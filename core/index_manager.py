import logging
import os
from functools import lru_cache
from typing import Dict, List, Optional

logger = logging.getLogger("bio_lab_index")

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.core.query_engine import BaseQueryEngine, RetrieverQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import chromadb.config

# 重依赖（torch 等）设为可选导入：CI 只跑不加载模型的测试时可不安这些包
try:
    from llama_index.embeddings.openai import OpenAIEmbedding
except ImportError:  # pragma: no cover
    OpenAIEmbedding = None
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:  # pragma: no cover
    HuggingFaceEmbedding = None
try:
    from llama_index.llms.openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

from config.config import Config
from core.retrievers import build_retriever
from core.rerankers import build_reranker, RerankerPostprocessor, TopKPostprocessor

COLLECTION_NAME = "bio_lab_index"


@lru_cache(maxsize=1)
def _build_llm():
    provider = (Config.LLM_PROVIDER or "openai").lower()
    if provider in ("vllm", "openai_like", "openailike"):
        from llama_index.llms.openai_like import OpenAILike

        extra_kwargs = {}
        if Config.VLLM_DISABLE_THINKING:
            # OpenAI 客户端需用 extra_body 透传 vLLM 的 chat_template_kwargs
            extra_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        logger.info("使用本地 vLLM: %s @ %s", Config.VLLM_MODEL, Config.VLLM_BASE_URL)
        return OpenAILike(
            model=Config.VLLM_MODEL,
            api_base=Config.VLLM_BASE_URL,
            api_key=Config.VLLM_API_KEY,
            is_chat_model=True,
            is_function_calling_model=False,
            context_window=Config.VLLM_CONTEXT_WINDOW,
            max_tokens=Config.VLLM_MAX_TOKENS,
            additional_kwargs=extra_kwargs,
        )
    return OpenAI(
        model=Config.OPENAI_MODEL,
        api_key=Config.OPENAI_API_KEY,
    )


@lru_cache(maxsize=1)
def _build_embed_model():
    if Config.EMBEDDING_PROVIDER == "huggingface":
        logger.info("使用本地 HuggingFace 模型: %s", Config.HF_EMBEDDING_MODEL)
        import torch

        # 评测/复现要求：关闭非确定性 kernel（warn_only 避免个别算子直接报错）
        torch.use_deterministic_algorithms(True, warn_only=True)
        return HuggingFaceEmbedding(
            model_name=Config.HF_EMBEDDING_MODEL,
            device=Config.HF_DEVICE,
            trust_remote_code=True,
            cache_folder=os.path.expanduser("~/.cache/huggingface/hub"),
            local_files_only=Config.HF_LOCAL_FILES_ONLY,
        )
    logger.info("使用 OpenAI Embedding 模型: %s", Config.EMBEDDING_MODEL)
    return OpenAIEmbedding(
        model=Config.EMBEDDING_MODEL,
        api_key=Config.OPENAI_API_KEY,
    )


class IndexManager:
    """索引管理：构建、持久化、增量插入、按文档删除。"""

    def __init__(self):
        # LLM / Embedding 进程内复用，避免每次实例化重复加载数 GB 模型
        self.llm = _build_llm()
        self.embed_model = _build_embed_model()

        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        self.node_parser = SentenceSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )

        self._query_engines: Dict[tuple, BaseQueryEngine] = {}
        self._retrievers: Dict[tuple, object] = {}

        self.chroma_client = chromadb.PersistentClient(
            path=Config.CHROMA_PATH,
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
        self._init_store()

    # ------------------------------------------------------------------ store
    def _load_storage_context(self) -> StorageContext:
        """有持久化文件则加载 docstore/index_store，否则用空白 context。"""
        if os.path.isdir(Config.STORAGE_DIR):
            try:
                return StorageContext.from_defaults(
                    persist_dir=Config.STORAGE_DIR,
                    vector_store=self.vector_store,
                )
            except (FileNotFoundError, ValueError) as e:
                logger.info("storage 目录暂无有效持久化文件，使用空白 context: %s", e)
        return StorageContext.from_defaults(vector_store=self.vector_store)

    def _init_store(self) -> None:
        """(重新)初始化向量 store / storage context，并尝试恢复已有索引。"""
        self.chroma_collection = self.chroma_client.get_or_create_collection(COLLECTION_NAME)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = self._load_storage_context()

        count = self.chroma_collection.count()
        if count > 0:
            try:
                self.index = self._restore_index()
                logger.info("从存储加载索引，含 %s 个向量。", count)
            except Exception as e:  # noqa: BLE001
                logger.warning("加载索引失败: %s", e)
                self.index = None
        else:
            logger.info("向量存储为空，需要初始化索引。")
            self.index = None

    def _restore_index(self) -> VectorStoreIndex:
        """优先从 index_store + docstore 恢复（保留节点文本/关系）；否则退回向量库。"""
        from llama_index.core import load_index_from_storage

        try:
            return load_index_from_storage(
                self.storage_context,
                store_nodes_override=True,
            )
        except ValueError:
            logger.info("storage 中无 index_store，退回从向量库构建（docstore 可能为空）。")
            return VectorStoreIndex.from_vector_store(
                self.vector_store,
                store_nodes_override=True,
            )

    def _persist(self) -> None:
        """持久化 docstore / index_store，使 BM25、查看、删除都能拿到文本与关系。"""
        if self.index is not None:
            self.index.storage_context.persist(persist_dir=Config.STORAGE_DIR)

    def reset(self) -> None:
        """清空向量与 docstore，供「重建索引」使用，避免重复向量。"""
        try:
            self.chroma_client.delete_collection(COLLECTION_NAME)
        except Exception as e:  # noqa: BLE001
            logger.warning("删除 collection 失败(可忽略): %s", e)

        for name in ("docstore.json", "index_store.json", "vector_store.json", "graph_store.json"):
            path = os.path.join(Config.STORAGE_DIR, name)
            if os.path.exists(path):
                os.remove(path)

        self._init_store()
        self.invalidate_query_engine_cache()
        logger.info("索引已重置。")

    # -------------------------------------------------------------- query side
    def get_retriever(self, *, retrieval: Optional[str] = None, topic: Optional[str] = None,
                      fetch_k: Optional[int] = None):
        """检索器（稠密 / 精确 / hybrid），带缓存（BM25 索引与向量加载开销大）。"""
        if self.index is None:
            raise ValueError("索引未初始化，请先创建索引")
        mode = retrieval or Config.RETRIEVAL_MODE
        fetch_k = fetch_k or Config.RETRIEVAL_FETCH_K
        key = (mode, topic, fetch_k)
        if key not in self._retrievers:
            self._retrievers[key] = build_retriever(
                self.index, mode=mode, topic=topic, fetch_k=fetch_k
            )
        return self._retrievers[key]

    def retrieve(self, query_text: str, top_k: int = 3, *, retrieval: Optional[str] = None,
                 rerank: Optional[str] = None, topic: Optional[str] = None) -> List:
        """只检索、不生成：返回 top_k 个 NodeWithScore。"""
        if self.index is None:
            raise ValueError("索引未初始化，请先创建索引")
        fetch_k = max(top_k, Config.RETRIEVAL_FETCH_K)
        nodes = self.get_retriever(retrieval=retrieval, topic=topic, fetch_k=fetch_k).retrieve(query_text)
        reranker = build_reranker(rerank)
        if reranker is not None:
            nodes = reranker.rerank_nodes(query_text, nodes[: Config.RERANK_MAX_CANDIDATES], top_k)
        else:
            nodes = nodes[:top_k]
        return nodes

    def get_query_engine(self, similarity_top_k: int = 3, *, retrieval: Optional[str] = None,
                         rerank: Optional[str] = None, topic: Optional[str] = None) -> BaseQueryEngine:
        """检索 + 重排 + 生成。重排/检索模式可按查询覆盖，带缓存。"""
        key = (similarity_top_k, retrieval, rerank, topic)
        if key not in self._query_engines:
            if self.index is None:
                raise ValueError("索引未初始化，请先创建索引")
            fetch_k = max(similarity_top_k, Config.RETRIEVAL_FETCH_K)
            retriever = self.get_retriever(retrieval=retrieval, topic=topic, fetch_k=fetch_k)
            reranker = build_reranker(rerank)
            if reranker is not None:
                postprocessors = [RerankerPostprocessor(reranker, top_n=similarity_top_k)]
            else:
                postprocessors = [TopKPostprocessor(similarity_top_k)]
            self._query_engines[key] = RetrieverQueryEngine.from_args(
                retriever, node_postprocessors=postprocessors
            )
        return self._query_engines[key]

    def invalidate_query_engine_cache(self) -> None:
        self._query_engines.clear()
        self._retrievers.clear()

    def query_index(self, query_text: str, top_k: int = 3, *, retrieval: Optional[str] = None,
                    rerank: Optional[str] = None, topic: Optional[str] = None):
        return self.get_query_engine(
            similarity_top_k=top_k, retrieval=retrieval, rerank=rerank, topic=topic
        ).query(query_text)

    # ------------------------------------------------------------ write side
    def create_index_from_documents(self, documents: List) -> VectorStoreIndex:
        """全量构建（调用方应先 reset，否则会与已有向量重复）。"""
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            transformations=[self.node_parser],
            store_nodes_override=True,
            show_progress=True,
        )
        self.invalidate_query_engine_cache()
        self._persist()
        return self.index

    def insert_documents(self, documents: List) -> int:
        """增量插入：批量切分后一次 insert_nodes，并持久化。"""
        if not self.index:
            raise ValueError("索引未初始化，请先创建索引")
        nodes = self.node_parser.get_nodes_from_documents(documents)
        if nodes:
            self.index.insert_nodes(nodes)
        self.invalidate_query_engine_cache()
        self._persist()
        return len(nodes)

    def delete_document(self, doc_id: str) -> int:
        """按稳定 doc_id 删除该文档的所有向量与 docstore 记录。"""
        if not self.index:
            raise ValueError("索引未初始化，请先创建索引")
        self.index.delete_ref_doc(doc_id, delete_from_docstore=True)
        self.invalidate_query_engine_cache()
        self._persist()
        if self.chroma_collection.count() == 0:
            self.index = None
        return 0

    # ---------------------------------------------------------------- view side
    def get_document_nodes(self, doc_id: str) -> List:
        """从 docstore 取该文档的 Node，保证「查看」与「问答引用」同源。"""
        if not self.index or not self.index.docstore:
            return []
        matched = []
        for node_id, doc in self.index.docstore.docs.items():
            if doc.metadata.get("doc_id") == doc_id:
                matched.append(doc)
        return matched

    def list_doc_ids(self) -> List[str]:
        if not self.index or not self.index.docstore:
            return []
        return sorted(
            {d.metadata.get("doc_id") for d in self.index.docstore.docs.values() if d.metadata.get("doc_id")}
        )

    def list_topics(self) -> Dict[str, int]:
        """返回 {专题: 文档数}，用于前端专题过滤下拉。"""
        if not self.index or not self.index.docstore:
            return {}
        seen: Dict[str, set] = {}
        for node in self.index.docstore.docs.values():
            topic = node.metadata.get("topic")
            doc_id = node.metadata.get("doc_id")
            if topic and doc_id:
                seen.setdefault(topic, set()).add(doc_id)
        return {t: len(ids) for t, ids in seen.items()}

    # ------------------------------------------------------------- compatibility
    def save_index(self, persist_dir: Optional[str] = None) -> None:
        if self.index:
            self.index.storage_context.persist(persist_dir=persist_dir or Config.STORAGE_DIR)

    def load_index(self, persist_dir: Optional[str] = None) -> None:
        try:
            if persist_dir:
                self.storage_context = StorageContext.from_defaults(
                    persist_dir=persist_dir,
                    vector_store=self.vector_store,
                )
            else:
                self.storage_context = self._load_storage_context()
            self.index = self._restore_index()
            self.invalidate_query_engine_cache()
            logger.info("索引已从 %s 加载。", persist_dir or Config.STORAGE_DIR)
        except Exception as e:  # noqa: BLE001
            logger.warning("加载索引失败: %s", e)
            self.index = None
