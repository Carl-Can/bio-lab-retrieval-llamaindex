import logging
import os
import json
import threading
import time
import uuid
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from config.config import Config
from core import ingestion
from core.document_processor import DocumentProcessor
from core.index_manager import IndexManager
from api.models import (
    InitializeResponse,
    UploadResponse,
    QueryResponse,
    SourceNode,
    DocumentContentResponse,
    DocumentChunk,
    HealthResponse,
    DeleteResponse,
    StatsResponse,
    DocumentListItem,
    DocumentListResponse,
    TopicItem,
    TopicListResponse,
    GroundingResult,
)

logger = logging.getLogger("bio_lab_api")
Config.setup_logging()

app = FastAPI(
    title="生物实验检索系统",
    description="基于LlamaIndex构建的生物实验检索平台",
    version="1.0.0"
)

cors_origins = [o.strip() for o in Config.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

index_manager = IndexManager()
doc_processor = DocumentProcessor(index_manager)

os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)


def _log_query(q: str, params: dict, source_nodes=None) -> None:
    """把真实 query 追加到 jsonl（best-effort，用于后续挖评测集）。"""
    try:
        os.makedirs(os.path.dirname(Config.QUERY_LOG), exist_ok=True)
        line = json.dumps({
            "ts": time.time(),
            "q": q,
            **params,
            "sources": [n.source_file for n in (source_nodes or [])],
        }, ensure_ascii=False)
        with _QUERY_LOG_LOCK:
            with open(Config.QUERY_LOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:  # noqa: BLE001
        logger.debug("写查询日志失败: %s", e)


_QUERY_LOG_LOCK = threading.Lock()


async def verify_api_key(x_api_key: str = Header(default="")):
    if Config.API_KEY and x_api_key != Config.API_KEY:
        raise HTTPException(status_code=403, detail="无效的 API Key")
    return True


@app.get("/", response_model=HealthResponse)
async def root():
    return {
        "message": "生物实验检索系统API",
        "version": "1.0.0",
        "description": "基于LlamaIndex构建的生物实验检索平台",
        "index_initialized": index_manager.index is not None
    }


@app.get("/stats", response_model=StatsResponse)
async def get_stats(authorized: bool = Depends(verify_api_key)):
    try:
        vector_count = 0
        if hasattr(index_manager, 'chroma_collection'):
            try:
                vector_count = index_manager.chroma_collection.count()
            except Exception:
                vector_count = 0
        return {
            "vector_count": vector_count,
            "index_initialized": index_manager.index is not None
        }
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail="获取统计信息失败")


@app.post("/initialize", response_model=InitializeResponse)
async def initialize_index(authorized: bool = Depends(verify_api_key)):
    try:
        docs_path = getattr(Config, 'DOCS_FOLDER', "./docs")
        if not os.path.exists(docs_path):
            raise HTTPException(status_code=404, detail=f"文档目录不存在: {docs_path}")

        doc_files = ingestion.iter_supported_files(docs_path)
        if not doc_files:
            raise HTTPException(status_code=400, detail="文档目录中没有支持的文件")

        documents = ingestion.load_documents_from_dir(docs_path)
        if not documents:
            raise HTTPException(status_code=400, detail="无法加载任何文档")

        # 重建前清空，避免与已有向量重复
        index_manager.reset()
        index_manager.create_index_from_documents(documents)
        logger.info("索引初始化完成: %s 篇文档, %s 个文件", len(documents), len(doc_files))

        return {
            "message": "索引初始化成功",
            "documents_count": len(documents),
            "files_processed": len(doc_files)
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"索引初始化失败: {e}")
        raise HTTPException(status_code=500, detail="索引初始化失败")


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), authorized: bool = Depends(verify_api_key)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")

        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1] or ".pdf"
        file_path = os.path.join(Config.UPLOAD_FOLDER, f"{file_id}{file_extension}")

        content = await file.read()
        if len(content) > Config.MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"文件大小超过限制 ({Config.MAX_FILE_SIZE} bytes)")

        with open(file_path, "wb") as buffer:
            buffer.write(content)

        doc = ingestion.load_document(
            file_path,
            source="uploaded",
            file_name=file.filename,
            doc_id=file_id,
        )
        if doc is None:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=400, detail=f"文档解析失败: {file.filename}")

        if index_manager.index is None:
            index_manager.create_index_from_documents([doc])
            documents_count = 1
        else:
            documents_count = index_manager.insert_documents([doc])

        logger.info(f"文档上传成功: {file.filename} ({file_id}), {documents_count} 个片段")

        return {
            "message": "文档上传并索引成功",
            "file_id": file_id,
            "filename": file.filename,
            "documents_count": documents_count
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"文档上传失败: {e}")
        raise HTTPException(status_code=500, detail="文档上传失败")


@app.get("/query", response_model=QueryResponse)
async def query_documents(
    q: str,
    top_k: int = 3,
    retrieval: Optional[str] = None,
    rerank: Optional[str] = None,
    topic: Optional[str] = None,
    generate: bool = True,
    verify: bool = False,
    authorized: bool = Depends(verify_api_key),
):
    """检索/生成解耦：generate=false 只返回检索片段（不调用 LLM）。
    verify=true 时对生成答案做 grounding 核验（Jev Yes/No，逐条陈述）。"""
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="查询内容不能为空")

    if top_k < 1 or top_k > 10:
        raise HTTPException(status_code=400, detail="top_k 必须在 1-10 之间")

    try:
        if index_manager.index is None:
            raise HTTPException(status_code=400, detail="索引未初始化，请先上传文档")

        def _to_source(node) -> SourceNode:
            score = float(node.score) if getattr(node, "score", None) is not None else 0.0
            metadata = getattr(node, "metadata", {}) or {}
            return SourceNode(
                text=(node.text or "")[:500],
                score=score,
                source_file=metadata.get("file_name", metadata.get("source", "")),
                page_number=metadata.get("page_label", metadata.get("page_number")),
            )

        if generate:
            engine = index_manager.get_query_engine(
                similarity_top_k=top_k, retrieval=retrieval, rerank=rerank, topic=topic
            )
            response = engine.query(q.strip())
            source_nodes = [_to_source(n) for n in getattr(response, "source_nodes", [])][:top_k]

            grounding = None
            if verify:
                from core.grounding import build_grounding_guard

                contexts = [n.text for n in getattr(response, "source_nodes", [])]
                # evaluator 内部用 asyncio.run，必须在无事件循环的线程里跑
                result = await run_in_threadpool(
                    build_grounding_guard().verify, str(response), contexts, q.strip()
                )
                grounding = GroundingResult(**result)

            _log_query(q.strip(), {"top_k": top_k, "retrieval": retrieval, "rerank": rerank,
                                   "topic": topic, "generate": True, "verify": verify}, source_nodes)
            return {
                "query": q,
                "response": str(response),
                "source_nodes": source_nodes,
                "grounding": grounding,
            }

        # 只检索：不触发 LLM 生成
        nodes = index_manager.retrieve(
            q.strip(), top_k=top_k, retrieval=retrieval, rerank=rerank, topic=topic
        )
        source_nodes = [_to_source(n) for n in nodes]
        _log_query(q.strip(), {"top_k": top_k, "retrieval": retrieval, "rerank": rerank,
                               "topic": topic, "generate": False}, source_nodes)
        return {
            "query": q,
            "response": "",
            "source_nodes": source_nodes,
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"查询失败: {q}: {e}")
        raise HTTPException(status_code=500, detail="查询失败")


def _resolve_doc_file(doc_id: str) -> Optional[str]:
    if os.path.exists(Config.UPLOAD_FOLDER):
        for f in os.listdir(Config.UPLOAD_FOLDER):
            if f.startswith(doc_id):
                return os.path.join(Config.UPLOAD_FOLDER, f)
    docs_path = getattr(Config, 'DOCS_FOLDER', "./docs")
    if os.path.exists(docs_path):
        for root, _, files in os.walk(docs_path):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), docs_path)
                id_part = os.path.splitext(rel)[0]
                if id_part == doc_id:
                    return os.path.join(root, f)
    return None


@app.get("/docs/{doc_id}", response_model=DocumentContentResponse)
async def get_document(doc_id: str, authorized: bool = Depends(verify_api_key)):
    try:
        if not doc_id or not doc_id.strip():
            raise HTTPException(status_code=400, detail="文档ID不能为空")

        # 优先读索引里的 Node，保证与问答引用同源
        chunks = doc_processor.get_indexed_chunks(doc_id)

        if not chunks:
            file_path = _resolve_doc_file(doc_id)
            if not file_path:
                raise HTTPException(status_code=404, detail="文档未找到")
            chunks = doc_processor.process_file(file_path)

        if not chunks:
            raise HTTPException(status_code=404, detail="文档未找到")

        return {
            "doc_id": doc_id,
            "chunks": [DocumentChunk(**chunk) for chunk in chunks]
        }
    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="文档文件不存在或无法访问")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"获取文档失败: {doc_id}: {e}")
        raise HTTPException(status_code=500, detail="获取文档失败")


@app.delete("/docs/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str, authorized: bool = Depends(verify_api_key)):
    try:
        file_path = _resolve_doc_file(doc_id)
        if not file_path and index_manager.index is None:
            raise HTTPException(status_code=404, detail="文档未找到")

        if file_path and os.path.exists(file_path) and file_path.startswith(
            os.path.abspath(Config.UPLOAD_FOLDER)
        ):
            os.remove(file_path)

        # 按稳定 doc_id 从向量库 + docstore 删除
        try:
            index_manager.delete_document(doc_id)
        except ValueError:
            pass

        return {"message": "文档已删除", "doc_id": doc_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文档失败: {doc_id}: {e}")
        raise HTTPException(status_code=500, detail="删除文档失败")


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents(authorized: bool = Depends(verify_api_key)):
    try:
        documents = []

        if os.path.exists(Config.UPLOAD_FOLDER):
            for filename in os.listdir(Config.UPLOAD_FOLDER):
                file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path):
                    stat = os.stat(file_path)
                    file_id = os.path.splitext(filename)[0]
                    documents.append(DocumentListItem(
                        id=file_id,
                        filename=filename,
                        size=stat.st_size,
                        upload_time=stat.st_mtime * 1000,
                        source="uploaded"
                    ))

        docs_path = getattr(Config, 'DOCS_FOLDER', "./docs")
        if os.path.exists(docs_path):
            supported = ('.pdf', '.md', '.docx', '.txt')
            for root, _, filenames in os.walk(docs_path):
                for filename in filenames:
                    if filename.lower().endswith(supported):
                        file_path = os.path.join(root, filename)
                        rel_path = os.path.relpath(file_path, docs_path)
                        stat_obj = os.stat(file_path)
                        file_id = os.path.splitext(rel_path)[0]
                        documents.append(DocumentListItem(
                            id=file_id,
                            filename=rel_path,
                            size=stat_obj.st_size,
                            upload_time=stat_obj.st_mtime * 1000,
                            source="indexed"
                        ))

        documents.sort(key=lambda x: x.upload_time, reverse=True)
        return {"documents": documents, "total": len(documents)}
    except Exception as e:
        logger.error(f"获取文档列表失败: {e}")
        raise HTTPException(status_code=500, detail="获取文档列表失败")


@app.get("/topics", response_model=TopicListResponse)
async def list_topics(authorized: bool = Depends(verify_api_key)):
    """返回专题（docs 一级目录）及各专题文档数，供前端专题过滤下拉使用。"""
    try:
        counts = index_manager.list_topics()
        if not counts:
            # 索引未建时退回扫描 docs 目录
            docs_path = getattr(Config, 'DOCS_FOLDER', "./docs")
            for path in ingestion.iter_supported_files(docs_path):
                topic = ingestion.derive_topic(path, docs_path)
                counts[topic] = counts.get(topic, 0) + 1

        topics = [
            TopicItem(topic=t, count=c)
            for t, c in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        ]
        return {"topics": topics, "total": len(topics)}
    except Exception as e:
        logger.error(f"获取专题列表失败: {e}")
        raise HTTPException(status_code=500, detail="获取专题列表失败")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=Config.DEBUG
    )
