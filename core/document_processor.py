"""
文档处理：不再使用 unstructured 另起一套切分，统一复用 ingestion + SentenceSplitter。
查看「文档」页时优先读索引里的 Node，保证与问答引用同源。
"""
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger("bio_lab_doc")

from llama_index.core.node_parser import SentenceSplitter

from config.config import Config
from core import ingestion

SUPPORTED_EXTENSIONS = ingestion.SUPPORTED_EXTENSIONS


class DocumentProcessor:
    """解析与查看。切分器与建索引时完全一致。"""

    def __init__(self, index_manager=None):
        self.chunk_size = Config.CHUNK_SIZE
        self.chunk_overlap = Config.CHUNK_OVERLAP
        self.upload_folder = Config.UPLOAD_FOLDER
        self.index_manager = index_manager
        self.splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        os.makedirs(self.upload_folder, exist_ok=True)

    def process_file(
        self,
        file_path: str,
        *,
        source: str = "uploaded",
        docs_root: Optional[str] = None,
    ) -> List[Dict]:
        """解析单个文件并按与索引一致的方式切分（用于索引未建好时的降级查看/测试）。"""
        doc = ingestion.load_document(file_path, docs_root=docs_root, source=source)
        if doc is None:
            return []
        nodes = self.splitter.get_nodes_from_documents([doc])
        return [
            {
                "chunk_id": i,
                "text": node.get_content(),
                "metadata": dict(node.metadata),
            }
            for i, node in enumerate(nodes)
        ]

    # 兼容旧调用名
    def process_document(self, file_path: str) -> List[Dict]:
        return self.process_file(file_path)

    def get_indexed_chunks(self, doc_id: str) -> List[Dict]:
        """从索引 docstore 读取该文档的 Node（与问答引用同源）。"""
        if self.index_manager is None:
            return []
        nodes = self.index_manager.get_document_nodes(doc_id)
        return [
            {
                "chunk_id": i,
                "text": node.get_content(),
                "metadata": dict(node.metadata),
            }
            for i, node in enumerate(nodes)
        ]
