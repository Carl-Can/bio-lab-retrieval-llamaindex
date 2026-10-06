"""
统一文档解析：所有入库链路（CLI 建库 / API initialize / API upload）都产出
带稳定元数据的 LlamaIndex Document，避免「查看」与「检索」两套切分。

元数据契约（Node 会继承）：
    doc_id   稳定、唯一、可复现（建库 = 相对 docs 的路径去扩展名；上传 = file_id）
    file_name 展示名（建库 = 相对路径；上传 = 原始文件名）
    topic    一级专题目录（建库 = 相对路径首段；上传 = "uploads"）
    source   "indexed" | "uploaded"
    file_path 磁盘路径
"""
import logging
import os
from collections import Counter
from typing import Iterable, List, Optional, Sequence

logger = logging.getLogger("bio_lab_ingestion")

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.readers.file import PyMuPDFReader

SUPPORTED_EXTENSIONS = (".pdf", ".md", ".txt", ".docx")


def derive_doc_id(file_path: str, docs_root: Optional[str] = None) -> str:
    """文件 -> 稳定 doc_id。建库文档用相对 docs 的路径去扩展名，保证可复现。"""
    if docs_root:
        rel = os.path.relpath(file_path, docs_root)
    else:
        rel = os.path.basename(file_path)
    return os.path.splitext(rel)[0].replace(os.sep, "/")


def derive_topic(file_path: str, docs_root: Optional[str] = None) -> str:
    """一级专题目录；没有子目录时归为 root / uploads。"""
    if docs_root:
        rel = os.path.relpath(file_path, docs_root).replace(os.sep, "/")
        parts = rel.split("/")
        return parts[0] if len(parts) > 1 else "root"
    return "uploads"


def _read_file(file_path: str) -> List[Document]:
    """读单文件为 Document 列表。PDF 按页返回，其余交给 SimpleDirectoryReader。"""
    if file_path.lower().endswith(".pdf"):
        return PyMuPDFReader().load(file_path=file_path)
    return SimpleDirectoryReader(input_files=[file_path]).load_data()


def _merge_pages(file_path: str, pages: Sequence[Document]) -> str:
    """把同一文件的多个 Document 合并为一段文本，并为 PDF 保留页码标记。"""
    if len(pages) == 1:
        return pages[0].text or ""
    is_pdf = file_path.lower().endswith(".pdf")
    parts = []
    for i, page in enumerate(pages):
        page_no = page.metadata.get("page_number") or page.metadata.get("source") or (i + 1)
        if is_pdf:
            parts.append(f"\n\n[第 {page_no} 页]\n{(page.text or '').strip()}")
        else:
            parts.append((page.text or "").strip())
    return "\n".join(parts).strip()


def load_document(
    file_path: str,
    *,
    docs_root: Optional[str] = None,
    source: str = "indexed",
    file_name: Optional[str] = None,
    doc_id: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
) -> Optional[Document]:
    """
    解析单个文件为一个 LlamaIndex Document（多页/多段会被合并，避免同文件多 ref_doc_id）。
    """
    try:
        pages = _read_file(file_path)
    except Exception as e:  # noqa: BLE001 - 单文件失败不应中断整批
        logger.warning("无法解析 %s: %s", file_path, e)
        return None

    if not pages:
        logger.warning("解析结果为空: %s", file_path)
        return None

    resolved_doc_id = doc_id or derive_doc_id(file_path, docs_root)
    resolved_name = file_name or (
        os.path.relpath(file_path, docs_root).replace(os.sep, "/")
        if docs_root
        else os.path.basename(file_path)
    )

    metadata = {
        "doc_id": resolved_doc_id,
        "file_name": resolved_name,
        "topic": derive_topic(file_path, docs_root),
        "source": source,
        "file_path": file_path,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    doc = Document(
        text=_merge_pages(file_path, pages),
        metadata=metadata,
        id_=resolved_doc_id,
    )
    return doc


def load_documents(
    file_paths: Iterable[str],
    *,
    docs_root: Optional[str] = None,
    source: str = "indexed",
) -> List[Document]:
    """批量解析，跳过失败项。同名不同扩展名的文件用扩展名消歧，保证 doc_id 唯一。"""
    paths = list(file_paths)
    base_ids = [derive_doc_id(p, docs_root) for p in paths]
    counts = Counter(base_ids)

    documents = []
    for path, base_id in zip(paths, base_ids):
        if counts[base_id] > 1 and docs_root:
            # 如 x.md 与 x.pdf 冲突 -> doc_id 保留扩展名
            doc_id = os.path.relpath(path, docs_root).replace(os.sep, "/")
        else:
            doc_id = base_id
        doc = load_document(path, docs_root=docs_root, source=source, doc_id=doc_id)
        if doc is not None:
            documents.append(doc)
    return documents


def iter_supported_files(docs_root: str) -> List[str]:
    """递归收集 docs 目录下支持的文件。"""
    files = []
    for root, _, filenames in os.walk(docs_root):
        for name in filenames:
            if name.lower().endswith(SUPPORTED_EXTENSIONS):
                files.append(os.path.join(root, name))
    return files


def load_documents_from_dir(docs_root: str) -> List[Document]:
    """建库入口：递归解析 docs 目录。"""
    return load_documents(iter_supported_files(docs_root), docs_root=docs_root, source="indexed")


def list_doc_ids(docs_root: str) -> List[str]:
    """不解析文件，仅按与 load_documents 相同的规则算出 doc_id 列表（用于校验评测金标）。"""
    paths = iter_supported_files(docs_root)
    base = [derive_doc_id(p, docs_root) for p in paths]
    counts = Counter(base)
    ids = []
    for path, bid in zip(paths, base):
        if counts[bid] > 1:
            ids.append(os.path.relpath(path, docs_root).replace(os.sep, "/"))
        else:
            ids.append(bid)
    return ids
