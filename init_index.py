"""
索引管理 CLI 工具
提供初始化数据库、索引文献、清除重复向量、查看状态等功能
"""
import logging
import os
import sys

logger = logging.getLogger("bio_lab_cli")
from llama_index.core import Document
from llama_index.readers.file import PyMuPDFReader
from core.index_manager import IndexManager
from config.config import Config


def clear_database():
    """清除所有向量数据"""
    import chromadb
    settings = chromadb.config.Settings(anonymized_telemetry=False)
    client = chromadb.PersistentClient(path=Config.CHROMA_PATH, settings=settings)
    
    try:
        client.delete_collection("bio_lab_index")
        print("✓ 已清除所有向量数据")
    except Exception:
        print("✓ 数据库已是空状态")


def index_documents():
    """索引 docs 目录中的文献（递归扫描子目录）"""
    docs_path = Config.DOCS_FOLDER
    if not os.path.exists(docs_path):
        print(f"错误: 文档目录不存在: {docs_path}")
        return
    
    supported_exts = ('.pdf', '.md', '.docx', '.txt')
    files = []
    for root, _, filenames in os.walk(docs_path):
        for f in filenames:
            if f.lower().endswith(supported_exts):
                files.append(os.path.join(root, f))
    
    print(f"找到 {len(files)} 个文档")
    
    if not files:
        print("文档目录为空")
        return
    
    from llama_index.core import SimpleDirectoryReader
    reader_pdf = PyMuPDFReader()
    documents = []
    
    for file_path in files:
        rel_path = os.path.relpath(file_path, docs_path)
        try:
            print(f"  加载: {rel_path}")
            if file_path.lower().endswith('.pdf'):
                docs = reader_pdf.load(file_path=file_path)
            else:
                docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
            for doc in docs:
                doc.metadata['file_name'] = rel_path
            documents.extend(docs)
        except Exception as e:
            print(f"  警告: 无法加载 {rel_path}: {e}")
    
    if not documents:
        print("没有成功加载任何文档")
        return
    
    print(f"成功加载 {len(documents)} 个文档片段")
    
    print("正在创建索引...")
    index_manager = IndexManager()
    index_manager.create_index_from_documents(documents)
    print("✓ 索引创建成功!")


def clear_duplicates():
    """清除重复向量"""
    import chromadb
    settings = chromadb.config.Settings(anonymized_telemetry=False)
    client = chromadb.PersistentClient(path=Config.CHROMA_PATH, settings=settings)
    
    try:
        collection = client.get_collection("bio_lab_index")
        total_before = collection.count()
        
        # 获取所有数据
        results = collection.get(include=["documents", "metadatas"])
        ids = results['ids']
        documents = results['documents']
        
        if not ids:
            print("集合为空，无需处理")
            return
        
        # 基于文档内容去重
        seen_content = {}
        duplicate_ids = []
        
        for i, (doc_id, doc_content) in enumerate(zip(ids, documents)):
            if doc_content in seen_content:
                duplicate_ids.append(doc_id)
            else:
                seen_content[doc_content] = doc_id
        
        if duplicate_ids:
            collection.delete(ids=duplicate_ids)
            print(f"✓ 已清除 {len(duplicate_ids)} 个重复向量")
            print(f"  清除前: {total_before} 个向量")
            print(f"  清除后: {collection.count()} 个向量")
        else:
            print("✓ 未发现重复向量")
            
    except Exception as e:
        print(f"错误: {e}")


def show_status():
    """显示系统状态"""
    import chromadb
    settings = chromadb.config.Settings(anonymized_telemetry=False)
    client = chromadb.PersistentClient(path=Config.CHROMA_PATH, settings=settings)
    
    print("\n" + "=" * 40)
    print("系统状态")
    print("=" * 40)
    
    # docs 目录统计（递归）
    docs_path = Config.DOCS_FOLDER
    if os.path.exists(docs_path):
        supported_exts = ('.pdf', '.md', '.docx', '.txt')
        all_files = []
        for root, _, filenames in os.walk(docs_path):
            for f in filenames:
                if f.lower().endswith(supported_exts):
                    all_files.append(os.path.relpath(os.path.join(root, f), docs_path))
        print(f"docs 目录文献: {len(all_files)} 个")
        for f in all_files:
            print(f"  - {f}")
    else:
        print("docs 目录: 不存在")
    
    # uploads 目录统计
    upload_path = Config.UPLOAD_FOLDER
    if os.path.exists(upload_path):
        uploaded_files = [f for f in os.listdir(upload_path) if f.endswith(('.pdf', '.docx', '.txt', '.md'))]
        print(f"上传文献: {len(uploaded_files)} 个")
    else:
        print("上传文献: 0 个")
    
    # 向量统计
    try:
        collection = client.get_collection("bio_lab_index")
        vector_count = collection.count()
        print(f"向量总数: {vector_count} 个")
    except Exception:
        print("向量总数: 0 (索引未创建)")
    
    print("=" * 40 + "\n")


def main():
    while True:
        print("\n" + "=" * 40)
        print("索引管理工具")
        print("=" * 40)
        print("1. 初始化数据库 (清除所有向量)")
        print("2. 索引 docs 目录文献")
        print("3. 清除重复向量")
        print("4. 查看状态")
        print("5. 重建索引 (清除+重新索引)")
        print("0. 退出")
        print("=" * 40)
        
        choice = input("请选择操作 [0-5]: ").strip()
        
        if choice == '1':
            confirm = input("确定要清除所有向量数据? (y/N): ").strip().lower()
            if confirm == 'y':
                clear_database()
            else:
                print("已取消")
                
        elif choice == '2':
            index_documents()
            
        elif choice == '3':
            clear_duplicates()
            
        elif choice == '4':
            show_status()
            
        elif choice == '5':
            confirm = input("将清除所有向量并重新索引，确定? (y/N): ").strip().lower()
            if confirm == 'y':
                clear_database()
                index_documents()
            else:
                print("已取消")
                
        elif choice == '0':
            print("再见!")
            break
            
        else:
            print("无效选项，请重新选择")


if __name__ == "__main__":
    main()
