"""
对比不同 Embedding 模型的效果
"""
from core.index_manager import IndexManager
from config.config import Config
import os

def test_query(index_manager, query, model_name):
    """测试单个查询"""
    print(f"\n{'='*80}")
    print(f"模型: {model_name}")
    print(f"查询: {query}")
    print('='*80)
    
    try:
        response = index_manager.query_index(query, top_k=3)
        print(f"\nAI 回答:\n{response}\n")
        
        if hasattr(response, 'source_nodes') and response.source_nodes:
            print(f"找到 {len(response.source_nodes)} 个相关片段:\n")
            for i, node in enumerate(response.source_nodes, 1):
                score = node.score if hasattr(node, 'score') else 'N/A'
                print(f"片段 {i} - 相似度: {score}")
                print(f"内容预览: {node.text[:150]}...")
                print()
    except Exception as e:
        print(f"查询失败: {e}")

def main():
    print("加载索引...")
    index_manager = IndexManager()
    
    if index_manager.index is None:
        print("错误: 索引未初始化，请先运行 init_index.py")
        return
    
    model_name = Config.HF_EMBEDDING_MODEL if Config.EMBEDDING_PROVIDER == "huggingface" else Config.EMBEDDING_MODEL
    
    # 测试查询
    queries = [
        "PCR experiment protocol",
        "DNA extraction method",
        "RNA purification steps",
        "如何进行PCR实验",
    ]
    
    for query in queries:
        test_query(index_manager, query, model_name)
        input("\n按 Enter 继续下一个查询...")

if __name__ == "__main__":
    main()
