"""
测试查询脚本
"""
from core.index_manager import IndexManager

def main():
    print("加载索引...")
    index_manager = IndexManager()
    
    if index_manager.index is None:
        print("错误: 索引未初始化")
        return
    
    print("索引加载成功！\n")
    
    # 测试多个查询
    queries = [
        "PCR experiment",
        "DNA extraction",
        "RNA extraction method",
        "protocol for PCR",
        "如何提取DNA"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"查询: {query}")
        print('='*60)
        
        try:
            response = index_manager.query_index(query, top_k=2)
            print(f"\n回答:\n{response}\n")
            
            if hasattr(response, 'source_nodes') and response.source_nodes:
                print(f"找到 {len(response.source_nodes)} 个相关片段:")
                for i, node in enumerate(response.source_nodes, 1):
                    print(f"\n片段 {i}:")
                    print(f"相关度: {node.score if hasattr(node, 'score') else 'N/A'}")
                    print(f"内容: {node.text[:200]}...")
        except Exception as e:
            print(f"查询失败: {e}")

if __name__ == "__main__":
    main()
