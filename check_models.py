"""
检查 OpenAI API 可用的模型
"""
import os
from openai import OpenAI
from config.config import Config

def main():
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    print("正在获取可用模型列表...\n")
    
    try:
        models = client.models.list()
        
        # 过滤出 GPT 和 o1 模型
        gpt_models = []
        for model in models.data:
            if any(x in model.id for x in ['gpt', 'o1', 'o3']):
                gpt_models.append(model.id)
        
        gpt_models.sort()
        
        print(f"找到 {len(gpt_models)} 个可用的语言模型:\n")
        for model in gpt_models:
            print(f"  - {model}")
        
        print(f"\n当前配置的模型: {Config.OPENAI_MODEL}")
        
        if Config.OPENAI_MODEL in gpt_models:
            print("✅ 当前模型可用")
        else:
            print("❌ 当前模型不可用，请选择上面列表中的模型")
            
    except Exception as e:
        print(f"错误: {e}")
        print("\n请检查:")
        print("1. API Key 是否正确")
        print("2. 网络连接是否正常")
        print("3. API Key 是否有足够的权限")

if __name__ == "__main__":
    main()
