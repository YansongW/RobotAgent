"""
OpenAI兼容的豆包服务使用示例
基于用户提供的示例代码实现
"""

import asyncio
import os
import sys
import json
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from services.doubao_service import DoubaoService, DoubaoOpenAIClient


async def load_config():
    """加载配置"""
    config_path = project_root / "config" / "config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config["doubao"]


async def example_single_turn_chat():
    """单轮对话示例"""
    print("\n=== 单轮对话示例 ===")
    
    config = await load_config()
    service = DoubaoService(config)
    
    # 使用便捷方法
    response = await service.single_turn_chat(
        user_message="常见的十字花科植物有哪些？",
        system_message="你是豆包，是由字节跳动开发的 AI 人工智能助手"
    )
    print(f"助手回复: {response}")
    
    # 使用OpenAI兼容接口
    messages = [
        {"role": "system", "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"},
        {"role": "user", "content": "常见的十字花科植物有哪些？"}
    ]
    
    openai_response = await service.openai_chat_completion(messages=messages)
    print(f"OpenAI格式回复: {openai_response['choices'][0]['message']['content']}")


async def example_multi_turn_chat():
    """多轮对话示例"""
    print("\n=== 多轮对话示例 ===")
    
    config = await load_config()
    service = DoubaoService(config)
    
    messages = [
        {"role": "system", "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"},
        {"role": "user", "content": "花椰菜是什么？"},
        {"role": "assistant", "content": "花椰菜又称菜花、花菜，是一种常见的蔬菜。"},
        {"role": "user", "content": "再详细点"}
    ]
    
    response = await service.multi_turn_chat(messages=messages)
    print(f"多轮对话回复: {response}")


async def example_stream_chat():
    """流式对话示例"""
    print("\n=== 流式对话示例 ===")
    
    config = await load_config()
    service = DoubaoService(config)
    
    messages = [
        {"role": "system", "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"},
        {"role": "user", "content": "常见的十字花科植物有哪些？"}
    ]
    
    print("流式回复: ", end="")
    async for chunk in service.stream_chat(messages=messages):
        print(chunk, end="", flush=True)
    print()  # 换行


async def example_function_calling():
    """工具调用示例"""
    print("\n=== 工具调用示例 ===")
    
    config = await load_config()
    service = DoubaoService(config)
    
    messages = [
        {"role": "user", "content": "北京今天天气如何？"}
    ]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "获取给定地点的天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "地点的位置信息，比如北京"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["摄氏度", "华氏度"]
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    response = await service.function_call_chat(messages=messages, tools=tools)
    print(f"工具调用回复: {json.dumps(response, ensure_ascii=False, indent=2)}")


async def example_embeddings():
    """文本向量化示例"""
    print("\n=== 文本向量化示例 ===")
    
    config = await load_config()
    service = DoubaoService(config)
    
    # 单个文本
    response = await service.openai_embeddings(
        input="花椰菜又称菜花、花菜，是一种常见的蔬菜。"
    )
    print(f"嵌入向量维度: {len(response['data'][0]['embedding'])}")
    print(f"前5个向量值: {response['data'][0]['embedding'][:5]}")
    
    # 多个文本
    texts = [
        "花椰菜又称菜花、花菜，是一种常见的蔬菜。",
        "白菜是十字花科芸薹属的植物。"
    ]
    
    batch_response = await service.openai_embeddings(input=texts)
    print(f"批量嵌入数量: {len(batch_response['data'])}")


async def example_direct_openai_client():
    """直接使用OpenAI兼容客户端示例"""
    print("\n=== 直接使用OpenAI客户端示例 ===")
    
    config = await load_config()
    
    # 设置环境变量（模拟）
    os.environ["ARK_API_KEY"] = config["api_key"]
    
    # 创建客户端
    client = DoubaoOpenAIClient(
        api_key=config["api_key"],
        base_url=config["base_url"]
    )
    
    # 聊天完成
    messages = [
        {"role": "system", "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"},
        {"role": "user", "content": "你好！"}
    ]
    
    response = await client.chat.create(
        model=config["models"]["chat"],
        messages=messages,
        max_tokens=1000,
        temperature=0.7
    )
    
    print(f"直接客户端回复: {response['choices'][0]['message']['content']}")
    
    # 嵌入向量
    embedding_response = await client.embeddings.create(
        model=config["models"]["text_embedding"],
        input="测试文本"
    )
    
    print(f"直接客户端嵌入维度: {len(embedding_response['data'][0]['embedding'])}")


async def example_async_stream():
    """异步流式示例"""
    print("\n=== 异步流式示例 ===")
    
    config = await load_config()
    client = DoubaoOpenAIClient(
        api_key=config["api_key"],
        base_url=config["base_url"]
    )
    
    messages = [
        {"role": "system", "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"},
        {"role": "user", "content": "常见的十字花科植物有哪些？"}
    ]
    
    print("异步流式回复: ", end="")
    stream = await client.chat.create(
        model=config["models"]["chat"],
        messages=messages,
        stream=True
    )
    
    async for chunk in stream:
        if "choices" in chunk and chunk["choices"]:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                print(content, end="", flush=True)
    print()  # 换行


async def main():
    """运行所有示例"""
    print("OpenAI兼容的豆包服务示例")
    print("=" * 50)
    
    try:
        await example_single_turn_chat()
        await example_multi_turn_chat()
        await example_stream_chat()
        await example_function_calling()
        await example_embeddings()
        await example_direct_openai_client()
        await example_async_stream()
        
        print("\n所有示例运行完成！")
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())