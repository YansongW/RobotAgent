# -*- coding: utf-8 -*-

# 火山方舟对话测试工具 (Volcengine Chat Test Tool)
# 验证火山方舟API配置和对话功能
# 作者: RobotAgent开发团队
# 版本: 0.0.1 (Initial Release)
# 更新时间: 2024-01-20
# 基于: BaseRobotAgent v0.0.1

import os
import json
import sys
from pathlib import Path
from openai import OpenAI

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入配置加载器
from src.utils.config_loader import config_loader

class VolcengineChatClient:
    """
    火山方舟对话客户端
    
    封装火山方舟API调用功能，提供简单易用的对话接口。
    支持配置文件加载、对话历史管理和系统提示词设置。
    """
    
    def __init__(self, api_key: str = None, model_id: str = None, config: dict = None):
        """
        初始化火山方舟Chat客户端
        
        从配置文件或参数中加载API配置，初始化OpenAI客户端。
        
        Args:
            api_key: 火山方舟API密钥，如果为None则从配置文件加载
            model_id: 模型ID，如果为None则从配置文件加载
            config: 配置字典，如果为None则从配置文件加载
            
        Raises:
            ValueError: 当API配置无效或缺失时
        """
        # 加载配置
        if config is None:
            try:
                config = config_loader.get_volcengine_config()
            except Exception as e:
                print(f"❌ 无法加载配置文件: {e}")
                print("请确保 config/api_config.yaml 文件存在且配置正确")
                print("参考 config/api_config.yaml.template 创建配置文件")
                raise ValueError("未找到火山方舟API配置")
        
        # 设置API密钥和模型ID
        self.api_key = api_key or config.get('api_key')
        self.model_id = model_id or config.get('default_model')
        self.base_url = config.get('base_url')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 2000)
        
        if not self.api_key:
            raise ValueError("未提供API密钥且配置文件中也未找到")
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.conversation_history = []
        self.system_prompt = self._load_system_prompt()
        
    def _load_system_prompt(self) -> str:
        """
        从配置文件加载系统提示词
        
        读取chat_agent_prompt_template.json文件，将JSON配置转换为
        格式化的系统提示词字符串。
        
        Returns:
            格式化后的系统提示词字符串
            
        Note:
            如果配置文件加载失败，将返回默认的系统提示词
        """
        try:
            config_path = project_root / "config" / "chat_agent_prompt_template.json"
            with open(config_path, 'r', encoding='utf-8') as f:
                prompt_config = json.load(f)
            
            # 将JSON配置转换为系统提示词
            system_prompt = f"""
你是一个基于多维行为树结构的智能助手。

设计理念：{prompt_config['overview']['设计理念']}

核心目标：
{chr(10).join(f"- {goal}" for goal in prompt_config['overview']['核心目标'])}

工作流程：
"""
            
            for stage in prompt_config['process']:
                system_prompt += f"\n{stage['阶段']}：{stage['目标']}\n"
                for detail in stage['细节']:
                    system_prompt += f"  • {detail['项']}：{detail['说明']}\n"
                    if '范例' in detail:
                        system_prompt += f"    范例：{detail['范例']}\n"
            
            system_prompt += f"\n期望效果：{prompt_config['expectation']}"
            
            return system_prompt
            
        except Exception as e:
            print(f"警告：无法加载系统提示词配置文件: {e}")
            return "你是一个智能助手，请用自然、友好的方式与用户对话。"
    
    def chat(self, user_message: str) -> str:
        """
        发送消息并获取回复
        
        向火山方舟API发送用户消息，获取AI助手的回复。
        自动管理对话历史记录。
        
        Args:
            user_message: 用户输入的消息
            
        Returns:
            AI助手的回复字符串
            
        Raises:
            Exception: 当API调用失败时
        """
        try:
            # 构建消息列表
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # 添加历史对话
            messages.extend(self.conversation_history)
            
            # 添加当前用户消息
            messages.append({"role": "user", "content": user_message})
            
            # 调用API
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # 获取回复
            assistant_reply = completion.choices[0].message.content
            
            # 更新对话历史
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_reply})
            
            # 保持对话历史在合理长度内（最近10轮对话）
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return assistant_reply
            
        except Exception as e:
            return f"抱歉，发生了错误：{str(e)}"
    
    def clear_history(self):
        # 清空对话历史
        self.conversation_history = []
        print("对话历史已清空。")

def main():
    # 主函数：实现命令行对话界面
    
    print("=" * 60)
    print("🤖 火山方舟Chat API测试工具")
    print("=" * 60)
    print("📋 正在加载配置...")
    
    # 初始化客户端
    try:
        chat_client = VolcengineChatClient()
        print(f"✅ 客户端初始化成功！")
        print(f"🔧 模型ID: {chat_client.model_id}")
        print(f"🌐 API地址: {chat_client.base_url}")
        print(f"🔑 API密钥: {chat_client.api_key[:8]}...{chat_client.api_key[-4:]}")
    except Exception as e:
        print(f"❌ 客户端初始化失败：{e}")
        print("请检查配置文件 config/api_config.yaml 是否存在且配置正确")
        return
    
    print("输入 'quit' 或 'exit' 退出程序")
    print("输入 'clear' 清空对话历史")
    print("输入 'help' 查看帮助信息")
    print("-" * 60)
    
    # 主对话循环
    while True:
        try:
            # 获取用户输入
            user_input = input("\n👤 您：").strip()
            
            # 处理特殊命令
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("👋 再见！")
                break
            elif user_input.lower() in ['clear', '清空']:
                chat_client.clear_history()
                continue
            elif user_input.lower() in ['help', '帮助']:
                print("""
📖 帮助信息：
- 直接输入消息与AI对话
- 'quit' 或 'exit'：退出程序
- 'clear'：清空对话历史
- 'help'：显示此帮助信息
                """)
                continue
            elif not user_input:
                print("请输入有效的消息。")
                continue
            
            # 发送消息并获取回复
            print("🤖 助手正在思考...")
            response = chat_client.chat(user_input)
            print(f"🤖 助手：{response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 程序被用户中断，再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误：{e}")

if __name__ == "__main__":
    main()