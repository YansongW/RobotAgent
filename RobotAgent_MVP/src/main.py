# -*- coding: utf-8 -*-

# RobotAgent MVP 主程序入口 (Main Entry Point)
# 启动和管理三智能体协作系统的主程序
# 作者: RobotAgent开发团队
# 版本: 0.0.1 (Initial Release)
# 更新时间: 2025-01-20
# 基于: BaseRobotAgent v0.0.1

# 导入标准库
import asyncio
import logging
import signal
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# 导入第三方库
from openai import OpenAI

# 导入项目基础组件
from agents.chat_agent import ChatAgent
from agents.action_agent import ActionAgent
from agents.memory_agent import MemoryAgent
from agents.agent_coordinator import AgentCoordinator
from communication.message_bus import get_message_bus
from utils.config_loader import config_loader
from utils.logger import setup_logger

# 导入CAMEL框架组件
try:
    from camel.agents import ChatAgent as CAMELChatAgent
    from camel.messages import BaseMessage
    CAMEL_AVAILABLE = True
except ImportError:
    CAMEL_AVAILABLE = False
    logging.warning("CAMEL框架未安装，使用模拟实现")


class RobotAgentSystem:
    """
    RobotAgent MVP 系统主类
    
    负责管理和协调三智能体系统的启动、运行和关闭。
    提供统一的系统接口和生命周期管理。
    
    Attributes:
        chat_agent (ChatAgent): 对话智能体实例
        action_agent (ActionAgent): 动作智能体实例
        memory_agent (MemoryAgent): 记忆智能体实例
        coordinator (AgentCoordinator): 智能体协调器
        message_bus: 消息总线实例
        config (Dict[str, Any]): 系统配置
        logger: 系统日志记录器
        is_running (bool): 系统运行状态
        
    Example:
        >>> system = RobotAgentSystem()
        >>> await system.start()
        >>> await system.run_interactive_mode()
        >>> await system.shutdown()
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化RobotAgent系统
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
            
        Raises:
            ValueError: 当配置文件无效时
            ImportError: 当依赖库未安装时
        """
        # 设置日志记录器
        self.logger = setup_logger("RobotAgentSystem")
        
        # 加载系统配置
        self.config = self._load_system_config(config_path)
        
        # 初始化系统组件
        self.chat_agent = None
        self.action_agent = None
        self.memory_agent = None
        self.coordinator = None
        self.message_bus = None
        self.volcengine_client = None
        
        # 系统状态
        self.is_running = False
        self._shutdown_event = asyncio.Event()
        
        # 注册信号处理器
        self._setup_signal_handlers()
        
        self.logger.info("RobotAgent系统初始化完成")
    
    def _load_system_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        加载系统配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            Dict[str, Any]: 系统配置字典
        """
        try:
            # 加载基础配置
            system_config = config_loader.get_system_config()
            agents_config = config_loader.get_agents_config()
            
            # 尝试加载火山API配置
            try:
                volcengine_config = config_loader.get_volcengine_config()
                system_config['volcengine'] = volcengine_config
                self.logger.info("火山API配置加载成功")
            except Exception as e:
                self.logger.warning(f"火山API配置加载失败: {e}")
                system_config['volcengine'] = None
            
            # 合并配置
            config = {
                'system': system_config,
                'agents': agents_config,
                'volcengine': system_config.get('volcengine')
            }
            
            return config
            
        except Exception as e:
            self.logger.error(f"配置加载失败: {e}")
            # 返回默认配置
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        获取默认配置
        
        Returns:
            Dict[str, Any]: 默认配置字典
        """
        return {
            'system': {
                'log_level': 'INFO',
                'max_agents': 10,
                'message_timeout': 30.0
            },
            'agents': {
                'chat_agent': {
                    'model_name': 'gpt-3.5-turbo',
                    'temperature': 0.7,
                    'max_tokens': 1000
                },
                'action_agent': {
                    'max_parallel_tasks': 5,
                    'task_timeout': 60.0
                },
                'memory_agent': {
                    'max_memory_items': 1000,
                    'cleanup_interval': 3600
                }
            },
            'volcengine': None
        }
    
    def _setup_signal_handlers(self):
        """
        设置信号处理器，用于优雅关闭系统
        """
        def signal_handler(signum, frame):
            self.logger.info(f"接收到信号 {signum}，开始关闭系统...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _init_volcengine_client(self):
        """初始化火山API客户端"""
        if not self.config.get("volcengine"):
            self.logger.warning("未配置火山API，将使用模拟模式")
            return
        
        try:
            volcengine_config = self.config["volcengine"]
            self.volcengine_client = OpenAI(
                api_key=volcengine_config["api_key"],
                base_url=volcengine_config["base_url"]
            )
            self.logger.info(f"火山API客户端初始化成功，模型: {volcengine_config['default_model']}")
        except Exception as e:
            self.logger.error(f"火山API客户端初始化失败: {e}")
            self.volcengine_client = None
    
    async def start(self) -> bool:
        """
        启动RobotAgent系统
        
        Returns:
            bool: 启动是否成功
        """
        try:
            self.logger.info("正在启动RobotAgent系统...")
            
            # 1. 初始化火山API客户端
            self._init_volcengine_client()
            
            # 2. 启动消息总线
            self.message_bus = get_message_bus()
            await self.message_bus.start()
            self.logger.info("消息总线启动成功")
            
            # 3. 创建智能体实例
            await self._create_agents()
            
            # 4. 启动智能体
            await self._start_agents()
            
            # 5. 创建并启动协调器
            await self._create_coordinator()
            
            # 6. 设置系统状态
            self.is_running = True
            
            self.logger.info("RobotAgent系统启动成功")
            return True
            
        except Exception as e:
            self.logger.error(f"系统启动失败: {e}")
            await self.shutdown()
            return False
    
    async def _create_agents(self):
        """
        创建智能体实例
        """
        try:
            # 创建对话智能体
            chat_config = self.config['agents'].get('chat_agent', {})
            
            self.chat_agent = ChatAgent(
                agent_id="chat_agent_001",
                config=chat_config,
                volcengine_client=self.volcengine_client
            )
            
            # 创建动作智能体
            action_config = self.config['agents'].get('action_agent', {})
            self.action_agent = ActionAgent(
                agent_id="action_agent_001",
                config=action_config
            )
            
            # 创建记忆智能体
            memory_config = self.config['agents'].get('memory_agent', {})
            self.memory_agent = MemoryAgent(
                agent_id="memory_agent_001",
                config=memory_config
            )
            
            self.logger.info("智能体实例创建成功")
            
        except Exception as e:
            self.logger.error(f"智能体创建失败: {e}")
            raise
    
    async def _start_agents(self):
        """
        启动所有智能体
        """
        try:
            # 并行启动所有智能体
            await asyncio.gather(
                self.chat_agent.start(),
                self.action_agent.start(),
                self.memory_agent.start()
            )
            
            self.logger.info("所有智能体启动成功")
            
        except Exception as e:
            self.logger.error(f"智能体启动失败: {e}")
            raise
    
    async def _create_coordinator(self):
        """
        创建并启动智能体协调器
        """
        try:
            # 创建协调器
            self.coordinator = AgentCoordinator()
            
            # 注册智能体
            await self.coordinator.register_agent(self.chat_agent)
            await self.coordinator.register_agent(self.action_agent)
            await self.coordinator.register_agent(self.memory_agent)
            
            # 启动协调器
            await self.coordinator.start()
            
            self.logger.info("智能体协调器启动成功")
            
        except Exception as e:
            self.logger.error(f"协调器启动失败: {e}")
            raise
    
    async def _chat_with_volcengine(self, user_message: str) -> str:
        """使用火山API进行对话"""
        if not self.volcengine_client:
            return "抱歉，火山API未配置，无法进行对话。"
        
        try:
            volcengine_config = self.config["volcengine"]
            
            # 构建消息
            messages = [
                {"role": "system", "content": "你是一个智能机器人助手，能够理解用户指令并协调执行各种任务。"},
                {"role": "user", "content": user_message}
            ]
            
            # 调用API
            completion = self.volcengine_client.chat.completions.create(
                model=volcengine_config["default_model"],
                messages=messages,
                temperature=volcengine_config.get("temperature", 0.7),
                max_tokens=volcengine_config.get("max_tokens", 2000)
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"火山API调用失败: {e}")
            return f"抱歉，处理您的请求时发生错误: {str(e)}"
    
    async def run_interactive_mode(self):
        """
        运行交互模式
        
        提供命令行交互界面，允许用户与智能体系统进行对话
        """
        self.logger.info("进入交互模式")
        
        print("\n" + "=" * 60)
        print("🤖 RobotAgent MVP 系统")
        print("=" * 60)
        
        # 显示系统状态
        if self.volcengine_client:
            volcengine_config = self.config["volcengine"]
            print(f"✅ 火山API已连接 - 模型: {volcengine_config['default_model']}")
        else:
            print("⚠️  火山API未配置，使用模拟模式")
        
        print("\n📋 系统已启动，可以开始对话")
        print("💡 输入 'help' 查看帮助信息")
        print("💡 输入 'quit' 或 'exit' 退出系统")
        print("-" * 60)
        
        while self.is_running and not self._shutdown_event.is_set():
            try:
                # 获取用户输入
                user_input = await self._get_user_input()
                
                if not user_input:
                    continue
                
                # 处理特殊命令
                if await self._handle_special_commands(user_input):
                    continue
                
                # 处理用户消息
                print("🤖 智能体正在处理...")
                
                # 优先使用火山API
                if self.volcengine_client:
                    response = await self._chat_with_volcengine(user_input)
                    print(f"🤖 系统: {response}")
                else:
                    await self._process_user_message(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 系统被用户中断")
                break
            except Exception as e:
                self.logger.error(f"交互模式错误: {e}")
                print(f"❌ 发生错误: {e}")
    
    async def _get_user_input(self) -> str:
        """
        异步获取用户输入
        
        Returns:
            str: 用户输入的文本
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, input, "\n👤 您: ")
    
    async def _handle_special_commands(self, user_input: str) -> bool:
        """
        处理特殊命令
        
        Args:
            user_input: 用户输入
            
        Returns:
            bool: 是否为特殊命令
        """
        command = user_input.strip().lower()
        
        if command in ['quit', 'exit', '退出']:
            print("👋 正在关闭系统...")
            await self.shutdown()
            return True
        
        elif command in ['help', '帮助']:
            self._show_help()
            return True
        
        elif command in ['status', '状态']:
            await self._show_status()
            return True
        
        elif command in ['clear', '清空']:
            print("\033[2J\033[H")  # 清屏
            return True
        
        return False
    
    def _show_help(self):
        """
        显示帮助信息
        """
        print("""
📖 RobotAgent MVP 帮助信息：

🎯 系统功能：
  - 智能对话 - 基于火山API的自然语言理解
  - 动作规划 - 智能分解和执行复杂任务
  - 记忆管理 - 学习和记住用户偏好
  - 多智能体协作 - 三个专业智能体协同工作

🔹 基本命令：
  - 直接输入消息与AI智能体对话
  - 'help' 或 '帮助'：显示此帮助信息
  - 'status' 或 '状态'：显示系统状态
  - 'clear' 或 '清空'：清屏
  - 'quit' 或 'exit'：退出系统

🤖 智能体介绍：
  - ChatAgent   - 对话智能体，负责理解和生成回复
  - ActionAgent - 动作智能体，负责规划和执行任务
  - MemoryAgent - 记忆智能体，负责记忆管理和学习

💡 使用提示：
  - 直接输入消息即可开始对话
  - 系统会自动选择最适合的智能体处理请求
  - 支持复杂任务的分解和协作执行

🔹 使用示例：
  - "帮我制定一个学习计划"
  - "执行文件整理任务"
  - "记住我喜欢喝咖啡"
        """)
    
    async def _show_status(self):
        """
        显示系统状态
        """
        try:
            print("\n📊 RobotAgent系统状态")
            print("-" * 50)
            
            # 系统总体状态
            print(f"🔧 系统运行状态: {'🟢 运行中' if self.is_running else '🔴 已停止'}")
            
            # 火山API状态
            if self.volcengine_client:
                volcengine_config = self.config["volcengine"]
                print(f"🌐 火山API状态: 🟢 已连接")
                print(f"   模型: {volcengine_config['default_model']}")
                print(f"   API地址: {volcengine_config['base_url']}")
            else:
                print(f"🌐 火山API状态: 🔴 未配置")
            
            # 智能体状态
            print("🤖 智能体状态：")
            if self.chat_agent:
                print(f"   ChatAgent: 🟢 运行中 (对话智能体)")
            else:
                print(f"   ChatAgent: 🔴 未创建 (对话智能体)")
                
            if self.action_agent:
                print(f"   ActionAgent: 🟢 运行中 (动作智能体)")
            else:
                print(f"   ActionAgent: 🔴 未创建 (动作智能体)")
                
            if self.memory_agent:
                print(f"   MemoryAgent: 🟢 运行中 (记忆智能体)")
            else:
                print(f"   MemoryAgent: 🔴 未创建 (记忆智能体)")
            
            # 协调器状态
            if self.coordinator:
                print(f"🎯 协调器状态: 🟢 运行中")
                try:
                    stats = await self.coordinator.get_stats()
                    print(f"   协调器统计: {stats}")
                except:
                    pass
            else:
                print(f"🎯 协调器状态: 🔴 未创建")
            
            # 消息总线状态
            if self.message_bus:
                try:
                    bus_stats = await self.message_bus.get_stats()
                    print(f"📡 消息总线状态: 🟢 正常")
                    print(f"   消息统计: {bus_stats}")
                except:
                    print(f"📡 消息总线状态: 🔴 异常")
            else:
                print(f"📡 消息总线状态: 🔴 未启动")
            
            # 配置信息
            print("\n⚙️  配置信息:")
            print(f"   日志级别: {self.config.get('system', {}).get('log_level', 'INFO')}")
            print(f"   最大智能体数: {self.config.get('system', {}).get('max_agents', 10)}")
            print(f"   消息超时: {self.config.get('system', {}).get('message_timeout', 30.0)}s")
            
            print("-" * 50)
                
        except Exception as e:
            print(f"❌ 获取状态信息失败: {e}")
    
    async def _process_user_message(self, user_input: str):
        """
        处理用户消息
        
        Args:
            user_input: 用户输入的消息
        """
        try:
            print("🤖 智能体正在处理...")
            
            # 通过协调器处理用户输入
            response = await self.coordinator.process_user_input(
                user_input=user_input,
                user_id="user_001",
                session_id="session_001"
            )
            
            # 显示响应
            if response:
                print(f"🤖 系统: {response}")
            else:
                print("🤖 系统: 抱歉，我无法处理您的请求。")
                
        except Exception as e:
            self.logger.error(f"处理用户消息失败: {e}")
            print(f"❌ 处理失败: {e}")
    
    async def shutdown(self):
        """
        关闭RobotAgent系统
        """
        if not self.is_running:
            return
        
        self.logger.info("正在关闭RobotAgent系统...")
        self.is_running = False
        self._shutdown_event.set()
        
        try:
            # 关闭协调器
            if self.coordinator:
                await self.coordinator.shutdown()
                self.logger.info("协调器已关闭")
            
            # 关闭智能体
            if self.chat_agent:
                await self.chat_agent.stop()
            if self.action_agent:
                await self.action_agent.stop()
            if self.memory_agent:
                await self.memory_agent.stop()
            self.logger.info("所有智能体已关闭")
            
            # 关闭消息总线
            if self.message_bus:
                await self.message_bus.stop()
                self.logger.info("消息总线已关闭")
            
            self.logger.info("RobotAgent系统已完全关闭")
            
        except Exception as e:
            self.logger.error(f"系统关闭过程中发生错误: {e}")


async def main():
    """
    主函数：启动RobotAgent MVP系统
    """
    system = None
    try:
        # 创建系统实例
        system = RobotAgentSystem()
        
        # 启动系统
        if await system.start():
            # 运行交互模式
            await system.run_interactive_mode()
        else:
            print("❌ 系统启动失败")
            return 1
            
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断")
    except Exception as e:
        print(f"❌ 程序运行错误: {e}")
        return 1
    finally:
        # 确保系统正确关闭
        if system:
            await system.shutdown()
    
    return 0


if __name__ == "__main__":
    # 设置事件循环策略（Windows兼容性）
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # 运行主程序
    exit_code = asyncio.run(main())
    sys.exit(exit_code)


# 变更历史:
# v0.0.1 (2025-01-20):
# - 初始实现
# - 三智能体系统集成
# - 交互模式实现
# - 配置管理和错误处理
# - 优雅关闭机制