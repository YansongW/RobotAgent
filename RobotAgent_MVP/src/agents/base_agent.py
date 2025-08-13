# -*- coding: utf-8 -*-
"""
智能体基类 (BaseRobotAgent)
基于CAMEL框架的机器人智能体基础实现

=== 设计理念与架构说明 ===

本文件实现了RobotAgent项目的核心智能体基类，基于CAMEL (Communicative Agents for 
"Mind" Exploration of Large Language Model Society) 框架设计。CAMEL是世界上第一个
多智能体系统框架，专注于研究大规模语言模型社会中智能体的行为、能力和潜在风险。

=== CAMEL框架四大核心设计原则 ===

1. 🧬 可进化性 (Evolvability)
   - 智能体系统能够通过生成数据和与环境交互来持续进化
   - 支持通过强化学习或监督学习驱动的自我改进
   - 本基类提供了学习接口和经验积累机制

2. 📈 可扩展性 (Scalability)  
   - 框架设计支持多达百万个智能体的系统
   - 确保大规模下的高效协调、通信和资源管理
   - 本基类采用异步消息传递和状态管理，支持大规模部署

3. 💾 状态性 (Statefulness)
   - 智能体维护状态化记忆，能够执行多步骤环境交互
   - 高效处理复杂任务，保持上下文连续性
   - 本基类实现了完整的状态管理和记忆系统

4. 📖 代码即提示 (Code-as-Prompt)
   - 每行代码和注释都作为智能体的提示
   - 代码应该清晰可读，确保人类和智能体都能有效解释
   - 本文件的详细注释正是这一原则的体现

=== 智能体架构组件说明 ===

本基类实现了CAMEL框架的核心组件：

1. **角色扮演框架**: 每个智能体都有明确的角色定义和职责
2. **消息系统**: 标准化的智能体间通信协议
3. **记忆系统**: 上下文记忆和外部记忆的统一管理
4. **工具集成**: 智能体与外部世界交互的函数集合
5. **推理能力**: 规划和奖励学习机制

=== 技术实现原理 ===

1. **基于LLM的智能体核心**:
   - 使用大语言模型作为智能体的"大脑"
   - 支持多种模型平台（OpenAI、Anthropic、本地模型等）
   - 可配置的模型参数（温度、最大token、提示词等）

2. **异步消息传递系统**:
   - 基于asyncio的异步编程模型
   - 支持智能体间的实时通信和协作
   - 消息队列和事件驱动的架构

3. **状态管理机制**:
   - 智能体生命周期管理（初始化、运行、暂停、停止）
   - 任务状态跟踪和错误恢复
   - 性能监控和资源管理

4. **记忆与学习系统**:
   - 短期记忆：对话历史和上下文状态
   - 长期记忆：经验积累和知识库
   - 学习机制：从交互中不断改进

=== 使用场景与扩展 ===

本基类设计为抽象基类，需要被具体的智能体类继承：
- ChatAgent: 处理自然语言对话
- ActionAgent: 执行具体动作和任务
- MemoryAgent: 管理记忆和学习
- PerceptionAgent: 处理感知和环境理解
- PlanningAgent: 进行任务规划和决策
- ROS2Agent: 与机器人硬件交互

每个具体智能体都会实现自己特定的业务逻辑，但共享相同的基础架构和通信协议。

=== 安全与可靠性 ===

1. **错误处理**: 完善的异常捕获和恢复机制
2. **资源管理**: 内存和计算资源的合理分配
3. **安全检查**: 输入验证和输出过滤
4. **监控日志**: 详细的运行日志和性能指标

作者: RobotAgent开发团队
版本: 1.0.0 (MVP)
更新时间: 2024年
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Union
from enum import Enum
import asyncio
import logging
import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

# CAMEL框架核心组件导入
# 注意：在MVP阶段，我们先定义接口，后续集成真实的CAMEL组件
try:
    # 尝试导入CAMEL框架组件
    from camel.agents import ChatAgent
    from camel.messages import BaseMessage
    from camel.models import BaseModelBackend
    from camel.prompts import TextPrompt
    CAMEL_AVAILABLE = True
except ImportError:
    # 如果CAMEL未安装，使用模拟类
    CAMEL_AVAILABLE = False
    print("警告: CAMEL框架未安装，使用模拟实现")


class AgentState(Enum):
    """
    智能体状态枚举
    
    定义智能体在生命周期中的各种状态，用于状态管理和监控。
    这是CAMEL框架"状态性"原则的具体体现。
    """
    INITIALIZING = "initializing"  # 初始化中
    IDLE = "idle"                  # 空闲状态，等待任务
    PROCESSING = "processing"      # 处理消息中
    EXECUTING = "executing"        # 执行任务中
    LEARNING = "learning"          # 学习和更新中
    ERROR = "error"                # 错误状态
    SHUTDOWN = "shutdown"          # 关闭状态


class MessageType(Enum):
    """
    消息类型枚举
    
    定义智能体间通信的消息类型，支持多种交互模式。
    这是CAMEL框架通信机制的基础。
    """
    TEXT = "text"                  # 文本消息
    TASK = "task"                  # 任务分配
    RESPONSE = "response"          # 响应消息
    STATUS = "status"              # 状态更新
    ERROR = "error"                # 错误报告
    HEARTBEAT = "heartbeat"        # 心跳检测


@dataclass
class AgentMessage:
    """
    智能体消息数据结构
    
    标准化的消息格式，确保智能体间通信的一致性和可靠性。
    包含消息的所有必要信息：发送者、接收者、内容、类型、时间戳等。
    """
    sender: str                    # 发送者ID
    recipient: str                 # 接收者ID
    content: Any                   # 消息内容
    message_type: MessageType      # 消息类型
    timestamp: float = field(default_factory=time.time)  # 时间戳
    message_id: str = field(default_factory=lambda: f"msg_{int(time.time()*1000)}")  # 消息ID
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据


@dataclass
class AgentCapability:
    """
    智能体能力描述
    
    定义智能体的具体能力和技能，用于任务分配和协作决策。
    这是CAMEL框架"角色扮演"机制的重要组成部分。
    """
    name: str                      # 能力名称
    description: str               # 能力描述
    input_types: List[str]         # 支持的输入类型
    output_types: List[str]        # 产生的输出类型
    confidence: float = 1.0        # 能力置信度 (0-1)
    enabled: bool = True           # 是否启用


class BaseRobotAgent(ABC):
    """
    机器人智能体基类
    
    这是所有RobotAgent智能体的抽象基类，实现了CAMEL框架的核心功能：
    1. 智能体生命周期管理
    2. 消息传递和通信
    3. 状态管理和监控
    4. 记忆和学习机制
    5. 工具集成和扩展
    
    === 设计模式说明 ===
    
    本类采用了以下设计模式：
    1. **抽象工厂模式**: 通过抽象方法定义智能体接口
    2. **观察者模式**: 通过消息总线实现智能体间通信
    3. **状态模式**: 通过AgentState管理智能体状态
    4. **策略模式**: 通过可配置的处理策略适应不同场景
    
    === 核心方法说明 ===
    
    1. **生命周期方法**:
       - __init__(): 初始化智能体
       - start(): 启动智能体
       - stop(): 停止智能体
       - reset(): 重置智能体状态
    
    2. **通信方法**:
       - send_message(): 发送消息
       - receive_message(): 接收消息
       - process_message(): 处理消息（抽象方法）
    
    3. **任务执行方法**:
       - execute_task(): 执行任务（抽象方法）
       - get_capabilities(): 获取能力列表
    
    4. **学习方法**:
       - learn_from_interaction(): 从交互中学习
       - update_knowledge(): 更新知识库
    """
    
    def __init__(self, 
                 agent_id: str,
                 agent_type: str,
                 config: Dict[str, Any] = None,
                 model_config: Dict[str, Any] = None):
        """
        初始化智能体基类
        
        这个初始化方法实现了CAMEL框架的核心组件初始化：
        1. 智能体身份和配置
        2. 状态管理系统
        3. 消息处理机制
        4. 记忆系统
        5. 日志和监控
        
        Args:
            agent_id (str): 智能体唯一标识符，用于消息路由和识别
            agent_type (str): 智能体类型，如"chat", "action", "memory"等
            config (Dict[str, Any], optional): 智能体配置参数
            model_config (Dict[str, Any], optional): 模型配置参数
        
        Raises:
            ValueError: 当agent_id为空或配置无效时
            RuntimeError: 当初始化失败时
        """
        # === 基础属性初始化 ===
        if not agent_id or not isinstance(agent_id, str):
            raise ValueError("agent_id必须是非空字符串")
        
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config or {}
        self.model_config = model_config or {}
        
        # === 状态管理初始化 ===
        # 智能体当前状态，初始为初始化状态
        self._state = AgentState.INITIALIZING
        # 状态变更历史，用于调试和监控
        self._state_history: List[tuple] = [(AgentState.INITIALIZING, time.time())]
        # 状态锁，确保状态变更的线程安全
        self._state_lock = asyncio.Lock()
        
        # === 消息系统初始化 ===
        # 消息队列，存储待处理的消息
        self._message_queue: asyncio.Queue = asyncio.Queue()
        # 消息处理任务，用于异步消息处理
        self._message_task: Optional[asyncio.Task] = None
        # 消息处理器映射，不同类型消息对应不同处理器
        self._message_handlers: Dict[MessageType, Callable] = {}
        
        # === 记忆系统初始化 ===
        # 短期记忆：最近的对话历史和上下文
        self._short_term_memory: List[AgentMessage] = []
        # 长期记忆：持久化的知识和经验
        self._long_term_memory: Dict[str, Any] = {}
        # 记忆容量限制，防止内存溢出
        self._memory_limit = self.config.get('memory_limit', 1000)
        
        # === 能力系统初始化 ===
        # 智能体能力列表，定义智能体可以执行的任务类型
        self._capabilities: List[AgentCapability] = []
        # 工具函数列表，智能体可以调用的外部函数
        self._tools: Dict[str, Callable] = {}
        
        # === 性能监控初始化 ===
        # 任务执行统计
        self._task_stats = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'average_response_time': 0.0
        }
        # 消息统计
        self._message_stats = {
            'sent': 0,
            'received': 0,
            'processed': 0,
            'errors': 0
        }
        
        # === 日志系统初始化 ===
        # 创建专用的日志记录器
        self.logger = logging.getLogger(f"RobotAgent.{agent_type}.{agent_id}")
        self.logger.setLevel(logging.INFO)
        
        # 如果没有处理器，添加控制台处理器
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # === CAMEL智能体核心初始化 ===
        self._camel_agent = None
        if CAMEL_AVAILABLE:
            try:
                # 初始化CAMEL ChatAgent
                # 这是智能体的"大脑"，负责理解和生成自然语言
                self._init_camel_agent()
            except Exception as e:
                self.logger.error(f"CAMEL智能体初始化失败: {e}")
                # 在MVP阶段，即使CAMEL初始化失败也继续运行
        
        # === 注册默认消息处理器 ===
        self._register_default_handlers()
        
        # === 初始化完成 ===
        self.logger.info(f"智能体 {self.agent_id} ({self.agent_type}) 初始化完成")
        self._set_state(AgentState.IDLE)
    
    def _init_camel_agent(self):
        """
        初始化CAMEL智能体核心
        
        这个方法创建CAMEL框架的ChatAgent实例，这是智能体的核心"大脑"。
        ChatAgent负责：
        1. 自然语言理解和生成
        2. 上下文管理
        3. 推理和决策
        4. 与大语言模型的交互
        
        在MVP阶段，如果CAMEL框架不可用，我们使用模拟实现。
        """
        if not CAMEL_AVAILABLE:
            self.logger.warning("CAMEL框架不可用，使用模拟实现")
            return
        
        try:
            # 构建系统提示消息
            # 这是CAMEL框架"代码即提示"原则的体现
            system_prompt = self._build_system_prompt()
            
            # 创建CAMEL ChatAgent
            # 这里会根据配置选择不同的模型后端
            model_backend = self._create_model_backend()
            
            self._camel_agent = ChatAgent(
                system_message=system_prompt,
                model=model_backend,
                message_window_size=self.config.get('message_window_size', 10)
            )
            
            self.logger.info("CAMEL智能体核心初始化成功")
            
        except Exception as e:
            self.logger.error(f"CAMEL智能体初始化失败: {e}")
            raise RuntimeError(f"无法初始化CAMEL智能体: {e}")
    
    def _build_system_prompt(self) -> str:
        """
        构建系统提示消息
        
        这个方法根据智能体的类型和配置构建系统提示消息。
        系统提示是智能体"人格"和"能力"的定义，告诉大语言模型：
        1. 你是谁（角色定义）
        2. 你能做什么（能力描述）
        3. 你应该如何行为（行为准则）
        4. 你的目标是什么（任务目标）
        
        Returns:
            str: 格式化的系统提示消息
        """
        # 基础角色定义
        role_definition = f"""
你是一个名为 {self.agent_id} 的 {self.agent_type} 类型智能体，
是RobotAgent多智能体系统的重要组成部分。
        """
        
        # 能力描述
        capabilities_desc = "\n".join([
            f"- {cap.name}: {cap.description}" 
            for cap in self._capabilities
        ]) if self._capabilities else "正在学习和发展中..."
        
        # 行为准则
        behavior_guidelines = """
你应该遵循以下行为准则：
1. 始终保持专业和友好的态度
2. 准确理解用户意图和任务要求
3. 与其他智能体积极协作
4. 在不确定时主动寻求帮助
5. 持续学习和改进自己的能力
        """
        
        # 组合完整的系统提示
        system_prompt = f"""
{role_definition}

你的主要能力包括：
{capabilities_desc}

{behavior_guidelines}

请根据接收到的消息和任务，发挥你的专业能力，
与用户和其他智能体进行有效的交流和协作。
        """
        
        return system_prompt.strip()
    
    def _create_model_backend(self):
        """
        创建模型后端
        
        根据配置创建合适的模型后端。在MVP阶段，我们支持：
        1. OpenAI GPT系列
        2. 本地模型
        3. 模拟模型（用于测试）
        
        Returns:
            BaseModelBackend: 模型后端实例
        """
        if not CAMEL_AVAILABLE:
            return None
        
        # 这里应该根据model_config创建具体的模型后端
        # 在MVP阶段，我们先返回None，后续实现
        return None
    
    def _register_default_handlers(self):
        """
        注册默认消息处理器
        
        为不同类型的消息注册默认的处理函数。
        这是消息驱动架构的核心，确保每种消息都有对应的处理逻辑。
        """
        self._message_handlers = {
            MessageType.TEXT: self._handle_text_message,
            MessageType.TASK: self._handle_task_message,
            MessageType.STATUS: self._handle_status_message,
            MessageType.ERROR: self._handle_error_message,
            MessageType.HEARTBEAT: self._handle_heartbeat_message,
        }
    
    async def _set_state(self, new_state: AgentState):
        """
        设置智能体状态
        
        线程安全的状态变更方法，记录状态变更历史。
        
        Args:
            new_state (AgentState): 新的状态
        """
        async with self._state_lock:
            old_state = self._state
            self._state = new_state
            self._state_history.append((new_state, time.time()))
            
            # 限制状态历史长度
            if len(self._state_history) > 100:
                self._state_history = self._state_history[-50:]
            
            self.logger.debug(f"状态变更: {old_state.value} -> {new_state.value}")
    
    @property
    def state(self) -> AgentState:
        """获取当前状态"""
        return self._state
    
    @property
    def is_running(self) -> bool:
        """检查智能体是否正在运行"""
        return self._state not in [AgentState.SHUTDOWN, AgentState.ERROR]
    
    async def start(self):
        """
        启动智能体
        
        启动智能体的主要服务：
        1. 消息处理循环
        2. 心跳检测
        3. 状态监控
        
        这个方法实现了智能体的"生命"开始。
        """
        if self._state == AgentState.SHUTDOWN:
            raise RuntimeError("无法启动已关闭的智能体")
        
        self.logger.info(f"启动智能体 {self.agent_id}")
        
        try:
            # 启动消息处理任务
            self._message_task = asyncio.create_task(self._message_processing_loop())
            
            # 设置为空闲状态，准备接收任务
            await self._set_state(AgentState.IDLE)
            
            self.logger.info(f"智能体 {self.agent_id} 启动成功")
            
        except Exception as e:
            self.logger.error(f"智能体启动失败: {e}")
            await self._set_state(AgentState.ERROR)
            raise
    
    async def stop(self):
        """
        停止智能体
        
        优雅地关闭智能体：
        1. 停止接收新消息
        2. 处理完当前任务
        3. 保存状态和记忆
        4. 释放资源
        """
        self.logger.info(f"停止智能体 {self.agent_id}")
        
        await self._set_state(AgentState.SHUTDOWN)
        
        # 停止消息处理任务
        if self._message_task and not self._message_task.done():
            self._message_task.cancel()
            try:
                await self._message_task
            except asyncio.CancelledError:
                pass
        
        # 保存状态（如果需要持久化）
        await self._save_state()
        
        self.logger.info(f"智能体 {self.agent_id} 已停止")
    
    async def reset(self):
        """
        重置智能体状态
        
        将智能体重置到初始状态：
        1. 清空消息队列
        2. 重置记忆
        3. 重置统计信息
        4. 恢复到空闲状态
        """
        self.logger.info(f"重置智能体 {self.agent_id}")
        
        # 清空消息队列
        while not self._message_queue.empty():
            try:
                self._message_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # 重置记忆
        self._short_term_memory.clear()
        
        # 重置统计信息
        self._task_stats = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'average_response_time': 0.0
        }
        self._message_stats = {
            'sent': 0,
            'received': 0,
            'processed': 0,
            'errors': 0
        }
        
        await self._set_state(AgentState.IDLE)
        self.logger.info(f"智能体 {self.agent_id} 重置完成")
    
    async def send_message(self, 
                          recipient: str, 
                          content: Any, 
                          message_type: MessageType = MessageType.TEXT,
                          metadata: Dict[str, Any] = None) -> str:
        """
        发送消息到其他智能体
        
        这是智能体间通信的核心方法，实现了CAMEL框架的通信机制。
        
        Args:
            recipient (str): 接收者智能体ID
            content (Any): 消息内容
            message_type (MessageType): 消息类型
            metadata (Dict[str, Any], optional): 消息元数据
        
        Returns:
            str: 消息ID
        
        Raises:
            RuntimeError: 当智能体未运行时
        """
        if not self.is_running:
            raise RuntimeError("智能体未运行，无法发送消息")
        
        # 创建消息对象
        message = AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            content=content,
            message_type=message_type,
            metadata=metadata or {}
        )
        
        # 记录到短期记忆
        self._add_to_memory(message)
        
        # 更新统计信息
        self._message_stats['sent'] += 1
        
        # 在实际实现中，这里应该通过消息总线发送消息
        # 在MVP阶段，我们先记录日志
        self.logger.info(
            f"发送消息 {message.message_id} 到 {recipient}: "
            f"{message_type.value} - {str(content)[:100]}..."
        )
        
        return message.message_id
    
    async def receive_message(self, message: AgentMessage):
        """
        接收消息
        
        将接收到的消息放入处理队列。
        
        Args:
            message (AgentMessage): 接收到的消息
        """
        if not self.is_running:
            self.logger.warning(f"智能体未运行，忽略消息 {message.message_id}")
            return
        
        # 将消息放入队列
        await self._message_queue.put(message)
        
        # 更新统计信息
        self._message_stats['received'] += 1
        
        self.logger.debug(f"接收消息 {message.message_id} 来自 {message.sender}")
    
    async def _message_processing_loop(self):
        """
        消息处理循环
        
        这是智能体的"心脏"，持续处理接收到的消息。
        实现了异步、非阻塞的消息处理机制。
        """
        self.logger.info("启动消息处理循环")
        
        while self.is_running:
            try:
                # 等待消息，设置超时避免无限等待
                message = await asyncio.wait_for(
                    self._message_queue.get(), 
                    timeout=1.0
                )
                
                # 处理消息
                await self._process_message_internal(message)
                
            except asyncio.TimeoutError:
                # 超时是正常的，继续循环
                continue
            except Exception as e:
                self.logger.error(f"消息处理循环错误: {e}")
                await asyncio.sleep(0.1)  # 短暂休息后继续
        
        self.logger.info("消息处理循环已停止")
    
    async def _process_message_internal(self, message: AgentMessage):
        """
        内部消息处理方法
        
        根据消息类型调用相应的处理器。
        
        Args:
            message (AgentMessage): 要处理的消息
        """
        start_time = time.time()
        
        try:
            await self._set_state(AgentState.PROCESSING)
            
            # 记录到短期记忆
            self._add_to_memory(message)
            
            # 根据消息类型选择处理器
            handler = self._message_handlers.get(message.message_type)
            if handler:
                await handler(message)
            else:
                # 调用抽象方法，由子类实现
                await self.process_message(message)
            
            # 更新统计信息
            self._message_stats['processed'] += 1
            processing_time = time.time() - start_time
            self._update_response_time(processing_time)
            
            await self._set_state(AgentState.IDLE)
            
        except Exception as e:
            self.logger.error(f"处理消息 {message.message_id} 时出错: {e}")
            self._message_stats['errors'] += 1
            await self._set_state(AgentState.ERROR)
            
            # 发送错误响应
            await self.send_message(
                recipient=message.sender,
                content=f"处理消息时出错: {str(e)}",
                message_type=MessageType.ERROR
            )
    
    def _add_to_memory(self, message: AgentMessage):
        """
        将消息添加到短期记忆
        
        实现记忆管理，保持对话上下文。
        
        Args:
            message (AgentMessage): 要记录的消息
        """
        self._short_term_memory.append(message)
        
        # 限制记忆大小，移除最旧的记忆
        if len(self._short_term_memory) > self._memory_limit:
            removed = self._short_term_memory.pop(0)
            self.logger.debug(f"移除旧记忆: {removed.message_id}")
    
    def _update_response_time(self, processing_time: float):
        """
        更新平均响应时间
        
        Args:
            processing_time (float): 本次处理时间
        """
        current_avg = self._task_stats['average_response_time']
        total_processed = self._message_stats['processed']
        
        if total_processed == 1:
            self._task_stats['average_response_time'] = processing_time
        else:
            # 计算移动平均
            self._task_stats['average_response_time'] = (
                (current_avg * (total_processed - 1) + processing_time) / total_processed
            )
    
    # === 默认消息处理器 ===
    
    async def _handle_text_message(self, message: AgentMessage):
        """处理文本消息"""
        self.logger.info(f"收到文本消息: {message.content}")
        # 默认实现：记录日志
        # 子类可以重写此方法实现具体逻辑
    
    async def _handle_task_message(self, message: AgentMessage):
        """处理任务消息"""
        self.logger.info(f"收到任务: {message.content}")
        
        try:
            # 调用抽象方法执行任务
            result = await self.execute_task(message.content)
            
            # 发送结果
            await self.send_message(
                recipient=message.sender,
                content=result,
                message_type=MessageType.RESPONSE
            )
            
            self._task_stats['successful_tasks'] += 1
            
        except Exception as e:
            self.logger.error(f"任务执行失败: {e}")
            self._task_stats['failed_tasks'] += 1
            
            # 发送错误响应
            await self.send_message(
                recipient=message.sender,
                content=f"任务执行失败: {str(e)}",
                message_type=MessageType.ERROR
            )
        
        self._task_stats['total_tasks'] += 1
    
    async def _handle_status_message(self, message: AgentMessage):
        """处理状态消息"""
        self.logger.debug(f"收到状态更新: {message.content}")
    
    async def _handle_error_message(self, message: AgentMessage):
        """处理错误消息"""
        self.logger.error(f"收到错误报告: {message.content}")
    
    async def _handle_heartbeat_message(self, message: AgentMessage):
        """处理心跳消息"""
        # 响应心跳
        await self.send_message(
            recipient=message.sender,
            content={"status": self._state.value, "timestamp": time.time()},
            message_type=MessageType.HEARTBEAT
        )
    
    # === 抽象方法 - 必须由子类实现 ===
    
    @abstractmethod
    async def process_message(self, message: AgentMessage) -> Any:
        """
        处理接收到的消息
        
        这是智能体的核心业务逻辑，每个具体的智能体类型都必须实现此方法。
        
        Args:
            message (AgentMessage): 接收到的消息
        
        Returns:
            Any: 处理结果
        
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("子类必须实现 process_message 方法")
    
    @abstractmethod
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行分配的任务
        
        这是智能体执行具体任务的方法，每个智能体类型都有不同的任务执行逻辑。
        
        Args:
            task (Dict[str, Any]): 任务描述和参数
        
        Returns:
            Dict[str, Any]: 任务执行结果
        
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("子类必须实现 execute_task 方法")
    
    # === 公共接口方法 ===
    
    def get_capabilities(self) -> List[AgentCapability]:
        """
        获取智能体能力列表
        
        Returns:
            List[AgentCapability]: 能力列表
        """
        return self._capabilities.copy()
    
    def add_capability(self, capability: AgentCapability):
        """
        添加新能力
        
        Args:
            capability (AgentCapability): 新能力
        """
        self._capabilities.append(capability)
        self.logger.info(f"添加新能力: {capability.name}")
    
    def remove_capability(self, capability_name: str) -> bool:
        """
        移除能力
        
        Args:
            capability_name (str): 能力名称
        
        Returns:
            bool: 是否成功移除
        """
        for i, cap in enumerate(self._capabilities):
            if cap.name == capability_name:
                removed = self._capabilities.pop(i)
                self.logger.info(f"移除能力: {removed.name}")
                return True
        return False
    
    def add_tool(self, name: str, func: Callable):
        """
        添加工具函数
        
        Args:
            name (str): 工具名称
            func (Callable): 工具函数
        """
        self._tools[name] = func
        self.logger.info(f"添加工具: {name}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取智能体状态信息
        
        Returns:
            Dict[str, Any]: 状态信息
        """
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'state': self._state.value,
            'uptime': time.time() - self._state_history[0][1],
            'capabilities': [cap.name for cap in self._capabilities],
            'task_stats': self._task_stats.copy(),
            'message_stats': self._message_stats.copy(),
            'memory_usage': {
                'short_term': len(self._short_term_memory),
                'limit': self._memory_limit
            }
        }
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        获取记忆摘要
        
        Returns:
            Dict[str, Any]: 记忆摘要
        """
        recent_messages = self._short_term_memory[-10:] if self._short_term_memory else []
        
        return {
            'total_messages': len(self._short_term_memory),
            'recent_messages': [
                {
                    'sender': msg.sender,
                    'type': msg.message_type.value,
                    'timestamp': msg.timestamp,
                    'content_preview': str(msg.content)[:50] + '...' if len(str(msg.content)) > 50 else str(msg.content)
                }
                for msg in recent_messages
            ],
            'message_types': {
                msg_type.value: sum(1 for msg in self._short_term_memory if msg.message_type == msg_type)
                for msg_type in MessageType
            }
        }
    
    async def learn_from_interaction(self, interaction_data: Dict[str, Any]):
        """
        从交互中学习
        
        这是CAMEL框架"可进化性"原则的体现，智能体可以从经验中学习和改进。
        
        Args:
            interaction_data (Dict[str, Any]): 交互数据
        """
        await self._set_state(AgentState.LEARNING)
        
        try:
            # 在这里实现学习逻辑
            # 例如：更新模型参数、调整策略、积累经验等
            
            # 记录学习事件
            self.logger.info(f"从交互中学习: {interaction_data.get('type', 'unknown')}")
            
            # 更新长期记忆
            timestamp = time.time()
            self._long_term_memory[f"learning_{timestamp}"] = {
                'data': interaction_data,
                'timestamp': timestamp,
                'type': 'learning_event'
            }
            
        except Exception as e:
            self.logger.error(f"学习过程中出错: {e}")
        finally:
            await self._set_state(AgentState.IDLE)
    
    async def _save_state(self):
        """
        保存智能体状态
        
        将重要的状态信息持久化，用于恢复和分析。
        """
        try:
            state_data = {
                'agent_id': self.agent_id,
                'agent_type': self.agent_type,
                'state_history': [(state.value, timestamp) for state, timestamp in self._state_history],
                'task_stats': self._task_stats,
                'message_stats': self._message_stats,
                'long_term_memory': self._long_term_memory,
                'timestamp': time.time()
            }
            
            # 在实际实现中，这里应该保存到数据库或文件
            # 在MVP阶段，我们先记录日志
            self.logger.info(f"保存状态: {len(self._long_term_memory)} 条长期记忆")
            
        except Exception as e:
            self.logger.error(f"保存状态失败: {e}")
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"BaseRobotAgent(id={self.agent_id}, type={self.agent_type}, state={self._state.value})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (
            f"BaseRobotAgent("
            f"id='{self.agent_id}', "
            f"type='{self.agent_type}', "
            f"state={self._state.value}, "
            f"capabilities={len(self._capabilities)}, "
            f"memory={len(self._short_term_memory)}"
            f")"
        )


# === 工具函数和辅助类 ===

class AgentFactory:
    """
    智能体工厂类
    
    用于创建不同类型的智能体实例。
    这是工厂模式的实现，简化智能体的创建过程。
    """
    
    @staticmethod
    def create_agent(agent_type: str, 
                    agent_id: str, 
                    config: Dict[str, Any] = None) -> BaseRobotAgent:
        """
        创建智能体实例
        
        Args:
            agent_type (str): 智能体类型
            agent_id (str): 智能体ID
            config (Dict[str, Any], optional): 配置参数
        
        Returns:
            BaseRobotAgent: 智能体实例
        
        Raises:
            ValueError: 当智能体类型不支持时
        """
        # 在实际实现中，这里会根据agent_type创建具体的智能体类
        # 例如：ChatAgent, ActionAgent, MemoryAgent等
        
        # 在MVP阶段，我们先返回基类的模拟实现
        if agent_type in ['chat', 'action', 'memory', 'perception', 'planning', 'ros2']:
            # 这里应该导入并创建具体的智能体类
            # 暂时返回None，表示需要具体实现
            raise NotImplementedError(f"智能体类型 {agent_type} 的具体实现尚未完成")
        else:
            raise ValueError(f"不支持的智能体类型: {agent_type}")


def load_agent_config(config_path: str) -> Dict[str, Any]:
    """
    加载智能体配置
    
    Args:
        config_path (str): 配置文件路径
    
    Returns:
        Dict[str, Any]: 配置数据
    
    Raises:
        FileNotFoundError: 当配置文件不存在时
        ValueError: 当配置格式无效时
    """
    import yaml
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"配置文件格式错误: {e}")


if __name__ == "__main__":
    """
    测试代码
    
    这里提供了基本的测试代码，演示如何使用BaseRobotAgent类。
    在实际项目中，应该有专门的测试文件。
    """
    import asyncio
    
    # 创建一个简单的测试智能体
    class TestAgent(BaseRobotAgent):
        """测试智能体"""
        
        async def process_message(self, message: AgentMessage) -> Any:
            """简单的消息处理"""
            return f"已处理消息: {message.content}"
        
        async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
            """简单的任务执行"""
            return {"result": f"已完成任务: {task.get('name', 'unknown')}"}
    
    async def test_agent():
        """测试函数"""
        # 创建测试智能体
        agent = TestAgent(
            agent_id="test_agent_001",
            agent_type="test",
            config={"memory_limit": 100}
        )
        
        # 添加能力
        agent.add_capability(AgentCapability(
            name="test_capability",
            description="测试能力",
            input_types=["text"],
            output_types=["text"]
        ))
        
        # 启动智能体
        await agent.start()
        
        # 发送测试消息
        await agent.send_message(
            recipient="test_recipient",
            content="Hello, World!",
            message_type=MessageType.TEXT
        )
        
        # 获取状态
        status = agent.get_status()
        print(f"智能体状态: {status}")
        
        # 停止智能体
        await agent.stop()
    
    # 运行测试
    print("开始测试 BaseRobotAgent...")
    asyncio.run(test_agent())
    print("测试完成！")