# -*- coding: utf-8 -*-

# 智能体基类 (BaseRobotAgent)
# 基于CAMEL框架的机器人智能体基础实现，融合Eigent和OWL项目的优势
# 作者: RobotAgent开发团队
# 版本: 0.0.1 (Initial Release)
# 更新时间: 2025年8月13日

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from enum import Enum
import asyncio
import logging
import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from copy import deepcopy
import uuid
import traceback

# 导入通信协议和消息总线
from ..communication.protocols import (
    AgentMessage, MessageType, MessagePriority, TaskMessage, ResponseMessage,
    StatusMessage, CollaborationRequest, CollaborationResponse, MemoryMessage,
    ToolMessage, MessageProtocol, MessageValidator, AgentRole
)
from ..communication.message_bus import MessageBus, get_message_bus

# CAMEL框架核心组件导入
try:
    from camel.agents import ChatAgent
    from camel.messages import BaseMessage
    from camel.models import BaseModelBackend
    from camel.prompts import TextPrompt
    from camel.toolkits.base import BaseToolkit
    from camel.toolkits.function_tool import FunctionTool
    CAMEL_AVAILABLE = True
except ImportError:
    CAMEL_AVAILABLE = False
    print("警告: CAMEL框架未安装，使用模拟实现")


class AgentState(Enum):
    # 智能体状态枚举
    # 定义智能体在生命周期中的各种状态，用于状态管理和监控
    # 这是CAMEL框架"状态性"原则的具体体现 
    INITIALIZING = "initializing"  # 初始化中
    IDLE = "idle"                  # 空闲状态，等待任务
    PROCESSING = "processing"      # 处理消息中
    EXECUTING = "executing"        # 执行任务中
    COLLABORATING = "collaborating" # 与其他智能体协作中
    LEARNING = "learning"          # 学习和更新中
    ERROR = "error"                # 错误状态
    SHUTDOWN = "shutdown"          # 关闭状态


# MessageType 现在从 protocols 模块导入


class TaskStatus(Enum):
    # 任务状态枚举
    # 基于Eigent项目的任务管理机制
    PENDING = "pending"            # 待处理
    IN_PROGRESS = "in_progress"    # 进行中
    COMPLETED = "completed"        # 已完成
    FAILED = "failed"              # 失败
    CANCELLED = "cancelled"        # 已取消
    DELEGATED = "delegated"        # 已委派给其他智能体


# AgentMessage 现在从 protocols 模块导入


@dataclass
class TaskDefinition:

    # 任务定义数据结构
    # 基于Eigent项目的任务管理系统

    task_id: str
    task_type: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1              # 优先级 (1-10)
    deadline: Optional[float] = None  # 截止时间
    dependencies: List[str] = field(default_factory=list)  # 依赖的任务ID
    assigned_agent: Optional[str] = None  # 分配的智能体ID
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    result: Optional[Any] = None
    error_info: Optional[str] = None


@dataclass
class AgentCapability:

    # 智能体能力描述
    # 定义智能体的具体能力和技能，用于任务分配和协作决策
    # 这是CAMEL框架"角色扮演"机制的重要组成部分

    name: str                      # 能力名称
    description: str               # 能力描述
    input_types: List[str]         # 支持的输入类型
    output_types: List[str]        # 产生的输出类型
    confidence: float = 1.0        # 能力置信度 (0-1)
    enabled: bool = True           # 是否启用
    tool_dependencies: List[str] = field(default_factory=list)  # 依赖的工具


@dataclass
class ToolDefinition:

    # 工具定义数据结构
    # 基于Eigent项目的MCP工具集成机制

    name: str
    description: str
    function: Callable
    parameters_schema: Dict[str, Any]
    return_schema: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    permissions: List[str] = field(default_factory=list)
    category: str = "general"


# CollaborationMode 现在从 protocols 模块导入
class LegacyCollaborationMode(Enum):
    # 保留原有的协作模式定义以兼容现有代码
    ROLE_PLAYING = "role_playing"  # 角色扮演模式（用户-助手）
    PEER_TO_PEER = "peer_to_peer"  # 对等协作模式
    HIERARCHICAL = "hierarchical"  # 层次化协作模式
    SOCIETY = "society"            # 智能体社会模式


class BaseRobotAgent(ABC):

    # 机器人智能体基类
    # 
    # 这是所有RobotAgent智能体的抽象基类，融合了Eigent和OWL项目的优势：
    # 1. Eigent的工具集成能力（MCP协议支持）
    # 2. OWL的智能体协作机制（角色扮演和任务分解）
    # 3. CAMEL框架的核心功能（状态管理、消息传递、学习机制）
    # 
    # === 核心设计原则 ===
    # 
    # 1. 🧬 可进化性 (Evolvability)
    #    - 智能体系统能够通过生成数据和与环境交互来持续进化
    #    - 支持通过强化学习或监督学习驱动的自我改进
    #    - 本基类提供了学习接口和经验积累机制
    # 
    # 2. 📈 可扩展性 (Scalability)
    #    - 框架设计支持多达百万个智能体的系统
    #    - 确保大规模下的高效协调、通信和资源管理
    #    - 本基类采用异步消息传递和状态管理，支持大规模部署
    # 
    # 3. 💾 状态性 (Statefulness)
    #    - 智能体维护状态化记忆，能够执行多步骤环境交互
    #    - 高效处理复杂任务，保持上下文连续性
    #    - 本基类实现了完整的状态管理和记忆系统
    # 
    # 4. 🔧 工具集成 (Tool Integration)
    #    - 智能体与外部世界交互的函数集合
    #    - 支持MCP协议和自定义工具
    #    - 动态工具注册和权限管理
    # 
    # === 智能体架构组件 ===
    # 
    # 1. **角色扮演框架**: 每个智能体都有明确的角色定义和职责
    # 2. **消息系统**: 标准化的智能体间通信协议
    # 3. **记忆系统**: 上下文记忆和外部记忆的统一管理
    # 4. **工具集成**: 智能体与外部世界交互的函数集合
    # 5. **协作机制**: 多智能体协作和任务分解
    # 6. **学习能力**: 从交互中持续学习和改进

    
    def __init__(self, 
                 agent_id: str,
                 agent_type: str,
                 config: Dict[str, Any] = None,
                 model_config: Dict[str, Any] = None,
                 collaboration_mode: LegacyCollaborationMode = LegacyCollaborationMode.PEER_TO_PEER):

        # 初始化智能体基类
        # 
        # 这个初始化方法实现了融合架构的核心组件初始化：
        # 1. 智能体身份和配置（基础CAMEL功能）
        # 2. 状态管理系统（增强的状态跟踪）
        # 3. 消息处理机制（支持工具调用和协作）
        # 4. 工具集成系统（MCP协议支持）
        # 5. 协作机制（OWL风格的智能体协作）
        # 6. 任务管理系统（Eigent风格的任务跟踪）
        # 7. 记忆和学习系统（多层次记忆管理）

        # === 基础属性初始化 ===
        if not agent_id or not isinstance(agent_id, str):
            raise ValueError("agent_id必须是非空字符串")
        
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config or {}
        self.model_config = model_config or {}
        self.collaboration_mode = collaboration_mode
        
        # === 状态管理初始化 ===
        self._state = AgentState.INITIALIZING
        self._state_history: List[Tuple[AgentState, float]] = [(AgentState.INITIALIZING, time.time())]
        self._state_lock = asyncio.Lock()
        
        # === 消息系统初始化 ===
        self.message_bus = get_message_bus()  # 获取全局消息总线实例
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._message_task: Optional[asyncio.Task] = None
        self._message_handlers: Dict[MessageType, Callable] = {}
        self._conversation_contexts: Dict[str, List[AgentMessage]] = {}  # 对话上下文管理
        
        # === 工具集成系统初始化（基于Eigent的MCP机制）===
        self._tools: Dict[str, ToolDefinition] = {}
        self._tool_permissions: Dict[str, List[str]] = {}
        self._tool_usage_stats: Dict[str, Dict[str, Any]] = {}
        
        # === 任务管理系统初始化（基于Eigent的任务管理）===
        self._active_tasks: Dict[str, TaskDefinition] = {}
        self._task_history: List[TaskDefinition] = []
        self._task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        
        # === 协作系统初始化（基于OWL的协作机制）===
        self._collaboration_partners: Dict[str, Dict[str, Any]] = {}
        self._role_definition: Optional[str] = None
        self._collaboration_history: List[Dict[str, Any]] = []
        
        # === 记忆系统初始化 ===
        self._short_term_memory: List[AgentMessage] = []
        self._long_term_memory: Dict[str, Any] = {}
        self._episodic_memory: List[Dict[str, Any]] = []  # 情节记忆
        self._semantic_memory: Dict[str, Any] = {}        # 语义记忆
        self._memory_limit = self.config.get('memory_limit', 1000)
        
        # === 能力系统初始化 ===
        self._capabilities: List[AgentCapability] = []
        self._skill_registry: Dict[str, Callable] = {}
        
        # === 性能监控初始化 ===
        self._performance_metrics = {
            'task_stats': {
                'total_tasks': 0,
                'successful_tasks': 0,
                'failed_tasks': 0,
                'average_response_time': 0.0,
                'collaboration_count': 0
            },
            'message_stats': {
                'sent': 0,
                'received': 0,
                'processed': 0,
                'errors': 0
            },
            'tool_stats': {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0
            }
        }
        
        # === 日志系统初始化 ===
        self.logger = logging.getLogger(f"RobotAgent.{agent_type}.{agent_id}")
        self.logger.setLevel(logging.INFO)
        
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
                self._init_camel_agent()
            except Exception as e:
                self.logger.error(f"CAMEL智能体初始化失败: {e}")
        
        # === 注册默认处理器和工具 ===
        self._register_default_handlers()
        self._register_default_tools()
        
        # === 初始化完成 ===
        self.logger.info(f"智能体 {self.agent_id} ({self.agent_type}) 初始化完成")
        asyncio.create_task(self._set_state(AgentState.IDLE))
    
    def _init_camel_agent(self):

        # 初始化CAMEL智能体核心
        # 
        # 创建CAMEL框架的ChatAgent实例，这是智能体的核心"大脑"。
        # 融合了OWL项目的角色扮演机制和Eigent项目的工具集成能力。

        if not CAMEL_AVAILABLE:
            self.logger.warning("CAMEL框架不可用，使用模拟实现")
            return
        
        try:
            # 构建系统提示消息（融合OWL的角色扮演机制）
            system_prompt = self._build_system_prompt()
            
            # 创建模型后端
            model_backend = self._create_model_backend()
            
            if model_backend:
                self._camel_agent = ChatAgent(
                    system_message=BaseMessage.make_assistant_message(
                        role_name=self.agent_type,
                        content=system_prompt
                    ),
                    model=model_backend,
                    message_window_size=self.config.get('message_window_size', 10)
                )
            
            self.logger.info("CAMEL智能体核心初始化成功")
            
        except Exception as e:
            self.logger.error(f"CAMEL智能体初始化失败: {e}")
            # 在MVP阶段，即使CAMEL初始化失败也继续运行
    
    def _build_system_prompt(self) -> str:

        # 构建系统提示消息
        # 
        # 基于OWL项目的角色扮演机制，构建智能体的系统提示。
        # 根据协作模式的不同，生成不同风格的提示消息。

        # 基础角色定义
        role_definition = f"""
你是一个名为 {self.agent_id} 的 {self.agent_type} 类型智能体，
是RobotAgent多智能体系统的重要组成部分。
        """
        
        # 根据协作模式调整角色描述
        if self.collaboration_mode == CollaborationMode.ROLE_PLAYING:
            # OWL风格的角色扮演模式
            if "user" in self.agent_type.lower():
                role_definition += """
你的主要职责是分析复杂任务并将其分解为可执行的步骤，
然后指导其他智能体完成这些步骤。你应该：
1. 仔细分析任务需求
2. 制定详细的执行计划
3. 指导助手智能体使用合适的工具
4. 验证执行结果的准确性
5. 在任务完成时明确表示 <TASK_DONE>
                """
            else:
                role_definition += """
你的主要职责是根据用户智能体的指导执行具体任务。你应该：
1. 理解并执行收到的指令
2. 主动使用可用的工具解决问题
3. 提供详细的执行过程和结果
4. 在遇到问题时寻求进一步指导
5. 验证答案的准确性
                """
        
        # 能力描述
        capabilities_desc = "\n".join([
            f"- {cap.name}: {cap.description}" 
            for cap in self._capabilities
        ]) if self._capabilities else "正在学习和发展中..."
        
        # 工具描述
        tools_desc = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self._tools.values() if tool.enabled
        ]) if self._tools else "暂无可用工具"
        
        # 行为准则
        behavior_guidelines = """
你应该遵循以下行为准则：
1. 始终保持专业和友好的态度
2. 准确理解用户意图和任务要求
3. 与其他智能体积极协作
4. 主动使用工具解决问题
5. 在不确定时主动寻求帮助
6. 持续学习和改进自己的能力
7. 确保输出结果的准确性和可靠性
        """
        
        # 组合完整的系统提示
        system_prompt = f"""
{role_definition}

你的主要能力包括：
{capabilities_desc}

你可以使用的工具包括：
{tools_desc}

{behavior_guidelines}

请根据接收到的消息和任务，发挥你的专业能力，
与用户和其他智能体进行有效的交流和协作。
如果需要使用工具，请明确说明你调用了哪个工具以及调用的结果。
        """
        
        return system_prompt.strip()
    
    def _create_model_backend(self):
        """
        # 创建模型后端
        # 
        # 根据配置创建合适的模型后端。支持多种模型平台。
        """
        if not CAMEL_AVAILABLE:
            return None
        
        # 在MVP阶段，返回None，后续根据具体需求实现
        # 可以支持OpenAI、Anthropic、本地模型等
        return None
    
    def _register_default_handlers(self):

        # 注册默认消息处理器
        # 为不同类型的消息注册默认的处理函数。
        # 支持工具调用和协作消息的处理。

        self._message_handlers = {
            MessageType.TEXT: self._handle_text_message,
            MessageType.TASK: self._handle_task_message,
            MessageType.INSTRUCTION: self._handle_instruction_message,
            MessageType.TOOL_CALL: self._handle_tool_call_message,
            MessageType.TOOL_RESULT: self._handle_tool_result_message,
            MessageType.STATUS: self._handle_status_message,
            MessageType.ERROR: self._handle_error_message,
            MessageType.HEARTBEAT: self._handle_heartbeat_message,
            MessageType.COLLABORATION: self._handle_collaboration_message,
        }
    
    def _register_default_tools(self):

        # 注册默认工具
        # 注册一些基础的工具函数，为智能体提供基本的操作能力。

        # 基础工具：获取当前时间
        self.register_tool(
            name="get_current_time",
            description="获取当前时间",
            function=self._tool_get_current_time,
            parameters_schema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
        
        # 基础工具：记录日志
        self.register_tool(
            name="log_message",
            description="记录日志消息",
            function=self._tool_log_message,
            parameters_schema={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "要记录的消息"},
                    "level": {"type": "string", "enum": ["info", "warning", "error"], "default": "info"}
                },
                "required": ["message"]
            }
        )
    
    async def _set_state(self, new_state: AgentState):

        # 设置智能体状态 
        # 线程安全的状态变更方法，记录状态变更历史。

        async with self._state_lock:
            old_state = self._state
            self._state = new_state
            self._state_history.append((new_state, time.time()))
            
            # 限制状态历史长度
            if len(self._state_history) > 100:
                self._state_history = self._state_history[-50:]
            
            self.logger.debug(f"状态变更: {old_state.value} -> {new_state.value}")
    
    # === 属性访问器 ===
    
    @property
    def state(self) -> AgentState:
        # 获取当前状态
        return self._state
    
    @property
    def is_running(self) -> bool:
        # 检查智能体是否正在运行
        return self._state not in [AgentState.SHUTDOWN, AgentState.ERROR]
    
    @property
    def capabilities(self) -> List[AgentCapability]:
        # 获取智能体能力列表
        return self._capabilities.copy()
    
    @property
    def available_tools(self) -> List[str]:
        # 获取可用工具列表
        return [name for name, tool in self._tools.items() if tool.enabled]
    
    # === 生命周期管理方法 ===
    
    async def start(self):
        # 启动智能体
        # 启动智能体的主要服务：
        # 1. 注册到消息总线
        # 2. 消息处理循环
        # 3. 任务处理循环
        # 4. 心跳检测
        # 5. 状态监控

        if self._state == AgentState.SHUTDOWN:
            raise RuntimeError("无法启动已关闭的智能体")
        
        self.logger.info(f"启动智能体 {self.agent_id}")
        
        try:
            # 注册到消息总线
            await self.message_bus.register_agent(self.agent_id, self)
            
            # 启动消息处理任务
            self._message_task = asyncio.create_task(self._message_processing_loop())
            
            # 启动任务处理任务
            self._task_processing_task = asyncio.create_task(self._task_processing_loop())
            
            # 设置为空闲状态，准备接收任务
            await self._set_state(AgentState.IDLE)
            
            self.logger.info(f"智能体 {self.agent_id} 启动成功")
            
        except Exception as e:
            self.logger.error(f"智能体启动失败: {e}")
            await self._set_state(AgentState.ERROR)
            raise
    
    async def stop(self):

        # 停止智能体
        # 优雅地关闭智能体：
        # 1. 停止接收新消息
        # 2. 处理完当前任务
        # 3. 保存状态和记忆
        # 4. 释放资源

        self.logger.info(f"停止智能体 {self.agent_id}")
        
        await self._set_state(AgentState.SHUTDOWN)
        
        # 停止消息处理任务
        if hasattr(self, '_message_task') and self._message_task and not self._message_task.done():
            self._message_task.cancel()
            try:
                await self._message_task
            except asyncio.CancelledError:
                pass
        
        # 停止任务处理任务
        if hasattr(self, '_task_processing_task') and self._task_processing_task and not self._task_processing_task.done():
            self._task_processing_task.cancel()
            try:
                await self._task_processing_task
            except asyncio.CancelledError:
                pass
        
        # 保存状态
        await self._save_state()
        
        self.logger.info(f"智能体 {self.agent_id} 已停止")
    
    async def reset(self):

        # 重置智能体状态
        # 将智能体重置到初始状态：
        # 1. 清空消息队列
        # 2. 重置记忆
        # 3. 重置统计信息
        # 4. 恢复到空闲状态

        self.logger.info(f"重置智能体 {self.agent_id}")
        
        # 清空消息队列
        while not self._message_queue.empty():
            try:
                self._message_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # 清空任务队列
        while not self._task_queue.empty():
            try:
                self._task_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # 重置记忆
        self._short_term_memory.clear()
        self._conversation_contexts.clear()
        self._active_tasks.clear()
        
        # 重置统计信息
        self._performance_metrics = {
            'task_stats': {
                'total_tasks': 0,
                'successful_tasks': 0,
                'failed_tasks': 0,
                'average_response_time': 0.0,
                'collaboration_count': 0
            },
            'message_stats': {
                'sent': 0,
                'received': 0,
                'processed': 0,
                'errors': 0
            },
            'tool_stats': {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0
            }
        }
        
        await self._set_state(AgentState.IDLE)
        self.logger.info(f"智能体 {self.agent_id} 重置完成")
    
    # === 消息处理方法 ===
    
    async def send_message(self, 
                          recipient: str, 
                          content: Any, 
                          message_type: MessageType = MessageType.TEXT,
                          metadata: Dict[str, Any] = None,
                          conversation_id: str = None) -> str:
        # 发送消息给其他智能体
        # 支持多种消息类型，包括工具调用和协作请求。
        message = AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            content=content,
            message_type=message_type,
            metadata=metadata or {},
            conversation_id=conversation_id
        )
        
        # 更新统计信息
        self._performance_metrics['message_stats']['sent'] += 1
        
        # 记录到短期记忆
        self._short_term_memory.append(message)
        self._manage_memory()
        
        # 如果有对话ID，记录到对话上下文
        if conversation_id:
            if conversation_id not in self._conversation_contexts:
                self._conversation_contexts[conversation_id] = []
            self._conversation_contexts[conversation_id].append(message)
        
        self.logger.debug(f"发送消息给 {recipient}: {message_type.value}")
        
        # 通过消息总线发送消息
        await self.message_bus.send_message(message)
        return message.message_id
    
    async def receive_message(self, message: AgentMessage):

        # 接收来自其他智能体的消息
        # 将消息放入处理队列，由消息处理循环异步处理。
        await self._message_queue.put(message)
        self._performance_metrics['message_stats']['received'] += 1
        
        # 记录到短期记忆
        self._short_term_memory.append(message)
        self._manage_memory()
        
        # 如果有对话ID，记录到对话上下文
        if message.conversation_id:
            if message.conversation_id not in self._conversation_contexts:
                self._conversation_contexts[message.conversation_id] = []
            self._conversation_contexts[message.conversation_id].append(message)
    
    async def _message_processing_loop(self):

        # 消息处理循环
        # 持续处理消息队列中的消息，直到智能体停止。
        while self.is_running:
            try:
                # 从消息总线接收消息
                message = await self.message_bus.receive_message(self.agent_id, timeout=1.0)
                if message is None:
                    continue
                
                await self._process_message(message)
                
            except asyncio.TimeoutError:
                # 超时是正常的，继续循环
                continue
            except Exception as e:
                self.logger.error(f"消息处理循环错误: {e}")
                self._performance_metrics['message_stats']['errors'] += 1
    
    async def _process_message(self, message: AgentMessage):

        # 处理单个消息
        # 根据消息类型调用相应的处理器。
        try:
            await self._set_state(AgentState.PROCESSING)
            
            # 获取对应的处理器
            handler = self._message_handlers.get(message.message_type)
            if handler:
                await handler(message)
            else:
                self.logger.warning(f"未找到消息类型 {message.message_type} 的处理器")
            
            self._performance_metrics['message_stats']['processed'] += 1
            
        except Exception as e:
            self.logger.error(f"处理消息失败: {e}")
            self._performance_metrics['message_stats']['errors'] += 1
            
            # 发送错误响应
            await self.send_message(
                recipient=message.sender,
                content=f"处理消息时发生错误: {str(e)}",
                message_type=MessageType.ERROR,
                conversation_id=message.conversation_id
            )
        finally:
            await self._set_state(AgentState.IDLE)
    
    # === 消息处理器实现 ===
    
    async def _handle_text_message(self, message: AgentMessage):

        # 处理文本消息
        # 使用CAMEL智能体处理自然语言消息。
        if self._camel_agent:
            try:
                # 使用CAMEL智能体生成响应
                response = await self._generate_camel_response(message.content)
                
                # 发送响应
                await self.send_message(
                    recipient=message.sender,
                    content=response,
                    message_type=MessageType.RESPONSE,
                    conversation_id=message.conversation_id
                )
            except Exception as e:
                self.logger.error(f"CAMEL智能体响应生成失败: {e}")
        else:
            # 简单的回显响应
            await self.send_message(
                recipient=message.sender,
                content=f"收到消息: {message.content}",
                message_type=MessageType.RESPONSE,
                conversation_id=message.conversation_id
            )
    
    async def _handle_task_message(self, message: AgentMessage):

        # 处理任务消息
        # 创建新任务并加入任务队列。
        try:
            task_data = message.content
            if isinstance(task_data, dict):
                task = TaskDefinition(
                    task_id=task_data.get('task_id', str(uuid.uuid4())),
                    task_type=task_data.get('task_type', 'general'),
                    description=task_data.get('description', ''),
                    parameters=task_data.get('parameters', {}),
                    priority=task_data.get('priority', 1),
                    assigned_agent=self.agent_id
                )
                
                # 添加到任务队列
                await self._task_queue.put((task.priority, task))
                self._active_tasks[task.task_id] = task
                
                self.logger.info(f"接收到新任务: {task.task_id}")
                
                # 发送确认响应
                await self.send_message(
                    recipient=message.sender,
                    content={
                        'status': 'accepted',
                        'task_id': task.task_id
                    },
                    message_type=MessageType.STATUS,
                    conversation_id=message.conversation_id
                )
            else:
                raise ValueError("任务数据格式无效")
                
        except Exception as e:
            self.logger.error(f"处理任务消息失败: {e}")
            await self.send_message(
                recipient=message.sender,
                content=f"任务处理失败: {str(e)}",
                message_type=MessageType.ERROR,
                conversation_id=message.conversation_id
            )
    
    async def _handle_instruction_message(self, message: AgentMessage):

        # 处理指令消息（OWL风格的用户智能体指令）
        # 这是OWL项目角色扮演模式的核心机制。
        try:
            instruction = message.content
            
            # 检查是否是任务完成指令
            if "TASK_DONE" in str(instruction):
                self.logger.info("收到任务完成指令")
                # 处理任务完成逻辑
                await self._handle_task_completion(message)
                return
            
            # 处理具体指令
            response = await self._execute_instruction(instruction)
            
            # 发送执行结果
            await self.send_message(
                recipient=message.sender,
                content=response,
                message_type=MessageType.RESPONSE,
                conversation_id=message.conversation_id
            )
            
        except Exception as e:
            self.logger.error(f"处理指令消息失败: {e}")
            await self.send_message(
                recipient=message.sender,
                content=f"指令执行失败: {str(e)}",
                message_type=MessageType.ERROR,
                conversation_id=message.conversation_id
            )
    
    async def _handle_tool_call_message(self, message: AgentMessage):

        # 处理工具调用消息
        # 执行工具调用并返回结果。
        try:
            tool_call_data = message.content
            tool_name = tool_call_data.get('tool_name')
            parameters = tool_call_data.get('parameters', {})
            
            # 执行工具调用
            result = await self.call_tool(tool_name, parameters)
            
            # 发送工具执行结果
            await self.send_message(
                recipient=message.sender,
                content={
                    'tool_name': tool_name,
                    'result': result,
                    'success': True
                },
                message_type=MessageType.TOOL_RESULT,
                conversation_id=message.conversation_id
            )
            
        except Exception as e:
            self.logger.error(f"工具调用失败: {e}")
            await self.send_message(
                recipient=message.sender,
                content={
                    'tool_name': tool_call_data.get('tool_name', 'unknown'),
                    'error': str(e),
                    'success': False
                },
                message_type=MessageType.TOOL_RESULT,
                conversation_id=message.conversation_id
            )
    
    async def _handle_tool_result_message(self, message: AgentMessage):
        # 处理工具执行结果消息
        # 
        # 记录工具执行结果，用于后续处理。
        result_data = message.content
        tool_name = result_data.get('tool_name')
        success = result_data.get('success', False)
        
        if success:
            self.logger.info(f"工具 {tool_name} 执行成功")
        else:
            self.logger.error(f"工具 {tool_name} 执行失败: {result_data.get('error')}")
        
        # 更新工具使用统计
        if tool_name not in self._tool_usage_stats:
            self._tool_usage_stats[tool_name] = {'success': 0, 'failure': 0}
        
        if success:
            self._tool_usage_stats[tool_name]['success'] += 1
            self._performance_metrics['tool_stats']['successful_calls'] += 1
        else:
            self._tool_usage_stats[tool_name]['failure'] += 1
            self._performance_metrics['tool_stats']['failed_calls'] += 1
    
    async def _handle_status_message(self, message: AgentMessage):
        # 处理状态消息
        # 
        # 更新其他智能体的状态信息。
        status_data = message.content
        sender_id = message.sender
        
        # 更新协作伙伴状态
        if sender_id not in self._collaboration_partners:
            self._collaboration_partners[sender_id] = {}
        
        self._collaboration_partners[sender_id].update({
            'last_status': status_data,
            'last_update': time.time()
        })
        
        self.logger.debug(f"更新智能体 {sender_id} 状态: {status_data}")
    
    async def _handle_error_message(self, message: AgentMessage):
        # 处理错误消息
        # 
        # 记录错误信息，必要时采取恢复措施。
        error_info = message.content
        sender_id = message.sender
        
        self.logger.error(f"收到来自 {sender_id} 的错误消息: {error_info}")
        
        # 记录错误到协作历史
        self._collaboration_history.append({
            'type': 'error',
            'sender': sender_id,
            'content': error_info,
            'timestamp': time.time()
        })
    
    async def _handle_heartbeat_message(self, message: AgentMessage):
        # 处理心跳消息
        # 
        # 响应心跳检测，维护连接状态。
        # 发送心跳响应
        await self.send_message(
            recipient=message.sender,
            content={'status': 'alive', 'timestamp': time.time()},
            message_type=MessageType.HEARTBEAT
        )
    
    async def _handle_collaboration_message(self, message: AgentMessage):
        # 处理协作消息
        # 
        # 处理智能体间的协作请求。
        collaboration_data = message.content
        collaboration_type = collaboration_data.get('type')
        
        if collaboration_type == 'request':
            # 处理协作请求
            await self._handle_collaboration_request(message)
        elif collaboration_type == 'response':
            # 处理协作响应
            await self._handle_collaboration_response(message)
        else:
            self.logger.warning(f"未知的协作消息类型: {collaboration_type}")
    
    # === 任务处理方法 ===
    
    async def _task_processing_loop(self):
        # 任务处理循环
        # 持续处理任务队列中的任务。
        while self.is_running:
            try:
                # 等待任务，设置超时避免无限阻塞
                priority, task = await asyncio.wait_for(
                    self._task_queue.get(),
                    timeout=1.0
                )
                
                await self._execute_task(task)
                
            except asyncio.TimeoutError:
                # 超时是正常的，继续循环
                continue
            except Exception as e:
                self.logger.error(f"任务处理循环错误: {e}")
    
    async def _execute_task(self, task: TaskDefinition):
        # 执行具体任务
        # 这是一个抽象方法，由子类实现具体的任务执行逻辑。
        try:
            await self._set_state(AgentState.EXECUTING)
            
            start_time = time.time()
            
            # 更新任务状态
            task.status = TaskStatus.IN_PROGRESS
            task.updated_at = time.time()
            
            # 调用抽象方法执行任务
            result = await self.execute_task(task)
            
            # 更新任务结果
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.updated_at = time.time()
            
            # 更新统计信息
            execution_time = time.time() - start_time
            self._update_task_stats(True, execution_time)
            
            # 移动到历史记录
            self._task_history.append(task)
            if task.task_id in self._active_tasks:
                del self._active_tasks[task.task_id]
            
            self.logger.info(f"任务 {task.task_id} 执行完成")
            
        except Exception as e:
            # 任务执行失败
            task.status = TaskStatus.FAILED
            task.error_info = str(e)
            task.updated_at = time.time()
            
            self._update_task_stats(False, 0)
            
            self.logger.error(f"任务 {task.task_id} 执行失败: {e}")
            
        finally:
            await self._set_state(AgentState.IDLE)
    
    def _update_task_stats(self, success: bool, execution_time: float):
        # 更新任务统计信息
        stats = self._performance_metrics['task_stats']
        stats['total_tasks'] += 1
        
        if success:
            stats['successful_tasks'] += 1
            # 更新平均响应时间
            total_time = stats['average_response_time'] * (stats['successful_tasks'] - 1)
            stats['average_response_time'] = (total_time + execution_time) / stats['successful_tasks']
        else:
            stats['failed_tasks'] += 1
    
    # === 工具集成方法 ===
    
    def register_tool(self, 
                     name: str, 
                     description: str, 
                     function: Callable,
                     parameters_schema: Dict[str, Any],
                     return_schema: Dict[str, Any] = None,
                     permissions: List[str] = None,
                     category: str = "general"):
        # 注册工具函数
        # 基于Eigent项目的MCP工具集成机制。
        tool = ToolDefinition(
            name=name,
            description=description,
            function=function,
            parameters_schema=parameters_schema,
            return_schema=return_schema or {},
            permissions=permissions or [],
            category=category
        )
        
        self._tools[name] = tool
        self.logger.info(f"注册工具: {name}")
    
    def unregister_tool(self, name: str):
        # 注销工具函数
        if name in self._tools:
            del self._tools[name]
            self.logger.info(f"注销工具: {name}")
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        # 调用工具函数
        # 执行指定的工具函数并返回结果。
        if tool_name not in self._tools:
            raise ValueError(f"工具 {tool_name} 不存在")
        
        tool = self._tools[tool_name]
        if not tool.enabled:
            raise ValueError(f"工具 {tool_name} 已禁用")
        
        try:
            # 验证参数（简化版本）
            # 在实际实现中应该根据parameters_schema进行详细验证
            
            # 更新统计信息
            self._performance_metrics['tool_stats']['total_calls'] += 1
            
            # 调用工具函数
            if asyncio.iscoroutinefunction(tool.function):
                result = await tool.function(**parameters)
            else:
                result = tool.function(**parameters)
            
            # 记录成功调用
            if tool_name not in self._tool_usage_stats:
                self._tool_usage_stats[tool_name] = {'success': 0, 'failure': 0}
            self._tool_usage_stats[tool_name]['success'] += 1
            
            self.logger.debug(f"工具 {tool_name} 调用成功")
            return result
            
        except Exception as e:
            # 记录失败调用
            if tool_name not in self._tool_usage_stats:
                self._tool_usage_stats[tool_name] = {'success': 0, 'failure': 0}
            self._tool_usage_stats[tool_name]['failure'] += 1
            
            self.logger.error(f"工具 {tool_name} 调用失败: {e}")
            raise
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        # 获取工具信息
        if tool_name not in self._tools:
            return None
        
        tool = self._tools[tool_name]
        return {
            'name': tool.name,
            'description': tool.description,
            'parameters_schema': tool.parameters_schema,
            'return_schema': tool.return_schema,
            'enabled': tool.enabled,
            'category': tool.category,
            'usage_stats': self._tool_usage_stats.get(tool_name, {'success': 0, 'failure': 0})
        }
    
    # === 默认工具实现 ===
    
    def _tool_get_current_time(self) -> str:
        # 获取当前时间的工具函数
        return datetime.now().isoformat()
    
    def _tool_log_message(self, message: str, level: str = "info"):
        # 记录日志的工具函数
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        return f"已记录{level}级别日志: {message}"
    
    # === 协作方法 ===
    
    async def request_collaboration(self, 
                                  partner_id: str, 
                                  collaboration_type: str,
                                  data: Dict[str, Any]) -> str:
        # 请求与其他智能体协作
        # 基于OWL项目的智能体协作机制。
        collaboration_request = {
            'type': 'request',
            'collaboration_type': collaboration_type,
            'data': data,
            'requester': self.agent_id,
            'timestamp': time.time()
        }
        
        message_id = await self.send_message(
            recipient=partner_id,
            content=collaboration_request,
            message_type=MessageType.COLLABORATION
        )
        
        # 记录协作请求
        self._collaboration_history.append({
            'type': 'request_sent',
            'partner': partner_id,
            'collaboration_type': collaboration_type,
            'message_id': message_id,
            'timestamp': time.time()
        })
        
        self._performance_metrics['task_stats']['collaboration_count'] += 1
        
        return message_id
    
    async def _handle_collaboration_request(self, message: AgentMessage):
        # 处理协作请求
        collaboration_data = message.content
        collaboration_type = collaboration_data.get('collaboration_type')
        requester = collaboration_data.get('requester')
        
        # 决定是否接受协作请求
        accept = await self._should_accept_collaboration(collaboration_data)
        
        response_data = {
            'type': 'response',
            'collaboration_type': collaboration_type,
            'accepted': accept,
            'responder': self.agent_id,
            'timestamp': time.time()
        }
        
        if accept:
            # 处理协作逻辑
            result = await self._process_collaboration(collaboration_data)
            response_data['result'] = result
        
        # 发送协作响应
        await self.send_message(
            recipient=requester,
            content=response_data,
            message_type=MessageType.COLLABORATION,
            conversation_id=message.conversation_id
        )
    
    async def _handle_collaboration_response(self, message: AgentMessage):
        """
        # 处理协作响应
        """
        response_data = message.content
        accepted = response_data.get('accepted', False)
        
        if accepted:
            self.logger.info(f"协作请求被 {message.sender} 接受")
            # 处理协作结果
            result = response_data.get('result')
            if result:
                await self._process_collaboration_result(result)
        else:
            self.logger.info(f"协作请求被 {message.sender} 拒绝")
    
    async def _should_accept_collaboration(self, collaboration_data: Dict[str, Any]) -> bool:
        """
        # 决定是否接受协作请求
        # 
        # 子类可以重写此方法实现自定义的协作策略。
        """
        # 默认接受所有协作请求
        return True
    
    async def _process_collaboration(self, collaboration_data: Dict[str, Any]) -> Any:
        # 处理协作任务
        # 
        # 子类应该重写此方法实现具体的协作逻辑。
        return {"status": "协作处理完成"}
    
    async def _process_collaboration_result(self, result: Any):
        # 处理协作结果
        # 
        # 处理从其他智能体收到的协作结果。
        self.logger.info(f"收到协作结果: {result}")
        
        # 记录到协作历史
        self._collaboration_history.append({
            'type': 'result_received',
            'result': result,
            'timestamp': time.time()
        })
    
    async def _handle_task_completion(self, message: AgentMessage):
        # 处理任务完成消息
        # 
        # 当收到TASK_DONE指令时的处理逻辑。
        self.logger.info("处理任务完成指令")
        
        # 更新所有活跃任务为完成状态
        for task in self._active_tasks.values():
            if task.status == TaskStatus.IN_PROGRESS:
                task.status = TaskStatus.COMPLETED
                task.updated_at = time.time()
        
        # 发送确认响应
        await self.send_message(
            recipient=message.sender,
            content="任务完成确认",
            message_type=MessageType.RESPONSE,
            conversation_id=message.conversation_id
        )
    
    async def _execute_instruction(self, instruction: str) -> str:
        # 执行具体指令
        # 
        # 基于OWL项目的指令执行机制。
        # 子类可以重写此方法实现具体的指令处理逻辑。
         try:
             # 简单的指令解析和执行
             if "时间" in instruction or "time" in instruction.lower():
                 return await self.call_tool("get_current_time", {})
             elif "日志" in instruction or "log" in instruction.lower():
                 return await self.call_tool("log_message", {"message": instruction})
             else:
                 # 使用CAMEL智能体处理复杂指令
                 if self._camel_agent:
                     return await self._generate_camel_response(instruction)
                 else:
                     return f"收到指令: {instruction}，正在处理..."
         except Exception as e:
             self.logger.error(f"指令执行失败: {e}")
             return f"指令执行失败: {str(e)}"
     
    async def _generate_camel_response(self, content: str) -> str:
        # 使用CAMEL智能体生成响应
        # 
        # 调用CAMEL框架的ChatAgent生成自然语言响应。
        if not self._camel_agent:
            return "CAMEL智能体不可用"
        
        try:
            # 创建用户消息
            user_message = BaseMessage.make_user_message(
                role_name="user",
                content=content
            )
            
            # 获取响应
            response = self._camel_agent.step(user_message)
            
            if response and hasattr(response, 'msg') and response.msg:
                return response.msg.content
            else:
                return "抱歉，我无法生成合适的响应。"
                
        except Exception as e:
            self.logger.error(f"CAMEL响应生成失败: {e}")
            return f"响应生成失败: {str(e)}"
    
    def _manage_memory(self):
        # 管理记忆系统
        # 
        # 维护记忆容量限制，清理过期记忆。
        # 限制短期记忆大小
        if len(self._short_term_memory) > self._memory_limit:
            # 移除最旧的记忆
            removed_count = len(self._short_term_memory) - self._memory_limit
            removed_messages = self._short_term_memory[:removed_count]
            self._short_term_memory = self._short_term_memory[removed_count:]
            
            # 将重要记忆转移到长期记忆
            for msg in removed_messages:
                if msg.message_type in [MessageType.TASK, MessageType.COLLABORATION]:
                    memory_key = f"important_{msg.message_id}"
                    self._long_term_memory[memory_key] = {
                        'message': msg,
                        'archived_at': time.time(),
                        'importance': 'high'
                    }
        
        # 清理过期的对话上下文（保留最近24小时）
        current_time = time.time()
        expired_conversations = []
        
        for conv_id, messages in self._conversation_contexts.items():
            if messages and (current_time - messages[-1].timestamp) > 86400:  # 24小时
                expired_conversations.append(conv_id)
        
        for conv_id in expired_conversations:
            del self._conversation_contexts[conv_id]
    
    async def _save_state(self):
        # 保存智能体状态
        # 将重要的状态信息持久化，用于恢复和分析。
        try:
            state_data = {
                'agent_id': self.agent_id,
                'agent_type': self.agent_type,
                'collaboration_mode': self.collaboration_mode.value,
                'state_history': [(state.value, timestamp) for state, timestamp in self._state_history],
                'performance_metrics': self._performance_metrics,
                'long_term_memory': self._long_term_memory,
                'collaboration_history': self._collaboration_history,
                'tool_usage_stats': self._tool_usage_stats,
                'timestamp': time.time()
            }
            
            # 在实际实现中，这里应该保存到数据库或文件
            # 在MVP阶段，我们先记录日志
            self.logger.info(f"保存状态: {len(self._long_term_memory)} 条长期记忆")
            
        except Exception as e:
            self.logger.error(f"保存状态失败: {e}")
    
    # === 公共接口方法 ===
    
    def get_status(self) -> Dict[str, Any]:
        # 获取智能体状态信息
        return {
             'agent_id': self.agent_id,
             'agent_type': self.agent_type,
             'state': self._state.value,
             'collaboration_mode': self.collaboration_mode.value,
             'uptime': time.time() - self._state_history[0][1] if self._state_history else 0,
             'capabilities': [cap.name for cap in self._capabilities],
             'available_tools': self.available_tools,
             'performance_metrics': self._performance_metrics.copy(),
             'memory_usage': {
                 'short_term': len(self._short_term_memory),
                 'long_term': len(self._long_term_memory),
                 'conversations': len(self._conversation_contexts),
                 'limit': self._memory_limit
             },
             'active_tasks': len(self._active_tasks),
             'collaboration_partners': len(self._collaboration_partners)
         }
     
    def get_memory_summary(self) -> Dict[str, Any]:
        # 获取记忆摘要
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
             },
             'long_term_entries': len(self._long_term_memory),
             'active_conversations': len(self._conversation_contexts)
         }
     
    def add_capability(self, capability: AgentCapability):
        # 添加新能力
        self._capabilities.append(capability)
        self.logger.info(f"添加新能力: {capability.name}")
    
    def remove_capability(self, capability_name: str) -> bool:
        # 移除能力
        for i, cap in enumerate(self._capabilities):
            if cap.name == capability_name:
                removed = self._capabilities.pop(i)
                self.logger.info(f"移除能力: {removed.name}")
                return True
        return False
     
    async def learn_from_interaction(self, interaction_data: Dict[str, Any]):
        # 从交互中学习
        # 这是CAMEL框架"可进化性"原则的体现，智能体可以从经验中学习和改进。
        await self._set_state(AgentState.LEARNING)
        
        try:
            # 记录学习事件到情节记忆
            learning_episode = {
                'type': 'learning_event',
                'data': interaction_data,
                'timestamp': time.time(),
                'context': {
                    'state': self._state.value,
                    'active_tasks': len(self._active_tasks),
                    'recent_performance': self._performance_metrics['task_stats']
                }
            }
            
            self._episodic_memory.append(learning_episode)
            
            # 更新语义记忆
            interaction_type = interaction_data.get('type', 'unknown')
            if interaction_type not in self._semantic_memory:
                self._semantic_memory[interaction_type] = {
                    'count': 0,
                    'success_rate': 0.0,
                    'patterns': []
                }
            
            self._semantic_memory[interaction_type]['count'] += 1
            
            # 记录学习事件
            self.logger.info(f"从交互中学习: {interaction_type}")
            
            # 更新长期记忆
            timestamp = time.time()
            self._long_term_memory[f"learning_{timestamp}"] = learning_episode
             
        except Exception as e:
            self.logger.error(f"学习过程中出错: {e}")
        finally:
            await self._set_state(AgentState.IDLE)
     
    # === 抽象方法 - 必须由子类实现 ===
    
    @abstractmethod
    async def execute_task(self, task: TaskDefinition) -> Any:
        # 执行分配的任务
        # 
        # 这是智能体执行具体任务的抽象方法，每个智能体类型都有不同的任务执行逻辑。
        # 子类必须实现此方法来定义具体的任务处理流程。
        # 
        # Args:
        #     task (TaskDefinition): 任务定义对象，包含任务的所有信息
        # 
        # Returns:
        #     Any: 任务执行结果
        # 
        # Raises:
        #     NotImplementedError: 子类必须实现此方法
        raise NotImplementedError("子类必须实现 execute_task 方法")
    
    # === 字符串表示方法 ===
    
    def __str__(self) -> str:
        # 字符串表示
        return f"BaseRobotAgent(id={self.agent_id}, type={self.agent_type}, state={self._state.value})"
    
    def __repr__(self) -> str:
        # 详细字符串表示
        return (
            f"BaseRobotAgent("
            f"id='{self.agent_id}', "
            f"type='{self.agent_type}', "
            f"state={self._state.value}, "
            f"capabilities={len(self._capabilities)}, "
            f"tools={len(self._tools)}, "
            f"memory={len(self._short_term_memory)}, "
            f"mode={self.collaboration_mode.value}"
            f")"
        )


# === 工具函数和辅助类 ===

class AgentFactory:
    # 智能体工厂类
    # 
    # 用于创建不同类型的智能体实例。
    # 这是工厂模式的实现，简化智能体的创建过程。
    
    @staticmethod
    def create_agent(agent_type: str, 
                    agent_id: str, 
                    config: Dict[str, Any] = None,
                    collaboration_mode: CollaborationMode = CollaborationMode.PEER_TO_PEER) -> 'BaseRobotAgent':
        # 创建智能体实例
        # 
        # Args:
        #     agent_type (str): 智能体类型
        #     agent_id (str): 智能体ID
        #     config (Dict[str, Any], optional): 配置参数
        #     collaboration_mode (CollaborationMode): 协作模式
        # 
        # Returns:
        #     BaseRobotAgent: 智能体实例
        # 
        # Raises:
        #     ValueError: 当智能体类型不支持时
        # 在实际实现中，这里会根据agent_type创建具体的智能体类
        # 例如：ChatAgent, ActionAgent, MemoryAgent等
        
        # 在MVP阶段，我们先返回基类的模拟实现
        if agent_type in ['chat', 'action', 'memory', 'perception', 'planning', 'ros2']:
            # 这里应该导入并创建具体的智能体类
            # 暂时抛出异常，表示需要具体实现
            raise NotImplementedError(f"智能体类型 {agent_type} 的具体实现尚未完成")
        else:
            raise ValueError(f"不支持的智能体类型: {agent_type}")


def load_agent_config(config_path: str) -> Dict[str, Any]:
    # 加载智能体配置
    # 
    # Args:
    #     config_path (str): 配置文件路径
    # 
    # Returns:
    #     Dict[str, Any]: 配置数据
    # 
    # Raises:
    #     FileNotFoundError: 当配置文件不存在时
    #     ValueError: 当配置格式无效时
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
    # 测试代码
    # 
    # 这里提供了基本的测试代码，演示如何使用BaseRobotAgent类。
    # 在实际项目中，应该有专门的测试文件。
    import asyncio
    
    # 创建一个简单的测试智能体
    class TestAgent(BaseRobotAgent):
        # 测试智能体
        
        async def execute_task(self, task: TaskDefinition) -> Any:
            # 简单的任务执行
            await asyncio.sleep(0.1)  # 模拟任务执行时间
            return {"result": f"已完成任务: {task.description}"}
    
    async def test_agent():
        # 测试函数
        # 创建测试智能体
        agent = TestAgent(
            agent_id="test_agent_001",
            agent_type="test",
            config={"memory_limit": 100},
            collaboration_mode=CollaborationMode.ROLE_PLAYING
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
        
        # 测试工具调用
        current_time = await agent.call_tool("get_current_time", {})
        print(f"当前时间: {current_time}")
        
        # 获取状态
        status = agent.get_status()
        print(f"智能体状态: {status}")
        
        # 获取记忆摘要
        memory = agent.get_memory_summary()
        print(f"记忆摘要: {memory}")
        
        # 停止智能体
        await agent.stop()
    
    # 运行测试
    print("开始测试 BaseRobotAgent...")
    asyncio.run(test_agent())
    print("测试完成！")