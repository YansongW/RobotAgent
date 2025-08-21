# -*- coding: utf-8 -*-

# 消息类型定义模块 (Message Types Definition)
# 统一定义系统中所有消息类型、协议和数据结构
# 作者: RobotAgent开发团队
# 版本: 0.0.1 (Initial Release)
# 更新时间: 2025-08-21
# 基于框架: CAMEL框架集成的消息通信系统

# 导入标准库
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Union, Callable
import uuid
import json


class MessageType(Enum):
    """消息类型枚举
    
    定义系统中所有支持的消息类型，用于消息路由和处理
    """
    # 基础消息类型
    TEXT = "text"                   # 文本消息
    COMMAND = "command"             # 命令消息
    INSTRUCTION = "instruction"     # 指令消息
    RESPONSE = "response"           # 响应消息
    ERROR = "error"                 # 错误消息
    STATUS = "status"               # 状态消息
    
    # 任务相关消息
    TASK = "task"                   # 任务消息（通用）
    TASK_REQUEST = "task_request"   # 任务请求
    TASK_RESPONSE = "task_response" # 任务响应
    TASK_UPDATE = "task_update"     # 任务更新
    TASK_COMPLETE = "task_complete" # 任务完成
    TASK_CANCEL = "task_cancel"     # 任务取消
    
    # 智能体通信消息
    AGENT_REGISTER = "agent_register"   # 智能体注册
    AGENT_HEARTBEAT = "agent_heartbeat" # 智能体心跳
    AGENT_STATUS = "agent_status"       # 智能体状态
    AGENT_SHUTDOWN = "agent_shutdown"   # 智能体关闭
    HEARTBEAT = "heartbeat"             # 心跳消息
    
    # 协作消息类型
    COLLABORATION_REQUEST = "collaboration_request"   # 协作请求
    COLLABORATION_RESPONSE = "collaboration_response" # 协作响应
    
    # 工具调用消息
    TOOL_CALL = "tool_call"         # 工具调用
    TOOL_RESULT = "tool_result"     # 工具结果
    TOOL_ERROR = "tool_error"       # 工具错误
    
    # 记忆系统消息
    MEMORY_STORE = "memory_store"   # 记忆存储
    MEMORY_RETRIEVE = "memory_retrieve" # 记忆检索
    MEMORY_UPDATE = "memory_update" # 记忆更新
    MEMORY_DELETE = "memory_delete" # 记忆删除
    
    # 系统控制消息
    SYSTEM_START = "system_start"   # 系统启动
    SYSTEM_STOP = "system_stop"     # 系统停止
    SYSTEM_CONFIG = "system_config" # 系统配置
    SYSTEM_STATUS = "system_status" # 系统状态
    
    # 多模态消息
    IMAGE = "image"                 # 图像消息
    AUDIO = "audio"                 # 音频消息
    VIDEO = "video"                 # 视频消息
    FILE = "file"                   # 文件消息


class MessagePriority(Enum):
    """消息优先级枚举
    
    定义消息处理的优先级，用于消息队列排序
    """
    LOW = 1         # 低优先级
    NORMAL = 2      # 普通优先级
    MEDIUM = 3      # 中等优先级
    HIGH = 4        # 高优先级
    URGENT = 5      # 紧急优先级
    CRITICAL = 6    # 关键优先级


class MessageStatus(Enum):
    """消息状态枚举
    
    定义消息在处理过程中的状态
    """
    PENDING = "pending"         # 待处理
    PROCESSING = "processing"   # 处理中
    COMPLETED = "completed"     # 已完成
    FAILED = "failed"           # 处理失败
    TIMEOUT = "timeout"         # 超时
    CANCELLED = "cancelled"     # 已取消


class IntentType(Enum):
    """意图类型枚举
    
    定义用户意图的分类，用于智能体理解和响应
    """
    QUESTION = "question"           # 问题询问
    COMMAND = "command"             # 命令执行
    CONVERSATION = "conversation"   # 对话交流
    TASK_ASSIGNMENT = "task_assignment" # 任务分配
    INFORMATION_REQUEST = "information_request" # 信息请求
    HELP_REQUEST = "help_request"   # 帮助请求
    GREETING = "greeting"           # 问候
    FAREWELL = "farewell"           # 告别
    UNKNOWN = "unknown"             # 未知意图


class TaskStatus(Enum):
    """任务状态枚举
    
    定义任务执行过程中的状态
    """
    CREATED = "created"         # 已创建
    QUEUED = "queued"           # 已排队
    ASSIGNED = "assigned"       # 已分配
    RUNNING = "running"         # 执行中
    PAUSED = "paused"           # 已暂停
    COMPLETED = "completed"     # 已完成
    FAILED = "failed"           # 执行失败
    CANCELLED = "cancelled"     # 已取消
    TIMEOUT = "timeout"         # 执行超时


class MemoryType(Enum):
    """记忆类型枚举
    
    定义记忆系统中的记忆分类
    """
    EPISODIC = "episodic"       # 情景记忆
    SEMANTIC = "semantic"       # 语义记忆
    PROCEDURAL = "procedural"   # 程序记忆
    WORKING = "working"         # 工作记忆
    LONG_TERM = "long_term"     # 长期记忆
    SHORT_TERM = "short_term"   # 短期记忆


class MemoryPriority(Enum):
    """记忆优先级枚举
    
    定义记忆的重要性级别
    """
    LOW = 1         # 低重要性
    NORMAL = 2      # 普通重要性
    HIGH = 3        # 高重要性
    CRITICAL = 4    # 关键重要性


@dataclass
class BaseMessage:
    """基础消息类
    
    所有消息类型的基础数据结构，包含消息的基本属性
    
    主要功能:
    - 消息唯一标识和时间戳管理
    - 消息类型和优先级定义
    - 消息状态跟踪
    - 消息元数据存储
    
    继承关系: 作为所有具体消息类的基类
    """
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.TEXT
    priority: MessagePriority = MessagePriority.NORMAL
    status: MessageStatus = MessageStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.now)
    sender_id: Optional[str] = None
    receiver_id: Optional[str] = None
    content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式
        
        Returns:
            消息字典表示
        """
        return {
            'message_id': self.message_id,
            'message_type': self.message_type.value,
            'priority': self.priority.value,
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'content': self.content,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseMessage':
        """从字典创建消息实例
        
        Args:
            data: 消息数据字典
            
        Returns:
            消息实例
        """
        return cls(
            message_id=data.get('message_id', str(uuid.uuid4())),
            message_type=MessageType(data.get('message_type', 'text')),
            priority=MessagePriority(data.get('priority', 2)),
            status=MessageStatus(data.get('status', 'pending')),
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
            sender_id=data.get('sender_id'),
            receiver_id=data.get('receiver_id'),
            content=data.get('content'),
            metadata=data.get('metadata', {})
        )


@dataclass
class AgentMessage(BaseMessage):
    """智能体消息类
    
    智能体间通信的标准消息格式
    
    主要功能:
    - 智能体身份标识
    - 消息内容和上下文管理
    - 响应回调机制
    - 消息链追踪
    
    继承关系: 继承自BaseMessage
    """
    agent_type: Optional[str] = None
    conversation_id: Optional[str] = None
    parent_message_id: Optional[str] = None
    requires_response: bool = False
    response_timeout: Optional[int] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def create_response(self, content: Any, 
                      message_type: MessageType = MessageType.RESPONSE) -> 'AgentMessage':
        """创建响应消息
        
        Args:
            content: 响应内容
            message_type: 消息类型
            
        Returns:
            响应消息实例
        """
        return AgentMessage(
            message_type=message_type,
            sender_id=self.receiver_id,
            receiver_id=self.sender_id,
            content=content,
            conversation_id=self.conversation_id,
            parent_message_id=self.message_id,
            context=self.context.copy()
        )


@dataclass
class TaskMessage(BaseMessage):
    """任务消息类
    
    任务相关的消息格式，用于任务分配、执行和状态更新
    
    主要功能:
    - 任务定义和参数传递
    - 任务状态跟踪
    - 任务结果返回
    - 任务依赖关系管理
    
    继承关系: 继承自BaseMessage
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = ""
    task_status: TaskStatus = TaskStatus.CREATED
    task_params: Dict[str, Any] = field(default_factory=dict)
    task_result: Optional[Any] = None
    task_error: Optional[str] = None
    assigned_agent: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    deadline: Optional[datetime] = None
    
    def update_status(self, status: TaskStatus, result: Optional[Any] = None, 
                     error: Optional[str] = None) -> None:
        """更新任务状态
        
        Args:
            status: 新状态
            result: 任务结果
            error: 错误信息
        """
        self.task_status = status
        if result is not None:
            self.task_result = result
        if error is not None:
            self.task_error = error
        self.timestamp = datetime.now()


@dataclass
class ResponseMessage(BaseMessage):
    """响应消息类
    
    用于返回处理结果和状态信息的消息格式
    
    主要功能:
    - 处理结果返回
    - 错误信息传递
    - 状态码定义
    - 响应时间记录
    
    继承关系: 继承自BaseMessage
    """
    request_id: Optional[str] = None
    status_code: int = 200
    success: bool = True
    error_message: Optional[str] = None
    response_data: Optional[Any] = None
    processing_time: Optional[float] = None
    
    @classmethod
    def create_success(cls, request_id: str, data: Any = None) -> 'ResponseMessage':
        """创建成功响应
        
        Args:
            request_id: 请求ID
            data: 响应数据
            
        Returns:
            成功响应消息
        """
        return cls(
            message_type=MessageType.RESPONSE,
            request_id=request_id,
            status_code=200,
            success=True,
            response_data=data
        )
    
    @classmethod
    def create_error(cls, request_id: str, error_message: str, 
                    status_code: int = 500) -> 'ResponseMessage':
        """创建错误响应
        
        Args:
            request_id: 请求ID
            error_message: 错误信息
            status_code: 状态码
            
        Returns:
            错误响应消息
        """
        return cls(
            message_type=MessageType.ERROR,
            request_id=request_id,
            status_code=status_code,
            success=False,
            error_message=error_message
        )


@dataclass
class ToolMessage(BaseMessage):
    """工具消息类
    
    工具调用和结果返回的消息格式
    
    主要功能:
    - 工具调用参数传递
    - 工具执行结果返回
    - 工具错误处理
    - 工具性能监控
    
    继承关系: 继承自BaseMessage
    """
    tool_name: str = ""
    tool_action: str = ""
    tool_params: Dict[str, Any] = field(default_factory=dict)
    tool_result: Optional[Any] = None
    tool_error: Optional[str] = None
    execution_time: Optional[float] = None
    
    @classmethod
    def create_call(cls, tool_name: str, action: str, 
                   params: Dict[str, Any] = None) -> 'ToolMessage':
        """创建工具调用消息
        
        Args:
            tool_name: 工具名称
            action: 工具动作
            params: 调用参数
            
        Returns:
            工具调用消息
        """
        return cls(
            message_type=MessageType.TOOL_CALL,
            tool_name=tool_name,
            tool_action=action,
            tool_params=params or {}
        )
    
    @classmethod
    def create_result(cls, tool_name: str, result: Any, 
                     execution_time: float = None) -> 'ToolMessage':
        """创建工具结果消息
        
        Args:
            tool_name: 工具名称
            result: 执行结果
            execution_time: 执行时间
            
        Returns:
            工具结果消息
        """
        return cls(
            message_type=MessageType.TOOL_RESULT,
            tool_name=tool_name,
            tool_result=result,
            execution_time=execution_time
        )


@dataclass
class MemoryItem:
    """记忆项数据结构
    
    记忆系统中单个记忆的完整信息
    
    主要功能:
    - 记忆内容和元数据存储
    - 记忆类型和优先级管理
    - 记忆访问统计
    - 记忆衰减机制
    
    继承关系: 无直接继承，作为数据容器使用
    """
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    memory_type: MemoryType = MemoryType.EPISODIC
    priority: MemoryPriority = MemoryPriority.NORMAL
    tags: List[str] = field(default_factory=list)
    source_agent: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    decay_factor: float = 1.0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式
        
        Returns:
            记忆项字典表示
        """
        return {
            'memory_id': self.memory_id,
            'content': self.content,
            'memory_type': self.memory_type.value,
            'priority': self.priority.value,
            'tags': self.tags,
            'source_agent': self.source_agent,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'decay_factor': self.decay_factor,
            'embedding': self.embedding,
            'metadata': self.metadata
        }
    
    def update_access(self) -> None:
        """更新访问信息
        
        更新最后访问时间和访问次数
        """
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class TaskDefinition:
    """任务定义数据结构
    
    定义任务的完整信息和执行参数
    
    主要功能:
    - 任务基本信息定义
    - 任务参数和约束设置
    - 任务依赖关系管理
    - 任务执行策略配置
    
    继承关系: 无直接继承，作为数据容器使用
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    task_type: str = ""
    priority: int = 1
    parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: Optional[int] = None
    max_retries: int = 3
    timeout: Optional[int] = None
    created_by: Optional[str] = None
    assigned_to: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式
        
        Returns:
            任务定义字典表示
        """
        return {
            'task_id': self.task_id,
            'name': self.name,
            'description': self.description,
            'task_type': self.task_type,
            'priority': self.priority,
            'parameters': self.parameters,
            'constraints': self.constraints,
            'dependencies': self.dependencies,
            'estimated_duration': self.estimated_duration,
            'max_retries': self.max_retries,
            'timeout': self.timeout,
            'created_by': self.created_by,
            'assigned_to': self.assigned_to,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class MessageAnalysis:
    """消息分析结果数据结构
    
    智能体对消息内容的分析结果
    
    主要功能:
    - 意图识别结果
    - 情感分析结果
    - 关键信息提取
    - 响应策略建议
    
    继承关系: 无直接继承，作为数据容器使用
    """
    intent: IntentType = IntentType.UNKNOWN
    confidence: float = 0.0
    entities: List[Dict[str, Any]] = field(default_factory=list)
    sentiment: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    context_requirements: List[str] = field(default_factory=list)
    suggested_response_type: Optional[MessageType] = None
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)


class MessageProtocol(ABC):
    """消息协议抽象基类
    
    定义消息处理的标准接口和协议
    
    主要功能:
    - 消息发送和接收接口
    - 消息序列化和反序列化
    - 消息验证和安全检查
    - 消息路由和分发
    
    继承关系: 抽象基类，由具体协议实现类继承
    """
    
    @abstractmethod
    def send_message(self, message: BaseMessage) -> bool:
        """发送消息
        
        Args:
            message: 要发送的消息
            
        Returns:
            发送是否成功
        """
        pass
    
    @abstractmethod
    def receive_message(self) -> Optional[BaseMessage]:
        """接收消息
        
        Returns:
            接收到的消息，如果没有消息则返回None
        """
        pass
    
    @abstractmethod
    def serialize_message(self, message: BaseMessage) -> str:
        """序列化消息
        
        Args:
            message: 要序列化的消息
            
        Returns:
            序列化后的消息字符串
        """
        pass
    
    @abstractmethod
    def deserialize_message(self, data: str) -> BaseMessage:
        """反序列化消息
        
        Args:
            data: 序列化的消息数据
            
        Returns:
            反序列化后的消息对象
        """
        pass


# 消息创建工厂函数
def create_agent_message(sender_id: str, receiver_id: str, content: Any,
                        message_type: MessageType = MessageType.TEXT,
                        priority: MessagePriority = MessagePriority.NORMAL,
                        **kwargs) -> AgentMessage:
    """创建智能体消息
    
    Args:
        sender_id: 发送者ID
        receiver_id: 接收者ID
        content: 消息内容
        message_type: 消息类型
        priority: 消息优先级
        **kwargs: 其他参数
        
    Returns:
        智能体消息实例
    """
    return AgentMessage(
        message_type=message_type,
        priority=priority,
        sender_id=sender_id,
        receiver_id=receiver_id,
        content=content,
        **kwargs
    )


def create_task_message(task_type: str, task_params: Dict[str, Any] = None,
                       assigned_agent: str = None,
                       priority: MessagePriority = MessagePriority.NORMAL,
                       **kwargs) -> TaskMessage:
    """创建任务消息
    
    Args:
        task_type: 任务类型
        task_params: 任务参数
        assigned_agent: 分配的智能体
        priority: 消息优先级
        **kwargs: 其他参数
        
    Returns:
        任务消息实例
    """
    return TaskMessage(
        message_type=MessageType.TASK_REQUEST,
        priority=priority,
        task_type=task_type,
        task_params=task_params or {},
        assigned_agent=assigned_agent,
        **kwargs
    )


def create_tool_message(tool_name: str, action: str, 
                       params: Dict[str, Any] = None,
                       **kwargs) -> ToolMessage:
    """创建工具消息
    
    Args:
        tool_name: 工具名称
        action: 工具动作
        params: 调用参数
        **kwargs: 其他参数
        
    Returns:
        工具消息实例
    """
    return ToolMessage(
        message_type=MessageType.TOOL_CALL,
        tool_name=tool_name,
        tool_action=action,
        tool_params=params or {},
        **kwargs
    )


def create_response_message(request_id: str, success: bool = True,
                          data: Any = None, error_message: str = None,
                          status_code: int = 200,
                          **kwargs) -> ResponseMessage:
    """创建响应消息
    
    Args:
        request_id: 请求ID
        success: 是否成功
        data: 响应数据
        error_message: 错误信息
        status_code: 状态码
        **kwargs: 其他参数
        
    Returns:
        响应消息实例
    """
    message_type = MessageType.RESPONSE if success else MessageType.ERROR
    return ResponseMessage(
        message_type=message_type,
        request_id=request_id,
        success=success,
        response_data=data,
        error_message=error_message,
        status_code=status_code,
        **kwargs
    )