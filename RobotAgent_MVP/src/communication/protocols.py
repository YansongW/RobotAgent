# -*- coding: utf-8 -*-

# 通信协议定义 (Communication Protocols)
# 定义智能体间的消息格式、协议和通信规范
# 作者: RobotAgent开发团队
# 版本: 0.0.1 (Initial Release)
# 更新时间: 2025-08-21

# 导入标准库
import uuid
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod


class MessageType(Enum):
    """消息类型枚举"""
    # 基础消息类型
    TASK = "task"                           # 任务消息
    INSTRUCTION = "instruction"             # 指令消息
    RESPONSE = "response"                   # 响应消息
    STATUS = "status"                       # 状态消息
    ERROR = "error"                         # 错误消息
    
    # 协作消息类型
    COLLABORATION_REQUEST = "collaboration_request"   # 协作请求
    COLLABORATION_RESPONSE = "collaboration_response" # 协作响应
    DELEGATION = "delegation"               # 任务委托
    FEEDBACK = "feedback"                   # 反馈消息
    
    # 记忆相关消息
    MEMORY_STORE = "memory_store"           # 记忆存储
    MEMORY_RETRIEVE = "memory_retrieve"     # 记忆检索
    MEMORY_UPDATE = "memory_update"         # 记忆更新
    MEMORY_DELETE = "memory_delete"         # 记忆删除
    
    # 系统消息类型
    HEARTBEAT = "heartbeat"                 # 心跳消息
    SHUTDOWN = "shutdown"                   # 关闭消息
    RESTART = "restart"                     # 重启消息
    CONFIG_UPDATE = "config_update"         # 配置更新
    
    # 工具相关消息
    TOOL_CALL = "tool_call"                 # 工具调用
    TOOL_RESULT = "tool_result"             # 工具结果
    TOOL_ERROR = "tool_error"               # 工具错误
    
    # 学习相关消息
    LEARNING_DATA = "learning_data"         # 学习数据
    MODEL_UPDATE = "model_update"           # 模型更新
    PATTERN_DISCOVERY = "pattern_discovery" # 模式发现


class MessagePriority(Enum):
    """消息优先级枚举"""
    CRITICAL = "critical"   # 关键消息（系统错误、紧急停止等）
    HIGH = "high"           # 高优先级（重要任务、错误处理等）
    MEDIUM = "medium"       # 中等优先级（普通任务、状态更新等）
    LOW = "low"             # 低优先级（日志、统计信息等）


class CollaborationMode(Enum):
    """协作模式枚举"""
    DIRECT = "direct"       # 直接协作
    CHAIN = "chain"         # 链式协作
    PARALLEL = "parallel"   # 并行协作
    FEEDBACK = "feedback"   # 反馈协作


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"     # 待处理
    RUNNING = "running"     # 执行中
    COMPLETED = "completed" # 已完成
    FAILED = "failed"       # 失败
    CANCELLED = "cancelled" # 已取消
    PAUSED = "paused"       # 暂停


class AgentRole(Enum):
    """智能体角色枚举"""
    CHAT_AGENT = "chat_agent"       # 对话协调智能体
    ACTION_AGENT = "action_agent"   # 任务执行智能体
    MEMORY_AGENT = "memory_agent"   # 记忆管理智能体


@dataclass
class AgentMessage:
    """智能体消息基础类
    
    定义智能体间通信的标准消息格式
    """
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: Optional[str] = None  # None表示广播消息
    message_type: MessageType = MessageType.RESPONSE
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: MessagePriority = MessagePriority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None  # 用于关联请求和响应
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['message_type'] = self.message_type.value
        data['priority'] = self.priority.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """从字典创建消息对象"""
        # 处理枚举类型
        if 'message_type' in data:
            data['message_type'] = MessageType(data['message_type'])
        if 'priority' in data:
            data['priority'] = MessagePriority(data['priority'])
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        return cls(**data)
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AgentMessage':
        """从JSON字符串创建消息对象"""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class TaskMessage(AgentMessage):
    """任务消息
    
    用于传递任务相关信息
    """
    message_type: MessageType = MessageType.TASK
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_description: str = ""
    task_parameters: Dict[str, Any] = field(default_factory=dict)
    expected_output: Optional[str] = None
    deadline: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # 将任务相关信息添加到content中
        self.content.update({
            "task_id": self.task_id,
            "task_description": self.task_description,
            "task_parameters": self.task_parameters,
            "expected_output": self.expected_output,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "dependencies": self.dependencies
        })


@dataclass
class ResponseMessage(AgentMessage):
    """响应消息
    
    用于返回处理结果
    """
    message_type: MessageType = MessageType.RESPONSE
    success: bool = True
    result: Any = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    
    def __post_init__(self):
        # 将响应相关信息添加到content中
        self.content.update({
            "success": self.success,
            "result": self.result,
            "error_message": self.error_message,
            "execution_time": self.execution_time
        })


@dataclass
class StatusMessage(AgentMessage):
    """状态消息
    
    用于报告智能体状态
    """
    message_type: MessageType = MessageType.STATUS
    agent_status: str = "idle"
    current_task: Optional[str] = None
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # 将状态相关信息添加到content中
        self.content.update({
            "agent_status": self.agent_status,
            "current_task": self.current_task,
            "resource_usage": self.resource_usage,
            "performance_metrics": self.performance_metrics
        })


@dataclass
class CollaborationRequest(AgentMessage):
    """协作请求消息
    
    用于请求其他智能体协作
    """
    message_type: MessageType = MessageType.COLLABORATION_REQUEST
    collaboration_mode: CollaborationMode = CollaborationMode.DIRECT
    requested_capability: str = ""
    context_data: Dict[str, Any] = field(default_factory=dict)
    expected_response_time: Optional[float] = None
    
    def __post_init__(self):
        # 将协作请求相关信息添加到content中
        self.content.update({
            "collaboration_mode": self.collaboration_mode.value,
            "requested_capability": self.requested_capability,
            "context_data": self.context_data,
            "expected_response_time": self.expected_response_time
        })


@dataclass
class CollaborationResponse(AgentMessage):
    """协作响应消息
    
    用于响应协作请求
    """
    message_type: MessageType = MessageType.COLLABORATION_RESPONSE
    accepted: bool = True
    capability_match: float = 1.0  # 能力匹配度 0-1
    estimated_completion_time: Optional[float] = None
    alternative_suggestions: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # 将协作响应相关信息添加到content中
        self.content.update({
            "accepted": self.accepted,
            "capability_match": self.capability_match,
            "estimated_completion_time": self.estimated_completion_time,
            "alternative_suggestions": self.alternative_suggestions
        })


@dataclass
class MemoryMessage(AgentMessage):
    """记忆消息
    
    用于记忆相关操作
    """
    operation: str = "store"  # store, retrieve, update, delete
    memory_type: str = "working"  # working, short_term, long_term
    memory_key: Optional[str] = None
    memory_data: Any = None
    search_query: Optional[str] = None
    search_filters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # 将记忆相关信息添加到content中
        self.content.update({
            "operation": self.operation,
            "memory_type": self.memory_type,
            "memory_key": self.memory_key,
            "memory_data": self.memory_data,
            "search_query": self.search_query,
            "search_filters": self.search_filters
        })


@dataclass
class ToolMessage(AgentMessage):
    """工具消息
    
    用于工具调用相关操作
    """
    tool_name: str = ""
    tool_parameters: Dict[str, Any] = field(default_factory=dict)
    tool_result: Any = None
    execution_status: str = "pending"  # pending, running, completed, failed
    
    def __post_init__(self):
        # 将工具相关信息添加到content中
        self.content.update({
            "tool_name": self.tool_name,
            "tool_parameters": self.tool_parameters,
            "tool_result": self.tool_result,
            "execution_status": self.execution_status
        })


class MessageProtocol(ABC):
    """消息协议抽象基类
    
    定义消息处理的标准接口
    """
    
    @abstractmethod
    async def send_message(self, message: AgentMessage) -> bool:
        """发送消息
        
        Args:
            message: 要发送的消息
            
        Returns:
            是否发送成功
        """
        pass
    
    @abstractmethod
    async def receive_message(self, timeout: Optional[float] = None) -> Optional[AgentMessage]:
        """接收消息
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            接收到的消息，如果超时则返回None
        """
        pass
    
    @abstractmethod
    async def subscribe(self, message_types: List[MessageType]) -> bool:
        """订阅消息类型
        
        Args:
            message_types: 要订阅的消息类型列表
            
        Returns:
            是否订阅成功
        """
        pass
    
    @abstractmethod
    async def unsubscribe(self, message_types: List[MessageType]) -> bool:
        """取消订阅消息类型
        
        Args:
            message_types: 要取消订阅的消息类型列表
            
        Returns:
            是否取消订阅成功
        """
        pass


class MessageValidator:
    """消息验证器
    
    用于验证消息格式和内容的正确性
    """
    
    @staticmethod
    def validate_message(message: AgentMessage) -> tuple[bool, Optional[str]]:
        """验证消息
        
        Args:
            message: 要验证的消息
            
        Returns:
            (是否有效, 错误信息)
        """
        try:
            # 检查必要字段
            if not message.message_id:
                return False, "消息ID不能为空"
            
            if not message.sender_id:
                return False, "发送者ID不能为空"
            
            if not isinstance(message.message_type, MessageType):
                return False, "消息类型无效"
            
            if not isinstance(message.priority, MessagePriority):
                return False, "消息优先级无效"
            
            if not isinstance(message.timestamp, datetime):
                return False, "时间戳格式无效"
            
            # 检查内容格式
            if not isinstance(message.content, dict):
                return False, "消息内容必须是字典格式"
            
            if not isinstance(message.metadata, dict):
                return False, "消息元数据必须是字典格式"
            
            return True, None
            
        except Exception as e:
            return False, f"消息验证异常: {str(e)}"
    
    @staticmethod
    def validate_task_message(message: TaskMessage) -> tuple[bool, Optional[str]]:
        """验证任务消息
        
        Args:
            message: 要验证的任务消息
            
        Returns:
            (是否有效, 错误信息)
        """
        # 先进行基础验证
        is_valid, error = MessageValidator.validate_message(message)
        if not is_valid:
            return is_valid, error
        
        try:
            # 检查任务特定字段
            if not message.task_id:
                return False, "任务ID不能为空"
            
            if not message.task_description:
                return False, "任务描述不能为空"
            
            if message.deadline and message.deadline <= datetime.now():
                return False, "任务截止时间不能是过去时间"
            
            return True, None
            
        except Exception as e:
            return False, f"任务消息验证异常: {str(e)}"


class MessageSerializer:
    """消息序列化器
    
    用于消息的序列化和反序列化
    """
    
    @staticmethod
    def serialize(message: AgentMessage) -> bytes:
        """序列化消息
        
        Args:
            message: 要序列化的消息
            
        Returns:
            序列化后的字节数据
        """
        try:
            json_str = message.to_json()
            return json_str.encode('utf-8')
        except Exception as e:
            raise ValueError(f"消息序列化失败: {str(e)}")
    
    @staticmethod
    def deserialize(data: bytes) -> AgentMessage:
        """反序列化消息
        
        Args:
            data: 要反序列化的字节数据
            
        Returns:
            反序列化后的消息对象
        """
        try:
            json_str = data.decode('utf-8')
            return AgentMessage.from_json(json_str)
        except Exception as e:
            raise ValueError(f"消息反序列化失败: {str(e)}")


class MessageRouter:
    """消息路由器
    
    用于消息路由和分发逻辑
    """
    
    def __init__(self):
        self.routing_rules: List[Dict[str, Any]] = []
    
    def add_routing_rule(self, rule: Dict[str, Any]):
        """添加路由规则
        
        Args:
            rule: 路由规则字典
                - condition: 路由条件函数
                - target: 目标智能体ID或ID列表
                - priority: 规则优先级
        """
        self.routing_rules.append(rule)
        # 按优先级排序
        self.routing_rules.sort(key=lambda x: x.get('priority', 0), reverse=True)
    
    def route_message(self, message: AgentMessage) -> List[str]:
        """路由消息
        
        Args:
            message: 要路由的消息
            
        Returns:
            目标智能体ID列表
        """
        targets = []
        
        # 如果消息指定了接收者，直接返回
        if message.receiver_id:
            return [message.receiver_id]
        
        # 应用路由规则
        for rule in self.routing_rules:
            try:
                condition = rule.get('condition')
                if condition and condition(message):
                    target = rule.get('target')
                    if isinstance(target, str):
                        targets.append(target)
                    elif isinstance(target, list):
                        targets.extend(target)
                    break  # 使用第一个匹配的规则
            except Exception:
                continue  # 忽略规则执行错误
        
        return targets


# 预定义的消息创建函数

def create_task_message(sender_id: str, receiver_id: str, task_description: str, 
                       task_parameters: Optional[Dict[str, Any]] = None,
                       priority: MessagePriority = MessagePriority.MEDIUM) -> TaskMessage:
    """创建任务消息"""
    return TaskMessage(
        sender_id=sender_id,
        receiver_id=receiver_id,
        task_description=task_description,
        task_parameters=task_parameters or {},
        priority=priority
    )


def create_response_message(sender_id: str, receiver_id: str, success: bool,
                          result: Any = None, error_message: Optional[str] = None,
                          correlation_id: Optional[str] = None) -> ResponseMessage:
    """创建响应消息"""
    return ResponseMessage(
        sender_id=sender_id,
        receiver_id=receiver_id,
        success=success,
        result=result,
        error_message=error_message,
        correlation_id=correlation_id
    )


def create_collaboration_request(sender_id: str, receiver_id: str, 
                               requested_capability: str,
                               collaboration_mode: CollaborationMode = CollaborationMode.DIRECT,
                               context_data: Optional[Dict[str, Any]] = None) -> CollaborationRequest:
    """创建协作请求消息"""
    return CollaborationRequest(
        sender_id=sender_id,
        receiver_id=receiver_id,
        requested_capability=requested_capability,
        collaboration_mode=collaboration_mode,
        context_data=context_data or {}
    )


def create_memory_message(sender_id: str, receiver_id: str, operation: str,
                         memory_type: str = "working", memory_key: Optional[str] = None,
                         memory_data: Any = None) -> MemoryMessage:
    """创建记忆消息"""
    message = MemoryMessage(
        sender_id=sender_id,
        receiver_id=receiver_id,
        operation=operation,
        memory_type=memory_type,
        memory_key=memory_key,
        memory_data=memory_data
    )
    
    # 根据操作类型设置消息类型
    if operation == "store":
        message.message_type = MessageType.MEMORY_STORE
    elif operation == "retrieve":
        message.message_type = MessageType.MEMORY_RETRIEVE
    elif operation == "update":
        message.message_type = MessageType.MEMORY_UPDATE
    elif operation == "delete":
        message.message_type = MessageType.MEMORY_DELETE
    
    return message


def create_status_message(sender_id: str, agent_status: str = "idle",
                         current_task: Optional[str] = None,
                         resource_usage: Optional[Dict[str, Any]] = None,
                         performance_metrics: Optional[Dict[str, Any]] = None) -> StatusMessage:
    """创建状态消息"""
    return StatusMessage(
        sender_id=sender_id,
        agent_status=agent_status,
        current_task=current_task,
        resource_usage=resource_usage or {},
        performance_metrics=performance_metrics or {}
    )