from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class MessageType(str, Enum):
    """消息类型枚举"""
    USER_INPUT = "user_input"
    NLP_PARSED = "nlp_parsed"
    MEMORY_RECORD = "memory_record"
    ROS2_COMMAND = "ros2_command"
    SYSTEM_STATUS = "system_status"
    ERROR = "error"

class Priority(str, Enum):
    """优先级枚举"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class UserInputMessage(BaseModel):
    """用户输入消息模型"""
    user_id: str = Field(..., description="用户ID")
    session_id: str = Field(..., description="会话ID")
    input_text: str = Field(..., description="用户输入的自然语言文本")
    language: str = Field(default="zh-CN", description="语言代码")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="附加元数据")

class ParsedMessage(BaseModel):
    """解析后的消息模型"""
    message_id: str = Field(..., description="消息ID")
    user_id: str = Field(..., description="用户ID")
    session_id: str = Field(..., description="会话ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    
    # 解析结果
    intent: str = Field(..., description="意图识别结果")
    action: str = Field(..., description="具体动作")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="动作参数")
    
    # 控制信息
    priority: Priority = Field(default=Priority.NORMAL, description="优先级")
    requires_confirmation: bool = Field(default=False, description="是否需要确认")
    estimated_duration: Optional[float] = Field(default=None, description="预估执行时间（秒）")
    
    # 原始信息
    original_text: str = Field(..., description="原始输入文本")
    confidence: float = Field(default=1.0, description="解析置信度")

class Position(BaseModel):
    """位置坐标模型"""
    x: float = Field(..., description="X坐标")
    y: float = Field(..., description="Y坐标") 
    z: float = Field(..., description="Z坐标")

class Orientation(BaseModel):
    """方向四元数模型"""
    x: float = Field(default=0.0, description="四元数X")
    y: float = Field(default=0.0, description="四元数Y")
    z: float = Field(default=0.0, description="四元数Z")
    w: float = Field(default=1.0, description="四元数W")

class Pose(BaseModel):
    """位姿模型"""
    position: Position = Field(..., description="位置")
    orientation: Orientation = Field(default_factory=Orientation, description="方向")

class ROS2CommandMessage(BaseModel):
    """ROS2命令消息模型"""
    message_id: str = Field(..., description="消息ID")
    command_type: str = Field(..., description="命令类型")
    command: str = Field(..., description="ROS2命令")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="命令参数")
    priority: Priority = Field(default=Priority.NORMAL, description="优先级")
    timeout: Optional[float] = Field(default=30.0, description="超时时间（秒）")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")

class MemoryRecordMessage(BaseModel):
    """记忆记录消息模型"""
    message_id: str = Field(..., description="消息ID")
    user_id: str = Field(..., description="用户ID")
    session_id: str = Field(..., description="会话ID")
    interaction_data: Dict[str, Any] = Field(..., description="交互数据")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    
class SystemStatusMessage(BaseModel):
    """系统状态消息模型"""
    component: str = Field(..., description="组件名称")
    status: str = Field(..., description="状态")
    message: str = Field(..., description="状态消息")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    details: Optional[Dict[str, Any]] = Field(default=None, description="详细信息")

class ErrorMessage(BaseModel):
    """错误消息模型"""
    error_id: str = Field(..., description="错误ID")
    error_type: str = Field(..., description="错误类型")
    error_message: str = Field(..., description="错误消息")
    component: str = Field(..., description="出错组件")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    context: Optional[Dict[str, Any]] = Field(default=None, description="错误上下文")
    stack_trace: Optional[str] = Field(default=None, description="堆栈跟踪")

class QueueMessage(BaseModel):
    """队列消息基础模型"""
    message_type: MessageType = Field(..., description="消息类型")
    message_id: str = Field(..., description="消息ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    data: Dict[str, Any] = Field(..., description="消息数据")
    priority: Priority = Field(default=Priority.NORMAL, description="优先级")
    ttl: Optional[int] = Field(default=3600, description="消息生存时间（秒）")