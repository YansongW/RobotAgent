from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from models.message_models import Priority

class APIResponse(BaseModel):
    """API响应基础模型"""
    success: bool = Field(..., description="请求是否成功")
    message: str = Field(..., description="响应消息")
    data: Optional[Dict[str, Any]] = Field(default=None, description="响应数据")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间戳")

class ProcessCommandRequest(BaseModel):
    """处理命令请求模型"""
    user_id: str = Field(..., description="用户ID")
    input_text: str = Field(..., description="输入文本", min_length=1)
    session_id: Optional[str] = Field(default=None, description="会话ID")
    language: str = Field(default="zh-CN", description="语言代码")

class ProcessCommandResponse(APIResponse):
    """处理命令响应模型"""
    class CommandData(BaseModel):
        message_id: str = Field(..., description="消息ID")
        parsed_intent: str = Field(..., description="解析的意图")
        parsed_action: str = Field(..., description="解析的动作")
        parameters: Dict[str, Any] = Field(..., description="动作参数")
        estimated_duration: Optional[float] = Field(default=None, description="预估执行时间")
        status: str = Field(..., description="处理状态")
    
    data: Optional[CommandData] = None

class SystemStatusResponse(APIResponse):
    """系统状态响应模型"""
    class StatusData(BaseModel):
        qwen_service: str = Field(..., description="Qwen服务状态")
        redis_connection: str = Field(..., description="Redis连接状态")
        memory_agent: str = Field(..., description="记忆Agent状态")
        ros2_agent: str = Field(..., description="ROS2 Agent状态")
        uptime: float = Field(..., description="运行时间（秒）")
        memory_usage: float = Field(..., description="内存使用率")
        cpu_usage: float = Field(..., description="CPU使用率")
    
    data: Optional[StatusData] = None

class LogsResponse(APIResponse):
    """日志响应模型"""
    class LogEntry(BaseModel):
        timestamp: str = Field(..., description="时间戳")
        level: str = Field(..., description="日志级别")
        component: str = Field(..., description="组件名称")
        message: str = Field(..., description="日志消息")
    
    class LogsData(BaseModel):
        logs: List[LogEntry] = Field(..., description="日志条目列表")
        total_count: int = Field(..., description="总日志数量")
        page: int = Field(..., description="当前页码")
        page_size: int = Field(..., description="每页大小")
    
    data: Optional[LogsData] = None

class MemoryRecordsResponse(APIResponse):
    """记忆记录响应模型"""
    class MemoryRecord(BaseModel):
        filename: str = Field(..., description="文件名")
        created_time: datetime = Field(..., description="创建时间")
        size: int = Field(..., description="文件大小（字节）")
        interaction_count: int = Field(..., description="交互次数")
    
    class MemoryRecordsData(BaseModel):
        records: List[MemoryRecord] = Field(..., description="记忆记录列表")
        total_count: int = Field(..., description="总记录数量")
        total_size: int = Field(..., description="总大小（字节）")
    
    data: Optional[MemoryRecordsData] = None

class WebSocketMessage(BaseModel):
    """WebSocket消息模型"""
    type: str = Field(..., description="消息类型")
    data: Dict[str, Any] = Field(..., description="消息数据")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")

class HealthCheckResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(..., description="健康状态")
    version: str = Field(..., description="版本号")
    uptime: float = Field(..., description="运行时间（秒）")
    timestamp: datetime = Field(default_factory=datetime.now, description="检查时间")
    
    class ComponentHealth(BaseModel):
        name: str = Field(..., description="组件名称")
        status: str = Field(..., description="组件状态")
        last_check: datetime = Field(..., description="最后检查时间")
        details: Optional[Dict[str, Any]] = Field(default=None, description="详细信息")
    
    components: List[ComponentHealth] = Field(default_factory=list, description="组件健康状态")

class PerformanceMetrics(BaseModel):
    """性能指标模型"""
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    
    # 系统指标
    cpu_usage: float = Field(..., description="CPU使用率")
    memory_usage: float = Field(..., description="内存使用率")
    disk_usage: float = Field(..., description="磁盘使用率")
    
    # 应用指标
    active_sessions: int = Field(..., description="活跃会话数")
    total_requests: int = Field(..., description="总请求数")
    error_rate: float = Field(..., description="错误率")
    average_response_time: float = Field(..., description="平均响应时间（秒）")
    
    # 组件指标
    qwen_api_calls: int = Field(default=0, description="Qwen API调用次数")
    qwen_avg_latency: float = Field(default=0.0, description="Qwen平均延迟（秒）")
    redis_connections: int = Field(default=0, description="Redis连接数")
    ros2_commands_sent: int = Field(default=0, description="ROS2命令发送数")
    memory_records_created: int = Field(default=0, description="创建的记忆记录数")