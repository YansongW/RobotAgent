# -*- coding: utf-8 -*-

# 配置模块初始化文件 (Configuration Module Initialization)
# 统一导出配置管理和消息类型定义
# 作者: RobotAgent开发团队
# 版本: 0.0.1 (Initial Release)
# 更新时间: 2025-08-21
# 基于框架: CAMEL框架集成的配置系统

# 导入配置管理器
from .config_manager import (
    ConfigManager,
    ConfigType,
    ConfigSource,
    ConfigItem,
    config_manager
)

# 导入消息类型定义
from .message_types import (
    # 枚举类型
    MessageType,
    MessagePriority,
    MessageStatus,
    IntentType,
    TaskStatus,
    MemoryType,
    MemoryPriority,
    
    # 消息类
    BaseMessage,
    AgentMessage,
    TaskMessage,
    ResponseMessage,
    ToolMessage,
    
    # 数据结构
    MemoryItem,
    TaskDefinition,
    MessageAnalysis,
    
    # 协议接口
    MessageProtocol,
    
    # 工厂函数
    create_agent_message,
    create_task_message,
    create_tool_message,
    create_response_message
)

# 模块版本信息
__version__ = "0.0.1"
__author__ = "RobotAgent开发团队"
__description__ = "RobotAgent配置管理和消息类型定义模块"

# 导出的公共接口
__all__ = [
    # 配置管理
    "ConfigManager",
    "ConfigType",
    "ConfigSource",
    "ConfigItem",
    "config_manager",
    
    # 消息类型枚举
    "MessageType",
    "MessagePriority",
    "MessageStatus",
    "IntentType",
    "TaskStatus",
    "MemoryType",
    "MemoryPriority",
    
    # 消息类
    "BaseMessage",
    "AgentMessage",
    "TaskMessage",
    "ResponseMessage",
    "ToolMessage",
    
    # 数据结构
    "MemoryItem",
    "TaskDefinition",
    "MessageAnalysis",
    
    # 协议接口
    "MessageProtocol",
    
    # 工厂函数
    "create_agent_message",
    "create_task_message",
    "create_tool_message",
    "create_response_message"
]