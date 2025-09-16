# -*- coding: utf-8 -*-

# 工具和插件系统设计 (Tools and Plugins System Design)
# 基于AgentScope框架重新设计的完整工具管理系统，包含工具生命周期管理、安全执行环境、插件架构、内置工具实现和配置部署方案
# 版本: 0.2.2
# 更新时间: 2025-01-08

# 工具和插件系统设计

## 1. 基于AgentScope的工具系统概述

### 1.1 设计理念

本工具系统基于AgentScope框架的ToolBase类进行设计，充分利用AgentScope的工具管理机制，实现智能体与外部功能的无缝集成。系统遵循AgentScope的设计哲学，提供统一、安全、可扩展的工具调用接口。

```python
# 基于AgentScope的工具系统核心设计
from agentscope.tool import ToolBase
from agentscope.message import Msg
from agentscope.agents import AgentBase
from typing import Any, Dict, List, Optional, Union
import asyncio
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# AgentScope工具系统的核心原则
class AgentScopeToolPrinciples:
    """
    AgentScope工具系统设计原则
    
    1. 统一接口原则: 所有工具继承自agentscope.tool.ToolBase
    2. 消息驱动原则: 工具调用通过Msg消息进行传递
    3. 类型安全原则: 严格的参数验证和类型检查
    4. 异步优先原则: 支持异步执行提升性能
    5. 可观测原则: 完整的执行日志和监控
    """
    pass
```

### 1.2 核心特性

基于AgentScope框架，工具系统具备以下核心特性：

- **AgentScope原生集成**: 完全基于agentscope.tool.ToolBase实现
- **消息驱动架构**: 通过agentscope.message.Msg进行工具调用
- **类型安全保障**: 严格的JSON Schema参数验证
- **异步执行支持**: 原生支持async/await模式
- **智能体感知**: 工具可感知调用的智能体上下文
- **会话状态集成**: 与AgentScope的会话管理无缝集成
- **插件化扩展**: 支持动态加载和热插拔

### 1.3 基于AgentScope的工具分类体系

```python
# AgentScope工具分类体系
from enum import Enum
from agentscope.tool import ToolBase

class AgentScopeToolCategory(Enum):
    """基于AgentScope的工具分类"""
    
    # 核心工具类别 - 基于AgentScope内置功能
    MESSAGE_PROCESSING = "message_processing"    # 消息处理工具
    AGENT_INTERACTION = "agent_interaction"      # 智能体交互工具
    MODEL_INTERFACE = "model_interface"          # 模型接口工具
    MEMORY_MANAGEMENT = "memory_management"      # 记忆管理工具
    
    # 扩展工具类别 - 基于业务需求
    FILE_OPERATION = "file_operation"            # 文件操作工具
    NETWORK_REQUEST = "network_request"          # 网络请求工具
    DATA_PROCESSING = "data_processing"          # 数据处理工具
    SYSTEM_INTEGRATION = "system_integration"    # 系统集成工具
    
    # 高级工具类别 - 基于复杂场景
    WORKFLOW_CONTROL = "workflow_control"        # 工作流控制工具
    PLUGIN_EXTENSION = "plugin_extension"        # 插件扩展工具
    CUSTOM_BUSINESS = "custom_business"          # 自定义业务工具

# AgentScope工具系统层次结构
class AgentScopeToolHierarchy:
    """
    AgentScope工具系统层次结构
    
    基于AgentScope框架的工具系统分层设计：
    
    Level 1: AgentScope核心工具层
    ├── MessageTool: 消息处理工具
    ├── AgentTool: 智能体管理工具
    ├── ModelTool: 模型调用工具
    └── MemoryTool: 记忆操作工具
    
    Level 2: 业务扩展工具层
    ├── FileTool: 文件操作工具
    ├── NetworkTool: 网络请求工具
    ├── DataTool: 数据处理工具
    └── SystemTool: 系统集成工具
    
    Level 3: 高级应用工具层
    ├── WorkflowTool: 工作流控制工具
    ├── PluginTool: 插件扩展工具
    └── CustomTool: 自定义业务工具
    """
    pass
```

## 2. 基于AgentScope的工具架构设计

### 2.1 AgentScope工具系统架构

```python
# AgentScope工具系统架构图
"""
┌─────────────────────────────────────────────────────────────┐
│                AgentScope智能体层 (Agent Layer)             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │AgentBase    │  │UserAgent    │  │AssistantAgent│       │
│  │智能体基类    │  │用户智能体    │  │助手智能体     │       │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│              AgentScope工具管理层 (Tool Management)         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ToolManager  │  │ToolRegistry │  │ToolExecutor │        │
│  │工具管理器    │  │工具注册表    │  │工具执行器     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│              AgentScope工具抽象层 (Tool Abstraction)        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ToolBase     │  │ToolSchema   │  │ToolResult   │        │
│  │工具基类      │  │工具模式      │  │工具结果      │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│              AgentScope消息层 (Message Layer)               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │Msg          │  │MessageHub   │  │Placeholder  │        │
│  │消息对象      │  │消息中心      │  │占位符        │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│              AgentScope会话层 (Session Layer)               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │Session      │  │SessionManager│ │StateManager │        │
│  │会话对象      │  │会话管理器     │  │状态管理器    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
"""

class AgentScopeToolArchitecture:
    """AgentScope工具系统架构实现"""
    
    def __init__(self):
        # 基于AgentScope的核心组件
        self.tool_manager = AgentScopeToolManager()
        self.tool_registry = AgentScopeToolRegistry()
        self.tool_executor = AgentScopeToolExecutor()
        
        # 集成AgentScope的消息系统
        self.message_hub = None  # 将在初始化时设置
        
        # 集成AgentScope的会话管理
        self.session_manager = None  # 将在初始化时设置
    
    def initialize_with_agentscope(self, session_manager, message_hub):
        """与AgentScope框架集成初始化"""
        self.session_manager = session_manager
        self.message_hub = message_hub
        
        # 配置工具管理器与AgentScope的集成
        self.tool_manager.set_session_manager(session_manager)
        self.tool_executor.set_message_hub(message_hub)
```

### 2.2 AgentScope工具生命周期管理

```python
# AgentScope工具生命周期
class AgentScopeToolLifecycle:
    """
    AgentScope工具生命周期管理
    
    生命周期阶段：
    1. 注册阶段 (Registration): 工具注册到AgentScope系统
    2. 验证阶段 (Validation): 验证工具的JSON Schema和接口
    3. 加载阶段 (Loading): 加载工具到智能体的工具集
    4. 绑定阶段 (Binding): 绑定工具到特定智能体实例
    5. 执行阶段 (Execution): 通过Msg消息调用工具
    6. 监控阶段 (Monitoring): 监控工具执行状态和性能
    7. 更新阶段 (Update): 动态更新工具配置和实现
    8. 卸载阶段 (Unloading): 从智能体中卸载工具
    """
    
    def __init__(self):
        self.lifecycle_stages = [
            "registration",
            "validation", 
            "loading",
            "binding",
            "execution",
            "monitoring",
            "update",
            "unloading"
        ]
        
        self.stage_handlers = {
            "registration": self._handle_registration,
            "validation": self._handle_validation,
            "loading": self._handle_loading,
            "binding": self._handle_binding,
            "execution": self._handle_execution,
            "monitoring": self._handle_monitoring,
            "update": self._handle_update,
            "unloading": self._handle_unloading
        }
    
    async def manage_tool_lifecycle(
        self, 
        tool: ToolBase, 
        stage: str, 
        context: dict = None
    ) -> dict:
        """管理工具生命周期
        
        Args:
            tool: AgentScope工具实例
            stage: 生命周期阶段
            context: 上下文信息
            
        Returns:
            生命周期管理结果
        """
        if stage not in self.stage_handlers:
            raise ValueError(f"Unknown lifecycle stage: {stage}")
        
        handler = self.stage_handlers[stage]
        result = await handler(tool, context or {})
        
        logger.info(f"Tool {tool.name} lifecycle stage {stage} completed")
        return result
    
    async def _handle_registration(self, tool: ToolBase, context: dict) -> dict:
        """处理工具注册阶段"""
        # 验证工具是否继承自AgentScope ToolBase
        if not isinstance(tool, ToolBase):
            raise TypeError("Tool must inherit from agentscope.tool.ToolBase")
        
        # 验证工具的必需方法
        required_methods = ['__call__', 'get_schema']
        for method in required_methods:
            if not hasattr(tool, method):
                raise AttributeError(f"Tool missing required method: {method}")
        
        return {
            "stage": "registration",
            "status": "success",
            "tool_name": tool.name,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_validation(self, tool: ToolBase, context: dict) -> dict:
        """处理工具验证阶段"""
        # 验证工具的JSON Schema
        try:
            schema = tool.get_schema()
            if not schema or 'name' not in schema:
                raise ValueError("Invalid tool schema")
        except Exception as e:
            raise ValueError(f"Tool schema validation failed: {e}")
        
        return {
            "stage": "validation",
            "status": "success",
            "schema_valid": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_loading(self, tool: ToolBase, context: dict) -> dict:
        """处理工具加载阶段"""
        # 加载工具到系统中
        # 这里可以包括依赖检查、资源分配等
        return {
            "stage": "loading",
            "status": "success",
            "loaded": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_binding(self, tool: ToolBase, context: dict) -> dict:
        """处理工具绑定阶段"""
        agent_id = context.get("agent_id")
        if not agent_id:
            raise ValueError("Agent ID required for tool binding")
        
        return {
            "stage": "binding",
            "status": "success",
            "agent_id": agent_id,
            "bound": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_execution(self, tool: ToolBase, context: dict) -> dict:
        """处理工具执行阶段"""
        # 工具执行的监控和日志记录
        return {
            "stage": "execution",
            "status": "success",
            "executed": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_monitoring(self, tool: ToolBase, context: dict) -> dict:
        """处理工具监控阶段"""
        # 监控工具的性能和状态
        return {
            "stage": "monitoring",
            "status": "success",
            "monitoring_active": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_update(self, tool: ToolBase, context: dict) -> dict:
        """处理工具更新阶段"""
        # 动态更新工具配置或实现
        return {
            "stage": "update",
            "status": "success",
            "updated": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_unloading(self, tool: ToolBase, context: dict) -> dict:
        """处理工具卸载阶段"""
        # 从系统中卸载工具，清理资源
        return {
            "stage": "unloading",
            "status": "success",
            "unloaded": True,
            "timestamp": datetime.utcnow().isoformat()
        }
```

## 3. 基于AgentScope的工具基础接口

### 3.1 AgentScope工具基类实现

```python
# 基于AgentScope的工具基础接口
from agentscope.tool import ToolBase
from agentscope.message import Msg
from agentscope.session import SessionManager
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
import asyncio
import logging
from datetime import datetime
import json

class AgentScopeToolMode(Enum):
    """基于AgentScope的工具执行模式"""
    SYNC = "sync"              # 同步执行
    ASYNC = "async"            # 异步执行
    STREAMING = "streaming"     # 流式执行
    MESSAGE_DRIVEN = "message_driven"  # 消息驱动执行

class AgentScopeToolBase(ToolBase):
    """基于AgentScope的工具基础抽象类
    
    继承自AgentScope的ToolBase，提供与AgentScope框架的原生集成。
    所有自定义工具都应该继承此类以确保与AgentScope的兼容性。
    """
    
    def __init__(
        self, 
        name: str, 
        description: str,
        mode: AgentScopeToolMode = AgentScopeToolMode.SYNC,
        timeout: Optional[int] = None,
        retry_count: int = 0,
        session_manager: Optional[SessionManager] = None
    ):
        # 调用AgentScope ToolBase的初始化
        super().__init__(name=name, description=description)
        
        # 扩展属性
        self.mode = mode
        self.timeout = timeout
        self.retry_count = retry_count
        self.session_manager = session_manager
        self.logger = logging.getLogger(f"agentscope.tool.{name}")
        self.created_at = datetime.utcnow()
        self.last_executed = None
        self.execution_count = 0
        self.error_count = 0
        
        # AgentScope集成属性
        self.message_history: List[Msg] = []
        self.agent_context: Dict[str, Any] = {}
    
    def __call__(self, **kwargs) -> Any:
        """AgentScope工具调用接口
        
        这是AgentScope框架调用工具的标准接口。
        根据工具模式选择合适的执行方式。
        
        Args:
            **kwargs: 工具执行参数
            
        Returns:
            工具执行结果
        """
        try:
            # 创建执行消息
            execution_msg = self._create_execution_message(**kwargs)
            
            # 根据模式执行
            if self.mode == AgentScopeToolMode.SYNC:
                return self._execute_sync(**kwargs)
            elif self.mode == AgentScopeToolMode.ASYNC:
                return asyncio.run(self._execute_async(**kwargs))
            elif self.mode == AgentScopeToolMode.STREAMING:
                return self._execute_streaming(**kwargs)
            elif self.mode == AgentScopeToolMode.MESSAGE_DRIVEN:
                return self._execute_message_driven(execution_msg, **kwargs)
            else:
                raise ValueError(f"Unsupported tool mode: {self.mode}")
                
        except Exception as e:
            self._handle_execution_error(e, **kwargs)
            raise
    
    def get_schema(self) -> Dict[str, Any]:
        """获取AgentScope兼容的工具Schema
        
        Returns:
            符合AgentScope标准的工具Schema定义
        """
        base_schema = {
            "name": self.name,
            "description": self.description,
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._get_parameters_schema()
            },
            "agentscope_metadata": {
                "mode": self.mode.value,
                "timeout": self.timeout,
                "retry_count": self.retry_count,
                "created_at": self.created_at.isoformat(),
                "version": "0.2.0"
            }
        }
        
        # 子类可以重写此方法来提供具体的参数Schema
        return base_schema
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """获取参数Schema定义
        
        子类应该重写此方法来定义具体的参数Schema。
        
        Returns:
            参数的JSON Schema定义
        """
        return {
            "type": "object",
            "properties": {},
            "required": []
        }
    
    def _create_execution_message(self, **kwargs) -> Msg:
        """创建工具执行消息
        
        Args:
            **kwargs: 执行参数
            
        Returns:
            AgentScope消息对象
        """
        return Msg(
            name=f"tool_{self.name}",
            content={
                "tool_name": self.name,
                "action": "execute",
                "parameters": kwargs,
                "timestamp": datetime.utcnow().isoformat()
            },
            role="tool",
            metadata={
                "tool_mode": self.mode.value,
                "execution_id": f"{self.name}_{self.execution_count + 1}"
            }
        )
    
    def _execute_sync(self, **kwargs) -> Any:
        """同步执行工具"""
        self.logger.info(f"Executing tool {self.name} in sync mode")
        
        # 验证输入
        self._validate_input(**kwargs)
        
        # 预处理
        processed_kwargs = self._pre_execute(**kwargs)
        
        # 执行核心逻辑
        result = self._execute_core(**processed_kwargs)
        
        # 后处理
        return self._post_execute(result, **kwargs)
    
    async def _execute_async(self, **kwargs) -> Any:
        """异步执行工具"""
        self.logger.info(f"Executing tool {self.name} in async mode")
        
        # 验证输入
        self._validate_input(**kwargs)
        
        # 预处理
        processed_kwargs = self._pre_execute(**kwargs)
        
        # 异步执行核心逻辑
        if asyncio.iscoroutinefunction(self._execute_core):
            result = await self._execute_core(**processed_kwargs)
        else:
            result = self._execute_core(**processed_kwargs)
        
        # 后处理
        return self._post_execute(result, **kwargs)
    
    def _execute_streaming(self, **kwargs):
        """流式执行工具"""
        self.logger.info(f"Executing tool {self.name} in streaming mode")
        
        # 验证输入
        self._validate_input(**kwargs)
        
        # 预处理
        processed_kwargs = self._pre_execute(**kwargs)
        
        # 流式执行
        for chunk in self._execute_core_streaming(**processed_kwargs):
            yield chunk
        
        # 后处理
        self._post_execute(None, **kwargs)
    
    def _execute_message_driven(self, msg: Msg, **kwargs) -> Any:
        """消息驱动执行工具
        
        这是AgentScope特有的执行模式，通过消息驱动工具执行。
        
        Args:
            msg: AgentScope消息对象
            **kwargs: 执行参数
            
        Returns:
            执行结果
        """
        self.logger.info(f"Executing tool {self.name} in message-driven mode")
        
        # 将消息添加到历史记录
        self.message_history.append(msg)
        
        # 从消息中提取参数
        if isinstance(msg.content, dict) and "parameters" in msg.content:
            msg_kwargs = msg.content["parameters"]
            kwargs.update(msg_kwargs)
        
        # 执行工具
        result = self._execute_sync(**kwargs)
        
        # 创建结果消息
        result_msg = Msg(
            name=f"tool_{self.name}_result",
            content={
                "tool_name": self.name,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            },
            role="tool",
            metadata={
                "execution_id": msg.metadata.get("execution_id"),
                "parent_message_id": getattr(msg, 'id', None)
            }
        )
        
        self.message_history.append(result_msg)
        return result
    
    def _execute_core(self, **kwargs) -> Any:
        """工具核心执行逻辑
        
        子类必须重写此方法来实现具体的工具功能。
        
        Args:
            **kwargs: 执行参数
            
        Returns:
            执行结果
        """
        raise NotImplementedError("Subclasses must implement _execute_core method")
    
    def _execute_core_streaming(self, **kwargs):
        """工具流式执行核心逻辑
        
        子类可以重写此方法来实现流式执行功能。
        
        Args:
            **kwargs: 执行参数
            
        Yields:
            执行过程中的数据块
        """
        # 默认实现：将同步执行结果作为单个块返回
        result = self._execute_core(**kwargs)
        yield result
    
    def _validate_input(self, **kwargs) -> bool:
        """验证输入参数
        
        Args:
            **kwargs: 待验证的参数
            
        Returns:
            验证是否通过
            
        Raises:
            ValueError: 参数验证失败时抛出
        """
        # 默认实现，子类可以重写
        return True
    
    def _pre_execute(self, **kwargs) -> Dict[str, Any]:
        """执行前的预处理
        
        Args:
            **kwargs: 执行参数
            
        Returns:
            预处理后的参数
        """
        self.logger.debug(f"Pre-executing tool {self.name} with args: {kwargs}")
        return kwargs
    
    def _post_execute(self, result: Any, **kwargs) -> Any:
        """执行后的后处理
        
        Args:
            result: 执行结果
            **kwargs: 执行参数
            
        Returns:
            后处理后的结果
        """
        self.last_executed = datetime.utcnow()
        self.execution_count += 1
        self.logger.debug(f"Tool {self.name} executed successfully")
        
        # 如果有会话管理器，记录执行信息
        if self.session_manager:
            self._record_execution_to_session(result, **kwargs)
        
        return result
    
    def _handle_execution_error(self, error: Exception, **kwargs) -> None:
        """处理执行错误
        
        Args:
            error: 发生的异常
            **kwargs: 执行参数
        """
        self.error_count += 1
        self.logger.error(f"Tool {self.name} execution failed: {error}")
        
        # 如果有会话管理器，记录错误信息
        if self.session_manager:
            self._record_error_to_session(error, **kwargs)
    
    def _record_execution_to_session(self, result: Any, **kwargs) -> None:
        """将执行信息记录到AgentScope会话中"""
        if self.session_manager:
            execution_record = {
                "tool_name": self.name,
                "execution_time": self.last_executed.isoformat(),
                "parameters": kwargs,
                "result": result,
                "status": "success"
            }
            # 这里可以调用session_manager的相关方法记录信息
            # self.session_manager.record_tool_execution(execution_record)
    
    def _record_error_to_session(self, error: Exception, **kwargs) -> None:
        """将错误信息记录到AgentScope会话中"""
        if self.session_manager:
            error_record = {
                "tool_name": self.name,
                "error_time": datetime.utcnow().isoformat(),
                "parameters": kwargs,
                "error": str(error),
                "error_type": type(error).__name__,
                "status": "error"
            }
            # 这里可以调用session_manager的相关方法记录错误
            # self.session_manager.record_tool_error(error_record)
    
    def get_agentscope_status(self) -> Dict[str, Any]:
        """获取AgentScope兼容的工具状态信息
        
        Returns:
            工具的状态信息
        """
        return {
            "name": self.name,
            "description": self.description,
            "mode": self.mode.value,
            "created_at": self.created_at.isoformat(),
            "last_executed": self.last_executed.isoformat() if self.last_executed else None,
            "execution_count": self.execution_count,
            "error_count": self.error_count,
            "success_rate": (self.execution_count - self.error_count) / max(self.execution_count, 1),
            "message_history_count": len(self.message_history),
            "agentscope_integration": {
                "session_manager_connected": self.session_manager is not None,
                "message_driven_capable": self.mode == AgentScopeToolMode.MESSAGE_DRIVEN,
                "framework_version": "0.2.0"
            }
        }
    
    def set_agent_context(self, context: Dict[str, Any]) -> None:
        """设置智能体上下文信息
        
        Args:
            context: 智能体上下文信息
        """
        self.agent_context.update(context)
    
    def get_message_history(self) -> List[Msg]:
        """获取工具的消息历史记录
        
        Returns:
            消息历史记录列表
        """
        return self.message_history.copy()
    
    def clear_message_history(self) -> None:
        """清空消息历史记录"""
        self.message_history.clear()
        self.logger.info(f"Cleared message history for tool {self.name}")
```

### 3.2 基于AgentScope的具体工具类型

#### 3.2.1 AgentScope同步工具
```python
class AgentScopeSyncTool(AgentScopeToolBase):
    """基于AgentScope的同步工具基类"""
    
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(
            name=name, 
            description=description, 
            mode=AgentScopeToolMode.SYNC,
            **kwargs
        )
    
    def _execute_core(self, **kwargs) -> Any:
        """同步执行核心逻辑
        
        子类必须重写此方法来实现具体的同步工具功能。
        
        Args:
            **kwargs: 执行参数
            
        Returns:
            执行结果
        """
        return self._execute_sync_logic(**kwargs)
    
    def _execute_sync_logic(self, **kwargs) -> Any:
        """同步执行逻辑
        
        子类重写此方法来实现具体的同步逻辑。
        
        Args:
            **kwargs: 执行参数
            
        Returns:
            执行结果
        """
        raise NotImplementedError("Subclasses must implement _execute_sync_logic method")

# 同步工具示例
class AgentScopeFileTool(AgentScopeSyncTool):
    """基于AgentScope的文件操作工具示例"""
    
    def __init__(self):
        super().__init__(
            name="file_tool",
            description="AgentScope文件操作工具，支持读取、写入、删除文件"
        )
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """获取文件工具的参数Schema"""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read", "write", "delete"],
                    "description": "文件操作类型"
                },
                "file_path": {
                    "type": "string",
                    "description": "文件路径"
                },
                "content": {
                    "type": "string",
                    "description": "写入文件的内容（仅在action为write时需要）"
                }
            },
            "required": ["action", "file_path"]
        }
    
    def _execute_sync_logic(self, **kwargs) -> Any:
        """执行文件操作"""
        action = kwargs.get("action")
        file_path = kwargs.get("file_path")
        content = kwargs.get("content")
        
        if action == "read":
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif action == "write":
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content or "")
            return f"File written successfully: {file_path}"
        elif action == "delete":
            import os
            os.remove(file_path)
            return f"File deleted successfully: {file_path}"
        else:
            raise ValueError(f"Unsupported action: {action}")
```

#### 3.2.2 AgentScope异步工具
```python
class AgentScopeAsyncTool(AgentScopeToolBase):
    """基于AgentScope的异步工具基类"""
    
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(
            name=name, 
            description=description, 
            mode=AgentScopeToolMode.ASYNC,
            **kwargs
        )
    
    async def _execute_core(self, **kwargs) -> Any:
        """异步执行核心逻辑"""
        return await self._execute_async_logic(**kwargs)
    
    async def _execute_async_logic(self, **kwargs) -> Any:
        """异步执行逻辑
        
        子类重写此方法来实现具体的异步逻辑。
        
        Args:
            **kwargs: 执行参数
            
        Returns:
            执行结果
        """
        raise NotImplementedError("Subclasses must implement _execute_async_logic method")

# 异步工具示例
class AgentScopeNetworkTool(AgentScopeAsyncTool):
    """基于AgentScope的网络请求工具示例"""
    
    def __init__(self):
        super().__init__(
            name="network_tool",
            description="AgentScope网络请求工具，支持HTTP GET/POST请求"
        )
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """获取网络工具的参数Schema"""
        return {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST"],
                    "description": "HTTP请求方法"
                },
                "url": {
                    "type": "string",
                    "description": "请求URL"
                },
                "data": {
                    "type": "object",
                    "description": "POST请求数据（可选）"
                },
                "headers": {
                    "type": "object",
                    "description": "请求头（可选）"
                }
            },
            "required": ["method", "url"]
        }
    
    async def _execute_async_logic(self, **kwargs) -> Any:
        """执行网络请求"""
        import aiohttp
        
        method = kwargs.get("method", "GET")
        url = kwargs.get("url")
        data = kwargs.get("data")
        headers = kwargs.get("headers", {})
        
        async with aiohttp.ClientSession() as session:
            if method == "GET":
                async with session.get(url, headers=headers) as response:
                    return {
                        "status": response.status,
                        "data": await response.text(),
                        "headers": dict(response.headers)
                    }
            elif method == "POST":
                async with session.post(url, json=data, headers=headers) as response:
                    return {
                        "status": response.status,
                        "data": await response.text(),
                        "headers": dict(response.headers)
                    }
```

#### 3.2.3 AgentScope流式工具
```python
class AgentScopeStreamingTool(AgentScopeToolBase):
    """基于AgentScope的流式工具基类"""
    
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(
            name=name, 
            description=description, 
            mode=AgentScopeToolMode.STREAMING,
            **kwargs
        )
    
    def _execute_core_streaming(self, **kwargs):
        """流式执行核心逻辑"""
        for chunk in self._execute_streaming_logic(**kwargs):
            yield chunk
    
    def _execute_streaming_logic(self, **kwargs):
        """流式执行逻辑
        
        子类重写此方法来实现具体的流式逻辑。
        
        Args:
            **kwargs: 执行参数
            
        Yields:
            执行过程中的数据块
        """
        raise NotImplementedError("Subclasses must implement _execute_streaming_logic method")

# 流式工具示例
class AgentScopeDataProcessingTool(AgentScopeStreamingTool):
    """基于AgentScope的数据处理工具示例"""
    
    def __init__(self):
        super().__init__(
            name="data_processing_tool",
            description="AgentScope数据处理工具，支持大数据集的流式处理"
        )
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """获取数据处理工具的参数Schema"""
        return {
            "type": "object",
            "properties": {
                "data_source": {
                    "type": "string",
                    "description": "数据源路径或URL"
                },
                "processing_type": {
                    "type": "string",
                    "enum": ["filter", "transform", "aggregate"],
                    "description": "数据处理类型"
                },
                "batch_size": {
                    "type": "integer",
                    "default": 100,
                    "description": "批处理大小"
                }
            },
            "required": ["data_source", "processing_type"]
        }
    
    def _execute_streaming_logic(self, **kwargs):
        """执行流式数据处理"""
        data_source = kwargs.get("data_source")
        processing_type = kwargs.get("processing_type")
        batch_size = kwargs.get("batch_size", 100)
        
        # 模拟数据流处理
        for i in range(0, 1000, batch_size):
            batch_data = self._load_batch_data(data_source, i, batch_size)
            processed_batch = self._process_batch(batch_data, processing_type)
            
            yield {
                "batch_id": i // batch_size,
                "processed_count": len(processed_batch),
                "data": processed_batch,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _load_batch_data(self, source: str, offset: int, size: int) -> List[Any]:
        """加载批次数据"""
        # 模拟数据加载
        return [f"data_{i}" for i in range(offset, offset + size)]
    
    def _process_batch(self, data: List[Any], processing_type: str) -> List[Any]:
        """处理批次数据"""
        if processing_type == "filter":
            return [item for item in data if "5" not in item]
        elif processing_type == "transform":
            return [item.upper() for item in data]
        elif processing_type == "aggregate":
            return [f"aggregated_{len(data)}_items"]
        else:
            return data
```

#### 3.2.4 AgentScope消息驱动工具
```python
class AgentScopeMessageDrivenTool(AgentScopeToolBase):
    """基于AgentScope的消息驱动工具基类"""
    
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(
            name=name, 
            description=description, 
            mode=AgentScopeToolMode.MESSAGE_DRIVEN,
            **kwargs
        )
    
    def _execute_core(self, **kwargs) -> Any:
        """消息驱动执行核心逻辑"""
        # 从智能体上下文中获取消息信息
        agent_context = self.agent_context
        message_history = self.message_history
        
        return self._execute_message_driven_logic(
            agent_context=agent_context,
            message_history=message_history,
            **kwargs
        )
    
    def _execute_message_driven_logic(
        self, 
        agent_context: Dict[str, Any],
        message_history: List[Msg],
        **kwargs
    ) -> Any:
        """消息驱动执行逻辑
        
        子类重写此方法来实现具体的消息驱动逻辑。
        
        Args:
            agent_context: 智能体上下文
            message_history: 消息历史
            **kwargs: 执行参数
            
        Returns:
            执行结果
        """
        raise NotImplementedError("Subclasses must implement _execute_message_driven_logic method")

# 消息驱动工具示例
class AgentScopeConversationTool(AgentScopeMessageDrivenTool):
    """基于AgentScope的对话工具示例"""
    
    def __init__(self):
        super().__init__(
            name="conversation_tool",
            description="AgentScope对话工具，基于消息历史进行智能对话"
        )
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """获取对话工具的参数Schema"""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用户查询内容"
                },
                "context_length": {
                    "type": "integer",
                    "default": 10,
                    "description": "考虑的历史消息数量"
                }
            },
            "required": ["query"]
        }
    
    def _execute_message_driven_logic(
        self, 
        agent_context: Dict[str, Any],
        message_history: List[Msg],
        **kwargs
    ) -> Any:
        """执行基于消息历史的对话"""
        query = kwargs.get("query")
        context_length = kwargs.get("context_length", 10)
        
        # 获取最近的消息历史
        recent_messages = message_history[-context_length:] if message_history else []
        
        # 构建对话上下文
        conversation_context = {
            "current_query": query,
            "recent_messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.metadata.get("timestamp") if msg.metadata else None
                }
                for msg in recent_messages
            ],
            "agent_info": agent_context
        }
        
        # 生成回复（这里是简化的示例）
        response = self._generate_response(conversation_context)
        
        return {
            "response": response,
            "context_used": len(recent_messages),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _generate_response(self, context: Dict[str, Any]) -> str:
        """生成对话回复"""
        query = context["current_query"]
        recent_count = len(context["recent_messages"])
        
        # 简化的回复生成逻辑
        return f"基于{recent_count}条历史消息，对于您的问题'{query}'，我的回复是..."
```

## 4. 基于AgentScope的工具管理系统

### 4.1 AgentScope工具注册表
```python
class AgentScopeToolRegistry:
    """基于AgentScope的工具注册表"""
    
    def __init__(self):
        self._tools: Dict[str, AgentScopeToolBase] = {}
        self._tool_categories: Dict[str, List[str]] = defaultdict(list)
        self._tool_modes: Dict[AgentScopeToolMode, List[str]] = defaultdict(list)
        self._tool_metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._logger = logging.getLogger(__name__)
    
    def register_tool(self, tool: AgentScopeToolBase) -> bool:
        """注册AgentScope工具
        
        Args:
            tool: 要注册的AgentScope工具实例
            
        Returns:
            注册是否成功
        """
        with self._lock:
            try:
                # 验证工具是否符合AgentScope规范
                if not self._validate_agentscope_tool(tool):
                    self._logger.error(f"工具 {tool.name} 不符合AgentScope规范")
                    return False
                
                # 检查名称冲突
                if tool.name in self._tools:
                    self._logger.warning(f"AgentScope工具 {tool.name} 已经注册")
                    return False
                
                # 注册工具
                self._tools[tool.name] = tool
                
                # 按类别分类
                if hasattr(tool, 'category') and tool.category:
                    self._tool_categories[tool.category].append(tool.name)
                
                # 按模式分类
                self._tool_modes[tool.mode].append(tool.name)
                
                # 保存工具元数据
                self._tool_metadata[tool.name] = {
                    "registered_at": datetime.utcnow().isoformat(),
                    "mode": tool.mode.value,
                    "category": getattr(tool, 'category', 'general'),
                    "version": getattr(tool, 'version', '1.0.0'),
                    "description": tool.description,
                    "schema": tool.get_schema()
                }
                
                self._logger.info(f"AgentScope工具 {tool.name} 注册成功")
                return True
                
            except Exception as e:
                self._logger.error(f"注册工具 {tool.name} 失败: {e}")
                return False
    
    def unregister_tool(self, name: str) -> bool:
        """注销工具
        
        Args:
            name: 工具名称
            
        Returns:
            注销是否成功
        """
        with self._lock:
            if name not in self._tools:
                self._logger.warning(f"工具 {name} 未找到")
                return False
            
            tool = self._tools[name]
            
            # 从分类中移除
            if hasattr(tool, 'category') and tool.category:
                if name in self._tool_categories[tool.category]:
                    self._tool_categories[tool.category].remove(name)
            
            # 从模式分类中移除
            if name in self._tool_modes[tool.mode]:
                self._tool_modes[tool.mode].remove(name)
            
            # 删除工具和元数据
            del self._tools[name]
            del self._tool_metadata[name]
            
            self._logger.info(f"工具 {name} 注销成功")
            return True
    
    def get_tool(self, name: str) -> Optional[AgentScopeToolBase]:
        """获取工具
        
        Args:
            name: 工具名称
            
        Returns:
            工具实例或None
        """
        return self._tools.get(name)
    
    def list_tools(
        self, 
        category: Optional[str] = None,
        mode: Optional[AgentScopeToolMode] = None
    ) -> List[str]:
        """列出工具
        
        Args:
            category: 工具类别（可选）
            mode: 工具模式（可选）
            
        Returns:
            工具名称列表
        """
        if category is not None:
            return self._tool_categories.get(category, [])
        
        if mode is not None:
            return self._tool_modes.get(mode, [])
        
        return list(self._tools.keys())
    
    def search_tools(self, keyword: str) -> List[Dict[str, Any]]:
        """搜索工具
        
        Args:
            keyword: 搜索关键词
            
        Returns:
            匹配的工具信息列表
        """
        keyword = keyword.lower()
        results = []
        
        for name, tool in self._tools.items():
            metadata = self._tool_metadata[name]
            
            # 搜索名称、描述和类别
            if (keyword in name.lower() or 
                keyword in tool.description.lower() or
                keyword in metadata.get('category', '').lower()):
                
                results.append({
                    "name": name,
                    "description": tool.description,
                    "mode": tool.mode.value,
                    "category": metadata.get('category', 'general'),
                    "registered_at": metadata.get('registered_at')
                })
        
        return results
    
    def get_tool_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """获取工具Schema
        
        Args:
            name: 工具名称
            
        Returns:
            工具Schema或None
        """
        tool = self.get_tool(name)
        if tool:
            return tool.get_schema()
        return None
    
    def get_tool_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """获取工具元数据
        
        Args:
            name: 工具名称
            
        Returns:
            工具元数据或None
        """
        return self._tool_metadata.get(name)
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """获取注册表统计信息
        
        Returns:
            注册表统计信息
        """
        total_tools = len(self._tools)
        
        # 按模式统计
        mode_stats = {}
        for mode, tools in self._tool_modes.items():
            mode_stats[mode.value] = len(tools)
        
        # 按类别统计
        category_stats = {}
        for category, tools in self._tool_categories.items():
            category_stats[category] = len(tools)
        
        return {
            "total_tools": total_tools,
            "mode_distribution": mode_stats,
            "category_distribution": category_stats,
            "registered_tools": list(self._tools.keys())
        }
    
    def _validate_agentscope_tool(self, tool: AgentScopeToolBase) -> bool:
        """验证工具是否符合AgentScope规范
        
        Args:
            tool: 要验证的工具
            
        Returns:
            是否符合规范
        """
        # 检查必要的属性
        required_attrs = ['name', 'description', 'mode']
        for attr in required_attrs:
            if not hasattr(tool, attr) or not getattr(tool, attr):
                self._logger.error(f"工具缺少必要属性: {attr}")
                return False
        
        # 检查必要的方法
        required_methods = ['__call__', 'get_schema']
        for method in required_methods:
            if not hasattr(tool, method) or not callable(getattr(tool, method)):
                self._logger.error(f"工具缺少必要方法: {method}")
                return False
        
        # 检查模式是否有效
        if not isinstance(tool.mode, AgentScopeToolMode):
            self._logger.error(f"无效的工具模式: {tool.mode}")
            return False
        
        # 验证Schema
        try:
            schema = tool.get_schema()
            if not isinstance(schema, dict) or 'name' not in schema:
                self._logger.error("工具Schema格式无效")
                return False
        except Exception as e:
            self._logger.error(f"获取工具Schema失败: {e}")
            return False
        
        return True
```

### 4.2 AgentScope工具管理器
```python
class AgentScopeToolManager:
    """基于AgentScope的工具管理器"""
    
    def __init__(self, session_manager: Optional[SessionManager] = None):
        self._registry = AgentScopeToolRegistry()
        self._executor = AgentScopeToolExecutor()
        self._session_manager = session_manager
        self._access_control = AgentScopeAccessControl()
        self._performance_monitor = ToolPerformanceMonitor()
        self._logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
    
    def register_tool(self, tool: AgentScopeToolBase) -> bool:
        """注册工具到管理器
        
        Args:
            tool: 要注册的AgentScope工具
            
        Returns:
            注册是否成功
        """
        success = self._registry.register_tool(tool)
        if success:
            self._logger.info(f"工具 {tool.name} 已注册到管理器")
        return success
    
    def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """执行AgentScope工具
        
        Args:
            tool_name: 工具名称
            parameters: 工具参数
            agent_context: 智能体上下文
            session_id: 会话ID
            
        Returns:
            工具执行结果
        """
        with self._lock:
            start_time = time.time()
            
            try:
                # 获取工具
                tool = self._registry.get_tool(tool_name)
                if not tool:
                    error_msg = f"工具 {tool_name} 未找到"
                    self._logger.error(error_msg)
                    return {
                        "success": False,
                        "error": error_msg,
                        "tool_name": tool_name,
                        "execution_time": 0
                    }
                
                # 检查访问权限
                if not self._access_control.check_permission(
                    tool_name, agent_context, session_id
                ):
                    error_msg = f"工具 {tool_name} 访问被拒绝"
                    self._logger.warning(error_msg)
                    return {
                        "success": False,
                        "error": error_msg,
                        "tool_name": tool_name,
                        "execution_time": 0
                    }
                
                # 验证参数
                validation_result = self._validate_parameters(tool, parameters)
                if not validation_result["valid"]:
                    error_msg = f"工具 {tool_name} 参数验证失败: {validation_result['error']}"
                    self._logger.error(error_msg)
                    return {
                        "success": False,
                        "error": error_msg,
                        "tool_name": tool_name,
                        "execution_time": 0
                    }
                
                # 执行工具
                result = self._executor.execute(
                    tool=tool,
                    parameters=parameters,
                    agent_context=agent_context,
                    session_id=session_id
                )
                
                # 记录性能指标
                execution_time = time.time() - start_time
                self._performance_monitor.record_execution(
                    tool_name=tool_name,
                    execution_time=execution_time,
                    success=result.get("success", False),
                    parameters_size=len(str(parameters))
                )
                
                # 更新会话状态
                if self._session_manager and session_id:
                    self._update_session_tool_usage(
                        session_id, tool_name, result
                    )
                
                result["execution_time"] = execution_time
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                error_msg = f"执行工具 {tool_name} 时发生异常: {str(e)}"
                self._logger.error(error_msg, exc_info=True)
                
                # 记录失败的性能指标
                self._performance_monitor.record_execution(
                    tool_name=tool_name,
                    execution_time=execution_time,
                    success=False,
                    error=str(e)
                )
                
                return {
                    "success": False,
                    "error": error_msg,
                    "tool_name": tool_name,
                    "execution_time": execution_time
                }
    
    def configure_tool(
        self, 
        tool_name: str, 
        config: Dict[str, Any]
    ) -> bool:
        """配置工具
        
        Args:
            tool_name: 工具名称
            config: 配置参数
            
        Returns:
            配置是否成功
        """
        tool = self._registry.get_tool(tool_name)
        if not tool:
            self._logger.error(f"工具 {tool_name} 未找到")
            return False
        
        try:
            if hasattr(tool, 'configure'):
                tool.configure(config)
                self._logger.info(f"工具 {tool_name} 配置成功")
                return True
            else:
                self._logger.warning(f"工具 {tool_name} 不支持配置")
                return False
        except Exception as e:
            self._logger.error(f"配置工具 {tool_name} 失败: {e}")
            return False
    
    def set_access_control_rules(self, rules: List[Dict[str, Any]]) -> None:
        """设置访问控制规则
        
        Args:
            rules: 访问控制规则列表
        """
        self._access_control.set_rules(rules)
        self._logger.info(f"已设置 {len(rules)} 条访问控制规则")
    
    def get_tool_performance_stats(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """获取工具性能统计
        
        Args:
            tool_name: 工具名称
            
        Returns:
            性能统计信息或None
        """
        return self._performance_monitor.get_tool_stats(tool_name)
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """获取管理器统计信息
        
        Returns:
            管理器统计信息
        """
        registry_stats = self._registry.get_registry_stats()
        performance_stats = self._performance_monitor.get_overall_stats()
        
        return {
            "registry": registry_stats,
            "performance": performance_stats,
            "access_control": self._access_control.get_stats()
        }
    
    def list_available_tools(
        self, 
        category: Optional[str] = None,
        mode: Optional[AgentScopeToolMode] = None
    ) -> List[Dict[str, Any]]:
        """列出可用工具
        
        Args:
            category: 工具类别过滤
            mode: 工具模式过滤
            
        Returns:
            可用工具列表
        """
        tool_names = self._registry.list_tools(category=category, mode=mode)
        tools_info = []
        
        for name in tool_names:
            tool = self._registry.get_tool(name)
            metadata = self._registry.get_tool_metadata(name)
            
            if tool and metadata:
                tools_info.append({
                    "name": name,
                    "description": tool.description,
                    "mode": tool.mode.value,
                    "category": metadata.get('category', 'general'),
                    "version": metadata.get('version', '1.0.0'),
                    "registered_at": metadata.get('registered_at')
                })
        
        return tools_info
    
    def _validate_parameters(
        self, 
        tool: AgentScopeToolBase, 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """验证工具参数
        
        Args:
            tool: 工具实例
            parameters: 参数字典
            
        Returns:
            验证结果
        """
        try:
            schema = tool.get_schema()
            
            # 基本的参数验证
            if 'parameters' in schema:
                required_params = schema['parameters'].get('required', [])
                
                # 检查必需参数
                for param in required_params:
                    if param not in parameters:
                        return {
                            "valid": False,
                            "error": f"缺少必需参数: {param}"
                        }
                
                # 检查参数类型（简单验证）
                param_properties = schema['parameters'].get('properties', {})
                for param_name, param_value in parameters.items():
                    if param_name in param_properties:
                        expected_type = param_properties[param_name].get('type')
                        if expected_type and not self._check_parameter_type(
                            param_value, expected_type
                        ):
                            return {
                                "valid": False,
                                "error": f"参数 {param_name} 类型不匹配，期望: {expected_type}"
                            }
            
            return {"valid": True}
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"参数验证异常: {str(e)}"
            }
    
    def _check_parameter_type(self, value: Any, expected_type: str) -> bool:
        """检查参数类型
        
        Args:
            value: 参数值
            expected_type: 期望的类型
            
        Returns:
            类型是否匹配
        """
        type_mapping = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True  # 未知类型默认通过
    
    def _update_session_tool_usage(
        self, 
        session_id: str, 
        tool_name: str, 
        result: Dict[str, Any]
    ) -> None:
        """更新会话工具使用记录
        
        Args:
            session_id: 会话ID
            tool_name: 工具名称
            result: 执行结果
        """
        try:
            if self._session_manager:
                usage_info = {
                    "tool_name": tool_name,
                    "timestamp": datetime.utcnow().isoformat(),
                    "success": result.get("success", False),
                    "execution_time": result.get("execution_time", 0)
                }
                
                # 这里假设SessionManager有记录工具使用的方法
                # 实际实现需要根据AgentScope的SessionManager API调整
                if hasattr(self._session_manager, 'record_tool_usage'):
                    self._session_manager.record_tool_usage(session_id, usage_info)
                    
        except Exception as e:
            self._logger.warning(f"更新会话工具使用记录失败: {e}")
```

### 4.3 AgentScope工具执行器
```python
class AgentScopeToolExecutor:
    """基于AgentScope的工具执行器"""
    
    def __init__(
        self, 
        sandbox_enabled: bool = True, 
        timeout: int = 30,
        max_concurrent_executions: int = 10
    ):
        self._sandbox_enabled = sandbox_enabled
        self._timeout = timeout
        self._max_concurrent_executions = max_concurrent_executions
        self._execution_history: List[AgentScopeExecutionRecord] = []
        self._active_executions: Dict[str, asyncio.Task] = {}
        self._execution_semaphore = asyncio.Semaphore(max_concurrent_executions)
        self._logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
    
    def execute(
        self,
        tool: AgentScopeToolBase,
        parameters: Dict[str, Any],
        agent_context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """执行AgentScope工具
        
        Args:
            tool: 要执行的工具
            parameters: 工具参数
            agent_context: 智能体上下文
            session_id: 会话ID
            
        Returns:
            执行结果
        """
        start_time = time.time()
        execution_id = self._generate_execution_id()
        
        # 创建执行记录
        record = AgentScopeExecutionRecord(
            execution_id=execution_id,
            tool_name=tool.name,
            tool_mode=tool.mode,
            start_time=start_time,
            parameters=parameters,
            agent_context=agent_context,
            session_id=session_id
        )
        
        try:
            # 根据工具模式选择执行方式
            if tool.mode == AgentScopeToolMode.SYNC:
                result = self._execute_sync_tool(tool, parameters, agent_context)
            elif tool.mode == AgentScopeToolMode.ASYNC:
                result = self._execute_async_tool(tool, parameters, agent_context)
            elif tool.mode == AgentScopeToolMode.STREAMING:
                result = self._execute_streaming_tool(tool, parameters, agent_context)
            elif tool.mode == AgentScopeToolMode.MESSAGE_DRIVEN:
                result = self._execute_message_driven_tool(tool, parameters, agent_context)
            else:
                raise ValueError(f"不支持的工具模式: {tool.mode}")
            
            # 记录执行结果
            record.end_time = time.time()
            record.execution_time = record.end_time - record.start_time
            record.success = result.get("success", False)
            record.result = result
            
            with self._lock:
                self._execution_history.append(record)
            
            return result
            
        except Exception as e:
            # 记录执行失败
            record.end_time = time.time()
            record.execution_time = record.end_time - record.start_time
            record.success = False
            record.error = str(e)
            
            with self._lock:
                self._execution_history.append(record)
            
            error_msg = f"工具 {tool.name} 执行失败: {str(e)}"
            self._logger.error(error_msg, exc_info=True)
            
            return {
                "success": False,
                "error": error_msg,
                "execution_id": execution_id,
                "execution_time": record.execution_time
            }
    
    def _execute_sync_tool(
        self,
        tool: AgentScopeToolBase,
        parameters: Dict[str, Any],
        agent_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """执行同步工具
        
        Args:
            tool: 同步工具
            parameters: 参数
            agent_context: 智能体上下文
            
        Returns:
            执行结果
        """
        try:
            if self._sandbox_enabled:
                result = self._execute_in_sandbox(tool, parameters, agent_context)
            else:
                result = tool(**parameters)
            
            return {
                "success": True,
                "result": result,
                "tool_name": tool.name,
                "mode": tool.mode.value
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool_name": tool.name,
                "mode": tool.mode.value
            }
    
    def _execute_async_tool(
        self,
        tool: AgentScopeToolBase,
        parameters: Dict[str, Any],
        agent_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """执行异步工具
        
        Args:
            tool: 异步工具
            parameters: 参数
            agent_context: 智能体上下文
            
        Returns:
            执行结果
        """
        try:
            # 对于异步工具，我们在同步上下文中运行
            import asyncio
            
            async def _async_execute():
                if self._sandbox_enabled:
                    return await self._execute_in_sandbox_async(tool, parameters, agent_context)
                else:
                    return await tool(**parameters)
            
            # 获取或创建事件循环
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if loop.is_running():
                # 如果循环正在运行，创建任务
                task = loop.create_task(_async_execute())
                result = task.result()  # 这可能需要调整
            else:
                # 如果循环未运行，直接运行
                result = loop.run_until_complete(_async_execute())
            
            return {
                "success": True,
                "result": result,
                "tool_name": tool.name,
                "mode": tool.mode.value
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool_name": tool.name,
                "mode": tool.mode.value
            }
    
    def _execute_streaming_tool(
        self,
        tool: AgentScopeToolBase,
        parameters: Dict[str, Any],
        agent_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """执行流式工具
        
        Args:
            tool: 流式工具
            parameters: 参数
            agent_context: 智能体上下文
            
        Returns:
            执行结果
        """
        try:
            # 流式工具返回生成器
            stream = tool(**parameters)
            
            # 收集流式结果
            results = []
            for chunk in stream:
                results.append(chunk)
            
            return {
                "success": True,
                "result": results,
                "stream_length": len(results),
                "tool_name": tool.name,
                "mode": tool.mode.value
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool_name": tool.name,
                "mode": tool.mode.value
            }
    
    def _execute_message_driven_tool(
        self,
        tool: AgentScopeToolBase,
        parameters: Dict[str, Any],
        agent_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """执行消息驱动工具
        
        Args:
            tool: 消息驱动工具
            parameters: 参数
            agent_context: 智能体上下文
            
        Returns:
            执行结果
        """
        try:
            # 消息驱动工具需要消息对象
            from agentscope.message import Msg
            
            # 创建消息对象
            if 'message' not in parameters:
                # 如果没有提供消息，从参数创建一个
                message = Msg(
                    name=agent_context.get('agent_id', 'unknown') if agent_context else 'unknown',
                    content=parameters.get('content', ''),
                    role='user'
                )
                parameters['message'] = message
            
            result = tool(**parameters)
            
            return {
                "success": True,
                "result": result,
                "tool_name": tool.name,
                "mode": tool.mode.value
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool_name": tool.name,
                "mode": tool.mode.value
            }
    
    def _execute_in_sandbox(
        self,
        tool: AgentScopeToolBase,
        parameters: Dict[str, Any],
        agent_context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """在沙箱中执行工具
        
        Args:
            tool: 工具实例
            parameters: 参数
            agent_context: 智能体上下文
            
        Returns:
            执行结果
        """
        # 创建沙箱环境
        sandbox = AgentScopeToolSandbox(
            timeout=self._timeout,
            agent_context=agent_context
        )
        
        try:
            return sandbox.execute(tool, parameters)
        finally:
            sandbox.cleanup()
    
    async def _execute_in_sandbox_async(
        self,
        tool: AgentScopeToolBase,
        parameters: Dict[str, Any],
        agent_context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """在沙箱中异步执行工具
        
        Args:
            tool: 工具实例
            parameters: 参数
            agent_context: 智能体上下文
            
        Returns:
            执行结果
        """
        # 创建异步沙箱环境
        sandbox = AgentScopeToolSandbox(
            timeout=self._timeout,
            agent_context=agent_context
        )
        
        try:
            return await sandbox.execute_async(tool, parameters)
        finally:
            await sandbox.cleanup_async()
    
    def _generate_execution_id(self) -> str:
        """生成执行ID
        
        Returns:
            唯一的执行ID
        """
        import uuid
        return f"agentscope_exec_{uuid.uuid4().hex[:8]}_{int(time.time() * 1000)}"
    
    def get_execution_history(
        self, 
        tool_name: Optional[str] = None, 
        session_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取执行历史
        
        Args:
            tool_name: 工具名称过滤
            session_id: 会话ID过滤
            limit: 返回记录数限制
            
        Returns:
            执行历史记录列表
        """
        with self._lock:
            history = self._execution_history.copy()
        
        # 应用过滤器
        if tool_name:
            history = [r for r in history if r.tool_name == tool_name]
        
        if session_id:
            history = [r for r in history if r.session_id == session_id]
        
        # 转换为字典格式并限制数量
        result = []
        for record in history[-limit:]:
            result.append({
                "execution_id": record.execution_id,
                "tool_name": record.tool_name,
                "tool_mode": record.tool_mode.value if record.tool_mode else None,
                "start_time": record.start_time,
                "end_time": record.end_time,
                "execution_time": record.execution_time,
                "success": record.success,
                "session_id": record.session_id,
                "error": record.error
            })
        
        return result
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计信息
        
        Returns:
            执行统计信息
        """
        with self._lock:
            total_executions = len(self._execution_history)
            successful_executions = sum(1 for r in self._execution_history if r.success)
            failed_executions = total_executions - successful_executions
            
            # 按工具统计
            tool_stats = {}
            for record in self._execution_history:
                tool_name = record.tool_name
                if tool_name not in tool_stats:
                    tool_stats[tool_name] = {
                        "total": 0,
                        "successful": 0,
                        "failed": 0,
                        "avg_execution_time": 0
                    }
                
                tool_stats[tool_name]["total"] += 1
                if record.success:
                    tool_stats[tool_name]["successful"] += 1
                else:
                    tool_stats[tool_name]["failed"] += 1
                
                if record.execution_time:
                    current_avg = tool_stats[tool_name]["avg_execution_time"]
                    total_count = tool_stats[tool_name]["total"]
                    tool_stats[tool_name]["avg_execution_time"] = (
                        (current_avg * (total_count - 1) + record.execution_time) / total_count
                    )
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
            "tool_stats": tool_stats,
            "active_executions": len(self._active_executions)
        }

@dataclass
class AgentScopeExecutionRecord:
    """AgentScope工具执行记录"""
    execution_id: str
    tool_name: str
    tool_mode: AgentScopeToolMode
    start_time: float
    parameters: Dict[str, Any]
    agent_context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    end_time: Optional[float] = None
    execution_time: Optional[float] = None
    success: Optional[bool] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
```

## 5. AgentScope沙箱执行环境

### 5.1 AgentScope工具沙箱
```python
class AgentScopeToolSandbox:
    """基于AgentScope的工具沙箱"""
    
    def __init__(
        self, 
        timeout: int = 30,
        agent_context: Optional[Dict[str, Any]] = None
    ):
        self._timeout = timeout
        self._agent_context = agent_context or {}
        self._resource_limits = {
            "max_memory": 200 * 1024 * 1024,  # 200MB
            "max_cpu_time": 30,  # 30秒
            "max_file_size": 50 * 1024 * 1024,  # 50MB
            "max_network_requests": 10,  # 最大网络请求数
            "allowed_modules": [
                "json", "re", "datetime", "math", "random", "uuid",
                "base64", "hashlib", "urllib.parse", "collections"
            ],
            "blocked_modules": [
                "os", "sys", "subprocess", "socket", "threading",
                "multiprocessing", "ctypes", "importlib"
            ]
        }
        self._temp_dir = None
        self._process = None
        self._session_id = self._agent_context.get('session_id')
        self._agent_id = self._agent_context.get('agent_id')
        self._logger = logging.getLogger(__name__)
    
    def execute(
        self, 
        tool: AgentScopeToolBase, 
        parameters: Dict[str, Any]
    ) -> Any:
        """在沙箱中同步执行工具
        
        Args:
            tool: AgentScope工具实例
            parameters: 工具参数
            
        Returns:
            执行结果
        """
        # 创建临时目录
        self._temp_dir = tempfile.mkdtemp(
            prefix=f"agentscope_sandbox_{self._agent_id or 'unknown'}_"
        )
        
        try:
            # 设置资源限制
            self._setup_resource_limits()
            
            # 创建安全执行环境
            safe_env = self._create_safe_environment()
            
            # 在受限环境中执行工具
            result = self._execute_with_limits(tool, parameters, safe_env)
            
            return result
            
        except Exception as e:
            error_msg = f"沙箱执行失败: {str(e)}"
            self._logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
            
        finally:
            # 清理资源
            self.cleanup()
    
    async def execute_async(
        self, 
        tool: AgentScopeToolBase, 
        parameters: Dict[str, Any]
    ) -> Any:
        """在沙箱中异步执行工具
        
        Args:
            tool: AgentScope工具实例
            parameters: 工具参数
            
        Returns:
            执行结果
        """
        # 创建临时目录
        self._temp_dir = tempfile.mkdtemp(
            prefix=f"agentscope_sandbox_{self._agent_id or 'unknown'}_"
        )
        
        try:
            # 设置资源限制
            self._setup_resource_limits()
            
            # 创建安全执行环境
            safe_env = self._create_safe_environment()
            
            # 在受限环境中异步执行工具
            result = await self._execute_with_limits_async(tool, parameters, safe_env)
            
            return result
            
        except Exception as e:
            error_msg = f"异步沙箱执行失败: {str(e)}"
            self._logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
            
        finally:
            # 清理资源
            await self.cleanup_async()
    
    def _setup_resource_limits(self):
        """设置资源限制
        
        注意: 在Windows环境下，某些资源限制可能不可用
        """
        try:
            import resource
            
            # 设置内存限制 (仅在Unix系统上可用)
            if hasattr(resource, 'RLIMIT_AS') and "max_memory" in self._resource_limits:
                try:
                    resource.setrlimit(
                        resource.RLIMIT_AS, 
                        (self._resource_limits["max_memory"], 
                         self._resource_limits["max_memory"])
                    )
                except (OSError, ValueError):
                    # Windows或其他系统可能不支持
                    self._logger.warning("无法设置内存限制")
            
            # 设置CPU时间限制
            if hasattr(resource, 'RLIMIT_CPU') and "max_cpu_time" in self._resource_limits:
                try:
                    resource.setrlimit(
                        resource.RLIMIT_CPU, 
                        (self._resource_limits["max_cpu_time"], 
                         self._resource_limits["max_cpu_time"])
                    )
                except (OSError, ValueError):
                    self._logger.warning("无法设置CPU时间限制")
                    
        except ImportError:
            # resource模块在某些系统上可能不可用
            self._logger.warning("resource模块不可用，跳过资源限制设置")
    
    def _create_safe_environment(self) -> Dict[str, Any]:
        """创建安全的执行环境
        
        Returns:
            安全的全局环境字典
        """
        allowed_modules = self._resource_limits.get("allowed_modules", [])
        blocked_modules = self._resource_limits.get("blocked_modules", [])
        
        # 创建受限的内置函数集合
        safe_builtins = {
            # 基本数据类型
            "len": len, "str": str, "int": int, "float": float, "bool": bool,
            "list": list, "dict": dict, "tuple": tuple, "set": set,
            
            # 迭代和序列操作
            "range": range, "enumerate": enumerate, "zip": zip,
            "map": map, "filter": filter, "sorted": sorted,
            
            # 数学函数
            "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
            "pow": pow, "divmod": divmod,
            
            # 类型检查
            "isinstance": isinstance, "issubclass": issubclass,
            "type": type, "hasattr": hasattr, "getattr": getattr,
            
            # 异常
            "Exception": Exception, "ValueError": ValueError,
            "TypeError": TypeError, "KeyError": KeyError,
            "IndexError": IndexError, "AttributeError": AttributeError,
            
            # 其他安全函数
            "print": print, "repr": repr, "format": format,
            "ord": ord, "chr": chr, "hex": hex, "oct": oct, "bin": bin
        }
        
        # 创建受限的全局环境
        safe_globals = {
            "__builtins__": safe_builtins,
            "__name__": "__agentscope_sandbox__",
            "__doc__": None,
            "__package__": None
        }
        
        # 添加允许的模块
        for module_name in allowed_modules:
            if module_name not in blocked_modules:
                try:
                    safe_globals[module_name] = __import__(module_name)
                except ImportError as e:
                    self._logger.warning(f"无法导入模块 {module_name}: {e}")
        
        # 添加AgentScope相关的安全上下文
        safe_globals["_agentscope_context"] = {
            "session_id": self._session_id,
            "agent_id": self._agent_id,
            "temp_dir": self._temp_dir,
            "resource_limits": self._resource_limits.copy()
        }
        
        return safe_globals
    
    def _execute_with_limits(
        self, 
        tool: AgentScopeToolBase, 
        parameters: Dict[str, Any],
        safe_env: Dict[str, Any]
    ) -> Any:
        """在限制条件下执行工具
        
        Args:
            tool: 工具实例
            parameters: 参数
            safe_env: 安全环境
            
        Returns:
            执行结果
        """
        import signal
        import threading
        
        result = None
        exception = None
        
        def target():
            nonlocal result, exception
            try:
                # 在安全环境中执行工具
                old_globals = globals().copy()
                try:
                    # 临时替换全局环境
                    globals().update(safe_env)
                    result = tool(**parameters)
                finally:
                    # 恢复原始全局环境
                    globals().clear()
                    globals().update(old_globals)
            except Exception as e:
                exception = e
        
        # 创建执行线程
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        
        # 等待执行完成或超时
        thread.join(timeout=self._timeout)
        
        if thread.is_alive():
            # 超时处理
            raise TimeoutError(f"工具执行超时 ({self._timeout}秒)")
        
        if exception:
            raise exception
        
        return result
    
    async def _execute_with_limits_async(
        self, 
        tool: AgentScopeToolBase, 
        parameters: Dict[str, Any],
        safe_env: Dict[str, Any]
    ) -> Any:
        """在限制条件下异步执行工具
        
        Args:
            tool: 工具实例
            parameters: 参数
            safe_env: 安全环境
            
        Returns:
            执行结果
        """
        import asyncio
        
        async def execute_coro():
            # 在安全环境中异步执行工具
            old_globals = globals().copy()
            try:
                # 临时替换全局环境
                globals().update(safe_env)
                if asyncio.iscoroutinefunction(tool):
                    return await tool(**parameters)
                else:
                    # 如果不是协程函数，在线程池中执行
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, lambda: tool(**parameters))
            finally:
                # 恢复原始全局环境
                globals().clear()
                globals().update(old_globals)
        
        # 使用asyncio.wait_for实现超时控制
        try:
            return await asyncio.wait_for(execute_coro(), timeout=self._timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"异步工具执行超时 ({self._timeout}秒)")
    
    def cleanup(self):
        """清理沙箱环境"""
        # 终止进程
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception as e:
                self._logger.warning(f"清理进程时出错: {e}")
                try:
                    self._process.kill()
                except:
                    pass
        
        # 删除临时目录
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            except Exception as e:
                self._logger.warning(f"清理临时目录时出错: {e}")
    
    async def cleanup_async(self):
        """异步清理沙箱环境"""
        # 异步终止进程
        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except Exception as e:
                self._logger.warning(f"异步清理进程时出错: {e}")
                try:
                    self._process.kill()
                except:
                    pass
        
        # 删除临时目录
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                # 在线程池中执行文件系统操作
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, 
                    lambda: shutil.rmtree(self._temp_dir, ignore_errors=True)
                )
            except Exception as e:
                self._logger.warning(f"异步清理临时目录时出错: {e}")
    
    def get_sandbox_info(self) -> Dict[str, Any]:
        """获取沙箱信息
        
        Returns:
            沙箱状态信息
        """
        return {
            "session_id": self._session_id,
            "agent_id": self._agent_id,
            "timeout": self._timeout,
            "temp_dir": self._temp_dir,
            "resource_limits": self._resource_limits.copy(),
            "process_active": self._process is not None and self._process.poll() is None
        }
```

## 6. AgentScope内置工具实现

### 6.1 AgentScope文件操作工具
```python
class AgentScopeFileReadTool(AgentScopeToolBase):
    """基于AgentScope的文件读取工具"""
    
    def __init__(self, agent_context: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="agentscope_file_read",
            description="基于AgentScope的安全文件读取工具",
            agent_context=agent_context
        )
        self._max_file_size = 10 * 1024 * 1024  # 10MB
        self._allowed_extensions = {
            ".txt", ".json", ".yaml", ".yml", ".md", ".csv",
            ".py", ".js", ".html", ".css", ".xml", ".log"
        }
        self._blocked_paths = {
            "/etc/passwd", "/etc/shadow", "/proc", "/sys",
            "C:\\Windows\\System32", "C:\\Windows\\SysWOW64"
        }
    
    def get_schema(self) -> Dict[str, Any]:
        """获取工具模式定义
        
        Returns:
            工具的JSON Schema定义
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "要读取的文件路径（相对路径或绝对路径）"
                        },
                        "encoding": {
                            "type": "string",
                            "description": "文件编码格式",
                            "default": "utf-8",
                            "enum": ["utf-8", "gbk", "ascii", "latin-1"]
                        },
                        "max_lines": {
                            "type": "integer",
                            "description": "最大读取行数（0表示读取全部）",
                            "default": 0,
                            "minimum": 0
                        }
                    },
                    "required": ["file_path"]
                }
            }
        }
    
    def __call__(
        self, 
        file_path: str, 
        encoding: str = "utf-8",
        max_lines: int = 0
    ) -> Dict[str, Any]:
        """执行文件读取操作
        
        Args:
            file_path: 文件路径
            encoding: 文件编码
            max_lines: 最大读取行数
            
        Returns:
            包含文件内容和元数据的字典
        """
        try:
            # 路径安全检查
            if not self._is_safe_path(file_path):
                return {
                    "success": False,
                    "error": "不安全的文件路径",
                    "error_type": "SecurityError",
                    "agent_id": self._agent_context.get('agent_id'),
                    "session_id": self._agent_context.get('session_id')
                }
            
            # 文件存在性检查
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return {
                    "success": False,
                    "error": f"文件不存在: {file_path}",
                    "error_type": "FileNotFoundError",
                    "agent_id": self._agent_context.get('agent_id'),
                    "session_id": self._agent_context.get('session_id')
                }
            
            # 文件大小检查
            file_size = file_path_obj.stat().st_size
            if file_size > self._max_file_size:
                return {
                    "success": False,
                    "error": f"文件过大: {file_size} bytes (最大: {self._max_file_size} bytes)",
                    "error_type": "FileSizeError",
                    "agent_id": self._agent_context.get('agent_id'),
                    "session_id": self._agent_context.get('session_id')
                }
            
            # 读取文件内容
            with open(file_path, 'r', encoding=encoding) as f:
                if max_lines > 0:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            break
                        lines.append(line.rstrip('\n\r'))
                    content = '\n'.join(lines)
                    truncated = True
                else:
                    content = f.read()
                    truncated = False
            
            # 记录执行日志
            self._log_execution({
                "action": "file_read",
                "file_path": file_path,
                "file_size": file_size,
                "encoding": encoding,
                "lines_read": len(content.split('\n')) if content else 0,
                "truncated": truncated
            })
            
            return {
                "success": True,
                "content": content,
                "metadata": {
                    "file_path": str(file_path_obj.absolute()),
                    "file_size": file_size,
                    "encoding": encoding,
                    "lines_count": len(content.split('\n')) if content else 0,
                    "truncated": truncated,
                    "read_time": datetime.utcnow().isoformat()
                },
                "agent_id": self._agent_context.get('agent_id'),
                "session_id": self._agent_context.get('session_id')
            }
            
        except UnicodeDecodeError as e:
            return {
                "success": False,
                "error": f"文件编码错误: {str(e)}",
                "error_type": "UnicodeDecodeError",
                "suggestion": "请尝试使用其他编码格式（如gbk、latin-1）",
                "agent_id": self._agent_context.get('agent_id'),
                "session_id": self._agent_context.get('session_id')
            }
        except PermissionError:
            return {
                "success": False,
                "error": f"权限不足: {file_path}",
                "error_type": "PermissionError",
                "agent_id": self._agent_context.get('agent_id'),
                "session_id": self._agent_context.get('session_id')
            }
        except Exception as e:
            self._log_error(f"文件读取失败: {str(e)}", {
                "file_path": file_path,
                "encoding": encoding,
                "exception_type": type(e).__name__
            })
            return {
                "success": False,
                "error": f"读取文件时发生错误: {str(e)}",
                "error_type": type(e).__name__,
                "agent_id": self._agent_context.get('agent_id'),
                "session_id": self._agent_context.get('session_id')
            }
    
    def _is_safe_path(self, file_path: str) -> bool:
        """检查文件路径是否安全
        
        Args:
            file_path: 文件路径
            
        Returns:
            路径是否安全
        """
        try:
            # 规范化路径
            normalized_path = Path(file_path).resolve()
            path_str = str(normalized_path)
            
            # 检查路径遍历攻击
            if ".." in file_path or "~" in file_path:
                return False
            
            # 检查被阻止的路径
            for blocked_path in self._blocked_paths:
                if path_str.startswith(blocked_path):
                    return False
            
            # 检查文件扩展名
            if normalized_path.suffix.lower() not in self._allowed_extensions:
                return False
            
            # 检查是否为文件（不是目录）
            if normalized_path.exists() and not normalized_path.is_file():
                return False
            
            return True
            
        except (OSError, ValueError):
            return False


class AgentScopeFileWriteTool(AgentScopeToolBase):
    """基于AgentScope的文件写入工具"""
    
    def __init__(self, agent_context: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="agentscope_file_write",
            description="基于AgentScope的安全文件写入工具",
            agent_context=agent_context
        )
        self._max_content_size = 5 * 1024 * 1024  # 5MB
        self._allowed_extensions = {
            ".txt", ".json", ".yaml", ".yml", ".md", ".csv",
            ".py", ".js", ".html", ".css", ".xml", ".log"
        }
        self._safe_directories = {
            "./output", "./temp", "./data", "./logs"
        }
    
    def get_schema(self) -> Dict[str, Any]:
        """获取工具模式定义"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "要写入的文件路径"
                        },
                        "content": {
                            "type": "string",
                            "description": "要写入的内容"
                        },
                        "encoding": {
                            "type": "string",
                            "description": "文件编码格式",
                            "default": "utf-8",
                            "enum": ["utf-8", "gbk", "ascii"]
                        },
                        "mode": {
                            "type": "string",
                            "description": "写入模式",
                            "default": "w",
                            "enum": ["w", "a", "x"]
                        }
                    },
                    "required": ["file_path", "content"]
                }
            }
        }
    
    def __call__(
        self, 
        file_path: str, 
        content: str,
        encoding: str = "utf-8",
        mode: str = "w"
    ) -> Dict[str, Any]:
        """执行文件写入操作"""
        try:
            # 内容大小检查
            content_size = len(content.encode(encoding))
            if content_size > self._max_content_size:
                return {
                    "success": False,
                    "error": f"内容过大: {content_size} bytes (最大: {self._max_content_size} bytes)",
                    "error_type": "ContentSizeError",
                    "agent_id": self._agent_context.get('agent_id'),
                    "session_id": self._agent_context.get('session_id')
                }
            
            # 路径安全检查
            if not self._is_safe_write_path(file_path):
                return {
                    "success": False,
                    "error": "不安全的文件路径或不允许的写入位置",
                    "error_type": "SecurityError",
                    "agent_id": self._agent_context.get('agent_id'),
                    "session_id": self._agent_context.get('session_id')
                }
            
            # 确保目录存在
            file_path_obj = Path(file_path)
            file_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # 写入文件
            with open(file_path, mode, encoding=encoding) as f:
                f.write(content)
            
            # 记录执行日志
            self._log_execution({
                "action": "file_write",
                "file_path": file_path,
                "content_size": content_size,
                "encoding": encoding,
                "mode": mode
            })
            
            return {
                "success": True,
                "message": "文件写入成功",
                "metadata": {
                    "file_path": str(file_path_obj.absolute()),
                    "content_size": content_size,
                    "encoding": encoding,
                    "mode": mode,
                    "write_time": datetime.utcnow().isoformat()
                },
                "agent_id": self._agent_context.get('agent_id'),
                "session_id": self._agent_context.get('session_id')
            }
            
        except Exception as e:
            self._log_error(f"文件写入失败: {str(e)}", {
                "file_path": file_path,
                "content_size": len(content),
                "encoding": encoding,
                "mode": mode,
                "exception_type": type(e).__name__
            })
            return {
                "success": False,
                "error": f"写入文件时发生错误: {str(e)}",
                "error_type": type(e).__name__,
                "agent_id": self._agent_context.get('agent_id'),
                "session_id": self._agent_context.get('session_id')
            }
    
    def _is_safe_write_path(self, file_path: str) -> bool:
        """检查写入路径是否安全"""
        try:
            normalized_path = Path(file_path).resolve()
            path_str = str(normalized_path)
            
            # 检查路径遍历攻击
            if ".." in file_path or "~" in file_path:
                return False
            
            # 检查文件扩展名
            if normalized_path.suffix.lower() not in self._allowed_extensions:
                return False
            
            # 检查是否在安全目录内
            for safe_dir in self._safe_directories:
                safe_dir_resolved = Path(safe_dir).resolve()
                try:
                    normalized_path.relative_to(safe_dir_resolved)
                    return True
                except ValueError:
                    continue
            
            return False
            
        except (OSError, ValueError):
            return False
```

### 6.2 AgentScope网络请求工具
```python
class AgentScopeHttpRequestTool(AgentScopeToolBase):
    """基于AgentScope的HTTP请求工具"""
    
    def __init__(self, agent_context: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="agentscope_http_request",
            description="基于AgentScope的安全HTTP请求工具",
            agent_context=agent_context
        )
        self._session = None
        self._max_response_size = 10 * 1024 * 1024  # 10MB
        self._default_timeout = 30
        self._allowed_schemes = {"http", "https"}
        self._blocked_domains = {
            "localhost", "127.0.0.1", "0.0.0.0", "::1",
            "169.254.0.0/16", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"
        }
        self._allowed_domains = set()  # 如果设置，则只允许这些域名
        self._request_count = 0
        self._max_requests_per_session = 100
    
    def get_schema(self) -> Dict[str, Any]:
        """获取工具模式定义"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "请求的URL地址",
                            "pattern": "^https?://.*"
                        },
                        "method": {
                            "type": "string",
                            "description": "HTTP请求方法",
                            "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"],
                            "default": "GET"
                        },
                        "headers": {
                            "type": "object",
                            "description": "请求头字典",
                            "additionalProperties": {"type": "string"}
                        },
                        "data": {
                            "type": ["object", "string"],
                            "description": "请求体数据（JSON对象或字符串）"
                        },
                        "params": {
                            "type": "object",
                            "description": "URL查询参数",
                            "additionalProperties": {"type": "string"}
                        },
                        "timeout": {
                            "type": "number",
                            "description": "请求超时时间（秒）",
                            "default": 30,
                            "minimum": 1,
                            "maximum": 300
                        },
                        "follow_redirects": {
                            "type": "boolean",
                            "description": "是否跟随重定向",
                            "default": True
                        }
                    },
                    "required": ["url"]
                }
            }
        }
    
    async def __call__(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Union[Dict[str, Any], str]] = None,
        params: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        follow_redirects: bool = True
    ) -> Dict[str, Any]:
        """异步执行HTTP请求
        
        Args:
            url: 请求URL
            method: HTTP方法
            headers: 请求头
            data: 请求体数据
            params: URL参数
            timeout: 超时时间
            follow_redirects: 是否跟随重定向
            
        Returns:
            包含响应数据的字典
        """
        import aiohttp
        import asyncio
        
        try:
            # 请求频率检查
            if self._request_count >= self._max_requests_per_session:
                return {
                    "success": False,
                    "error": f"超出会话最大请求数限制: {self._max_requests_per_session}",
                    "error_type": "RateLimitError",
                    "agent_id": self._agent_context.get('agent_id'),
                    "session_id": self._agent_context.get('session_id')
                }
            
            # URL安全检查
            if not self._is_safe_url(url):
                return {
                    "success": False,
                    "error": "不安全的URL或被阻止的域名",
                    "error_type": "SecurityError",
                    "agent_id": self._agent_context.get('agent_id'),
                    "session_id": self._agent_context.get('session_id')
                }
            
            # 创建或复用会话
            if not self._session:
                connector = aiohttp.TCPConnector(
                    limit=10,
                    limit_per_host=5,
                    ttl_dns_cache=300,
                    use_dns_cache=True
                )
                self._session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                    headers={
                        "User-Agent": f"AgentScope-HttpTool/1.0 (Agent: {self._agent_context.get('agent_id', 'unknown')})"
                    }
                )
            
            # 准备请求参数
            request_headers = headers or {}
            request_headers.setdefault("Accept", "application/json, text/plain, */*")
            
            # 处理请求体
            json_data = None
            text_data = None
            if data:
                if isinstance(data, dict):
                    json_data = data
                    request_headers.setdefault("Content-Type", "application/json")
                else:
                    text_data = str(data)
                    request_headers.setdefault("Content-Type", "text/plain")
            
            # 发送请求
            start_time = asyncio.get_event_loop().time()
            async with self._session.request(
                method=method.upper(),
                url=url,
                headers=request_headers,
                params=params,
                json=json_data,
                data=text_data,
                allow_redirects=follow_redirects,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                # 检查响应大小
                content_length = response.headers.get('Content-Length')
                if content_length and int(content_length) > self._max_response_size:
                    return {
                        "success": False,
                        "error": f"响应过大: {content_length} bytes (最大: {self._max_response_size} bytes)",
                        "error_type": "ResponseSizeError",
                        "agent_id": self._agent_context.get('agent_id'),
                        "session_id": self._agent_context.get('session_id')
                    }
                
                # 读取响应内容
                try:
                    # 尝试解析为JSON
                    if 'application/json' in response.headers.get('Content-Type', ''):
                        content = await response.json()
                        content_type = 'json'
                    else:
                        content = await response.text()
                        content_type = 'text'
                        
                        # 检查实际响应大小
                        if len(content) > self._max_response_size:
                            return {
                                "success": False,
                                "error": f"响应内容过大: {len(content)} bytes",
                                "error_type": "ResponseSizeError",
                                "agent_id": self._agent_context.get('agent_id'),
                                "session_id": self._agent_context.get('session_id')
                            }
                except aiohttp.ContentTypeError:
                    # 如果JSON解析失败，回退到文本
                    content = await response.text()
                    content_type = 'text'
                
                end_time = asyncio.get_event_loop().time()
                request_duration = end_time - start_time
                
                # 增加请求计数
                self._request_count += 1
                
                # 记录执行日志
                self._log_execution({
                    "action": "http_request",
                    "url": url,
                    "method": method,
                    "status_code": response.status,
                    "response_size": len(str(content)),
                    "duration": request_duration,
                    "content_type": content_type
                })
                
                return {
                    "success": True,
                    "status_code": response.status,
                    "headers": dict(response.headers),
                    "content": content,
                    "metadata": {
                        "url": str(response.url),
                        "method": method.upper(),
                        "content_type": content_type,
                        "response_size": len(str(content)),
                        "duration": request_duration,
                        "redirected": str(response.url) != url,
                        "request_time": datetime.utcnow().isoformat()
                    },
                    "agent_id": self._agent_context.get('agent_id'),
                    "session_id": self._agent_context.get('session_id')
                }
                
        except aiohttp.ClientTimeout:
            return {
                "success": False,
                "error": "请求超时",
                "error_type": "TimeoutError",
                "agent_id": self._agent_context.get('agent_id'),
                "session_id": self._agent_context.get('session_id')
            }
        except aiohttp.ClientError as e:
            return {
                "success": False,
                "error": f"客户端错误: {str(e)}",
                "error_type": "ClientError",
                "agent_id": self._agent_context.get('agent_id'),
                "session_id": self._agent_context.get('session_id')
            }
        except Exception as e:
            self._log_execution({
                "action": "http_request_error",
                "url": url,
                "method": method,
                "error": str(e),
                "error_type": type(e).__name__
            })
            return {
                "success": False,
                "error": f"请求失败: {str(e)}",
                "error_type": type(e).__name__,
                "agent_id": self._agent_context.get('agent_id'),
                "session_id": self._agent_context.get('session_id')
            }
    
    def _is_safe_url(self, url: str) -> bool:
        """检查URL安全性
        
        Args:
            url: 要检查的URL
            
        Returns:
            是否为安全URL
        """
        from urllib.parse import urlparse
        import ipaddress
        
        try:
            parsed = urlparse(url)
            
            # 检查协议
            if parsed.scheme not in self._allowed_schemes:
                return False
            
            # 检查域名
            hostname = parsed.hostname
            if not hostname:
                return False
            
            # 如果设置了允许域名列表，只允许这些域名
            if self._allowed_domains and hostname not in self._allowed_domains:
                return False
            
            # 检查是否为IP地址
            try:
                ip = ipaddress.ip_address(hostname)
                # 阻止私有IP地址
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    return False
            except ValueError:
                # 不是IP地址，检查域名
                if hostname in self._blocked_domains:
                    return False
                
                # 检查是否包含被阻止的域名模式
                for blocked in self._blocked_domains:
                    if blocked in hostname:
                        return False
            
            # 检查端口
            port = parsed.port
            if port and port not in [80, 443, 8080, 8443]:
                # 只允许常见的HTTP端口
                return False
            
            return True
        except Exception:
            return False
    
    def set_allowed_domains(self, domains: Set[str]):
        """设置允许的域名列表
        
        Args:
            domains: 允许访问的域名集合
        """
        self._allowed_domains = domains
    
    def add_blocked_domain(self, domain: str):
        """添加被阻止的域名
        
        Args:
            domain: 要阻止的域名
        """
        self._blocked_domains.add(domain)
    
    async def close(self):
        """关闭HTTP会话"""
        if self._session:
            await self._session.close()
            self._session = None
            self._request_count = 0
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
```

## 7. AgentScope插件系统

### 7.1 AgentScope插件基础框架
```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import asyncio
import importlib
import inspect
from pathlib import Path

class AgentScopePluginBase(ABC):
    """基于AgentScope的插件基础抽象类"""
    
    def __init__(
        self, 
        name: str, 
        version: str, 
        author: str,
        agent_context: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.version = version
        self.author = author
        self.enabled = False
        self.agent_context = agent_context or {}
        self.tools: List[AgentScopeToolBase] = []
        self.dependencies: List[str] = []
        self.required_agentscope_version = ">=0.0.1"
        self.plugin_config: Dict[str, Any] = {}
        self.initialization_time: Optional[datetime] = None
        self.last_used_time: Optional[datetime] = None
        self.usage_count = 0
        self.error_count = 0
        self.status = "uninitialized"  # uninitialized, initializing, ready, error, disabled
        self._lock = asyncio.Lock()
    
    @abstractmethod
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化插件
        
        Args:
            config: 插件配置参数
            
        Returns:
            初始化是否成功
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """清理插件资源"""
        pass
    
    @abstractmethod
    def get_tools(self) -> List[AgentScopeToolBase]:
        """获取插件提供的工具
        
        Returns:
            工具列表
        """
        pass
    
    @abstractmethod
    def validate_dependencies(self) -> bool:
        """验证插件依赖
        
        Returns:
            依赖是否满足
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取插件元数据
        
        Returns:
            插件元数据字典
        """
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "enabled": self.enabled,
            "status": self.status,
            "dependencies": self.dependencies,
            "required_agentscope_version": self.required_agentscope_version,
            "tools_count": len(self.tools),
            "initialization_time": self.initialization_time.isoformat() if self.initialization_time else None,
            "last_used_time": self.last_used_time.isoformat() if self.last_used_time else None,
            "usage_count": self.usage_count,
            "error_count": self.error_count,
            "agent_context": {
                "agent_id": self.agent_context.get('agent_id'),
                "session_id": self.agent_context.get('session_id')
            }
        }
    
    def get_schema(self) -> Dict[str, Any]:
        """获取插件模式定义
        
        Returns:
            插件模式字典
        """
        return {
            "name": self.name,
            "version": self.version,
            "description": self.__doc__ or f"{self.name} plugin",
            "author": self.author,
            "dependencies": self.dependencies,
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "schema": tool.get_schema()
                }
                for tool in self.tools
            ],
            "config_schema": self.get_config_schema()
        }
    
    def get_config_schema(self) -> Dict[str, Any]:
        """获取插件配置模式
        
        Returns:
            配置模式字典
        """
        return {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "description": "是否启用插件",
                    "default": True
                },
                "max_usage_count": {
                    "type": "integer",
                    "description": "最大使用次数限制",
                    "default": 1000,
                    "minimum": 1
                },
                "timeout": {
                    "type": "number",
                    "description": "操作超时时间（秒）",
                    "default": 30,
                    "minimum": 1
                }
            }
        }
    
    async def enable(self) -> bool:
        """启用插件
        
        Returns:
            启用是否成功
        """
        async with self._lock:
            if self.status != "ready":
                return False
            
            self.enabled = True
            self.last_used_time = datetime.utcnow()
            return True
    
    async def disable(self) -> bool:
        """禁用插件
        
        Returns:
            禁用是否成功
        """
        async with self._lock:
            self.enabled = False
            self.status = "disabled"
            return True
    
    async def reload(self) -> bool:
        """重新加载插件
        
        Returns:
            重新加载是否成功
        """
        async with self._lock:
            try:
                # 清理现有资源
                await self.cleanup()
                
                # 重新初始化
                success = await self.initialize(self.plugin_config)
                if success:
                    self.status = "ready"
                    self.initialization_time = datetime.utcnow()
                else:
                    self.status = "error"
                    self.error_count += 1
                
                return success
            except Exception as e:
                self.status = "error"
                self.error_count += 1
                return False
    
    def update_usage_stats(self):
        """更新使用统计"""
        self.usage_count += 1
        self.last_used_time = datetime.utcnow()
    
    def log_error(self, error: Exception):
        """记录错误
        
        Args:
            error: 错误对象
        """
        self.error_count += 1
         # 这里可以集成到AgentScope的日志系统
         print(f"Plugin {self.name} error: {str(error)}")
```

### 7.2 AgentScope插件管理器
```python
class AgentScopePluginManager:
    """基于AgentScope的插件管理器"""
    
    def __init__(
        self, 
        plugin_dir: str, 
        tool_manager: 'AgentScopeToolManager',
        agent_context: Optional[Dict[str, Any]] = None
    ):
        self.plugin_dir = Path(plugin_dir)
        self.tool_manager = tool_manager
        self.agent_context = agent_context or {}
        self.plugins: Dict[str, AgentScopePluginBase] = {}
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
        self.plugin_dependencies: Dict[str, Set[str]] = {}
        self.load_order: List[str] = []
        self.enabled_plugins: Set[str] = set()
        self.failed_plugins: Set[str] = set()
        self._lock = asyncio.Lock()
        self.plugin_stats = {
            "total_loaded": 0,
            "total_enabled": 0,
            "total_failed": 0,
            "last_scan_time": None
        }
    
    async def scan_plugins(self) -> List[str]:
        """扫描插件目录
        
        Returns:
            发现的插件文件路径列表
        """
        plugin_files = []
        
        if not self.plugin_dir.exists():
            self.plugin_dir.mkdir(parents=True, exist_ok=True)
            return plugin_files
        
        # 扫描Python文件
        for file_path in self.plugin_dir.rglob("*.py"):
            if file_path.name.startswith("plugin_") or file_path.name.endswith("_plugin.py"):
                plugin_files.append(str(file_path))
        
        # 扫描插件包目录
        for dir_path in self.plugin_dir.iterdir():
            if dir_path.is_dir() and (dir_path / "__init__.py").exists():
                plugin_files.append(str(dir_path))
        
        self.plugin_stats["last_scan_time"] = datetime.utcnow().isoformat()
        return plugin_files
    
    async def load_plugin(self, plugin_path: str) -> bool:
        """加载单个插件
        
        Args:
            plugin_path: 插件文件或目录路径
            
        Returns:
            加载是否成功
        """
        async with self._lock:
            try:
                # 动态导入插件模块
                plugin_path_obj = Path(plugin_path)
                
                if plugin_path_obj.is_file():
                    # 单文件插件
                    spec = importlib.util.spec_from_file_location(
                        f"plugin_{plugin_path_obj.stem}", 
                        plugin_path
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                else:
                    # 包插件
                    spec = importlib.util.spec_from_file_location(
                        f"plugin_{plugin_path_obj.name}",
                        plugin_path_obj / "__init__.py"
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                
                # 查找插件类
                plugin_class = None
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, AgentScopePluginBase) and 
                        obj != AgentScopePluginBase):
                        plugin_class = obj
                        break
                
                if not plugin_class:
                    raise ValueError(f"未找到有效的插件类: {plugin_path}")
                
                # 创建插件实例
                plugin_instance = plugin_class(agent_context=self.agent_context)
                
                # 验证依赖
                if not plugin_instance.validate_dependencies():
                    raise ValueError(f"插件依赖验证失败: {plugin_instance.name}")
                
                # 加载配置
                config = self.plugin_configs.get(plugin_instance.name, {})
                
                # 初始化插件
                plugin_instance.status = "initializing"
                success = await plugin_instance.initialize(config)
                
                if success:
                    # 注册插件
                    self.plugins[plugin_instance.name] = plugin_instance
                    plugin_instance.status = "ready"
                    plugin_instance.initialization_time = datetime.utcnow()
                    
                    # 注册插件工具
                    for tool in plugin_instance.get_tools():
                        await self.tool_manager.register_tool(tool)
                    
                    # 更新依赖关系
                    self.plugin_dependencies[plugin_instance.name] = set(plugin_instance.dependencies)
                    
                    # 更新统计
                    self.plugin_stats["total_loaded"] += 1
                    
                    return True
                else:
                    plugin_instance.status = "error"
                    self.failed_plugins.add(plugin_instance.name)
                    self.plugin_stats["total_failed"] += 1
                    return False
                    
            except Exception as e:
                self.failed_plugins.add(plugin_path)
                self.plugin_stats["total_failed"] += 1
                print(f"加载插件失败 {plugin_path}: {str(e)}")
                return False
    
    async def load_all_plugins(self) -> Dict[str, bool]:
        """加载所有插件
        
        Returns:
            插件加载结果字典
        """
        plugin_files = await self.scan_plugins()
        results = {}
        
        # 按依赖顺序加载
        loaded_plugins = set()
        remaining_plugins = set(plugin_files)
        
        while remaining_plugins:
            progress_made = False
            
            for plugin_path in list(remaining_plugins):
                # 检查是否可以加载（依赖已满足）
                can_load = True
                # 这里可以添加依赖检查逻辑
                
                if can_load:
                    success = await self.load_plugin(plugin_path)
                    results[plugin_path] = success
                    remaining_plugins.remove(plugin_path)
                    if success:
                        loaded_plugins.add(plugin_path)
                    progress_made = True
            
            # 如果没有进展，说明存在循环依赖或无法满足的依赖
            if not progress_made:
                for plugin_path in remaining_plugins:
                    results[plugin_path] = False
                    self.failed_plugins.add(plugin_path)
                break
        
        return results
            spec.loader.exec_module(module)
            
            # 获取插件类
            plugin_class = getattr(module, "Plugin", None)
            if not plugin_class or not issubclass(plugin_class, PluginBase):
                logger.error(f"Invalid plugin: {plugin_path}")
                return False
            
            # 创建插件实例
            plugin = plugin_class()
            
            # 检查依赖
            if not await self._check_dependencies(plugin):
                logger.error(f"Plugin dependencies not met: {plugin.name}")
                return False
            
            # 初始化插件
            if not await plugin.initialize():
                logger.error(f"Plugin initialization failed: {plugin.name}")
                return False
            
            # 注册插件工具
            tools = plugin.get_tools()
            for tool in tools:
                if not self.tool_manager.registry.register_tool(tool):
                    logger.warning(f"Failed to register tool: {tool.name}")
            
            # 保存插件
            self.plugins[plugin.name] = plugin
            plugin.enabled = True
            
            logger.info(f"Plugin loaded: {plugin.name} v{plugin.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_path}: {e}")
            return False
    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """卸载插件"""
        if plugin_name not in self.plugins:
            return False
        
        plugin = self.plugins[plugin_name]
        
        try:
            # 注销插件工具
            tools = plugin.get_tools()
            for tool in tools:
                self.tool_manager.registry.unregister_tool(tool.name)
            
            # 清理插件资源
            await plugin.cleanup()
            
            # 移除插件
            del self.plugins[plugin_name]
            
            logger.info(f"Plugin unloaded: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload plugin {plugin_name}: {e}")
            return False
    
    async def reload_plugin(self, plugin_name: str) -> bool:
        """重新加载插件"""
        if plugin_name in self.plugins:
            plugin_path = self._get_plugin_path(plugin_name)
            await self.unload_plugin(plugin_name)
            return await self.load_plugin(plugin_path)
        return False
    
    async def load_all_plugins(self) -> None:
        """加载所有插件"""
        plugin_files = self.plugin_dir.glob("*.py")
        
        for plugin_file in plugin_files:
            await self.load_plugin(str(plugin_file))
    
    def list_plugins(self) -> List[Dict]:
        """列出所有插件"""
        return [plugin.get_metadata() for plugin in self.plugins.values()]
    
    def get_plugin(self, plugin_name: str) -> Optional[PluginBase]:
        """获取插件"""
        return self.plugins.get(plugin_name)
    
    async def _check_dependencies(self, plugin: PluginBase) -> bool:
        """检查插件依赖"""
        for dependency in plugin.dependencies:
            if dependency not in self.plugins:
                return False
        return True
    
    def _get_plugin_path(self, plugin_name: str) -> str:
        """获取插件文件路径"""
        return str(self.plugin_dir / f"{plugin_name}.py")
```

## 8. AgentScope工具配置和部署

### 8.1 AgentScope工具配置文件
```yaml
# agentscope_tools_config.yaml
agentscope:
  # AgentScope框架配置
  framework:
    version: "0.1.0"
    logging_level: "INFO"
    session_timeout: 3600
    max_agents: 100

  # 工具系统配置
  tools:
    # 内置工具配置
    builtin:
      agentscope_file_read:
        enabled: true
        max_file_size: 10485760  # 10MB
        allowed_extensions: [".txt", ".json", ".yaml", ".md", ".py"]
        security_check: true
        agent_context: true
      
      agentscope_file_write:
        enabled: true
        max_file_size: 5242880  # 5MB
        backup_enabled: true
        security_check: true
        agent_context: true
      
      agentscope_http_request:
        enabled: true
        timeout: 30
        max_response_size: 5242880  # 5MB
        allowed_domains: ["api.example.com", "*.github.com", "*.agentscope.io"]
        rate_limit: 100  # requests per minute
        session_management: true
      
      agentscope_web_scraping:
        enabled: true
        timeout: 45
        max_page_size: 10485760  # 10MB
        javascript_enabled: false
        user_agent: "AgentScope/1.0"
    
    # 插件工具配置
    plugins:
      agentscope_database_plugin:
        enabled: true
        config:
          connection_string: "postgresql://user:pass@localhost/db"
          pool_size: 5
          query_timeout: 30
          agent_isolation: true
      
      agentscope_ai_plugin:
        enabled: false
        config:
          api_key: "${AI_API_KEY}"
          model: "gpt-3.5-turbo"
          max_tokens: 4096
          temperature: 0.7
          agent_context: true

# AgentScope执行环境配置
execution:
  agentscope_sandbox:
    enabled: true
    isolation_level: "high"
    resource_limits:
      max_memory: 536870912  # 512MB
      max_cpu_time: 30
      max_file_size: 104857600  # 100MB
      max_network_connections: 10
    
    security:
      filesystem_isolation: true
      network_isolation: true
      process_isolation: true
  
  timeout: 60
  max_concurrent: 10
  agent_context_enabled: true
  session_tracking: true

# AgentScope访问控制配置
access_control:
  default_policy: "allow"
  agent_based_control: true
  
  rules:
    - tool: "agentscope_file_read"
      agents: ["file_agent", "admin_agent", "research_agent"]
      action: "allow"
      conditions:
        - "agent.role == 'file_handler'"
        - "session.authenticated == true"
    
    - tool: "agentscope_http_request"
      agents: ["web_agent", "api_agent"]
      action: "allow"
      conditions:
        - "agent.permissions.network == true"
    
    - tool: "*"
      agents: ["restricted_agent"]
      action: "deny"
      exceptions: ["agentscope_file_read"]

# AgentScope监控配置
monitoring:
  enabled: true
  agent_tracking: true
  session_tracking: true
  
  metrics:
    - "execution_count"
    - "execution_time"
    - "success_rate"
    - "error_rate"
    - "agent_usage"
    - "session_duration"
    - "memory_usage"
  
  alerts:
    - condition: "error_rate > 0.1"
      action: "disable_tool"
      notify_agents: true
    
    - condition: "execution_time > 60"
      action: "log_warning"
      escalate_to_admin: true
    
    - condition: "memory_usage > 0.8"
      action: "cleanup_sessions"
      auto_scale: true
```

### 8.2 AgentScope工具部署脚本
```python
# agentscope_deploy_tools.py
import asyncio
import yaml
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# AgentScope框架导入
from agentscope import init
from agentscope.agents import AgentBase
from agentscope.models import ModelWrapperBase
from agentscope.memory import MemoryBase
from agentscope.tool import ToolBase
from agentscope.session import SessionManager
from agentscope.logging import setup_logger

# 自定义组件导入
from mvp_0_2_0.tools.registry import AgentScopeToolRegistry
from mvp_0_2_0.tools.executor import AgentScopeToolExecutor
from mvp_0_2_0.tools.manager import AgentScopeToolManager
from mvp_0_2_0.tools.sandbox import AgentScopeToolSandbox
from mvp_0_2_0.tools.builtin import (
    AgentScopeFileReadTool,
    AgentScopeFileWriteTool,
    AgentScopeHttpRequestTool,
    AgentScopeWebScrapingTool
)
from mvp_0_2_0.plugins.manager import AgentScopePluginManager
from mvp_0_2_0.monitoring.tool_monitor import AgentScopeToolMonitor

logger = setup_logger(__name__)

class AgentScopeToolDeployer:
    """AgentScope工具系统部署器"""
    
    def __init__(self, config_path: str = "agentscope_tools_config.yaml"):
        self.config_path = Path(config_path)
        self.config = None
        self.tool_manager = None
        self.plugin_manager = None
        self.monitor = None
        self.session_manager = None
    
    async def deploy(self) -> AgentScopeToolManager:
        """部署AgentScope工具系统"""
        try:
            # 1. 加载配置
            await self._load_config()
            
            # 2. 初始化AgentScope框架
            await self._init_agentscope()
            
            # 3. 创建核心组件
            await self._create_core_components()
            
            # 4. 注册内置工具
            await self._register_builtin_tools()
            
            # 5. 加载插件
            await self._load_plugins()
            
            # 6. 配置访问控制
            await self._setup_access_control()
            
            # 7. 启动监控
            await self._start_monitoring()
            
            # 8. 验证部署
            await self._validate_deployment()
            
            logger.info("AgentScope tool system deployed successfully")
            return self.tool_manager
            
        except Exception as e:
            logger.error(f"Failed to deploy tool system: {e}")
            await self._cleanup_on_failure()
            raise
    
    async def _load_config(self) -> None:
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {self.config_path}")
    
    async def _init_agentscope(self) -> None:
        """初始化AgentScope框架"""
        framework_config = self.config.get("agentscope", {}).get("framework", {})
        
        # 初始化AgentScope
        init(
            project="RobotAgent-MVP-0.2.0",
            name="tool-system",
            logging_level=framework_config.get("logging_level", "INFO"),
            save_dir="./logs",
            save_log=True,
            save_code=True
        )
        
        # 创建会话管理器
        self.session_manager = SessionManager(
            timeout=framework_config.get("session_timeout", 3600),
            max_sessions=framework_config.get("max_agents", 100)
        )
        
        logger.info("AgentScope framework initialized")
    
    async def _create_core_components(self) -> None:
        """创建核心组件"""
        execution_config = self.config.get("execution", {})
        
        # 创建工具注册表
        registry = AgentScopeToolRegistry()
        
        # 创建沙箱环境
        sandbox_config = execution_config.get("agentscope_sandbox", {})
        sandbox = AgentScopeToolSandbox(
            enabled=sandbox_config.get("enabled", True),
            isolation_level=sandbox_config.get("isolation_level", "high"),
            resource_limits=sandbox_config.get("resource_limits", {}),
            security_config=sandbox_config.get("security", {})
        )
        
        # 创建工具执行器
        executor = AgentScopeToolExecutor(
            sandbox=sandbox,
            timeout=execution_config.get("timeout", 60),
            max_concurrent=execution_config.get("max_concurrent", 10),
            agent_context_enabled=execution_config.get("agent_context_enabled", True),
            session_tracking=execution_config.get("session_tracking", True)
        )
        
        # 创建工具管理器
        self.tool_manager = AgentScopeToolManager(
            registry=registry,
            executor=executor,
            session_manager=self.session_manager
        )
        
        logger.info("Core components created")
    
    async def _register_builtin_tools(self) -> None:
        """注册内置工具"""
        builtin_config = self.config.get("agentscope", {}).get("tools", {}).get("builtin", {})
        
        # 注册文件读取工具
        if builtin_config.get("agentscope_file_read", {}).get("enabled", True):
            file_read_config = builtin_config["agentscope_file_read"]
            file_read_tool = AgentScopeFileReadTool(
                max_file_size=file_read_config.get("max_file_size", 10485760),
                allowed_extensions=file_read_config.get("allowed_extensions", []),
                security_check=file_read_config.get("security_check", True),
                agent_context=file_read_config.get("agent_context", True)
            )
            await self.tool_manager.registry.register_tool(file_read_tool)
        
        # 注册文件写入工具
        if builtin_config.get("agentscope_file_write", {}).get("enabled", True):
            file_write_config = builtin_config["agentscope_file_write"]
            file_write_tool = AgentScopeFileWriteTool(
                max_file_size=file_write_config.get("max_file_size", 5242880),
                backup_enabled=file_write_config.get("backup_enabled", True),
                security_check=file_write_config.get("security_check", True),
                agent_context=file_write_config.get("agent_context", True)
            )
            await self.tool_manager.registry.register_tool(file_write_tool)
        
        # 注册HTTP请求工具
        if builtin_config.get("agentscope_http_request", {}).get("enabled", True):
            http_config = builtin_config["agentscope_http_request"]
            http_tool = AgentScopeHttpRequestTool(
                timeout=http_config.get("timeout", 30),
                max_response_size=http_config.get("max_response_size", 5242880),
                allowed_domains=http_config.get("allowed_domains", []),
                rate_limit=http_config.get("rate_limit", 100),
                session_management=http_config.get("session_management", True)
            )
            await self.tool_manager.registry.register_tool(http_tool)
        
        # 注册网页抓取工具
        if builtin_config.get("agentscope_web_scraping", {}).get("enabled", True):
            scraping_config = builtin_config["agentscope_web_scraping"]
            scraping_tool = AgentScopeWebScrapingTool(
                timeout=scraping_config.get("timeout", 45),
                max_page_size=scraping_config.get("max_page_size", 10485760),
                javascript_enabled=scraping_config.get("javascript_enabled", False),
                user_agent=scraping_config.get("user_agent", "AgentScope/1.0")
            )
            await self.tool_manager.registry.register_tool(scraping_tool)
        
        logger.info("Built-in tools registered")
    
    async def _load_plugins(self) -> None:
        """加载插件"""
        plugin_config = self.config.get("agentscope", {}).get("tools", {}).get("plugins", {})
        
        # 创建插件管理器
        self.plugin_manager = AgentScopePluginManager(
            plugin_dir="./plugins",
            tool_manager=self.tool_manager
        )
        
        # 加载所有插件
        await self.plugin_manager.load_all_plugins()
        
        # 配置插件
        for plugin_name, plugin_config in plugin_config.items():
            if plugin_config.get("enabled", False):
                plugin = self.plugin_manager.get_plugin(plugin_name)
                if plugin:
                    await plugin.configure(plugin_config.get("config", {}))
        
        logger.info("Plugins loaded and configured")
    
    async def _setup_access_control(self) -> None:
        """设置访问控制"""
        access_config = self.config.get("access_control", {})
        
        # 设置默认策略
        default_policy = access_config.get("default_policy", "allow")
        self.tool_manager.set_default_access_policy(default_policy)
        
        # 配置访问规则
        for rule in access_config.get("rules", []):
            tool_name = rule["tool"]
            agents = rule["agents"]
            action = rule["action"]
            conditions = rule.get("conditions", [])
            exceptions = rule.get("exceptions", [])
            
            await self.tool_manager.set_access_rule(
                tool_name=tool_name,
                agents=agents,
                action=action,
                conditions=conditions,
                exceptions=exceptions
            )
        
        logger.info("Access control configured")
    
    async def _start_monitoring(self) -> None:
        """启动监控"""
        monitoring_config = self.config.get("monitoring", {})
        
        if monitoring_config.get("enabled", True):
            self.monitor = AgentScopeToolMonitor(
                tool_manager=self.tool_manager,
                config=monitoring_config
            )
            await self.monitor.start()
            logger.info("Monitoring started")
    
    async def _validate_deployment(self) -> None:
        """验证部署"""
        # 检查工具注册
        tools = await self.tool_manager.registry.list_tools()
        logger.info(f"Registered tools: {len(tools)}")
        
        # 检查插件加载
        if self.plugin_manager:
            plugins = self.plugin_manager.list_plugins()
            logger.info(f"Loaded plugins: {len(plugins)}")
        
        # 检查监控状态
        if self.monitor:
            status = await self.monitor.get_status()
            logger.info(f"Monitoring status: {status}")
        
        logger.info("Deployment validation completed")
    
    async def _cleanup_on_failure(self) -> None:
        """失败时清理资源"""
        try:
            if self.monitor:
                await self.monitor.stop()
            
            if self.plugin_manager:
                await self.plugin_manager.cleanup()
            
            if self.tool_manager:
                await self.tool_manager.cleanup()
            
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

async def deploy_agentscope_tools(config_path: str = "agentscope_tools_config.yaml") -> AgentScopeToolManager:
    """部署AgentScope工具系统的便捷函数"""
    deployer = AgentScopeToolDeployer(config_path)
    return await deployer.deploy()

if __name__ == "__main__":
    asyncio.run(deploy_agentscope_tools())
```

---

**文档版本**: v0.1.0  
**创建日期**: 2025-01-08  
**更新日期**: 2025-01-08  
**负责人**: 开发团队  
**审核状态**: 待审核