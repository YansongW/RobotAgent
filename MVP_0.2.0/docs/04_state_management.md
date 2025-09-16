# -*- coding: utf-8 -*-

# 状态管理和持久化策略 (State Management and Persistence Strategy)
# 基于AgentScope StateModule的状态管理架构设计
# 版本: 0.2.1
# 更新时间: 2025-09-10

# 状态管理和持久化策略

## 概述

本文档基于AgentScope框架的StateModule设计，详细阐述了RobotAgent MVP 0.2.0的状态管理和持久化策略。AgentScope的状态管理系统提供了自动状态注册、嵌套状态序列化、版本兼容性等核心功能，为构建可靠的智能体应用奠定了基础。

## 1. AgentScope状态管理架构

### 1.1 StateModule核心设计

基于AgentScope源码分析，StateModule提供了以下核心功能：

```python
# AgentScope StateModule核心架构
class StateModule:
    """状态管理模块基类"""
    
    def __init__(self) -> None:
        # 状态注册表 - 管理可序列化的属性
        self._state_registry: dict[str, dict] = {}
        # 嵌套状态模块 - 管理StateModule类型的属性
        self._nested_state_modules: list[str] = []
        
        # 自动注册StateModule属性
        self._auto_register_state_modules()
    
    def register_state(
        self,
        attr_name: str,
        custom_to_json: Callable | None = None,
        custom_from_json: Callable | None = None,
    ) -> None:
        """注册属性作为状态
        
        Args:
            attr_name: 属性名称
            custom_to_json: 自定义序列化函数
            custom_from_json: 自定义反序列化函数
        """
        self._state_registry[attr_name] = {
            "to_json": custom_to_json,
            "from_json": custom_from_json,
        }
```

### 1.2 状态管理原则

**基于AgentScope的设计原则**：
- **自动注册**: StateModule属性自动注册为嵌套状态
- **类型安全**: 支持自定义序列化/反序列化函数
- **嵌套支持**: 递归处理嵌套StateModule对象
- **版本兼容**: 支持状态迁移和版本管理
- **异步优先**: 与AgentScope的异步架构保持一致

### 1.3 状态分类体系

```
AgentScope状态体系
├── 核心状态 (Core State)
│   ├── AgentBase状态 (智能体状态)
│   ├── MemoryBase状态 (记忆状态)
│   ├── Toolkit状态 (工具状态)
│   └── SessionManager状态 (会话状态)
├── 扩展状态 (Extended State)
│   ├── 自定义智能体状态
│   ├── 业务逻辑状态
│   └── 用户数据状态
└── 运行时状态 (Runtime State)
    ├── 临时缓存状态
    ├── 性能监控状态
    └── 系统配置状态
```

## 2. 基于AgentScope的状态管理架构

### 2.1 StateModule层次结构

```
┌─────────────────────────────────────────────────────────────┐
│                    应用层 (Application Layer)               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ AgentBase   │  │ MemoryBase  │  │   Toolkit   │        │
│  │  智能体      │  │    记忆      │  │    工具      │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│                 StateModule层 (State Module Layer)         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │State Registry│ │Nested Modules│ │Serialization│        │
│  │  状态注册表   │  │  嵌套模块    │  │  序列化机制   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│                   持久化层 (Persistence Layer)              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ state_dict  │  │load_state_dict│ │ JSON/Pickle │        │
│  │  状态字典    │  │  状态加载     │  │  序列化格式   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 StateModule核心组件

#### 2.2.1 状态注册机制

```python
# 基于AgentScope的状态注册实现
class EnhancedStateModule(StateModule):
    """增强的状态模块 - 基于AgentScope StateModule"""
    
    def __init__(self) -> None:
        super().__init__()
        
        # 状态版本管理
        self._state_version = "1.0.0"
        self._migration_handlers: dict[tuple[str, str], Callable] = {}
        
        # 状态变更监听器
        self._state_listeners: dict[str, list[Callable]] = {}
        
        # 注册版本信息
        self.register_state("_state_version")
    
    def register_state_with_validation(
        self,
        attr_name: str,
        validator: Callable[[Any], bool] | None = None,
        custom_to_json: Callable | None = None,
        custom_from_json: Callable | None = None,
    ) -> None:
        """注册状态并添加验证器
        
        Args:
            attr_name: 属性名称
            validator: 状态值验证函数
            custom_to_json: 自定义序列化函数
            custom_from_json: 自定义反序列化函数
        """
        # 包装序列化函数以添加验证
        def validated_to_json(value):
            if validator and not validator(value):
                raise ValueError(f"Invalid state value for {attr_name}: {value}")
            return custom_to_json(value) if custom_to_json else value
        
        def validated_from_json(value):
            result = custom_from_json(value) if custom_from_json else value
            if validator and not validator(result):
                raise ValueError(f"Invalid deserialized value for {attr_name}: {result}")
            return result
        
        self.register_state(
            attr_name,
            custom_to_json=validated_to_json,
            custom_from_json=validated_from_json
        )
    
    def add_state_listener(self, attr_name: str, listener: Callable) -> None:
        """添加状态变更监听器"""
        if attr_name not in self._state_listeners:
            self._state_listeners[attr_name] = []
        self._state_listeners[attr_name].append(listener)
    
    def _notify_state_change(self, attr_name: str, old_value: Any, new_value: Any) -> None:
        """通知状态变更"""
        if attr_name in self._state_listeners:
            for listener in self._state_listeners[attr_name]:
                try:
                    listener(attr_name, old_value, new_value)
                except Exception as e:
                    logger.warning(f"State listener error for {attr_name}: {e}")
```

#### 2.2.2 状态序列化机制

```python
# AgentScope状态序列化的增强实现
class StateSerializationManager:
    """状态序列化管理器"""
    
    def __init__(self):
        self._custom_serializers: dict[type, Callable] = {}
        self._custom_deserializers: dict[str, Callable] = {}
    
    def register_serializer(self, obj_type: type, serializer: Callable, deserializer: Callable) -> None:
        """注册自定义序列化器
        
        Args:
            obj_type: 对象类型
            serializer: 序列化函数
            deserializer: 反序列化函数
        """
        self._custom_serializers[obj_type] = serializer
        self._custom_deserializers[obj_type.__name__] = deserializer
    
    def serialize_state(self, state_module: StateModule) -> dict:
        """序列化StateModule对象
        
        Args:
            state_module: StateModule实例
            
        Returns:
            序列化后的状态字典
        """
        state_dict = state_module.state_dict()
        
        # 添加类型信息
        state_dict["__class__"] = state_module.__class__.__name__
        state_dict["__module__"] = state_module.__class__.__module__
        
        # 递归处理嵌套对象
        return self._deep_serialize(state_dict)
    
    def deserialize_state(self, state_dict: dict, target_class: type = None) -> StateModule:
        """反序列化状态字典
        
        Args:
            state_dict: 状态字典
            target_class: 目标类型
            
        Returns:
            StateModule实例
        """
        # 获取类信息
        class_name = state_dict.pop("__class__", None)
        module_name = state_dict.pop("__module__", None)
        
        if target_class is None and class_name and module_name:
            # 动态导入类
            module = importlib.import_module(module_name)
            target_class = getattr(module, class_name)
        
        if target_class is None:
            raise ValueError("Cannot determine target class for deserialization")
        
        # 创建实例
        instance = target_class()
        
        # 反序列化状态
        deserialized_dict = self._deep_deserialize(state_dict)
        instance.load_state_dict(deserialized_dict)
        
        return instance
    
    def _deep_serialize(self, obj: Any) -> Any:
        """深度序列化对象"""
        if isinstance(obj, dict):
            return {k: self._deep_serialize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._deep_serialize(item) for item in obj]
        elif type(obj) in self._custom_serializers:
            return {
                "__serialized_type__": type(obj).__name__,
                "__serialized_data__": self._custom_serializers[type(obj)](obj)
            }
        else:
            return obj
    
    def _deep_deserialize(self, obj: Any) -> Any:
        """深度反序列化对象"""
        if isinstance(obj, dict):
            if "__serialized_type__" in obj and "__serialized_data__" in obj:
                type_name = obj["__serialized_type__"]
                if type_name in self._custom_deserializers:
                    return self._custom_deserializers[type_name](obj["__serialized_data__"])
            return {k: self._deep_deserialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_deserialize(item) for item in obj]
        else:
            return obj
```

#### 2.2.3 状态元数据管理

```python
# 基于AgentScope的状态元数据设计
@dataclass
class StateMetadata:
    """状态元数据 - 扩展AgentScope的状态信息"""
    
    # 基础信息
    created_at: datetime
    updated_at: datetime
    version: str
    
    # 状态特征
    size_bytes: int | None = None
    checksum: str | None = None
    
    # 分类标签
    tags: list[str] = field(default_factory=list)
    category: str = "general"
    
    # 生命周期管理
    ttl_seconds: int | None = None
    access_count: int = 0
    last_access: datetime | None = None
    
    # AgentScope特定信息
    state_module_class: str | None = None
    nested_modules: list[str] = field(default_factory=list)
    registered_attributes: list[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        """检查状态是否过期"""
        if self.ttl_seconds is None:
            return False
        
        elapsed = (datetime.utcnow() - self.created_at).total_seconds()
        return elapsed > self.ttl_seconds
    
    def update_access(self) -> None:
        """更新访问信息"""
        self.access_count += 1
        self.last_access = datetime.utcnow()
    
    def calculate_checksum(self, data: Any) -> str:
        """计算状态数据的校验和"""
        import hashlib
        import json
        
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()
```

#### 2.2.4 状态变更追踪

```python
# 基于AgentScope的状态变更追踪实现
class StateChangeTracker:
    """状态变更追踪器 - 监控StateModule的状态变化"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.change_history: list[StateChange] = []
        self.state_snapshots: dict[str, dict] = {}
        self.tracked_modules: dict[str, StateModule] = {}
    
    def register_module(self, module_id: str, state_module: StateModule) -> None:
        """注册要追踪的StateModule
        
        Args:
            module_id: 模块标识符
            state_module: StateModule实例
        """
        self.tracked_modules[module_id] = state_module
        
        # 创建初始快照
        self.create_snapshot(module_id)
        
        # 如果模块支持状态监听，添加监听器
        if hasattr(state_module, 'add_state_listener'):
            state_module.add_state_listener(
                "*",  # 监听所有属性
                lambda attr, old, new: self._on_state_change(module_id, attr, old, new)
            )
    
    def create_snapshot(self, module_id: str, snapshot_name: str = None) -> str:
        """创建状态快照
        
        Args:
            module_id: 模块标识符
            snapshot_name: 快照名称，默认使用时间戳
            
        Returns:
            快照标识符
        """
        if module_id not in self.tracked_modules:
            raise ValueError(f"Module {module_id} not registered")
        
        if snapshot_name is None:
            snapshot_name = f"{module_id}_{datetime.utcnow().isoformat()}"
        
        state_module = self.tracked_modules[module_id]
        self.state_snapshots[snapshot_name] = state_module.state_dict()
        
        return snapshot_name
    
    def restore_snapshot(self, module_id: str, snapshot_name: str) -> bool:
        """恢复到指定快照
        
        Args:
            module_id: 模块标识符
            snapshot_name: 快照名称
            
        Returns:
            是否成功恢复
        """
        if module_id not in self.tracked_modules:
            return False
        
        if snapshot_name not in self.state_snapshots:
            return False
        
        try:
            state_module = self.tracked_modules[module_id]
            snapshot_data = self.state_snapshots[snapshot_name]
            state_module.load_state_dict(snapshot_data)
            
            # 记录恢复操作
            self._record_change(
                module_id,
                "__snapshot_restore__",
                None,
                snapshot_name,
                "snapshot_restore"
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to restore snapshot {snapshot_name}: {e}")
            return False
    
    def get_change_history(
        self, 
        module_id: str = None, 
        attribute: str = None, 
        limit: int = None
    ) -> list[StateChange]:
        """获取状态变更历史
        
        Args:
            module_id: 模块标识符过滤
            attribute: 属性名过滤
            limit: 返回记录数限制
            
        Returns:
            状态变更记录列表
        """
        filtered_history = self.change_history
        
        if module_id:
            filtered_history = [c for c in filtered_history if c.module_id == module_id]
        
        if attribute:
            filtered_history = [c for c in filtered_history if c.attribute == attribute]
        
        if limit:
            filtered_history = filtered_history[-limit:]
        
        return filtered_history
    
    def _on_state_change(self, module_id: str, attribute: str, old_value: Any, new_value: Any) -> None:
        """状态变更回调"""
        self._record_change(module_id, attribute, old_value, new_value, "attribute_change")
    
    def _record_change(
        self, 
        module_id: str, 
        attribute: str, 
        old_value: Any, 
        new_value: Any, 
        change_type: str
    ) -> None:
        """记录状态变更"""
        change = StateChange(
            module_id=module_id,
            attribute=attribute,
            old_value=old_value,
            new_value=new_value,
            timestamp=datetime.utcnow(),
            change_type=change_type
        )
        
        self.change_history.append(change)
        
        # 限制历史记录长度
        if len(self.change_history) > self.max_history:
            self.change_history.pop(0)

@dataclass
class StateChange:
    """状态变更记录"""
    module_id: str
    attribute: str
    old_value: Any
    new_value: Any
    timestamp: datetime
    change_type: str  # attribute_change, snapshot_restore, etc.
```

## 3. 基于AgentScope的智能体状态管理

### 3.1 智能体状态模块设计

```python
# 基于AgentScope StateModule的智能体状态实现
class AgentStateModule(EnhancedStateModule):
    """智能体状态模块 - 继承自AgentScope StateModule"""
    
    def __init__(self, agent_id: str):
        super().__init__()
        
        # 基础状态属性
        self.agent_id = agent_id
        self.status = AgentStatus.IDLE
        self.current_task_id: str | None = None
        self.conversation_context: list[dict] = []
        self.performance_metrics = AgentMetrics()
        
        # 注册状态属性
        self.register_state("agent_id")
        self.register_state("status")
        self.register_state("current_task_id")
        self.register_state_with_validation(
            "conversation_context",
            validator=lambda x: isinstance(x, list),
            custom_to_json=self._serialize_conversation,
            custom_from_json=self._deserialize_conversation
        )
        self.register_state(
            "performance_metrics",
            custom_to_json=lambda x: x.__dict__,
            custom_from_json=lambda x: AgentMetrics(**x)
        )
        
        # 添加状态变更监听器
        self.add_state_listener("status", self._on_status_change)
        self.add_state_listener("current_task_id", self._on_task_change)
    
    def update_status(self, new_status: AgentStatus, reason: str = None) -> None:
        """更新智能体状态
        
        Args:
            new_status: 新状态
            reason: 状态变更原因
        """
        old_status = self.status
        self.status = new_status
        
        # 更新性能指标
        self.performance_metrics.last_activity = datetime.utcnow()
        
        logger.info(
            f"Agent {self.agent_id} status changed: {old_status} -> {new_status}"
            f"{f' (reason: {reason})' if reason else ''}"
        )
    
    def start_task(self, task_id: str) -> None:
        """开始执行任务
        
        Args:
            task_id: 任务标识符
        """
        self.current_task_id = task_id
        self.update_status(AgentStatus.BUSY, f"Starting task {task_id}")
    
    def complete_task(self, success: bool = True) -> None:
        """完成任务
        
        Args:
            success: 任务是否成功完成
        """
        if success:
            self.performance_metrics.successful_tasks += 1
        else:
            self.performance_metrics.failed_tasks += 1
        
        self.current_task_id = None
        self.update_status(AgentStatus.IDLE, "Task completed")
    
    def add_conversation_message(self, message: dict) -> None:
        """添加对话消息
        
        Args:
            message: 消息字典
        """
        self.conversation_context.append({
            **message,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        self.performance_metrics.total_messages += 1
        
        # 限制对话上下文长度
        max_context_length = 1000
        if len(self.conversation_context) > max_context_length:
            self.conversation_context = self.conversation_context[-max_context_length:]
    
    def get_conversation_summary(self, last_n: int = 10) -> list[dict]:
        """获取对话摘要
        
        Args:
            last_n: 返回最近N条消息
            
        Returns:
            对话消息列表
        """
        return self.conversation_context[-last_n:] if last_n > 0 else self.conversation_context
    
    def _serialize_conversation(self, conversation: list[dict]) -> list[dict]:
        """序列化对话上下文"""
        # 可以在这里添加压缩或过滤逻辑
        return conversation
    
    def _deserialize_conversation(self, data: list[dict]) -> list[dict]:
        """反序列化对话上下文"""
        return data
    
    def _on_status_change(self, attr_name: str, old_value: Any, new_value: Any) -> None:
        """状态变更回调"""
        # 可以在这里添加状态变更的副作用处理
        pass
    
    def _on_task_change(self, attr_name: str, old_value: Any, new_value: Any) -> None:
        """任务变更回调"""
        if new_value is not None:
            logger.info(f"Agent {self.agent_id} started task: {new_value}")
        elif old_value is not None:
            logger.info(f"Agent {self.agent_id} finished task: {old_value}")

class AgentStatus(Enum):
    """智能体状态枚举"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"
    INITIALIZING = "initializing"

@dataclass
class AgentMetrics:
    """智能体性能指标"""
    total_messages: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_response_time: float = 0.0
    memory_usage_mb: float = 0.0
    last_activity: datetime | None = None
    
    def calculate_success_rate(self) -> float:
        """计算任务成功率"""
        total_tasks = self.successful_tasks + self.failed_tasks
        return self.successful_tasks / total_tasks if total_tasks > 0 else 0.0
    
    def update_response_time(self, response_time: float) -> None:
        """更新平均响应时间"""
        if self.average_response_time == 0.0:
            self.average_response_time = response_time
        else:
            # 使用指数移动平均
            alpha = 0.1
            self.average_response_time = (
                alpha * response_time + (1 - alpha) * self.average_response_time
            )
```

### 3.2 智能体状态管理器

```python
# 基于AgentScope的智能体状态管理器
class AgentStateManager:
    """智能体状态管理器 - 管理多个智能体的状态"""
    
    def __init__(self, serialization_manager: StateSerializationManager = None):
        self.agent_states: dict[str, AgentStateModule] = {}
        self.state_tracker = StateChangeTracker()
        self.serialization_manager = serialization_manager or StateSerializationManager()
        
        # 注册AgentStateModule的序列化器
        self.serialization_manager.register_serializer(
            AgentStateModule,
            lambda x: x.state_dict(),
            lambda x: self._deserialize_agent_state(x)
        )
    
    def create_agent_state(self, agent_id: str) -> AgentStateModule:
        """创建智能体状态模块
        
        Args:
            agent_id: 智能体标识符
            
        Returns:
            智能体状态模块实例
        """
        if agent_id in self.agent_states:
            raise ValueError(f"Agent state for {agent_id} already exists")
        
        agent_state = AgentStateModule(agent_id)
        self.agent_states[agent_id] = agent_state
        
        # 注册到状态追踪器
        self.state_tracker.register_module(agent_id, agent_state)
        
        logger.info(f"Created state module for agent: {agent_id}")
        return agent_state
    
    def get_agent_state(self, agent_id: str) -> AgentStateModule | None:
        """获取智能体状态模块
        
        Args:
            agent_id: 智能体标识符
            
        Returns:
            智能体状态模块实例或None
        """
        return self.agent_states.get(agent_id)
    
    def remove_agent_state(self, agent_id: str) -> bool:
        """移除智能体状态
        
        Args:
            agent_id: 智能体标识符
            
        Returns:
            是否成功移除
        """
        if agent_id in self.agent_states:
            del self.agent_states[agent_id]
            logger.info(f"Removed state module for agent: {agent_id}")
            return True
        return False
    
    def get_all_agent_ids(self) -> list[str]:
        """获取所有智能体ID列表
        
        Returns:
            智能体ID列表
        """
        return list(self.agent_states.keys())
    
    def get_agents_by_status(self, status: AgentStatus) -> list[str]:
        """根据状态获取智能体列表
        
        Args:
            status: 智能体状态
            
        Returns:
            符合条件的智能体ID列表
        """
        return [
            agent_id for agent_id, state_module in self.agent_states.items()
            if state_module.status == status
        ]
    
    def save_all_states(self, file_path: str) -> None:
        """保存所有智能体状态到文件
        
        Args:
            file_path: 保存文件路径
        """
        states_data = {}
        
        for agent_id, state_module in self.agent_states.items():
            states_data[agent_id] = self.serialization_manager.serialize_state(state_module)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(states_data, f, indent=2, default=str)
        
        logger.info(f"Saved {len(states_data)} agent states to {file_path}")
    
    def load_all_states(self, file_path: str) -> None:
        """从文件加载所有智能体状态
        
        Args:
            file_path: 状态文件路径
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                states_data = json.load(f)
            
            for agent_id, state_dict in states_data.items():
                try:
                    state_module = self.serialization_manager.deserialize_state(
                        state_dict, AgentStateModule
                    )
                    self.agent_states[agent_id] = state_module
                    
                    # 重新注册到状态追踪器
                    self.state_tracker.register_module(agent_id, state_module)
                    
                except Exception as e:
                    logger.error(f"Failed to load state for agent {agent_id}: {e}")
            
            logger.info(f"Loaded {len(self.agent_states)} agent states from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load states from {file_path}: {e}")
    
    def create_global_snapshot(self, snapshot_name: str = None) -> str:
        """创建全局状态快照
        
        Args:
            snapshot_name: 快照名称
            
        Returns:
            快照标识符
        """
        if snapshot_name is None:
            snapshot_name = f"global_{datetime.utcnow().isoformat()}"
        
        for agent_id in self.agent_states.keys():
            self.state_tracker.create_snapshot(agent_id, f"{snapshot_name}_{agent_id}")
        
        logger.info(f"Created global snapshot: {snapshot_name}")
        return snapshot_name
    
    def get_system_metrics(self) -> dict:
        """获取系统级性能指标
        
        Returns:
            系统指标字典
        """
        total_agents = len(self.agent_states)
        active_agents = len(self.get_agents_by_status(AgentStatus.BUSY))
        idle_agents = len(self.get_agents_by_status(AgentStatus.IDLE))
        error_agents = len(self.get_agents_by_status(AgentStatus.ERROR))
        
        total_messages = sum(
            state.performance_metrics.total_messages 
            for state in self.agent_states.values()
        )
        
        total_successful_tasks = sum(
            state.performance_metrics.successful_tasks 
            for state in self.agent_states.values()
        )
        
        total_failed_tasks = sum(
            state.performance_metrics.failed_tasks 
            for state in self.agent_states.values()
        )
        
        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "idle_agents": idle_agents,
            "error_agents": error_agents,
            "total_messages": total_messages,
            "total_successful_tasks": total_successful_tasks,
            "total_failed_tasks": total_failed_tasks,
            "overall_success_rate": (
                total_successful_tasks / (total_successful_tasks + total_failed_tasks)
                if (total_successful_tasks + total_failed_tasks) > 0 else 0.0
            )
        }
    
    def _deserialize_agent_state(self, state_dict: dict) -> AgentStateModule:
        """反序列化智能体状态"""
        agent_id = state_dict.get("agent_id", "unknown")
        agent_state = AgentStateModule(agent_id)
        agent_state.load_state_dict(state_dict)
        return agent_state
```

## 4. 基于AgentScope的会话状态管理

### 4.1 会话状态模块设计

```python
# 基于AgentScope StateModule的会话状态实现
class SessionStateModule(EnhancedStateModule):
    """会话状态模块 - 继承自AgentScope StateModule"""
    
    def __init__(self, session_id: str):
        super().__init__()
        
        # 基础会话属性
        self.session_id = session_id
        self.participants: list[str] = []
        self.messages: list[dict] = []
        self.context: dict[str, Any] = {}
        self.status = SessionStatus.ACTIVE
        self.created_at = datetime.utcnow()
        self.expires_at: datetime | None = None
        
        # 会话统计信息
        self.message_count = 0
        self.participant_activity: dict[str, datetime] = {}
        
        # 注册状态属性
        self.register_state("session_id")
        self.register_state_with_validation(
            "participants",
            validator=lambda x: isinstance(x, list) and all(isinstance(p, str) for p in x)
        )
        self.register_state_with_validation(
            "messages",
            validator=lambda x: isinstance(x, list),
            custom_to_json=self._serialize_messages,
            custom_from_json=self._deserialize_messages
        )
        self.register_state("context")
        self.register_state("status")
        self.register_state(
            "created_at",
            custom_to_json=lambda x: x.isoformat(),
            custom_from_json=lambda x: datetime.fromisoformat(x)
        )
        self.register_state(
            "expires_at",
            custom_to_json=lambda x: x.isoformat() if x else None,
            custom_from_json=lambda x: datetime.fromisoformat(x) if x else None
        )
        self.register_state("message_count")
        self.register_state(
            "participant_activity",
            custom_to_json=lambda x: {k: v.isoformat() for k, v in x.items()},
            custom_from_json=lambda x: {k: datetime.fromisoformat(v) for k, v in x.items()}
        )
        
        # 添加状态变更监听器
        self.add_state_listener("status", self._on_status_change)
        self.add_state_listener("participants", self._on_participants_change)
    
    def add_participant(self, participant_id: str) -> None:
        """添加参与者
        
        Args:
            participant_id: 参与者标识符
        """
        if participant_id not in self.participants:
            self.participants.append(participant_id)
            self.participant_activity[participant_id] = datetime.utcnow()
            logger.info(f"Added participant {participant_id} to session {self.session_id}")
    
    def remove_participant(self, participant_id: str) -> bool:
        """移除参与者
        
        Args:
            participant_id: 参与者标识符
            
        Returns:
            是否成功移除
        """
        if participant_id in self.participants:
            self.participants.remove(participant_id)
            if participant_id in self.participant_activity:
                del self.participant_activity[participant_id]
            logger.info(f"Removed participant {participant_id} from session {self.session_id}")
            return True
        return False
    
    def add_message(self, message: dict, sender_id: str = None) -> None:
        """添加消息到会话
        
        Args:
            message: 消息字典
            sender_id: 发送者ID
        """
        # 添加时间戳和消息ID
        enhanced_message = {
            "id": f"msg_{self.message_count + 1}",
            "timestamp": datetime.utcnow().isoformat(),
            "sender_id": sender_id,
            **message
        }
        
        self.messages.append(enhanced_message)
        self.message_count += 1
        
        # 更新发送者活动时间
        if sender_id and sender_id in self.participants:
            self.participant_activity[sender_id] = datetime.utcnow()
        
        # 限制消息历史长度
        max_messages = 10000
        if len(self.messages) > max_messages:
            # 保留最近的消息
            self.messages = self.messages[-max_messages:]
    
    def get_messages(self, limit: int = None, sender_id: str = None) -> list[dict]:
        """获取消息列表
        
        Args:
            limit: 返回消息数量限制
            sender_id: 按发送者过滤
            
        Returns:
            消息列表
        """
        filtered_messages = self.messages
        
        if sender_id:
            filtered_messages = [
                msg for msg in filtered_messages 
                if msg.get("sender_id") == sender_id
            ]
        
        if limit:
            filtered_messages = filtered_messages[-limit:]
        
        return filtered_messages
    
    def update_context(self, key: str, value: Any) -> None:
        """更新会话上下文
        
        Args:
            key: 上下文键
            value: 上下文值
        """
        self.context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """获取会话上下文
        
        Args:
            key: 上下文键
            default: 默认值
            
        Returns:
            上下文值
        """
        return self.context.get(key, default)
    
    def set_expiration(self, expires_in_seconds: int) -> None:
        """设置会话过期时间
        
        Args:
            expires_in_seconds: 过期时间（秒）
        """
        self.expires_at = datetime.utcnow() + timedelta(seconds=expires_in_seconds)
    
    def is_expired(self) -> bool:
        """检查会话是否过期
        
        Returns:
            是否过期
        """
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def pause_session(self, reason: str = None) -> None:
        """暂停会话
        
        Args:
            reason: 暂停原因
        """
        self.status = SessionStatus.PAUSED
        if reason:
            self.update_context("pause_reason", reason)
        logger.info(f"Session {self.session_id} paused{f': {reason}' if reason else ''}")
    
    def resume_session(self) -> None:
        """恢复会话"""
        self.status = SessionStatus.ACTIVE
        self.update_context("pause_reason", None)
        logger.info(f"Session {self.session_id} resumed")
    
    def end_session(self, reason: str = None) -> None:
        """结束会话
        
        Args:
            reason: 结束原因
        """
        self.status = SessionStatus.ENDED
        if reason:
            self.update_context("end_reason", reason)
        logger.info(f"Session {self.session_id} ended{f': {reason}' if reason else ''}")
    
    def get_session_summary(self) -> dict:
        """获取会话摘要
        
        Returns:
            会话摘要字典
        """
        duration = datetime.utcnow() - self.created_at
        
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "participant_count": len(self.participants),
            "message_count": self.message_count,
            "duration_seconds": duration.total_seconds(),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_expired": self.is_expired()
        }
    
    def _serialize_messages(self, messages: list[dict]) -> list[dict]:
        """序列化消息列表"""
        # 可以在这里添加消息压缩或过滤逻辑
        return messages
    
    def _deserialize_messages(self, data: list[dict]) -> list[dict]:
        """反序列化消息列表"""
        return data
    
    def _on_status_change(self, attr_name: str, old_value: Any, new_value: Any) -> None:
        """状态变更回调"""
        logger.info(f"Session {self.session_id} status changed: {old_value} -> {new_value}")
    
    def _on_participants_change(self, attr_name: str, old_value: Any, new_value: Any) -> None:
        """参与者变更回调"""
        old_set = set(old_value) if old_value else set()
        new_set = set(new_value) if new_value else set()
        
        added = new_set - old_set
        removed = old_set - new_set
        
        if added:
            logger.info(f"Session {self.session_id} added participants: {added}")
        if removed:
            logger.info(f"Session {self.session_id} removed participants: {removed}")

class SessionStatus(Enum):
    """会话状态枚举"""
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"
    EXPIRED = "expired"
```

### 4.2 会话状态管理器

```python
# 基于AgentScope的会话状态管理器
class SessionStateManager:
    """会话状态管理器 - 管理多个会话的状态"""
    
    def __init__(self, serialization_manager: StateSerializationManager = None):
        self.session_states: dict[str, SessionStateModule] = {}
        self.state_tracker = StateChangeTracker()
        self.serialization_manager = serialization_manager or StateSerializationManager()
        
        # 注册SessionStateModule的序列化器
        self.serialization_manager.register_serializer(
            SessionStateModule,
            lambda x: x.state_dict(),
            lambda x: self._deserialize_session_state(x)
        )
        
        # 会话清理任务
        self._cleanup_task: asyncio.Task | None = None
        self._cleanup_interval = 3600  # 1小时
    
    def create_session(
        self, 
        session_id: str, 
        participants: list[str] = None,
        expires_in_seconds: int = None
    ) -> SessionStateModule:
        """创建新会话
        
        Args:
            session_id: 会话标识符
            participants: 初始参与者列表
            expires_in_seconds: 过期时间（秒）
            
        Returns:
            会话状态模块实例
        """
        if session_id in self.session_states:
            raise ValueError(f"Session {session_id} already exists")
        
        session_state = SessionStateModule(session_id)
        
        # 添加初始参与者
        if participants:
            for participant_id in participants:
                session_state.add_participant(participant_id)
        
        # 设置过期时间
        if expires_in_seconds:
            session_state.set_expiration(expires_in_seconds)
        
        self.session_states[session_id] = session_state
        
        # 注册到状态追踪器
        self.state_tracker.register_module(session_id, session_state)
        
        logger.info(f"Created session: {session_id}")
        return session_state
    
    def get_session(self, session_id: str) -> SessionStateModule | None:
        """获取会话状态模块
        
        Args:
            session_id: 会话标识符
            
        Returns:
            会话状态模块实例或None
        """
        return self.session_states.get(session_id)
    
    def end_session(self, session_id: str, reason: str = None) -> bool:
        """结束会话
        
        Args:
            session_id: 会话标识符
            reason: 结束原因
            
        Returns:
            是否成功结束
        """
        session = self.get_session(session_id)
        if session:
            session.end_session(reason)
            return True
        return False
    
    def remove_session(self, session_id: str) -> bool:
        """移除会话
        
        Args:
            session_id: 会话标识符
            
        Returns:
            是否成功移除
        """
        if session_id in self.session_states:
            del self.session_states[session_id]
            logger.info(f"Removed session: {session_id}")
            return True
        return False
    
    def get_all_session_ids(self) -> list[str]:
        """获取所有会话ID列表
        
        Returns:
            会话ID列表
        """
        return list(self.session_states.keys())
    
    def get_sessions_by_status(self, status: SessionStatus) -> list[str]:
        """根据状态获取会话列表
        
        Args:
            status: 会话状态
            
        Returns:
            符合条件的会话ID列表
        """
        return [
            session_id for session_id, session_state in self.session_states.items()
            if session_state.status == status
        ]
    
    def get_sessions_by_participant(self, participant_id: str) -> list[str]:
        """根据参与者获取会话列表
        
        Args:
            participant_id: 参与者标识符
            
        Returns:
            包含该参与者的会话ID列表
        """
        return [
            session_id for session_id, session_state in self.session_states.items()
            if participant_id in session_state.participants
        ]
    
    def cleanup_expired_sessions(self) -> int:
        """清理过期会话
        
        Returns:
            清理的会话数量
        """
        expired_sessions = []
        
        for session_id, session_state in self.session_states.items():
            if session_state.is_expired():
                session_state.status = SessionStatus.EXPIRED
                expired_sessions.append(session_id)
        
        # 移除过期会话
        for session_id in expired_sessions:
            self.remove_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)
    
    def save_all_sessions(self, file_path: str) -> None:
        """保存所有会话状态到文件
        
        Args:
            file_path: 保存文件路径
        """
        sessions_data = {}
        
        for session_id, session_state in self.session_states.items():
            sessions_data[session_id] = self.serialization_manager.serialize_state(session_state)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(sessions_data, f, indent=2, default=str)
        
        logger.info(f"Saved {len(sessions_data)} session states to {file_path}")
    
    def load_all_sessions(self, file_path: str) -> None:
        """从文件加载所有会话状态
        
        Args:
            file_path: 状态文件路径
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sessions_data = json.load(f)
            
            for session_id, state_dict in sessions_data.items():
                try:
                    session_state = self.serialization_manager.deserialize_state(
                        state_dict, SessionStateModule
                    )
                    self.session_states[session_id] = session_state
                    
                    # 重新注册到状态追踪器
                    self.state_tracker.register_module(session_id, session_state)
                    
                except Exception as e:
                    logger.error(f"Failed to load session {session_id}: {e}")
            
            logger.info(f"Loaded {len(self.session_states)} session states from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load sessions from {file_path}: {e}")
    
    def get_system_metrics(self) -> dict:
        """获取会话系统指标
        
        Returns:
            系统指标字典
        """
        total_sessions = len(self.session_states)
        active_sessions = len(self.get_sessions_by_status(SessionStatus.ACTIVE))
        paused_sessions = len(self.get_sessions_by_status(SessionStatus.PAUSED))
        ended_sessions = len(self.get_sessions_by_status(SessionStatus.ENDED))
        
        total_messages = sum(
            session.message_count for session in self.session_states.values()
        )
        
        total_participants = len(set(
            participant_id
            for session in self.session_states.values()
            for participant_id in session.participants
        ))
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "paused_sessions": paused_sessions,
            "ended_sessions": ended_sessions,
            "total_messages": total_messages,
            "total_participants": total_participants,
            "average_messages_per_session": (
                total_messages / total_sessions if total_sessions > 0 else 0
            )
        }
    
    def start_cleanup_task(self) -> None:
        """启动自动清理任务"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            logger.info("Started session cleanup task")
    
    def stop_cleanup_task(self) -> None:
        """停止自动清理任务"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            logger.info("Stopped session cleanup task")
    
    async def _periodic_cleanup(self) -> None:
        """定期清理任务"""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                self.cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup task: {e}")
    
    def _deserialize_session_state(self, state_dict: dict) -> SessionStateModule:
        """反序列化会话状态"""
        session_id = state_dict.get("session_id", "unknown")
        session_state = SessionStateModule(session_id)
        session_state.load_state_dict(state_dict)
        return session_state
```

## 5. 基于AgentScope的持久化策略

### 5.1 状态序列化管理器

```python
# 基于AgentScope的状态序列化管理器
class StateSerializationManager:
    """状态序列化管理器 - 处理状态的序列化和反序列化"""
    
    def __init__(self):
        self.serializers: dict[type, tuple[callable, callable]] = {}
        self.custom_encoders: dict[type, callable] = {}
        self.custom_decoders: dict[str, callable] = {}
        
        # 注册默认序列化器
        self._register_default_serializers()
    
    def register_serializer(
        self, 
        state_type: type, 
        to_dict_func: callable, 
        from_dict_func: callable
    ) -> None:
        """注册状态类型的序列化器
        
        Args:
            state_type: 状态类型
            to_dict_func: 序列化函数
            from_dict_func: 反序列化函数
        """
        self.serializers[state_type] = (to_dict_func, from_dict_func)
        logger.info(f"Registered serializer for {state_type.__name__}")
    
    def register_custom_encoder(self, data_type: type, encoder: callable) -> None:
        """注册自定义编码器
        
        Args:
            data_type: 数据类型
            encoder: 编码函数
        """
        self.custom_encoders[data_type] = encoder
    
    def register_custom_decoder(self, type_name: str, decoder: callable) -> None:
        """注册自定义解码器
        
        Args:
            type_name: 类型名称
            decoder: 解码函数
        """
        self.custom_decoders[type_name] = decoder
    
    def serialize_state(self, state_module: StateModule) -> dict:
        """序列化状态模块
        
        Args:
            state_module: 状态模块实例
            
        Returns:
            序列化后的字典
        """
        state_type = type(state_module)
        
        if state_type in self.serializers:
            to_dict_func, _ = self.serializers[state_type]
            state_dict = to_dict_func(state_module)
        else:
            # 使用默认的state_dict方法
            state_dict = state_module.state_dict()
        
        # 添加类型信息
        serialized = {
            "__type__": f"{state_type.__module__}.{state_type.__name__}",
            "__version__": getattr(state_module, "__version__", "1.0.0"),
            "__timestamp__": datetime.utcnow().isoformat(),
            "state_data": self._encode_complex_types(state_dict)
        }
        
        return serialized
    
    def deserialize_state(self, data: dict, expected_type: type = None) -> StateModule:
        """反序列化状态模块
        
        Args:
            data: 序列化数据
            expected_type: 期望的类型
            
        Returns:
            状态模块实例
        """
        type_name = data.get("__type__")
        if not type_name:
            raise ValueError("Missing type information in serialized data")
        
        # 解析类型
        try:
            module_name, class_name = type_name.rsplit(".", 1)
            module = importlib.import_module(module_name)
            state_type = getattr(module, class_name)
        except (ValueError, ImportError, AttributeError) as e:
            raise ValueError(f"Cannot resolve type {type_name}: {e}")
        
        # 验证类型
        if expected_type and not issubclass(state_type, expected_type):
            raise TypeError(f"Expected {expected_type}, got {state_type}")
        
        # 反序列化状态数据
        state_data = self._decode_complex_types(data["state_data"])
        
        if state_type in self.serializers:
            _, from_dict_func = self.serializers[state_type]
            state_module = from_dict_func(state_data)
        else:
            # 创建实例并加载状态
            state_module = state_type()
            state_module.load_state_dict(state_data)
        
        return state_module
    
    def _encode_complex_types(self, obj: Any) -> Any:
        """编码复杂类型"""
        if isinstance(obj, dict):
            return {k: self._encode_complex_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._encode_complex_types(item) for item in obj]
        elif type(obj) in self.custom_encoders:
            encoder = self.custom_encoders[type(obj)]
            return {
                "__custom_type__": type(obj).__name__,
                "__data__": encoder(obj)
            }
        elif isinstance(obj, datetime):
            return {
                "__custom_type__": "datetime",
                "__data__": obj.isoformat()
            }
        elif isinstance(obj, Enum):
            return {
                "__custom_type__": "enum",
                "__enum_type__": f"{type(obj).__module__}.{type(obj).__name__}",
                "__data__": obj.value
            }
        else:
            return obj
    
    def _decode_complex_types(self, obj: Any) -> Any:
        """解码复杂类型"""
        if isinstance(obj, dict):
            if "__custom_type__" in obj:
                type_name = obj["__custom_type__"]
                data = obj["__data__"]
                
                if type_name in self.custom_decoders:
                    decoder = self.custom_decoders[type_name]
                    return decoder(data)
                elif type_name == "datetime":
                    return datetime.fromisoformat(data)
                elif type_name == "enum":
                    enum_type_name = obj["__enum_type__"]
                    module_name, class_name = enum_type_name.rsplit(".", 1)
                    module = importlib.import_module(module_name)
                    enum_type = getattr(module, class_name)
                    return enum_type(data)
                else:
                    logger.warning(f"Unknown custom type: {type_name}")
                    return data
            else:
                return {k: self._decode_complex_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._decode_complex_types(item) for item in obj]
        else:
            return obj
    
    def _register_default_serializers(self) -> None:
        """注册默认序列化器"""
        # 注册常用类型的编码器和解码器
        self.register_custom_encoder(set, list)
        self.register_custom_decoder("set", set)
        
        self.register_custom_encoder(frozenset, list)
        self.register_custom_decoder("frozenset", frozenset)
        
        # 注册UUID编码器
        try:
            import uuid
            self.register_custom_encoder(uuid.UUID, str)
            self.register_custom_decoder("UUID", uuid.UUID)
        except ImportError:
            pass
```

### 5.2 文件系统持久化后端

```python
# 基于AgentScope的文件系统持久化后端
class FileSystemPersistenceBackend:
    """文件系统持久化后端 - 处理状态的文件存储"""
    
    def __init__(self, base_path: str = "./state_storage", compression: bool = True):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.lock = asyncio.Lock()
        self.serialization_manager = StateSerializationManager()
        
        # 创建子目录
        (self.base_path / "agents").mkdir(exist_ok=True)
        (self.base_path / "sessions").mkdir(exist_ok=True)
        (self.base_path / "conversations").mkdir(exist_ok=True)
        (self.base_path / "backups").mkdir(exist_ok=True)
    
    async def save_state(
        self, 
        state_id: str, 
        state_module: StateModule, 
        category: str = "agents"
    ) -> None:
        """保存状态模块到文件系统
        
        Args:
            state_id: 状态标识符
            state_module: 状态模块实例
            category: 状态分类（agents, sessions, conversations等）
        """
        try:
            # 序列化状态
            serialized_data = self.serialization_manager.serialize_state(state_module)
            
            # 构建文件路径
            category_path = self.base_path / category
            file_path = category_path / f"{state_id}.json"
            
            # 创建备份（如果文件已存在）
            if file_path.exists():
                await self._create_backup(file_path)
            
            async with self.lock:
                if self.compression:
                    # 使用压缩存储
                    await self._save_compressed(file_path, serialized_data)
                else:
                    # 直接JSON存储
                    await self._save_json(file_path, serialized_data)
            
            logger.info(f"State {state_id} saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save state {state_id}: {e}")
            raise
    
    async def load_state(
        self, 
        state_id: str, 
        expected_type: type = None, 
        category: str = "agents"
    ) -> Optional[StateModule]:
        """从文件系统加载状态模块
        
        Args:
            state_id: 状态标识符
            expected_type: 期望的状态类型
            category: 状态分类
            
        Returns:
            状态模块实例或None
        """
        try:
            category_path = self.base_path / category
            file_path = category_path / f"{state_id}.json"
            
            if not file_path.exists():
                return None
            
            async with self.lock:
                if self.compression and file_path.suffix == ".gz":
                    serialized_data = await self._load_compressed(file_path)
                else:
                    serialized_data = await self._load_json(file_path)
            
            # 反序列化状态
            state_module = self.serialization_manager.deserialize_state(
                serialized_data, expected_type
            )
            
            logger.info(f"State {state_id} loaded from {file_path}")
            return state_module
            
        except Exception as e:
            logger.error(f"Failed to load state {state_id}: {e}")
            return None
    
    async def delete_state(self, state_id: str, category: str = "agents") -> bool:
        """删除状态文件
        
        Args:
            state_id: 状态标识符
            category: 状态分类
            
        Returns:
            是否成功删除
        """
        try:
            category_path = self.base_path / category
            file_path = category_path / f"{state_id}.json"
            
            async with self.lock:
                if file_path.exists():
                    # 创建删除前备份
                    await self._create_backup(file_path, suffix="_deleted")
                    file_path.unlink()
                    logger.info(f"State {state_id} deleted from {file_path}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete state {state_id}: {e}")
            return False
    
    async def exists_state(self, state_id: str, category: str = "agents") -> bool:
        """检查状态文件是否存在"""
        category_path = self.base_path / category
        file_path = category_path / f"{state_id}.json"
        return file_path.exists()
    
    async def list_states(self, category: str = "agents") -> List[str]:
        """列出指定分类下的所有状态ID"""
        category_path = self.base_path / category
        if not category_path.exists():
            return []
        
        state_files = category_path.glob("*.json")
        return [f.stem for f in state_files]
    
    async def get_state_info(self, state_id: str, category: str = "agents") -> Optional[dict]:
        """获取状态文件信息"""
        category_path = self.base_path / category
        file_path = category_path / f"{state_id}.json"
        
        if not file_path.exists():
            return None
        
        stat = file_path.stat()
        return {
            "state_id": state_id,
            "category": category,
            "file_path": str(file_path),
            "size_bytes": stat.st_size,
            "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat()
        }
    
    async def _save_json(self, file_path: Path, data: dict) -> None:
        """保存JSON数据"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    async def _load_json(self, file_path: Path) -> dict:
        """加载JSON数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    async def _save_compressed(self, file_path: Path, data: dict) -> None:
        """保存压缩数据"""
        import gzip
        compressed_path = file_path.with_suffix(file_path.suffix + ".gz")
        json_str = json.dumps(data, ensure_ascii=False)
        
        with gzip.open(compressed_path, 'wt', encoding='utf-8') as f:
            f.write(json_str)
    
    async def _load_compressed(self, file_path: Path) -> dict:
        """加载压缩数据"""
        import gzip
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    
    async def _create_backup(self, file_path: Path, suffix: str = "_backup") -> None:
        """创建文件备份"""
        if not file_path.exists():
            return
        
        backup_dir = self.base_path / "backups"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}{suffix}_{timestamp}{file_path.suffix}"
        backup_path = backup_dir / backup_name
        
        import shutil
        shutil.copy2(file_path, backup_path)
        logger.debug(f"Created backup: {backup_path}")
```

### 5.3 数据库持久化后端

```python
# 基于AgentScope的数据库持久化后端
class DatabasePersistenceBackend:
    """数据库持久化后端 - 处理状态的数据库存储"""
    
    def __init__(self, connection_config: dict):
        self.connection_config = connection_config
        self.pool = None
        self.serialization_manager = StateSerializationManager()
        self.db_type = connection_config.get("type", "postgresql")
    
    async def initialize(self) -> None:
        """初始化数据库连接池和表结构"""
        if self.db_type == "postgresql":
            await self._initialize_postgresql()
        elif self.db_type == "sqlite":
            await self._initialize_sqlite()
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
        
        logger.info(f"Database persistence backend initialized ({self.db_type})")
    
    async def _initialize_postgresql(self) -> None:
        """初始化PostgreSQL连接和表"""
        import asyncpg
        
        self.pool = await asyncpg.create_pool(
            host=self.connection_config["host"],
            port=self.connection_config.get("port", 5432),
            user=self.connection_config["user"],
            password=self.connection_config["password"],
            database=self.connection_config["database"],
            min_size=self.connection_config.get("min_connections", 5),
            max_size=self.connection_config.get("max_connections", 20)
        )
        
        # 创建状态表
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_states (
                    state_id VARCHAR(255) NOT NULL,
                    category VARCHAR(100) NOT NULL,
                    state_type VARCHAR(255) NOT NULL,
                    state_version VARCHAR(50) NOT NULL,
                    state_data JSONB NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (state_id, category)
                )
            """)
            
            # 创建索引
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_states_category 
                ON agent_states(category)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_states_type 
                ON agent_states(state_type)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_states_updated 
                ON agent_states(updated_at)
            """)
    
    async def _initialize_sqlite(self) -> None:
        """初始化SQLite连接和表"""
        import aiosqlite
        
        db_path = self.connection_config.get("path", "./state_storage.db")
        self.connection = await aiosqlite.connect(db_path)
        
        # 创建状态表
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS agent_states (
                state_id TEXT NOT NULL,
                category TEXT NOT NULL,
                state_type TEXT NOT NULL,
                state_version TEXT NOT NULL,
                state_data TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (state_id, category)
            )
        """)
        
        # 创建索引
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_states_category 
            ON agent_states(category)
        """)
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_states_type 
            ON agent_states(state_type)
        """)
        
        await self.connection.commit()
    
    async def save_state(
        self, 
        state_id: str, 
        state_module: StateModule, 
        category: str = "agents",
        metadata: dict = None
    ) -> None:
        """保存状态模块到数据库
        
        Args:
            state_id: 状态标识符
            state_module: 状态模块实例
            category: 状态分类
            metadata: 额外元数据
        """
        try:
            # 序列化状态
            serialized_data = self.serialization_manager.serialize_state(state_module)
            
            state_type = serialized_data["__type__"]
            state_version = serialized_data["__version__"]
            state_data_json = json.dumps(serialized_data["state_data"])
            metadata_json = json.dumps(metadata) if metadata else None
            
            if self.db_type == "postgresql":
                await self._save_postgresql(
                    state_id, category, state_type, state_version, 
                    state_data_json, metadata_json
                )
            elif self.db_type == "sqlite":
                await self._save_sqlite(
                    state_id, category, state_type, state_version, 
                    state_data_json, metadata_json
                )
            
            logger.info(f"State {state_id} saved to database ({category})")
            
        except Exception as e:
            logger.error(f"Failed to save state {state_id}: {e}")
            raise
    
    async def _save_postgresql(
        self, state_id: str, category: str, state_type: str, 
        state_version: str, state_data: str, metadata: str
    ) -> None:
        """保存到PostgreSQL"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO agent_states 
                (state_id, category, state_type, state_version, state_data, metadata, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP)
                ON CONFLICT (state_id, category) DO UPDATE SET
                    state_type = EXCLUDED.state_type,
                    state_version = EXCLUDED.state_version,
                    state_data = EXCLUDED.state_data,
                    metadata = EXCLUDED.metadata,
                    updated_at = EXCLUDED.updated_at
            """, state_id, category, state_type, state_version, state_data, metadata)
    
    async def _save_sqlite(
        self, state_id: str, category: str, state_type: str, 
        state_version: str, state_data: str, metadata: str
    ) -> None:
        """保存到SQLite"""
        await self.connection.execute("""
            INSERT OR REPLACE INTO agent_states 
            (state_id, category, state_type, state_version, state_data, metadata, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (state_id, category, state_type, state_version, state_data, metadata))
        await self.connection.commit()
    
    async def load_state(
        self, 
        state_id: str, 
        expected_type: type = None, 
        category: str = "agents"
    ) -> Optional[StateModule]:
        """从数据库加载状态模块
        
        Args:
            state_id: 状态标识符
            expected_type: 期望的状态类型
            category: 状态分类
            
        Returns:
            状态模块实例或None
        """
        try:
            if self.db_type == "postgresql":
                row = await self._load_postgresql(state_id, category)
            elif self.db_type == "sqlite":
                row = await self._load_sqlite(state_id, category)
            else:
                return None
            
            if not row:
                return None
            
            # 重构序列化数据
            serialized_data = {
                "__type__": row["state_type"],
                "__version__": row["state_version"],
                "__timestamp__": row["updated_at"].isoformat() if hasattr(row["updated_at"], "isoformat") else str(row["updated_at"]),
                "state_data": json.loads(row["state_data"])
            }
            
            # 反序列化状态
            state_module = self.serialization_manager.deserialize_state(
                serialized_data, expected_type
            )
            
            logger.info(f"State {state_id} loaded from database ({category})")
            return state_module
            
        except Exception as e:
            logger.error(f"Failed to load state {state_id}: {e}")
            return None
    
    async def _load_postgresql(self, state_id: str, category: str) -> Optional[dict]:
        """从PostgreSQL加载"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT state_type, state_version, state_data, metadata, updated_at
                FROM agent_states 
                WHERE state_id = $1 AND category = $2
            """, state_id, category)
            return dict(row) if row else None
    
    async def _load_sqlite(self, state_id: str, category: str) -> Optional[dict]:
        """从SQLite加载"""
        cursor = await self.connection.execute("""
            SELECT state_type, state_version, state_data, metadata, updated_at
            FROM agent_states 
            WHERE state_id = ? AND category = ?
        """, (state_id, category))
        row = await cursor.fetchone()
        
        if row:
            return {
                "state_type": row[0],
                "state_version": row[1], 
                "state_data": row[2],
                "metadata": row[3],
                "updated_at": row[4]
            }
        return None
    
    async def delete_state(self, state_id: str, category: str = "agents") -> bool:
        """删除状态记录"""
        try:
            if self.db_type == "postgresql":
                result = await self._delete_postgresql(state_id, category)
            elif self.db_type == "sqlite":
                result = await self._delete_sqlite(state_id, category)
            else:
                return False
            
            if result:
                logger.info(f"State {state_id} deleted from database ({category})")
            return result
            
        except Exception as e:
            logger.error(f"Failed to delete state {state_id}: {e}")
            return False
    
    async def _delete_postgresql(self, state_id: str, category: str) -> bool:
        """从PostgreSQL删除"""
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM agent_states 
                WHERE state_id = $1 AND category = $2
            """, state_id, category)
            return result.split()[-1] == "1"  # 检查影响行数
    
    async def _delete_sqlite(self, state_id: str, category: str) -> bool:
        """从SQLite删除"""
        cursor = await self.connection.execute("""
            DELETE FROM agent_states 
            WHERE state_id = ? AND category = ?
        """, (state_id, category))
        await self.connection.commit()
        return cursor.rowcount > 0
    
    async def list_states(self, category: str = "agents") -> List[str]:
        """列出指定分类下的所有状态ID"""
        try:
            if self.db_type == "postgresql":
                return await self._list_postgresql(category)
            elif self.db_type == "sqlite":
                return await self._list_sqlite(category)
            else:
                return []
        except Exception as e:
            logger.error(f"Failed to list states: {e}")
            return []
    
    async def _list_postgresql(self, category: str) -> List[str]:
        """从PostgreSQL列出状态"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT state_id FROM agent_states 
                WHERE category = $1 ORDER BY updated_at DESC
            """, category)
            return [row["state_id"] for row in rows]
    
    async def _list_sqlite(self, category: str) -> List[str]:
        """从SQLite列出状态"""
        cursor = await self.connection.execute("""
            SELECT state_id FROM agent_states 
            WHERE category = ? ORDER BY updated_at DESC
        """, (category,))
        rows = await cursor.fetchall()
        return [row[0] for row in rows]
    
    async def close(self) -> None:
        """关闭数据库连接"""
        if self.db_type == "postgresql" and self.pool:
            await self.pool.close()
        elif self.db_type == "sqlite" and hasattr(self, "connection"):
            await self.connection.close()
        
        logger.info("Database persistence backend closed")
```
            return None
```

### 5.2 持久化策略配置
```yaml
persistence:
  # 默认后端
  default_backend: "file"
  
  # 后端配置
  backends:
    memory:
      type: "memory"
    
    file:
      type: "filesystem"
      base_path: "./data/states"
      compression: true
      encryption: false
    
    database:
      type: "postgresql"
      connection_string: "postgresql://user:pass@localhost/agentdb"
      pool_size: 10
      timeout: 30
  
  # 持久化规则
  rules:
    - pattern: "agent:*"
      backend: "database"
      ttl: 86400  # 24小时
    
    - pattern: "session:*"
      backend: "file"
      ttl: 3600   # 1小时
    
    - pattern: "cache:*"
      backend: "memory"
      ttl: 300    # 5分钟
```

## 6. 状态同步和一致性

### 6.1 分布式状态同步
```python
class DistributedStateManager:
    """分布式状态管理器"""
    
    def __init__(self, node_id: str, cluster_nodes: List[str]):
        self.node_id = node_id
        self.cluster_nodes = cluster_nodes
        self.local_state = StateStore()
        self.sync_queue = asyncio.Queue()
        self.vector_clock = VectorClock(node_id, cluster_nodes)
    
    async def set_state(self, key: str, value: Any) -> None:
        """设置状态并同步到集群"""
        # 更新本地状态
        timestamp = self.vector_clock.tick()
        await self.local_state.set_state(key, value)
        
        # 创建同步消息
        sync_message = StateSyncMessage(
            key=key,
            value=value,
            timestamp=timestamp,
            node_id=self.node_id
        )
        
        # 广播到其他节点
        await self._broadcast_sync_message(sync_message)
    
    async def handle_sync_message(self, message: StateSyncMessage) -> None:
        """处理来自其他节点的同步消息"""
        # 更新向量时钟
        self.vector_clock.update(message.timestamp)
        
        # 检查是否需要更新本地状态
        current_value = await self.local_state.get_state(message.key)
        if self._should_update(current_value, message):
            await self.local_state.set_state(message.key, message.value)
    
    def _should_update(self, current_value: Any, message: StateSyncMessage) -> bool:
        """判断是否应该更新本地状态"""
        # 基于向量时钟的冲突解决
        if current_value is None:
            return True
        
        current_timestamp = getattr(current_value, 'timestamp', None)
        if current_timestamp is None:
            return True
        
        return self.vector_clock.happens_before(current_timestamp, message.timestamp)
```

### 6.2 状态一致性检查
```python
class StateConsistencyChecker:
    """状态一致性检查器"""
    
    def __init__(self, state_store: StateStore):
        self.state_store = state_store
    
    async def check_consistency(self) -> ConsistencyReport:
        """检查状态一致性"""
        report = ConsistencyReport()
        
        # 检查状态完整性
        await self._check_integrity(report)
        
        # 检查状态关联性
        await self._check_relationships(report)
        
        # 检查状态有效性
        await self._check_validity(report)
        
        return report
    
    async def _check_integrity(self, report: ConsistencyReport) -> None:
        """检查状态完整性"""
        # 检查必需的状态是否存在
        required_states = ["system:config", "system:status"]
        for state_key in required_states:
            if not await self.state_store.get_state(state_key):
                report.add_error(f"Missing required state: {state_key}")
    
    async def _check_relationships(self, report: ConsistencyReport) -> None:
        """检查状态关联性"""
        # 检查智能体和会话的关联关系
        # 实现具体的关联性检查逻辑
        pass
    
    async def _check_validity(self, report: ConsistencyReport) -> None:
        """检查状态有效性"""
        # 检查状态值的有效性
        # 实现具体的有效性检查逻辑
        pass

@dataclass
class ConsistencyReport:
    """一致性检查报告"""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def add_error(self, error: str):
        self.errors.append(error)
    
    def add_warning(self, warning: str):
        self.warnings.append(warning)
    
    @property
    def is_consistent(self) -> bool:
        return len(self.errors) == 0
```

## 7. 状态备份和恢复

### 7.1 状态备份
```python
class StateBackupManager:
    """状态备份管理器"""
    
    def __init__(self, state_store: StateStore, backup_path: str):
        self.state_store = state_store
        self.backup_path = Path(backup_path)
        self.backup_path.mkdir(parents=True, exist_ok=True)
    
    async def create_backup(self, backup_name: str = None) -> str:
        """创建状态备份"""
        if not backup_name:
            backup_name = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        backup_file = self.backup_path / f"{backup_name}.json"
        
        # 获取所有状态数据
        all_states = await self._get_all_states()
        
        # 创建备份元数据
        backup_data = {
            "metadata": {
                "backup_name": backup_name,
                "created_at": datetime.utcnow().isoformat(),
                "version": "1.0",
                "total_states": len(all_states)
            },
            "states": all_states
        }
        
        # 写入备份文件
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, ensure_ascii=False, indent=2, default=str)
        
        return backup_name
    
    async def restore_backup(self, backup_name: str, selective: List[str] = None) -> bool:
        """恢复状态备份"""
        backup_file = self.backup_path / f"{backup_name}.json"
        
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_file}")
        
        try:
            with open(backup_file, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            states = backup_data.get("states", {})
            
            # 选择性恢复
            if selective:
                states = {k: v for k, v in states.items() if k in selective}
            
            # 恢复状态
            for key, value in states.items():
                await self.state_store.set_state(key, value)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore backup {backup_name}: {e}")
            return False
    
    async def _get_all_states(self) -> Dict[str, Any]:
        """获取所有状态数据"""
        # 这里需要根据具体的状态存储实现来获取所有状态
        # 示例实现
        return self.state_store.states.copy()
```

### 7.2 自动备份策略
```python
class AutoBackupScheduler:
    """自动备份调度器"""
    
    def __init__(self, backup_manager: StateBackupManager, config: Dict):
        self.backup_manager = backup_manager
        self.config = config
        self.scheduler = None
    
    def start(self):
        """启动自动备份"""
        interval = self.config.get("interval", 3600)  # 默认1小时
        self.scheduler = asyncio.create_task(self._backup_loop(interval))
    
    def stop(self):
        """停止自动备份"""
        if self.scheduler:
            self.scheduler.cancel()
    
    async def _backup_loop(self, interval: int):
        """备份循环"""
        while True:
            try:
                await asyncio.sleep(interval)
                backup_name = await self.backup_manager.create_backup()
                logger.info(f"Auto backup created: {backup_name}")
                
                # 清理旧备份
                await self._cleanup_old_backups()
                
            except Exception as e:
                logger.error(f"Auto backup failed: {e}")
    
    async def _cleanup_old_backups(self):
        """清理旧备份"""
        max_backups = self.config.get("max_backups", 10)
        backup_files = list(self.backup_manager.backup_path.glob("backup_*.json"))
        
        if len(backup_files) > max_backups:
            # 按修改时间排序，删除最旧的备份
            backup_files.sort(key=lambda x: x.stat().st_mtime)
            for old_backup in backup_files[:-max_backups]:
                old_backup.unlink()
                logger.info(f"Deleted old backup: {old_backup.name}")
```

## 8. 基于AgentScope的性能优化

### 8.1 状态性能管理器

```python
# 基于AgentScope的状态性能管理器
class StatePerformanceManager:
    """状态性能管理器 - 提供缓存、压缩和优化功能"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.cache_enabled = self.config.get("cache_enabled", True)
        self.compression_enabled = self.config.get("compression_enabled", True)
        self.max_cache_size = self.config.get("max_cache_size", 1000)
        self.cache_ttl = self.config.get("cache_ttl", 3600)
        
        # 初始化缓存
        if self.cache_enabled:
            self.cache = StateLRUCache(
                max_size=self.max_cache_size,
                ttl=self.cache_ttl
            )
        
        # 初始化压缩器
        if self.compression_enabled:
            self.compressor = StateCompressor()
        
        # 性能统计
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "compression_ratio": 0.0,
            "avg_load_time": 0.0,
            "avg_save_time": 0.0
        }
    
    async def get_cached_state(
        self, 
        state_id: str, 
        category: str = "agents"
    ) -> Optional[StateModule]:
        """从缓存获取状态
        
        Args:
            state_id: 状态标识符
            category: 状态分类
            
        Returns:
            缓存的状态模块或None
        """
        if not self.cache_enabled:
            return None
        
        cache_key = f"{category}:{state_id}"
        cached_state = await self.cache.get(cache_key)
        
        if cached_state:
            self.stats["cache_hits"] += 1
            logger.debug(f"Cache hit for state {state_id}")
            return cached_state
        else:
            self.stats["cache_misses"] += 1
            logger.debug(f"Cache miss for state {state_id}")
            return None
    
    async def cache_state(
        self, 
        state_id: str, 
        state_module: StateModule, 
        category: str = "agents"
    ) -> None:
        """缓存状态模块
        
        Args:
            state_id: 状态标识符
            state_module: 状态模块实例
            category: 状态分类
        """
        if not self.cache_enabled:
            return
        
        cache_key = f"{category}:{state_id}"
        await self.cache.set(cache_key, state_module)
        logger.debug(f"State {state_id} cached")
    
    def compress_state_data(self, data: dict) -> bytes:
        """压缩状态数据
        
        Args:
            data: 状态数据字典
            
        Returns:
            压缩后的字节数据
        """
        if not self.compression_enabled:
            return json.dumps(data).encode('utf-8')
        
        original_size = len(json.dumps(data).encode('utf-8'))
        compressed_data = self.compressor.compress(data)
        compressed_size = len(compressed_data)
        
        # 更新压缩比统计
        if original_size > 0:
            ratio = compressed_size / original_size
            self.stats["compression_ratio"] = (
                self.stats["compression_ratio"] * 0.9 + ratio * 0.1
            )
        
        logger.debug(f"Compressed {original_size} bytes to {compressed_size} bytes")
        return compressed_data
    
    def get_performance_stats(self) -> dict:
        """获取性能统计信息
        
        Returns:
            性能统计字典
        """
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        hit_rate = (
            self.stats["cache_hits"] / total_requests 
            if total_requests > 0 else 0.0
        )
        
        return {
            **self.stats,
            "cache_hit_rate": hit_rate,
            "total_cache_requests": total_requests
        }


class StateLRUCache:
    """状态LRU缓存实现"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: dict[str, CacheEntry] = {}
        self.access_order: list[str] = []
        self.lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        async with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # 检查是否过期
            if time.time() - entry.timestamp > self.ttl:
                await self._remove(key)
                return None
            
            # 更新访问顺序
            self._update_access_order(key)
            return entry.value
    
    async def set(self, key: str, value: Any) -> None:
        """设置缓存值"""
        async with self.lock:
            # 如果缓存已满且不是更新现有键，则淘汰LRU项
            if len(self.cache) >= self.max_size and key not in self.cache:
                await self._evict_lru()
            
            self.cache[key] = CacheEntry(value, time.time())
            self._update_access_order(key)
    
    async def _remove(self, key: str) -> None:
        """移除缓存项"""
        if key in self.cache:
            del self.cache[key]
            if key in self.access_order:
                self.access_order.remove(key)
    
    async def _evict_lru(self) -> None:
        """淘汰最近最少使用的项"""
        if self.access_order:
            lru_key = self.access_order[0]
            await self._remove(lru_key)
    
    def _update_access_order(self, key: str) -> None:
        """更新访问顺序"""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)


class StateCompressor:
    """状态压缩器"""
    
    def __init__(self, algorithm: str = "gzip"):
        self.algorithm = algorithm
    
    def compress(self, data: Any) -> bytes:
        """压缩状态数据
        
        Args:
            data: 要压缩的数据
            
        Returns:
            压缩后的字节数据
        """
        json_str = json.dumps(data, default=str, separators=(',', ':'))
        
        if self.algorithm == "gzip":
            return gzip.compress(json_str.encode('utf-8'))
        elif self.algorithm == "zlib":
            import zlib
            return zlib.compress(json_str.encode('utf-8'))
        else:
            return json_str.encode('utf-8')
    
    def decompress(self, compressed_data: bytes) -> Any:
        """解压缩状态数据
        
        Args:
            compressed_data: 压缩的字节数据
            
        Returns:
            解压缩后的数据
        """
        if self.algorithm == "gzip":
            json_str = gzip.decompress(compressed_data).decode('utf-8')
        elif self.algorithm == "zlib":
            import zlib
            json_str = zlib.decompress(compressed_data).decode('utf-8')
        else:
            json_str = compressed_data.decode('utf-8')
        
        return json.loads(json_str)


@dataclass
class CacheEntry:
    """缓存条目"""
    value: Any
    timestamp: float
```

### 8.2 性能监控和优化

```python
class StatePerformanceMonitor:
    """状态性能监控器"""
    
    def __init__(self, performance_manager: StatePerformanceManager):
        self.performance_manager = performance_manager
        self.metrics_history = []
        self.monitoring_enabled = True
    
    async def collect_metrics(self) -> dict:
        """收集性能指标
        
        Returns:
            性能指标字典
        """
        if not self.monitoring_enabled:
            return {}
        
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "performance_stats": self.performance_manager.get_performance_stats(),
            "memory_usage": self._get_memory_usage(),
            "cache_stats": await self._get_cache_stats()
        }
        
        # 保存历史记录
        self.metrics_history.append(metrics)
        
        # 限制历史记录数量
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        return metrics
    
    def _get_memory_usage(self) -> dict:
        """获取内存使用情况"""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss": memory_info.rss,
            "vms": memory_info.vms,
            "percent": process.memory_percent()
        }
    
    async def _get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        if hasattr(self.performance_manager, 'cache'):
            return await self.performance_manager.cache.get_stats()
        return {}
    
    async def generate_performance_report(self) -> dict:
        """生成性能报告
        
        Returns:
            性能报告字典
        """
        current_metrics = await self.collect_metrics()
        
        # 分析趋势
        trends = self._analyze_trends()
        
        # 生成建议
        recommendations = self._generate_recommendations(current_metrics, trends)
        
        return {
            "current_metrics": current_metrics,
            "trends": trends,
            "recommendations": recommendations,
            "report_generated_at": datetime.utcnow().isoformat()
        }
    
    def _analyze_trends(self) -> dict:
        """分析性能趋势"""
        if len(self.metrics_history) < 2:
            return {"insufficient_data": True}
        
        # 分析缓存命中率趋势
        hit_rates = [
            m["performance_stats"].get("cache_hit_rate", 0) 
            for m in self.metrics_history[-10:]
        ]
        
        # 分析内存使用趋势
        memory_usage = [
            m["memory_usage"].get("percent", 0) 
            for m in self.metrics_history[-10:]
        ]
        
        return {
            "cache_hit_rate_trend": self._calculate_trend(hit_rates),
            "memory_usage_trend": self._calculate_trend(memory_usage),
            "data_points": len(hit_rates)
        }
    
    def _calculate_trend(self, values: list) -> str:
        """计算趋势方向"""
        if len(values) < 2:
            return "stable"
        
        recent_avg = sum(values[-3:]) / len(values[-3:])
        earlier_avg = sum(values[:-3]) / len(values[:-3]) if len(values) > 3 else values[0]
        
        if recent_avg > earlier_avg * 1.1:
            return "increasing"
        elif recent_avg < earlier_avg * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def _generate_recommendations(self, metrics: dict, trends: dict) -> list:
        """生成性能优化建议"""
        recommendations = []
        
        # 缓存相关建议
        hit_rate = metrics["performance_stats"].get("cache_hit_rate", 0)
        if hit_rate < 0.5:
            recommendations.append({
                "type": "cache_optimization",
                "priority": "high",
                "message": "缓存命中率较低，建议增加缓存大小或调整TTL"
            })
        
        # 内存相关建议
        memory_percent = metrics["memory_usage"].get("percent", 0)
        if memory_percent > 80:
            recommendations.append({
                "type": "memory_optimization",
                "priority": "high",
                "message": "内存使用率过高，建议启用状态压缩或清理缓存"
            })
        
        # 趋势相关建议
        if trends.get("memory_usage_trend") == "increasing":
            recommendations.append({
                "type": "trend_warning",
                "priority": "medium",
                "message": "内存使用呈上升趋势，需要监控内存泄漏"
            })
        
        return recommendations
```

---

**文档版本**: v0.2.1  
**创建日期**: 2025-01-08  
**更新日期**: 2025-01-08  
**负责人**: 开发团队  
**审核状态**: 已完成基于AgentScope的重新设计