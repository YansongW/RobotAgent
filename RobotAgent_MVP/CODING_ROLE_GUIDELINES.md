# RobotAgent MVP 代码开发规范与角色指南

## 📋 文档概述

本文档为RobotAgent MVP项目的代码开发提供详细的规范指导，确保代码质量、一致性和可维护性。所有开发人员必须严格遵循本规范。

## 🎯 核心开发原则

### 1. 基于已有架构扩展 (Architecture Extension)

**核心理念**: 基于已完成的 `base_agent.py` 进行扩展开发，严禁破坏现有架构。

**具体要求**:
- 所有新智能体必须继承 `BaseRobotAgent` 抽象基类
- 遵循已定义的状态管理、消息传递、工具集成机制
- 保持与现有枚举类型（AgentState、MessageType、TaskStatus等）的兼容性

### 2. CAMEL框架集成原则 (CAMEL Integration)

**设计理念**: 融合CAMEL、Eigent、OWL三大项目的优势

**实现要求**:
- 遵循CAMEL的四大核心原则：可进化性、可扩展性、状态性、代码即提示
- 集成CAMEL的ChatAgent、BaseMessage、工具系统
- 参考Eigent的多智能体协作模式
- 借鉴OWL的优化学习机制

### 3. 代码即提示原则 (Code-as-Prompt)

**核心要求**: 每行代码和注释都应作为智能体的提示

**实现标准**:
```python
# ✅ 正确示例：详细的功能说明注释
class ChatAgent(BaseRobotAgent):
    
    # 对话智能体 (Chat Agent)
    
    # 基于CAMEL框架的自然语言处理智能体，专注于：
    # 1. 多轮对话管理和上下文维护
    # 2. 自然语言理解和生成
    # 3. 情感分析和意图识别
    # 4. 与其他智能体的协作通信
    
    # 继承自BaseRobotAgent，实现了完整的对话处理流程。
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        # 初始化对话智能体
        # Args:
        #     agent_id: 智能体唯一标识符
        #     config: 配置参数字典，包含模型配置、提示模板等
        # 调用父类初始化，设置智能体类型为"chat"
        super().__init__(agent_id, "chat", config)
```

## 🚫 严禁修改规则

### 禁止直接修改的文件

**严格禁止直接修改以下已完成的文件**:
- `src/agents/base_agent.py` - 智能体基类（已完成）
- `src/agents/__init__.py` - 模块初始化文件
- 所有已通过测试的配置文件

### 修改申请流程

**如需修改已完成代码，必须遵循以下流程**:

1. **申请阶段**:
   - 向项目负责人提交修改申请
   - 详细说明修改原因和影响范围
   - 获得明确批准后方可进行

2. **修改方式**:
   ```python
   # ❌ 错误方式：直接删除或替换原代码
   # def old_function():
   #     return "old implementation"
   
   # ✅ 正确方式：注释原代码，添加新实现
   # 原始实现 - 由于性能问题需要优化 (修改原因)
   # 原代码在处理大量消息时存在内存泄漏问题 (错误原因)
   # def old_function():
   #     return "old implementation"
   
   def old_function():
       
       # 优化后的函数实现
       
       # 修改说明:
       # - 原因: 原实现存在内存泄漏问题
       # - 方法: 使用上下文管理器确保资源释放
       # - 影响: 提升内存使用效率，不影响外部接口
       
       return "optimized implementation"
   ```

3. **修改标注要求**:
   ```python
   # 修改标注模板
   
   # 代码修改记录:
   # - 修改时间: 2025-XX-Xx
   # - 修改人员: [开发者姓名]
   # - 修改原因: [详细说明为什么需要修改]
   # - 修改方法: [具体的修改方式和技术方案]
   # - 影响分析: [对系统其他部分的影响评估]
   # - 测试状态: [修改后的测试结果]
   ```

## 📝 代码规范标准

### 1. 文件结构规范

**标准文件头部**:
```python
# -*- coding: utf-8 -*-

# [智能体名称] ([英文名称])
# [功能描述]
# 作者: RobotAgent开发团队
# 版本: 0.0.1 (Initial Release)
# 更新时间: [当前日期]
# 基于: BaseRobotAgent v0.0.1

# 导入标准库
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# 导入项目基础组件
from .base_agent import (
    BaseRobotAgent, AgentState, MessageType, TaskStatus,
    AgentMessage, TaskDefinition, AgentCapability
)

# 导入CAMEL框架组件
try:
    from camel.agents import ChatAgent
    from camel.messages import BaseMessage
    from camel.models import ModelFactory
    CAMEL_AVAILABLE = True
except ImportError:
    CAMEL_AVAILABLE = False
    logging.warning("CAMEL框架未安装，使用模拟实现")
```

### 2. 类设计规范

**智能体类设计模板**:
```python
class [AgentName](BaseRobotAgent):
    
    # [智能体中文名称] ([Agent English Name])
    
    # [详细功能描述，包括：]
    # 1. 主要职责和功能
    # 2. 与其他智能体的协作方式
    # 3. 使用的核心技术和算法
    # 4. 输入输出格式说明
    
    # 继承自XXXX_Agent，实现了[具体实现的抽象方法]。
    
    Attributes:
        [属性名称]: [属性描述]
        
    Example:
        >>> agent = [AgentName]("agent_001")
        >>> await agent.start()
        >>> result = await agent.execute_task(task)
    # 
    
    def __init__(self, 
                 agent_id: str,
                 config: Dict[str, Any] = None,
                 **kwargs):
        # 初始化[智能体名称]
        # 
        # Args:
        #     agent_id: 智能体唯一标识符
            config: 配置参数字典
            **kwargs: 其他初始化参数
            
        Raises:
            ValueError: 当配置参数无效时
            ImportError: 当依赖库未安装时
        #
        # 设置智能体类型和默认配置
        super().__init__(agent_id, "[agent_type]", config, **kwargs)
        
        # 初始化专业化组件
        self._init_specialized_components()
        
        # 注册专业化工具
        self._register_specialized_tools()
        
        # 添加专业化能力
        self._add_specialized_capabilities()
```

### 3. 方法实现规范

**抽象方法实现**:
```python
async def execute_task(self, task: TaskDefinition) -> Any:
    
    # 执行任务的核心方法实现
    
    # 这是BaseRobotAgent的抽象方法，必须在子类中实现。
    # [智能体名称]的任务执行流程包括：
    # 1. 任务分析和预处理
    # 2. [具体处理步骤]
    # 3. 结果生成和后处理
    
    Args:
        task: 任务定义对象，包含任务类型、参数等信息
        
    Returns:
        Any: 任务执行结果，格式取决于任务类型
        
    Raises:
        ValueError: 当任务参数无效时
        RuntimeError: 当任务执行失败时
    #
    try:
        # 更新智能体状态
        await self._set_state(AgentState.EXECUTING)
        
        # 记录任务开始
        self.logger.info(f"开始执行任务: {task.task_id}")
        
        # 任务执行逻辑
        result = await self._execute_task_logic(task)
        
        # 更新任务状态
        task.status = TaskStatus.COMPLETED
        task.result = result
        
        # 记录任务完成
        self.logger.info(f"任务执行完成: {task.task_id}")
        
        return result
        
    except Exception as e:
        # 错误处理
        task.status = TaskStatus.FAILED
        task.error_info = str(e)
        await self._set_state(AgentState.ERROR)
        
        self.logger.error(f"任务执行失败: {task.task_id}, 错误: {e}")
        raise
    
    finally:
        # 清理和状态恢复
        await self._set_state(AgentState.IDLE)
```

### 4. 错误处理规范

**统一错误处理模式**:
```python
try:
    # 主要逻辑
    result = await self._process_logic()
    
except ValueError as e:
    # 参数错误处理
    self.logger.error(f"参数错误: {e}")
    await self._handle_parameter_error(e)
    raise
    
except RuntimeError as e:
    # 运行时错误处理
    self.logger.error(f"运行时错误: {e}")
    await self._handle_runtime_error(e)
    raise
    
except Exception as e:
    # 通用错误处理
    self.logger.error(f"未知错误: {e}")
    await self._handle_unknown_error(e)
    raise
    
finally:
    # 清理资源
    await self._cleanup_resources()
```

## 🔧 技术实现指南

### 1. CAMEL框架集成

**ChatAgent集成模式**:
```python
def _init_camel_agent(self):
    
    # 初始化CAMEL ChatAgent
    
    # 基于配置创建CAMEL ChatAgent实例，用于自然语言处理。
    # 如果CAMEL框架不可用，将使用模拟实现.

    if not CAMEL_AVAILABLE:
        self.logger.warning("CAMEL框架不可用，使用模拟实现")
        self._camel_agent = None
        return
    
    try:
        # 构建系统提示
        system_prompt = self._build_system_prompt()
        
        # 创建CAMEL ChatAgent
        self._camel_agent = ChatAgent(
            system_message=system_prompt,
            model=self._create_model_backend(),
            message_window_size=self.config.get('message_window_size', 10)
        )
        
        self.logger.info("CAMEL ChatAgent初始化成功")
        
    except Exception as e:
        self.logger.error(f"CAMEL ChatAgent初始化失败: {e}")
        self._camel_agent = None
```

### 2. 异步编程规范

**异步方法设计**:
```python
async def _async_method_template(self, param: Any) -> Any:
    
    # 异步方法模板
    
    # 所有异步方法都应该遵循这个模板，确保：
    # 1. 正确的错误处理
    # 2. 资源管理
    # 3. 状态更新
    # 4. 日志记录
    
    # 参数验证
    if not self._validate_parameters(param):
        raise ValueError(f"无效参数: {param}")
    
    # 状态检查
    if not self.is_running:
        raise RuntimeError("智能体未运行")
    
    try:
        # 异步操作
        result = await self._perform_async_operation(param)
        
        # 结果验证
        if not self._validate_result(result):
            raise RuntimeError("操作结果无效")
        
        return result
        
    except asyncio.CancelledError:
        self.logger.info("操作被取消")
        raise
        
    except Exception as e:
        self.logger.error(f"异步操作失败: {e}")
        raise
```

### 3. 配置管理规范

**配置加载和验证**:
```python
def _load_and_validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
    
    # 加载和验证配置
    
    # 确保配置参数的完整性和有效性。
    
    # 默认配置
    default_config = {
        'model_name': 'gpt-3.5-turbo',
        'temperature': 0.7,
        'max_tokens': 1000,
        'timeout': 30.0,
        'retry_count': 3
    }
    
    # 合并配置
    merged_config = {**default_config, **(config or {})}
    
    # 验证配置
    self._validate_config(merged_config)
    
    return merged_config

def _validate_config(self, config: Dict[str, Any]):
    # 验证配置参数
    required_keys = ['model_name', 'temperature']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"缺少必需的配置参数: {key}")
    
    # 参数范围验证
    if not 0 <= config['temperature'] <= 2:
        raise ValueError("temperature必须在0-2之间")
```

## 📊 测试规范

### 1. 单元测试要求

**测试文件命名**: `test_[agent_name].py`

**测试类结构**:
```python
import pytest
import asyncio
from unittest.mock import Mock, patch

from src.agents.chat_agent import ChatAgent
from src.agents.base_agent import TaskDefinition, AgentState

class TestChatAgent:
    
    # ChatAgent单元测试类
    
    # 测试覆盖范围：
    # 1. 初始化和配置
    # 2. 任务执行逻辑
    # 3. 消息处理
    # 4. 错误处理
    # 5. 状态管理
    
    @pytest.fixture
    async def chat_agent(self):
        # 创建测试用的ChatAgent实例
        agent = ChatAgent("test_chat_agent")
        await agent.start()
        yield agent
        await agent.stop()
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        # 测试智能体初始化
        agent = ChatAgent("test_agent")
        assert agent.agent_id == "test_agent"
        assert agent.agent_type == "chat"
        assert agent.state == AgentState.INITIALIZING
    
    @pytest.mark.asyncio
    async def test_task_execution(self, chat_agent):
        # 测试任务执行
        task = TaskDefinition(
            task_id="test_task",
            task_type="chat",
            description="测试对话任务"
        )
        
        result = await chat_agent.execute_task(task)
        assert result is not None
        assert task.status == TaskStatus.COMPLETED
```

### 2. 集成测试要求

**多智能体协作测试**:
```python
@pytest.mark.asyncio
async def test_multi_agent_collaboration():
    # 测试多智能体协作
    # 创建多个智能体
    chat_agent = ChatAgent("chat_001")
    action_agent = ActionAgent("action_001")
    
    # 启动智能体
    await chat_agent.start()
    await action_agent.start()
    
    try:
        # 测试协作流程
        collaboration_id = await chat_agent.request_collaboration(
            "action_001", "task_delegation", {"task": "执行动作"}
        )
        
        # 验证协作结果
        assert collaboration_id is not None
        
    finally:
        # 清理资源
        await chat_agent.stop()
        await action_agent.stop()
```

## 📚 文档规范

### 1. 代码文档

**类文档模板**:
```python
class ExampleAgent(BaseRobotAgent):
    
    # 示例智能体类
    
    # 这是一个示例智能体，展示了如何正确继承XXXX_Agent
    # 并实现所需的功能。
    
    Attributes:
        specialized_config (Dict[str, Any]): 专业化配置参数
        processing_queue (asyncio.Queue): 处理队列
        
    Example:
        创建和使用示例智能体：
        
        >>> agent = ExampleAgent("example_001")
        >>> await agent.start()
        >>> task = TaskDefinition("task_001", "example", "示例任务")
        >>> result = await agent.execute_task(task)
        >>> await agent.stop()
        
    Note:
        这个智能体需要特定的配置参数才能正常工作。
        请参考配置文档了解详细信息。
        
    See Also:
        BaseRobotAgent: 基础智能体类
        TaskDefinition: 任务定义数据结构
    #
```

### 2. 变更日志

**每个文件都应包含变更记录**:
```python

# 变更历史:

# v0.0.1 (2025-01-XX):
# - 初始实现
# - 基础对话功能
# - CAMEL框架集成

# v0.0.2 (2025-01-XX):
# - 添加情感分析功能
# - 优化响应生成算法
# - 修复内存泄漏问题

```

## ✅ 代码审查清单

### 提交前检查项

- [ ] 代码遵循PEP 8规范
- [ ] 所有方法都有完整的文档字符串
- [ ] 错误处理完整且合理
- [ ] 单元测试覆盖率 > 80%
- [ ] 没有硬编码的配置值
- [ ] 日志记录适当且有意义
- [ ] 异步方法正确处理取消和超时
- [ ] 资源管理正确（文件、连接等）
- [ ] 与BaseRobotAgent接口兼容
- [ ] CAMEL框架集成正确

### 性能检查项

- [ ] 没有阻塞的同步调用
- [ ] 内存使用合理
- [ ] 响应时间在可接受范围内
- [ ] 并发处理能力满足要求
- [ ] 错误恢复机制有效

## 🎯 总结

本规范文档确保了RobotAgent MVP项目的代码质量和一致性。所有开发人员必须：

1. **严格遵循**已定义的架构和接口
2. **禁止直接修改**已完成的核心文件
3. **遵循CAMEL框架**的设计原则
4. **保持代码质量**和文档完整性
5. **确保测试覆盖**和性能要求

通过遵循这些规范，我们能够构建一个高质量、可维护、可扩展的智能体系统。

---

*RobotAgent开发团队*  
*版本: 1.0*  
*更新时间: 2025年8月15日*