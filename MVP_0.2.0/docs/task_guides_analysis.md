# -*- coding: utf-8 -*-

# AgentScope Task Guides深度分析 (AgentScope Task Guides Deep Analysis)
# 基于AgentScope v0.0.3.3源码的任务指南系统架构分析
# 版本: 0.2.0
# 更新时间: 2025-09-10

# AgentScope Task Guides深度分析

## 概述

Task Guides是AgentScope框架中的核心任务指南模块，提供了完整的开发指导和最佳实践。本文档基于AgentScope v0.0.3.3源码，深度分析Task Guides的架构设计、核心实现和应用场景。

## 核心架构

### 1. 模型系统 (Model)

#### 统一模型接口

AgentScope提供了统一的模型接口，支持多种主流LLM提供商：

```python
# 支持的模型提供商和功能矩阵
SUPPORTED_MODELS = {
    "OpenAI": {
        "class": "OpenAIChatModel",
        "compatible": ["vLLM", "DeepSeek"],
        "features": ["streaming", "tools", "vision", "reasoning"]
    },
    "DashScope": {
        "class": "DashScopeChatModel", 
        "features": ["streaming", "tools", "vision", "reasoning"]
    },
    "Anthropic": {
        "class": "AnthropicChatModel",
        "features": ["streaming", "tools", "vision", "reasoning"]
    },
    "Gemini": {
        "class": "GeminiChatModel",
        "features": ["streaming", "tools", "vision", "reasoning"]
    },
    "Ollama": {
        "class": "OllamaChatModel",
        "features": ["streaming", "tools", "vision", "reasoning"]
    }
}
```

#### 模型调用示例

```python
# 基础模型调用
async def example_model_call() -> None:
    """模型调用示例"""
    model = DashScopeChatModel(
        model_name="qwen-max",
        api_key=os.environ["DASHSCOPE_API_KEY"],
        stream=False,
    )
    
    res = await model(
        messages=[
            {"role": "user", "content": "Hi!"},
        ],
    )
    
    # 直接创建Msg对象
    msg_res = Msg("Friday", res.content, "assistant")
    print("The response:", res)
    print("The response as Msg:", msg_res)
```

#### 流式响应处理

```python
# 流式模型调用
async def example_streaming() -> None:
    """流式响应示例"""
    model = DashScopeChatModel(
        model_name="qwen-max",
        api_key=os.environ["DASHSCOPE_API_KEY"],
        stream=True,
    )
    
    generator = await model(
        messages=[
            {
                "role": "user",
                "content": "Count from 1 to 20, and just report the number without any other information.",
            },
        ],
    )
    
    # 累积式流式响应处理
    i = 0
    async for chunk in generator:
        print(f"Chunk {i}")
        print(f"\ttype: {type(chunk.content)}")
        print(f"\t{chunk}\n")
        i += 1
```

#### 推理模型支持

```python
# 推理模型调用
async def example_reasoning() -> None:
    """推理模型示例"""
    model = DashScopeChatModel(
        model_name="qwen-turbo",
        api_key=os.environ["DASHSCOPE_API_KEY"],
        enable_thinking=True,
    )
    
    res = await model(
        messages=[
            {"role": "user", "content": "Who am I?"},
        ],
    )
    
    # 处理ThinkingBlock
    last_chunk = None
    async for chunk in res:
        last_chunk = chunk
    print("The final response:")
    print(last_chunk)
```

### 2. 提示词格式化器 (Prompt Formatter)

#### 格式化器类型

AgentScope提供两种类型的格式化器：

```python
# 格式化器分类
FORMATTER_TYPES = {
    "ChatFormatter": {
        "description": "标准用户-助手场景",
        "identification": "使用role字段识别用户和助手",
        "use_case": "聊天机器人场景"
    },
    "MultiAgentFormatter": {
        "description": "多智能体场景", 
        "identification": "使用name字段识别不同智能体",
        "use_case": "多智能体协作场景"
    }
}
```

#### 内置格式化器支持矩阵

```python
# 内置格式化器功能支持
FORMATTER_SUPPORT_MATRIX = {
    "OpenAIChatFormatter": {
        "tool_use_result": True,
        "image": True,
        "audio": True,
        "video": False,
        "thinking": False
    },
    "DashScopeChatFormatter": {
        "tool_use_result": True,
        "image": True,
        "audio": True,
        "video": False,
        "thinking": False
    },
    "AnthropicChatFormatter": {
        "tool_use_result": True,
        "image": True,
        "audio": False,
        "video": False,
        "thinking": True
    },
    "GeminiChatFormatter": {
        "tool_use_result": True,
        "image": True,
        "audio": True,
        "video": True,
        "thinking": False
    }
}
```

#### 多智能体消息格式化

```python
# 多智能体消息格式化示例
async def run_formatter_example() -> list[dict]:
    """多智能体消息格式化示例"""
    formatter = DashScopeMultiAgentFormatter()
    
    input_msgs = [
        # 系统提示
        Msg("system", "You're a helpful assistant named Friday", "system"),
        # 对话历史
        Msg("Bob", "Hi, Alice, do you know the nearest library?", "assistant"),
        Msg("Alice", "Sorry, I don't know. Do you have any idea, Charlie?", "assistant"),
        Msg("Charlie", "No, let's ask Friday. Friday, get me the nearest library.", "assistant"),
        # 工具调用序列
        Msg(
            "Friday",
            [
                ToolUseBlock(
                    type="tool_use",
                    name="get_current_location",
                    id="1",
                    input={},
                ),
            ],
            "assistant",
        ),
        # 工具结果
        Msg(
            "system",
            [
                ToolResultBlock(
                    type="tool_result",
                    name="get_current_location",
                    id="1",
                    output=[TextBlock(type="text", text="104.48, 36.30")],
                ),
            ],
            "system",
        ),
    ]
    
    formatted_message = await formatter.format(input_msgs)
    print("The formatted message:")
    print(json.dumps(formatted_message, indent=4))
    return formatted_message
```

### 3. 工具系统 (Tool)

#### 工具函数架构

```python
# 工具函数模板
def tool_function(a: int, b: str) -> ToolResponse:
    """{function description}
    
    Args:
        a (int):
            {description of the first parameter}
        b (str):
            {description of the second parameter}
    """
    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=f"Processing {a} and {b}"
            )
        ]
    )
```

#### Toolkit管理系统

```python
# 自定义工具函数
async def my_search(query: str, api_key: str) -> ToolResponse:
    """简单的搜索工具函数示例
    
    Args:
        query (str):
            搜索查询
        api_key (str):
            API密钥用于认证
    """
    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=f"Searching for '{query}' with API key '{api_key}'",
            ),
        ],
    )

# 工具注册和管理
toolkit = Toolkit()
toolkit.register_tool_function(my_search)

# 获取工具JSON Schema
print("Tool JSON Schemas:")
print(json.dumps(toolkit.get_json_schemas(), indent=4, ensure_ascii=False))

# 预设参数注册
toolkit.register_tool_function(my_search, preset_kwargs={"api_key": "xxx"})
```

#### 工具执行机制

```python
# 工具调用执行
async def example_tool_execution() -> None:
    """工具执行示例"""
    res = await toolkit.call_tool_function(
        ToolUseBlock(
            type="tool_use",
            id="123",
            name="my_search",
            input={"query": "AgentScope"},
        ),
    )
    
    # 统一异步生成器处理
    print("Tool Response:")
    async for tool_response in res:
        print(tool_response)
```

### 4. 记忆系统 (Memory)

#### 基础记忆接口

```python
# 记忆系统基类方法
class MemoryBase:
    """记忆系统基类"""
    
    def add(self, msgs: List[Msg]) -> None:
        """添加消息到记忆"""
        pass
    
    def delete(self, index: int) -> None:
        """从记忆中删除项目"""
        pass
    
    def size(self) -> int:
        """记忆大小"""
        pass
    
    def clear(self) -> None:
        """清空记忆内容"""
        pass
    
    def get_memory(self) -> List[Msg]:
        """获取记忆内容作为Msg对象列表"""
        pass
    
    def state_dict(self) -> Dict:
        """获取记忆的状态字典"""
        pass
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """加载记忆的状态字典"""
        pass
```

### 5. 长期记忆系统 (Long-Term Memory)

#### Mem0集成架构

```python
# 长期记忆系统配置
long_term_memory = Mem0LongTermMemory(
    agent_name="Friday",
    user_name="user_123",
    model=DashScopeChatModel(
        model_name="qwen-max-latest",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        stream=False,
    ),
    embedding_model=DashScopeTextEmbedding(
        model_name="text-embedding-v2",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
    ),
    on_disk=False,
)
```

#### 记忆操作模式

```python
# 长期记忆操作模式
LONG_TERM_MEMORY_MODES = {
    "agent_control": {
        "description": "智能体自主管理长期记忆",
        "mechanism": "通过工具调用",
        "tools": ["record_to_memory", "retrieve_from_memory"]
    },
    "static_control": {
        "description": "开发者显式控制长期记忆操作",
        "mechanism": "手动调用记忆方法",
        "tools": []
    },
    "both": {
        "description": "激活两种记忆管理模式",
        "mechanism": "混合控制",
        "tools": ["record_to_memory", "retrieve_from_memory"]
    }
}
```

#### 记忆操作示例

```python
# 基础记忆操作
async def basic_usage():
    """基础使用示例"""
    # 记录记忆
    await long_term_memory.record(
        [Msg("user", "I like staying in homestays", "user")],
    )
    
    # 检索记忆
    results = await long_term_memory.retrieve(
        [Msg("user", "My accommodation preferences", "user")],
    )
    print(f"Retrieval results: {results}")
```

### 6. 智能体系统 (Agent)

#### ReActAgent核心特性

```python
# ReActAgent功能特性
REACT_AGENT_FEATURES = {
    "realtime_steering": {
        "description": "支持实时控制",
        "implementation": "基于asyncio取消机制",
        "method": "interrupt()和handle_interrupt()"
    },
    "parallel_tool_calls": {
        "description": "支持并行工具调用",
        "implementation": "asyncio.gather函数",
        "parameter": "parallel_tool_calls=True"
    },
    "structured_output": {
        "description": "支持结构化输出",
        "implementation": "Pydantic模型",
        "parameter": "structured_model"
    },
    "mcp_control": {
        "description": "细粒度MCP控制",
        "reference": "MCP章节"
    },
    "meta_tool": {
        "description": "智能体控制的工具管理",
        "reference": "Tool章节"
    },
    "long_term_memory": {
        "description": "自控制长期记忆",
        "reference": "Long-term Memory章节"
    },
    "state_management": {
        "description": "自动状态管理",
        "reference": "State章节"
    }
}
```

#### 实时控制机制

```python
# 中断处理机制
class AgentBase:
    async def __call__(self, *args: Any, **kwargs: Any) -> Msg:
        reply_msg: Msg | None = None
        try:
            self._reply_task = asyncio.current_task()
            reply_msg = await self.reply(*args, **kwargs)
        except asyncio.CancelledError:
            # 捕获中断并通过handle_interrupt方法处理
            reply_msg = await self.handle_interrupt(*args, **kwargs)
        return reply_msg
    
    @abstractmethod
    async def handle_interrupt(self, *args: Any, **kwargs: Any) -> Msg:
        """中断处理方法"""
        pass
```

#### 并行工具调用

```python
# 并行工具调用示例
def example_tool_function(tag: str) -> ToolResponse:
    """示例工具函数"""
    start_time = datetime.now().strftime("%H:%M:%S.%f")
    time.sleep(3)  # 模拟长时间运行的任务
    end_time = datetime.now().strftime("%H:%M:%S.%f")
    
    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=f"Tag {tag} started at {start_time} and ended at {end_time}. ",
            ),
        ],
    )

# 创建支持并行工具调用的ReAct智能体
agent = ReActAgent(
    name="Jarvis",
    sys_prompt="You are a helpful assistant.",
    model=DashScopeChatModel(
        model_name="qwen-max",
        api_key=os.environ["DASHSCOPE_API_KEY"],
    ),
    formatter=DashScopeChatFormatter(),
    toolkit=toolkit,
    parallel_tool_calls=True,  # 启用并行工具调用
)
```

### 7. 管道系统 (Pipeline)

#### MsgHub消息广播

```python
# 消息广播机制
async def example_broadcast_message():
    """消息广播示例"""
    # 创建消息中心
    async with MsgHub(
        participants=[alice, bob, charlie],
        announcement=Msg(
            "user",
            "Now introduce yourself in one sentence, including your name, age and career.",
            "user",
        ),
    ) as hub:
        # 无需手动消息传递的群聊
        await alice()
        await bob()
        await charlie()
```

#### 动态参与者管理

```python
# 动态参与者管理
async def dynamic_participant_management():
    """动态参与者管理示例"""
    async with MsgHub(participants=[alice]) as hub:
        # 添加新参与者
        hub.add(david)
        
        # 移除参与者
        hub.delete(alice)
        
        # 广播消息给所有当前参与者
        await hub.broadcast(
            Msg("system", "Now we begin to ...", "system"),
        )
```

#### 管道类型

```python
# 管道实现类型
PIPELINE_TYPES = {
    "Sequential Pipeline": {
        "description": "按预定义顺序逐一执行智能体",
        "implementation": ["sequential_pipeline", "SequentialPipeline"],
        "pattern": "链式执行"
    },
    "Fanout Pipeline": {
        "description": "将相同输入分发给多个智能体",
        "implementation": ["fanout_pipeline", "FanoutPipeline"],
        "pattern": "扇出执行"
    }
}
```

### 8. 状态/会话管理 (State/Session Management)

#### StateModule基础架构

```python
# StateModule核心方法
class StateModule:
    """状态模块基类"""
    
    def register_state(
        self, 
        attr_name: str,
        custom_to_json: Optional[Callable] = None,
        custom_from_json: Optional[Callable] = None
    ) -> None:
        """注册属性作为状态"""
        pass
    
    def state_dict(self) -> Dict:
        """获取当前对象的状态字典"""
        pass
    
    def load_state_dict(
        self, 
        state_dict: Dict, 
        strict: bool = True
    ) -> None:
        """加载状态字典到当前对象"""
        pass
```

#### 嵌套状态管理

```python
# 嵌套状态管理示例
class ClassA(StateModule):
    def __init__(self) -> None:
        super().__init__()
        self.cnt = 123
        # 注册cnt属性作为状态
        self.register_state("cnt")

class ClassB(StateModule):
    def __init__(self) -> None:
        super().__init__()
        # 属性"a"继承自StateModule
        self.a = ClassA()
        # 手动注册属性"c"作为状态
        self.c = "Hello, world!"
        self.register_state("c")

# 自动嵌套状态管理
obj_b = ClassB()
print("State of obj_b:")
print(json.dumps(obj_b.state_dict(), indent=4))
```

#### 会话管理系统

```python
# 会话管理基类
class SessionBase:
    """会话管理基类"""
    
    @abstractmethod
    def save_session_state(self, session_id: str) -> None:
        """保存会话状态"""
        pass
    
    @abstractmethod
    def load_session_state(self, session_id: str) -> None:
        """加载会话状态"""
        pass

# JSON会话实现
class JSONSession(SessionBase):
    """基于JSON的会话管理实现"""
    
    def save_session_state(self, session_id: str) -> None:
        """保存会话状态到JSON文件"""
        pass
    
    def load_session_state(self, session_id: str) -> None:
        """从JSON文件加载会话状态"""
        pass
```

## 核心实现分析

### 1. 统一接口设计

**设计原理**：
- 抽象基类定义标准接口
- 具体实现类提供特定功能
- 插件化架构支持扩展
- 配置驱动的灵活性

**实现特点**：
```python
# 统一模型接口示例
class ModelWrapperBase:
    def __call__(
        self, 
        messages: List[Dict], 
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None
    ) -> Union[ChatResponse, AsyncGenerator[ChatResponse, None]]:
        """统一的模型调用接口"""
        pass
```

### 2. 异步编程模式

**核心特性**：
- 基于asyncio的异步执行
- 并发任务管理
- 流式数据处理
- 中断和恢复机制

**实现模式**：
```python
# 异步执行模式
async def async_execution_pattern():
    # 并发执行多个任务
    results = await asyncio.gather(
        task1(),
        task2(),
        task3(),
        return_exceptions=True
    )
    
    # 流式数据处理
    async for chunk in stream_generator:
        process_chunk(chunk)
```

### 3. 状态管理机制

**设计原理**：
- 自动状态注册
- 嵌套状态序列化
- 会话级状态管理
- 状态恢复机制

**实现特点**：
```python
# 状态管理机制
class StateManagement:
    def __init__(self):
        self._state_registry = {}
        self._nested_states = []
    
    def auto_register_states(self):
        """自动注册StateModule属性"""
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, StateModule):
                self._nested_states.append(attr_name)
```

### 4. 工具生态系统

**架构特点**：
- 函数式工具定义
- 自动Schema提取
- 动态工具注册
- 并行工具执行

**实现机制**：
```python
# 工具生态系统
class ToolEcosystem:
    def extract_tool_schema(self, func: Callable) -> Dict:
        """从函数docstring提取工具Schema"""
        signature = inspect.signature(func)
        docstring = inspect.getdoc(func)
        
        # 解析参数和描述
        schema = self.parse_docstring_to_schema(docstring, signature)
        return schema
    
    async def execute_tool_parallel(self, tool_calls: List[ToolUseBlock]):
        """并行执行工具调用"""
        tasks = [self.call_tool_function(call) for call in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
```

## 最佳实践

### 1. 模型选择策略

**选择原则**：
- 根据任务类型选择合适的模型
- 考虑成本和性能平衡
- 评估特殊功能需求（视觉、推理等）
- 测试模型兼容性

**实施建议**：
```python
# 模型选择策略
MODEL_SELECTION_STRATEGY = {
    "text_generation": ["qwen-max", "gpt-4", "claude-3"],
    "code_generation": ["qwen-coder", "gpt-4", "claude-3"],
    "vision_tasks": ["qwen-vl", "gpt-4-vision", "gemini-pro-vision"],
    "reasoning_tasks": ["qwen-turbo", "claude-3", "gpt-4"]
}
```

### 2. 工具设计规范

**设计原则**：
- 单一职责原则
- 清晰的参数定义
- 完整的错误处理
- 异步执行支持

**实现规范**：
```python
# 工具设计规范
async def well_designed_tool(
    required_param: str,
    optional_param: Optional[int] = None
) -> ToolResponse:
    """设计良好的工具函数
    
    Args:
        required_param (str):
            必需参数的描述
        optional_param (Optional[int]):
            可选参数的描述
    
    Returns:
        ToolResponse: 工具执行结果
    
    Raises:
        ValueError: 参数验证失败时抛出
        RuntimeError: 执行过程中出现错误时抛出
    """
    try:
        # 参数验证
        if not required_param:
            raise ValueError("required_param cannot be empty")
        
        # 执行逻辑
        result = await perform_operation(required_param, optional_param)
        
        return ToolResponse(
            content=[
                TextBlock(type="text", text=str(result))
            ]
        )
    
    except Exception as e:
        return ToolResponse(
            content=[
                TextBlock(type="text", text=f"Error: {str(e)}")
            ],
            error=str(e)
        )
```

### 3. 记忆管理策略

**管理原则**：
- 短期记忆用于上下文维护
- 长期记忆用于知识积累
- 定期清理过期记忆
- 记忆检索优化

**实施策略**：
```python
# 记忆管理策略
class MemoryManagementStrategy:
    def __init__(self):
        self.short_term_limit = 100  # 短期记忆条目限制
        self.long_term_threshold = 0.8  # 长期记忆相关性阈值
    
    async def manage_memory(self, agent):
        """记忆管理策略"""
        # 短期记忆管理
        if agent.memory.size() > self.short_term_limit:
            await self.compress_short_term_memory(agent)
        
        # 长期记忆管理
        if hasattr(agent, 'long_term_memory'):
            await self.update_long_term_memory(agent)
    
    async def compress_short_term_memory(self, agent):
        """压缩短期记忆"""
        # 保留最近的重要消息
        important_msgs = self.extract_important_messages(
            agent.memory.get_memory()
        )
        agent.memory.clear()
        agent.memory.add(important_msgs)
```

### 4. 状态管理最佳实践

**管理原则**：
- 明确状态边界
- 实现状态版本控制
- 支持状态回滚
- 优化序列化性能

**实施方案**：
```python
# 状态管理最佳实践
class StateManagementBestPractice:
    def __init__(self):
        self.state_version = "1.0.0"
        self.state_history = []
    
    def save_checkpoint(self, agent):
        """保存状态检查点"""
        checkpoint = {
            "version": self.state_version,
            "timestamp": datetime.utcnow().isoformat(),
            "state": agent.state_dict()
        }
        self.state_history.append(checkpoint)
    
    def rollback_to_checkpoint(self, agent, checkpoint_index: int):
        """回滚到指定检查点"""
        if 0 <= checkpoint_index < len(self.state_history):
            checkpoint = self.state_history[checkpoint_index]
            agent.load_state_dict(checkpoint["state"])
            return True
        return False
```

## 技术特色

### 1. 模块化架构

- **高度解耦**：各模块独立开发和测试
- **插件化设计**：支持自定义扩展
- **标准化接口**：统一的API设计
- **配置驱动**：灵活的配置管理

### 2. 异步优先

- **高并发支持**：基于asyncio的异步编程
- **流式处理**：支持实时数据流
- **中断机制**：支持任务中断和恢复
- **资源优化**：高效的资源利用

### 3. 状态感知

- **自动状态管理**：无需手动状态维护
- **嵌套状态支持**：复杂对象状态管理
- **状态持久化**：支持状态保存和恢复
- **版本控制**：状态版本管理

### 4. 工具生态

- **丰富的内置工具**：覆盖常用功能
- **自动Schema提取**：从函数自动生成工具描述
- **并行执行**：支持工具并行调用
- **动态扩展**：运行时动态添加工具

## 总结与展望

AgentScope的Task Guides提供了完整的开发指南和最佳实践，具有以下优势：

1. **架构完整性**：覆盖了智能体开发的各个方面
2. **实现优雅性**：基于现代Python特性的优雅实现
3. **扩展灵活性**：支持自定义扩展和插件开发
4. **工程实用性**：提供了丰富的示例和最佳实践
5. **性能优化**：基于异步编程的高性能实现

该指南系统为构建复杂的智能体应用提供了坚实的基础，是AgentScope框架的重要组成部分。通过深入理解其设计原理和实现机制，开发者可以构建出高效、可靠、可扩展的智能体应用。

## 参考资料

- AgentScope官方文档
- Python asyncio官方文档
- Pydantic官方文档
- OpenAI API文档
- Anthropic API文档
- Google Gemini API文档
- Mem0长期记忆库文档