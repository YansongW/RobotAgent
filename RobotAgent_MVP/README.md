# RobotAgent MVP

基于CAMEL框架的智能体机器人系统MVP版本

## 架构概述

本项目采用**分层智能体架构**，融合了CAMEL、Eigent和OWL项目的优势，构建了一个可扩展、状态化、可进化的智能体系统。

### 核心设计理念

#### 🧬 可进化性 (Evolvability)
- 智能体通过与环境交互持续学习和改进
- 支持强化学习和监督学习驱动的自我优化
- 多层次记忆系统支持经验积累和知识迁移

#### 📈 可扩展性 (Scalability) 
- 异步消息传递机制支持大规模智能体协作
- 模块化设计便于功能扩展和性能优化
- 支持分布式部署和负载均衡

#### 💾 状态性 (Statefulness)
- 完整的状态管理系统跟踪智能体生命周期
- 多层次记忆架构维护上下文连续性
- 支持复杂多步骤任务的执行和恢复

#### 🔧 工具集成 (Tool Integration)
- 动态工具注册和权限管理
- 支持MCP协议和自定义工具扩展
- 工具调用链追踪和错误处理

## 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    RobotAgent MVP 系统                      │
├─────────────────────────────────────────────────────────────┤
│  输入层 (Input Layer)                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ 语音输入     │  │ 文本输入     │  │ 传感器数据   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  智能体层 (Agent Layer)                                     │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              BaseRobotAgent (抽象基类)                  ││
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐      ││
│  │  │ ChatAgent   │ │ ActionAgent │ │ MemoryAgent │      ││
│  │  │ 对话处理     │ │ 动作规划     │ │ 记忆管理     │      ││
│  │  └─────────────┘ └─────────────┘ └─────────────┘      ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│  通信层 (Communication Layer)                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ 消息路由     │  │ 协作协调     │  │ 状态同步     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  工具层 (Tool Layer)                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ 内置工具     │  │ 外部API     │  │ 硬件接口     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  输出层 (Output Layer)                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ 语音合成     │  │ 动作指令     │  │ 状态反馈     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件详解

#### 1. BaseRobotAgent (智能体基类)

**设计模式**: 抽象工厂模式 + 观察者模式 + 状态机模式

**核心功能**:
- **状态管理**: 8种智能体状态的完整生命周期管理
- **消息系统**: 10种消息类型的标准化通信协议
- **工具集成**: 动态工具注册、权限控制、调用链追踪
- **协作机制**: 4种协作模式支持多智能体协调
- **记忆系统**: 短期、长期、情节、语义四层记忆架构
- **学习能力**: 交互式学习和经验积累机制

**技术实现**:
```python
# 状态枚举
class AgentState(Enum):
    INITIALIZING = "initializing"  # 初始化中
    IDLE = "idle"                  # 空闲状态
    PROCESSING = "processing"      # 处理消息中
    EXECUTING = "executing"        # 执行任务中
    COLLABORATING = "collaborating" # 协作中
    LEARNING = "learning"          # 学习中
    ERROR = "error"                # 错误状态
    SHUTDOWN = "shutdown"          # 关闭状态

# 消息类型
class MessageType(Enum):
    TEXT = "text"                  # 文本消息
    TASK = "task"                  # 任务分配
    INSTRUCTION = "instruction"    # 指令消息
    RESPONSE = "response"          # 响应消息
    TOOL_CALL = "tool_call"        # 工具调用
    TOOL_RESULT = "tool_result"    # 工具结果
    STATUS = "status"              # 状态更新
    ERROR = "error"                # 错误报告
    HEARTBEAT = "heartbeat"        # 心跳检测
    COLLABORATION = "collaboration" # 协作请求
```

#### 2. 专业化智能体

**ChatAgent (对话智能体)**:
- 继承BaseRobotAgent，专注于自然语言处理
- 集成CAMEL的对话生成能力
- 支持多轮对话上下文管理
- 实现情感分析和意图识别

**ActionAgent (动作智能体)**:
- 继承BaseRobotAgent，专注于任务规划和执行
- 实现分层任务分解算法
- 支持动作序列优化和执行监控
- 集成机器人控制接口

**MemoryAgent (记忆智能体)**:
- 继承BaseRobotAgent，专注于知识管理
- 实现向量化记忆存储和检索
- 支持知识图谱构建和推理
- 提供个性化学习和适应能力

## 项目结构

```
RobotAgent_MVP/
├── src/
│   ├── agents/                    # 智能体模块
│   │   ├── __init__.py
│   │   ├── base_agent.py          # 智能体基类 (已完成)
│   │   ├── chat_agent.py          # 对话智能体 (待实现)
│   │   ├── action_agent.py        # 动作智能体 (待实现)
│   │   └── memory_agent.py        # 记忆智能体 (待实现)
│   ├── communication/             # 通信模块
│   │   ├── __init__.py
│   │   ├── message_router.py      # 消息路由 (待实现)
│   │   ├── collaboration.py       # 协作协调 (待实现)
│   │   └── protocols.py           # 通信协议 (待实现)
│   ├── memory/                    # 记忆模块
│   │   ├── __init__.py
│   │   ├── memory_manager.py      # 记忆管理器 (待实现)
│   │   ├── vector_store.py        # 向量存储 (待实现)
│   │   └── knowledge_graph.py     # 知识图谱 (待实现)
│   ├── output/                    # 输出模块
│   │   ├── __init__.py
│   │   ├── tts_engine.py          # 语音合成 (待实现)
│   │   ├── action_executor.py     # 动作执行 (待实现)
│   │   └── response_formatter.py  # 响应格式化 (待实现)
│   ├── utils/                     # 工具模块
│   │   ├── __init__.py
│   │   ├── config_loader.py       # 配置加载 (待实现)
│   │   ├── logger.py              # 日志系统 (待实现)
│   │   └── validators.py          # 数据验证 (待实现)
│   └── main.py                    # 主程序入口 (待实现)
├── config/                        # 配置文件
│   ├── agents_config.yaml         # 智能体配置
│   ├── system_config.yaml         # 系统配置
│   ├── chat_agent_prompt_template.json # 对话模板
│   └── api_config.yaml.template   # API配置模板
├── tests/                         # 测试代码
│   ├── __init__.py
│   ├── test_agents.py             # 智能体测试
│   ├── test_communication.py      # 通信测试
│   └── volcengine_chat_test.py    # 火山引擎测试
├── requirements.txt               # 依赖包
├── demo.py                        # 演示程序
└── README.md                      # 说明文档
```

## 技术实现细节

### 1. 异步消息处理机制

```python
# 消息处理流程
async def _message_processing_loop(self):
    """异步消息处理循环"""
    while self._running:
        try:
            if self._message_queue:
                message = self._message_queue.popleft()
                await self._process_message(message)
            else:
                await asyncio.sleep(0.01)  # 避免CPU占用过高
        except Exception as e:
            self.logger.error(f"消息处理错误: {e}")
            await self._set_state(AgentState.ERROR)
```

### 2. 工具集成架构

```python
# 工具定义数据结构
@dataclass
class ToolDefinition:
    name: str                          # 工具名称
    description: str                   # 工具描述
    function: Callable                 # 工具函数
    parameters_schema: Dict[str, Any]  # 参数模式
    return_schema: Dict[str, Any]      # 返回模式
    enabled: bool = True               # 是否启用
    permissions: List[str] = field(default_factory=list)  # 权限列表
    category: str = "general"          # 工具分类
```

### 3. 多层记忆系统

```python
# 记忆系统架构
class MemorySystem:
    def __init__(self):
        self._short_term_memory = deque(maxlen=100)    # 短期记忆
        self._long_term_memory = {}                    # 长期记忆
        self._episodic_memory = []                     # 情节记忆
        self._semantic_memory = {}                     # 语义记忆
        self._conversation_memory = {}                 # 对话记忆
```

### 4. 协作模式实现

```python
# 协作模式枚举
class CollaborationMode(Enum):
    ROLE_PLAYING = "role_playing"      # 角色扮演模式
    PEER_TO_PEER = "peer_to_peer"      # 对等协作模式
    HIERARCHICAL = "hierarchical"      # 层次化协作模式
    SOCIETY = "society"                # 智能体社会模式
```

## 开发路线图

### Phase 1: 核心智能体实现 (当前阶段)
- [x] BaseRobotAgent基类设计和实现
- [ ] ChatAgent对话智能体实现
- [ ] ActionAgent动作智能体实现
- [ ] MemoryAgent记忆智能体实现

### Phase 2: 通信和协作系统
- [ ] 消息路由系统实现
- [ ] 智能体协作协调机制
- [ ] 分布式通信协议
- [ ] 负载均衡和容错处理

### Phase 3: 高级功能集成
- [ ] 多模态输入处理
- [ ] 语音合成和识别
- [ ] 机器人硬件接口
- [ ] 知识图谱集成

### Phase 4: 优化和部署
- [ ] 性能优化和监控
- [ ] 容器化部署
- [ ] 云原生架构适配
- [ ] 生产环境测试

## 快速开始

### 环境准备

1. **Python环境**: Python 3.8+
2. **依赖安装**:
```bash
pip install -r requirements.txt
```

3. **配置文件**:
```bash
# 复制API配置模板
cp config/api_config.yaml.template config/api_config.yaml
# 编辑配置文件，填入你的API密钥
```

### 运行系统

```bash
# 运行基础测试
python src/agents/base_agent.py

# 运行完整系统 (开发中)
python src/main.py

# 运行演示程序 (开发中)
python demo.py
```

### 测试验证

```bash
# 运行单元测试
python -m pytest tests/ -v

# 运行特定测试
python -m pytest tests/test_agents.py -v
```

## 核心特性

### ✨ 已实现功能
- **完整的智能体基类**: 包含状态管理、消息处理、工具集成等核心功能
- **异步消息系统**: 支持10种消息类型的标准化通信
- **动态工具注册**: 支持运行时工具添加和权限管理
- **多层记忆架构**: 短期、长期、情节、语义四层记忆系统
- **协作机制**: 支持4种协作模式的多智能体协调
- **学习能力**: 交互式学习和经验积累机制

### 🚧 开发中功能
- **专业化智能体**: ChatAgent、ActionAgent、MemoryAgent实现
- **通信路由系统**: 智能体间消息路由和协调
- **输出处理模块**: TTS语音合成、动作执行、响应格式化
- **配置管理系统**: 动态配置加载和热更新
- **监控和日志**: 完整的系统监控和日志记录

### 🎯 规划中功能
- **多模态处理**: 语音、图像、传感器数据集成
- **知识图谱**: 结构化知识存储和推理
- **分布式部署**: 云原生架构和容器化部署
- **可视化界面**: Web界面和实时监控面板
- **插件系统**: 第三方插件和扩展支持

## 贡献指南

我们欢迎社区贡献！请参考以下指南：

1. **代码规范**: 遵循PEP 8 Python代码规范
2. **测试要求**: 新功能需要包含相应的单元测试
3. **文档更新**: 重要功能需要更新相关文档
4. **提交格式**: 使用清晰的commit message描述变更

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系我们

- **项目主页**: [GitHub Repository]
- **问题反馈**: [GitHub Issues]
- **技术讨论**: [GitHub Discussions]

---

*RobotAgent MVP - 构建下一代智能机器人系统* 🤖✨
