# RobotAgent MVP

基于CAMEL框架的智能体机器人系统MVP版本

## 架构概述

本项目采用**三智能体协作架构**，融合了CAMEL、Eigent和OWL项目的优势，构建了一个可扩展、状态化、可进化的智能体系统。系统由ChatAgent（对话协调）、ActionAgent（任务执行）、MemoryAgent（记忆管理）三个专业化智能体组成，通过标准化的消息传递机制实现高效协作。

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

### 三智能体协作架构图

```
┌─────────────────────────────────────────────────────────────┐
│                 RobotAgent MVP 三智能体系统                  │
├─────────────────────────────────────────────────────────────┤
│  用户交互层 (User Interface Layer)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ 语音输入     │  │ 文本输入     │  │ 多模态输入   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  智能体协作层 (Agent Collaboration Layer)                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                BaseRobotAgent (抽象基类)                ││
│  │                                                         ││
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ ││
│  │  │ ChatAgent   │◄──►│ ActionAgent │◄──►│ MemoryAgent │ ││
│  │  │ 对话协调     │    │ 任务执行     │    │ 记忆管理     │ ││
│  │  │ • 意图识别   │    │ • 任务分解   │    │ • 知识存储   │ ││
│  │  │ • 上下文管理 │    │ • 动作规划   │    │ • 向量检索   │ ││
│  │  │ • 智能体调度 │    │ • 工具调用   │    │ • 知识图谱   │ ││
│  │  └─────────────┘    └─────────────┘    └─────────────┘ ││
│  │           │               │                    │       ││
│  │           └───────────────┼────────────────────┘       ││
│  │                          │                            ││
│  │                   协作消息传递                         ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│  数据流处理层 (Data Flow Layer)                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ 消息路由     │  │ 状态同步     │  │ 结果整合     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  工具执行层 (Tool Execution Layer)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ CAMEL工具   │  │ 外部API     │  │ 硬件接口     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  输出响应层 (Output Response Layer)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ 自然语言响应 │  │ 动作指令     │  │ 状态反馈     │        │
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

#### 2. 三智能体协作系统

**数据流设计**:

```
用户输入 → ChatAgent → ActionAgent → MemoryAgent
    ↑                ↓         ↓         ↓
    └─── 响应整合 ←── 协作消息传递 ←── 知识检索
```

### 智能体协作工作流

#### 标准协作流程

```
1. 用户输入 → ChatAgent
   ├─ 意图识别和分析
   ├─ 上下文理解
   └─ 决策路由

2. ChatAgent → ActionAgent (任务执行场景)
   ├─ 发送任务描述和参数
   ├─ ActionAgent执行任务分解
   ├─ 返回执行计划和结果
   └─ ChatAgent整合响应

3. ChatAgent → MemoryAgent (知识查询场景)
   ├─ 发送查询请求
   ├─ MemoryAgent检索相关记忆
   ├─ 返回知识和上下文
   └─ ChatAgent生成回复

4. ActionAgent ↔ MemoryAgent (执行过程中)
   ├─ ActionAgent查询执行经验
   ├─ MemoryAgent提供历史数据
   ├─ ActionAgent存储执行结果
   └─ MemoryAgent更新知识库
```

#### 消息传递协议

**标准消息格式**：
```json
{
  "message_id": "msg_uuid",
  "sender_agent": "ChatAgent|ActionAgent|MemoryAgent",
  "receiver_agent": "ChatAgent|ActionAgent|MemoryAgent",
  "message_type": "TASK|QUERY|RESPONSE|STATUS",
  "timestamp": "2024-01-01T10:00:00Z",
  "payload": {
    "content": "消息内容",
    "metadata": {"priority": "HIGH|MEDIUM|LOW"}
  },
  "correlation_id": "关联消息ID"
}
```

**协作模式**：
- **直接协作**：ChatAgent直接调用其他智能体
- **链式协作**：ChatAgent → ActionAgent → MemoryAgent
- **并行协作**：ChatAgent同时查询ActionAgent和MemoryAgent
- **反馈协作**：智能体间的双向信息交换

**ChatAgent (对话协调智能体)**:

**核心职责**：作为系统的对话入口和协调中心

**主要功能**：
- **意图识别与分析**：解析用户输入，识别任务类型和执行意图
- **对话上下文管理**：维护多轮对话状态和会话历史
- **智能体调度协调**：根据任务需求决定是否调用ActionAgent或MemoryAgent
- **响应整合与生成**：整合各智能体的输出，生成最终用户响应
- **情感分析与适应**：分析用户情感状态，调整交互策略

**输出格式**：
```json
{
  "intent_type": "TASK_EXECUTION|INFORMATION_QUERY|CONVERSATION",
  "confidence": 0.95,
  "context_summary": "对话上下文摘要",
  "next_action": "CALL_ACTION_AGENT|CALL_MEMORY_AGENT|DIRECT_RESPONSE",
  "parameters": {"task_description": "具体任务描述"}
}
```

**ActionAgent (任务执行智能体)**:

**核心职责**：负责复杂任务的分解、规划和执行

**主要功能**：
- **任务分解与建模**：将复杂任务分解为可执行的子任务序列
- **动作序列规划**：制定最优的执行路径和策略
- **工具调用与集成**：协调CAMEL工具、外部API和硬件接口
- **执行状态监控**：实时跟踪任务执行进度和状态
- **异常处理与恢复**：处理执行过程中的错误和异常情况
- **结果验证与反馈**：验证执行结果的正确性和完整性

**输出格式**：
```json
{
  "task_tree": {
    "root_task": "主任务描述",
    "subtasks": [
      {
        "id": "subtask_1",
        "description": "子任务描述",
        "dependencies": [],
        "tools_required": ["tool_name"],
        "estimated_time": "2min",
        "status": "PENDING|IN_PROGRESS|COMPLETED|FAILED"
      }
    ]
  },
  "execution_plan": {
    "sequence": ["subtask_1", "subtask_2"],
    "parallel_groups": [["subtask_3", "subtask_4"]]
  },
  "execution_result": {
    "status": "SUCCESS|PARTIAL|FAILED",
    "completed_tasks": ["subtask_1"],
    "failed_tasks": [],
    "output_data": "执行结果数据"
  }
}
```

**MemoryAgent (记忆管理智能体)**:

**核心职责**：管理系统的多层记忆和知识图谱

**主要功能**：
- **多模态记忆存储**：存储文本、图像、音频等多种形式的记忆数据
- **向量化知识检索**：基于语义相似度的高效知识检索
- **知识图谱构建**：构建和维护实体关系网络
- **经验学习与积累**：从历史交互中提取和存储经验模式
- **上下文关联分析**：分析当前对话与历史记忆的关联性
- **记忆优化与清理**：定期优化记忆结构，清理冗余信息

**记忆层次结构**：
- **工作记忆**：当前会话的临时信息（容量限制：50条消息）
- **短期记忆**：近期会话历史（保留时间：7天）
- **长期记忆**：重要经验和知识（永久存储）
- **知识图谱**：结构化的实体关系网络

**输出格式**：
```json
{
  "memory_query_result": {
    "relevant_memories": [
      {
        "memory_id": "mem_001",
        "content": "相关记忆内容",
        "similarity_score": 0.89,
        "memory_type": "EXPERIENCE|KNOWLEDGE|CONTEXT",
        "timestamp": "2024-01-01T10:00:00Z"
      }
    ],
    "knowledge_graph_entities": [
      {
        "entity": "实体名称",
        "relations": [{"relation": "关系类型", "target": "目标实体"}]
      }
    ]
  },
  "memory_storage_result": {
    "stored_memory_id": "mem_002",
    "storage_type": "WORKING|SHORT_TERM|LONG_TERM",
    "indexed_keywords": ["关键词1", "关键词2"]
  }
}
```

## 项目结构

```
RobotAgent_MVP/
├── src/
│   ├── agents/                    # 智能体模块
│   │   ├── __init__.py
│   │   ├── base_agent.py          # 智能体基类 (已完成)
│   │   ├── chat_agent.py          # 对话智能体 (已完成)
│   │   ├── action_agent.py        # 动作智能体 (已完成)
│   │   ├── memory_agent.py        # 记忆智能体 (已完成)
│   │   └── agent_coordinator.py   # 智能体协调器 (已完成)
│   ├── communication/             # 通信模块
│   │   ├── __init__.py
│   │   ├── protocols.py           # 通信协议 (已完成)
│   │   └── message_bus.py         # 消息总线 (已完成)
│   ├── memory/                    # 记忆模块
│   │   ├── __init__.py
│   │   ├── simple_memory.py       # 简单记忆系统 (已完成)
│   │   └── conversation_history.py # 对话历史 (已完成)
│   ├── output/                    # 输出模块
│   │   ├── __init__.py
│   │   ├── tts_handler.py         # 语音合成处理 (已完成)
│   │   └── action_file_generator.py # 动作文件生成 (已完成)
│   ├── utils/                     # 工具模块
│   │   ├── __init__.py
│   │   ├── config_loader.py       # 配置加载 (已完成)
│   │   ├── logger.py              # 日志系统 (已完成)
│   │   ├── config.py              # 配置管理 (已完成)
│   │   └── message_types.py       # 消息类型定义 (已完成)
│   └── main.py                    # 主程序入口 (基础框架)
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

## 实现计划与开发步骤

### Phase 1: 三智能体核心架构实现

**Week 1-2: ActionAgent实现** ✅ 已完成
- [x] 创建ActionAgent类，继承BaseRobotAgent
- [x] 实现任务分解算法和任务树数据结构
- [x] 开发动作序列规划引擎
- [x] 集成CAMEL工具调用接口
- [x] 实现执行状态监控和异常处理

**Week 3-4: MemoryAgent实现** ✅ 已完成
- [x] 创建MemoryAgent类，继承BaseRobotAgent
- [x] 实现多层记忆存储系统（工作/短期/长期记忆）
- [x] 开发向量化知识检索引擎
- [x] 构建知识图谱存储和查询机制
- [x] 实现记忆优化和清理算法

**Week 5-6: ChatAgent协调功能增强** ✅ 已完成
- [x] 扩展ChatAgent的智能体调度功能
- [x] 实现意图识别和任务路由逻辑
- [x] 开发响应整合和生成机制
- [x] 完善情感分析和上下文管理

### Phase 2: 智能体协作系统

**Week 7-8: 消息传递协议** ✅ 已完成
- [x] 实现标准化消息格式和传递机制
- [x] 开发智能体间的异步通信系统
- [x] 构建消息路由和状态同步机制
- [x] 实现协作模式（直接/链式/并行/反馈）

**Week 9-10: 协作工作流** ✅ 已完成
- [x] 实现ChatAgent → ActionAgent任务执行流程
- [x] 开发ChatAgent → MemoryAgent知识查询流程
- [x] 构建ActionAgent ↔ MemoryAgent协作机制
- [x] 完善错误处理和恢复策略

### Phase 3: 高级功能与优化

**Week 11-12: 多模态处理**
- [ ] 扩展MemoryAgent支持图像、音频记忆
- [ ] 实现跨模态知识检索
- [ ] 开发多模态任务分解能力

**Week 13-14: 学习与适应**
- [ ] 实现经验学习和模式识别
- [ ] 开发个性化适应机制
- [ ] 构建知识图谱自动更新

**Week 15: 性能优化**
- [ ] 优化消息传递性能
- [ ] 改进记忆检索效率
- [ ] 调优任务执行速度

### Phase 4: 测试与部署 

**Week 16-17: 系统测试**
- [ ] 单元测试（各智能体功能）
- [ ] 集成测试（智能体协作）
- [ ] 性能测试（并发和负载）
- [ ] 用户验收测试

**Week 18: 部署准备**
- [ ] 文档完善和API文档
- [ ] 配置管理和环境部署
- [ ] 监控和日志系统
- [ ] 用户界面优化

### 关键里程碑

- **里程碑1** (Week 6): 三智能体基础功能完成 ✅
- **里程碑2** (Week 10): 智能体协作系统可用 ✅
- **里程碑3** (Week 15): 高级功能集成完成 🚧
- **里程碑4** (Week 18): 系统部署就绪 📋

## 核心特性

### ✨ 已实现功能
- **完整的智能体基类**: 包含状态管理、消息处理、工具集成等核心功能
- **专业化智能体**: ChatAgent、ActionAgent、MemoryAgent完整实现
- **智能体协调器**: AgentCoordinator智能体协调机制
- **异步消息系统**: 支持10种消息类型的标准化通信
- **通信路由系统**: 智能体间消息路由和协调
- **动态工具注册**: 支持运行时工具添加和权限管理
- **多层记忆架构**: 短期、长期、情节、语义四层记忆系统
- **协作机制**: 支持4种协作模式的多智能体协调
- **学习能力**: 交互式学习和经验积累机制
- **输出处理模块**: TTS语音合成、动作文件生成器
- **配置管理系统**: 动态配置加载和热更新
- **监控和日志**: 完整的系统监控和日志记录

### 🚧 开发中功能
- **主程序入口**: 系统集成和启动流程
- **演示程序**: 完整的功能演示脚本
- **集成测试**: 端到端测试和验证

### 🎯 规划中功能
- **多模态处理**: 语音、图像、传感器数据集成
- **知识图谱**: 结构化知识存储和推理
- **分布式部署**: 云原生架构和容器化部署
- **可视化界面**: Web界面和实时监控面板
- **插件系统**: 第三方插件和扩展支持
