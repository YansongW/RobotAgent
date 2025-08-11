# RobotAgent - CAMEL-ROS2智能机器人系统 (MVP开发阶段)

## ⚠️ 项目状态

**当前阶段**: MVP (最小可行产品) 开发阶段

本项目目前处于早期开发阶段，正在进行基础架构设计和核心组件的原型开发。大部分功能仍在设计和开发中。

## 项目愿景

RobotAgent计划成为一个基于CAMEL.AI多智能体框架和ROS2机器人操作系统的智能机器人控制系统。采用"大脑（CAMEL认知）+ 小脑（ROS2控制）"的创新架构设计，通过多个专门的CAMEL智能体协同工作，实现自然语言交互的智能机器人控制。

### 设计理念 (规划中)
- **🧠 大脑-小脑架构**: CAMEL智能体负责认知决策，ROS2负责精确控制
- **🤝 多智能体协作**: 多个专门智能体分工合作，各司其职
- **🧠 多模态记忆**: 集成向量数据库和知识图谱的GraphRAG系统
- **⚡ 实时通信**: 基于Redis消息总线的高效智能体通信

## 当前已完成的功能

### ✅ 基础配置系统
- 项目目录结构搭建
- 基础配置文件管理
- API密钥安全管理（配置文件化）

### ✅ 火山方舟Chat API集成
- 火山方舟Chat API测试工具
- 配置文件化的API密钥管理
- 多轮对话支持
- 智能助手prompt模板加载

### ✅ 开发工具
- 配置加载器工具类
- API配置模板系统
- 安全的.gitignore配置

## 正在开发中的功能

### 🚧 CAMEL智能体系统 (设计阶段)
- Dialog Agent (对话智能体)
- Planning Agent (规划智能体) 
- Decision Agent (决策智能体)
- Perception Agent (感知智能体)
- Learning Agent (学习智能体)
- ROS2 Agent (机器人控制智能体)

### 🚧 多模态记忆系统 (设计阶段)
- 向量数据库集成 (Milvus)
- 知识图谱系统 (Neo4j)
- GraphRAG检索增强生成
- 记忆分层存储策略

### 🚧 ROS2集成 (设计阶段)
- ROS2接口模块
- 机器人控制接口
- 安全监控系统

## 当前可用的工具

### 火山方舟Chat API测试工具

位置: `RobotAgent_MVP/tests/volcengine_chat_test.py`

这是一个完整的火山方舟Chat API测试工具，支持：
- 配置文件管理API密钥
- 多轮对话
- 智能助手prompt模板
- 命令行交互界面

使用方法请参考: [火山方舟Chat API测试工具说明](RobotAgent_MVP/tests/README_volcengine_chat.md)

## 📚 文档导航

### 当前可用文档
- **[项目目录结构](PROJECT_STRUCTURE.md)** - 完整的项目文件组织说明
- **[火山方舟Chat API测试工具说明](RobotAgent_MVP/tests/README_volcengine_chat.md)** - 当前可用工具的使用指南
- **[API配置迁移总结](RobotAgent_MVP/docs/API_CONFIG_MIGRATION.md)** - API密钥配置文件化改造说明

### 设计文档 (规划阶段)
> ⚠️ 以下文档描述的是项目的设计愿景，大部分功能尚未实现

- **[CAMEL-ROS2架构设计](docs/camel_ros2_architecture.md)** - CAMEL与ROS2集成的架构设计
- **[多模态记忆系统](docs/multimodal_memory_system.md)** - GraphRAG和向量数据库设计
- **[代码架构说明](docs/code_architecture.md)** - 详细的代码结构设计
- **[CAMEL智能体实现](docs/camel_agent_implementation.md)** - 各智能体的设计方案
- **[系统配置与部署](docs/system_configuration_deployment.md)** - 系统配置和部署设计
- **[API接口与开发指南](docs/api_development_guide.md)** - API接口设计文档

## 当前项目结构

```
RobotAgent/
├── README.md                           # 项目说明文档 (已更新)
├── PROJECT_STRUCTURE.md               # 项目结构说明
├── QUICK_START.md                      # 快速开始指南 (需要更新)
├── DOCUMENTATION_INDEX.md             # 文档索引
├── .gitignore                          # Git忽略文件配置
├── docs/                               # 设计文档目录
│   ├── camel_ros2_architecture.md      # CAMEL-ROS2架构设计 (设计阶段)
│   ├── multimodal_memory_system.md     # 多模态记忆系统设计 (设计阶段)
│   ├── camel_agent_implementation.md   # CAMEL智能体设计 (设计阶段)
│   ├── system_configuration_deployment.md # 系统配置设计 (设计阶段)
│   ├── api_development_guide.md        # API接口设计 (设计阶段)
│   └── code_architecture.md            # 代码架构设计 (设计阶段)
└── RobotAgent_MVP/                     # MVP开发目录 ✅
    ├── README.md                       # MVP说明文档
    ├── requirements.txt                # Python依赖 ✅
    ├── api_config.yaml.template        # API配置模板 ✅
    ├── .gitignore                      # Git忽略配置 ✅
    ├── config/                         # 配置文件目录 ✅
    │   ├── __init__.py
    │   ├── api_config.yaml             # API配置文件 ✅
    │   ├── api_config.yaml.template    # API配置模板 ✅
    │   ├── agents_config.yaml          # 智能体配置 (基础)
    │   ├── chat_agent_prompt_template.json # 聊天智能体提示模板 ✅
    │   └── system_config.yaml          # 系统配置 (基础)
    ├── src/                            # 源代码目录
    │   ├── __init__.py
    │   ├── main.py                     # 主程序入口 (基础)
    │   ├── agents/                     # 智能体模块 (基础结构)
    │   ├── communication/              # 通信模块 (基础结构)
    │   ├── memory/                     # 记忆系统 (基础结构)
    │   ├── output/                     # 输出模块 (基础结构)
    │   └── utils/                      # 工具模块 ✅
    │       ├── __init__.py
    │       └── config_loader.py        # 配置加载器 ✅
    ├── tests/                          # 测试代码目录 ✅
    │   ├── __init__.py
    │   ├── volcengine_chat_test.py     # 火山方舟Chat API测试工具 ✅
    │   ├── README_volcengine_chat.md   # 测试工具说明文档 ✅
    │   ├── test_agents.py              # 智能体测试 (基础)
    │   └── test_communication.py       # 通信测试 (基础)
    ├── docs/                           # MVP文档目录 ✅
    │   └── API_CONFIG_MIGRATION.md     # API配置迁移说明 ✅
    └── demo.py                         # 演示程序 (基础)
```

### 图例说明
- ✅ **已完成**: 功能完整，可正常使用
- **(基础)**: 基础文件结构已创建，功能待开发
- **(设计阶段)**: 设计文档，对应功能尚未实现

## 开发进度

### ✅ 已完成的组件

#### 1. 配置管理系统
- **配置加载器**: 安全的API密钥和系统配置管理
- **配置模板系统**: 便于团队协作的配置模板
- **环境隔离**: 开发/生产环境配置分离

#### 2. 火山方舟Chat API集成
- **API客户端**: 基于OpenAI SDK的火山方舟API调用
- **多轮对话**: 支持上下文记忆的对话系统
- **Prompt模板**: 智能助手行为配置和模板加载
- **交互界面**: 命令行对话测试工具

### 🚧 规划中的组件 (设计阶段)

#### 1. CAMEL智能体系统
- **对话智能体 (Dialog Agent)**: 自然语言理解和对话管理
- **规划智能体 (Planning Agent)**: 任务规划和路径规划
- **决策智能体 (Decision Agent)**: 智能决策和动作选择
- **感知智能体 (Perception Agent)**: 多模态感知和环境理解
- **学习智能体 (Learning Agent)**: 经验学习和知识更新
- **ROS2智能体 (ROS2 Agent)**: 机器人控制和状态管理

#### 2. 多模态记忆系统
- **LangGraph工作流引擎**: 智能记忆处理和分析工作流
- **记忆分类系统**: 智能体记忆、任务经验、领域知识三大类别
- **向量数据库 (Milvus)**: 多模态嵌入存储和语义检索
- **知识图谱 (Neo4j)**: 结构化知识表示和关系推理
- **GraphRAG**: 图增强检索生成系统
- **分层存储**: 热/温/冷/归档四层存储策略

#### 3. ROS2接口模块
- **话题通信**: 标准ROS2话题发布和订阅
- **服务调用**: ROS2服务的同步通信
- **动作执行**: 长时间任务的异步执行
- **安全监控**: 实时安全状态监控

#### 4. 通信架构
- **消息总线**: 基于Redis的智能体间通信
- **REST API**: 标准化的外部接口
- **WebSocket**: 实时数据推送
- **安全机制**: 认证、授权、监控

## 快速体验

### 火山方舟Chat API测试工具

1. **安装依赖**:
   ```bash
   cd RobotAgent_MVP
   pip install -r requirements.txt
   ```

2. **配置API密钥**:
   ```bash
   cp config/api_config.yaml.template config/api_config.yaml
   # 编辑 config/api_config.yaml，填入您的火山方舟API密钥
   ```

3. **运行测试工具**:
   ```bash
   python tests/volcengine_chat_test.py
   ```

详细使用说明请参考: [火山方舟Chat API测试工具说明](RobotAgent_MVP/tests/README_volcengine_chat.md)

## 开发计划

### 近期目标 (MVP阶段)
- [ ] 完善基础智能体框架
- [ ] 实现简单的多智能体通信
- [ ] 集成基础的记忆系统
- [ ] 添加更多API服务支持

### 中期目标
- [ ] 实现完整的CAMEL智能体系统
- [ ] 集成ROS2接口
- [ ] 开发多模态记忆系统
- [ ] 构建Web管理界面

### 长期目标
- [ ] 完整的机器人控制系统
- [ ] 分布式部署支持
- [ ] 生产环境优化
- [ ] 社区生态建设

## 贡献指南

项目目前处于早期开发阶段，欢迎各种形式的贡献：

- 🐛 **Bug报告**: 发现问题请提交Issue
- 💡 **功能建议**: 欢迎提出新的功能想法
- 📝 **文档改进**: 帮助完善项目文档
- 🔧 **代码贡献**: 提交Pull Request改进代码
- 🧪 **测试用例**: 添加测试用例提高代码质量

### 开发环境设置
```bash
# 克隆项目
git clone <repository-url>
cd RobotAgent

# 进入MVP开发目录
cd RobotAgent_MVP

# 安装依赖
pip install -r requirements.txt

# 配置开发环境
cp config/api_config.yaml.template config/api_config.yaml
```

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系方式

- 📧 **邮件**: 通过GitHub Issue联系
- 💬 **讨论**: 欢迎在GitHub Discussions中参与讨论
- 🐛 **问题反馈**: 请在GitHub Issues中提交

---

> ⚠️ **重要提醒**: 本项目目前处于MVP开发阶段，大部分功能仍在开发中。如果您对项目感兴趣，建议先体验火山方舟Chat API测试工具，了解项目的基础能力。
