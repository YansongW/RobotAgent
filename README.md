# RobotAgent - CAMEL-ROS2智能机器人系统

## 项目概述

RobotAgent是一个基于CAMEL.AI多智能体框架和ROS2机器人操作系统的智能机器人控制系统。采用"大脑（CAMEL认知）+ 小脑（ROS2控制）"的创新架构设计，通过6个专门的CAMEL智能体协同工作，实现自然语言交互的智能机器人控制。

### 核心设计理念
- **🧠 大脑-小脑架构**: CAMEL智能体负责认知决策，ROS2负责精确控制
- **🤝 多智能体协作**: 6个专门智能体分工合作，各司其职
- **🧠 多模态记忆**: 集成向量数据库和知识图谱的GraphRAG系统
- **⚡ 实时通信**: 基于Redis消息总线的高效智能体通信

## 系统特性

- 🤖 **CAMEL多智能体系统**：Dialog、Planning、Decision、Perception、Learning、ROS2六大智能体协同工作
- 🧠 **多模态记忆系统**：Milvus向量数据库 + Neo4j知识图谱 + LangGraph工作流引擎
- 🔄 **智能记忆管理**：自动分类、重要性评分、分层存储、生命周期管理
- 📊 **记忆可视化**：3D记忆空间、知识图谱可视化、交互式仪表板
- 🔧 **ROS2深度集成**：ROS2作为平等智能体成员，标准化机器人控制接口
- 🛡️ **多层安全保障**：智能体级、ROS2级、系统级三重安全机制
- 📈 **实时监控系统**：完整的智能体状态监控和协作可视化
- 🌐 **分布式架构**：支持单机和分布式部署，可扩展性强

## 快速开始

### 🚀 快速部署
详细的安装和部署指南请参考：[快速开始指南](QUICK_START.md)

### 环境要求
- Ubuntu 20.04 LTS
- Python 3.8+
- ROS2 Humble Hawksbill
- NVIDIA GPU（推荐，用于AI模型推理）
- 最少16GB RAM

### 一键启动
```bash
# 克隆项目
git clone <repository-url>
cd RobotAgent

# 运行安装脚本
./scripts/install.sh

# 启动系统
./scripts/start.sh
```

详细的安装步骤和配置说明请查看 [系统配置与部署指南](docs/system_configuration_deployment.md)。

## 📚 文档导航

> 💡 **快速导航**: 查看 [📋 文档索引](DOCUMENTATION_INDEX.md) 获取完整的文档导航和阅读建议

### 核心文档
- **[快速开始指南](QUICK_START.md)** - 快速部署和运行系统
- **[项目目录结构](PROJECT_STRUCTURE.md)** - 完整的项目文件组织说明
- **[项目总结](PROJECT_SUMMARY.md)** - 项目概览和技术栈总结

### CAMEL-ROS2集成文档
- **[CAMEL-ROS2架构设计](docs/camel_ros2_architecture.md)** - CAMEL与ROS2集成的完整架构
- **[多模态记忆系统](docs/multimodal_memory_system.md)** - GraphRAG和向量数据库设计
- **[代码架构说明](docs/code_architecture.md)** - 详细的代码结构和实现
- **[CAMEL智能体实现](docs/camel_agent_implementation.md)** - 各智能体的具体实现
- **[ROS2智能体实现](docs/ros2_agent_implementation.md)** - ROS2智能体的详细实现
- **[系统配置与部署](docs/system_configuration_deployment.md)** - 完整的配置和部署指南
- **[API接口与开发指南](docs/api_development_guide.md)** - REST API、WebSocket和SDK文档

### 开发指南
- **[API参考文档](docs/api_development_guide.md)** - 详细的API接口说明
- **[开发者指南](docs/api_development_guide.md#4-开发指南)** - 二次开发和扩展指南

## 项目结构

```
RobotAgent/
├── README.md                 # 项目说明文档
├── docs/                     # 详细文档目录
│   ├── camel_ros2_architecture.md      # CAMEL-ROS2架构设计
│   ├── multimodal_memory_system.md     # 多模态记忆系统
│   ├── camel_agent_implementation.md   # CAMEL智能体实现
│   ├── ros2_agent_implementation.md    # ROS2智能体实现
│   ├── system_configuration_deployment.md # 系统配置与部署
│   └── api_development_guide.md        # API接口与开发指南
├── src/                      # 源代码目录
│   ├── camel_agents/         # CAMEL智能体模块
│   │   ├── dialog_agent/     # 对话智能体
│   │   ├── planning_agent/   # 规划智能体
│   │   ├── decision_agent/   # 决策智能体
│   │   ├── perception_agent/ # 感知智能体
│   │   └── learning_agent/   # 学习智能体
│   ├── ros2_interface/       # ROS2接口模块
│   ├── memory_system/        # 多模态记忆系统
│   ├── communication/        # 通信模块
│   └── safety/              # 安全模块
├── config/                   # 配置文件
│   ├── agents/              # 智能体配置
│   ├── ros2/                # ROS2配置
│   ├── memory/              # 记忆系统配置
│   └── communication/       # 通信配置
├── tests/                    # 测试代码
├── requirements.txt          # Python依赖
└── main.py                  # 主程序入口
```

## 核心组件

### 1. CAMEL智能体系统
- **对话智能体 (Dialog Agent)**: 自然语言理解和对话管理
- **规划智能体 (Planning Agent)**: 任务规划和路径规划
- **决策智能体 (Decision Agent)**: 智能决策和动作选择
- **感知智能体 (Perception Agent)**: 多模态感知和环境理解
- **学习智能体 (Learning Agent)**: 经验学习和知识更新
- **ROS2智能体 (ROS2 Agent)**: 机器人控制和状态管理

### 2. 多模态记忆系统
- **LangGraph工作流引擎**: 智能记忆处理和分析工作流
- **记忆分类系统**: 智能体记忆、任务经验、领域知识三大类别
- **向量数据库 (Milvus)**: 多模态嵌入存储和语义检索
- **知识图谱 (Neo4j)**: 结构化知识表示和关系推理
- **GraphRAG**: 图增强检索生成系统
- **分层存储**: 热/温/冷/归档四层存储策略
- **可视化系统**: 3D记忆空间和知识图谱可视化
- **缓存系统**: 多层缓存优化和智能预取

### 3. ROS2接口模块
- **话题通信**: 标准ROS2话题发布和订阅
- **服务调用**: ROS2服务的同步通信
- **动作执行**: 长时间任务的异步执行
- **安全监控**: 实时安全状态监控

### 4. 通信架构
- **消息总线**: 基于Redis的智能体间通信
- **REST API**: 标准化的外部接口
- **WebSocket**: 实时数据推送
- **安全机制**: 认证、授权、监控

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系方式

如有问题或建议，请通过Issue或邮件联系项目维护者。# RobotAgent
