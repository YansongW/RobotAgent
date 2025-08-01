# 项目目录结构

基于CAMEL-ROS2集成架构的智能机器人系统项目结构。

```
RobotAgent/
├── README.md                           # 项目主要说明文档
├── requirements.txt                    # Python依赖包列表
├── .env                               # 环境变量配置
├── .gitignore                         # Git忽略文件配置
├── main.py                            # 主程序入口
├── setup.py                           # 项目安装配置
├── LICENSE                            # 开源许可证
│
├── docs/                              # 文档目录
│   ├── PROJECT_SUMMARY.md             # 项目总览文档
│   ├── camel_ros2_architecture.md     # CAMEL-ROS2架构设计
│   ├── multimodal_memory_system.md    # 多模态记忆系统
│   ├── code_architecture.md           # 代码架构说明
│   ├── camel_agent_implementation.md  # CAMEL智能体实现
│   ├── camel_agent_implementation_part2.md # CAMEL智能体实现(第二部分)
│   ├── ros2_agent_implementation.md   # ROS2智能体实现
│   ├── system_configuration_deployment.md # 系统配置与部署
│   ├── api_development_guide.md       # API接口与开发指南
│   └── deployment.md                  # 传统部署指南(已弃用)
│
├── src/                               # 源代码目录
│   ├── __init__.py
│   ├── main.py                        # 系统主入口
│   │
│   ├── camel_agents/                  # CAMEL智能体模块
│   │   ├── __init__.py
│   │   ├── base_robot_agent.py        # 机器人智能体基类
│   │   ├── dialog_agent.py            # 对话智能体
│   │   ├── planning_agent.py          # 规划智能体
│   │   ├── decision_agent.py          # 决策智能体
│   │   ├── perception_agent.py        # 感知智能体
│   │   ├── learning_agent.py          # 学习智能体
│   │   ├── ros2_agent.py              # ROS2智能体
│   │   └── prompts/                   # 智能体提示词
│   │       ├── dialog_prompts.txt
│   │       ├── planning_prompts.txt
│   │       ├── decision_prompts.txt
│   │       ├── perception_prompts.txt
│   │       ├── learning_prompts.txt
│   │       └── ros2_prompts.txt
│   │
│   ├── ros2_interface/                # ROS2接口模块
│   │   ├── __init__.py
│   │   ├── ros2_wrapper.py            # ROS2包装器
│   │   ├── command_executor_node.py   # 指令执行节点
│   │   ├── sensor_data_node.py        # 传感器数据节点
│   │   ├── safety_monitor_node.py     # 安全监控节点
│   │   ├── interfaces/                # ROS2接口定义
│   │   │   ├── msg/                   # 消息类型
│   │   │   │   ├── AgentCommand.msg
│   │   │   │   ├── AgentResponse.msg
│   │   │   │   ├── RobotState.msg
│   │   │   │   └── SafetyStatus.msg
│   │   │   ├── srv/                   # 服务类型
│   │   │   │   ├── ExecuteTask.srv
│   │   │   │   └── GetSystemStatus.srv
│   │   │   └── action/                # 动作类型
│   │   │       ├── ComplexTask.action
│   │   │       └── NavigateToGoal.action
│   │   ├── launch/                    # 启动文件
│   │   │   ├── robot_agent_system.launch.py
│   │   │   ├── camel_agents.launch.py
│   │   │   └── ros2_nodes.launch.py
│   │   └── package.xml                # ROS2包配置
│   │
│   ├── memory_system/                 # 多模态记忆系统
│   │   ├── __init__.py
│   │   ├── langgraph_engine.py        # LangGraph工作流引擎
│   │   ├── storage_manager.py         # 多模态分类存储管理器
│   │   ├── multimodal_processor.py    # 多模态数据处理器
│   │   ├── classifiers/               # 记忆分类器
│   │   │   ├── __init__.py
│   │   │   ├── memory_classifier.py   # 记忆分类器
│   │   │   └── modality_detector.py   # 模态检测器
│   │   ├── processors/                # 记忆处理器
│   │   │   ├── __init__.py
│   │   │   ├── agent_memory_processor.py    # 智能体记忆处理器
│   │   │   ├── task_experience_processor.py # 任务经验处理器
│   │   │   └── domain_knowledge_processor.py # 领域知识处理器
│   │   ├── vector_store/              # 向量数据库
│   │   │   ├── __init__.py
│   │   │   ├── milvus_client.py       # Milvus客户端
│   │   │   ├── embedding_manager.py   # 嵌入管理器
│   │   │   └── vector_operations.py   # 向量操作
│   │   ├── knowledge_graph/           # 知识图谱
│   │   │   ├── __init__.py
│   │   │   ├── neo4j_client.py        # Neo4j客户端
│   │   │   ├── graph_builder.py       # 图谱构建器
│   │   │   └── graph_query.py         # 图谱查询
│   │   ├── graphrag/                  # GraphRAG引擎
│   │   │   ├── __init__.py
│   │   │   ├── retrieval_engine.py    # 检索引擎
│   │   │   ├── reasoning_engine.py    # 推理引擎
│   │   │   └── fusion_engine.py       # 融合引擎
│   │   ├── visualization/             # 知识图谱可视化系统
│   │   │   ├── __init__.py
│   │   │   ├── graph_visualizer.py    # 图谱可视化器
│   │   │   ├── web_app.py             # Web可视化应用
│   │   │   ├── dashboard.py           # 仪表板
│   │   │   └── templates/             # 可视化模板
│   │   │       ├── dashboard.html
│   │   │       ├── graph_view.html
│   │   │       └── analytics.html
│   │   ├── storage/                   # 分层存储系统
│   │   │   ├── __init__.py
│   │   │   ├── tiered_storage.py      # 分层存储管理
│   │   │   ├── hot_storage.py         # 热存储（内存）
│   │   │   ├── warm_storage.py        # 温存储（SSD）
│   │   │   ├── cold_storage.py        # 冷存储（HDD）
│   │   │   └── archive_storage.py     # 归档存储（对象存储）
│   │   └── cache/                     # 缓存系统
│   │       ├── __init__.py
│   │       ├── redis_cache.py         # Redis缓存
│   │       └── memory_cache.py        # 内存缓存
│   │
│   ├── communication/                 # 通信模块
│   │   ├── __init__.py
│   │   ├── message_bus.py             # 消息总线
│   │   ├── redis_client.py            # Redis客户端
│   │   ├── websocket_server.py        # WebSocket服务器
│   │   └── api/                       # REST API
│   │       ├── __init__.py
│   │       ├── main_api.py            # 主API服务
│   │       ├── agent_api.py           # 智能体API
│   │       ├── robot_api.py           # 机器人API
│   │       ├── memory_api.py          # 记忆系统API
│   │       └── schemas.py             # 数据模式
│   │
│   ├── safety/                        # 安全模块
│   │   ├── __init__.py
│   │   ├── safety_monitor.py          # 安全监控器
│   │   ├── emergency_stop.py          # 紧急停止
│   │   ├── collision_detection.py     # 碰撞检测
│   │   ├── workspace_limits.py        # 工作空间限制
│   │   └── safety_rules.py            # 安全规则
│   │
│   └── utils/                         # 工具模块
│       ├── __init__.py
│       ├── config_loader.py           # 配置加载器
│       ├── logger.py                  # 日志工具
│       ├── file_utils.py              # 文件工具
│       ├── time_utils.py              # 时间工具
│       └── validation.py              # 验证工具
│
├── config/                            # 配置文件目录
│   ├── main.yaml                      # 主配置文件
│   ├── agents/                        # 智能体配置
│   │   ├── dialog_agent.yaml          # 对话智能体配置
│   │   ├── planning_agent.yaml        # 规划智能体配置
│   │   ├── decision_agent.yaml        # 决策智能体配置
│   │   ├── perception_agent.yaml      # 感知智能体配置
│   │   ├── learning_agent.yaml        # 学习智能体配置
│   │   └── ros2_agent.yaml            # ROS2智能体配置
│   ├── ros2/                          # ROS2配置
│   │   ├── topics.yaml                # 话题配置
│   │   ├── services.yaml              # 服务配置
│   │   ├── actions.yaml               # 动作配置
│   │   └── safety.yaml                # 安全配置
│   ├── memory/                        # 记忆系统配置
│   │   ├── langgraph.yaml             # LangGraph工作流配置
│   │   ├── memory_classification.yaml # 记忆分类配置
│   │   ├── storage_tiers.yaml         # 分层存储配置
│   │   ├── milvus.yaml                # Milvus配置
│   │   ├── neo4j.yaml                 # Neo4j配置
│   │   ├── text_processing.yaml       # 文本处理配置
│   │   ├── image_processing.yaml      # 图像处理配置
│   │   ├── video_processing.yaml      # 视频处理配置
│   │   ├── visualization.yaml         # 可视化配置
│   │   └── caching.yaml               # 缓存配置
│   ├── communication/                 # 通信配置
│   │   ├── message_bus.yaml           # 消息总线配置
│   │   ├── api.yaml                   # API配置
│   │   └── websocket.yaml             # WebSocket配置
│   ├── security/                      # 安全配置
│   │   ├── authentication.yaml        # 认证配置
│   │   ├── authorization.yaml         # 授权配置
│   │   └── monitoring.yaml            # 监控配置
│   ├── prompts/                       # 提示词文件
│   │   ├── dialog_prompts.txt         # 对话提示词
│   │   ├── planning_prompts.txt       # 规划提示词
│   │   └── ros2_prompts.txt           # ROS2提示词
│   └── robot/                         # 机器人配置
│       ├── robot.urdf                 # 机器人模型
│       ├── robot.srdf                 # 语义机器人描述
│       ├── joint_limits.yaml          # 关节限制
│       └── kinematics.yaml            # 运动学配置
│
├── data/                              # 数据目录
│   ├── vector_store/                  # 向量数据库
│   │   ├── collections/               # Milvus集合
│   │   └── embeddings/                # 嵌入向量
│   ├── knowledge_graph/               # 知识图谱
│   │   ├── nodes/                     # 图谱节点
│   │   └── relationships/             # 图谱关系
│   ├── cache/                         # 缓存数据
│   │   ├── redis/                     # Redis缓存
│   │   ├── memory/                    # 内存缓存
│   │   └── model_cache/               # 模型缓存
│   ├── logs/                          # 日志文件
│   │   ├── agents/                    # 智能体日志
│   │   ├── ros2/                      # ROS2日志
│   │   ├── memory/                    # 记忆系统日志
│   │   └── system/                    # 系统日志
│   ├── uploads/                       # 上传文件
│   │   ├── images/                    # 图像文件
│   │   ├── videos/                    # 视频文件
│   │   └── audio/                     # 音频文件
│   └── backups/                       # 备份文件
│       ├── config_backups/            # 配置备份
│       ├── data_backups/              # 数据备份
│       └── model_backups/             # 模型备份
│
├── models/                            # AI模型目录
│   ├── llm/                           # 大语言模型
│   │   ├── qwen/                      # Qwen模型
│   │   ├── llama/                     # LLaMA模型
│   │   └── custom/                    # 自定义模型
│   ├── embedding/                     # 嵌入模型
│   │   ├── text/                      # 文本嵌入
│   │   ├── image/                     # 图像嵌入
│   │   └── multimodal/                # 多模态嵌入
│   ├── vision/                        # 视觉模型
│   │   ├── clip/                      # CLIP模型
│   │   ├── yolo/                      # YOLO模型
│   │   └── detection/                 # 检测模型
│   ├── audio/                         # 音频模型
│   │   ├── whisper/                   # Whisper模型
│   │   └── speech/                    # 语音模型
│   └── cache/                         # 模型缓存
│
├── tests/                             # 测试代码目录
│   ├── __init__.py
│   ├── conftest.py                    # pytest配置
│   ├── unit/                          # 单元测试
│   │   ├── __init__.py
│   │   ├── test_camel_agents.py       # CAMEL智能体测试
│   │   ├── test_ros2_interface.py     # ROS2接口测试
│   │   ├── test_memory_system.py      # 记忆系统测试
│   │   ├── test_communication.py      # 通信模块测试
│   │   └── test_safety.py             # 安全模块测试
│   ├── integration/                   # 集成测试
│   │   ├── __init__.py
│   │   ├── test_agent_collaboration.py # 智能体协作测试
│   │   ├── test_ros2_integration.py   # ROS2集成测试
│   │   ├── test_memory_integration.py # 记忆系统集成测试
│   │   └── test_end_to_end.py         # 端到端测试
│   ├── fixtures/                      # 测试数据
│   │   ├── test_audio.wav             # 测试音频
│   │   ├── test_image.jpg             # 测试图像
│   │   ├── test_video.mp4             # 测试视频
│   │   └── test_configs/              # 测试配置
│   └── mocks/                         # 模拟对象
│       ├── __init__.py
│       ├── mock_llm.py                # 模拟LLM
│       ├── mock_ros2.py               # 模拟ROS2
│       └── mock_memory.py             # 模拟记忆系统
│
├── scripts/                           # 脚本目录
│   ├── install.sh                     # 安装脚本
│   ├── start.sh                       # 启动脚本
│   ├── stop.sh                        # 停止脚本
│   ├── restart.sh                     # 重启脚本
│   ├── backup.sh                      # 备份脚本
│   ├── init_system.py                 # 系统初始化
│   ├── setup_memory.py                # 记忆系统设置
│   ├── validate_config.py             # 配置验证
│   ├── performance_test.py            # 性能测试
│   └── deployment/                    # 部署脚本
│       ├── docker/                    # Docker部署
│       │   ├── Dockerfile
│       │   ├── docker-compose.yml
│       │   └── entrypoint.sh
│       ├── kubernetes/                # Kubernetes部署
│       │   ├── namespace.yaml
│       │   ├── configmap.yaml
│       │   ├── deployment.yaml
│       │   └── service.yaml
│       └── systemd/                   # Systemd服务
│           ├── robot-agent.service
│           └── robot-agent.target
│
├── tools/                             # 开发工具
│   ├── agent_generator/               # 智能体生成器
│   │   ├── generate_agent.py
│   │   └── agent_templates/
│   ├── memory_tools/                  # 记忆系统工具
│   │   ├── vector_store_manager.py
│   │   ├── knowledge_graph_builder.py
│   │   └── data_migration.py
│   ├── monitoring/                    # 监控工具
│   │   ├── grafana_dashboard.json
│   │   ├── prometheus_config.yml
│   │   └── agent_metrics.py
│   └── debugging/                     # 调试工具
│       ├── log_analyzer.py
│       ├── performance_profiler.py
│       └── agent_debugger.py
│
├── examples/                          # 示例代码
│   ├── basic_usage.py                 # 基础使用示例
│   ├── agent_interaction.py           # 智能体交互示例
│   ├── multimodal_demo.py             # 多模态演示
│   ├── ros2_integration.py            # ROS2集成示例
│   └── custom_agents/                 # 自定义智能体示例
│       ├── custom_dialog_agent.py
│       ├── custom_planning_agent.py
│       └── custom_perception_agent.py
│
└── assets/                            # 资源文件
    ├── images/                        # 图片资源
    │   ├── architecture_diagram.png
    │   ├── agent_collaboration.png
    │   ├── memory_system.png
    │   └── ui_screenshots/
    ├── videos/                        # 视频资源
    │   ├── system_demo.mp4
    │   ├── agent_tutorial.mp4
    │   └── deployment_guide.mp4
    └── documents/                     # 文档资源
        ├── presentation.pptx
        ├── technical_specs.pdf
        └── research_papers/
```

## 架构特点

### CAMEL智能体系统
- **多智能体协作**: 6个专门的CAMEL智能体协同工作
- **统一基类**: 所有智能体继承自BaseRobotAgent
- **消息总线**: 基于Redis的分布式通信机制
- **提示词管理**: 集中化的提示词配置和管理

### ROS2集成
- **平等地位**: ROS2作为CAMEL框架中的智能体成员
- **标准接口**: 遵循ROS2消息、服务、动作规范
- **安全监控**: 实时安全状态监控和紧急停止
- **模块化设计**: 独立的ROS2接口模块

### 多模态记忆系统
- **向量数据库**: 使用Milvus存储多模态嵌入
- **知识图谱**: 使用Neo4j构建结构化知识
- **GraphRAG**: 结合检索和推理的增强系统
- **缓存优化**: 多层缓存提升性能

### 通信架构
- **消息总线**: 智能体间异步通信
- **REST API**: 标准化的外部接口
- **WebSocket**: 实时数据推送
- **安全机制**: 认证、授权、监控

### 安全保障
- **多层安全**: 智能体级、ROS2级、系统级安全
- **实时监控**: 持续的安全状态检查
- **紧急停止**: 快速响应的安全机制
- **工作空间限制**: 物理安全边界

这个项目结构完全基于CAMEL-ROS2集成架构，实现了"大脑（CAMEL认知）+ 小脑（ROS2控制）"的设计理念，通过多模态记忆系统实现智能体间的知识共享和协作。