# RobotAgent 快速开始指南

本指南将帮助您快速部署和运行基于CAMEL-ROS2集成的RobotAgent智能机器人系统。

## 系统概述

RobotAgent是一个集成了CAMEL.AI多智能体框架和ROS2机器人操作系统的智能机器人系统，采用"大脑（CAMEL认知）+ 小脑（ROS2控制）"的架构设计。

### 核心组件
- **6个CAMEL智能体**: Dialog、Planning、Decision、Perception、Learning、ROS2
- **多模态记忆系统**: Milvus向量数据库 + Neo4j知识图谱
- **ROS2接口**: 标准化的机器人控制接口
- **通信总线**: 基于Redis的分布式消息系统

## 环境要求

### 硬件要求
- **CPU**: 8核心以上（推荐16核心）
- **内存**: 16GB以上（推荐32GB）
- **存储**: 100GB以上可用空间
- **GPU**: NVIDIA GPU（可选，用于AI模型加速）

### 软件要求
- **操作系统**: Ubuntu 20.04/22.04 或 Windows 10/11
- **Python**: 3.8+
- **ROS2**: Humble/Iron
- **Docker**: 20.10+
- **Docker Compose**: 2.0+

## 快速安装

### 1. 克隆项目
```bash
git clone https://github.com/your-org/RobotAgent.git
cd RobotAgent
```

### 2. 运行安装脚本
```bash
# Linux/macOS
chmod +x scripts/install.sh
./scripts/install.sh

# Windows (PowerShell)
.\scripts\install.ps1
```

安装脚本将自动完成：
- Python虚拟环境创建
- 依赖包安装
- ROS2环境配置
- Docker服务启动
- 数据库初始化

### 3. 配置系统
```bash
# 复制配置模板
cp config/main.yaml.template config/main.yaml
cp .env.template .env

# 编辑配置文件
nano config/main.yaml
nano .env
```

### 4. 启动系统
```bash
# 使用启动脚本
./scripts/start.sh

# 或者使用Docker Compose
docker-compose up -d
```

## 配置说明

### 主配置文件 (config/main.yaml)

```yaml
# CAMEL智能体配置
camel_agents:
  dialog_agent:
    model: "qwen-7b"
    max_tokens: 2048
    temperature: 0.7
  
  planning_agent:
    model: "qwen-7b"
    planning_horizon: 10
    safety_check: true
  
  # ... 其他智能体配置

# ROS2配置
ros2:
  domain_id: 42
  topics:
    cmd_vel: "/cmd_vel"
    joint_states: "/joint_states"
  safety:
    emergency_stop_topic: "/emergency_stop"
    workspace_limits:
      x: [-2.0, 2.0]
      y: [-2.0, 2.0]
      z: [0.0, 2.0]

# 记忆系统配置
memory_system:
  # LangGraph工作流引擎
  langgraph:
    enabled: true
    workflow_timeout: 300
    max_concurrent_workflows: 10
    checkpoint_interval: 30
  
  # 记忆分类配置
  classification:
    enabled: true
    categories:
      - agent_memory
      - task_experience
      - domain_knowledge
    importance_threshold: 0.5
    auto_classification: true
  
  # 向量数据库配置
  milvus:
    host: "localhost"
    port: 19530
    collections:
      agent_memory: "agent_memory_collection"
      task_experience: "task_experience_collection"
      domain_knowledge: "domain_knowledge_collection"
    embedding_dim: 768
  
  # 知识图谱配置
  neo4j:
    uri: "bolt://localhost:7687"
    username: "neo4j"
    password: "password"
    database: "robot_memory"
    max_pool_size: 50
  
  # 分层存储配置
  storage_tiers:
    hot_storage:
      enabled: true
      retention_days: 7
      max_size_gb: 10
    warm_storage:
      enabled: true
      retention_days: 30
      max_size_gb: 50
    cold_storage:
      enabled: true
      retention_days: 365
      max_size_gb: 200
    archive_storage:
      enabled: true
      retention_days: -1  # 永久保存
      compression: true
  
  # 可视化配置
  visualization:
    web_interface:
      enabled: true
      port: 8001
    graph_visualization:
      enabled: true
      max_nodes: 1000
    3d_visualization:
      enabled: true
      physics_enabled: true

# 通信配置
communication:
  message_bus:
    redis_url: "redis://localhost:6379"
    channels:
      - "agent_communication"
      - "ros2_bridge"
  
  api:
    host: "0.0.0.0"
    port: 8000
    cors_origins: ["*"]
```

### 环境变量 (.env)

```bash
# 系统配置
ROBOT_AGENT_HOME=/path/to/RobotAgent
PYTHONPATH=${ROBOT_AGENT_HOME}/src:${PYTHONPATH}

# ROS2配置
ROS_DOMAIN_ID=42
RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# AI模型配置
OPENAI_API_KEY=your_openai_key
HUGGINGFACE_TOKEN=your_hf_token

# 数据库配置
MILVUS_HOST=localhost
MILVUS_PORT=19530
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
REDIS_URL=redis://localhost:6379

# GPU配置（可选）
CUDA_VISIBLE_DEVICES=0
```

## 验证安装

### 1. 检查服务状态
```bash
# 检查Docker容器
docker ps

# 检查系统健康状态
curl http://localhost:8000/health

# 检查各智能体状态
curl http://localhost:8000/api/v1/agents/status
```

### 2. 运行测试
```bash
# 运行单元测试
python -m pytest tests/unit/ -v

# 运行集成测试
python -m pytest tests/integration/ -v

# 运行端到端测试
python tests/integration/test_end_to_end.py
```

### 3. 功能测试

#### 文本交互测试
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "请帮我移动机器人到桌子旁边",
    "user_id": "test_user"
  }'
```

#### 语音交互测试
```bash
curl -X POST http://localhost:8000/api/v1/audio \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@test_audio.wav" \
  -F "user_id=test_user"
```

#### ROS2通信测试
```bash
# 检查ROS2话题
ros2 topic list

# 发布测试消息
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  '{linear: {x: 0.1, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.1}}'

# 检查智能体状态
ros2 topic echo /robot_agent/status
```

#### 记忆系统测试
```bash
# 存储文本记忆
curl -X POST http://localhost:8000/api/v1/memory/store \
  -H "Content-Type: application/json" \
  -d '{
    "data": "机器人学会了如何抓取易碎物品",
    "data_type": "text",
    "memory_category": "domain_knowledge",
    "importance_score": 0.9,
    "metadata": {
      "skill": "fragile_object_handling",
      "difficulty": "high"
    }
  }'

# 搜索相关记忆
curl -X POST http://localhost:8000/api/v1/memory/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "抓取物品的技巧",
    "query_type": "semantic",
    "memory_categories": ["domain_knowledge"],
    "limit": 5
  }'

# 启动LangGraph记忆工作流
curl -X POST http://localhost:8000/api/v1/memory/workflow/start \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_type": "knowledge_integration",
    "input_data": {
      "domain": "object_manipulation",
      "context": "learning_session"
    }
  }'

# 获取记忆可视化数据
curl -X GET "http://localhost:8000/api/v1/memory/visualization/graph?node_types=task,object&max_nodes=50"

# 获取存储统计信息
curl -X GET http://localhost:8000/api/v1/memory/storage/stats

# 访问记忆系统Web界面
# 浏览器打开: http://localhost:8001
```

## 基本使用

### 1. Web界面访问

打开浏览器访问：
- **系统仪表板**: http://localhost:8000
- **API文档**: http://localhost:8000/docs
- **智能体监控**: http://localhost:8000/agents
- **记忆系统**: http://localhost:8000/memory

### 2. Python API调用

```python
import asyncio
from robot_agent_sdk import RobotAgentClient

async def main():
    # 创建客户端
    client = RobotAgentClient(base_url="http://localhost:8000")
    
    # 发送文本消息
    response = await client.chat(
        message="请帮我拿起桌上的杯子",
        user_id="user123"
    )
    print(f"机器人回复: {response.message}")
    
    # 获取智能体状态
    agents_status = await client.get_agents_status()
    for agent_name, status in agents_status.items():
        print(f"{agent_name}: {status}")
    
    # 控制机器人移动
    await client.move_robot(
        linear_x=0.1,
        angular_z=0.2,
        duration=5.0
    )
    
    # 记忆系统使用示例
    
    # 存储多模态记忆
    memory_result = await client.store_memory(
        data="机器人成功拿起了桌上的红色杯子",
        data_type="text",
        memory_category="task_experience",
        importance_score=0.8,
        metadata={
            "task_id": "pick_cup_001",
            "object": "red_cup",
            "location": "table",
            "success": True
        }
    )
    print(f"记忆存储成功: {memory_result.memory_id}")
    
    # 搜索相关记忆
    search_results = await client.search_memory(
        query="拿杯子的经验",
        query_type="semantic",
        memory_categories=["task_experience"],
        limit=5,
        importance_threshold=0.5
    )
    for memory in search_results.memories:
        print(f"相关记忆: {memory.content} (相似度: {memory.similarity})")
    
    # 启动LangGraph记忆工作流
    workflow_result = await client.start_memory_workflow(
        workflow_type="experience_analysis",
        input_data={
            "task_type": "object_manipulation",
            "context": "kitchen_environment"
        }
    )
    print(f"工作流启动: {workflow_result.workflow_id}")
    
    # 获取记忆可视化数据
    graph_data = await client.get_memory_graph(
        node_types=["task", "object", "location"],
        max_nodes=100
    )
    print(f"知识图谱节点数: {len(graph_data.nodes)}")
    
    # 获取3D记忆空间数据
    space_data = await client.get_3d_memory_space(
        memory_categories=["task_experience"],
        time_range={"start": "2024-01-01", "end": "2024-12-31"}
    )
    print(f"3D空间记忆点数: {len(space_data.memory_points)}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. ROS2命令行工具

```bash
# 查看可用服务
ros2 service list | grep robot_agent

# 调用规划服务
ros2 service call /robot_agent/plan_task \
  robot_agent_msgs/srv/PlanTask \
  '{task_description: "移动到厨房并拿起杯子"}'

# 查看智能体协作状态
ros2 topic echo /robot_agent/agent_collaboration

# 紧急停止
ros2 topic pub /emergency_stop std_msgs/msg/Bool '{data: true}'
```

## 常见问题

### Q: 智能体启动失败
**A**: 检查以下项目：
- Python虚拟环境是否激活
- 依赖包是否完整安装
- 配置文件是否正确
- 端口是否被占用

```bash
# 检查端口占用
netstat -tulpn | grep :8000

# 重新安装依赖
pip install -r requirements.txt

# 验证配置
python scripts/validate_config.py
```

### Q: ROS2通信问题
**A**: 检查ROS2环境：
```bash
# 检查ROS2环境
echo $ROS_DOMAIN_ID
echo $RMW_IMPLEMENTATION

# 重新source环境
source /opt/ros/humble/setup.bash
source install/setup.bash

# 检查网络配置
ros2 doctor
```

### Q: AI模型加载错误
**A**: 确认模型文件和配置：
```bash
# 检查模型文件
ls -la models/

# 检查GPU可用性
nvidia-smi

# 检查模型配置
python -c "from transformers import AutoTokenizer; print('模型加载正常')"
```

### Q: 数据库连接错误
**A**: 检查数据库服务：
```bash
# 检查Milvus
curl http://localhost:19530/health

# 检查Neo4j
curl http://localhost:7474

# 检查Redis
redis-cli ping

# 重启数据库服务
docker-compose restart milvus neo4j redis
```

### Q: 记忆系统问题
**A**: 检查记忆系统组件：
```bash
# 检查记忆系统状态
curl http://localhost:8000/api/v1/memory/health

# 检查向量数据库连接
python -c "
from src.memory_system.vector_database.milvus_client import MilvusClient
client = MilvusClient()
print('Milvus连接正常' if client.health_check() else 'Milvus连接失败')
"

# 检查知识图谱连接
python -c "
from src.memory_system.knowledge_graph.neo4j_client import Neo4jClient
client = Neo4jClient()
print('Neo4j连接正常' if client.health_check() else 'Neo4j连接失败')
"

# 检查LangGraph工作流
curl http://localhost:8000/api/v1/memory/workflow/status

# 重启记忆系统服务
docker-compose restart memory-system

# 清理记忆系统缓存
curl -X POST http://localhost:8000/api/v1/memory/cache/clear
```

### Q: 记忆分类不准确
**A**: 调整分类配置：
```bash
# 检查分类器状态
curl http://localhost:8000/api/v1/memory/classification/status

# 重新训练分类器
curl -X POST http://localhost:8000/api/v1/memory/classification/retrain

# 手动分类记忆
curl -X POST http://localhost:8000/api/v1/memory/classification/manual \
  -H "Content-Type: application/json" \
  -d '{
    "memory_id": "memory_123",
    "category": "task_experience",
    "importance_score": 0.8
  }'
```

### Q: 可视化界面无法访问
**A**: 检查可视化服务：
```bash
# 检查可视化服务状态
curl http://localhost:8001/health

# 检查端口占用
netstat -tulpn | grep :8001

# 重启可视化服务
docker-compose restart memory-visualization

# 检查防火墙设置
sudo ufw status
sudo ufw allow 8001
```

## 下一步

### 1. 深入学习
- 阅读 [CAMEL-ROS2架构文档](docs/camel_ros2_architecture.md)
- 了解 [多模态记忆系统](docs/multimodal_memory_system.md)
- 学习 [智能体实现指南](docs/camel_agent_implementation.md)

### 2. 自定义配置
- 修改智能体配置以适应特定任务
- 调整ROS2话题和服务配置
- 优化记忆系统参数

### 3. 扩展功能
- 添加自定义智能体
- 集成新的传感器和执行器
- 开发特定领域的技能包

### 4. 性能优化
- GPU加速配置
- 分布式部署
- 缓存策略优化

## 获取帮助

- **文档**: [完整文档](docs/)
- **示例**: [使用示例](examples/)
- **问题反馈**: [GitHub Issues](https://github.com/your-org/RobotAgent/issues)
- **社区讨论**: [Discussions](https://github.com/your-org/RobotAgent/discussions)

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。