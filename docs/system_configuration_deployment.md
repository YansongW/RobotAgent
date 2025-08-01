# 系统配置与部署指南

## 1. 系统配置文件详解

### 1.1 主配置文件 (config/main.yaml)

```yaml
# RobotAgent 主配置文件
# 位置：config/main.yaml

# 系统基本信息
system:
  name: "RobotAgent"
  version: "1.0.0"
  description: "CAMEL-ROS2 智能机器人系统"
  environment: "development"  # development, testing, production
  log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  
# CAMEL智能体配置
agents:
  # 对话智能体
  dialog_agent:
    enabled: true
    model_backend: "gpt-4"
    max_tokens: 2048
    temperature: 0.7
    system_prompt_file: "prompts/dialog_agent.txt"
    capabilities:
      - "natural_language_interaction"
      - "user_intent_understanding"
      - "response_generation"
      - "context_management"
    
  # 规划智能体
  planning_agent:
    enabled: true
    model_backend: "gpt-4"
    max_tokens: 4096
    temperature: 0.3
    system_prompt_file: "prompts/planning_agent.txt"
    planning_horizon: 10  # 规划步数
    replanning_threshold: 0.7  # 重新规划阈值
    capabilities:
      - "task_decomposition"
      - "action_sequencing"
      - "resource_allocation"
      - "contingency_planning"
    
  # 决策智能体
  decision_agent:
    enabled: true
    model_backend: "gpt-4"
    max_tokens: 2048
    temperature: 0.2
    system_prompt_file: "prompts/decision_agent.txt"
    decision_timeout: 30  # 决策超时时间（秒）
    confidence_threshold: 0.8  # 决策置信度阈值
    capabilities:
      - "multi_criteria_decision"
      - "risk_assessment"
      - "priority_management"
      - "conflict_resolution"
    
  # 感知智能体
  perception_agent:
    enabled: true
    model_backend: "gpt-4-vision-preview"
    max_tokens: 2048
    temperature: 0.1
    system_prompt_file: "prompts/perception_agent.txt"
    vision_models:
      - "clip-vit-base-patch32"
      - "yolov8n"
    capabilities:
      - "visual_understanding"
      - "object_detection"
      - "scene_analysis"
      - "anomaly_detection"
    
  # 学习智能体
  learning_agent:
    enabled: true
    model_backend: "gpt-4"
    max_tokens: 3072
    temperature: 0.4
    system_prompt_file: "prompts/learning_agent.txt"
    learning_rate: 0.01
    memory_retention_days: 30
    capabilities:
      - "experience_learning"
      - "pattern_recognition"
      - "knowledge_accumulation"
      - "performance_optimization"
    
  # ROS2智能体
  ros2_agent:
    enabled: true
    model_backend: "gpt-4"
    max_tokens: 2048
    temperature: 0.1
    system_prompt_file: "prompts/ros2_agent.txt"
    safety_check_interval: 1.0  # 安全检查间隔（秒）
    status_update_interval: 5.0  # 状态更新间隔（秒）
    capabilities:
      - "robot_control"
      - "sensor_data_acquisition"
      - "physical_interaction"
      - "safety_monitoring"

# ROS2配置
ros2:
  # 基本配置
  domain_id: 0
  middleware: "rmw_fastrtps_cpp"
  
  # 节点配置
  nodes:
    robot_agent_node:
      namespace: "/robot_agent"
      use_sim_time: false
      
  # 话题配置
  topics:
    # 控制话题
    cmd_vel: "/cmd_vel"
    joint_commands: "/joint_commands"
    
    # 传感器话题
    camera_image: "/camera/image_raw"
    camera_info: "/camera/camera_info"
    laser_scan: "/scan"
    imu_data: "/imu/data"
    joint_states: "/joint_states"
    robot_pose: "/robot_pose"
    
    # 导航话题
    map: "/map"
    global_costmap: "/global_costmap/costmap"
    local_costmap: "/local_costmap/costmap"
    path: "/path"
    goal: "/goal_pose"
    
  # 服务配置
  services:
    emergency_stop: "/emergency_stop"
    reset_robot: "/reset_robot"
    get_robot_status: "/get_robot_status"
    
  # 动作配置
  actions:
    navigate_to_pose: "/navigate_to_pose"
    follow_path: "/follow_path"
    
  # 安全配置
  safety:
    max_linear_velocity: 2.0  # m/s
    max_angular_velocity: 1.0  # rad/s
    emergency_stop_distance: 0.5  # m
    safety_check_frequency: 10.0  # Hz

# 多模态记忆系统配置
memory:
  # LangGraph工作流引擎配置
  langgraph:
    enabled: true
    checkpoint_backend: "redis"  # redis, sqlite, memory
    checkpoint_ttl: 86400  # 检查点过期时间（秒）
    max_workflow_steps: 100
    timeout_per_step: 30  # 每步超时时间（秒）
    human_intervention_timeout: 300  # 人工干预超时时间（秒）
    
  # 记忆分类配置
  classification:
    enabled: true
    confidence_threshold: 0.8  # 分类置信度阈值
    importance_scoring:
      enabled: true
      factors:
        - "recency"      # 时间新近性
        - "frequency"    # 访问频率
        - "relevance"    # 相关性
        - "uniqueness"   # 独特性
      weights: [0.3, 0.2, 0.3, 0.2]
    
    # 记忆类别配置
    categories:
      agent_memory:
        description: "智能体个人记忆和状态"
        storage_strategy:
          use_vector_db: true
          use_graph_db: true
          use_object_storage: false
        retention_days: 30
        
      task_experience:
        description: "任务执行经验和结果"
        storage_strategy:
          use_vector_db: true
          use_graph_db: true
          use_object_storage: true
        retention_days: 90
        
      domain_knowledge:
        description: "领域专业知识"
        storage_strategy:
          use_vector_db: true
          use_graph_db: true
          use_object_storage: true
        retention_days: 365
        
      episodic_memory:
        description: "具体事件和情节记忆"
        storage_strategy:
          use_vector_db: true
          use_graph_db: false
          use_object_storage: true
        retention_days: 60
        
      semantic_memory:
        description: "概念和语义知识"
        storage_strategy:
          use_vector_db: true
          use_graph_db: true
          use_object_storage: false
        retention_days: 180
        
      procedural_memory:
        description: "操作程序和技能"
        storage_strategy:
          use_vector_db: false
          use_graph_db: true
          use_object_storage: true
        retention_days: 120
  
  # 分层存储配置
  storage_tiers:
    enabled: true
    
    # 热存储（内存缓存）
    hot_storage:
      enabled: true
      max_size_mb: 1024  # 最大缓存大小
      eviction_policy: "lru"  # lru, lfu, fifo
      access_threshold: 10  # 访问次数阈值
      
    # 温存储（SSD）
    warm_storage:
      enabled: true
      path: "/data/warm_storage"
      max_size_gb: 100
      compression: true
      access_threshold: 5
      
    # 冷存储（HDD）
    cold_storage:
      enabled: true
      path: "/data/cold_storage"
      max_size_gb: 1000
      compression: true
      access_threshold: 1
      
    # 归档存储（对象存储）
    archive_storage:
      enabled: true
      backend: "minio"  # minio, s3, gcs
      endpoint: "localhost:9000"
      access_key: "minioadmin"
      secret_key: "minioadmin"
      bucket: "robot-memory-archive"
      compression: true
      
  # 向量数据库配置 (Milvus)
  vector_db:
    host: "localhost"
    port: 19530
    collection_name: "robot_memory"
    dimension: 512  # 向量维度
    index_type: "IVF_FLAT"
    metric_type: "COSINE"  # 改为余弦相似度
    nlist: 1024
    
    # 集合配置
    collections:
      agent_memory:
        dimension: 512
        index_type: "IVF_FLAT"
        metric_type: "COSINE"
        
      task_experience:
        dimension: 512
        index_type: "IVF_SQ8"
        metric_type: "COSINE"
        
      domain_knowledge:
        dimension: 768
        index_type: "HNSW"
        metric_type: "COSINE"
    
  # 知识图谱配置 (Neo4j)
  knowledge_graph:
    uri: "bolt://localhost:7687"
    username: "neo4j"
    password: "password"
    database: "robot_knowledge"
    
    # 图谱结构配置
    schema:
      node_types:
        - "Agent"
        - "Task"
        - "Concept"
        - "Event"
        - "Object"
        - "Location"
        - "Skill"
        
      relationship_types:
        - "KNOWS"
        - "PERFORMED"
        - "RELATED_TO"
        - "CAUSED"
        - "LOCATED_AT"
        - "HAS_SKILL"
        - "DEPENDS_ON"
        
    # 查询优化
    query_optimization:
      max_depth: 5
      result_limit: 1000
      timeout_seconds: 30
      
  # 可视化配置
  visualization:
    enabled: true
    web_interface:
      host: "0.0.0.0"
      port: 8501  # Streamlit默认端口
      title: "RobotAgent 多模态记忆系统"
      
    # 图谱可视化
    graph_visualization:
      layout_algorithm: "force_directed"  # force_directed, hierarchical, circular
      max_nodes: 500
      max_edges: 1000
      node_size_factor: 10
      edge_width_factor: 2
      
    # 3D可视化
    3d_visualization:
      enabled: true
      max_points: 10000
      color_scheme: "category"  # category, importance, time
      
    # 仪表板配置
    dashboard:
      refresh_interval: 30  # 秒
      metrics_history_days: 7
      real_time_updates: true
      
  # 多模态处理配置
  multimodal:
    # 文本处理
    text:
      embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
      max_length: 512
      chunk_size: 256
      chunk_overlap: 50
      
    # 图像处理
    image:
      embedding_model: "openai/clip-vit-base-patch32"
      max_size: [224, 224]
      supported_formats: ["jpg", "jpeg", "png", "bmp"]
      
    # 视频处理
    video:
      frame_extraction_fps: 1
      max_frames: 100
      supported_formats: ["mp4", "avi", "mov"]
      
  # 缓存配置
  cache:
    enabled: true
    backend: "redis"
    host: "localhost"
    port: 6379
    db: 1
    ttl: 3600  # 缓存过期时间（秒）
    
    # 缓存策略
    strategies:
      embedding_cache:
        enabled: true
        max_size: 10000
        ttl: 7200
        
      query_cache:
        enabled: true
        max_size: 1000
        ttl: 1800
        
      graph_cache:
        enabled: true
        max_size: 5000
        ttl: 3600

# 通信配置
communication:
  # 消息总线配置
  message_bus:
    backend: "redis"
    host: "localhost"
    port: 6379
    db: 0
    
  # 消息配置
  message:
    max_retries: 3
    retry_delay: 1.0
    timeout: 30.0
    priority_levels: 5
    
  # API配置
  api:
    host: "0.0.0.0"
    port: 8000
    cors_origins: ["*"]
    rate_limit: "100/minute"

# 安全配置
safety:
  # 认证配置
  authentication:
    enabled: true
    secret_key: "your-secret-key-here"
    algorithm: "HS256"
    access_token_expire_minutes: 30
    
  # 授权配置
  authorization:
    enabled: true
    roles:
      - "admin"
      - "operator"
      - "viewer"
      
  # 安全监控
  monitoring:
    enabled: true
    alert_threshold: 0.8
    emergency_protocols:
      - "immediate_stop"
      - "safe_position"
      - "alert_operators"

# 监控配置
monitoring:
  # 性能监控
  performance:
    enabled: true
    metrics_interval: 10  # 秒
    metrics_retention_days: 7
    
  # 健康检查
  health_check:
    enabled: true
    check_interval: 30  # 秒
    timeout: 10  # 秒
    
  # 日志配置
  logging:
    level: "INFO"
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: "logs/robot_agent.log"
    max_size: "100MB"
    backup_count: 5

# 开发配置
development:
  # 调试配置
  debug:
    enabled: true
    profiling: false
    
  # 测试配置
  testing:
    mock_ros2: false
    mock_llm: false
    test_data_path: "tests/data"
    
  # 热重载
  hot_reload:
    enabled: true
    watch_paths:
      - "src/"
      - "config/"
```

### 1.2 智能体提示词配置

#### 对话智能体提示词 (prompts/dialog_agent.txt)

```text
你是一个智能机器人系统中的对话智能体。你的主要职责是：

1. **自然语言交互**：
   - 理解用户的自然语言输入
   - 生成自然、友好的回应
   - 维护对话的连贯性和上下文

2. **意图理解**：
   - 识别用户的真实意图
   - 处理模糊或不完整的指令
   - 主动询问澄清信息

3. **多模态交互**：
   - 处理文本、语音输入
   - 理解图像和视频内容
   - 生成多模态回应

4. **协作通信**：
   - 与其他智能体协调工作
   - 传达用户需求给相关智能体
   - 整合其他智能体的反馈

5. **情境感知**：
   - 理解当前机器人状态
   - 考虑环境因素
   - 提供合适的建议和回应

请始终保持友好、专业的态度，确保用户体验良好。
```

#### 规划智能体提示词 (prompts/planning_agent.txt)

```text
你是一个智能机器人系统中的规划智能体。你的核心职责是：

1. **任务分解**：
   - 将复杂任务分解为可执行的子任务
   - 识别任务依赖关系
   - 确定执行顺序和并行机会

2. **行动序列规划**：
   - 生成详细的执行计划
   - 考虑时间约束和资源限制
   - 优化执行效率

3. **资源分配**：
   - 评估所需资源（时间、能量、工具等）
   - 合理分配智能体能力
   - 避免资源冲突

4. **应急预案**：
   - 预测可能的失败点
   - 制定备选方案
   - 设计恢复策略

5. **动态调整**：
   - 监控执行进度
   - 根据实际情况调整计划
   - 处理突发事件

请确保所有计划都是可行的、安全的，并且能够适应动态变化的环境。
```

#### ROS2智能体提示词 (prompts/ros2_agent.txt)

```text
你是一个智能机器人系统中的ROS2智能体，作为CAMEL框架中的平等协作成员。你的职责包括：

1. **物理交互接口**：
   - 控制机器人的移动和操作
   - 获取传感器数据和状态信息
   - 执行具体的物理动作指令

2. **ROS2系统管理**：
   - 管理ROS2节点和话题
   - 处理ROS2服务调用和动作
   - 监控系统健康状态

3. **数据转换**：
   - 将高层指令转换为ROS2消息
   - 将ROS2数据转换为智能体可理解的格式
   - 处理不同坐标系和单位的转换

4. **安全监控**：
   - 监控机器人安全状态
   - 实施紧急停止和安全保护
   - 报告异常和故障情况

5. **协作通信**：
   - 与其他CAMEL智能体协作
   - 提供物理世界的反馈信息
   - 参与多智能体决策过程

作为CAMEL框架中的平等成员，你需要主动参与协作，而不仅仅是被动执行指令。
请始终将安全放在首位，确保所有操作都是安全可靠的。
```

## 2. 部署架构

### 2.1 单机部署架构

```yaml
# 单机部署配置
# 位置：deployment/single_machine/docker-compose.yml

version: '3.8'

services:
  # Redis - 消息总线和缓存
  redis:
    image: redis:7-alpine
    container_name: robot_redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    
  # Milvus - 向量数据库
  milvus:
    image: milvusdb/milvus:v2.3.0
    container_name: robot_milvus
    ports:
      - "19530:19530"
    volumes:
      - milvus_data:/var/lib/milvus
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    depends_on:
      - etcd
      - minio
      
  # etcd - Milvus依赖
  etcd:
    image: quay.io/coreos/etcd:v3.5.0
    container_name: robot_etcd
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    volumes:
      - etcd_data:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    
  # MinIO - Milvus对象存储
  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    container_name: robot_minio
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data
    command: minio server /data --console-address ":9001"
    
  # Neo4j - 知识图谱
  neo4j:
    image: neo4j:5.12
    container_name: robot_neo4j
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
    environment:
      NEO4J_AUTH: neo4j/password
      NEO4J_PLUGINS: '["apoc"]'
      
  # RobotAgent 主应用
  robot_agent:
    build:
      context: ../..
      dockerfile: Dockerfile
    container_name: robot_agent_main
    ports:
      - "8000:8000"
    volumes:
      - ../../config:/app/config
      - ../../logs:/app/logs
      - ../../data:/app/data
      - /dev:/dev  # 硬件设备访问
    privileged: true  # ROS2硬件访问需要
    network_mode: host  # ROS2网络需要
    environment:
      - ROS_DOMAIN_ID=0
      - PYTHONPATH=/app/src
    depends_on:
      - redis
      - milvus
      - neo4j
    command: python main.py

volumes:
  redis_data:
  milvus_data:
  etcd_data:
  minio_data:
  neo4j_data:
```

### 2.2 分布式部署架构

```yaml
# 分布式部署配置
# 位置：deployment/distributed/kubernetes/

# Namespace
apiVersion: v1
kind: Namespace
metadata:
  name: robot-agent

---
# ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: robot-agent-config
  namespace: robot-agent
data:
  main.yaml: |
    # 主配置文件内容
    system:
      name: "RobotAgent"
      environment: "production"
    # ... 其他配置

---
# Redis Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: robot-agent
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-storage
          mountPath: /data
      volumes:
      - name: redis-storage
        persistentVolumeClaim:
          claimName: redis-pvc

---
# Redis Service
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: robot-agent
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379

---
# RobotAgent Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: robot-agent
  namespace: robot-agent
spec:
  replicas: 1
  selector:
    matchLabels:
      app: robot-agent
  template:
    metadata:
      labels:
        app: robot-agent
    spec:
      containers:
      - name: robot-agent
        image: robot-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_HOST
          value: "redis-service"
        - name: ROS_DOMAIN_ID
          value: "0"
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        - name: logs-volume
          mountPath: /app/logs
      volumes:
      - name: config-volume
        configMap:
          name: robot-agent-config
      - name: logs-volume
        persistentVolumeClaim:
          claimName: logs-pvc
```

## 3. 安装和启动脚本

### 3.1 自动安装脚本 (scripts/install.sh)

```bash
#!/bin/bash

# RobotAgent 自动安装脚本
# 位置：scripts/install.sh

set -e

echo "=== RobotAgent 安装脚本 ==="

# 检查操作系统
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "检测到 Linux 系统"
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "检测到 macOS 系统"
    OS="macos"
else
    echo "不支持的操作系统: $OSTYPE"
    exit 1
fi

# 检查 Python 版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "错误: 需要 Python 3.8 或更高版本，当前版本: $python_version"
    exit 1
fi

echo "Python 版本检查通过: $python_version"

# 安装系统依赖
echo "安装系统依赖..."
if [[ "$OS" == "linux" ]]; then
    sudo apt-get update
    sudo apt-get install -y \
        curl \
        wget \
        git \
        build-essential \
        cmake \
        pkg-config \
        libssl-dev \
        libffi-dev \
        python3-dev \
        python3-pip \
        python3-venv
elif [[ "$OS" == "macos" ]]; then
    # 检查 Homebrew
    if ! command -v brew &> /dev/null; then
        echo "安装 Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    brew update
    brew install \
        curl \
        wget \
        git \
        cmake \
        pkg-config \
        openssl \
        libffi \
        python@3.11
fi

# 安装 ROS2
echo "安装 ROS2..."
if [[ "$OS" == "linux" ]]; then
    # Ubuntu/Debian ROS2 安装
    sudo apt update && sudo apt install -y curl gnupg lsb-release
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
    sudo apt update
    sudo apt install -y ros-humble-desktop
    
    # 设置环境
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
    source /opt/ros/humble/setup.bash
    
elif [[ "$OS" == "macos" ]]; then
    # macOS ROS2 安装
    brew install ros2
fi

# 安装 Docker 和 Docker Compose
echo "安装 Docker..."
if [[ "$OS" == "linux" ]]; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    
    # 安装 Docker Compose
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    
elif [[ "$OS" == "macos" ]]; then
    brew install --cask docker
fi

# 创建 Python 虚拟环境
echo "创建 Python 虚拟环境..."
python3 -m venv venv
source venv/bin/activate

# 升级 pip
pip install --upgrade pip

# 安装 Python 依赖
echo "安装 Python 依赖..."
pip install -r requirements.txt

# 创建必要的目录
echo "创建项目目录..."
mkdir -p logs data/images data/videos data/audio models

# 设置权限
chmod +x scripts/*.sh

# 初始化配置文件
echo "初始化配置文件..."
if [ ! -f config/main.yaml ]; then
    cp config/main.yaml.example config/main.yaml
    echo "请编辑 config/main.yaml 文件以配置您的系统"
fi

# 下载预训练模型
echo "下载预训练模型..."
python scripts/download_models.py

echo "=== 安装完成 ==="
echo ""
echo "下一步："
echo "1. 编辑配置文件: config/main.yaml"
echo "2. 启动服务: ./scripts/start.sh"
echo "3. 查看文档: docs/QUICK_START.md"
```

### 3.2 启动脚本 (scripts/start.sh)

```bash
#!/bin/bash

# RobotAgent 启动脚本
# 位置：scripts/start.sh

set -e

echo "=== 启动 RobotAgent 系统 ==="

# 检查配置文件
if [ ! -f config/main.yaml ]; then
    echo "错误: 配置文件 config/main.yaml 不存在"
    echo "请运行: cp config/main.yaml.example config/main.yaml"
    exit 1
fi

# 激活虚拟环境
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "已激活 Python 虚拟环境"
fi

# 设置 ROS2 环境
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
    echo "已设置 ROS2 环境"
fi

# 检查 Docker 服务
if ! docker info > /dev/null 2>&1; then
    echo "错误: Docker 服务未运行"
    echo "请启动 Docker 服务"
    exit 1
fi

# 启动基础服务
echo "启动基础服务..."
docker-compose -f deployment/single_machine/docker-compose.yml up -d redis milvus neo4j

# 等待服务启动
echo "等待服务启动..."
sleep 10

# 检查服务状态
echo "检查服务状态..."
python scripts/check_services.py

# 启动 RobotAgent
echo "启动 RobotAgent 主程序..."
export PYTHONPATH=$PWD/src:$PYTHONPATH
python main.py

echo "=== RobotAgent 启动完成 ==="
```

这个配置和部署文档提供了：

1. **完整的系统配置**：包括所有智能体、ROS2、内存系统、通信等配置
2. **智能体提示词**：为每个智能体定制的系统提示词
3. **部署架构**：单机和分布式两种部署方案
4. **自动化脚本**：安装和启动的自动化脚本

这样就完成了整个RobotAgent项目的核心文档体系，涵盖了架构设计、实现细节、配置部署等各个方面。