# API接口与开发指南

## 1. REST API 接口文档

### 1.1 API 基础信息

```yaml
# API 基础配置
base_url: "http://localhost:8000"
version: "v1"
authentication: "Bearer Token"
content_type: "application/json"
```

### 1.2 认证接口

#### 登录获取Token
```http
POST /api/v1/auth/login
Content-Type: application/json

{
    "username": "admin",
    "password": "password"
}

# 响应
{
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "token_type": "bearer",
    "expires_in": 1800
}
```

#### 刷新Token
```http
POST /api/v1/auth/refresh
Authorization: Bearer <access_token>

# 响应
{
    "access_token": "new_token_here",
    "token_type": "bearer",
    "expires_in": 1800
}
```

### 1.3 智能体交互接口

#### 发送消息给智能体
```http
POST /api/v1/agents/{agent_name}/message
Authorization: Bearer <access_token>
Content-Type: application/json

{
    "content": "请帮我规划一个从A点到B点的路径",
    "message_type": "task_request",
    "priority": 2,
    "metadata": {
        "user_id": "user123",
        "session_id": "session456"
    }
}

# 响应
{
    "message_id": "msg_789",
    "status": "sent",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

#### 获取智能体响应
```http
GET /api/v1/agents/{agent_name}/messages/{message_id}
Authorization: Bearer <access_token>

# 响应
{
    "message_id": "msg_789",
    "status": "completed",
    "response": {
        "content": "我已经为您规划了最优路径...",
        "result_data": {
            "path": [...],
            "estimated_time": 120,
            "distance": 50.5
        }
    },
    "timestamp": "2024-01-15T10:30:15Z"
}
```

#### 广播消息
```http
POST /api/v1/agents/broadcast
Authorization: Bearer <access_token>
Content-Type: application/json

{
    "content": "系统维护通知：将在30分钟后进行系统重启",
    "message_type": "system_notification",
    "priority": 3
}
```

### 1.4 机器人控制接口

#### 移动控制
```http
POST /api/v1/robot/move
Authorization: Bearer <access_token>
Content-Type: application/json

{
    "movement_type": "velocity",
    "parameters": {
        "linear": {"x": 0.5, "y": 0.0, "z": 0.0},
        "angular": {"x": 0.0, "y": 0.0, "z": 0.2},
        "duration": 5.0
    }
}

# 响应
{
    "task_id": "task_123",
    "status": "executing",
    "estimated_completion": "2024-01-15T10:30:20Z"
}
```

#### 导航到目标点
```http
POST /api/v1/robot/navigate
Authorization: Bearer <access_token>
Content-Type: application/json

{
    "target_position": {
        "x": 10.0,
        "y": 5.0,
        "z": 0.0
    },
    "target_orientation": {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "w": 1.0
    },
    "navigation_options": {
        "avoid_obstacles": true,
        "max_speed": 1.0,
        "timeout": 300
    }
}
```

#### 紧急停止
```http
POST /api/v1/robot/emergency_stop
Authorization: Bearer <access_token>
Content-Type: application/json

{
    "reason": "用户请求紧急停止"
}

# 响应
{
    "status": "stopped",
    "timestamp": "2024-01-15T10:30:00Z",
    "message": "机器人已紧急停止"
}
```

### 1.5 传感器数据接口

#### 获取传感器数据
```http
GET /api/v1/sensors/data?types=camera,lidar,imu
Authorization: Bearer <access_token>

# 响应
{
    "timestamp": "2024-01-15T10:30:00Z",
    "sensor_data": {
        "camera": {
            "image_url": "/api/v1/sensors/camera/latest",
            "resolution": [640, 480],
            "timestamp": "2024-01-15T10:29:59Z"
        },
        "lidar": {
            "ranges": [1.2, 1.5, 2.0, ...],
            "angle_min": -1.57,
            "angle_max": 1.57,
            "range_min": 0.1,
            "range_max": 10.0
        },
        "imu": {
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "linear_acceleration": {"x": 0.0, "y": 0.0, "z": 9.8}
        }
    }
}
```

#### 获取相机图像
```http
GET /api/v1/sensors/camera/latest
Authorization: Bearer <access_token>

# 响应: 图像文件 (JPEG/PNG)
```

### 1.6 记忆系统接口

#### 存储多模态记忆数据
```http
POST /api/v1/memory/store
Authorization: Bearer <access_token>
Content-Type: multipart/form-data

{
    "data_type": "image",
    "file": <image_file>,
    "metadata": {
        "description": "客厅场景图像",
        "location": "living_room",
        "timestamp": "2024-01-15T10:30:00Z",
        "agent_id": "agent_001",
        "task_id": "task_123"
    },
    "memory_category": "episodic_memory",  # 新增：记忆分类
    "importance_score": 0.8,               # 新增：重要性评分
    "storage_tier": "hot"                  # 新增：存储层级
}

# 响应
{
    "memory_id": "mem_456",
    "status": "stored",
    "embedding_id": "emb_789",
    "category": "episodic_memory",
    "storage_locations": {
        "vector_db": "collection_episodic",
        "graph_db": "node_12345",
        "object_storage": "bucket/mem_456.jpg",
        "tier": "hot_storage"
    },
    "workflow_id": "wf_789"  # LangGraph工作流ID
}
```

#### 检索相关记忆
```http
POST /api/v1/memory/search
Authorization: Bearer <access_token>
Content-Type: application/json

{
    "query": "客厅里的沙发",
    "query_type": "text",
    "limit": 10,
    "similarity_threshold": 0.7,
    "memory_categories": ["episodic_memory", "semantic_memory"],  # 新增：按类别过滤
    "time_range": {                                               # 新增：时间范围过滤
        "start": "2024-01-01T00:00:00Z",
        "end": "2024-01-15T23:59:59Z"
    },
    "importance_threshold": 0.5,                                  # 新增：重要性阈值
    "agent_id": "agent_001"                                       # 新增：按智能体过滤
}

# 响应
{
    "results": [
        {
            "memory_id": "mem_456",
            "similarity_score": 0.85,
            "data_type": "image",
            "category": "episodic_memory",
            "importance_score": 0.8,
            "metadata": {
                "description": "客厅场景图像",
                "location": "living_room",
                "agent_id": "agent_001",
                "task_id": "task_123"
            },
            "url": "/api/v1/memory/mem_456",
            "storage_tier": "hot"
        }
    ],
    "total_count": 25,
    "search_time_ms": 150
}
```

#### LangGraph记忆工作流管理
```http
POST /api/v1/memory/workflow/start
Authorization: Bearer <access_token>
Content-Type: application/json

{
    "workflow_type": "memory_processing",
    "input_data": {
        "memory_data": {...},
        "processing_options": {
            "enable_classification": true,
            "enable_importance_scoring": true,
            "enable_graph_update": true
        }
    },
    "checkpoint_config": {
        "enable_checkpoints": true,
        "checkpoint_interval": 5
    }
}

# 响应
{
    "workflow_id": "wf_789",
    "status": "running",
    "current_step": "classify_memory",
    "checkpoint_id": "cp_001"
}
```

#### 获取工作流状态
```http
GET /api/v1/memory/workflow/{workflow_id}/status
Authorization: Bearer <access_token>

# 响应
{
    "workflow_id": "wf_789",
    "status": "completed",
    "current_step": "update_knowledge_graph",
    "progress": 100,
    "steps_completed": [
        "classify_memory",
        "process_agent_memory",
        "store_memory",
        "update_knowledge_graph"
    ],
    "execution_time_ms": 2500,
    "checkpoint_id": "cp_005"
}
```

#### 记忆分类和重要性评分
```http
POST /api/v1/memory/classify
Authorization: Bearer <access_token>
Content-Type: application/json

{
    "content": "机器人在客厅成功完成了物品抓取任务",
    "metadata": {
        "data_type": "text",
        "agent_id": "agent_001",
        "task_id": "task_123",
        "timestamp": "2024-01-15T10:30:00Z"
    }
}

# 响应
{
    "classification": {
        "category": "task_experience",
        "modality": "text",
        "confidence": 0.92
    },
    "importance_score": 0.85,
    "scoring_factors": {
        "recency": 0.9,
        "frequency": 0.7,
        "relevance": 0.8,
        "uniqueness": 0.9
    },
    "storage_recommendation": {
        "tier": "warm",
        "retention_days": 90
    }
}
```

#### 知识图谱可视化
```http
GET /api/v1/memory/visualization/graph
Authorization: Bearer <access_token>
Query Parameters:
- memory_ids: mem_456,mem_789 (可选)
- categories: agent_memory,task_experience (可选)
- time_range: 2024-01-01,2024-01-15 (可选)
- max_nodes: 500 (可选)
- layout: force_directed (可选)

# 响应
{
    "graph_data": {
        "nodes": [
            {
                "id": "mem_456",
                "label": "客厅场景图像",
                "category": "episodic_memory",
                "importance": 0.8,
                "position": {"x": 100, "y": 200}
            }
        ],
        "edges": [
            {
                "source": "mem_456",
                "target": "mem_789",
                "relationship": "RELATED_TO",
                "weight": 0.7
            }
        ]
    },
    "statistics": {
        "total_nodes": 150,
        "total_edges": 300,
        "categories_count": {
            "agent_memory": 50,
            "task_experience": 60,
            "domain_knowledge": 40
        }
    }
}
```

#### 3D记忆空间可视化
```http
GET /api/v1/memory/visualization/3d
Authorization: Bearer <access_token>
Query Parameters:
- categories: agent_memory,task_experience
- max_points: 1000
- color_scheme: category

# 响应
{
    "visualization_data": {
        "points": [
            {
                "id": "mem_456",
                "coordinates": {"x": 0.5, "y": 0.3, "z": 0.8},
                "color": "#FF5733",
                "size": 10,
                "label": "客厅场景图像",
                "category": "episodic_memory"
            }
        ],
        "clusters": [
            {
                "center": {"x": 0.4, "y": 0.3, "z": 0.7},
                "radius": 0.2,
                "category": "episodic_memory",
                "point_count": 25
            }
        ]
    },
    "metadata": {
        "total_points": 500,
        "dimension_reduction": "umap",
        "clustering_algorithm": "dbscan"
    }
}
```

#### 分层存储管理
```http
POST /api/v1/memory/storage/migrate
Authorization: Bearer <access_token>
Content-Type: application/json

{
    "memory_id": "mem_456",
    "source_tier": "hot",
    "target_tier": "warm",
    "reason": "access_frequency_decreased"
}

# 响应
{
    "migration_id": "mig_123",
    "status": "completed",
    "source_tier": "hot",
    "target_tier": "warm",
    "migration_time_ms": 500
}
```

#### 获取存储统计信息
```http
GET /api/v1/memory/storage/stats
Authorization: Bearer <access_token>

# 响应
{
    "storage_tiers": {
        "hot": {
            "total_memories": 1000,
            "size_mb": 512,
            "utilization": 0.5,
            "avg_access_frequency": 15.2
        },
        "warm": {
            "total_memories": 5000,
            "size_gb": 50,
            "utilization": 0.5,
            "avg_access_frequency": 3.1
        },
        "cold": {
            "total_memories": 20000,
            "size_gb": 200,
            "utilization": 0.2,
            "avg_access_frequency": 0.5
        },
        "archive": {
            "total_memories": 100000,
            "size_gb": 1000,
            "utilization": 0.1,
            "avg_access_frequency": 0.01
        }
    },
    "total_memories": 126000,
    "total_size_gb": 1250.5
}
```

### 1.7 系统状态接口

#### 获取系统状态
```http
GET /api/v1/system/status
Authorization: Bearer <access_token>

# 响应
{
    "system_status": "running",
    "agents": {
        "DialogAgent": {"status": "active", "load": 0.3},
        "PlanningAgent": {"status": "active", "load": 0.5},
        "ROS2Agent": {"status": "active", "load": 0.2}
    },
    "robot_status": {
        "position": {"x": 1.0, "y": 2.0, "z": 0.0},
        "battery_level": 85,
        "system_health": "good"
    },
    "services": {
        "redis": {"status": "connected"},
        "milvus": {"status": "connected"},
        "neo4j": {"status": "connected"}
    }
}
```

#### 获取性能指标
```http
GET /api/v1/system/metrics
Authorization: Bearer <access_token>

# 响应
{
    "timestamp": "2024-01-15T10:30:00Z",
    "metrics": {
        "cpu_usage": 45.2,
        "memory_usage": 68.5,
        "disk_usage": 32.1,
        "network_io": {
            "bytes_sent": 1024000,
            "bytes_received": 2048000
        },
        "agent_performance": {
            "average_response_time": 1.2,
            "message_throughput": 150,
            "error_rate": 0.01
        }
    }
}
```

## 2. WebSocket 实时接口

### 2.1 连接WebSocket
```javascript
// 连接WebSocket
const ws = new WebSocket('ws://localhost:8000/ws');

// 认证
ws.onopen = function() {
    ws.send(JSON.stringify({
        type: 'auth',
        token: 'your_access_token'
    }));
};
```

### 2.2 实时消息订阅
```javascript
// 订阅智能体消息
ws.send(JSON.stringify({
    type: 'subscribe',
    channels: ['agent_messages', 'robot_status', 'sensor_data']
}));

// 接收消息
ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    
    switch(message.type) {
        case 'agent_message':
            console.log('智能体消息:', message.data);
            break;
        case 'robot_status':
            console.log('机器人状态:', message.data);
            break;
        case 'sensor_data':
            console.log('传感器数据:', message.data);
            break;
    }
};
```

## 3. Python SDK

### 3.1 安装SDK
```bash
pip install robotagent-sdk
```

### 3.2 基础使用
```python
from robotagent_sdk import RobotAgentClient

# 创建客户端
client = RobotAgentClient(
    base_url="http://localhost:8000",
    username="admin",
    password="password"
)

# 发送消息给智能体
response = client.send_message(
    agent_name="DialogAgent",
    content="你好，请介绍一下你的功能",
    message_type="text"
)

print(f"响应: {response.content}")

# 控制机器人移动
task = client.robot.move(
    movement_type="velocity",
    linear={"x": 0.5, "y": 0.0, "z": 0.0},
    angular={"x": 0.0, "y": 0.0, "z": 0.2},
    duration=5.0
)

print(f"移动任务ID: {task.task_id}")

# 获取传感器数据
sensor_data = client.sensors.get_data(types=["camera", "lidar"])
print(f"传感器数据: {sensor_data}")

# 存储多模态记忆数据
memory_id = client.memory.store(
    data_type="image",
    file_path="path/to/image.jpg",
    metadata={
        "description": "客厅场景",
        "location": "living_room",
        "agent_id": "agent_001",
        "task_id": "task_123"
    },
    memory_category="episodic_memory",
    importance_score=0.8,
    storage_tier="hot"
)

print(f"存储的记忆ID: {memory_id}")

# 搜索记忆（增强版）
memories = client.memory.search(
    query="客厅里的沙发",
    query_type="text",
    memory_categories=["episodic_memory", "semantic_memory"],
    time_range={
        "start": "2024-01-01T00:00:00Z",
        "end": "2024-01-15T23:59:59Z"
    },
    importance_threshold=0.5,
    agent_id="agent_001",
    limit=10
)

print(f"找到 {len(memories)} 条相关记忆")

# LangGraph记忆工作流管理
workflow = client.memory.start_workflow(
    workflow_type="memory_processing",
    input_data={
        "memory_data": {...},
        "processing_options": {
            "enable_classification": True,
            "enable_importance_scoring": True,
            "enable_graph_update": True
        }
    }
)

print(f"工作流ID: {workflow.workflow_id}")

# 检查工作流状态
status = client.memory.get_workflow_status(workflow.workflow_id)
print(f"工作流状态: {status.status}, 进度: {status.progress}%")

# 记忆分类和重要性评分
classification = client.memory.classify(
    content="机器人在客厅成功完成了物品抓取任务",
    metadata={
        "data_type": "text",
        "agent_id": "agent_001",
        "task_id": "task_123"
    }
)

print(f"记忆分类: {classification.category}, 重要性: {classification.importance_score}")

# 知识图谱可视化
graph_data = client.memory.visualize_graph(
    memory_ids=["mem_456", "mem_789"],
    categories=["agent_memory", "task_experience"],
    time_range=("2024-01-01", "2024-01-15"),
    max_nodes=500,
    layout="force_directed"
)

print(f"图谱节点数: {len(graph_data.nodes)}, 边数: {len(graph_data.edges)}")

# 3D记忆空间可视化
visualization_3d = client.memory.visualize_3d(
    categories=["agent_memory", "task_experience"],
    max_points=1000,
    color_scheme="category"
)

print(f"3D可视化点数: {len(visualization_3d.points)}")

# 分层存储管理
migration = client.memory.migrate_storage(
    memory_id="mem_456",
    source_tier="hot",
    target_tier="warm",
    reason="access_frequency_decreased"
)

print(f"存储迁移状态: {migration.status}")

# 获取存储统计信息
storage_stats = client.memory.get_storage_stats()
print(f"总记忆数: {storage_stats.total_memories}")
print(f"热存储使用率: {storage_stats.storage_tiers.hot.utilization}")

# 记忆系统监控
monitor = client.memory.create_monitor()
monitor.on_memory_stored = lambda memory_id: print(f"新记忆存储: {memory_id}")
monitor.on_memory_accessed = lambda memory_id: print(f"记忆被访问: {memory_id}")
monitor.start()
```

### 3.3 异步使用
```python
import asyncio
from robotagent_sdk import AsyncRobotAgentClient

async def main():
    # 创建异步客户端
    client = AsyncRobotAgentClient(
        base_url="http://localhost:8000",
        username="admin",
        password="password"
    )
    
    # 异步发送消息
    response = await client.send_message(
        agent_name="PlanningAgent",
        content="请规划从A到B的路径",
        message_type="task_request"
    )
    
    # 异步控制机器人
    task = await client.robot.navigate(
        target_position={"x": 10.0, "y": 5.0, "z": 0.0}
    )
    
    # 等待任务完成
    result = await client.robot.wait_for_task(task.task_id)
    print(f"导航结果: {result}")

# 运行异步代码
asyncio.run(main())
```

## 4. 开发指南

### 4.1 添加新的智能体

#### 步骤1：创建智能体类
```python
# src/camel_agents/custom_agent.py

from .base_agent import BaseRobotAgent
from typing import Dict, Any

class CustomAgent(BaseRobotAgent):
    """自定义智能体"""
    
    def __init__(self, model_backend, message_bus, config: Dict[str, Any] = None):
        system_message = """
        你是一个自定义智能体，负责...
        """
        super().__init__("CustomAgent", model_backend, system_message, message_bus, config)
        self.capabilities = [
            "custom_capability_1",
            "custom_capability_2"
        ]
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行自定义任务"""
        task_type = task.get("type")
        
        if task_type == "custom_task":
            return await self._handle_custom_task(task)
        else:
            return await super().execute_task(task)
    
    async def _handle_custom_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """处理自定义任务"""
        # 实现自定义逻辑
        return {
            "success": True,
            "result": "自定义任务完成"
        }
```

#### 步骤2：注册智能体
```python
# src/agent_manager.py

from camel_agents.custom_agent import CustomAgent

class AgentManager:
    def __init__(self, config, message_bus):
        # ... 现有代码
        
        # 添加自定义智能体
        if config.get("agents", {}).get("custom_agent", {}).get("enabled", False):
            self.agents["CustomAgent"] = CustomAgent(
                model_backend=self._create_model_backend(config["agents"]["custom_agent"]),
                message_bus=message_bus,
                config=config["agents"]["custom_agent"]
            )
```

#### 步骤3：添加配置
```yaml
# config/main.yaml

agents:
  custom_agent:
    enabled: true
    model_backend: "gpt-4"
    max_tokens: 2048
    temperature: 0.7
    system_prompt_file: "prompts/custom_agent.txt"
    capabilities:
      - "custom_capability_1"
      - "custom_capability_2"
```

### 4.2 添加新的API端点

#### 步骤1：创建路由
```python
# src/api/routes/custom_routes.py

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from ..dependencies import get_current_user, get_agent_manager

router = APIRouter(prefix="/api/v1/custom", tags=["custom"])

@router.post("/custom_action")
async def custom_action(
    request_data: Dict[str, Any],
    current_user = Depends(get_current_user),
    agent_manager = Depends(get_agent_manager)
):
    """自定义动作端点"""
    try:
        # 获取自定义智能体
        custom_agent = agent_manager.get_agent("CustomAgent")
        
        # 执行自定义任务
        result = await custom_agent.execute_task({
            "type": "custom_task",
            "parameters": request_data
        })
        
        return {
            "success": True,
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### 步骤2：注册路由
```python
# src/api/main.py

from .routes import custom_routes

app = FastAPI()

# 注册自定义路由
app.include_router(custom_routes.router)
```

### 4.3 扩展记忆系统

#### 添加新的数据类型
```python
# src/memory_system/processors/custom_processor.py

from .base_processor import BaseProcessor
import numpy as np

class CustomDataProcessor(BaseProcessor):
    """自定义数据处理器"""
    
    def __init__(self, config):
        super().__init__(config)
        self.supported_types = ["custom_data"]
    
    async def process(self, data: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """处理自定义数据"""
        # 实现自定义数据处理逻辑
        embedding = self._generate_embedding(data)
        
        return {
            "embedding": embedding,
            "processed_data": data,
            "metadata": metadata
        }
    
    def _generate_embedding(self, data: Any) -> np.ndarray:
        """生成自定义数据的嵌入向量"""
        # 实现嵌入生成逻辑
        return np.random.rand(512)  # 示例
```

### 4.4 测试指南

#### 单元测试
```python
# tests/test_custom_agent.py

import pytest
from unittest.mock import Mock, AsyncMock
from src.camel_agents.custom_agent import CustomAgent

@pytest.fixture
def custom_agent():
    model_backend = Mock()
    message_bus = Mock()
    config = {"test": True}
    return CustomAgent(model_backend, message_bus, config)

@pytest.mark.asyncio
async def test_custom_task_execution(custom_agent):
    """测试自定义任务执行"""
    task = {
        "type": "custom_task",
        "parameters": {"test_param": "test_value"}
    }
    
    result = await custom_agent.execute_task(task)
    
    assert result["success"] is True
    assert "result" in result
```

#### 集成测试
```python
# tests/test_api_integration.py

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_custom_action_endpoint():
    """测试自定义动作端点"""
    # 首先获取认证token
    login_response = client.post("/api/v1/auth/login", json={
        "username": "test_user",
        "password": "test_password"
    })
    token = login_response.json()["access_token"]
    
    # 调用自定义端点
    response = client.post(
        "/api/v1/custom/custom_action",
        json={"test_param": "test_value"},
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 200
    assert response.json()["success"] is True
```

## 5. 部署和运维

### 5.1 Docker化部署

#### Dockerfile
```dockerfile
# Dockerfile

FROM python:3.11-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 设置环境变量
ENV PYTHONPATH=/app/src

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "main.py"]
```

### 5.2 监控和日志

#### 日志配置
```python
# src/utils/logging_config.py

import logging
import logging.handlers
from pathlib import Path

def setup_logging(config):
    """设置日志配置"""
    log_level = getattr(logging, config.get("log_level", "INFO"))
    log_format = config.get("log_format", 
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # 创建日志目录
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 配置根日志器
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.handlers.RotatingFileHandler(
                log_dir / "robot_agent.log",
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
        ]
    )
    
    # 配置特定模块的日志级别
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
```

这个API接口与开发指南文档提供了：

1. **完整的REST API接口**：包括认证、智能体交互、机器人控制、传感器数据、记忆系统等
2. **WebSocket实时接口**：支持实时消息推送和状态更新
3. **Python SDK**：提供同步和异步的客户端库
4. **开发指南**：详细说明如何扩展系统功能
5. **测试和部署**：包括单元测试、集成测试和Docker化部署

至此，我已经为您创建了完整的RobotAgent项目文档体系，涵盖了：

- 架构设计文档
- 智能体实现指南
- 多模态记忆系统
- 系统配置与部署
- API接口与开发指南

这些文档详细说明了如何将CAMEL.AI和ROS2框架结合，构建一个具备高级认知、规划、协作能力的智能机器人系统。