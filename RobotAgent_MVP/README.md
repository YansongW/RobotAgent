# RobotAgent MVP - 最小可行性验证项目

## 项目概述

RobotAgent MVP是基于主项目架构的最小可行性验证子项目，实现自然语言到ROS2机械臂控制的完整流程。项目采用前后端分离架构，使用Qwen系列模型进行自然语言处理，通过消息管道实现异步通信。

## 核心功能

### 🎯 主要功能
- **自然语言解析**: 使用Qwen模型将自然语言转换为标准JSON格式
- **异步消息管道**: 基于Redis的消息队列实现Agent间通信
- **记忆Agent**: 记录完整交互历史，生成Markdown格式的本地文档
- **ROS2Agent**: 解析指令并控制睿尔曼机械臂（Gazebo仿真）
- **完整日志系统**: 记录每次交互的输入输出、时间戳和延迟信息
- **Web界面**: 提供用户友好的前端交互界面

### 🏗️ 技术架构

```
RobotAgent_MVP/
├── backend/                    # 后端服务
│   ├── app.py                 # FastAPI主应用
│   ├── models/                # 数据模型
│   │   ├── __init__.py
│   │   ├── message_models.py  # 消息数据模型
│   │   └── response_models.py # 响应数据模型
│   ├── services/              # 核心服务
│   │   ├── __init__.py
│   │   ├── qwen_service.py    # Qwen模型服务
│   │   ├── message_queue.py   # 消息队列服务
│   │   ├── memory_agent.py    # 记忆Agent
│   │   └── ros2_agent.py      # ROS2Agent
│   ├── utils/                 # 工具模块
│   │   ├── __init__.py
│   │   ├── logger.py          # 日志工具
│   │   └── config.py          # 配置管理
│   └── requirements.txt       # Python依赖
├── frontend/                  # 前端界面
│   ├── index.html            # 主页面
│   ├── static/               # 静态资源
│   │   ├── css/
│   │   │   └── style.css     # 样式文件
│   │   └── js/
│   │       └── app.js        # 前端逻辑
│   └── templates/            # 模板文件
├── logs/                     # 日志文件目录
├── memory_records/           # 记忆记录目录
├── config/                   # 配置文件
│   ├── config.yaml          # 主配置文件
│   └── ros2_commands.json   # ROS2命令映射
├── scripts/                  # 脚本文件
│   ├── start_services.sh    # 启动脚本
│   └── setup_environment.sh # 环境配置脚本
├── tests/                    # 测试文件
│   ├── test_qwen_service.py
│   ├── test_message_queue.py
│   └── test_integration.py
├── docker-compose.yml        # Docker编排文件
├── Dockerfile               # Docker镜像文件
└── README.md               # 项目说明文档
```

## 系统流程

### 🔄 数据流程图

```
用户输入 → Web前端 → FastAPI后端 → Qwen模型解析 → JSON格式化
                                                        ↓
记忆Agent ← 消息队列(Redis) ← JSON消息 ← 消息分发器
    ↓                              ↓
本地MD文档                    ROS2Agent
                                ↓
                        睿尔曼机械臂控制(Gazebo)
```

### 📋 处理步骤

1. **用户输入**: 通过Web界面输入自然语言指令
2. **模型解析**: Qwen模型将自然语言转换为结构化JSON
3. **消息分发**: 通过Redis消息队列异步分发给各Agent
4. **记忆存储**: Memory Agent记录完整交互历史
5. **机械臂控制**: ROS2 Agent解析指令并控制机械臂
6. **日志记录**: 完整记录处理过程和性能指标

## JSON消息格式规范

### 输入消息格式
```json
{
  "user_id": "user_123",
  "timestamp": "2024-01-15T10:30:00Z",
  "session_id": "session_456",
  "input_text": "请让机械臂移动到位置(0.3, 0.2, 0.5)",
  "language": "zh-CN"
}
```

### 解析后的标准格式
```json
{
  "message_id": "msg_789",
  "timestamp": "2024-01-15T10:30:01Z",
  "user_id": "user_123",
  "session_id": "session_456",
  "intent": "robot_control",
  "action": "move_to_position",
  "parameters": {
    "target_position": {
      "x": 0.3,
      "y": 0.2,
      "z": 0.5
    },
    "coordinate_frame": "base_link",
    "speed": "normal",
    "precision": "high"
  },
  "priority": "normal",
  "requires_confirmation": false,
  "estimated_duration": 5.0
}
```

### ROS2命令映射
```json
{
  "move_to_position": {
    "ros2_command": "ros2 action send_goal",
    "action_type": "/rm_driver/move_to_pose",
    "parameters_mapping": {
      "target_position.x": "pose.position.x",
      "target_position.y": "pose.position.y", 
      "target_position.z": "pose.position.z"
    }
  }
}
```

## 环境要求

### 系统要求
- **操作系统**: Ubuntu 20.04/22.04 或 Windows 10/11
- **Python**: 3.8+
- **ROS2**: Humble Hawksbill
- **Redis**: 6.0+
- **内存**: 最少8GB RAM
- **存储**: 最少10GB可用空间

### 依赖服务
- **Qwen API**: 通义千问模型API访问
- **Redis**: 消息队列服务
- **Gazebo**: 机械臂仿真环境
- **睿尔曼ROS2包**: rm_robot功能包

## 快速开始

### 1. 环境配置
```bash
# 克隆项目
cd RobotAgent_MVP

# 安装Python依赖
pip install -r backend/requirements.txt

# 配置环境变量
cp config/config.yaml.template config/config.yaml
# 编辑config.yaml，填入Qwen API密钥等配置

# 启动Redis服务
sudo systemctl start redis-server

# 配置ROS2环境
source /opt/ros/humble/setup.bash
```

### 2. 启动服务
```bash
# 启动后端服务
cd backend
python app.py

# 启动Gazebo仿真（新终端）
source ~/ros2_ws/install/setup.bash
ros2 launch rm_gazebo gazebo_65_demo.launch.py

# 启动MoveIt2（新终端）
ros2 launch rm_65_config gazebo_moveit_demo.launch.py
```

### 3. 访问界面
- **Web界面**: http://localhost:8000
- **API文档**: http://localhost:8000/docs
- **日志查看**: logs/目录下的日志文件
- **记忆记录**: memory_records/目录下的Markdown文件

## API接口

### 主要端点
- `POST /api/v1/process_command` - 处理自然语言指令
- `GET /api/v1/status` - 获取系统状态
- `GET /api/v1/logs` - 获取日志信息
- `GET /api/v1/memory_records` - 获取记忆记录
- `WebSocket /ws` - 实时状态推送

### 使用示例
```bash
# 发送控制指令
curl -X POST "http://localhost:8000/api/v1/process_command" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "input_text": "让机械臂移动到桌子上方",
    "session_id": "test_session"
  }'
```

## 日志系统

### 日志级别
- **DEBUG**: 详细的调试信息
- **INFO**: 一般信息记录
- **WARNING**: 警告信息
- **ERROR**: 错误信息
- **CRITICAL**: 严重错误

### 日志文件
- `logs/app.log` - 应用主日志
- `logs/qwen_service.log` - Qwen服务日志
- `logs/ros2_agent.log` - ROS2 Agent日志
- `logs/memory_agent.log` - Memory Agent日志
- `logs/performance.log` - 性能指标日志

## 测试

### 单元测试
```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定测试
python -m pytest tests/test_qwen_service.py -v
```

### 集成测试
```bash
# 端到端测试
python tests/test_integration.py
```

## 性能指标

### 关键指标
- **响应时间**: Qwen模型调用延迟
- **处理时间**: 完整流程处理时间
- **成功率**: 指令解析和执行成功率
- **并发能力**: 同时处理的请求数量

### 监控方式
- 实时日志监控
- 性能指标统计
- Web界面状态显示
- 告警机制

## 故障排除

### 常见问题
1. **Qwen API调用失败**: 检查API密钥和网络连接
2. **Redis连接失败**: 确认Redis服务运行状态
3. **ROS2通信问题**: 检查ROS2环境配置
4. **Gazebo启动失败**: 确认显卡驱动和依赖包

### 调试方法
- 查看详细日志文件
- 使用API测试工具
- 检查系统资源使用
- 验证配置文件正确性

## 扩展计划

### 短期目标
- [ ] 支持更多机械臂动作类型
- [ ] 增加语音输入功能
- [ ] 优化响应速度
- [ ] 添加安全检查机制

### 长期目标
- [ ] 集成视觉感知
- [ ] 支持多机械臂协作
- [ ] 添加学习能力
- [ ] 云端部署支持

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。

## 许可证

本项目采用MIT许可证。