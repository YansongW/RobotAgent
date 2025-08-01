# RobotAgent MVP

基于 Qwen 大模型的智能机器人聊天控制系统 MVP 版本。

## 🚀 功能特性

- 🤖 **智能聊天**: 基于 Qwen 大模型的自然语言对话功能
- 🎯 **机器人控制**: 将自然语言指令转换为 ROS2 机器人控制命令
- 🔄 **多 Agent 架构**: QwenService、ROS2Agent、MemoryAgent 协同工作
- 📊 **实时状态反馈**: 机器人执行状态实时转换为自然语言反馈
- 💬 **测试模式**: 无需真实 API 密钥即可运行测试
- 📝 **记忆管理**: 自动记录和管理交互历史

## 🏗️ 项目结构

```
RobotAgent_MVP/
├── backend/                    # 后端服务
│   ├── app.py                 # FastAPI 主应用
│   ├── agents/                # Agent 模块
│   │   ├── memory_agent.py    # 记忆 Agent
│   │   └── ros2_agent.py      # ROS2 Agent
│   ├── services/              # 核心服务
│   │   ├── qwen_service.py    # Qwen 模型服务
│   │   └── message_queue.py   # 消息队列服务
│   ├── models/                # 数据模型
│   │   ├── message_models.py  # 消息数据模型
│   │   └── response_models.py # 响应数据模型
│   ├── utils/                 # 工具模块
│   │   ├── config.py          # 配置管理
│   │   └── logger.py          # 日志工具
│   ├── static/                # 静态文件
│   │   └── chat_test.html     # 聊天测试页面
│   └── requirements.txt       # Python 依赖
├── config/                    # 配置文件
│   ├── config.yaml           # 主配置文件
│   └── ros2_commands.json    # ROS2 命令映射
├── frontend/                 # 前端界面
│   └── index.html           # 主页面
├── logs/                    # 日志文件目录
├── memory_records/          # 记忆记录目录
├── docker-compose.yml       # Docker 编排文件
├── Dockerfile              # Docker 镜像文件
├── setup_env.bat          # 环境配置脚本
├── start.bat              # Windows 启动脚本
└── start.sh               # Linux/Mac 启动脚本
```

## 🔄 系统架构

### 聊天控制流程

```
用户输入 → Web界面 → FastAPI后端 → QwenService
                                        ↓
                                   解析为JSON格式
                                   {
                                     "user_reply": "自然语言回复",
                                     "ros2_command": "ROS2指令"
                                   }
                                        ↓
                              ROS2Agent处理指令
                                        ↓
                              执行状态反馈给用户
                                        ↓
                              MemoryAgent记录历史
```

### 核心组件

1. **QwenService**: 处理自然语言对话，生成机器人控制指令
2. **ROS2Agent**: 解析和执行 ROS2 机器人控制命令
3. **MemoryAgent**: 记录完整的交互历史和状态
4. **MessageQueue**: 组件间异步消息传递

## 🚀 快速开始

### 1. 环境准备

确保您的系统已安装：
- Python 3.8+
- Redis (可选，用于消息队列)

### 2. 安装依赖

```bash
cd RobotAgent_MVP
pip install -r backend/requirements.txt
```

### 3. 配置 API 密钥（可选）

如需使用真实的 Qwen API：

```bash
# Windows
set DASHSCOPE_API_KEY=your_api_key_here

# Linux/Mac
export DASHSCOPE_API_KEY=your_api_key_here
```

获取 API 密钥：访问 [阿里云百炼平台](https://bailian.console.aliyun.com/)

### 4. 启动系统

```bash
# 方式1: 使用启动脚本
# Windows
start.bat

# Linux/Mac
./start.sh

# 方式2: 手动启动
cd backend
python app.py
```

### 5. 访问界面

- **聊天测试页面**: http://localhost:8000/static/chat_test.html
- **主界面**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs

## 💬 使用说明

### 聊天测试界面

访问 http://localhost:8000/static/chat_test.html 进行测试：

#### 支持的指令类型

1. **移动控制**
   - "前进"、"move forward"
   - "后退"、"backward"
   - "左转"、"右转"

2. **夹爪控制**
   - "抓取物体"、"grab"
   - "打开夹爪"、"gripper open"
   - "关闭夹爪"、"gripper close"

3. **机械臂控制**
   - "控制机械臂"、"arm control"
   - "移动关节"、"joint movement"

4. **状态查询**
   - "查询状态"、"status"
   - "获取位置"、"position"

5. **普通聊天**
   - 任何其他对话内容

### API 接口

#### 主要端点

- `POST /api/chat` - 新的聊天接口
- `POST /api/process-command` - 兼容旧版本的命令处理
- `GET /api/status` - 获取系统状态
- `GET /health` - 健康检查

#### 聊天 API 示例

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "让机器人前进",
    "conversation_id": "test-conversation"
  }'
```

响应格式：
```json
{
  "response": "好的，我将执行移动指令：让机器人前进",
  "ros2_command": {
    "command_type": "movement",
    "topic": "/cmd_vel",
    "message_type": "geometry_msgs/Twist",
    "data": {
      "linear": {"x": 0.5, "y": 0.0, "z": 0.0},
      "angular": {"x": 0.0, "y": 0.0, "z": 0.0}
    }
  },
  "conversation_id": "test-conversation"
}
```

## ⚙️ 配置说明

### 主配置文件: `config/config.yaml`

```yaml
# Qwen 模型配置
qwen:
  api_key: "${DASHSCOPE_API_KEY}"
  model_name: "qwen-max-latest"
  max_tokens: 2048
  temperature: 0.7

# 服务器配置
server:
  host: "0.0.0.0"
  port: 8000
  debug: false

# 记忆 Agent 配置
memory_agent:
  storage_path: "./memory_records"
  max_records: 1000

# ROS2 Agent 配置
ros2_agent:
  command_mapping_file: "./config/ros2_commands.json"
  default_timeout: 30
```

### ROS2 命令映射: `config/ros2_commands.json`

定义了自然语言指令到 ROS2 命令的映射关系。

## 🔧 开发指南

### 测试模式

系统支持测试模式，无需真实 API 密钥：

- 未设置 `DASHSCOPE_API_KEY` 时自动启用
- 基于关键词匹配生成模拟响应
- 支持所有基本功能测试

### 扩展新功能

1. **添加新的指令类型**
   - 修改 `qwen_service.py` 中的 `_create_test_response` 方法
   - 更新 `ros2_agent.py` 中的 `_execute_ros2_command` 方法

2. **自定义 Agent**
   - 在 `agents/` 目录下创建新的 Agent 类
   - 在 `app.py` 中注册新 Agent

3. **修改响应格式**
   - 更新 `models/` 目录下的数据模型
   - 调整相应的处理逻辑

## 📊 日志系统

### 日志文件位置

- `backend/logs/` - 后端服务日志
- `logs/` - 项目根目录日志

### 主要日志文件

- `FastAPI.log` - 主应用日志
- `QwenService.log` - Qwen 服务日志
- `ROS2Agent.log` - ROS2 Agent 日志
- `MemoryAgent.log` - 记忆 Agent 日志
- `MessageQueue.log` - 消息队列日志

## 🐳 Docker 部署

```bash
# 构建镜像
docker build -t robotagent-mvp .

# 运行容器
docker run -p 8000:8000 -e DASHSCOPE_API_KEY=your_key robotagent-mvp

# 使用 docker-compose
docker-compose up -d
```

## 🔍 故障排除

### 常见问题

1. **服务启动失败**
   - 检查端口 8000 是否被占用
   - 确认 Python 依赖已正确安装

2. **API 调用失败**
   - 检查网络连接
   - 验证 API 密钥（如果使用真实 API）

3. **ROS2 环境检查失败**
   - 在 Windows 环境下这是正常现象
   - 不影响系统基本功能

### 调试方法

- 查看 `backend/logs/` 目录下的日志文件
- 使用 `/health` 端点检查服务状态
- 在测试模式下验证基本功能

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 📞 联系方式

如有问题，请通过 GitHub Issues 联系。