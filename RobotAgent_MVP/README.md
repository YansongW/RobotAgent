# RobotAgent MVP

基于豆包大模型的智能机器人对话系统，采用 ASR + 大模型 + TTS 三段式处理架构。

## 功能特性

- 🤖 **智能对话**: 基于豆包 Doubao-Seed-1.6 大模型的自然语言理解
- 🎤 **语音识别**: 集成豆包 ASR 服务，支持实时语音转文字
- 🔊 **语音合成**: 集成豆包 TTS 服务，支持文字转语音
- 💾 **记忆功能**: 本地化对话历史保存和上下文复用
- 🦾 **ROS2集成**: 支持机器人动作库管理和执行
- 🌐 **Web界面**: 现代化的对话界面，支持文本和语音交互

## 技术架构

```
ASR (语音识别) → 大模型 (对话生成) → TTS (语音合成)
                      ↓
              记忆服务 (上下文管理)
                      ↓
              ROS2服务 (动作执行)
```

## 项目结构

```
RobotAgent_MVP/
├── main.py                 # 主程序入口
├── config.json            # 配置文件
├── requirements.txt        # 依赖列表
├── services/              # 服务模块
│   ├── doubao_service.py  # 豆包大模型服务
│   ├── asr_service.py     # 语音识别服务
│   ├── tts_service.py     # 语音合成服务
│   ├── memory_service.py  # 记忆管理服务
│   └── ros2_service.py    # ROS2动作服务
├── prompts/               # 提示词模板
│   ├── system_prompt.json
│   ├── conversation_prompt.json
│   └── task_execution_prompt.json
├── ros2_actions/          # ROS2动作库
│   ├── basic_movements.py
│   └── manipulation_actions.py
├── static/                # 前端文件
│   ├── index.html
│   ├── style.css
│   └── app.js
├── memory/                # 记忆文件存储
├── logs/                  # 日志文件
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥

编辑 `config.json` 文件，填入您的豆包API密钥：

```json
{
  "doubao_api": {
    "api_key": "您的API密钥",
    "base_url": "https://ark.cn-beijing.volces.com/api/v3"
  }
}
```

### 3. 启动服务

```bash
python main.py
```

### 4. 访问界面

打开浏览器访问: http://localhost:8000

## API接口

### 文本对话
```
POST /chat/text
{
  "message": "用户消息",
  "conversation_id": "对话ID"
}
```

### 语音对话
```
POST /chat/voice
Content-Type: multipart/form-data
- audio: 音频文件
- conversation_id: 对话ID
```

### 获取对话历史
```
GET /chat/history/{conversation_id}
```

### 获取ROS2动作库
```
GET /ros2/actions
```

### 健康检查
```
GET /health
```

## 配置说明

### 豆包模型配置

- **Chat模型**: doubao-seed-1.6 (对话生成)
- **Embedding模型**: doubao-embedding-large (文本向量化)
- **Vision模型**: doubao-embedding-vision (图文向量化)

### ASR配置

- **引擎**: doubao-streaming (流式识别)
- **语言**: zh-CN (中文)
- **采样率**: 16000Hz

### TTS配置

- **语音**: BV700_streaming (流式合成)
- **语言**: zh-CN (中文)
- **音频格式**: mp3

## 记忆功能

系统支持多层次的记忆管理：

1. **对话历史**: 保存完整的用户-机器人对话记录
2. **上下文缓存**: 利用豆包的上下文缓存API提高响应效率
3. **Prompt复用**: 支持从记忆文件导入上下文到新对话

## ROS2集成

### 动作库管理

- 动态加载 `ros2_actions/` 目录下的Python文件
- 支持基础移动和机械臂操作动作
- 实时更新动作库（手动添加文件即可）

### 动作格式

```python
{
    "action_name": {
        "name": "动作名称",
        "description": "动作描述",
        "category": "动作类别",
        "parameters": {
            "param1": "参数描述"
        },
        "example": "使用示例"
    }
}
```

## 开发指南

### 添加新的ROS2动作

1. 在 `ros2_actions/` 目录下创建Python文件
2. 按照标准格式定义动作字典
3. 系统会自动加载新动作

### 自定义Prompt

1. 在 `prompts/` 目录下创建JSON文件
2. 按照模板格式定义提示词
3. 在服务中引用新的提示词

### 扩展服务

1. 在 `services/` 目录下创建新的服务文件
2. 在 `main.py` 中注册新服务
3. 添加相应的API路由

## 注意事项

- 确保API密钥有效且有足够的配额
- 语音功能需要浏览器支持麦克风访问
- 记忆文件会随着使用增长，建议定期清理
- ROS2功能目前为模拟实现，实际部署需要ROS2环境

## 许可证

MIT License

## 支持

如有问题，请查看日志文件或联系开发团队。