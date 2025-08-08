# RobotAgent MVP

基于CAMEL框架的三智能体机器人系统MVP版本

## 架构设计

```
Input → [Chat Agent + Action Agent + Memory Agent] → Output (TTS/Voice + Action File)
```

### 核心智能体

1. **Chat Agent (对话智能体)**
   - 处理自然语言理解和对话生成
   - 负责与用户的交互沟通

2. **Action Agent (动作智能体)**
   - 负责动作规划和执行逻辑
   - 处理具体的任务执行

3. **Memory Agent (记忆智能体)**
   - 处理学习和记忆功能
   - 存储对话历史、经验和知识

## 项目结构

```
RobotAgent_MVP/
├── src/
│   ├── agents/              # 三个核心智能体
│   ├── communication/       # 智能体通信
│   ├── memory/             # 记忆系统
│   ├── output/             # 输出处理
│   ├── utils/              # 工具模块
│   └── main.py             # 主程序
├── config/                 # 配置文件
├── tests/                  # 测试代码
├── requirements.txt        # 依赖包
└── README.md              # 说明文档
```

## 快速开始

1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 运行系统
```bash
python src/main.py
```

## 功能特性

- 基于CAMEL框架的多智能体协作
- 自然语言对话交互
- 动作规划和执行
- 记忆和学习能力
- TTS语音输出
- 动作文件生成