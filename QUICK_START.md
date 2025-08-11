# RobotAgent 快速开始指南 (MVP阶段)

## ⚠️ 重要说明

**当前状态**: 本项目处于MVP (最小可行产品) 开发阶段

目前可用的功能有限，主要包括火山方舟Chat API测试工具和基础配置系统。大部分高级功能仍在设计和开发中。

## 项目愿景

RobotAgent计划成为一个集成了CAMEL.AI多智能体框架和ROS2机器人操作系统的智能机器人系统，采用"大脑（CAMEL认知）+ 小脑（ROS2控制）"的架构设计。

### 规划中的核心组件
- **多个CAMEL智能体**: Dialog、Planning、Decision、Perception、Learning、ROS2 (设计阶段)
- **多模态记忆系统**: Milvus向量数据库 + Neo4j知识图谱 (设计阶段)
- **ROS2接口**: 标准化的机器人控制接口 (设计阶段)
- **通信总线**: 基于Redis的分布式消息系统 (设计阶段)

## 当前环境要求 (MVP阶段)

### 最低要求
- **操作系统**: Windows 10/11, macOS, 或 Linux
- **Python**: 3.8+
- **内存**: 4GB以上
- **存储**: 1GB可用空间

### 推荐配置
- **内存**: 8GB以上
- **网络**: 稳定的互联网连接（用于API调用）

## 快速开始

### 1. 克隆项目
```bash
git clone <repository-url>
cd RobotAgent
```

### 2. 进入MVP开发目录
```bash
cd RobotAgent_MVP
```

### 3. 安装Python依赖
```bash
pip install -r requirements.txt
```

主要依赖包：
- `openai>=1.0.0` - 用于火山方舟API调用
- `pyyaml>=6.0` - 配置文件解析
- `camel-ai>=0.1.0` - CAMEL框架 (基础)

### 4. 配置API密钥
```bash
# 复制配置模板
cp config/api_config.yaml.template config/api_config.yaml

# 编辑配置文件，填入您的火山方舟API密钥
# 使用您喜欢的文本编辑器编辑 config/api_config.yaml
```

### 5. 运行火山方舟Chat API测试工具
```bash
python tests/volcengine_chat_test.py
```

## 配置说明 (MVP阶段)

### API配置文件 (config/api_config.yaml)

这是当前唯一需要配置的文件：

```yaml
volcengine:
  # 火山方舟API配置
  api_key: "your-volcengine-api-key-here"  # 替换为您的真实API密钥
  base_url: "https://ark.cn-beijing.volces.com/api/v3"
  default_model: "doubao-seed-1-6-250615"
  
  # API调用参数
  temperature: 0.7
  max_tokens: 2000
  
  # 对话历史管理
  max_history_turns: 10  # 保持最近N轮对话

# 其他API配置示例 (可选)
# openai:
#   api_key: "your-openai-api-key-here"
#   base_url: "https://api.openai.com/v1"
#   default_model: "gpt-3.5-turbo"
#   temperature: 0.7
#   max_tokens: 2000
```

### 配置步骤

1. **复制配置模板**:
   ```bash
   cp config/api_config.yaml.template config/api_config.yaml
   ```

2. **编辑配置文件**:
   - 打开 `config/api_config.yaml`
   - 将 `your-volcengine-api-key-here` 替换为您的真实API密钥
   - 根据需要调整其他参数

3. **验证配置**:
   ```bash
   python tests/volcengine_chat_test.py
   ```

### 安全注意事项

- ⚠️ **重要**: `api_config.yaml` 包含敏感信息，已添加到 `.gitignore`
- 🔒 **不要将包含真实API密钥的配置文件提交到版本控制系统**
- 📋 使用 `api_config.yaml.template` 作为配置模板分享给其他开发者

## 验证安装 (MVP阶段)

### 1. 验证Python环境
```bash
# 检查Python版本
python --version

# 检查依赖包安装
pip list | grep -E "(requests|pyyaml)"
```

### 2. 验证配置文件
```bash
# 检查配置文件是否存在
ls -la config/api_config.yaml

# 验证配置文件格式
python -c "import yaml; print('配置文件格式正确' if yaml.safe_load(open('config/api_config.yaml')) else '配置文件格式错误')"
```

### 3. 运行可用测试
```bash
# 运行火山方舟Chat API测试
cd RobotAgent_MVP
python tests/volcengine_chat_test.py

# 运行现有的单元测试
python -m pytest tests/ -v
```

## 基本使用 (MVP阶段)

### 1. 火山方舟Chat API测试工具

这是当前唯一可用的功能模块：

```bash
# 进入MVP目录
cd RobotAgent_MVP

# 运行交互式聊天工具
python tests/volcengine_chat_test.py
```

### 2. 使用示例

运行测试工具后，您可以：

1. **发送消息**: 直接输入文本与AI对话
2. **查看历史**: 输入 `history` 查看对话历史
3. **清除历史**: 输入 `clear` 清除对话历史
4. **退出程序**: 输入 `quit` 或 `exit`

```
=== 火山方舟Chat API测试工具 ===
配置加载成功！
模型: doubao-seed-1-6-250615
温度: 0.7
最大令牌数: 2000

您: 你好，请介绍一下自己
AI: 您好！我是豆包，字节跳动开发的AI助手...

您: history
=== 对话历史 ===
[2024-01-15 10:30:15] 用户: 你好，请介绍一下自己
[2024-01-15 10:30:16] 助手: 您好！我是豆包...

您: quit
感谢使用！再见！
```

### 3. 配置自定义

您可以修改 `config/api_config.yaml` 来调整：

- **模型选择**: 更改 `default_model`
- **响应创造性**: 调整 `temperature` (0.0-1.0)
- **响应长度**: 修改 `max_tokens`
- **历史记录**: 设置 `max_history_turns`

## 常见问题 (MVP阶段)

### Q: 配置文件不存在
**A**: 确保已复制配置模板：
```bash
# 检查配置文件是否存在
ls -la config/api_config.yaml

# 如果不存在，复制模板
cp config/api_config.yaml.template config/api_config.yaml

# 编辑配置文件，添加真实的API密钥
```

### Q: API密钥错误
**A**: 检查API密钥配置：
- 确认API密钥格式正确
- 检查是否有多余的空格或换行符
- 验证API密钥是否有效且未过期
- 确认账户余额充足

```bash
# 验证配置文件格式
python -c "
import yaml
try:
    with open('config/api_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print('配置文件格式正确')
    print(f'API密钥长度: {len(config[\"volcengine\"][\"api_key\"])}')
except Exception as e:
    print(f'配置文件错误: {e}')
"
```

### Q: 网络连接问题
**A**: 检查网络连接：
```bash
# 测试网络连接
ping ark.cn-beijing.volces.com

# 检查防火墙设置
# Windows: 检查Windows防火墙
# Linux: sudo ufw status

# 检查代理设置
echo $HTTP_PROXY
echo $HTTPS_PROXY
```

### Q: Python依赖问题
**A**: 重新安装依赖：
```bash
# 检查Python版本
python --version

# 检查pip版本
pip --version

# 重新安装依赖
pip install -r RobotAgent_MVP/requirements.txt

# 检查特定包
pip show requests pyyaml
```

### Q: 程序运行错误
**A**: 检查错误信息：
```bash
# 运行时显示详细错误信息
cd RobotAgent_MVP
python -v tests/volcengine_chat_test.py

# 检查Python路径
python -c "import sys; print('\\n'.join(sys.path))"

# 检查当前工作目录
pwd
```

## 下一步 (MVP阶段)

### 1. 当前可以做的
- 熟悉火山方舟Chat API的使用
- 尝试不同的对话场景和参数配置
- 阅读项目的设计文档了解未来规划

### 2. 等待开发的功能
- CAMEL智能体系统 (规划中)
- 多模态记忆系统 (规划中)
- ROS2集成 (规划中)
- Web界面 (规划中)

### 3. 如何参与开发
- 查看项目的开发计划
- 关注代码仓库的更新
- 提供反馈和建议

## 获取帮助

### 当前可用资源
- **API配置说明**: [README_volcengine_chat.md](RobotAgent_MVP/tests/README_volcengine_chat.md)
- **配置迁移文档**: [API_CONFIG_MIGRATION.md](API_CONFIG_MIGRATION.md)
- **项目概述**: [README.md](README.md)

### 设计文档 (规划阶段)
- CAMEL-ROS2架构文档 (待开发)
- 多模态记忆系统文档 (待开发)
- 智能体实现指南 (待开发)

### 联系方式
- 问题反馈: 通过项目仓库提交Issue
- 功能建议: 通过项目仓库提交Feature Request

---

**重要提醒**: 本项目目前处于MVP开发阶段，功能有限。请关注项目更新以获取最新功能。