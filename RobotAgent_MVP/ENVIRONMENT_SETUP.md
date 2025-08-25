# 环境变量配置指南

本文档指导您如何正确配置 RobotAgent MVP 系统的环境变量。

## 快速开始

### 1. 复制环境变量模板

```bash
cp .env.template .env
```

### 2. 编辑 .env 文件

打开 `.env` 文件，填入您的实际配置值：

```bash
# 必填：火山方舟 API 密钥
VOLCENGINE_API_KEY=your_actual_api_key_here

# 可选：其他配置
LOG_LEVEL=INFO
DEVELOPMENT_MODE=true
```

## 获取火山方舟 API 密钥

1. 访问 [火山方舟控制台](https://console.volcengine.com/ark)
2. 登录您的账户
3. 创建或选择一个应用
4. 在 API 密钥管理页面生成新的密钥
5. 复制密钥并粘贴到 `.env` 文件中

## 环境变量说明

### 必需的环境变量

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `VOLCENGINE_API_KEY` | 火山方舟 API 密钥 | `your_api_key_here` |

### 可选的环境变量

| 变量名 | 说明 | 默认值 | 示例 |
|--------|------|--------|------|
| `VOLCENGINE_BASE_URL` | 火山方舟 API 基础URL | 配置文件中的值 | `https://ark.cn-beijing.volces.com/api/v3` |
| `VOLCENGINE_DEFAULT_MODEL` | 默认模型 | 配置文件中的值 | `doubao-seed-1-6-250615` |
| `LOG_LEVEL` | 日志级别 | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `DEVELOPMENT_MODE` | 开发模式 | `true` | `true`, `false` |
| `DEBUG_MODE` | 调试模式 | `false` | `true`, `false` |
| `API_TIMEOUT` | API 超时时间(秒) | `30` | `30` |
| `MAX_RETRIES` | 最大重试次数 | `3` | `3` |

## 安全注意事项

⚠️ **重要安全提醒**

1. **永远不要将 `.env` 文件提交到版本控制系统**
   - `.env` 文件已添加到 `.gitignore` 中
   - 确保您的 API 密钥不会意外泄露

2. **定期轮换 API 密钥**
   - 建议定期更新您的 API 密钥
   - 如果怀疑密钥泄露，立即更换

3. **限制文件权限**
   ```bash
   # Linux/macOS
   chmod 600 .env
   ```

4. **使用不同环境的不同密钥**
   - 开发环境和生产环境应使用不同的 API 密钥
   - 避免在开发中使用生产密钥

## 验证配置

配置完成后，您可以运行以下命令验证配置是否正确：

```bash
# 运行配置测试
python tests/volcengine_chat_test.py

# 或运行完整的系统测试
python test_chat_agent_registration.py
```

## 故障排除

### 常见问题

1. **API 密钥无效**
   - 检查密钥是否正确复制
   - 确认密钥在火山方舟控制台中是否有效
   - 检查密钥是否有足够的权限

2. **环境变量未加载**
   - 确认 `.env` 文件在项目根目录
   - 检查文件名是否正确（不是 `.env.txt`）
   - 重启应用程序

3. **配置文件错误**
   - 检查 `config/api_config.yaml` 中的占位符格式
   - 确认环境变量名称匹配

### 调试步骤

1. 设置调试模式：
   ```bash
   DEBUG_MODE=true
   LOG_LEVEL=DEBUG
   ```

2. 检查环境变量是否正确加载：
   ```python
   import os
   print(f"API Key: {os.getenv('VOLCENGINE_API_KEY', 'Not found')}")
   ```

3. 查看详细日志输出

## 支持

如果您在配置过程中遇到问题，请：

1. 检查本文档的故障排除部分
2. 查看项目的 README.md 文件
3. 提交 Issue 到项目仓库

---

**注意**: 本配置指南适用于 RobotAgent MVP v0.0.1。不同版本的配置可能有所差异。