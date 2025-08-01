FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY backend/requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY backend/ .
COPY config/ ./config/

# 创建必要的目录
RUN mkdir -p logs memory_records

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "app.py"]