#!/bin/bash

# RobotAgent MVP 启动脚本
# 用于启动整个系统

echo "=== RobotAgent MVP 启动脚本 ==="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python3"
    exit 1
fi

# 检查Docker环境
if ! command -v docker &> /dev/null; then
    echo "警告: 未找到Docker，将使用本地模式启动"
    USE_DOCKER=false
else
    USE_DOCKER=true
fi

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)/backend"

# 创建必要的目录
mkdir -p logs
mkdir -p memory_records
mkdir -p data

echo "创建必要目录完成"

# 选择启动模式
if [ "$USE_DOCKER" = true ]; then
    echo "使用Docker模式启动..."
    
    # 检查docker-compose
    if command -v docker-compose &> /dev/null; then
        echo "启动Redis服务..."
        docker-compose up -d redis
        
        echo "等待Redis启动..."
        sleep 5
        
        echo "启动后端服务..."
        python3 backend/app.py
    else
        echo "未找到docker-compose，使用Docker直接启动Redis..."
        
        # 启动Redis容器
        docker run -d --name robotagent-redis \
            -p 6379:6379 \
            -v $(pwd)/data/redis:/data \
            redis:7-alpine redis-server --appendonly yes
        
        echo "等待Redis启动..."
        sleep 5
        
        echo "启动后端服务..."
        python3 backend/app.py
    fi
else
    echo "使用本地模式启动..."
    echo "警告: 需要手动启动Redis服务"
    echo "请确保Redis在localhost:6379运行"
    
    read -p "按Enter继续，或Ctrl+C取消..."
    
    echo "启动后端服务..."
    python3 backend/app.py
fi