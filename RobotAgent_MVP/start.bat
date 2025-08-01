@echo off
REM RobotAgent MVP Windows 启动脚本

echo === RobotAgent MVP 启动脚本 ===

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python
    pause
    exit /b 1
)

REM 检查Docker环境
docker --version >nul 2>&1
if errorlevel 1 (
    echo 警告: 未找到Docker，将使用本地模式启动
    set USE_DOCKER=false
) else (
    set USE_DOCKER=true
)

REM 设置环境变量
set PYTHONPATH=%PYTHONPATH%;%cd%\backend

REM 检查并设置 DASHSCOPE_API_KEY
if not defined DASHSCOPE_API_KEY (
    echo 检查系统环境变量中的 DASHSCOPE_API_KEY...
    for /f "tokens=*" %%i in ('powershell -Command "[System.Environment]::GetEnvironmentVariable('DASHSCOPE_API_KEY', 'Machine')"') do set DASHSCOPE_API_KEY=%%i
    
    if not defined DASHSCOPE_API_KEY (
        for /f "tokens=*" %%i in ('powershell -Command "[System.Environment]::GetEnvironmentVariable('DASHSCOPE_API_KEY', 'User')"') do set DASHSCOPE_API_KEY=%%i
    )
    
    if defined DASHSCOPE_API_KEY (
        echo 成功从系统环境变量加载 DASHSCOPE_API_KEY
    ) else (
        echo 警告: 未找到 DASHSCOPE_API_KEY 环境变量，将使用测试模式
    )
) else (
    echo DASHSCOPE_API_KEY 已设置
)

REM 创建必要的目录
if not exist logs mkdir logs
if not exist memory_records mkdir memory_records
if not exist data mkdir data

echo 创建必要目录完成

REM 选择启动模式
if "%USE_DOCKER%"=="true" (
    echo 使用Docker模式启动...
    
    REM 检查docker-compose
    docker-compose --version >nul 2>&1
    if not errorlevel 1 (
        echo 启动Redis服务...
        docker-compose up -d redis
        
        echo 等待Redis启动...
        timeout /t 5 /nobreak >nul
        
        echo 启动后端服务...
        python backend\app.py
    ) else (
        echo 未找到docker-compose，使用Docker直接启动Redis...
        
        REM 启动Redis容器
        docker run -d --name robotagent-redis -p 6379:6379 -v %cd%\data\redis:/data redis:7-alpine redis-server --appendonly yes
        
        echo 等待Redis启动...
        timeout /t 5 /nobreak >nul
        
        echo 启动后端服务...
        python backend\app.py
    )
) else (
    echo 使用本地模式启动...
    echo 警告: 需要手动启动Redis服务
    echo 请确保Redis在localhost:6379运行
    
    pause
    
    echo 启动后端服务...
    python backend\app.py
)

pause