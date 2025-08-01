import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from contextlib import asynccontextmanager
import uvicorn
import psutil
import os
from pathlib import Path

# 导入自定义模块
from utils.config import Config
from utils.logger import CustomLogger, PerformanceLogger
from services.qwen_service import QwenService
from services.message_queue import MessageQueue
from agents.memory_agent import MemoryAgent
from agents.ros2_agent import ROS2Agent
from models import (
    ProcessCommandRequest, ProcessCommandResponse, SystemStatusResponse,
    LogsResponse, MemoryRecordsResponse, WebSocketMessage, HealthCheckResponse,
    PerformanceMetrics, APIResponse, QueueMessage, MessageType, Priority
)

# 全局变量
config: Config = None
qwen_service: QwenService = None
message_queue: MessageQueue = None
memory_agent: MemoryAgent = None
ros2_agent: ROS2Agent = None
logger: CustomLogger = None
perf_logger: PerformanceLogger = None
websocket_connections: List[WebSocket] = []
app_start_time: datetime = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global config, qwen_service, message_queue, memory_agent, ros2_agent, logger, perf_logger, app_start_time
    
    # 启动时初始化
    app_start_time = datetime.now()
    
    # 加载配置
    config = Config()
    
    # 初始化日志
    logger = CustomLogger("FastAPI")
    perf_logger = PerformanceLogger()
    
    logger.info("正在启动RobotAgent MVP服务...")
    
    try:
        # 初始化服务
        qwen_service = QwenService(config)
        message_queue = MessageQueue(config)
        
        # 连接消息队列
        await message_queue.connect()
        
        # 初始化Agent
        memory_agent = MemoryAgent(config, message_queue)
        ros2_agent = ROS2Agent(config, message_queue)
        
        # 启动Agent
        await memory_agent.start()
        await ros2_agent.start()
        
        logger.info("RobotAgent MVP服务启动成功")
        
        yield
        
    except Exception as e:
        logger.error(f"服务启动失败: {str(e)}")
        raise
    
    # 关闭时清理
    logger.info("正在关闭RobotAgent MVP服务...")
    
    try:
        # 停止Agent
        if memory_agent:
            await memory_agent.stop()
        if ros2_agent:
            await ros2_agent.stop()
        
        # 关闭服务
        if message_queue:
            await message_queue.disconnect()
        if qwen_service:
            await qwen_service.close()
        
        logger.info("RobotAgent MVP服务已关闭")
        
    except Exception as e:
        logger.error(f"服务关闭时出错: {str(e)}")

# 创建FastAPI应用
app = FastAPI(
    title="RobotAgent MVP API",
    description="基于Qwen模型的机器人控制系统最小可行性验证",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务
static_dir = Path(__file__).parent.parent / "frontend"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    """根路径，返回前端页面"""
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    else:
        return HTMLResponse("""
        <html>
            <head><title>RobotAgent MVP</title></head>
            <body>
                <h1>RobotAgent MVP API</h1>
                <p>API文档: <a href="/docs">/docs</a></p>
                <p>健康检查: <a href="/health">/health</a></p>
            </body>
        </html>
        """)

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """健康检查接口"""
    try:
        uptime = (datetime.now() - app_start_time).total_seconds()
        
        # 检查各组件健康状态
        components = []
        
        # Qwen服务健康检查
        qwen_health = await qwen_service.health_check()
        components.append({
            "name": "qwen_service",
            "status": qwen_health.get("status", "unknown"),
            "last_check": datetime.now(),
            "details": qwen_health
        })
        
        # 消息队列健康检查
        queue_health = await message_queue.health_check()
        components.append({
            "name": "message_queue",
            "status": queue_health.get("status", "unknown"),
            "last_check": datetime.now(),
            "details": queue_health
        })
        
        # 记忆Agent健康检查
        memory_health = await memory_agent.health_check()
        components.append({
            "name": "memory_agent",
            "status": memory_health.get("status", "unknown"),
            "last_check": datetime.now(),
            "details": memory_health
        })
        
        # ROS2 Agent健康检查
        ros2_health = await ros2_agent.health_check()
        components.append({
            "name": "ros2_agent",
            "status": ros2_health.get("status", "unknown"),
            "last_check": datetime.now(),
            "details": ros2_health
        })
        
        # 判断整体健康状态
        overall_status = "healthy"
        for component in components:
            if component["status"] != "healthy":
                overall_status = "degraded"
                break
        
        return HealthCheckResponse(
            status=overall_status,
            version="1.0.0",
            uptime=uptime,
            components=components
        )
        
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"健康检查失败: {str(e)}")

@app.post("/api/process-command", response_model=ProcessCommandResponse)
async def process_command(request: ProcessCommandRequest):
    """处理用户命令"""
    start_time = datetime.now()
    message_id = str(uuid.uuid4())
    
    try:
        logger.info(
            f"接收到用户命令",
            extra={
                "user_id": request.user_id,
                "session_id": request.session_id,
                "input_length": len(request.input_text)
            }
        )
        
        # 使用Qwen解析自然语言
        with perf_logger.timer("qwen_parsing"):
            parse_result = await qwen_service.parse_natural_language(
                request.input_text,
                request.user_id
            )
        
        if not parse_result["success"]:
            raise HTTPException(
                status_code=400,
                detail=f"命令解析失败: {parse_result.get('error', '未知错误')}"
            )
        
        parsed_data = parse_result["result"]
        
        # 创建用户输入消息
        user_message = QueueMessage(
            message_id=message_id,
            message_type=MessageType.USER_INPUT,
            data={
                "user_id": request.user_id,
                "session_id": request.session_id or str(uuid.uuid4()),
                "input_text": request.input_text,
                "language": request.language
            },
            priority=Priority.MEDIUM
        )
        
        # 创建解析后的命令消息
        parsed_message = QueueMessage(
            message_id=message_id,
            message_type=MessageType.PARSED_COMMAND,
            data={
                "user_id": request.user_id,
                "session_id": request.session_id or str(uuid.uuid4()),
                "input_text": request.input_text,
                "intent": parsed_data["intent"],
                "action": parsed_data["action"],
                "parameters": parsed_data["parameters"],
                "estimated_duration": parsed_data.get("estimated_duration", 5.0)
            },
            priority=Priority(parsed_data.get("priority", "medium"))
        )
        
        # 发送消息到队列
        with perf_logger.timer("queue_operations"):
            # 发送到记忆Agent
            await message_queue.send_to_memory_agent(user_message)
            await message_queue.send_to_memory_agent(parsed_message)
            
            # 发送到ROS2 Agent
            await message_queue.send_to_ros2_agent(parsed_message)
        
        # 计算总处理时间
        total_time = (datetime.now() - start_time).total_seconds()
        
        # 通过WebSocket广播状态更新
        await broadcast_websocket_message({
            "type": "command_processed",
            "data": {
                "message_id": message_id,
                "intent": parsed_data["intent"],
                "action": parsed_data["action"],
                "status": "queued"
            }
        })
        
        logger.info(
            f"命令处理完成",
            extra={
                "message_id": message_id,
                "total_time": total_time,
                "intent": parsed_data["intent"],
                "action": parsed_data["action"]
            }
        )
        
        return ProcessCommandResponse(
            success=True,
            message="命令处理成功",
            data=ProcessCommandResponse.CommandData(
                message_id=message_id,
                parsed_intent=parsed_data["intent"],
                parsed_action=parsed_data["action"],
                parameters=parsed_data["parameters"],
                estimated_duration=parsed_data.get("estimated_duration"),
                status="queued"
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        total_time = (datetime.now() - start_time).total_seconds()
        logger.error(
            f"处理命令失败: {str(e)}",
            extra={
                "message_id": message_id,
                "total_time": total_time,
                "error": str(e)
            }
        )
        raise HTTPException(status_code=500, detail=f"处理命令失败: {str(e)}")

@app.get("/api/system-status", response_model=SystemStatusResponse)
async def get_system_status():
    """获取系统状态"""
    try:
        # 获取系统资源使用情况
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        uptime = (datetime.now() - app_start_time).total_seconds()
        
        # 获取各组件状态
        qwen_health = await qwen_service.health_check()
        queue_health = await message_queue.health_check()
        memory_health = await memory_agent.health_check()
        ros2_health = await ros2_agent.health_check()
        
        return SystemStatusResponse(
            success=True,
            message="系统状态获取成功",
            data=SystemStatusResponse.StatusData(
                qwen_service=qwen_health.get("status", "unknown"),
                redis_connection=queue_health.get("status", "unknown"),
                memory_agent=memory_health.get("status", "unknown"),
                ros2_agent=ros2_health.get("status", "unknown"),
                uptime=uptime,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage
            )
        )
        
    except Exception as e:
        logger.error(f"获取系统状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取系统状态失败: {str(e)}")

@app.get("/api/logs", response_model=LogsResponse)
async def get_logs(page: int = 1, page_size: int = 50, level: str = None):
    """获取系统日志"""
    try:
        # 这里简化实现，实际应该从日志文件读取
        logs = []
        
        # 模拟日志数据
        sample_logs = [
            {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "component": "FastAPI",
                "message": "系统运行正常"
            },
            {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "component": "QwenService",
                "message": f"已处理 {qwen_service.total_calls} 次API调用"
            }
        ]
        
        # 应用过滤和分页
        filtered_logs = sample_logs
        if level:
            filtered_logs = [log for log in filtered_logs if log["level"] == level.upper()]
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_logs = filtered_logs[start_idx:end_idx]
        
        return LogsResponse(
            success=True,
            message="日志获取成功",
            data=LogsResponse.LogsData(
                logs=[LogsResponse.LogEntry(**log) for log in page_logs],
                total_count=len(filtered_logs),
                page=page,
                page_size=page_size
            )
        )
        
    except Exception as e:
        logger.error(f"获取日志失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取日志失败: {str(e)}")

@app.get("/api/memory-records", response_model=MemoryRecordsResponse)
async def get_memory_records(limit: int = 20):
    """获取记忆记录"""
    try:
        records = await memory_agent.get_memory_records(limit)
        
        return MemoryRecordsResponse(
            success=True,
            message="记忆记录获取成功",
            data=MemoryRecordsResponse.MemoryRecordsData(
                records=[
                    MemoryRecordsResponse.MemoryRecord(**record)
                    for record in records
                ],
                total_count=len(records),
                total_size=sum(record["size"] for record in records)
            )
        )
        
    except Exception as e:
        logger.error(f"获取记忆记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取记忆记录失败: {str(e)}")

@app.get("/api/performance-metrics", response_model=PerformanceMetrics)
async def get_performance_metrics():
    """获取性能指标"""
    try:
        # 系统指标
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # 应用指标
        qwen_stats = qwen_service.get_stats()
        queue_stats = await message_queue.get_all_queue_stats()
        
        return PerformanceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            disk_usage=disk.percent,
            active_sessions=len(websocket_connections),
            total_requests=qwen_stats["total_calls"],
            error_rate=qwen_stats["error_count"] / max(qwen_stats["total_calls"], 1) * 100,
            average_response_time=perf_logger.get_average_time("qwen_parsing"),
            qwen_api_calls=qwen_stats["total_calls"],
            qwen_avg_latency=perf_logger.get_average_time("qwen_parsing"),
            redis_connections=1 if queue_stats.get("connection_status") == "connected" else 0,
            ros2_commands_sent=ros2_agent.processed_count,
            memory_records_created=memory_agent.processed_count
        )
        
    except Exception as e:
        logger.error(f"获取性能指标失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取性能指标失败: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket连接端点"""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        logger.info("新的WebSocket连接已建立")
        
        # 发送欢迎消息
        await websocket.send_json({
            "type": "connection_established",
            "data": {"message": "WebSocket连接已建立"},
            "timestamp": datetime.now().isoformat()
        })
        
        # 保持连接
        while True:
            try:
                # 接收客户端消息
                data = await websocket.receive_json()
                
                # 处理客户端消息（如心跳等）
                if data.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket消息处理错误: {str(e)}")
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket连接错误: {str(e)}")
    finally:
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)
        logger.info("WebSocket连接已断开")

async def broadcast_websocket_message(message: Dict[str, Any]):
    """广播WebSocket消息"""
    if not websocket_connections:
        return
    
    message_with_timestamp = {
        **message,
        "timestamp": datetime.now().isoformat()
    }
    
    # 移除已断开的连接
    disconnected = []
    for websocket in websocket_connections:
        try:
            await websocket.send_json(message_with_timestamp)
        except:
            disconnected.append(websocket)
    
    # 清理断开的连接
    for websocket in disconnected:
        websocket_connections.remove(websocket)

if __name__ == "__main__":
    # 运行服务器
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # 生产环境应设为False
        log_level="info"
    )