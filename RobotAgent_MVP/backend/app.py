from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import asyncio
import uvicorn
import os

from utils.config import Config
from utils.logger import CustomLogger
from services.qwen_service import QwenService
from agents.ros2_agent import ROS2Agent
from agents.memory_agent import MemoryAgent

# 配置和日志
config = Config()
logger = CustomLogger("FastAPI")

# 创建FastAPI应用
app = FastAPI(
    title="RobotAgent API",
    description="智能机器人控制API",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 全局服务实例
qwen_service = None
ros2_agent = None
memory_agent = None

# 请求模型
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = "default"
    user_id: Optional[str] = "default"

class CommandRequest(BaseModel):
    command: str
    user_id: Optional[str] = "default"

# 响应模型
class ChatResponse(BaseModel):
    success: bool
    user_reply: str
    execution_status: Optional[str] = None
    conversation_id: str
    timestamp: float

class StatusResponse(BaseModel):
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化服务"""
    global qwen_service, ros2_agent, memory_agent
    
    try:
        logger.info("正在启动RobotAgent服务...")
        
        # 初始化消息队列
        from services.message_queue import MessageQueue
        message_queue = MessageQueue(config)
        
        # 初始化服务
        qwen_service = QwenService()
        ros2_agent = ROS2Agent()
        memory_agent = MemoryAgent(config, message_queue)
        
        # 启动代理
        await ros2_agent.start()
        await memory_agent.start()
        
        # 启动状态反馈任务
        asyncio.create_task(status_feedback_loop())
        
        logger.info("RobotAgent服务启动成功")
        
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    global ros2_agent, memory_agent
    
    try:
        logger.info("正在关闭RobotAgent服务...")
        
        if ros2_agent:
            await ros2_agent.stop()
        
        if memory_agent:
            await memory_agent.stop()
        
        logger.info("RobotAgent服务已关闭")
        
    except Exception as e:
        logger.error(f"服务关闭失败: {e}")

async def status_feedback_loop():
    """状态反馈循环 - 将执行状态转换为自然语言并存储"""
    while True:
        try:
            # 获取ROS2执行状态
            status_info = await ros2_agent.get_status_feedback()
            
            if status_info:
                # 将状态转换为自然语言
                natural_language_status = await qwen_service.convert_status_to_natural_language(
                    status_info["execution_result"]
                )
                
                # 存储状态反馈到记忆
                await memory_agent.store_interaction({
                    "type": "status_feedback",
                    "original_reply": status_info["user_reply"],
                    "execution_result": status_info["execution_result"],
                    "natural_language_status": natural_language_status,
                    "timestamp": status_info["timestamp"]
                })
                
                logger.info(f"状态反馈已处理: {natural_language_status}")
            
            await asyncio.sleep(1)  # 每秒检查一次
            
        except Exception as e:
            logger.error(f"状态反馈循环错误: {e}")
            await asyncio.sleep(5)

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "RobotAgent API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        health_status = {
            "api": "healthy",
            "qwen_service": "healthy" if qwen_service else "unhealthy",
            "ros2_agent": ros2_agent.get_health_status() if ros2_agent else {"status": "unhealthy"},
            "memory_agent": memory_agent.get_health_status() if memory_agent else {"status": "unhealthy"}
        }
        
        overall_status = "healthy" if all(
            status.get("status") == "healthy" if isinstance(status, dict) else status == "healthy"
            for status in health_status.values()
        ) else "unhealthy"
        
        return {
            "status": overall_status,
            "services": health_status,
            "timestamp": asyncio.get_event_loop().time()
        }
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/api/chat", response_model=ChatResponse)
async def robot_chat(request: ChatRequest):
    """
    机器人聊天接口 - 主要API接口
    
    用户通过此接口与机器人进行自然语言对话
    系统会自动判断是否需要执行机器人动作
    """
    try:
        logger.info(f"收到聊天请求: {request.message}")
        
        # 获取对话历史
        conversation_history = await memory_agent.get_conversation_history(
            request.conversation_id, 
            limit=10
        )
        
        # 获取最近的执行状态（用于状态反馈）
        recent_status = await memory_agent.get_recent_status_feedback(request.conversation_id)
        execution_status = recent_status.get("natural_language_status") if recent_status else None
        
        # 调用Qwen API获取响应
        robot_response = await qwen_service.robot_chat(
            user_input=request.message,
            conversation_history=conversation_history,
            execution_status=execution_status
        )
        
        if not robot_response:
            raise HTTPException(status_code=500, detail="无法获取机器人响应")
        
        # 存储用户输入到记忆
        await memory_agent.store_interaction({
            "type": "user_input",
            "conversation_id": request.conversation_id,
            "user_id": request.user_id,
            "message": request.message,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        # 处理机器人响应
        execution_result = await ros2_agent.process_robot_response(robot_response)
        
        # 存储机器人响应到记忆
        await memory_agent.store_interaction({
            "type": "robot_response",
            "conversation_id": request.conversation_id,
            "user_id": request.user_id,
            "user_reply": robot_response["user_reply"],
            "ros2_command": robot_response["ros2_command"],
            "execution_result": execution_result,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        # 返回响应
        return ChatResponse(
            success=True,
            user_reply=robot_response["user_reply"],
            execution_status=execution_result.get("message"),
            conversation_id=request.conversation_id,
            timestamp=asyncio.get_event_loop().time()
        )
        
    except Exception as e:
        logger.error(f"聊天请求处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@app.post("/api/process-command")
async def process_command(request: CommandRequest):
    """
    处理命令接口 - 兼容旧版本
    
    直接处理用户命令，不进行对话管理
    """
    try:
        logger.info(f"收到命令请求: {request.command}")
        
        # 调用Qwen API
        robot_response = await qwen_service.robot_chat(
            user_input=request.command,
            conversation_history=None,
            execution_status=None
        )
        
        if not robot_response:
            raise HTTPException(status_code=500, detail="无法获取机器人响应")
        
        # 处理机器人响应
        execution_result = await ros2_agent.process_robot_response(robot_response)
        
        return {
            "success": True,
            "user_reply": robot_response["user_reply"],
            "execution_status": execution_result.get("message"),
            "details": execution_result
        }
        
    except Exception as e:
        logger.error(f"命令处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@app.get("/api/status")
async def get_system_status():
    """获取系统状态"""
    try:
        return {
            "qwen_service": {
                "total_requests": qwen_service.total_requests,
                "successful_requests": qwen_service.successful_requests,
                "failed_requests": qwen_service.failed_requests,
                "average_response_time": qwen_service.total_response_time / max(qwen_service.total_requests, 1)
            },
            "ros2_agent": ros2_agent.get_status(),
            "memory_agent": memory_agent.get_status()
        }
        
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")

@app.get("/api/conversation/{conversation_id}/history")
async def get_conversation_history(conversation_id: str, limit: int = 20):
    """获取对话历史"""
    try:
        history = await memory_agent.get_conversation_history(conversation_id, limit)
        return {
            "conversation_id": conversation_id,
            "history": history,
            "count": len(history)
        }
        
    except Exception as e:
        logger.error(f"获取对话历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取历史失败: {str(e)}")

@app.delete("/api/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """清除对话历史"""
    try:
        await memory_agent.clear_conversation(conversation_id)
        return {
            "success": True,
            "message": f"对话 {conversation_id} 已清除"
        }
        
    except Exception as e:
        logger.error(f"清除对话失败: {e}")
        raise HTTPException(status_code=500, detail=f"清除失败: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )