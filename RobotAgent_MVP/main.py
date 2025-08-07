#!/usr/bin/env python3
"""
RobotAgent MVP - 主程序
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from services.doubao_service import DoubaoService
from services.asr_service import ASRService
from services.tts_service import TTSService
from services.memory_service import MemoryService
from services.ros2_service import ROS2Service

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/main.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global doubao_service, asr_service, tts_service, memory_service, ros2_service, robot_agent
    
    # 启动时执行
    logger.info("正在启动RobotAgent MVP...")
    
    try:
        # 加载配置 - 修复路径问题
        config_path = Path(__file__).parent / "config.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 初始化服务
        doubao_service = DoubaoService(config["doubao_api"])
        asr_service = ASRService(config["asr_config"])
        tts_service = TTSService(config["tts_config"])
        memory_service = MemoryService(config["memory_config"])
        ros2_service = ROS2Service(config["memory_config"])
        
        # 初始化机器人代理
        robot_agent = RobotAgent()
        
        logger.info("所有服务初始化完成")
        
    except Exception as e:
        logger.error(f"服务初始化失败: {e}")
        raise
    
    yield  # 应用运行期间
    
    # 关闭时执行
    logger.info("正在关闭RobotAgent MVP...")

# 创建FastAPI应用
app = FastAPI(
    title="RobotAgent MVP",
    description="基于豆包模型的机器人对话系统",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 请求/响应模型
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    success: bool
    response: str
    conversation_id: str
    ros2_command: Optional[Dict[str, Any]] = None
    audio_url: Optional[str] = None

class VoiceChatRequest(BaseModel):
    conversation_id: Optional[str] = None

# 全局服务实例
doubao_service: Optional[DoubaoService] = None
asr_service: Optional[ASRService] = None
tts_service: Optional[TTSService] = None
memory_service: Optional[MemoryService] = None
ros2_service: Optional[ROS2Service] = None
robot_agent = None  # 将在startup_event中初始化

class RobotAgent:
    def __init__(self):
        # 使用全局服务实例
        self.doubao_service = doubao_service
        self.asr_service = asr_service
        self.tts_service = tts_service
        self.memory_service = memory_service
        self.ros2_service = ros2_service
        
        # 加载系统提示词
        self.system_prompt = self.load_system_prompt()
        
        # 流式语音模式开关
        self.voice_streaming_enabled = False
        
        logger.info("RobotAgent 初始化完成")
    
    def load_system_prompt(self):
        """加载系统提示词"""
        try:
            with open("prompts/system_prompt.json", "r", encoding="utf-8") as f:
                prompt_data = json.load(f)
                return json.dumps(prompt_data["system_prompt"], ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"加载系统提示词失败: {e}")
            return "你是一个智能机器人助手。"
    
    async def process_message(self, message: str, conversation_history: List[Dict] = None) -> Dict:
        """处理用户消息 - 双逻辑处理系统"""
        try:
            # 构建对话历史
            if conversation_history is None:
                conversation_history = []
            
            # 检测是否包含动作任务
            has_action_task = await self.detect_action_task(message)
            
            # 构建提示词
            system_prompt = self.build_system_prompt(has_action_task)
            
            # 添加系统提示词
            messages = [{"role": "system", "content": system_prompt}]
            
            # 添加历史对话
            messages.extend(conversation_history)
            
            # 添加当前用户消息
            user_message = self.build_user_message(message, has_action_task)
            messages.append({"role": "user", "content": user_message})
            
            # 调用豆包模型
            if self.voice_streaming_enabled:
                response = await self.process_streaming_response(messages)
            else:
                response = await self.doubao_service.chat_completion(messages)
            
            # 解析响应
            response_data = await self.parse_response(response, has_action_task)
            
            # 保存到记忆
            await self.memory_service.save_conversation(message, response_data.get("user_reply", ""))
            
            # 如果包含动作任务，异步处理行为树
            if has_action_task and "behavior_tree" in response_data:
                asyncio.create_task(self.execute_behavior_tree(response_data["behavior_tree"]))
            
            return response_data
            
        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            return {
                "type": "error",
                "user_reply": "抱歉，我遇到了一些问题，请稍后再试。",
                "error": str(e)
            }
    
    async def detect_action_task(self, message: str) -> bool:
        """检测用户输入是否包含机器人动作任务"""
        action_keywords = [
            "帮我", "取", "拿", "放", "移动", "去", "找", "抓", "推", "拉",
            "打开", "关闭", "整理", "清理", "搬", "运", "送", "带来", "拿来"
        ]
        
        return any(keyword in message for keyword in action_keywords)
    
    def build_system_prompt(self, has_action_task: bool) -> str:
        """构建系统提示词"""
        base_prompt = self.system_prompt
        
        if has_action_task:
            mode_instruction = """
当前模式：双逻辑处理模式
- 请同时使用Agent逻辑和Action规划逻辑处理用户输入
- Agent逻辑：提供友好的对话回复和情感支持
- Action逻辑：将任务分解为详细的行为树结构
- 返回格式：dual_response 或 action_response
"""
        else:
            mode_instruction = """
当前模式：Agent对话模式
- 专注于自然对话、问答和情感交流
- 返回格式：agent_response
"""
        
        return f"{base_prompt}\n\n{mode_instruction}"
    
    def build_user_message(self, message: str, has_action_task: bool) -> str:
        """构建用户消息"""
        if has_action_task:
            return f"""用户输入：{message}

请使用双逻辑处理：
1. Agent逻辑：理解用户情感和意图，提供友好回复
2. Action逻辑：分析任务并生成详细的行为树

可用的ROS2动作库：{json.dumps(self.ros2_service.get_available_actions(), ensure_ascii=False)}

请按照system_prompt中的格式返回JSON响应。"""
        else:
            return f"""用户输入：{message}

请使用Agent逻辑进行自然对话，按照agent_response格式返回。"""
    
    async def parse_response(self, response, has_action_task: bool) -> Dict:
        """解析模型响应"""
        try:
            # 如果response已经是字典（豆包服务返回的格式）
            if isinstance(response, dict):
                # 检查是否是豆包服务的标准返回格式
                if "response" in response:
                    response_text = response.get("response", "")
                    ros2_command = response.get("ros2_command")
                    
                    # 尝试解析response_text中的JSON
                    try:
                        if response_text.strip().startswith('{'):
                            parsed_data = json.loads(response_text)
                            return parsed_data
                    except json.JSONDecodeError:
                        pass
                    
                    # 如果不是JSON格式，包装成标准格式
                    if has_action_task:
                        return {
                            "type": "action_response",
                            "user_reply": response_text,
                            "action_analysis": {"main_task": "简单回复"},
                            "behavior_tree": None,
                            "task_decomposition": {},
                            "execution_plan": {},
                            "ros2_command": ros2_command
                        }
                    else:
                        return {
                            "type": "agent_response",
                            "user_reply": response_text,
                            "thinking_process": "正常对话回复",
                            "memory_update": {},
                            "emotion_state": "正常"
                        }
                else:
                    # 如果是其他格式的字典，直接返回
                    return response
            
            # 如果是字符串，尝试解析JSON格式的响应
            response_data = json.loads(response)
            return response_data
        except json.JSONDecodeError:
            # 如果不是JSON格式，包装成标准格式
            if has_action_task:
                return {
                    "type": "action_response",
                    "user_reply": str(response),
                    "action_analysis": {"main_task": "解析失败"},
                    "behavior_tree": None,
                    "task_decomposition": {},
                    "execution_plan": {}
                }
            else:
                return {
                    "type": "agent_response",
                    "user_reply": str(response),
                    "thinking_process": "模型返回非结构化响应",
                    "memory_update": {},
                    "emotion_state": "正常"
                }
    
    async def process_streaming_response(self, messages: List[Dict]) -> str:
        """处理流式响应（用于语音对话）"""
        try:
            # 调用豆包服务的流式API
            response = await self.doubao_service._call_chat_api(messages, stream=True)
            
            if response is None:
                return "抱歉，流式响应处理失败。"
            
            # 收集流式响应内容
            full_response = ""
            async for chunk in self.doubao_service._handle_stream_response(response):
                full_response += chunk
            
            return full_response
            
        except Exception as e:
            logger.error(f"流式响应处理失败: {e}")
            # 回退到普通响应
            return await self.doubao_service.chat_completion(messages)
    
    async def execute_behavior_tree(self, behavior_tree: Dict):
        """异步执行行为树"""
        try:
            logger.info(f"开始执行行为树: {behavior_tree.get('root', {}).get('name', 'Unknown')}")
            # 这里可以实现具体的行为树执行逻辑
            # 目前只是记录日志
            await self.traverse_behavior_tree(behavior_tree.get("root", {}))
        except Exception as e:
            logger.error(f"执行行为树失败: {e}")
    
    async def traverse_behavior_tree(self, node: Dict):
        """遍历并执行行为树节点"""
        if not node:
            return
        
        node_type = node.get("type", "")
        node_name = node.get("name", "Unknown")
        
        logger.info(f"执行节点: {node_name} (类型: {node_type})")
        
        if node_type == "action":
            # 执行ROS2动作
            ros2_action = node.get("ros2_action")
            if ros2_action:
                logger.info(f"执行ROS2动作: {ros2_action}")
                # 这里可以调用实际的ROS2动作
        
        elif node_type in ["sequence", "selector", "parallel"]:
            # 处理组合节点
            children = node.get("children", [])
            for child in children:
                await self.traverse_behavior_tree(child)
        
        elif node_type == "condition":
            # 处理条件节点
            logger.info(f"检查条件: {node_name}")
    
    def toggle_voice_streaming(self, enabled: bool):
        """切换语音流式模式"""
        self.voice_streaming_enabled = enabled
        logger.info(f"语音流式模式: {'开启' if enabled else '关闭'}")



@app.get("/", response_class=HTMLResponse)
async def root():
    """根路径，返回主页面"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RobotAgent MVP</title>
        <meta charset="utf-8">
    </head>
    <body>
        <h1>RobotAgent MVP</h1>
        <p>基于豆包模型的机器人对话系统</p>
        <p><a href="/static/index.html">进入聊天界面</a></p>
    </body>
    </html>
    """

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """文本聊天接口 - 双逻辑处理系统"""
    try:
        logger.info(f"收到聊天请求: {request.message}")
        
        # 获取对话历史
        conversation_history = await memory_service.get_conversation_history(
            request.conversation_id or "default"
        )
        
        # 使用RobotAgent处理消息
        response_data = await robot_agent.process_message(
            message=request.message,
            conversation_history=conversation_history
        )
        
        # 解析ROS2命令
        ros2_command = None
        if response_data.get("behavior_tree"):
            ros2_command = response_data["behavior_tree"]
        
        # 生成语音
        audio_url = None
        if response_data.get("user_reply"):
            audio_file = await tts_service.text_to_speech(response_data["user_reply"])
            if audio_file:
                audio_url = f"/static/audio/{audio_file}"
        
        # 保存对话记录
        conversation_id = request.conversation_id or "default"
        await memory_service.save_conversation(
            conversation_id=conversation_id,
            user_message=request.message,
            bot_response=response_data.get("user_reply", ""),
            ros2_command=ros2_command
        )
        
        return ChatResponse(
            success=True,
            response=response_data.get("user_reply", ""),
            conversation_id=conversation_id,
            ros2_command=ros2_command,
            audio_url=audio_url
        )
        
    except Exception as e:
        logger.error(f"聊天处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice_chat")
async def voice_chat(
    audio_file: UploadFile = File(...),
    conversation_id: Optional[str] = None
):
    """语音聊天接口"""
    try:
        logger.info("收到语音聊天请求")
        
        # 保存上传的音频文件
        audio_content = await audio_file.read()
        
        # ASR: 语音转文字
        text_message = await asr_service.speech_to_text(audio_content)
        if not text_message:
            raise HTTPException(status_code=400, detail="语音识别失败")
        
        logger.info(f"语音识别结果: {text_message}")
        
        # 调用文本聊天接口
        chat_request = ChatRequest(
            message=text_message,
            conversation_id=conversation_id
        )
        
        return await chat(chat_request)
        
    except Exception as e:
        logger.error(f"语音聊天处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """获取对话历史"""
    try:
        history = await memory_service.get_conversation_history(conversation_id)
        return {"success": True, "history": history}
    except Exception as e:
        logger.error(f"获取对话历史失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ros2_actions")
async def get_ros2_actions():
    """获取ROS2动作库"""
    try:
        actions = await ros2_service.get_actions_library()
        return {"success": True, "actions": actions}
    except Exception as e:
        logger.error(f"获取ROS2动作库失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/toggle_voice_streaming")
async def toggle_voice_streaming(request: dict):
    """切换语音流式模式"""
    try:
        enabled = request.get("enabled", False)
        robot_agent.toggle_voice_streaming(enabled)
        return {
            "success": True, 
            "data": {
                "voice_streaming_enabled": robot_agent.voice_streaming_enabled,
                "message": f"语音流式模式已{'开启' if enabled else '关闭'}"
            }
        }
    except Exception as e:
        logger.error(f"切换语音流式模式错误: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/status")
async def get_status():
    """获取系统状态"""
    try:
        return {
            "success": True,
            "data": {
                "voice_streaming_enabled": robot_agent.voice_streaming_enabled,
                "services": {
                    "doubao": "运行中",
                    "asr": "运行中", 
                    "tts": "运行中",
                    "memory": "运行中",
                    "ros2": "运行中"
                },
                "available_actions": robot_agent.ros2_service.get_available_actions()
            }
        }
    except Exception as e:
        logger.error(f"获取状态错误: {e}")
        return {"success": False, "error": str(e)}

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "message": "RobotAgent MVP is running"}

def main():
    """主函数"""
    # 确保必要的目录存在
    os.makedirs("logs", exist_ok=True)
    os.makedirs("static/audio", exist_ok=True)
    
    # 启动服务器
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()