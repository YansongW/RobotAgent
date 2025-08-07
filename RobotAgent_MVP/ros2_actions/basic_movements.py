"""
基础移动动作库
包含机器人的基本移动功能
"""

import asyncio
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def ros2_action(name: str, description: str, category: str = "general", 
                parameters: Dict = None, examples: list = None):
    """ROS2动作装饰器"""
    def decorator(func):
        func.__ros2_action__ = {
            "name": name,
            "description": description,
            "category": category,
            "parameters": parameters or {},
            "examples": examples or []
        }
        return func
    return decorator

@ros2_action(
    name="move_forward",
    description="控制机器人向前移动指定距离",
    category="movement",
    parameters={
        "distance": {
            "type": "float",
            "description": "移动距离（米）",
            "default": 1.0,
            "min": 0.1,
            "max": 5.0
        },
        "speed": {
            "type": "float", 
            "description": "移动速度（米/秒）",
            "default": 0.5,
            "min": 0.1,
            "max": 2.0
        }
    },
    examples=[
        {"distance": 1.0, "speed": 0.5},
        {"distance": 2.5, "speed": 1.0}
    ]
)
async def move_forward(distance: float = 1.0, speed: float = 0.5) -> Dict[str, Any]:
    """向前移动"""
    try:
        logger.info(f"开始向前移动 {distance}米，速度 {speed}米/秒")
        
        # 模拟移动时间
        move_time = distance / speed
        await asyncio.sleep(min(move_time, 3.0))  # 最多等待3秒
        
        return {
            "status": "completed",
            "message": f"成功向前移动 {distance}米",
            "distance_moved": distance,
            "time_taken": move_time
        }
    except Exception as e:
        logger.error(f"向前移动失败: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }

@ros2_action(
    name="move_backward",
    description="控制机器人向后移动指定距离",
    category="movement",
    parameters={
        "distance": {
            "type": "float",
            "description": "移动距离（米）",
            "default": 1.0,
            "min": 0.1,
            "max": 3.0
        },
        "speed": {
            "type": "float",
            "description": "移动速度（米/秒）",
            "default": 0.3,
            "min": 0.1,
            "max": 1.0
        }
    },
    examples=[
        {"distance": 0.5, "speed": 0.3},
        {"distance": 1.0, "speed": 0.5}
    ]
)
async def move_backward(distance: float = 1.0, speed: float = 0.3) -> Dict[str, Any]:
    """向后移动"""
    try:
        logger.info(f"开始向后移动 {distance}米，速度 {speed}米/秒")
        
        move_time = distance / speed
        await asyncio.sleep(min(move_time, 3.0))
        
        return {
            "status": "completed",
            "message": f"成功向后移动 {distance}米",
            "distance_moved": distance,
            "time_taken": move_time
        }
    except Exception as e:
        logger.error(f"向后移动失败: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }

@ros2_action(
    name="turn_left",
    description="控制机器人向左转动指定角度",
    category="movement",
    parameters={
        "angle": {
            "type": "float",
            "description": "转动角度（度）",
            "default": 90.0,
            "min": 1.0,
            "max": 360.0
        },
        "angular_speed": {
            "type": "float",
            "description": "角速度（度/秒）",
            "default": 45.0,
            "min": 10.0,
            "max": 180.0
        }
    },
    examples=[
        {"angle": 90.0, "angular_speed": 45.0},
        {"angle": 180.0, "angular_speed": 60.0}
    ]
)
async def turn_left(angle: float = 90.0, angular_speed: float = 45.0) -> Dict[str, Any]:
    """向左转动"""
    try:
        logger.info(f"开始向左转动 {angle}度，角速度 {angular_speed}度/秒")
        
        turn_time = angle / angular_speed
        await asyncio.sleep(min(turn_time, 2.0))
        
        return {
            "status": "completed",
            "message": f"成功向左转动 {angle}度",
            "angle_turned": angle,
            "time_taken": turn_time
        }
    except Exception as e:
        logger.error(f"向左转动失败: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }

@ros2_action(
    name="turn_right",
    description="控制机器人向右转动指定角度",
    category="movement",
    parameters={
        "angle": {
            "type": "float",
            "description": "转动角度（度）",
            "default": 90.0,
            "min": 1.0,
            "max": 360.0
        },
        "angular_speed": {
            "type": "float",
            "description": "角速度（度/秒）",
            "default": 45.0,
            "min": 10.0,
            "max": 180.0
        }
    },
    examples=[
        {"angle": 90.0, "angular_speed": 45.0},
        {"angle": 45.0, "angular_speed": 30.0}
    ]
)
async def turn_right(angle: float = 90.0, angular_speed: float = 45.0) -> Dict[str, Any]:
    """向右转动"""
    try:
        logger.info(f"开始向右转动 {angle}度，角速度 {angular_speed}度/秒")
        
        turn_time = angle / angular_speed
        await asyncio.sleep(min(turn_time, 2.0))
        
        return {
            "status": "completed",
            "message": f"成功向右转动 {angle}度",
            "angle_turned": angle,
            "time_taken": turn_time
        }
    except Exception as e:
        logger.error(f"向右转动失败: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }

@ros2_action(
    name="stop_movement",
    description="立即停止机器人的所有移动",
    category="movement",
    parameters={},
    examples=[{}]
)
async def stop_movement() -> Dict[str, Any]:
    """停止移动"""
    try:
        logger.info("执行紧急停止")
        
        return {
            "status": "completed",
            "message": "机器人已停止所有移动",
            "timestamp": "immediate"
        }
    except Exception as e:
        logger.error(f"停止移动失败: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }