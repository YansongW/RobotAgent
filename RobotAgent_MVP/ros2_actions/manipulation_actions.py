"""
机械臂操作动作库
包含抓取、放置等机械臂操作功能
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
    name="pick_up_object",
    description="使用机械臂抓取指定物体",
    category="manipulation",
    parameters={
        "object_type": {
            "type": "string",
            "description": "物体类型",
            "required": True,
            "examples": ["cup", "bottle", "apple", "book"]
        },
        "location": {
            "type": "string",
            "description": "物体位置",
            "default": "table",
            "examples": ["table", "shelf", "floor", "counter"]
        },
        "approach": {
            "type": "string",
            "description": "抓取方式",
            "default": "gentle",
            "options": ["gentle", "firm", "precise"]
        },
        "safety_check": {
            "type": "boolean",
            "description": "是否进行安全检查",
            "default": True
        }
    },
    examples=[
        {
            "object_type": "cup",
            "location": "table",
            "approach": "gentle",
            "safety_check": True
        },
        {
            "object_type": "bottle",
            "location": "shelf",
            "approach": "firm",
            "safety_check": True
        }
    ]
)
async def pick_up_object(
    object_type: str,
    location: str = "table",
    approach: str = "gentle",
    safety_check: bool = True
) -> Dict[str, Any]:
    """抓取物体"""
    try:
        logger.info(f"开始抓取 {object_type}，位置: {location}，方式: {approach}")
        
        # 模拟安全检查
        if safety_check:
            logger.info("执行安全检查...")
            await asyncio.sleep(0.5)
        
        # 模拟移动到目标位置
        logger.info("移动机械臂到目标位置...")
        await asyncio.sleep(1.0)
        
        # 模拟抓取动作
        logger.info(f"使用{approach}方式抓取{object_type}...")
        await asyncio.sleep(1.5)
        
        return {
            "status": "completed",
            "message": f"成功抓取 {object_type}",
            "object_grasped": object_type,
            "grasp_quality": "good",
            "location": location
        }
    except Exception as e:
        logger.error(f"抓取物体失败: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }

@ros2_action(
    name="place_object",
    description="将抓取的物体放置到指定位置",
    category="manipulation",
    parameters={
        "target_location": {
            "type": "string",
            "description": "目标位置",
            "required": True,
            "examples": ["table", "shelf", "box", "tray"]
        },
        "placement_style": {
            "type": "string",
            "description": "放置方式",
            "default": "gentle",
            "options": ["gentle", "precise", "quick"]
        },
        "orientation": {
            "type": "string",
            "description": "物体朝向",
            "default": "upright",
            "options": ["upright", "sideways", "flat"]
        }
    },
    examples=[
        {
            "target_location": "table",
            "placement_style": "gentle",
            "orientation": "upright"
        },
        {
            "target_location": "shelf",
            "placement_style": "precise",
            "orientation": "upright"
        }
    ]
)
async def place_object(
    target_location: str,
    placement_style: str = "gentle",
    orientation: str = "upright"
) -> Dict[str, Any]:
    """放置物体"""
    try:
        logger.info(f"开始放置物体到 {target_location}，方式: {placement_style}")
        
        # 模拟移动到目标位置
        logger.info("移动到目标位置...")
        await asyncio.sleep(1.0)
        
        # 模拟调整朝向
        logger.info(f"调整物体朝向为 {orientation}...")
        await asyncio.sleep(0.5)
        
        # 模拟放置动作
        logger.info(f"使用{placement_style}方式放置物体...")
        await asyncio.sleep(1.0)
        
        return {
            "status": "completed",
            "message": f"成功将物体放置到 {target_location}",
            "placement_location": target_location,
            "final_orientation": orientation
        }
    except Exception as e:
        logger.error(f"放置物体失败: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }

@ros2_action(
    name="open_gripper",
    description="打开机械臂夹爪",
    category="manipulation",
    parameters={
        "opening_width": {
            "type": "float",
            "description": "打开宽度（厘米）",
            "default": 8.0,
            "min": 0.0,
            "max": 15.0
        }
    },
    examples=[
        {"opening_width": 8.0},
        {"opening_width": 12.0}
    ]
)
async def open_gripper(opening_width: float = 8.0) -> Dict[str, Any]:
    """打开夹爪"""
    try:
        logger.info(f"打开夹爪，宽度: {opening_width}厘米")
        
        await asyncio.sleep(0.5)
        
        return {
            "status": "completed",
            "message": f"夹爪已打开到 {opening_width}厘米",
            "gripper_width": opening_width
        }
    except Exception as e:
        logger.error(f"打开夹爪失败: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }

@ros2_action(
    name="close_gripper",
    description="关闭机械臂夹爪",
    category="manipulation",
    parameters={
        "force": {
            "type": "float",
            "description": "夹持力度（0-100）",
            "default": 50.0,
            "min": 10.0,
            "max": 100.0
        }
    },
    examples=[
        {"force": 30.0},
        {"force": 70.0}
    ]
)
async def close_gripper(force: float = 50.0) -> Dict[str, Any]:
    """关闭夹爪"""
    try:
        logger.info(f"关闭夹爪，力度: {force}%")
        
        await asyncio.sleep(0.5)
        
        return {
            "status": "completed",
            "message": f"夹爪已关闭，力度: {force}%",
            "gripper_force": force
        }
    except Exception as e:
        logger.error(f"关闭夹爪失败: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }

@ros2_action(
    name="move_arm_to_position",
    description="移动机械臂到指定位置",
    category="manipulation",
    parameters={
        "x": {
            "type": "float",
            "description": "X坐标（米）",
            "required": True
        },
        "y": {
            "type": "float",
            "description": "Y坐标（米）",
            "required": True
        },
        "z": {
            "type": "float",
            "description": "Z坐标（米）",
            "required": True
        },
        "speed": {
            "type": "float",
            "description": "移动速度（0-100）",
            "default": 50.0,
            "min": 10.0,
            "max": 100.0
        }
    },
    examples=[
        {"x": 0.5, "y": 0.0, "z": 0.3, "speed": 50.0},
        {"x": 0.3, "y": 0.2, "z": 0.4, "speed": 30.0}
    ]
)
async def move_arm_to_position(
    x: float,
    y: float,
    z: float,
    speed: float = 50.0
) -> Dict[str, Any]:
    """移动机械臂到指定位置"""
    try:
        logger.info(f"移动机械臂到位置 ({x}, {y}, {z})，速度: {speed}%")
        
        # 计算移动时间（基于速度）
        move_time = 2.0 * (100 - speed) / 100 + 0.5
        await asyncio.sleep(move_time)
        
        return {
            "status": "completed",
            "message": f"机械臂已移动到位置 ({x}, {y}, {z})",
            "final_position": {"x": x, "y": y, "z": z},
            "move_time": move_time
        }
    except Exception as e:
        logger.error(f"移动机械臂失败: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }