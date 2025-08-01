# 测试配置文件
import os

# 测试环境配置
TEST_CONFIG = {
    "server_url": "http://localhost:8000",
    "test_user_id": "test_user_001",
    "test_session_timeout": 30,  # 秒
    "max_response_time": 5.0,    # 秒
}

# 测试命令集
TEST_COMMANDS = [
    # 基础移动命令
    {
        "command": "移动机械臂到位置 (0.5, 0.3, 0.2)",
        "expected_intent": "move_to_position",
        "expected_action": "move_arm",
        "description": "测试基础位置移动"
    },
    {
        "command": "将机械臂移动到 x=0.2, y=0.4, z=0.3",
        "expected_intent": "move_to_position", 
        "expected_action": "move_arm",
        "description": "测试参数化位置移动"
    },
    
    # 夹爪控制命令
    {
        "command": "打开夹爪",
        "expected_intent": "gripper_control",
        "expected_action": "open_gripper",
        "description": "测试夹爪打开"
    },
    {
        "command": "关闭夹爪",
        "expected_intent": "gripper_control",
        "expected_action": "close_gripper", 
        "description": "测试夹爪关闭"
    },
    
    # 复合动作命令
    {
        "command": "抓取桌上的红色方块",
        "expected_intent": "pick_object",
        "expected_action": "pick_and_place",
        "description": "测试物体抓取"
    },
    {
        "command": "将物体放到指定位置",
        "expected_intent": "place_object",
        "expected_action": "pick_and_place",
        "description": "测试物体放置"
    },
    
    # 状态查询命令
    {
        "command": "获取当前机械臂位置",
        "expected_intent": "get_current_pose",
        "expected_action": "query_status",
        "description": "测试状态查询"
    },
    {
        "command": "检查机械臂状态",
        "expected_intent": "get_arm_status",
        "expected_action": "query_status",
        "description": "测试状态检查"
    },
    
    # 控制命令
    {
        "command": "停止所有运动",
        "expected_intent": "stop_motion",
        "expected_action": "stop_arm",
        "description": "测试紧急停止"
    },
    {
        "command": "回到初始位置",
        "expected_intent": "go_home",
        "expected_action": "move_arm",
        "description": "测试回零位"
    },
    
    # 关节控制命令
    {
        "command": "将第一个关节转动30度",
        "expected_intent": "joint_control",
        "expected_action": "move_joint",
        "description": "测试单关节控制"
    },
    {
        "command": "设置所有关节角度为 [0, 30, 45, 0, 90, 0]",
        "expected_intent": "joint_control",
        "expected_action": "move_joints",
        "description": "测试多关节控制"
    },
    
    # 边界测试命令
    {
        "command": "这是一个无效的命令",
        "expected_intent": "unknown",
        "expected_action": "unknown",
        "description": "测试无效命令处理"
    },
    {
        "command": "",
        "expected_intent": "unknown", 
        "expected_action": "unknown",
        "description": "测试空命令处理"
    }
]

# 性能测试配置
PERFORMANCE_TEST_CONFIG = {
    "concurrent_requests": 5,
    "total_requests": 50,
    "request_interval": 0.1,  # 秒
    "timeout": 10.0,  # 秒
}

# 预期的系统状态
EXPECTED_SYSTEM_STATUS = {
    "qwen_service": ["healthy", "unhealthy"],
    "redis_connection": ["healthy", "unhealthy"], 
    "memory_agent": ["healthy", "unhealthy"],
    "ros2_agent": ["healthy", "unhealthy"]
}

# 测试数据目录
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
TEST_LOGS_DIR = os.path.join(os.path.dirname(__file__), "test_logs")

# 创建测试目录
os.makedirs(TEST_DATA_DIR, exist_ok=True)
os.makedirs(TEST_LOGS_DIR, exist_ok=True)