# RobotAgent 代码架构详细设计

## 1. 项目总体架构

### 1.1 目录结构
```
RobotAgent/
├── src/
│   ├── camel_agents/           # CAMEL智能体模块
│   │   ├── __init__.py
│   │   ├── base_agent.py       # 基础智能体类
│   │   ├── dialog_agent.py     # 对话智能体
│   │   ├── planning_agent.py   # 规划智能体
│   │   ├── decision_agent.py   # 决策智能体
│   │   ├── perception_agent.py # 感知智能体
│   │   ├── learning_agent.py   # 学习智能体
│   │   ├── ros2_agent.py       # ROS2智能体
│   │   └── agent_manager.py    # 智能体管理器
│   ├── ros2_interface/         # ROS2接口模块
│   │   ├── __init__.py
│   │   ├── ros2_wrapper.py     # ROS2包装器
│   │   ├── nodes/              # ROS2节点
│   │   │   ├── __init__.py
│   │   │   ├── command_executor.py
│   │   │   ├── state_monitor.py
│   │   │   ├── safety_controller.py
│   │   │   └── sensor_processor.py
│   │   ├── controllers/        # 控制器
│   │   │   ├── __init__.py
│   │   │   ├── arm_controller.py
│   │   │   ├── base_controller.py
│   │   │   └── gripper_controller.py
│   │   └── messages/           # 消息定义
│   │       ├── __init__.py
│   │       └── custom_msgs.py
│   ├── memory_system/          # 记忆系统模块
│   │   ├── __init__.py
│   │   ├── multimodal_processor.py
│   │   ├── vector_db/          # 向量数据库
│   │   │   ├── __init__.py
│   │   │   ├── milvus_client.py
│   │   │   └── embedding_engine.py
│   │   ├── knowledge_graph/    # 知识图谱
│   │   │   ├── __init__.py
│   │   │   ├── neo4j_client.py
│   │   │   └── graph_builder.py
│   │   ├── rag_engine/         # RAG引擎
│   │   │   ├── __init__.py
│   │   │   ├── retriever.py
│   │   │   └── graph_rag.py
│   │   └── data_strategies/    # 数据策略
│   │       ├── __init__.py
│   │       ├── text_strategy.py
│   │       ├── image_strategy.py
│   │       └── video_strategy.py
│   ├── communication/          # 通信模块
│   │   ├── __init__.py
│   │   ├── message_bus.py      # 消息总线
│   │   ├── protocols/          # 通信协议
│   │   │   ├── __init__.py
│   │   │   ├── camel_protocol.py
│   │   │   └── ros2_protocol.py
│   │   └── serializers/        # 序列化器
│   │       ├── __init__.py
│   │       └── message_serializer.py
│   ├── safety/                 # 安全模块
│   │   ├── __init__.py
│   │   ├── safety_monitor.py   # 安全监控
│   │   ├── emergency_stop.py   # 紧急停止
│   │   └── constraint_checker.py # 约束检查
│   ├── utils/                  # 工具模块
│   │   ├── __init__.py
│   │   ├── config_manager.py   # 配置管理
│   │   ├── logger.py           # 日志管理
│   │   ├── metrics.py          # 性能指标
│   │   └── validators.py       # 验证器
│   └── main.py                 # 主入口文件
├── config/                     # 配置文件
│   ├── agents/                 # 智能体配置
│   ├── ros2/                   # ROS2配置
│   ├── memory/                 # 记忆系统配置
│   └── system/                 # 系统配置
├── tests/                      # 测试文件
├── scripts/                    # 脚本文件
├── requirements.txt            # Python依赖
└── setup.py                    # 安装脚本
```

## 2. 核心模块详细设计

### 2.1 CAMEL智能体模块 (src/camel_agents/)

#### 2.1.1 base_agent.py - 基础智能体类
```python
"""
基础智能体抽象类，定义所有智能体的通用接口和行为
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import BaseModelBackend

class BaseRobotAgent(ABC):
    """机器人智能体基类"""
    
    def __init__(self, 
                 name: str,
                 model_backend: BaseModelBackend,
                 system_message: str,
                 message_bus: 'MessageBus'):
        self.name = name
        self.agent = ChatAgent(
            system_message=system_message,
            model=model_backend
        )
        self.message_bus = message_bus
        self.state = "idle"
        self.capabilities = []
        
    @abstractmethod
    async def process_message(self, message: BaseMessage) -> BaseMessage:
        """处理接收到的消息"""
        pass
        
    @abstractmethod
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行分配的任务"""
        pass
        
    async def send_message(self, recipient: str, content: str, message_type: str = "text"):
        """发送消息到其他智能体"""
        await self.message_bus.send_message(
            sender=self.name,
            recipient=recipient,
            content=content,
            message_type=message_type
        )
        
    def get_status(self) -> Dict[str, Any]:
        """获取智能体状态"""
        return {
            "name": self.name,
            "state": self.state,
            "capabilities": self.capabilities
        }
```

#### 2.1.2 dialog_agent.py - 对话智能体
```python
"""
对话智能体：负责自然语言交互、用户意图理解和响应生成
"""

from typing import Dict, Any, List
from camel.messages import BaseMessage
from .base_agent import BaseRobotAgent

class DialogAgent(BaseRobotAgent):
    """对话智能体"""
    
    def __init__(self, model_backend, message_bus):
        system_message = """
        你是一个智能机器人的对话智能体。你的职责包括：
        1. 理解用户的自然语言指令
        2. 与用户进行友好的对话交互
        3. 将用户意图转换为结构化的任务描述
        4. 协调其他智能体完成用户请求
        """
        super().__init__("DialogAgent", model_backend, system_message, message_bus)
        self.capabilities = ["natural_language_understanding", "intent_recognition", "response_generation"]
        self.conversation_history = []
        
    async def process_message(self, message: BaseMessage) -> BaseMessage:
        """处理用户输入或其他智能体的消息"""
        if message.meta.get("source") == "user":
            return await self._handle_user_input(message)
        else:
            return await self._handle_agent_message(message)
            
    async def _handle_user_input(self, message: BaseMessage) -> BaseMessage:
        """处理用户输入"""
        # 保存对话历史
        self.conversation_history.append({
            "role": "user",
            "content": message.content,
            "timestamp": message.meta.get("timestamp")
        })
        
        # 使用CAMEL智能体理解用户意图
        response = await self.agent.step(message)
        
        # 解析意图并生成任务
        intent = await self._extract_intent(message.content)
        
        if intent["type"] == "robot_command":
            # 发送任务给规划智能体
            await self.send_message(
                recipient="PlanningAgent",
                content=intent,
                message_type="task_request"
            )
            
        return response
        
    async def _extract_intent(self, user_input: str) -> Dict[str, Any]:
        """提取用户意图"""
        intent_prompt = f"""
        分析以下用户输入，提取意图和参数：
        用户输入：{user_input}
        
        请返回JSON格式的意图信息，包括：
        - type: 意图类型（robot_command, question, greeting等）
        - action: 具体动作（如果是机器人指令）
        - parameters: 相关参数
        - priority: 优先级（high, medium, low）
        """
        
        intent_message = BaseMessage.make_user_message(
            role_name="user",
            content=intent_prompt
        )
        
        response = await self.agent.step(intent_message)
        # 解析JSON响应
        return self._parse_intent_response(response.content)
```

#### 2.1.3 planning_agent.py - 规划智能体
```python
"""
规划智能体：负责任务分解、路径规划和执行策略制定
"""

from typing import Dict, Any, List
from .base_agent import BaseRobotAgent

class PlanningAgent(BaseRobotAgent):
    """规划智能体"""
    
    def __init__(self, model_backend, message_bus):
        system_message = """
        你是一个智能机器人的规划智能体。你的职责包括：
        1. 将复杂任务分解为可执行的子任务
        2. 制定任务执行计划和时间安排
        3. 考虑环境约束和安全要求
        4. 协调各个执行模块完成任务
        """
        super().__init__("PlanningAgent", model_backend, system_message, message_bus)
        self.capabilities = ["task_decomposition", "path_planning", "resource_allocation"]
        self.current_plan = None
        self.task_queue = []
        
    async def process_message(self, message: BaseMessage) -> BaseMessage:
        """处理任务请求"""
        if message.meta.get("message_type") == "task_request":
            return await self._create_execution_plan(message.content)
        elif message.meta.get("message_type") == "status_update":
            return await self._update_plan_status(message.content)
            
    async def _create_execution_plan(self, task_description: Dict[str, Any]) -> BaseMessage:
        """创建执行计划"""
        planning_prompt = f"""
        为以下任务创建详细的执行计划：
        任务描述：{task_description}
        
        请考虑：
        1. 任务分解：将复杂任务分解为简单子任务
        2. 执行顺序：确定子任务的执行顺序
        3. 资源需求：每个子任务需要的资源和能力
        4. 安全约束：执行过程中的安全要求
        5. 错误处理：可能的错误情况和应对策略
        
        返回JSON格式的执行计划。
        """
        
        plan_message = BaseMessage.make_user_message(
            role_name="user",
            content=planning_prompt
        )
        
        response = await self.agent.step(plan_message)
        execution_plan = self._parse_plan_response(response.content)
        
        # 保存当前计划
        self.current_plan = execution_plan
        
        # 发送计划给决策智能体
        await self.send_message(
            recipient="DecisionAgent",
            content=execution_plan,
            message_type="execution_plan"
        )
        
        return response
        
    async def _update_plan_status(self, status_update: Dict[str, Any]) -> BaseMessage:
        """更新计划执行状态"""
        if self.current_plan:
            # 更新子任务状态
            task_id = status_update.get("task_id")
            new_status = status_update.get("status")
            
            for subtask in self.current_plan.get("subtasks", []):
                if subtask["id"] == task_id:
                    subtask["status"] = new_status
                    break
                    
            # 检查是否需要重新规划
            if status_update.get("status") == "failed":
                await self._replan_on_failure(task_id, status_update.get("error"))
                
        return BaseMessage.make_assistant_message(
            role_name="assistant",
            content="Plan status updated"
        )
```

#### 2.1.4 ros2_agent.py - ROS2智能体
```python
"""
ROS2智能体：将ROS2系统封装为CAMEL智能体，负责物理世界交互
"""

from typing import Dict, Any, List
import asyncio
from .base_agent import BaseRobotAgent
from ..ros2_interface.ros2_wrapper import ROS2Wrapper

class ROS2Agent(BaseRobotAgent):
    """ROS2智能体 - 机器人的"小脑"，负责低层运动控制"""
    
    def __init__(self, model_backend, message_bus, ros2_config: Dict[str, Any]):
        system_message = """
        你是一个智能机器人的ROS2控制智能体。你的职责包括：
        1. 执行具体的机器人动作指令
        2. 监控机器人硬件状态
        3. 处理传感器数据
        4. 确保运动安全性
        5. 向其他智能体报告执行状态
        """
        super().__init__("ROS2Agent", model_backend, system_message, message_bus)
        self.capabilities = ["motion_control", "sensor_processing", "hardware_monitoring"]
        
        # 初始化ROS2接口
        self.ros2_wrapper = ROS2Wrapper(ros2_config)
        self.robot_state = {
            "position": None,
            "orientation": None,
            "joint_states": None,
            "sensor_data": {},
            "status": "idle"
        }
        
    async def process_message(self, message: BaseMessage) -> BaseMessage:
        """处理来自其他智能体的指令"""
        message_type = message.meta.get("message_type")
        
        if message_type == "motion_command":
            return await self._execute_motion_command(message.content)
        elif message_type == "sensor_request":
            return await self._get_sensor_data(message.content)
        elif message_type == "status_request":
            return await self._get_robot_status()
            
    async def _execute_motion_command(self, command: Dict[str, Any]) -> BaseMessage:
        """执行运动指令"""
        try:
            command_type = command.get("type")
            parameters = command.get("parameters", {})
            
            if command_type == "move_to_position":
                result = await self.ros2_wrapper.move_to_position(
                    x=parameters.get("x"),
                    y=parameters.get("y"),
                    z=parameters.get("z")
                )
            elif command_type == "move_arm":
                result = await self.ros2_wrapper.move_arm(
                    joint_positions=parameters.get("joint_positions")
                )
            elif command_type == "grasp_object":
                result = await self.ros2_wrapper.grasp_object(
                    object_id=parameters.get("object_id")
                )
            else:
                raise ValueError(f"Unknown command type: {command_type}")
                
            # 更新机器人状态
            await self._update_robot_state()
            
            # 报告执行结果
            await self.send_message(
                recipient="DecisionAgent",
                content={
                    "command_id": command.get("id"),
                    "status": "completed" if result["success"] else "failed",
                    "result": result,
                    "robot_state": self.robot_state
                },
                message_type="execution_result"
            )
            
            return BaseMessage.make_assistant_message(
                role_name="assistant",
                content=f"Motion command executed: {result}"
            )
            
        except Exception as e:
            error_msg = f"Failed to execute motion command: {str(e)}"
            
            # 报告错误
            await self.send_message(
                recipient="DecisionAgent",
                content={
                    "command_id": command.get("id"),
                    "status": "failed",
                    "error": error_msg,
                    "robot_state": self.robot_state
                },
                message_type="execution_result"
            )
            
            return BaseMessage.make_assistant_message(
                role_name="assistant",
                content=error_msg
            )
            
    async def _update_robot_state(self):
        """更新机器人状态"""
        self.robot_state = await self.ros2_wrapper.get_robot_state()
        
    async def start_monitoring(self):
        """启动状态监控"""
        while True:
            await self._update_robot_state()
            
            # 检查异常状态
            if self._detect_anomaly():
                await self.send_message(
                    recipient="DecisionAgent",
                    content={
                        "type": "anomaly_detected",
                        "robot_state": self.robot_state,
                        "timestamp": asyncio.get_event_loop().time()
                    },
                    message_type="alert"
                )
                
            await asyncio.sleep(0.1)  # 100ms监控周期
```

### 2.2 ROS2接口模块 (src/ros2_interface/)

#### 2.2.1 ros2_wrapper.py - ROS2包装器
```python
"""
ROS2包装器：提供统一的ROS2接口，封装底层ROS2复杂性
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import asyncio
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

class ROS2Wrapper:
    """ROS2系统包装器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.executor = None
        self.nodes = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # 初始化ROS2
        rclpy.init()
        self._setup_nodes()
        
    def _setup_nodes(self):
        """设置ROS2节点"""
        from .nodes.command_executor import CommandExecutorNode
        from .nodes.state_monitor import StateMonitorNode
        from .nodes.safety_controller import SafetyControllerNode
        from .nodes.sensor_processor import SensorProcessorNode
        
        # 创建节点
        self.nodes['command_executor'] = CommandExecutorNode()
        self.nodes['state_monitor'] = StateMonitorNode()
        self.nodes['safety_controller'] = SafetyControllerNode()
        self.nodes['sensor_processor'] = SensorProcessorNode()
        
        # 设置执行器
        self.executor = MultiThreadedExecutor()
        for node in self.nodes.values():
            self.executor.add_node(node)
            
        # 在后台线程中运行执行器
        self.executor_thread = self.thread_pool.submit(self.executor.spin)
        
    async def move_to_position(self, x: float, y: float, z: float) -> Dict[str, Any]:
        """移动到指定位置"""
        command = {
            "type": "move_to_position",
            "target": {"x": x, "y": y, "z": z}
        }
        
        # 异步执行ROS2命令
        future = self.thread_pool.submit(
            self.nodes['command_executor'].execute_command, 
            command
        )
        
        # 等待执行完成
        result = await asyncio.wrap_future(future)
        return result
        
    async def move_arm(self, joint_positions: List[float]) -> Dict[str, Any]:
        """移动机械臂"""
        command = {
            "type": "move_arm",
            "joint_positions": joint_positions
        }
        
        future = self.thread_pool.submit(
            self.nodes['command_executor'].execute_command,
            command
        )
        
        result = await asyncio.wrap_future(future)
        return result
        
    async def get_robot_state(self) -> Dict[str, Any]:
        """获取机器人状态"""
        future = self.thread_pool.submit(
            self.nodes['state_monitor'].get_current_state
        )
        
        state = await asyncio.wrap_future(future)
        return state
        
    async def get_sensor_data(self, sensor_types: List[str] = None) -> Dict[str, Any]:
        """获取传感器数据"""
        future = self.thread_pool.submit(
            self.nodes['sensor_processor'].get_sensor_data,
            sensor_types
        )
        
        data = await asyncio.wrap_future(future)
        return data
        
    def shutdown(self):
        """关闭ROS2系统"""
        if self.executor:
            self.executor.shutdown()
        rclpy.shutdown()
        self.thread_pool.shutdown(wait=True)
```

#### 2.2.2 nodes/command_executor.py - 命令执行节点
```python
"""
命令执行节点：负责执行具体的机器人动作指令
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Twist
from moveit_msgs.action import MoveGroup
from control_msgs.action import FollowJointTrajectory
from typing import Dict, Any, List
import threading

class CommandExecutorNode(Node):
    """命令执行节点"""
    
    def __init__(self):
        super().__init__('command_executor')
        
        # 动作客户端
        self.move_group_client = ActionClient(self, MoveGroup, '/move_group')
        self.joint_trajectory_client = ActionClient(
            self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory'
        )
        
        # 发布器
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pose_publisher = self.create_publisher(PoseStamped, '/move_base_simple/goal', 10)
        
        # 执行状态
        self.execution_lock = threading.Lock()
        self.current_execution = None
        
        self.get_logger().info('Command Executor Node initialized')
        
    def execute_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """执行命令"""
        with self.execution_lock:
            try:
                command_type = command.get("type")
                
                if command_type == "move_to_position":
                    return self._execute_move_to_position(command)
                elif command_type == "move_arm":
                    return self._execute_move_arm(command)
                elif command_type == "move_base":
                    return self._execute_move_base(command)
                else:
                    return {
                        "success": False,
                        "error": f"Unknown command type: {command_type}"
                    }
                    
            except Exception as e:
                self.get_logger().error(f"Command execution failed: {str(e)}")
                return {
                    "success": False,
                    "error": str(e)
                }
                
    def _execute_move_to_position(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """执行位置移动"""
        target = command.get("target")
        
        # 创建目标姿态
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "base_link"
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = target["x"]
        goal_msg.pose.position.y = target["y"]
        goal_msg.pose.position.z = target["z"]
        goal_msg.pose.orientation.w = 1.0
        
        # 发布目标
        self.pose_publisher.publish(goal_msg)
        
        # 等待执行完成（简化版本）
        # 实际实现中需要监控执行状态
        
        return {
            "success": True,
            "final_position": target,
            "execution_time": 0.0
        }
        
    def _execute_move_arm(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """执行机械臂移动"""
        joint_positions = command.get("joint_positions")
        
        # 创建关节轨迹目标
        goal = FollowJointTrajectory.Goal()
        
        # 设置轨迹点
        trajectory_point = JointTrajectoryPoint()
        trajectory_point.positions = joint_positions
        trajectory_point.time_from_start.sec = 2  # 2秒内完成
        
        goal.trajectory.points = [trajectory_point]
        goal.trajectory.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint", 
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]
        
        # 发送目标并等待结果
        if not self.joint_trajectory_client.wait_for_server(timeout_sec=5.0):
            return {
                "success": False,
                "error": "Joint trajectory action server not available"
            }
            
        future = self.joint_trajectory_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        
        goal_handle = future.result()
        if not goal_handle.accepted:
            return {
                "success": False,
                "error": "Goal rejected by action server"
            }
            
        # 等待执行完成
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        
        result = result_future.result()
        
        return {
            "success": result.result.error_code == 0,
            "final_joint_positions": joint_positions,
            "execution_time": 2.0
        }
```

### 2.3 记忆系统模块 (src/memory_system/)

#### 2.3.1 LangGraph工作流引擎 (langgraph_engine.py)
```python
"""
LangGraph工作流引擎：管理记忆处理的状态图工作流
"""

from langgraph import StateGraph, END
from typing import Dict, Any, List, Optional
import asyncio
from enum import Enum

class MemoryCategory(Enum):
    """记忆分类枚举"""
    AGENT_MEMORY = "agent_memory"           # 智能体记忆
    TASK_EXPERIENCE = "task_experience"     # 任务经验
    DOMAIN_KNOWLEDGE = "domain_knowledge"   # 领域知识
    EPISODIC_MEMORY = "episodic_memory"     # 情节记忆
    SEMANTIC_MEMORY = "semantic_memory"     # 语义记忆
    PROCEDURAL_MEMORY = "procedural_memory" # 程序记忆

class MemoryWorkflowState:
    """记忆工作流状态"""
    
    def __init__(self):
        self.memory_data: Dict[str, Any] = {}
        self.classification: Dict[str, Any] = {}
        self.processing_status: str = "pending"
        self.storage_results: Dict[str, str] = {}
        self.error_info: Optional[str] = None
        self.checkpoint_id: Optional[str] = None
        
class MemoryWorkflowEngine:
    """记忆工作流引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.workflow = self._create_memory_workflow()
        self.checkpoints = {}
        
    def _create_memory_workflow(self) -> StateGraph:
        """创建记忆处理工作流"""
        workflow = StateGraph(MemoryWorkflowState)
        
        # 添加节点
        workflow.add_node("classify_memory", self._classify_memory_node)
        workflow.add_node("process_agent_memory", self._process_agent_memory_node)
        workflow.add_node("process_task_experience", self._process_task_experience_node)
        workflow.add_node("process_domain_knowledge", self._process_domain_knowledge_node)
        workflow.add_node("store_memory", self._store_memory_node)
        workflow.add_node("update_knowledge_graph", self._update_knowledge_graph_node)
        workflow.add_node("human_intervention", self._human_intervention_node)
        
        # 设置入口点
        workflow.set_entry_point("classify_memory")
        
        # 添加条件边
        workflow.add_conditional_edges(
            "classify_memory",
            self._route_by_category,
            {
                "agent_memory": "process_agent_memory",
                "task_experience": "process_task_experience", 
                "domain_knowledge": "process_domain_knowledge",
                "human_intervention": "human_intervention"
            }
        )
        
        # 添加边
        workflow.add_edge("process_agent_memory", "store_memory")
        workflow.add_edge("process_task_experience", "store_memory")
        workflow.add_edge("process_domain_knowledge", "store_memory")
        workflow.add_edge("store_memory", "update_knowledge_graph")
        workflow.add_edge("update_knowledge_graph", END)
        workflow.add_edge("human_intervention", "store_memory")
        
        return workflow.compile()
        
    async def _classify_memory_node(self, state: MemoryWorkflowState) -> MemoryWorkflowState:
        """记忆分类节点"""
        from .classifiers.memory_classifier import MemoryClassifier
        
        classifier = MemoryClassifier()
        classification = classifier.classify_memory(
            state.memory_data.get("content"),
            state.memory_data.get("metadata", {})
        )
        
        state.classification = classification
        state.processing_status = "classified"
        return state
        
    async def _process_agent_memory_node(self, state: MemoryWorkflowState) -> MemoryWorkflowState:
        """智能体记忆处理节点"""
        from .processors.agent_memory_processor import AgentMemoryProcessor
        
        processor = AgentMemoryProcessor()
        processed_data = await processor.process(state.memory_data, state.classification)
        
        state.memory_data.update(processed_data)
        state.processing_status = "agent_processed"
        return state
        
    def _route_by_category(self, state: MemoryWorkflowState) -> str:
        """根据记忆类别路由"""
        category = state.classification.get("category")
        
        # 检查是否需要人工干预
        if state.classification.get("requires_human_intervention", False):
            return "human_intervention"
            
        if category == MemoryCategory.AGENT_MEMORY.value:
            return "agent_memory"
        elif category == MemoryCategory.TASK_EXPERIENCE.value:
            return "task_experience"
        elif category == MemoryCategory.DOMAIN_KNOWLEDGE.value:
            return "domain_knowledge"
        else:
            return "agent_memory"  # 默认路由
```

#### 2.3.2 多模态分类存储管理器 (storage_manager.py)
```python
"""
多模态分类存储管理器：智能分类和分层存储多模态记忆数据
"""

import asyncio
from typing import Dict, Any, List, Optional
from enum import Enum

class StorageTier(Enum):
    """存储层级"""
    HOT = "hot"           # 热存储 - 内存缓存
    WARM = "warm"         # 温存储 - SSD
    COLD = "cold"         # 冷存储 - HDD
    ARCHIVE = "archive"   # 归档存储 - 对象存储

class MultiModalStorageManager:
    """多模态存储管理器"""
    
    def __init__(self):
        self.vector_db = MilvusClient()
        self.graph_db = Neo4jClient()
        self.object_storage = MinIOClient()
        self.tiered_storage = TieredStorageSystem()
        self.classifier = MemoryClassifier()
        
    async def store_memory(self, memory_data: Dict[str, Any]) -> str:
        """存储记忆数据"""
        # 1. 分类记忆
        classification = self.classifier.classify_memory(
            memory_data["content"],
            memory_data["metadata"]
        )
        
        # 2. 确定存储策略
        storage_strategy = classification["storage_strategy"]
        
        # 3. 多源存储
        storage_results = {}
        
        if storage_strategy["use_vector_db"]:
            vector_id = await self.store_in_vector_db(memory_data, classification)
            storage_results["vector_id"] = vector_id
            
        if storage_strategy["use_graph_db"]:
            graph_id = await self.store_in_graph_db(memory_data, classification)
            storage_results["graph_id"] = graph_id
            
        if storage_strategy["use_object_storage"]:
            object_id = await self.store_in_object_storage(memory_data, classification)
            storage_results["object_id"] = object_id
            
        # 4. 分层存储
        tier = self.tiered_storage.determine_storage_tier(classification)
        await self.tiered_storage.store_in_tier(memory_data, tier)
        
        # 5. 创建记忆索引
        memory_id = self.create_memory_index(storage_results, classification)
        
        return memory_id
```

#### 2.3.3 知识图谱可视化系统 (visualization/)
```python
"""
知识图谱可视化系统：提供交互式图谱可视化和分析功能
"""

import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, Any, List, Optional

class KnowledgeGraphVisualizer:
    """知识图谱可视化器"""
    
    def __init__(self, graph_db):
        self.graph_db = graph_db
        self.layout_algorithms = {
            "force_directed": nx.spring_layout,
            "hierarchical": nx.nx_agraph.graphviz_layout,
            "circular": nx.circular_layout,
            "random": nx.random_layout
        }
        
    def visualize_memory_graph(self,
                              memory_ids: Optional[List[str]] = None,
                              categories: Optional[List[str]] = None,
                              time_range: Optional[Dict[str, str]] = None,
                              layout: str = "force_directed") -> go.Figure:
        """可视化记忆知识图谱"""
        
        # 1. 构建查询条件
        query_conditions = self.build_query_conditions(memory_ids, categories, time_range)
        
        # 2. 从图数据库获取数据
        graph_data = self.graph_db.query_graph(query_conditions)
        
        # 3. 构建NetworkX图
        G = self.build_networkx_graph(graph_data)
        
        # 4. 计算布局
        layout_func = self.layout_algorithms.get(layout, nx.spring_layout)
        pos = layout_func(G)
        
        # 5. 创建Plotly图形
        fig = self.create_plotly_graph(G, pos)
        
        return fig
        
class MemoryVisualizationWebApp:
    """记忆可视化Web应用"""
    
    def __init__(self, memory_system):
        self.memory_system = memory_system
        self.visualizer = KnowledgeGraphVisualizer(memory_system.graph_db)
        
    def run_app(self):
        """运行Streamlit应用"""
        st.set_page_config(
            page_title="RobotAgent 多模态记忆系统",
            page_icon="🧠",
            layout="wide"
        )
        
        st.title("🧠 多模态记忆系统可视化")
        
        # 创建标签页
        tabs = st.tabs([
            "📊 总览仪表板", "🕸️ 知识图谱", "📈 分析报告",
            "🔍 记忆搜索", "⚙️ 系统管理"
        ])
        
        with tabs[0]:
            self.create_overview_dashboard()
            
        with tabs[1]:
            self.create_knowledge_graph_view()
            
        with tabs[2]:
            self.create_analytics_view()
```
#### 2.3.4 多模态处理器 (multimodal_processor.py)
```python
"""
多模态数据处理器：统一处理文本、图像、视频等多种模态数据
"""

import asyncio
from typing import Dict, Any, List, Union, Optional
import numpy as np
from PIL import Image
import cv2
from transformers import CLIPProcessor, CLIPModel
import torch

class MultimodalProcessor:
    """多模态数据处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 加载多模态模型
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model.to(self.device)
        
    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理多模态数据"""
        data_type = data.get("type")
        content = data.get("content")
        
        if data_type == "text":
            return await self._process_text(content, data.get("metadata", {}))
        elif data_type == "image":
            return await self._process_image(content, data.get("metadata", {}))
        elif data_type == "video":
            return await self._process_video(content, data.get("metadata", {}))
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
            
    async def _process_text(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """处理文本数据"""
        # 生成文本嵌入
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            embedding = text_features.cpu().numpy().flatten()
            
        # 提取关键信息
        keywords = await self._extract_keywords(text)
        entities = await self._extract_entities(text)
        
        return {
            "type": "text",
            "content": text,
            "embedding": embedding.tolist(),
            "keywords": keywords,
            "entities": entities,
            "metadata": metadata,
            "processed_at": asyncio.get_event_loop().time()
        }
```

#### 2.3.5 向量数据库客户端 (vector_db/milvus_client.py)
```python
"""
Milvus向量数据库客户端：管理多模态数据的向量存储和检索
"""

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from typing import Dict, Any, List, Optional
import numpy as np
import asyncio

class MilvusClient:
    """Milvus向量数据库客户端"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 19530)
        self.collections = {}
        
        # 连接到Milvus
        connections.connect(
            alias="default",
            host=self.host,
            port=self.port
        )
        
    async def create_collection(self, collection_name: str, dimension: int = 512):
        """创建集合"""
        if utility.has_collection(collection_name):
            return
            
        # 定义字段模式
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="memory_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="modality", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
            FieldSchema(name="importance_score", dtype=DataType.FLOAT)
        ]
        
        # 创建集合模式
        schema = CollectionSchema(fields, f"Collection for {collection_name}")
        collection = Collection(collection_name, schema)
        
        # 创建索引
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index("embedding", index_params)
        
        self.collections[collection_name] = collection
        
    async def insert_memory(self, collection_name: str, memory_data: Dict[str, Any]) -> str:
        """插入记忆数据"""
        if collection_name not in self.collections:
            await self.create_collection(collection_name)
            
        collection = self.collections[collection_name]
        
        # 准备数据
        entities = [
            [memory_data["memory_id"]],
            [memory_data["category"]],
            [memory_data["modality"]],
            [memory_data["embedding"]],
            [memory_data["timestamp"]],
            [memory_data["importance_score"]]
        ]
        
        # 插入数据
        mr = collection.insert(entities)
        collection.flush()
        
        return mr.primary_keys[0]
        
    async def search_similar_memories(self, 
                                    collection_name: str,
                                    query_embedding: List[float],
                                    top_k: int = 10,
                                    filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """搜索相似记忆"""
        if collection_name not in self.collections:
            return []
            
        collection = self.collections[collection_name]
        collection.load()
        
        # 构建搜索参数
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        
        # 构建过滤表达式
        expr = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, str):
                    conditions.append(f'{key} == "{value}"')
                else:
                    conditions.append(f'{key} == {value}')
            expr = " and ".join(conditions)
        
        # 执行搜索
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["memory_id", "category", "modality", "timestamp", "importance_score"]
        )
        
        # 格式化结果
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.id,
                    "memory_id": hit.entity.get("memory_id"),
                    "category": hit.entity.get("category"),
                    "modality": hit.entity.get("modality"),
                    "timestamp": hit.entity.get("timestamp"),
                    "importance_score": hit.entity.get("importance_score"),
                    "similarity_score": hit.score
                })
                
        return formatted_results
```
        
        # 初始化集合
        self._setup_collections()
        
    def _setup_collections(self):
        """设置向量集合"""
        # 多模态数据集合
        multimodal_fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="data_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
            FieldSchema(name="timestamp", dtype=DataType.DOUBLE),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        
        multimodal_schema = CollectionSchema(
            fields=multimodal_fields,
            description="Multimodal data collection"
        )
        
        # 创建或获取集合
        collection_name = "multimodal_data"
        if utility.has_collection(collection_name):
            self.collections[collection_name] = Collection(collection_name)
        else:
            self.collections[collection_name] = Collection(
                name=collection_name,
                schema=multimodal_schema
            )
            
        # 创建索引
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        
        self.collections[collection_name].create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        # 加载集合
        self.collections[collection_name].load()
        
    async def insert_data(self, data: Dict[str, Any]) -> str:
        """插入数据"""
        collection = self.collections["multimodal_data"]
        
        # 准备插入数据
        insert_data = [
            [data["type"]],                    # data_type
            [data.get("content_hash", "")],    # content_hash
            [data["embedding"]],               # embedding
            [data["processed_at"]],            # timestamp
            [data.get("metadata", {})]         # metadata
        ]
        
        # 插入数据
        result = collection.insert(insert_data)
        
        # 刷新以确保数据持久化
        collection.flush()
        
        return result.primary_keys[0]
        
    async def search_similar(self, 
                           query_embedding: List[float],
                           data_type: Optional[str] = None,
                           top_k: int = 10,
                           score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """搜索相似数据"""
        collection = self.collections["multimodal_data"]
        
        # 构建搜索参数
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        # 构建过滤表达式
        expr = None
        if data_type:
            expr = f'data_type == "{data_type}"'
            
        # 执行搜索
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["data_type", "content_hash", "timestamp", "metadata"]
        )
        
        # 处理结果
        similar_data = []
        for hits in results:
            for hit in hits:
                if hit.score >= score_threshold:
                    similar_data.append({
                        "id": hit.id,
                        "score": hit.score,
                        "data_type": hit.entity.get("data_type"),
                        "content_hash": hit.entity.get("content_hash"),
                        "timestamp": hit.entity.get("timestamp"),
                        "metadata": hit.entity.get("metadata")
                    })
                    
        return similar_data
        
    async def get_by_id(self, data_id: int) -> Optional[Dict[str, Any]]:
        """根据ID获取数据"""
        collection = self.collections["multimodal_data"]
        
        result = collection.query(
            expr=f"id == {data_id}",
            output_fields=["data_type", "content_hash", "embedding", "timestamp", "metadata"]
        )
        
        if result:
            return result[0]
        return None
        
    async def delete_data(self, data_id: int) -> bool:
        """删除数据"""
        collection = self.collections["multimodal_data"]
        
        try:
            collection.delete(expr=f"id == {data_id}")
            collection.flush()
            return True
        except Exception as e:
            print(f"Failed to delete data: {e}")
            return False
            
    def close(self):
        """关闭连接"""
        connections.disconnect("default")
```

### 2.4 通信模块 (src/communication/)

#### 2.4.1 message_bus.py - 消息总线
```python
"""
消息总线：实现智能体间的异步通信和消息路由
"""

import asyncio
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import json
import uuid
from datetime import datetime

class MessageType(Enum):
    """消息类型枚举"""
    TEXT = "text"
    TASK_REQUEST = "task_request"
    EXECUTION_PLAN = "execution_plan"
    MOTION_COMMAND = "motion_command"
    SENSOR_REQUEST = "sensor_request"
    STATUS_UPDATE = "status_update"
    EXECUTION_RESULT = "execution_result"
    ALERT = "alert"
    BROADCAST = "broadcast"

@dataclass
class Message:
    """消息数据结构"""
    id: str
    sender: str
    recipient: str
    message_type: MessageType
    content: Any
    timestamp: datetime
    priority: int = 1  # 1=低, 2=中, 3=高
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "sender": self.sender,
            "recipient": self.recipient,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority,
            "metadata": self.metadata or {}
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """从字典创建消息"""
        return cls(
            id=data["id"],
            sender=data["sender"],
            recipient=data["recipient"],
            message_type=MessageType(data["message_type"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            priority=data.get("priority", 1),
            metadata=data.get("metadata", {})
        )

class MessageBus:
    """消息总线"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.message_history: List[Message] = []
        self.running = False
        self.worker_task = None
        
    async def start(self):
        """启动消息总线"""
        self.running = True
        self.worker_task = asyncio.create_task(self._message_worker())
        
    async def stop(self):
        """停止消息总线"""
        self.running = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
                
    def subscribe(self, agent_name: str, callback: Callable[[Message], None]):
        """订阅消息"""
        if agent_name not in self.subscribers:
            self.subscribers[agent_name] = []
        self.subscribers[agent_name].append(callback)
        
    def unsubscribe(self, agent_name: str, callback: Callable[[Message], None]):
        """取消订阅"""
        if agent_name in self.subscribers:
            self.subscribers[agent_name].remove(callback)
            
    async def send_message(self, 
                          sender: str,
                          recipient: str,
                          content: Any,
                          message_type: str = "text",
                          priority: int = 1,
                          metadata: Dict[str, Any] = None) -> str:
        """发送消息"""
        message = Message(
            id=str(uuid.uuid4()),
            sender=sender,
            recipient=recipient,
            message_type=MessageType(message_type),
            content=content,
            timestamp=datetime.now(),
            priority=priority,
            metadata=metadata
        )
        
        # 添加到队列
        await self.message_queue.put(message)
        
        # 记录消息历史
        self.message_history.append(message)
        
        return message.id
        
    async def broadcast_message(self,
                              sender: str,
                              content: Any,
                              message_type: str = "broadcast",
                              priority: int = 1,
                              metadata: Dict[str, Any] = None) -> str:
        """广播消息"""
        return await self.send_message(
            sender=sender,
            recipient="*",  # 广播标识
            content=content,
            message_type=message_type,
            priority=priority,
            metadata=metadata
        )
        
    async def _message_worker(self):
        """消息处理工作线程"""
        while self.running:
            try:
                # 获取消息（按优先级排序）
                message = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=1.0
                )
                
                # 路由消息
                await self._route_message(message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Message worker error: {e}")
                
    async def _route_message(self, message: Message):
        """路由消息到目标智能体"""
        if message.recipient == "*":
            # 广播消息
            for agent_name, callbacks in self.subscribers.items():
                if agent_name != message.sender:  # 不发送给自己
                    for callback in callbacks:
                        try:
                            await self._call_callback(callback, message)
                        except Exception as e:
                            print(f"Callback error for {agent_name}: {e}")
        else:
            # 单播消息
            if message.recipient in self.subscribers:
                for callback in self.subscribers[message.recipient]:
                    try:
                        await self._call_callback(callback, message)
                    except Exception as e:
                        print(f"Callback error for {message.recipient}: {e}")
                        
    async def _call_callback(self, callback: Callable, message: Message):
        """调用回调函数"""
        if asyncio.iscoroutinefunction(callback):
            await callback(message)
        else:
            callback(message)
            
    def get_message_history(self, 
                          agent_name: Optional[str] = None,
                          message_type: Optional[MessageType] = None,
                          limit: int = 100) -> List[Message]:
        """获取消息历史"""
        filtered_messages = self.message_history
        
        if agent_name:
            filtered_messages = [
                msg for msg in filtered_messages 
                if msg.sender == agent_name or msg.recipient == agent_name
            ]
            
        if message_type:
            filtered_messages = [
                msg for msg in filtered_messages 
                if msg.message_type == message_type
            ]
            
        return filtered_messages[-limit:]
```

## 3. 主入口文件 (src/main.py)

```python
"""
RobotAgent主入口文件：初始化和启动整个系统
"""

import asyncio
import signal
import sys
from typing import Dict, Any
import yaml
import logging
from pathlib import Path

# 导入核心模块
from camel_agents.agent_manager import AgentManager
from communication.message_bus import MessageBus
from memory_system.multimodal_processor import MultimodalProcessor
from memory_system.vector_db.milvus_client import MilvusClient
from memory_system.knowledge_graph.neo4j_client import Neo4jClient
from ros2_interface.ros2_wrapper import ROS2Wrapper
from safety.safety_monitor import SafetyMonitor
from utils.config_manager import ConfigManager
from utils.logger import setup_logging

class RobotAgent:
    """机器人智能体主类"""
    
    def __init__(self, config_path: str = "config/system/main.yaml"):
        self.config_path = config_path
        self.config = None
        self.message_bus = None
        self.agent_manager = None
        self.memory_system = None
        self.ros2_wrapper = None
        self.safety_monitor = None
        self.running = False
        
    async def initialize(self):
        """初始化系统"""
        try:
            # 加载配置
            self.config = ConfigManager.load_config(self.config_path)
            
            # 设置日志
            setup_logging(self.config.get("logging", {}))
            logging.info("Starting RobotAgent initialization...")
            
            # 初始化消息总线
            self.message_bus = MessageBus()
            await self.message_bus.start()
            logging.info("Message bus started")
            
            # 初始化记忆系统
            await self._initialize_memory_system()
            logging.info("Memory system initialized")
            
            # 初始化ROS2接口
            if self.config.get("ros2", {}).get("enabled", True):
                self.ros2_wrapper = ROS2Wrapper(self.config["ros2"])
                logging.info("ROS2 interface initialized")
            
            # 初始化安全监控
            self.safety_monitor = SafetyMonitor(
                self.config.get("safety", {}),
                self.message_bus
            )
            await self.safety_monitor.start()
            logging.info("Safety monitor started")
            
            # 初始化智能体管理器
            self.agent_manager = AgentManager(
                config=self.config["agents"],
                message_bus=self.message_bus,
                memory_system=self.memory_system,
                ros2_wrapper=self.ros2_wrapper
            )
            await self.agent_manager.initialize()
            logging.info("Agent manager initialized")
            
            logging.info("RobotAgent initialization completed successfully")
            
        except Exception as e:
            logging.error(f"Initialization failed: {e}")
            raise
            
    async def _initialize_memory_system(self):
        """初始化记忆系统"""
        memory_config = self.config.get("memory", {})
        
        # 初始化多模态处理器
        multimodal_processor = MultimodalProcessor(memory_config.get("multimodal", {}))
        
        # 初始化向量数据库
        vector_db = MilvusClient(memory_config.get("vector_db", {}))
        
        # 初始化知识图谱
        knowledge_graph = Neo4jClient(memory_config.get("knowledge_graph", {}))
        
        self.memory_system = {
            "multimodal_processor": multimodal_processor,
            "vector_db": vector_db,
            "knowledge_graph": knowledge_graph
        }
        
    async def start(self):
        """启动系统"""
        if not self.agent_manager:
            await self.initialize()
            
        self.running = True
        logging.info("Starting RobotAgent...")
        
        try:
            # 启动所有智能体
            await self.agent_manager.start_all_agents()
            
            # 启动安全监控
            if self.safety_monitor:
                await self.safety_monitor.start_monitoring()
                
            logging.info("RobotAgent started successfully")
            
            # 保持运行
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logging.error(f"Runtime error: {e}")
            await self.shutdown()
            
    async def shutdown(self):
        """关闭系统"""
        logging.info("Shutting down RobotAgent...")
        self.running = False
        
        try:
            # 停止智能体
            if self.agent_manager:
                await self.agent_manager.stop_all_agents()
                
            # 停止安全监控
            if self.safety_monitor:
                await self.safety_monitor.stop()
                
            # 停止消息总线
            if self.message_bus:
                await self.message_bus.stop()
                
            # 关闭ROS2接口
            if self.ros2_wrapper:
                self.ros2_wrapper.shutdown()
                
            # 关闭记忆系统
            if self.memory_system:
                self.memory_system["vector_db"].close()
                self.memory_system["knowledge_graph"].close()
                
            logging.info("RobotAgent shutdown completed")
            
        except Exception as e:
            logging.error(f"Shutdown error: {e}")

def signal_handler(signum, frame):
    """信号处理器"""
    print(f"\nReceived signal {signum}, shutting down...")
    sys.exit(0)

async def main():
    """主函数"""
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 创建并启动RobotAgent
    robot_agent = RobotAgent()
    
    try:
        await robot_agent.start()
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt")
    finally:
        await robot_agent.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

## 4. 配置文件结构

### 4.1 主配置文件 (config/system/main.yaml)
```yaml
# RobotAgent主配置文件

# 系统信息
system:
  name: "RobotAgent"
  version: "1.0.0"
  environment: "development"  # development, testing, production

# 日志配置
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/robot_agent.log"
  max_size: "100MB"
  backup_count: 5

# 智能体配置
agents:
  model_backend:
    type: "openai"  # openai, anthropic, local
    model_name: "gpt-4"
    api_key: "${OPENAI_API_KEY}"
    temperature: 0.7
    max_tokens: 2048
    
  dialog_agent:
    enabled: true
    system_message_file: "config/agents/dialog_agent_prompt.txt"
    
  planning_agent:
    enabled: true
    system_message_file: "config/agents/planning_agent_prompt.txt"
    
  decision_agent:
    enabled: true
    system_message_file: "config/agents/decision_agent_prompt.txt"
    
  perception_agent:
    enabled: true
    system_message_file: "config/agents/perception_agent_prompt.txt"
    
  learning_agent:
    enabled: true
    system_message_file: "config/agents/learning_agent_prompt.txt"
    
  ros2_agent:
    enabled: true
    system_message_file: "config/agents/ros2_agent_prompt.txt"

# ROS2配置
ros2:
  enabled: true
  domain_id: 0
  nodes:
    command_executor:
      enabled: true
      namespace: "/robot"
    state_monitor:
      enabled: true
      namespace: "/robot"
    safety_controller:
      enabled: true
      namespace: "/robot"
    sensor_processor:
      enabled: true
      namespace: "/robot"
      
  controllers:
    arm_controller:
      type: "moveit"
      planning_group: "arm"
      max_velocity: 1.0
      max_acceleration: 1.0
    base_controller:
      type: "nav2"
      max_linear_velocity: 1.0
      max_angular_velocity: 1.0

# 记忆系统配置
memory:
  multimodal:
    clip_model: "openai/clip-vit-base-patch32"
    device: "auto"  # auto, cpu, cuda
    
  vector_db:
    type: "milvus"
    host: "localhost"
    port: 19530
    collection_name: "multimodal_data"
    dimension: 512
    
  knowledge_graph:
    type: "neo4j"
    uri: "bolt://localhost:7687"
    username: "neo4j"
    password: "${NEO4J_PASSWORD}"
    database: "robot_memory"

# 安全配置
safety:
  enabled: true
  emergency_stop:
    enabled: true
    trigger_conditions:
      - "collision_detected"
      - "human_too_close"
      - "system_error"
  constraints:
    workspace_limits:
      x_min: -2.0
      x_max: 2.0
      y_min: -2.0
      y_max: 2.0
      z_min: 0.0
      z_max: 2.0
    velocity_limits:
      linear_max: 1.0
      angular_max: 1.0
    force_limits:
      max_force: 100.0  # N

# 通信配置
communication:
  message_bus:
    queue_size: 1000
    worker_threads: 4
    message_timeout: 30.0  # seconds
    
# 性能配置
performance:
  monitoring:
    enabled: true
    metrics_interval: 1.0  # seconds
    
  optimization:
    batch_processing: true
    cache_size: 1000
    parallel_processing: true
```

这个详细的代码架构文档涵盖了：

1. **完整的目录结构**：展示了项目的组织方式
2. **核心模块设计**：包括CAMEL智能体、ROS2接口、记忆系统、通信模块
3. **具体代码实现**：提供了关键类和函数的详细实现
4. **配置管理**：展示了系统配置的结构和内容
5. **系统集成**：说明了各模块如何协同工作

这个架构实现了您要求的核心思想：
- 以CAMEL的Agent架构为核心
- ROS2作为独立的简单Agent
- 多模态记忆系统（RAG + GraphRAG）
- "大脑-小脑"架构（认知决策 + 运动控制）

您希望我继续完善某个特定模块的实现细节吗？