import json
import asyncio
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from utils.config import Config
from utils.logger import CustomLogger
from services.message_queue import MessageQueue
from models.message_models import QueueMessage, MessageType, ROS2CommandMessage

class ROS2Agent:
    """ROS2 Agent - 负责处理ROS2机械臂控制命令"""
    
    def __init__(self, config: Config, message_queue: MessageQueue):
        self.config = config
        self.message_queue = message_queue
        self.logger = CustomLogger("ROS2Agent")
        
        # 加载ROS2命令映射
        self.command_mappings = self._load_command_mappings()
        
        # 运行状态
        self.is_running = False
        self.ros2_available = False
        self.gazebo_running = False
        self.processed_count = 0
        
        # 机械臂状态
        self.current_pose = {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        self.gripper_state = "unknown"
        self.arm_status = "idle"
    
    def _load_command_mappings(self) -> Dict[str, Any]:
        """加载ROS2命令映射配置"""
        try:
            config_path = Path(self.config.system.config_dir) / "ros2_commands.json"
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"加载ROS2命令映射失败: {str(e)}")
            return {}
    
    async def start(self):
        """启动ROS2 Agent"""
        self.is_running = True
        self.logger.info("ROS2 Agent已启动")
        
        # 检查ROS2环境
        await self._check_ros2_environment()
        
        # 启动Gazebo仿真（如果配置启用）
        if self.config.ros2.auto_start_gazebo:
            await self._start_gazebo_simulation()
        
        # 启动消息处理循环
        asyncio.create_task(self._message_processing_loop())
    
    async def stop(self):
        """停止ROS2 Agent"""
        self.is_running = False
        
        # 停止Gazebo仿真
        if self.gazebo_running:
            await self._stop_gazebo_simulation()
        
        self.logger.info("ROS2 Agent已停止")
    
    async def _check_ros2_environment(self):
        """检查ROS2环境"""
        try:
            # 检查ROS2是否可用
            result = subprocess.run(
                ["ros2", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                self.ros2_available = True
                self.logger.info(f"ROS2环境检查通过: {result.stdout.strip()}")
            else:
                self.ros2_available = False
                self.logger.warning("ROS2环境不可用")
                
        except Exception as e:
            self.ros2_available = False
            self.logger.error(f"ROS2环境检查失败: {str(e)}")
    
    async def _start_gazebo_simulation(self):
        """启动Gazebo仿真"""
        try:
            if not self.ros2_available:
                self.logger.warning("ROS2不可用，无法启动Gazebo仿真")
                return
            
            self.logger.info("正在启动Gazebo仿真...")
            
            # 启动Gazebo世界
            gazebo_cmd = [
                "ros2", "launch", 
                self.config.ros2.robot_package,
                self.config.ros2.gazebo_launch_file
            ]
            
            # 异步启动Gazebo
            process = await asyncio.create_subprocess_exec(
                *gazebo_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # 等待一段时间让Gazebo启动
            await asyncio.sleep(5)
            
            # 检查进程是否还在运行
            if process.returncode is None:
                self.gazebo_running = True
                self.logger.info("Gazebo仿真启动成功")
            else:
                self.gazebo_running = False
                self.logger.error("Gazebo仿真启动失败")
                
        except Exception as e:
            self.gazebo_running = False
            self.logger.error(f"启动Gazebo仿真失败: {str(e)}")
    
    async def _stop_gazebo_simulation(self):
        """停止Gazebo仿真"""
        try:
            # 杀死Gazebo进程
            subprocess.run(["pkill", "-f", "gazebo"], timeout=5)
            self.gazebo_running = False
            self.logger.info("Gazebo仿真已停止")
            
        except Exception as e:
            self.logger.error(f"停止Gazebo仿真失败: {str(e)}")
    
    async def _message_processing_loop(self):
        """消息处理循环"""
        while self.is_running:
            try:
                # 从队列接收消息
                message = await self.message_queue.receive_from_queue(
                    self.message_queue.ros2_queue,
                    timeout=5
                )
                
                if message:
                    await self._process_message(message)
                    
            except Exception as e:
                self.logger.error(f"消息处理循环出错: {str(e)}")
                await asyncio.sleep(1)
    
    async def _process_message(self, message: QueueMessage):
        """处理接收到的消息"""
        try:
            self.logger.info(
                f"处理ROS2消息",
                extra={
                    "message_id": message.message_id,
                    "message_type": message.message_type.value
                }
            )
            
            if message.message_type == MessageType.PARSED_COMMAND:
                await self._execute_parsed_command(message)
            elif message.message_type == MessageType.ROS2_COMMAND:
                await self._execute_ros2_command(message)
            else:
                self.logger.warning(f"未知的消息类型: {message.message_type}")
            
            self.processed_count += 1
            
        except Exception as e:
            self.logger.error(
                f"处理消息失败: {str(e)}",
                extra={
                    "message_id": message.message_id,
                    "error": str(e)
                }
            )
    
    async def _execute_parsed_command(self, message: QueueMessage):
        """执行解析后的命令"""
        try:
            data = message.data
            action = data.get("action", "")
            parameters = data.get("parameters", {})
            
            # 查找对应的ROS2命令
            ros2_command = self._map_to_ros2_command(action, parameters)
            
            if ros2_command:
                # 执行ROS2命令
                result = await self._execute_ros2_command_direct(ros2_command)
                
                # 发送执行结果到记忆Agent
                await self._send_execution_result(message, ros2_command, result)
            else:
                self.logger.warning(f"未找到对应的ROS2命令: {action}")
                
        except Exception as e:
            self.logger.error(f"执行解析命令失败: {str(e)}")
    
    async def _execute_ros2_command(self, message: QueueMessage):
        """执行ROS2命令消息"""
        try:
            data = message.data
            command_type = data.get("command_type", "")
            topic = data.get("topic", "")
            parameters = data.get("parameters", {})
            
            # 构建ROS2命令
            ros2_command = {
                "command_type": command_type,
                "topic": topic,
                "parameters": parameters
            }
            
            # 执行命令
            result = await self._execute_ros2_command_direct(ros2_command)
            
            # 发送执行结果
            await self._send_execution_result(message, ros2_command, result)
            
        except Exception as e:
            self.logger.error(f"执行ROS2命令失败: {str(e)}")
    
    def _map_to_ros2_command(self, action: str, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """将动作映射为ROS2命令"""
        try:
            # 查找命令映射
            for command_name, command_config in self.command_mappings.items():
                if command_config.get("action") == action:
                    # 构建ROS2命令
                    ros2_command = {
                        "command_type": command_config.get("command_type", "topic_pub"),
                        "topic": command_config.get("topic", ""),
                        "message_type": command_config.get("message_type", ""),
                        "parameters": self._map_parameters(command_config, parameters)
                    }
                    
                    return ros2_command
            
            return None
            
        except Exception as e:
            self.logger.error(f"命令映射失败: {str(e)}")
            return None
    
    def _map_parameters(self, command_config: Dict[str, Any], input_params: Dict[str, Any]) -> Dict[str, Any]:
        """映射参数"""
        try:
            mapped_params = {}
            param_mapping = command_config.get("parameter_mapping", {})
            default_values = command_config.get("default_values", {})
            
            # 应用参数映射
            for ros2_param, input_param in param_mapping.items():
                if input_param in input_params:
                    mapped_params[ros2_param] = input_params[input_param]
                elif ros2_param in default_values:
                    mapped_params[ros2_param] = default_values[ros2_param]
            
            # 添加默认值
            for param, value in default_values.items():
                if param not in mapped_params:
                    mapped_params[param] = value
            
            return mapped_params
            
        except Exception as e:
            self.logger.error(f"参数映射失败: {str(e)}")
            return {}
    
    async def _execute_ros2_command_direct(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """直接执行ROS2命令"""
        start_time = datetime.now()
        
        try:
            if not self.ros2_available:
                return {
                    "success": False,
                    "error": "ROS2环境不可用",
                    "execution_time": 0
                }
            
            command_type = command.get("command_type", "topic_pub")
            topic = command.get("topic", "")
            parameters = command.get("parameters", {})
            
            if command_type == "topic_pub":
                # 发布话题消息
                result = await self._publish_topic(topic, parameters)
            elif command_type == "service_call":
                # 调用服务
                result = await self._call_service(topic, parameters)
            elif command_type == "action_call":
                # 调用动作
                result = await self._call_action(topic, parameters)
            else:
                result = {
                    "success": False,
                    "error": f"不支持的命令类型: {command_type}"
                }
            
            execution_time = (datetime.now() - start_time).total_seconds()
            result["execution_time"] = execution_time
            
            # 更新机械臂状态
            await self._update_arm_status(command, result)
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            }
    
    async def _publish_topic(self, topic: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """发布ROS2话题"""
        try:
            # 构建ros2 topic pub命令
            message_data = json.dumps(parameters)
            
            cmd = [
                "ros2", "topic", "pub", "--once",
                topic,
                "geometry_msgs/msg/Twist",  # 默认消息类型，实际应根据配置确定
                message_data
            ]
            
            # 执行命令
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.logger.info(f"成功发布话题 {topic}")
                return {
                    "success": True,
                    "output": stdout.decode(),
                    "topic": topic,
                    "parameters": parameters
                }
            else:
                self.logger.error(f"发布话题失败: {stderr.decode()}")
                return {
                    "success": False,
                    "error": stderr.decode(),
                    "topic": topic
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "topic": topic
            }
    
    async def _call_service(self, service: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """调用ROS2服务"""
        try:
            # 构建ros2 service call命令
            message_data = json.dumps(parameters)
            
            cmd = [
                "ros2", "service", "call",
                service,
                "std_srvs/srv/Empty",  # 默认服务类型
                message_data
            ]
            
            # 执行命令
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.logger.info(f"成功调用服务 {service}")
                return {
                    "success": True,
                    "output": stdout.decode(),
                    "service": service,
                    "parameters": parameters
                }
            else:
                self.logger.error(f"调用服务失败: {stderr.decode()}")
                return {
                    "success": False,
                    "error": stderr.decode(),
                    "service": service
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "service": service
            }
    
    async def _call_action(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """调用ROS2动作"""
        try:
            # 构建ros2 action send_goal命令
            message_data = json.dumps(parameters)
            
            cmd = [
                "ros2", "action", "send_goal",
                action,
                "control_msgs/action/FollowJointTrajectory",  # 默认动作类型
                message_data
            ]
            
            # 执行命令
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.logger.info(f"成功调用动作 {action}")
                return {
                    "success": True,
                    "output": stdout.decode(),
                    "action": action,
                    "parameters": parameters
                }
            else:
                self.logger.error(f"调用动作失败: {stderr.decode()}")
                return {
                    "success": False,
                    "error": stderr.decode(),
                    "action": action
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action": action
            }
    
    async def _update_arm_status(self, command: Dict[str, Any], result: Dict[str, Any]):
        """更新机械臂状态"""
        try:
            if result.get("success"):
                parameters = command.get("parameters", {})
                
                # 更新位置信息
                if "position" in parameters:
                    pos = parameters["position"]
                    self.current_pose.update({
                        "x": pos.get("x", self.current_pose["x"]),
                        "y": pos.get("y", self.current_pose["y"]),
                        "z": pos.get("z", self.current_pose["z"])
                    })
                
                # 更新姿态信息
                if "orientation" in parameters:
                    ori = parameters["orientation"]
                    self.current_pose.update({
                        "roll": ori.get("roll", self.current_pose["roll"]),
                        "pitch": ori.get("pitch", self.current_pose["pitch"]),
                        "yaw": ori.get("yaw", self.current_pose["yaw"])
                    })
                
                # 更新夹爪状态
                if "gripper_action" in parameters:
                    self.gripper_state = parameters["gripper_action"]
                
                self.arm_status = "executing"
            else:
                self.arm_status = "error"
                
        except Exception as e:
            self.logger.error(f"更新机械臂状态失败: {str(e)}")
    
    async def _send_execution_result(self, original_message: QueueMessage, command: Dict[str, Any], result: Dict[str, Any]):
        """发送执行结果到记忆Agent"""
        try:
            # 构建记忆记录消息
            memory_data = {
                "message_id": original_message.message_id,
                "session_id": original_message.data.get("session_id", "default"),
                "user_id": original_message.data.get("user_id", "default"),
                "user_input": original_message.data.get("input_text", ""),
                "intent": original_message.data.get("intent", ""),
                "action": original_message.data.get("action", ""),
                "parameters": original_message.data.get("parameters", {}),
                "priority": original_message.priority.value,
                "ros2_command_type": command.get("command_type", ""),
                "ros2_topic": command.get("topic", ""),
                "execution_status": "success" if result.get("success") else "failed",
                "execution_time": result.get("execution_time", 0),
                "error_message": result.get("error", "")
            }
            
            # 发送到记忆队列
            memory_message = QueueMessage(
                message_type=MessageType.MEMORY_RECORD,
                data=memory_data,
                priority=original_message.priority
            )
            
            await self.message_queue.send_to_memory_agent(memory_message)
            
        except Exception as e:
            self.logger.error(f"发送执行结果失败: {str(e)}")
    
    async def get_current_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            "is_running": self.is_running,
            "ros2_available": self.ros2_available,
            "gazebo_running": self.gazebo_running,
            "current_pose": self.current_pose,
            "gripper_state": self.gripper_state,
            "arm_status": self.arm_status,
            "processed_count": self.processed_count
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            status = await self.get_current_status()
            
            # 检查ROS2节点
            ros2_nodes_ok = False
            if self.ros2_available:
                try:
                    result = subprocess.run(
                        ["ros2", "node", "list"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    ros2_nodes_ok = result.returncode == 0
                except:
                    ros2_nodes_ok = False
            
            overall_status = "healthy" if (
                self.is_running and 
                self.ros2_available and 
                ros2_nodes_ok
            ) else "unhealthy"
            
            return {
                "status": overall_status,
                "details": status,
                "ros2_nodes_available": ros2_nodes_ok
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "is_running": self.is_running
            }