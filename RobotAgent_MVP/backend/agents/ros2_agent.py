import json
import subprocess
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from utils.config import Config
from utils.logger import CustomLogger

class ROS2Agent:
    """ROS2代理 - 处理机器人控制指令"""
    
    def __init__(self):
        self.config = Config()
        self.logger = CustomLogger("ROS2Agent")
        self.is_running = False
        self.command_mappings = {}
        self.execution_history = []
        self.current_status = "idle"
        
        # 状态反馈队列
        self.status_queue = asyncio.Queue()
        
        self.logger.info("ROS2Agent初始化完成")
    
    def _load_command_mappings(self):
        """加载命令映射配置"""
        try:
            # 使用相对路径加载配置文件
            config_path = Path(__file__).parent.parent.parent / "config" / "ros2_commands.json"
            
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.command_mappings = json.load(f)
                self.logger.info(f"成功加载命令映射: {len(self.command_mappings)} 个命令")
            else:
                self.logger.warning(f"命令映射文件不存在: {config_path}")
                self.command_mappings = {}
        except Exception as e:
            self.logger.error(f"加载命令映射失败: {e}")
            self.command_mappings = {}
    
    def _check_ros2_environment(self):
        """检查ROS2环境"""
        try:
            result = subprocess.run(['ros2', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.logger.info(f"ROS2环境检查成功: {result.stdout.strip()}")
            else:
                self.logger.warning("ROS2环境检查失败")
        except Exception as e:
            self.logger.warning(f"ROS2环境检查失败: {e}")
    
    async def start(self):
        """启动ROS2代理"""
        if self.is_running:
            self.logger.warning("ROS2Agent已经在运行")
            return
        
        self.logger.info("启动ROS2Agent...")
        
        # 加载命令映射
        self._load_command_mappings()
        
        # 检查ROS2环境
        self._check_ros2_environment()
        
        # 启动Gazebo仿真（如果配置了）
        if self.config.ros2.get("auto_start_gazebo", False):
            await self._start_gazebo_simulation()
        
        # 启动状态监控
        asyncio.create_task(self._status_monitor())
        
        self.is_running = True
        self.current_status = "ready"
        self.logger.info("ROS2Agent启动成功")
    
    async def stop(self):
        """停止ROS2代理"""
        if not self.is_running:
            return
        
        self.logger.info("停止ROS2Agent...")
        
        # 停止Gazebo进程
        if hasattr(self, 'gazebo_process') and self.gazebo_process:
            try:
                self.gazebo_process.terminate()
                await self.gazebo_process.wait()
                self.logger.info("Gazebo仿真已停止")
            except Exception as e:
                self.logger.error(f"停止Gazebo仿真失败: {e}")
        
        self.is_running = False
        self.logger.info("ROS2Agent已停止")
    
    async def _start_gazebo_simulation(self):
        """启动Gazebo仿真"""
        try:
            robot_package = self.config.ros2["robot_package"]
            launch_file = self.config.ros2["gazebo_launch_file"]
            
            self.logger.info(f"启动Gazebo仿真: {robot_package} {launch_file}")
            
            # 启动Gazebo
            cmd = f"ros2 launch {robot_package} {launch_file}"
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            self.gazebo_process = process
            self.logger.info("Gazebo仿真启动成功")
            
        except Exception as e:
            self.logger.error(f"启动Gazebo仿真失败: {e}")
    
    async def process_robot_response(self, robot_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理机器人聊天响应
        
        Args:
            robot_response: 包含user_reply和ros2_command的响应
            
        Returns:
            处理结果，包含执行状态
        """
        try:
            # 验证响应格式
            if not self._validate_robot_response(robot_response):
                return {
                    "status": "error",
                    "message": "无效的机器人响应格式",
                    "user_reply": robot_response.get("user_reply", "响应格式错误")
                }
            
            user_reply = robot_response["user_reply"]
            ros2_command = robot_response["ros2_command"]
            
            # 如果没有ROS2指令，直接返回用户回复
            if ros2_command is None:
                self.current_status = "standby"
                return {
                    "status": "standby",
                    "message": "无需机器人动作，仅对话",
                    "user_reply": user_reply
                }
            
            # 验证ROS2指令格式
            if not self._validate_ros2_command(ros2_command):
                return {
                    "status": "error",
                    "message": "ROS2指令格式验证失败",
                    "user_reply": user_reply,
                    "details": "指令格式不符合ROS2通信标准"
                }
            
            # 执行ROS2指令
            execution_result = await self._execute_ros2_command(ros2_command)
            
            # 记录执行历史
            self.execution_history.append({
                "timestamp": time.time(),
                "command": ros2_command,
                "result": execution_result,
                "user_reply": user_reply
            })
            
            # 更新状态
            self.current_status = execution_result["status"]
            
            # 将状态放入队列供状态反馈使用
            await self.status_queue.put({
                "execution_result": execution_result,
                "user_reply": user_reply,
                "timestamp": time.time()
            })
            
            return {
                "status": execution_result["status"],
                "message": execution_result["message"],
                "user_reply": user_reply,
                "execution_details": execution_result.get("details", {})
            }
            
        except Exception as e:
            self.logger.error(f"处理机器人响应时发生错误: {e}")
            return {
                "status": "error",
                "message": f"处理失败: {str(e)}",
                "user_reply": robot_response.get("user_reply", "处理失败")
            }
    
    def _validate_robot_response(self, response: Dict[str, Any]) -> bool:
        """验证机器人响应格式"""
        try:
            if not isinstance(response, dict):
                return False
            
            if "type" not in response or response["type"] != "robot_response":
                return False
            
            if "user_reply" not in response or not isinstance(response["user_reply"], str):
                return False
            
            if "ros2_command" not in response:
                return False
            
            return True
        except Exception:
            return False
    
    def _validate_ros2_command(self, command: Dict[str, Any]) -> bool:
        """验证ROS2指令格式"""
        try:
            if not isinstance(command, dict):
                return False
            
            required_fields = ["command_type", "topic", "message_type", "data"]
            if not all(field in command for field in required_fields):
                return False
            
            # 验证指令类型
            valid_types = ["movement", "manipulation", "gripper", "sensor", "status"]
            if command["command_type"] not in valid_types:
                return False
            
            # 验证话题名称格式
            topic = command["topic"]
            if not isinstance(topic, str) or not topic.startswith("/"):
                return False
            
            # 验证消息类型格式
            message_type = command["message_type"]
            if not isinstance(message_type, str) or "/" not in message_type:
                return False
            
            # 验证数据字段
            if not isinstance(command["data"], dict):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"ROS2指令验证失败: {e}")
            return False
    
    async def _execute_ros2_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """执行ROS2指令"""
        try:
            command_type = command["command_type"]
            topic = command["topic"]
            message_type = command["message_type"]
            data = command["data"]
            
            self.logger.info(f"执行ROS2指令: {command_type} -> {topic}")
            
            # 根据指令类型执行相应操作
            if command_type == "movement":
                return await self._execute_movement_command(topic, message_type, data)
            elif command_type == "manipulation":
                return await self._execute_manipulation_command(topic, message_type, data)
            elif command_type == "gripper":
                return await self._execute_gripper_command(topic, message_type, data)
            elif command_type == "sensor":
                return await self._execute_sensor_command(topic, message_type, data)
            elif command_type == "status":
                return await self._execute_status_command(topic, message_type, data)
            else:
                return {
                    "status": "error",
                    "message": f"不支持的指令类型: {command_type}"
                }
                
        except Exception as e:
            self.logger.error(f"执行ROS2指令时发生错误: {e}")
            return {
                "status": "error",
                "message": f"指令执行失败: {str(e)}"
            }
    
    async def _execute_movement_command(self, topic: str, message_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行移动指令"""
        try:
            # 模拟ROS2发布指令
            self.logger.info(f"发布移动指令到话题 {topic}")
            
            # 这里应该是实际的ROS2发布代码
            # 由于当前环境没有ROS2，我们模拟执行
            await asyncio.sleep(0.1)  # 模拟执行时间
            
            return {
                "status": "success",
                "message": f"移动指令已发送到 {topic}",
                "details": {
                    "topic": topic,
                    "message_type": message_type,
                    "data": data
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"移动指令执行失败: {str(e)}"
            }
    
    async def _execute_manipulation_command(self, topic: str, message_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行机械臂操作指令"""
        try:
            self.logger.info(f"发布机械臂指令到话题 {topic}")
            await asyncio.sleep(0.1)
            
            return {
                "status": "success",
                "message": f"机械臂指令已发送到 {topic}",
                "details": {
                    "topic": topic,
                    "message_type": message_type,
                    "data": data
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"机械臂指令执行失败: {str(e)}"
            }
    
    async def _execute_gripper_command(self, topic: str, message_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行夹爪控制指令"""
        try:
            self.logger.info(f"发布夹爪指令到话题 {topic}")
            await asyncio.sleep(0.1)
            
            return {
                "status": "success",
                "message": f"夹爪指令已发送到 {topic}",
                "details": {
                    "topic": topic,
                    "message_type": message_type,
                    "data": data
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"夹爪指令执行失败: {str(e)}"
            }
    
    async def _execute_sensor_command(self, topic: str, message_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行传感器查询指令"""
        try:
            self.logger.info(f"查询传感器数据从话题 {topic}")
            await asyncio.sleep(0.1)
            
            # 模拟传感器数据
            sensor_data = {
                "timestamp": time.time(),
                "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            }
            
            return {
                "status": "success",
                "message": f"传感器数据已获取从 {topic}",
                "details": {
                    "topic": topic,
                    "message_type": message_type,
                    "sensor_data": sensor_data
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"传感器查询失败: {str(e)}"
            }
    
    async def _execute_status_command(self, topic: str, message_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行状态查询指令"""
        try:
            self.logger.info(f"查询机器人状态从话题 {topic}")
            await asyncio.sleep(0.1)
            
            # 模拟机器人状态
            robot_status = {
                "timestamp": time.time(),
                "position": {"x": 1.0, "y": 2.0, "z": 0.0},
                "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": 1.57},
                "battery": 85.5,
                "connection": "connected"
            }
            
            return {
                "status": "success",
                "message": f"机器人状态已获取从 {topic}",
                "details": {
                    "topic": topic,
                    "message_type": message_type,
                    "robot_status": robot_status
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"状态查询失败: {str(e)}"
            }
    
    def get_status(self) -> Dict[str, Any]:
        """获取Agent状态"""
        return {
            "is_running": self.is_running,
            "current_status": self.current_status,
            "execution_history_count": len(self.execution_history),
            "command_mappings_count": len(self.command_mappings)
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        return {
            "status": "healthy" if self.is_running else "unhealthy",
            "current_status": self.current_status,
            "last_execution": self.execution_history[-1] if self.execution_history else None
        }
    

    
    async def _handle_move_control(self, action: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """处理移动控制指令"""
        try:
            ros2_commands = []
            
            if action == "move_to_position":
                # 移动到指定位置
                command = {
                    "command_type": "action_call",
                    "topic": "/rm_driver/move_to_pose",
                    "action_type": "rm_ros_interfaces/action/MoveTopose",
                    "parameters": {
                        "pose": {
                            "position": {
                                "x": parameters.get("x", 0.0),
                                "y": parameters.get("y", 0.0),
                                "z": parameters.get("z", 0.0)
                            },
                            "orientation": {
                                "x": parameters.get("orientation_x", 0.0),
                                "y": parameters.get("orientation_y", 0.0),
                                "z": parameters.get("orientation_z", 0.0),
                                "w": parameters.get("orientation_w", 1.0)
                            }
                        },
                        "speed": parameters.get("speed", 0.1)
                    }
                }
                ros2_commands.append(command)
                
            elif action == "move_relative":
                # 相对移动
                current_x = self.current_pose["x"]
                current_y = self.current_pose["y"]
                current_z = self.current_pose["z"]
                
                command = {
                    "command_type": "action_call",
                    "topic": "/rm_driver/move_to_pose",
                    "action_type": "rm_ros_interfaces/action/MoveTopose",
                    "parameters": {
                        "pose": {
                            "position": {
                                "x": current_x + parameters.get("dx", 0.0),
                                "y": current_y + parameters.get("dy", 0.0),
                                "z": current_z + parameters.get("dz", 0.0)
                            },
                            "orientation": {
                                "x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0
                            }
                        },
                        "speed": parameters.get("speed", 0.1)
                    }
                }
                ros2_commands.append(command)
                
            elif action == "home_position":
                # 回到初始位置
                command = {
                    "command_type": "action_call",
                    "topic": "/rm_driver/move_to_pose",
                    "action_type": "rm_ros_interfaces/action/MoveTopose",
                    "parameters": {
                        "pose": {
                            "position": {"x": 0.0, "y": 0.0, "z": 0.5},
                            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                        },
                        "speed": 0.1
                    }
                }
                ros2_commands.append(command)
            
            return ros2_commands
            
        except Exception as e:
            self.logger.error(f"处理移动控制指令失败: {str(e)}")
            return []
    
    async def _handle_gripper_control(self, action: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """处理夹爪控制指令"""
        try:
            ros2_commands = []
            
            if action in ["open_gripper", "close_gripper"]:
                gripper_state = "open" if action == "open_gripper" else "close"
                
                command = {
                    "command_type": "service_call",
                    "topic": "/rm_driver/set_gripper",
                    "service_type": "rm_ros_interfaces/srv/SetGripper",
                    "parameters": {
                        "gripper_state": gripper_state,
                        "force": parameters.get("force", 50)
                    }
                }
                ros2_commands.append(command)
                
                # 更新夹爪状态
                self.gripper_state = gripper_state
            
            return ros2_commands
            
        except Exception as e:
            self.logger.error(f"处理夹爪控制指令失败: {str(e)}")
            return []
    
    async def _handle_joint_control(self, action: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """处理关节控制指令"""
        try:
            ros2_commands = []
            
            if action == "move_joints":
                joint_positions = parameters.get("joint_positions", [])
                
                if len(joint_positions) == 6:  # 6轴机械臂
                    command = {
                        "command_type": "action_call",
                        "topic": "/rm_driver/move_joints",
                        "action_type": "rm_ros_interfaces/action/MoveJoints",
                        "parameters": {
                            "joint_positions": joint_positions,
                            "speed": parameters.get("speed", 0.1)
                        }
                    }
                    ros2_commands.append(command)
                else:
                    self.logger.error(f"关节位置数量不正确: {len(joint_positions)}, 期望6个")
            
            return ros2_commands
            
        except Exception as e:
            self.logger.error(f"处理关节控制指令失败: {str(e)}")
            return []
    
    async def _handle_complex_action(self, action: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """处理复合动作指令"""
        try:
            ros2_commands = []
            
            if action == "pick_object":
                # 抓取物体的复合动作序列
                target_pos = parameters.get("target_position", {})
                
                # 1. 移动到物体上方
                ros2_commands.extend(await self._handle_move_control("move_to_position", {
                    "x": target_pos.get("x", 0.0),
                    "y": target_pos.get("y", 0.0),
                    "z": target_pos.get("z", 0.0) + 0.1  # 上方10cm
                }))
                
                # 2. 打开夹爪
                ros2_commands.extend(await self._handle_gripper_control("open_gripper", {}))
                
                # 3. 下降到物体位置
                ros2_commands.extend(await self._handle_move_control("move_to_position", target_pos))
                
                # 4. 夹取物体
                ros2_commands.extend(await self._handle_gripper_control("close_gripper", {}))
                
                # 5. 提升物体
                ros2_commands.extend(await self._handle_move_control("move_to_position", {
                    "x": target_pos.get("x", 0.0),
                    "y": target_pos.get("y", 0.0),
                    "z": target_pos.get("z", 0.0) + 0.1
                }))
                
            elif action == "place_object":
                # 放置物体的复合动作序列
                target_pos = parameters.get("target_position", {})
                
                # 1. 移动到目标位置上方
                ros2_commands.extend(await self._handle_move_control("move_to_position", {
                    "x": target_pos.get("x", 0.0),
                    "y": target_pos.get("y", 0.0),
                    "z": target_pos.get("z", 0.0) + 0.1
                }))
                
                # 2. 下降到放置位置
                ros2_commands.extend(await self._handle_move_control("move_to_position", target_pos))
                
                # 3. 释放物体
                ros2_commands.extend(await self._handle_gripper_control("open_gripper", {}))
                
                # 4. 提升夹爪
                ros2_commands.extend(await self._handle_move_control("move_to_position", {
                    "x": target_pos.get("x", 0.0),
                    "y": target_pos.get("y", 0.0),
                    "z": target_pos.get("z", 0.0) + 0.1
                }))
            
            return ros2_commands
            
        except Exception as e:
            self.logger.error(f"处理复合动作指令失败: {str(e)}")
            return []
    
    async def _handle_status_query(self, action: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """处理状态查询指令"""
        try:
            ros2_commands = []
            
            if action == "get_current_pose":
                command = {
                    "command_type": "service_call",
                    "topic": "/rm_driver/get_current_pose",
                    "service_type": "rm_ros_interfaces/srv/GetCurrentPose",
                    "parameters": {}
                }
                ros2_commands.append(command)
            
            return ros2_commands
            
        except Exception as e:
            self.logger.error(f"处理状态查询指令失败: {str(e)}")
            return []
    
    async def _handle_system_control(self, action: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """处理系统控制指令"""
        try:
            ros2_commands = []
            
            if action == "stop_motion":
                command = {
                    "command_type": "service_call",
                    "topic": "/rm_driver/stop_motion",
                    "service_type": "std_srvs/srv/Empty",
                    "parameters": {}
                }
                ros2_commands.append(command)
            
            return ros2_commands
            
        except Exception as e:
            self.logger.error(f"处理系统控制指令失败: {str(e)}")
            return []
    
    async def _handle_generic_action(self, action: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """处理通用动作指令（基于配置文件映射）"""
        try:
            ros2_commands = []
            
            # 查找命令映射
            if action in self.command_mappings:
                command_config = self.command_mappings[action]
                
                # 构建ROS2命令
                command = {
                    "command_type": "topic_pub",  # 默认类型
                    "topic": "",
                    "parameters": {}
                }
                
                # 解析ROS2命令字符串
                ros2_cmd_str = command_config.get("ros2_command", "")
                if "action send_goal" in ros2_cmd_str:
                    command["command_type"] = "action_call"
                elif "service call" in ros2_cmd_str:
                    command["command_type"] = "service_call"
                
                # 提取topic
                parts = ros2_cmd_str.split()
                if len(parts) > 3:
                    command["topic"] = parts[3]
                
                # 映射参数
                param_mapping = command_config.get("parameters_mapping", {})
                mapped_params = {}
                for ros2_param, input_param in param_mapping.items():
                    if input_param in parameters:
                        mapped_params[ros2_param] = parameters[input_param]
                
                # 添加默认值
                default_values = command_config.get("default_values", {})
                for param, value in default_values.items():
                    if param not in mapped_params:
                        mapped_params[param] = value
                
                command["parameters"] = mapped_params
                ros2_commands.append(command)
            
            return ros2_commands
            
        except Exception as e:
            self.logger.error(f"处理通用动作指令失败: {str(e)}")
            return []
    
    async def _publish_to_ros2(self, ros2_command: Dict[str, Any]) -> Dict[str, Any]:
        """发布命令到ROS2系统执行物理控制"""
        start_time = datetime.now()
        
        try:
            if not self.ros2_available:
                return {
                    "success": False,
                    "error": "ROS2环境不可用",
                    "execution_time": 0
                }
            
            command_type = ros2_command.get("command_type", "topic_pub")
            topic = ros2_command.get("topic", "")
            parameters = ros2_command.get("parameters", {})
            
            self.logger.info(f"发布ROS2命令到 {topic} - 类型: {command_type}")
            
            if command_type == "topic_pub":
                result = await self._publish_topic(topic, parameters)
            elif command_type == "service_call":
                result = await self._call_service(topic, parameters)
            elif command_type == "action_call":
                result = await self._call_action(topic, parameters)
            else:
                result = {
                    "success": False,
                    "error": f"不支持的命令类型: {command_type}"
                }
            
            execution_time = (datetime.now() - start_time).total_seconds()
            result["execution_time"] = execution_time
            
            # 更新机械臂状态
            if result.get("success", False):
                await self._update_arm_status(ros2_command, result)
            
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
                "geometry_msgs/msg/Twist",  # 默认消息类型
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
                "rm_ros_interfaces/action/MoveTopose",  # 默认动作类型
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
            # 根据执行的命令更新状态
            if "pose" in command.get("parameters", {}):
                pose = command["parameters"]["pose"]
                if "position" in pose:
                    self.current_pose.update({
                        "x": pose["position"].get("x", self.current_pose["x"]),
                        "y": pose["position"].get("y", self.current_pose["y"]),
                        "z": pose["position"].get("z", self.current_pose["z"])
                    })
            
            # 更新状态时间戳
            self.arm_status = "moving" if result.get("success", False) else "error"
            
        except Exception as e:
            self.logger.error(f"更新机械臂状态失败: {str(e)}")
    

    
    async def _status_monitor(self):
        """状态监控任务"""
        while self.is_running:
            try:
                # 定期检查系统状态
                await asyncio.sleep(5)
                
                # 这里可以添加实际的ROS2状态监控逻辑
                self.logger.debug(f"当前状态: {self.current_status}")
                
            except Exception as e:
                self.logger.error(f"状态监控错误: {e}")
    
    async def get_status_feedback(self) -> Optional[Dict[str, Any]]:
        """获取状态反馈"""
        try:
            # 非阻塞获取状态
            status_info = await asyncio.wait_for(self.status_queue.get(), timeout=0.1)
            return status_info
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            self.logger.error(f"获取状态反馈失败: {e}")
            return None