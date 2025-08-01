import json
import os
import time
import asyncio
from typing import Dict, Any, Optional, List
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role
import dashscope

from utils.config import Config
from utils.logger import CustomLogger, PerformanceLogger

class QwenService:
    """Qwen模型服务 - 使用DashScope SDK"""
    
    def __init__(self):
        self.config = Config()
        self.logger = CustomLogger("QwenService")
        self.perf_logger = PerformanceLogger()
        
        # 设置API Key
        api_key = self.config.qwen["api_key"]
        if api_key.startswith("${") and api_key.endswith("}"):
            # 从环境变量读取
            env_var = api_key[2:-1]
            api_key = os.getenv(env_var)
            if not api_key:
                self.logger.warning(f"环境变量 {env_var} 未设置，将使用测试模式")
                self.test_mode = True
                api_key = "test_key"
            else:
                self.test_mode = False
        else:
            self.test_mode = False
        
        dashscope.api_key = api_key
        
        # 统计信息
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_response_time = 0.0
        
        # 机器人聊天系统提示词
        self.robot_chat_prompt = """你是一个智能机器人助手，能够理解用户的自然语言指令并控制机器人执行相应动作。

请严格按照以下JSON格式返回响应：

{
    "type": "robot_response",
    "user_reply": "给用户的自然语言回复",
    "ros2_command": {
        "command_type": "指令类型",
        "topic": "ROS2话题名称",
        "message_type": "消息类型",
        "data": {
            "具体的控制参数"
        }
    }
}

指令类型说明：
- "movement": 机器人移动控制
- "manipulation": 机械臂操作
- "gripper": 夹爪控制
- "sensor": 传感器查询
- "status": 状态查询
- "null": 无需机器人动作，仅对话

当用户只是聊天而不需要机器人执行动作时，将ros2_command设置为null。

ROS2话题和消息类型示例：
- 移动控制: topic="/cmd_vel", message_type="geometry_msgs/Twist"
- 机械臂控制: topic="/arm_controller/joint_trajectory", message_type="trajectory_msgs/JointTrajectory"
- 夹爪控制: topic="/gripper_controller/command", message_type="control_msgs/GripperCommand"
- 状态查询: topic="/robot_state", message_type="sensor_msgs/JointState"

请确保返回的JSON格式完全正确，不要添加任何额外的文本或解释。"""
        
        self.logger.info("QwenService初始化完成")
    
    async def robot_chat(self, user_input: str, conversation_history: List[Dict] = None, execution_status: str = None) -> Optional[Dict[str, Any]]:
        """
        机器人聊天接口 - 返回包含用户回复和ROS2指令的JSON
        
        Args:
            user_input: 用户输入的自然语言
            conversation_history: 对话历史
            execution_status: 上次执行状态（用于状态反馈）
            
        Returns:
            包含user_reply和ros2_command的字典
        """
        # 测试模式：返回模拟响应
        if self.test_mode:
            self.logger.info(f"测试模式：处理用户输入 - {user_input}")
            return self._create_test_response(user_input)
        
        with self.perf_logger.timer("robot_chat"):
            self.total_requests += 1
            start_time = time.time()
            
            try:
                # 构建消息
                messages = [
                    {
                        'role': Role.SYSTEM,
                        'content': self.robot_chat_prompt
                    }
                ]
                
                # 添加对话历史
                if conversation_history:
                    messages.extend(conversation_history)
                
                # 添加执行状态反馈
                if execution_status:
                    status_message = f"上次指令执行状态：{execution_status}\n\n用户输入：{user_input}"
                else:
                    status_message = user_input
                
                messages.append({
                    'role': Role.USER,
                    'content': status_message
                })
                
                # 调用DashScope API
                response = await asyncio.to_thread(
                    Generation.call,
                    model=self.config.qwen["model_name"],
                    messages=messages,
                    max_tokens=self.config.qwen["max_tokens"],
                    temperature=self.config.qwen["temperature"],
                    top_p=self.config.qwen.get('top_p', 0.8),
                    result_format='message'
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                self.total_response_time += response_time
                
                if response.status_code == 200:
                    self.successful_requests += 1
                    
                    # 解析响应
                    content = response.output.choices[0].message.content
                    
                    # 尝试解析JSON
                    try:
                        # 提取JSON部分
                        if "```json" in content:
                            json_start = content.find("```json") + 7
                            json_end = content.find("```", json_start)
                            json_content = content[json_start:json_end].strip()
                        elif "```" in content:
                            json_start = content.find("```") + 3
                            json_end = content.find("```", json_start)
                            json_content = content[json_start:json_end].strip()
                        else:
                            json_content = content.strip()
                        
                        result = json.loads(json_content)
                        
                        # 验证JSON格式
                        if self._validate_robot_response(result):
                            self.logger.info(f"成功生成机器人响应: {result.get('type', 'unknown')}")
                            return result
                        else:
                            self.logger.warning(f"响应格式验证失败，重新请求")
                            return await self._retry_with_format_correction(user_input, content)
                            
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON解析失败: {e}, 原始内容: {content}")
                        return await self._retry_with_format_correction(user_input, content)
                        
                else:
                    self.failed_requests += 1
                    self.logger.error(f"API调用失败: {response.status_code}, {response.message}")
                    return self._create_fallback_chat_response(user_input)
                    
            except Exception as e:
                self.failed_requests += 1
                end_time = time.time()
                self.total_response_time += (end_time - start_time)
                self.logger.error(f"调用Qwen API时发生错误: {e}")
                return self._create_fallback_chat_response(user_input)
    
    async def _retry_with_format_correction(self, user_input: str, previous_response: str) -> Dict[str, Any]:
        """重新请求并纠正格式"""
        try:
            correction_prompt = f"""之前的响应格式不正确：
{previous_response}

请严格按照以下JSON格式重新生成响应，不要添加任何其他文本：

{{
    "type": "robot_response",
    "user_reply": "给用户的自然语言回复",
    "ros2_command": {{
        "command_type": "指令类型或null",
        "topic": "ROS2话题名称或null",
        "message_type": "消息类型或null",
        "data": {{}}
    }}
}}

用户原始输入：{user_input}"""

            messages = [
                {
                    'role': Role.SYSTEM,
                    'content': self.robot_chat_prompt
                },
                {
                    'role': Role.USER,
                    'content': correction_prompt
                }
            ]
            
            response = await asyncio.to_thread(
                Generation.call,
                model=self.config.qwen["model_name"],
                messages=messages,
                max_tokens=self.config.qwen["max_tokens"],
                temperature=0.1,  # 降低温度以获得更稳定的格式
                result_format='message'
            )
            
            if response.status_code == 200:
                content = response.output.choices[0].message.content
                
                # 提取JSON
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    json_content = content[json_start:json_end].strip()
                else:
                    json_content = content.strip()
                
                result = json.loads(json_content)
                
                if self._validate_robot_response(result):
                    return result
                    
        except Exception as e:
            self.logger.error(f"格式纠正重试失败: {e}")
        
        # 如果重试仍然失败，返回备用响应
        return self._create_fallback_chat_response(user_input)
    
    def _validate_robot_response(self, response: Dict[str, Any]) -> bool:
        """验证机器人响应格式"""
        try:
            # 检查必要字段
            if not isinstance(response, dict):
                return False
            
            if "type" not in response or response["type"] != "robot_response":
                return False
            
            if "user_reply" not in response or not isinstance(response["user_reply"], str):
                return False
            
            # ros2_command可以是null或字典
            if "ros2_command" not in response:
                return False
            
            ros2_cmd = response["ros2_command"]
            if ros2_cmd is not None:
                if not isinstance(ros2_cmd, dict):
                    return False
                
                required_fields = ["command_type", "topic", "message_type", "data"]
                if not all(field in ros2_cmd for field in required_fields):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _create_fallback_chat_response(self, user_input: str) -> Dict[str, Any]:
        """创建备用聊天响应"""
        return {
            "type": "robot_response",
            "user_reply": f"抱歉，我无法理解您的指令：{user_input}。请尝试重新表达。",
            "ros2_command": None
        }
    
    async def convert_status_to_natural_language(self, execution_status: Dict[str, Any]) -> str:
        """将执行状态转换为自然语言"""
        try:
            status_prompt = f"""请将以下机器人执行状态转换为自然语言回复给用户：

执行状态：{json.dumps(execution_status, ensure_ascii=False, indent=2)}

请用简洁、友好的语言告诉用户机器人的执行情况。只返回自然语言回复，不要包含JSON或其他格式。"""

            messages = [
                {
                    'role': Role.SYSTEM,
                    'content': "你是一个机器人助手，需要将技术状态信息转换为用户友好的自然语言。"
                },
                {
                    'role': Role.USER,
                    'content': status_prompt
                }
            ]
            
            response = await asyncio.to_thread(
                Generation.call,
                model=self.config.qwen["model_name"],
                messages=messages,
                max_tokens=200,
                temperature=0.3,
                result_format='message'
            )
            
            if response.status_code == 200:
                return response.output.choices[0].message.content.strip()
            else:
                return f"执行状态：{execution_status.get('status', '未知')}"
                
        except Exception as e:
            self.logger.error(f"状态转换失败: {e}")
            return f"执行状态：{execution_status.get('status', '未知')}"
        
    async def parse_natural_language(self, user_input: str, user_id: str = "default") -> Optional[Dict[str, Any]]:
        """
        解析自然语言输入为结构化命令
        
        Args:
            user_input: 用户输入的自然语言
            user_id: 用户ID
            
        Returns:
            解析结果字典或None（如果失败）
        """
        with self.perf_logger.timer("qwen_api_call"):
            self.total_requests += 1
            start_time = time.time()
            
            try:
                # 构建消息
                messages = [
                    {
                        'role': Role.SYSTEM,
                        'content': self.config.qwen["system_prompt"]
                    },
                    {
                        'role': Role.USER,
                        'content': f"请解析以下指令：{user_input}"
                    }
                ]
                
                # 调用DashScope API
                response = await asyncio.to_thread(
                    Generation.call,
                    model=self.config.qwen["model_name"],
                    messages=messages,
                    max_tokens=self.config.qwen["max_tokens"],
                    temperature=self.config.qwen["temperature"],
                    top_p=self.config.qwen.get('top_p', 0.8),
                    result_format='message'
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                self.total_response_time += response_time
                
                if response.status_code == 200:
                    self.successful_requests += 1
                    
                    # 解析响应
                    content = response.output.choices[0].message.content
                    
                    # 尝试解析JSON
                    try:
                        # 提取JSON部分（可能包含在代码块中）
                        if "```json" in content:
                            json_start = content.find("```json") + 7
                            json_end = content.find("```", json_start)
                            json_content = content[json_start:json_end].strip()
                        elif "```" in content:
                            json_start = content.find("```") + 3
                            json_end = content.find("```", json_start)
                            json_content = content[json_start:json_end].strip()
                        else:
                            json_content = content.strip()
                        
                        result = json.loads(json_content)
                        
                        # 验证必要字段
                        required_fields = ['intent', 'action', 'parameters', 'confidence']
                        if all(field in result for field in required_fields):
                            self.logger.info(f"成功解析用户输入: {user_input} -> {result['intent']}")
                            return result
                        else:
                            self.logger.warning(f"解析结果缺少必要字段: {result}")
                            return self._create_fallback_response(user_input)
                            
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON解析失败: {e}, 原始内容: {content}")
                        return self._create_fallback_response(user_input)
                        
                else:
                    self.failed_requests += 1
                    self.logger.error(f"API调用失败: {response.status_code}, {response.message}")
                    return None
                    
            except Exception as e:
                self.failed_requests += 1
                end_time = time.time()
                self.total_response_time += (end_time - start_time)
                self.logger.error(f"调用Qwen API时发生错误: {e}")
                return None
    
    def _create_fallback_response(self, user_input: str) -> Dict[str, Any]:
        """创建备用响应"""
        return {
            "intent": "unknown",
            "action": "unknown", 
            "parameters": {"original_input": user_input},
            "confidence": 0.1
        }
    
    async def simple_chat(self, user_input: str, conversation_history: list = None) -> Optional[str]:
        """
        简单对话功能
        
        Args:
            user_input: 用户输入
            conversation_history: 对话历史
            
        Returns:
            模型回复或None
        """
        with self.perf_logger.timer("qwen_chat"):
            try:
                # 构建消息历史
                messages = []
                
                if conversation_history:
                    messages.extend(conversation_history)
                
                messages.append({
                    'role': Role.USER,
                    'content': user_input
                })
                
                # 调用API
                response = await asyncio.to_thread(
                    Generation.call,
                    model=self.config.qwen["model_name"],
                    messages=messages,
                    max_tokens=self.config.qwen["max_tokens"],
                    temperature=self.config.qwen["temperature"],
                    result_format='message'
                )
                
                if response.status_code == 200:
                    return response.output.choices[0].message.content
                else:
                    self.logger.error(f"对话API调用失败: {response.status_code}")
                    return None
                    
            except Exception as e:
                self.logger.error(f"对话时发生错误: {e}")
                return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        avg_response_time = (
            self.total_response_time / self.total_requests 
            if self.total_requests > 0 else 0
        )
        
        success_rate = (
            self.successful_requests / self.total_requests 
            if self.total_requests > 0 else 0
        )
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate,
            "average_response_time": avg_response_time
        }
    
    def _create_test_response(self, user_input: str) -> Dict[str, Any]:
        """创建测试模式响应"""
        # 简单的关键词匹配来模拟智能响应
        user_input_lower = user_input.lower()
        
        if any(word in user_input_lower for word in ["移动", "前进", "后退", "左转", "右转", "move", "forward", "backward"]):
            return {
                "type": "robot_response",
                "user_reply": f"好的，我将执行移动指令：{user_input}",
                "ros2_command": {
                    "command_type": "movement",
                    "topic": "/cmd_vel",
                    "message_type": "geometry_msgs/Twist",
                    "data": {
                        "linear": {"x": 0.5, "y": 0.0, "z": 0.0},
                        "angular": {"x": 0.0, "y": 0.0, "z": 0.0}
                    }
                }
            }
        elif any(word in user_input_lower for word in ["抓取", "夹爪", "gripper", "grab", "pick"]):
            return {
                "type": "robot_response", 
                "user_reply": f"好的，我将控制夹爪：{user_input}",
                "ros2_command": {
                    "command_type": "gripper",
                    "topic": "/gripper_controller/command",
                    "message_type": "control_msgs/GripperCommand",
                    "data": {
                        "position": 0.5,
                        "max_effort": 10.0
                    }
                }
            }
        elif any(word in user_input_lower for word in ["机械臂", "arm", "关节", "joint"]):
            return {
                "type": "robot_response",
                "user_reply": f"好的，我将控制机械臂：{user_input}",
                "ros2_command": {
                    "command_type": "manipulation",
                    "topic": "/arm_controller/joint_trajectory",
                    "message_type": "trajectory_msgs/JointTrajectory",
                    "data": {
                        "joint_names": ["joint1", "joint2", "joint3"],
                        "points": [{"positions": [0.0, 0.5, 1.0], "time_from_start": {"sec": 2}}]
                    }
                }
            }
        elif any(word in user_input_lower for word in ["状态", "status", "位置", "position"]):
            return {
                "type": "robot_response",
                "user_reply": f"我将查询机器人状态：{user_input}",
                "ros2_command": {
                    "command_type": "status",
                    "topic": "/robot_state",
                    "message_type": "sensor_msgs/JointState",
                    "data": {}
                }
            }
        else:
            # 纯聊天，无需机器人动作
            return {
                "type": "robot_response",
                "user_reply": f"我理解了您的话：{user_input}。这是一个测试模式的回复。",
                "ros2_command": None
            }
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            # 发送简单的测试请求
            test_response = await self.simple_chat("你好")
            return test_response is not None
        except Exception as e:
            self.logger.error(f"健康检查失败: {e}")
            return False