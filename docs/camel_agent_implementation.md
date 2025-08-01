# CAMEL智能体实现指南

## 1. 概述

本文档详细说明了RobotAgent项目中各个CAMEL智能体的具体实现方法，包括基础架构、消息处理、任务执行和协作机制。

## 2. 基础智能体架构

### 2.1 BaseRobotAgent类设计

```python
"""
基础智能体抽象类 - 所有机器人智能体的基类
位置：src/camel_agents/base_agent.py
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import BaseModelBackend
from camel.prompts import TextPrompt
import asyncio
import logging
from datetime import datetime

class AgentState(Enum):
    """智能体状态枚举"""
    IDLE = "idle"
    PROCESSING = "processing"
    EXECUTING = "executing"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class BaseRobotAgent(ABC):
    """机器人智能体基类"""
    
    def __init__(self, 
                 name: str,
                 model_backend: BaseModelBackend,
                 system_message: str,
                 message_bus: 'MessageBus',
                 config: Dict[str, Any] = None):
        """
        初始化基础智能体
        
        Args:
            name: 智能体名称
            model_backend: 模型后端
            system_message: 系统提示消息
            message_bus: 消息总线
            config: 配置参数
        """
        self.name = name
        self.config = config or {}
        self.message_bus = message_bus
        self.logger = logging.getLogger(f"RobotAgent.{name}")
        
        # 初始化CAMEL智能体
        self.agent = ChatAgent(
            system_message=TextPrompt(system_message),
            model=model_backend
        )
        
        # 智能体状态
        self.state = AgentState.IDLE
        self.capabilities = []
        self.task_queue = asyncio.Queue()
        self.current_task = None
        self.performance_metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_response_time": 0.0,
            "last_activity": None
        }
        
        # LangGraph记忆系统集成
        self.memory_system = None
        self.memory_workflow_state = None
        self.agent_memory_processor = None
        
        # 消息处理器映射
        self.message_handlers = {
            "task_request": self._handle_task_request,
            "status_request": self._handle_status_request,
            "shutdown": self._handle_shutdown,
            "ping": self._handle_ping
        }
        
        # 注册消息订阅
        self.message_bus.subscribe(self.name, self._on_message_received)
        
        # 初始化记忆系统
        asyncio.create_task(self._initialize_memory_system())
        
    async def _initialize_memory_system(self):
        """初始化LangGraph记忆系统"""
        try:
            from memory_system.langgraph_engine import MemoryWorkflowEngine
            from memory_system.processors.agent_memory_processor import AgentMemoryProcessor
            
            # 初始化记忆工作流引擎
            memory_config = self.config.get("memory", {})
            self.memory_system = MemoryWorkflowEngine(memory_config)
            
            # 初始化智能体记忆处理器
            self.agent_memory_processor = AgentMemoryProcessor(
                agent_name=self.name,
                memory_system=self.memory_system
            )
            
            # 创建记忆工作流状态
            self.memory_workflow_state = await self.memory_system.create_initial_state({
                "agent_id": self.name,
                "agent_type": self.__class__.__name__,
                "capabilities": self.capabilities
            })
            
            self.logger.info(f"Memory system initialized for agent: {self.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory system: {e}")
            # 记忆系统初始化失败不应该阻止智能体启动
            self.memory_system = None
        
    @abstractmethod
    async def process_message(self, message: BaseMessage) -> BaseMessage:
        """处理接收到的消息 - 子类必须实现"""
        pass
        
    @abstractmethod
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行分配的任务 - 子类必须实现"""
        pass
        
    async def _on_message_received(self, message: 'Message'):
        """消息接收回调"""
        try:
            self.logger.debug(f"Received message: {message.message_type} from {message.sender}")
            
            # 更新活动时间
            self.performance_metrics["last_activity"] = datetime.now()
            
            # 路由到对应的处理器
            handler = self.message_handlers.get(
                message.message_type.value, 
                self._handle_custom_message
            )
            
            await handler(message)
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            await self._send_error_response(message, str(e))
            
    async def _handle_task_request(self, message: 'Message'):
        """处理任务请求"""
        task = message.content
        task["id"] = message.id
        task["sender"] = message.sender
        
        # 添加到任务队列
        await self.task_queue.put(task)
        
        # 发送确认
        await self.send_message(
            recipient=message.sender,
            content={"status": "accepted", "task_id": task["id"]},
            message_type="task_accepted"
        )
        
    async def _handle_status_request(self, message: 'Message'):
        """处理状态请求"""
        status = self.get_status()
        await self.send_message(
            recipient=message.sender,
            content=status,
            message_type="status_response"
        )
        
    async def _handle_shutdown(self, message: 'Message'):
        """处理关闭请求"""
        self.logger.info("Received shutdown request")
        self.state = AgentState.SHUTDOWN
        
    async def _handle_ping(self, message: 'Message'):
        """处理ping请求"""
        await self.send_message(
            recipient=message.sender,
            content={"status": "alive", "timestamp": datetime.now().isoformat()},
            message_type="pong"
        )
        
    async def _handle_custom_message(self, message: 'Message'):
        """处理自定义消息 - 委托给子类实现"""
        camel_message = BaseMessage.make_user_message(
            role_name="user",
            content=message.content
        )
        
        response = await self.process_message(camel_message)
        
        if response:
            await self.send_message(
                recipient=message.sender,
                content=response.content,
                message_type="response"
            )
            
    async def send_message(self, 
                          recipient: str,
                          content: Any,
                          message_type: str = "text",
                          priority: int = 1,
                          metadata: Dict[str, Any] = None) -> str:
        """发送消息到其他智能体"""
        return await self.message_bus.send_message(
            sender=self.name,
            recipient=recipient,
            content=content,
            message_type=message_type,
            priority=priority,
            metadata=metadata
        )
        
    async def broadcast_message(self,
                              content: Any,
                              message_type: str = "broadcast",
                              priority: int = 1,
                              metadata: Dict[str, Any] = None) -> str:
        """广播消息"""
        return await self.message_bus.broadcast_message(
            sender=self.name,
            content=content,
            message_type=message_type,
            priority=priority,
            metadata=metadata
        )
        
    def get_status(self) -> Dict[str, Any]:
        """获取智能体状态"""
        return {
            "name": self.name,
            "state": self.state.value,
            "capabilities": self.capabilities,
            "current_task": self.current_task,
            "queue_size": self.task_queue.qsize(),
            "performance_metrics": self.performance_metrics
        }
        
    async def start(self):
        """启动智能体"""
        self.logger.info(f"Starting agent: {self.name}")
        self.state = AgentState.IDLE
        
        # 启动任务处理循环
        asyncio.create_task(self._task_processing_loop())
        
    async def stop(self):
        """停止智能体"""
        self.logger.info(f"Stopping agent: {self.name}")
        self.state = AgentState.SHUTDOWN
        
    async def _task_processing_loop(self):
        """任务处理循环"""
        while self.state != AgentState.SHUTDOWN:
            try:
                # 等待任务
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # 执行任务
                self.state = AgentState.EXECUTING
                self.current_task = task
                
                start_time = datetime.now()
                result = await self.execute_task(task)
                end_time = datetime.now()
                
                # 更新性能指标
                execution_time = (end_time - start_time).total_seconds()
                self._update_performance_metrics(execution_time, success=True)
                
                # 发送结果
                await self.send_message(
                    recipient=task.get("sender", "system"),
                    content=result,
                    message_type="task_result"
                )
                
                self.current_task = None
                self.state = AgentState.IDLE
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Task execution failed: {e}")
                self._update_performance_metrics(0, success=False)
                
                if self.current_task:
                    await self.send_message(
                        recipient=self.current_task.get("sender", "system"),
                        content={"error": str(e), "task_id": self.current_task.get("id")},
                        message_type="task_error"
                    )
                    
                self.current_task = None
                self.state = AgentState.IDLE
                
    def _update_performance_metrics(self, execution_time: float, success: bool):
        """更新性能指标"""
        if success:
            self.performance_metrics["tasks_completed"] += 1
        else:
            self.performance_metrics["tasks_failed"] += 1
            
        # 更新平均响应时间
        total_tasks = (self.performance_metrics["tasks_completed"] + 
                      self.performance_metrics["tasks_failed"])
        if total_tasks > 0:
            current_avg = self.performance_metrics["average_response_time"]
            self.performance_metrics["average_response_time"] = (
                (current_avg * (total_tasks - 1) + execution_time) / total_tasks
            )
            
    # ==================== LangGraph记忆系统方法 ====================
    
    async def store_agent_memory(self, memory_data: Dict[str, Any]) -> Optional[str]:
        """存储智能体记忆"""
        if not self.memory_system or not self.agent_memory_processor:
            self.logger.warning("Memory system not initialized, skipping memory storage")
            return None
            
        try:
            # 添加智能体上下文信息
            memory_data.update({
                "agent_id": self.name,
                "agent_type": self.__class__.__name__,
                "timestamp": datetime.now().isoformat(),
                "memory_category": "agent_memory"
            })
            
            # 通过记忆处理器存储
            memory_id = await self.agent_memory_processor.process_memory(
                memory_data, self.memory_workflow_state
            )
            
            self.logger.debug(f"Stored agent memory: {memory_id}")
            return memory_id
            
        except Exception as e:
            self.logger.error(f"Failed to store agent memory: {e}")
            return None
            
    async def retrieve_agent_memory(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检索智能体记忆"""
        if not self.memory_system or not self.agent_memory_processor:
            self.logger.warning("Memory system not initialized, returning empty results")
            return []
            
        try:
            # 添加智能体过滤条件
            query.update({
                "agent_id": self.name,
                "memory_category": "agent_memory"
            })
            
            # 通过记忆处理器检索
            memories = await self.agent_memory_processor.retrieve_memory(
                query, self.memory_workflow_state
            )
            
            self.logger.debug(f"Retrieved {len(memories)} agent memories")
            return memories
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve agent memory: {e}")
            return []
            
    async def update_memory_workflow_state(self, state_updates: Dict[str, Any]):
        """更新记忆工作流状态"""
        if not self.memory_system or not self.memory_workflow_state:
            return
            
        try:
            # 更新工作流状态
            self.memory_workflow_state.update(state_updates)
            
            # 保存检查点
            await self.memory_system.save_checkpoint(
                self.memory_workflow_state, 
                checkpoint_id=f"{self.name}_checkpoint"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update memory workflow state: {e}")
            
    async def get_memory_context(self, context_type: str = "recent") -> Dict[str, Any]:
        """获取记忆上下文"""
        if not self.memory_system:
            return {}
            
        try:
            if context_type == "recent":
                # 获取最近的记忆
                query = {
                    "limit": 10,
                    "sort_by": "timestamp",
                    "order": "desc"
                }
            elif context_type == "relevant":
                # 获取相关记忆（基于当前任务）
                query = {
                    "task_related": True,
                    "current_task": self.current_task,
                    "limit": 5
                }
            else:
                query = {"limit": 5}
                
            memories = await self.retrieve_agent_memory(query)
            
            return {
                "context_type": context_type,
                "memory_count": len(memories),
                "memories": memories,
                "agent_id": self.name
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get memory context: {e}")
            return {}
```

## 3. 具体智能体实现

### 3.1 对话智能体 (DialogAgent)

```python
"""
对话智能体实现
位置：src/camel_agents/dialog_agent.py
"""

from typing import Dict, Any, List, Optional
import json
import re
from camel.messages import BaseMessage
from .base_agent import BaseRobotAgent

class DialogAgent(BaseRobotAgent):
    """对话智能体 - 负责自然语言交互和意图理解"""
    
    def __init__(self, model_backend, message_bus, config: Dict[str, Any] = None):
        system_message = """
你是一个智能机器人的对话智能体。你的职责包括：

1. **自然语言理解**：
   - 理解用户的自然语言指令和问题
   - 识别用户的意图和情感状态
   - 处理多轮对话的上下文

2. **意图分类**：
   - robot_command: 机器人动作指令（移动、抓取、操作等）
   - question: 信息查询问题
   - greeting: 问候和社交对话
   - help: 帮助请求
   - emergency: 紧急情况

3. **响应生成**：
   - 生成友好、自然的回复
   - 确认用户指令的理解
   - 提供必要的解释和反馈

4. **任务协调**：
   - 将用户意图转换为结构化任务
   - 协调其他智能体完成用户请求
   - 跟踪任务执行状态并向用户报告

请始终保持友好、专业的语调，确保用户体验良好。
"""
        super().__init__("DialogAgent", model_backend, system_message, message_bus, config)
        self.capabilities = [
            "natural_language_understanding",
            "intent_recognition", 
            "response_generation",
            "conversation_management"
        ]
        
        # 对话状态
        self.conversation_history = []
        self.user_context = {}
        self.active_tasks = {}
        
        # 意图识别模式
        self.intent_patterns = {
            "robot_command": [
                r"(移动|去|到|走|前进|后退|左转|右转)",
                r"(抓取|拿|取|放|放下|举起)",
                r"(打开|关闭|启动|停止)",
                r"(看|观察|检查|扫描)"
            ],
            "question": [
                r"(什么|哪里|怎么|为什么|何时)",
                r"(状态|位置|信息|数据)",
                r"(能否|可以|是否)"
            ],
            "greeting": [
                r"(你好|早上好|下午好|晚上好)",
                r"(谢谢|感谢|再见|拜拜)"
            ]
        }
        
    async def process_message(self, message: BaseMessage) -> BaseMessage:
        """处理消息"""
        content = message.content
        
        if isinstance(content, dict):
            # 处理结构化消息
            if content.get("type") == "user_input":
                return await self._handle_user_input(content["text"])
            elif content.get("type") == "task_update":
                return await self._handle_task_update(content)
        else:
            # 处理文本消息
            return await self._handle_user_input(str(content))
            
    async def _handle_user_input(self, user_text: str) -> BaseMessage:
        """处理用户输入"""
        # 记录对话历史
        self.conversation_history.append({
            "role": "user",
            "content": user_text,
            "timestamp": datetime.now().isoformat()
        })
        
        # 意图识别
        intent = await self._recognize_intent(user_text)
        
        # 生成响应
        response_text = await self._generate_response(user_text, intent)
        
        # 记录响应
        self.conversation_history.append({
            "role": "assistant", 
            "content": response_text,
            "timestamp": datetime.now().isoformat()
        })
        
        # 如果是机器人指令，创建任务
        if intent["type"] == "robot_command":
            await self._create_robot_task(intent, user_text)
            
        return BaseMessage.make_assistant_message(
            role_name="assistant",
            content=response_text
        )
        
    async def _recognize_intent(self, user_text: str) -> Dict[str, Any]:
        """识别用户意图"""
        # 使用模式匹配进行初步分类
        intent_type = "question"  # 默认类型
        confidence = 0.5
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_text, re.IGNORECASE):
                    intent_type = intent
                    confidence = 0.8
                    break
                    
        # 使用LLM进行详细意图分析
        intent_prompt = f"""
分析以下用户输入的详细意图：

用户输入："{user_text}"
对话历史：{json.dumps(self.conversation_history[-5:], ensure_ascii=False, indent=2)}

请返回JSON格式的意图分析：
{{
    "type": "意图类型（robot_command/question/greeting/help/emergency）",
    "action": "具体动作（如果是机器人指令）",
    "parameters": {{
        "target": "目标对象",
        "location": "位置信息",
        "quantity": "数量",
        "other": "其他参数"
    }},
    "priority": "优先级（high/medium/low）",
    "confidence": "置信度（0-1）",
    "context": "上下文信息"
}}
"""
        
        intent_message = BaseMessage.make_user_message(
            role_name="user",
            content=intent_prompt
        )
        
        response = await self.agent.step(intent_message)
        
        try:
            detailed_intent = json.loads(response.content)
            # 合并基础分类和详细分析
            detailed_intent["type"] = intent_type
            detailed_intent["confidence"] = max(confidence, detailed_intent.get("confidence", 0))
            return detailed_intent
        except json.JSONDecodeError:
            # 如果解析失败，返回基础分类
            return {
                "type": intent_type,
                "confidence": confidence,
                "action": None,
                "parameters": {},
                "priority": "medium"
            }
            
    async def _generate_response(self, user_text: str, intent: Dict[str, Any]) -> str:
        """生成响应"""
        response_prompt = f"""
基于用户输入和意图分析，生成合适的回复：

用户输入："{user_text}"
意图分析：{json.dumps(intent, ensure_ascii=False, indent=2)}
对话历史：{json.dumps(self.conversation_history[-3:], ensure_ascii=False, indent=2)}

回复要求：
1. 友好、自然的语调
2. 确认对用户意图的理解
3. 如果是机器人指令，说明将要执行的动作
4. 如果是问题，提供有用的信息或说明如何获取答案
5. 保持简洁但信息完整

请直接返回回复内容，不要包含其他格式。
"""
        
        response_message = BaseMessage.make_user_message(
            role_name="user",
            content=response_prompt
        )
        
        response = await self.agent.step(response_message)
        return response.content.strip()
        
    async def _create_robot_task(self, intent: Dict[str, Any], original_text: str):
        """创建机器人任务"""
        task = {
            "type": "robot_command",
            "original_text": original_text,
            "intent": intent,
            "priority": intent.get("priority", "medium"),
            "created_by": "DialogAgent",
            "timestamp": datetime.now().isoformat()
        }
        
        # 发送给规划智能体
        task_id = await self.send_message(
            recipient="PlanningAgent",
            content=task,
            message_type="task_request",
            priority=2 if intent.get("priority") == "high" else 1
        )
        
        # 记录活跃任务
        self.active_tasks[task_id] = {
            "task": task,
            "status": "planning",
            "created_at": datetime.now()
        }
        
    async def _handle_task_update(self, update: Dict[str, Any]) -> BaseMessage:
        """处理任务状态更新"""
        task_id = update.get("task_id")
        status = update.get("status")
        
        if task_id in self.active_tasks:
            self.active_tasks[task_id]["status"] = status
            
            # 生成状态更新响应
            if status == "completed":
                response = "任务已完成！"
            elif status == "failed":
                error = update.get("error", "未知错误")
                response = f"任务执行失败：{error}"
            elif status == "executing":
                response = "正在执行任务..."
            else:
                response = f"任务状态更新：{status}"
                
            # 可以选择是否向用户报告状态
            # 这里简化为记录日志
            self.logger.info(f"Task {task_id} status: {status}")
            
        return BaseMessage.make_assistant_message(
            role_name="assistant",
            content="Task update processed"
        )
        
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行对话相关任务"""
        task_type = task.get("type")
        
        if task_type == "generate_response":
            # 生成响应任务
            user_input = task.get("user_input")
            intent = await self._recognize_intent(user_input)
            response = await self._generate_response(user_input, intent)
            
            return {
                "success": True,
                "response": response,
                "intent": intent
            }
            
        elif task_type == "analyze_conversation":
            # 分析对话任务
            analysis = await self._analyze_conversation()
            return {
                "success": True,
                "analysis": analysis
            }
            
        else:
            return {
                "success": False,
                "error": f"Unknown task type: {task_type}"
            }
            
    async def _analyze_conversation(self) -> Dict[str, Any]:
        """分析对话模式和用户偏好"""
        if len(self.conversation_history) < 2:
            return {"message": "Insufficient conversation data"}
            
        analysis_prompt = f"""
分析以下对话历史，提取用户偏好和对话模式：

对话历史：
{json.dumps(self.conversation_history, ensure_ascii=False, indent=2)}

请分析：
1. 用户的常用指令类型
2. 用户的语言风格偏好
3. 对话中的重复模式
4. 用户可能的需求趋势

返回JSON格式的分析结果。
"""
        
        analysis_message = BaseMessage.make_user_message(
            role_name="user",
            content=analysis_prompt
        )
        
        response = await self.agent.step(analysis_message)
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"analysis": response.content}
```

### 3.2 规划智能体 (PlanningAgent)

```python
"""
规划智能体实现
位置：src/camel_agents/planning_agent.py
"""

from typing import Dict, Any, List, Optional
import json
from datetime import datetime, timedelta
from .base_agent import BaseRobotAgent

class PlanningAgent(BaseRobotAgent):
    """规划智能体 - 负责任务分解和执行计划制定"""
    
    def __init__(self, model_backend, message_bus, config: Dict[str, Any] = None):
        system_message = """
你是一个智能机器人的规划智能体。你的职责包括：

1. **任务分解**：
   - 将复杂任务分解为可执行的子任务
   - 识别任务间的依赖关系
   - 确定执行顺序和并行可能性

2. **路径规划**：
   - 制定机器人移动路径
   - 考虑障碍物和安全约束
   - 优化执行效率

3. **资源分配**：
   - 分析任务所需资源
   - 协调多个执行模块
   - 处理资源冲突

4. **风险评估**：
   - 识别潜在风险和失败点
   - 制定应急预案
   - 设置安全检查点

5. **计划优化**：
   - 根据执行反馈调整计划
   - 处理异常情况的重新规划
   - 学习和改进规划策略

请确保所有计划都是安全、可行和高效的。
"""
        super().__init__("PlanningAgent", model_backend, system_message, message_bus, config)
        self.capabilities = [
            "task_decomposition",
            "path_planning", 
            "resource_allocation",
            "risk_assessment",
            "plan_optimization"
        ]
        
        # 规划状态
        self.active_plans = {}
        self.plan_templates = {}
        self.execution_history = []
        
        # 加载规划模板
        self._load_plan_templates()
        
    def _load_plan_templates(self):
        """加载常用任务的规划模板"""
        self.plan_templates = {
            "move_to_location": {
                "subtasks": [
                    {"id": "path_planning", "type": "navigation", "description": "规划移动路径"},
                    {"id": "obstacle_check", "type": "perception", "description": "检查路径障碍"},
                    {"id": "execute_movement", "type": "motion", "description": "执行移动"},
                    {"id": "verify_arrival", "type": "verification", "description": "验证到达目标"}
                ],
                "dependencies": {
                    "obstacle_check": ["path_planning"],
                    "execute_movement": ["obstacle_check"],
                    "verify_arrival": ["execute_movement"]
                }
            },
            "pick_and_place": {
                "subtasks": [
                    {"id": "locate_object", "type": "perception", "description": "定位目标物体"},
                    {"id": "approach_object", "type": "navigation", "description": "接近物体"},
                    {"id": "grasp_object", "type": "manipulation", "description": "抓取物体"},
                    {"id": "move_to_target", "type": "navigation", "description": "移动到目标位置"},
                    {"id": "place_object", "type": "manipulation", "description": "放置物体"},
                    {"id": "verify_placement", "type": "verification", "description": "验证放置结果"}
                ],
                "dependencies": {
                    "approach_object": ["locate_object"],
                    "grasp_object": ["approach_object"],
                    "move_to_target": ["grasp_object"],
                    "place_object": ["move_to_target"],
                    "verify_placement": ["place_object"]
                }
            }
        }
        
    async def process_message(self, message: BaseMessage) -> BaseMessage:
        """处理消息"""
        # 委托给基类的消息处理
        return await super().process_message(message)
        
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行规划任务"""
        task_type = task.get("type")
        
        if task_type == "robot_command":
            return await self._create_execution_plan(task)
        elif task_type == "replan":
            return await self._replan_task(task)
        elif task_type == "optimize_plan":
            return await self._optimize_plan(task)
        else:
            return {
                "success": False,
                "error": f"Unknown task type: {task_type}"
            }
            
    async def _create_execution_plan(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """创建执行计划"""
        intent = task.get("intent", {})
        action = intent.get("action")
        parameters = intent.get("parameters", {})
        
        # 使用LLM分析任务并生成计划
        planning_prompt = f"""
为以下机器人任务创建详细的执行计划：

任务描述：{task.get("original_text", "")}
意图分析：{json.dumps(intent, ensure_ascii=False, indent=2)}

请考虑以下方面：
1. 任务分解：将任务分解为具体的子任务
2. 执行顺序：确定子任务的执行顺序和依赖关系
3. 资源需求：每个子任务需要的能力和资源
4. 安全约束：执行过程中的安全要求和检查点
5. 错误处理：可能的失败情况和应对策略
6. 时间估算：每个子任务的预计执行时间

返回JSON格式的执行计划：
{{
    "plan_id": "唯一计划ID",
    "task_type": "任务类型",
    "priority": "优先级",
    "estimated_duration": "预计总时间（秒）",
    "subtasks": [
        {{
            "id": "子任务ID",
            "type": "子任务类型",
            "description": "子任务描述",
            "executor": "执行者（哪个智能体或模块）",
            "parameters": {{}},
            "estimated_time": "预计时间（秒）",
            "dependencies": ["依赖的子任务ID"],
            "safety_checks": ["安全检查项"],
            "failure_handling": "失败处理策略"
        }}
    ],
    "safety_constraints": [
        "全局安全约束"
    ],
    "success_criteria": [
        "成功标准"
    ],
    "rollback_plan": "回滚计划"
}}
"""
        
        plan_message = BaseMessage.make_user_message(
            role_name="user",
            content=planning_prompt
        )
        
        response = await self.agent.step(plan_message)
        
        try:
            execution_plan = json.loads(response.content)
            
            # 生成唯一计划ID
            plan_id = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.active_plans)}"
            execution_plan["plan_id"] = plan_id
            execution_plan["created_at"] = datetime.now().isoformat()
            execution_plan["status"] = "created"
            execution_plan["original_task"] = task
            
            # 保存计划
            self.active_plans[plan_id] = execution_plan
            
            # 发送计划给决策智能体
            await self.send_message(
                recipient="DecisionAgent",
                content=execution_plan,
                message_type="execution_plan",
                priority=2 if intent.get("priority") == "high" else 1
            )
            
            return {
                "success": True,
                "plan_id": plan_id,
                "execution_plan": execution_plan
            }
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse execution plan: {e}")
            return {
                "success": False,
                "error": f"Failed to create execution plan: {str(e)}"
            }
            
    async def _replan_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """重新规划任务"""
        plan_id = task.get("plan_id")
        failure_info = task.get("failure_info", {})
        
        if plan_id not in self.active_plans:
            return {
                "success": False,
                "error": f"Plan {plan_id} not found"
            }
            
        original_plan = self.active_plans[plan_id]
        
        replan_prompt = f"""
原始计划执行失败，需要重新规划：

原始计划：
{json.dumps(original_plan, ensure_ascii=False, indent=2)}

失败信息：
{json.dumps(failure_info, ensure_ascii=False, indent=2)}

请分析失败原因并创建新的执行计划：
1. 识别失败的根本原因
2. 调整计划以避免相同问题
3. 考虑替代方案和备用策略
4. 增强错误处理和安全措施

返回JSON格式的新执行计划。
"""
        
        replan_message = BaseMessage.make_user_message(
            role_name="user",
            content=replan_prompt
        )
        
        response = await self.agent.step(replan_message)
        
        try:
            new_plan = json.loads(response.content)
            
            # 更新计划
            new_plan_id = f"{plan_id}_replan_{datetime.now().strftime('%H%M%S')}"
            new_plan["plan_id"] = new_plan_id
            new_plan["created_at"] = datetime.now().isoformat()
            new_plan["status"] = "replanned"
            new_plan["original_plan_id"] = plan_id
            new_plan["replan_reason"] = failure_info
            
            self.active_plans[new_plan_id] = new_plan
            
            # 标记原计划为失败
            original_plan["status"] = "failed"
            original_plan["failure_reason"] = failure_info
            
            # 发送新计划
            await self.send_message(
                recipient="DecisionAgent",
                content=new_plan,
                message_type="execution_plan",
                priority=3  # 重新规划的任务优先级较高
            )
            
            return {
                "success": True,
                "new_plan_id": new_plan_id,
                "execution_plan": new_plan
            }
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse replan: {e}")
            return {
                "success": False,
                "error": f"Failed to create replan: {str(e)}"
            }
            
    async def _optimize_plan(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """优化执行计划"""
        plan_id = task.get("plan_id")
        optimization_criteria = task.get("criteria", ["time", "safety", "efficiency"])
        
        if plan_id not in self.active_plans:
            return {
                "success": False,
                "error": f"Plan {plan_id} not found"
            }
            
        current_plan = self.active_plans[plan_id]
        
        optimize_prompt = f"""
优化以下执行计划：

当前计划：
{json.dumps(current_plan, ensure_ascii=False, indent=2)}

优化目标：{optimization_criteria}

请从以下方面优化计划：
1. 时间效率：减少总执行时间
2. 安全性：增强安全措施和检查
3. 资源利用：优化资源分配和使用
4. 并行性：识别可并行执行的子任务
5. 鲁棒性：提高计划的容错能力

返回优化后的JSON格式执行计划。
"""
        
        optimize_message = BaseMessage.make_user_message(
            role_name="user",
            content=optimize_prompt
        )
        
        response = await self.agent.step(optimize_message)
        
        try:
            optimized_plan = json.loads(response.content)
            
            # 更新计划
            optimized_plan["plan_id"] = plan_id
            optimized_plan["optimized_at"] = datetime.now().isoformat()
            optimized_plan["optimization_criteria"] = optimization_criteria
            optimized_plan["status"] = "optimized"
            
            self.active_plans[plan_id] = optimized_plan
            
            return {
                "success": True,
                "optimized_plan": optimized_plan
            }
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse optimized plan: {e}")
            return {
                "success": False,
                "error": f"Failed to optimize plan: {str(e)}"
            }
```

### 3.3 决策智能体 (DecisionAgent)

```python
"""
决策智能体实现
位置：src/camel_agents/decision_agent.py
"""

from typing import Dict, Any, List, Optional
import json
from datetime import datetime
from .base_agent import BaseRobotAgent

class DecisionAgent(BaseRobotAgent):
    """决策智能体 - 负责执行决策和任务协调"""
    
    def __init__(self, model_backend, message_bus, config: Dict[str, Any] = None):
        system_message = """
你是一个智能机器人的决策智能体。你的职责包括：

1. **执行决策**：
   - 评估执行计划的可行性
   - 决定任务的执行时机和顺序
   - 处理多任务间的优先级冲突

2. **资源协调**：
   - 协调各个执行模块的工作
   - 管理系统资源的分配
   - 处理资源竞争和冲突

3. **状态监控**：
   - 监控任务执行状态
   - 检测异常情况和错误
   - 触发必要的干预措施

4. **风险管理**：
   - 评估执行风险
   - 做出安全相关的决策
   - 在必要时中止或修改任务

5. **学习优化**：
   - 从执行结果中学习
   - 优化决策策略
   - 改进系统性能

请确保所有决策都是安全、合理和高效的。
"""
        super().__init__("DecisionAgent", model_backend, system_message, message_bus, config)
        self.capabilities = [
            "execution_decision",
            "resource_coordination",
            "status_monitoring", 
            "risk_management",
            "learning_optimization"
        ]
        
        # 决策状态
        self.active_executions = {}
        self.resource_status = {}
        self.decision_history = []
        self.risk_thresholds = {
            "safety": 0.8,
            "success_probability": 0.6,
            "resource_availability": 0.7
        }
        
    async def process_message(self, message: BaseMessage) -> BaseMessage:
        """处理消息"""
        return await super().process_message(message)
        
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行决策任务"""
        task_type = task.get("type")
        
        if task_type == "execution_plan":
            return await self._evaluate_and_execute_plan(task)
        elif task_type == "status_update":
            return await self._handle_status_update(task)
        elif task_type == "risk_assessment":
            return await self._assess_risk(task)
        elif task_type == "resource_request":
            return await self._handle_resource_request(task)
        else:
            return {
                "success": False,
                "error": f"Unknown task type: {task_type}"
            }
            
    async def _evaluate_and_execute_plan(self, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估并执行计划"""
        plan = plan_data.get("execution_plan", plan_data)
        plan_id = plan.get("plan_id")
        
        # 风险评估
        risk_assessment = await self._assess_plan_risk(plan)
        
        # 资源可用性检查
        resource_check = await self._check_resource_availability(plan)
        
        # 决策评估
        decision = await self._make_execution_decision(plan, risk_assessment, resource_check)
        
        if decision["approved"]:
            # 开始执行
            execution_id = await self._start_execution(plan)
            
            return {
                "success": True,
                "execution_id": execution_id,
                "decision": decision,
                "risk_assessment": risk_assessment
            }
        else:
            # 拒绝执行
            await self.send_message(
                recipient="PlanningAgent",
                content={
                    "plan_id": plan_id,
                    "status": "rejected",
                    "reason": decision["reason"],
                    "suggestions": decision.get("suggestions", [])
                },
                message_type="plan_feedback"
            )
            
            return {
                "success": False,
                "reason": decision["reason"],
                "suggestions": decision.get("suggestions", [])
            }
            
    async def _assess_plan_risk(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """评估计划风险"""
        risk_prompt = f"""
评估以下执行计划的风险：

执行计划：
{json.dumps(plan, ensure_ascii=False, indent=2)}

请从以下方面评估风险：
1. 安全风险：可能的安全隐患和事故风险
2. 技术风险：技术实现的难度和失败概率
3. 环境风险：环境因素对执行的影响
4. 时间风险：执行时间超出预期的可能性
5. 资源风险：资源不足或冲突的风险

返回JSON格式的风险评估：
{{
    "overall_risk_level": "风险等级（low/medium/high）",
    "risk_score": "风险分数（0-1）",
    "risk_factors": [
        {{
            "category": "风险类别",
            "description": "风险描述", 
            "probability": "发生概率（0-1）",
            "impact": "影响程度（0-1）",
            "mitigation": "缓解措施"
        }}
    ],
    "recommendations": [
        "风险缓解建议"
    ]
}}
"""
        
        risk_message = BaseMessage.make_user_message(
            role_name="user",
            content=risk_prompt
        )
        
        response = await self.agent.step(risk_message)
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {
                "overall_risk_level": "medium",
                "risk_score": 0.5,
                "risk_factors": [],
                "recommendations": []
            }
            
    async def _check_resource_availability(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """检查资源可用性"""
        required_resources = set()
        
        # 从子任务中提取所需资源
        for subtask in plan.get("subtasks", []):
            executor = subtask.get("executor", "")
            if executor:
                required_resources.add(executor)
                
        # 检查每个资源的可用性
        resource_status = {}
        for resource in required_resources:
            # 简化实现：假设所有资源都可用
            # 实际实现中需要查询各个模块的状态
            resource_status[resource] = {
                "available": True,
                "load": 0.3,  # 当前负载
                "estimated_availability": 0.9
            }
            
        overall_availability = min(
            res["estimated_availability"] for res in resource_status.values()
        ) if resource_status else 1.0
        
        return {
            "overall_availability": overall_availability,
            "resource_status": resource_status,
            "bottlenecks": []  # 资源瓶颈
        }
        
    async def _make_execution_decision(self, 
                                     plan: Dict[str, Any],
                                     risk_assessment: Dict[str, Any],
                                     resource_check: Dict[str, Any]) -> Dict[str, Any]:
        """做出执行决策"""
        decision_prompt = f"""
基于以下信息做出执行决策：

执行计划：
{json.dumps(plan, ensure_ascii=False, indent=2)}

风险评估：
{json.dumps(risk_assessment, ensure_ascii=False, indent=2)}

资源状态：
{json.dumps(resource_check, ensure_ascii=False, indent=2)}

决策标准：
- 安全风险阈值：{self.risk_thresholds['safety']}
- 成功概率阈值：{self.risk_thresholds['success_probability']}
- 资源可用性阈值：{self.risk_thresholds['resource_availability']}

请做出执行决策并返回JSON格式结果：
{{
    "approved": "是否批准执行（true/false）",
    "confidence": "决策置信度（0-1）",
    "reason": "决策理由",
    "conditions": [
        "执行条件或要求"
    ],
    "suggestions": [
        "改进建议（如果拒绝执行）"
    ],
    "monitoring_points": [
        "需要重点监控的执行点"
    ]
}}
"""
        
        decision_message = BaseMessage.make_user_message(
            role_name="user",
            content=decision_prompt
        )
        
        response = await self.agent.step(decision_message)
        
        try:
            decision = json.loads(response.content)
            
            # 记录决策历史
            self.decision_history.append({
                "timestamp": datetime.now().isoformat(),
                "plan_id": plan.get("plan_id"),
                "decision": decision,
                "risk_assessment": risk_assessment,
                "resource_check": resource_check
            })
            
            return decision
            
        except json.JSONDecodeError:
            # 默认保守决策
            return {
                "approved": False,
                "confidence": 0.0,
                "reason": "Failed to parse decision response",
                "conditions": [],
                "suggestions": ["Review plan and try again"],
                "monitoring_points": []
            }
            
    async def _start_execution(self, plan: Dict[str, Any]) -> str:
        """开始执行计划"""
        plan_id = plan.get("plan_id")
        execution_id = f"exec_{plan_id}_{datetime.now().strftime('%H%M%S')}"
        
        # 创建执行记录
        execution_record = {
            "execution_id": execution_id,
            "plan": plan,
            "status": "executing",
            "started_at": datetime.now().isoformat(),
            "current_subtask": None,
            "completed_subtasks": [],
            "failed_subtasks": []
        }
        
        self.active_executions[execution_id] = execution_record
        
        # 开始执行第一个子任务
        await self._execute_next_subtask(execution_id)
        
        return execution_id
        
    async def _execute_next_subtask(self, execution_id: str):
        """执行下一个子任务"""
        execution = self.active_executions.get(execution_id)
        if not execution:
            return
            
        plan = execution["plan"]
        completed = set(execution["completed_subtasks"])
        
        # 找到下一个可执行的子任务
        next_subtask = None
        for subtask in plan.get("subtasks", []):
            subtask_id = subtask["id"]
            
            # 跳过已完成的子任务
            if subtask_id in completed:
                continue
                
            # 检查依赖是否满足
            dependencies = subtask.get("dependencies", [])
            if all(dep in completed for dep in dependencies):
                next_subtask = subtask
                break
                
        if next_subtask:
            # 执行子任务
            execution["current_subtask"] = next_subtask["id"]
            
            await self.send_message(
                recipient=next_subtask.get("executor", "ROS2Agent"),
                content={
                    "execution_id": execution_id,
                    "subtask": next_subtask,
                    "plan_context": plan
                },
                message_type="subtask_execution",
                priority=2
            )
        else:
            # 所有子任务完成
            execution["status"] = "completed"
            execution["completed_at"] = datetime.now().isoformat()
            
            # 通知完成
            await self.send_message(
                recipient="DialogAgent",
                content={
                    "execution_id": execution_id,
                    "status": "completed",
                    "plan_id": plan.get("plan_id")
                },
                message_type="execution_completed"
            )
            
    async def _handle_status_update(self, update: Dict[str, Any]) -> Dict[str, Any]:
        """处理状态更新"""
        execution_id = update.get("execution_id")
        subtask_id = update.get("subtask_id")
        status = update.get("status")
        
        if execution_id not in self.active_executions:
            return {
                "success": False,
                "error": f"Execution {execution_id} not found"
            }
            
        execution = self.active_executions[execution_id]
        
        if status == "completed":
            # 子任务完成
            execution["completed_subtasks"].append(subtask_id)
            execution["current_subtask"] = None
            
            # 执行下一个子任务
            await self._execute_next_subtask(execution_id)
            
        elif status == "failed":
            # 子任务失败
            execution["failed_subtasks"].append(subtask_id)
            execution["status"] = "failed"
            execution["failure_reason"] = update.get("error")
            
            # 通知失败并请求重新规划
            await self.send_message(
                recipient="PlanningAgent",
                content={
                    "plan_id": execution["plan"].get("plan_id"),
                    "failure_info": {
                        "failed_subtask": subtask_id,
                        "error": update.get("error"),
                        "execution_context": execution
                    }
                },
                message_type="replan_request"
            )
            
        return {
            "success": True,
            "execution_status": execution["status"]
        }
```
## 4. 感知智能体 (PerceptionAgent)

### 4.1 感知智能体实现

```python
"""
感知智能体实现
位置：src/camel_agents/perception_agent.py
"""

from typing import Dict, Any, List, Optional, Tuple
import json
import cv2
import numpy as np
from datetime import datetime
import base64
from .base_agent import BaseRobotAgent

class PerceptionAgent(BaseRobotAgent):
    """感知智能体 - 负责多模态感知和环境理解"""
    
    def __init__(self, model_backend, message_bus, config: Dict[str, Any] = None):
        system_message = """
你是一个智能机器人的感知智能体。你的职责包括：

1. **视觉感知**：
   - 物体检测和识别
   - 场景理解和分析
   - 深度估计和3D重建
   - 视觉导航和定位

2. **多模态融合**：
   - 融合视觉、听觉、触觉等多种感知信息
   - 构建统一的环境表示
   - 处理传感器数据的不确定性

3. **环境建模**：
   - 构建和更新环境地图
   - 识别动态和静态物体
   - 预测环境变化

4. **异常检测**：
   - 检测异常情况和潜在危险
   - 监控系统状态和健康度
   - 触发安全警报

5. **感知记忆**：
   - 存储和检索感知历史
   - 学习环境模式和规律
   - 支持长期记忆和经验积累

请确保感知结果准确、及时和可靠。
"""
        super().__init__("PerceptionAgent", model_backend, system_message, message_bus, config)
        self.capabilities = [
            "visual_perception",
            "multimodal_fusion",
            "environment_modeling",
            "anomaly_detection", 
            "perception_memory"
        ]
        
        # 感知状态
        self.perception_history = []
        self.environment_model = {}
        self.detected_objects = {}
        self.sensor_data_buffer = {}
        
        # 感知配置
        self.perception_config = {
            "object_detection_threshold": 0.7,
            "scene_analysis_interval": 1.0,
            "memory_retention_days": 30,
            "anomaly_threshold": 0.8
        }
        
        # 初始化感知模块
        self._initialize_perception_modules()
        
    def _initialize_perception_modules(self):
        """初始化感知模块"""
        # 这里会初始化各种感知模型和处理器
        # 实际实现中会加载预训练模型
        self.perception_modules = {
            "object_detector": None,  # YOLO/DETR等物体检测模型
            "depth_estimator": None,  # 深度估计模型
            "scene_analyzer": None,   # 场景分析模型
            "anomaly_detector": None  # 异常检测模型
        }
        
    async def process_message(self, message: BaseMessage) -> BaseMessage:
        """处理消息"""
        return await super().process_message(message)
        
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行感知任务"""
        task_type = task.get("type")
        
        if task_type == "visual_perception":
            return await self._process_visual_data(task)
        elif task_type == "object_detection":
            return await self._detect_objects(task)
        elif task_type == "scene_analysis":
            return await self._analyze_scene(task)
        elif task_type == "environment_mapping":
            return await self._update_environment_map(task)
        elif task_type == "anomaly_detection":
            return await self._detect_anomalies(task)
        else:
            return {
                "success": False,
                "error": f"Unknown task type: {task_type}"
            }
            
    async def _process_visual_data(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """处理视觉数据"""
        image_data = task.get("image_data")
        camera_info = task.get("camera_info", {})
        processing_options = task.get("options", {})
        
        if not image_data:
            return {
                "success": False,
                "error": "No image data provided"
            }
            
        try:
            # 解码图像数据
            if isinstance(image_data, str):
                # Base64编码的图像
                image_bytes = base64.b64decode(image_data)
                image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            else:
                # 直接的numpy数组
                image = np.array(image_data)
                
            # 使用LLM进行视觉理解
            visual_analysis = await self._analyze_image_with_llm(image, processing_options)
            
            # 物体检测
            detected_objects = await self._detect_objects_in_image(image)
            
            # 场景分析
            scene_info = await self._analyze_scene_context(image, detected_objects)
            
            # 构建感知结果
            perception_result = {
                "timestamp": datetime.now().isoformat(),
                "camera_info": camera_info,
                "visual_analysis": visual_analysis,
                "detected_objects": detected_objects,
                "scene_info": scene_info,
                "image_metadata": {
                    "width": image.shape[1],
                    "height": image.shape[0],
                    "channels": image.shape[2] if len(image.shape) > 2 else 1
                }
            }
            
            # 存储到感知历史
            self.perception_history.append(perception_result)
            
            # 更新环境模型
            await self._update_environment_model(perception_result)
            
            return {
                "success": True,
                "perception_result": perception_result
            }
            
        except Exception as e:
            self.logger.error(f"Visual processing failed: {e}")
            return {
                "success": False,
                "error": f"Visual processing failed: {str(e)}"
            }
            
    async def _analyze_image_with_llm(self, image: np.ndarray, options: Dict[str, Any]) -> Dict[str, Any]:
        """使用LLM分析图像"""
        # 将图像转换为base64用于LLM分析
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        analysis_prompt = f"""
分析以下图像并提供详细的视觉理解：

图像数据：[Base64编码的图像]

分析要求：
1. 场景描述：描述图像中的整体场景和环境
2. 物体识别：识别图像中的主要物体和它们的位置
3. 空间关系：分析物体之间的空间关系和布局
4. 环境特征：识别环境的特征（室内/室外、光照、天气等）
5. 潜在风险：识别可能的安全隐患或障碍物
6. 导航信息：提供对机器人导航有用的信息

处理选项：{json.dumps(options, ensure_ascii=False)}

返回JSON格式的分析结果：
{{
    "scene_description": "场景整体描述",
    "objects": [
        {{
            "name": "物体名称",
            "confidence": "识别置信度（0-1）",
            "location": "位置描述",
            "properties": ["物体属性"]
        }}
    ],
    "spatial_relationships": [
        "空间关系描述"
    ],
    "environment_features": {{
        "lighting": "光照条件",
        "weather": "天气状况",
        "indoor_outdoor": "室内/室外",
        "surface_type": "地面类型"
    }},
    "safety_assessment": {{
        "risk_level": "风险等级（low/medium/high）",
        "hazards": ["潜在危险"],
        "safe_areas": ["安全区域"]
    }},
    "navigation_info": {{
        "traversable_areas": ["可通行区域"],
        "obstacles": ["障碍物"],
        "landmarks": ["地标"]
    }}
}}
"""
        
        # 注意：实际实现中需要使用支持视觉的LLM（如GPT-4V）
        analysis_message = BaseMessage.make_user_message(
            role_name="user",
            content=analysis_prompt
        )
        
        response = await self.agent.step(analysis_message)
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {
                "scene_description": response.content,
                "objects": [],
                "spatial_relationships": [],
                "environment_features": {},
                "safety_assessment": {"risk_level": "unknown"},
                "navigation_info": {}
            }
            
    async def _detect_objects_in_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """在图像中检测物体"""
        # 简化实现：实际中会使用YOLO、DETR等模型
        detected_objects = []
        
        # 模拟物体检测结果
        # 实际实现中会调用物体检测模型
        mock_detections = [
            {
                "class_name": "person",
                "confidence": 0.95,
                "bbox": [100, 150, 200, 400],  # [x1, y1, x2, y2]
                "center": [150, 275]
            },
            {
                "class_name": "chair", 
                "confidence": 0.87,
                "bbox": [300, 200, 450, 350],
                "center": [375, 275]
            }
        ]
        
        for detection in mock_detections:
            if detection["confidence"] >= self.perception_config["object_detection_threshold"]:
                detected_objects.append({
                    "id": f"obj_{len(detected_objects)}_{datetime.now().strftime('%H%M%S')}",
                    "class_name": detection["class_name"],
                    "confidence": detection["confidence"],
                    "bounding_box": detection["bbox"],
                    "center_point": detection["center"],
                    "timestamp": datetime.now().isoformat()
                })
                
        return detected_objects
        
    async def _analyze_scene_context(self, image: np.ndarray, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析场景上下文"""
        scene_prompt = f"""
基于检测到的物体分析场景上下文：

检测到的物体：
{json.dumps(objects, ensure_ascii=False, indent=2)}

图像尺寸：{image.shape}

请分析：
1. 场景类型（办公室、家庭、工厂等）
2. 活动推断（人们在做什么）
3. 环境状态（整洁、杂乱、正常等）
4. 时间推断（基于光照和活动）
5. 功能区域（工作区、休息区等）

返回JSON格式的场景分析：
{{
    "scene_type": "场景类型",
    "activity_inference": "活动推断",
    "environment_state": "环境状态",
    "time_inference": "时间推断",
    "functional_areas": [
        {{
            "area_type": "区域类型",
            "description": "区域描述",
            "objects": ["相关物体"]
        }}
    ],
    "scene_complexity": "场景复杂度（simple/medium/complex）",
    "dynamic_elements": ["动态元素"]
}}
"""
        
        scene_message = BaseMessage.make_user_message(
            role_name="user",
            content=scene_prompt
        )
        
        response = await self.agent.step(scene_message)
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {
                "scene_type": "unknown",
                "activity_inference": "unknown",
                "environment_state": "normal",
                "functional_areas": [],
                "scene_complexity": "medium",
                "dynamic_elements": []
            }
            
    async def _update_environment_model(self, perception_result: Dict[str, Any]):
        """更新环境模型"""
        timestamp = perception_result["timestamp"]
        objects = perception_result["detected_objects"]
        scene_info = perception_result["scene_info"]
        
        # 更新物体跟踪
        for obj in objects:
            obj_id = obj["id"]
            self.detected_objects[obj_id] = {
                "last_seen": timestamp,
                "object_info": obj,
                "tracking_history": self.detected_objects.get(obj_id, {}).get("tracking_history", [])
            }
            self.detected_objects[obj_id]["tracking_history"].append({
                "timestamp": timestamp,
                "position": obj["center_point"],
                "confidence": obj["confidence"]
            })
            
        # 更新环境特征
        if "environment_features" in perception_result["visual_analysis"]:
            env_features = perception_result["visual_analysis"]["environment_features"]
            self.environment_model.update({
                "last_updated": timestamp,
                "current_features": env_features,
                "scene_type": scene_info.get("scene_type", "unknown")
            })
            
    async def _detect_anomalies(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """检测异常情况"""
        current_perception = task.get("perception_data")
        
        if not current_perception:
            return {
                "success": False,
                "error": "No perception data provided"
            }
            
        anomaly_prompt = f"""
基于当前感知数据和历史模式检测异常：

当前感知数据：
{json.dumps(current_perception, ensure_ascii=False, indent=2)}

历史环境模型：
{json.dumps(self.environment_model, ensure_ascii=False, indent=2)}

最近检测到的物体：
{json.dumps(list(self.detected_objects.values())[-5:], ensure_ascii=False, indent=2)}

请检测以下类型的异常：
1. 新出现的未知物体
2. 物体位置的异常变化
3. 环境状态的突然改变
4. 潜在的安全威胁
5. 传感器数据的异常模式

返回JSON格式的异常检测结果：
{{
    "anomalies_detected": "是否检测到异常（true/false）",
    "anomaly_count": "异常数量",
    "anomalies": [
        {{
            "type": "异常类型",
            "severity": "严重程度（low/medium/high/critical）",
            "description": "异常描述",
            "confidence": "检测置信度（0-1）",
            "location": "异常位置",
            "recommended_action": "建议采取的行动"
        }}
    ],
    "overall_risk_level": "整体风险等级",
    "immediate_action_required": "是否需要立即行动（true/false）"
}}
"""
        
        anomaly_message = BaseMessage.make_user_message(
            role_name="user",
            content=anomaly_prompt
        )
        
        response = await self.agent.step(anomaly_message)
        
        try:
            anomaly_result = json.loads(response.content)
            
            # 如果检测到高风险异常，立即通知其他智能体
            if anomaly_result.get("immediate_action_required"):
                await self.broadcast_message(
                    content={
                        "alert_type": "anomaly_detected",
                        "anomalies": anomaly_result["anomalies"],
                        "risk_level": anomaly_result["overall_risk_level"]
                    },
                    message_type="safety_alert",
                    priority=3
                )
                
            return {
                "success": True,
                "anomaly_result": anomaly_result
            }
            
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": "Failed to parse anomaly detection result"
            }
```

## 5. 学习智能体 (LearningAgent)

### 5.1 学习智能体实现

```python
"""
学习智能体实现
位置：src/camel_agents/learning_agent.py
"""

from typing import Dict, Any, List, Optional
import json
import numpy as np
from datetime import datetime, timedelta
from .base_agent import BaseRobotAgent

class LearningAgent(BaseRobotAgent):
    """学习智能体 - 负责经验学习和知识积累"""
    
    def __init__(self, model_backend, message_bus, config: Dict[str, Any] = None):
        system_message = """
你是一个智能机器人的学习智能体。你的职责包括：

1. **经验学习**：
   - 从任务执行中学习成功和失败的模式
   - 识别最佳实践和优化策略
   - 积累领域知识和专业技能

2. **模式识别**：
   - 识别用户行为模式和偏好
   - 发现环境变化的规律
   - 学习任务执行的最优路径

3. **知识管理**：
   - 组织和结构化学习到的知识
   - 建立知识图谱和关联关系
   - 支持知识检索和推理

4. **适应性优化**：
   - 根据学习结果调整系统行为
   - 优化决策策略和执行方法
   - 提高系统的适应性和鲁棒性

5. **元学习**：
   - 学习如何更好地学习
   - 优化学习策略和方法
   - 提高学习效率和效果

请确保学习过程是持续的、有效的和可解释的。
"""
        super().__init__("LearningAgent", model_backend, system_message, message_bus, config)
        self.capabilities = [
            "experience_learning",
            "pattern_recognition",
            "knowledge_management",
            "adaptive_optimization",
            "meta_learning"
        ]
        
        # 学习状态
        self.experience_database = []
        self.learned_patterns = {}
        self.knowledge_graph = {}
        self.performance_metrics = {}
        self.learning_objectives = []
        
        # 学习配置
        self.learning_config = {
            "experience_retention_days": 90,
            "pattern_confidence_threshold": 0.7,
            "learning_rate": 0.1,
            "knowledge_update_interval": 3600  # 1小时
        }
        
    async def process_message(self, message: BaseMessage) -> BaseMessage:
        """处理消息"""
        return await super().process_message(message)
        
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行学习任务"""
        task_type = task.get("type")
        
        if task_type == "learn_from_experience":
            return await self._learn_from_experience(task)
        elif task_type == "pattern_analysis":
            return await self._analyze_patterns(task)
        elif task_type == "knowledge_update":
            return await self._update_knowledge(task)
        elif task_type == "performance_analysis":
            return await self._analyze_performance(task)
        elif task_type == "optimization_suggestion":
            return await self._suggest_optimizations(task)
        else:
            return {
                "success": False,
                "error": f"Unknown task type: {task_type}"
            }
            
    async def _learn_from_experience(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """从经验中学习"""
        experience_data = task.get("experience_data")
        
        if not experience_data:
            return {
                "success": False,
                "error": "No experience data provided"
            }
            
        # 结构化经验数据
        structured_experience = await self._structure_experience(experience_data)
        
        # 提取学习要点
        learning_insights = await self._extract_learning_insights(structured_experience)
        
        # 更新知识库
        await self._update_knowledge_base(learning_insights)
        
        # 识别改进机会
        improvement_opportunities = await self._identify_improvements(structured_experience)
        
        # 存储经验
        self.experience_database.append({
            "timestamp": datetime.now().isoformat(),
            "raw_data": experience_data,
            "structured_data": structured_experience,
            "learning_insights": learning_insights,
            "improvements": improvement_opportunities
        })
        
        return {
            "success": True,
            "learning_insights": learning_insights,
            "improvement_opportunities": improvement_opportunities
        }
        
    async def _structure_experience(self, experience_data: Dict[str, Any]) -> Dict[str, Any]:
        """结构化经验数据"""
        structure_prompt = f"""
将以下原始经验数据结构化为标准格式：

原始经验数据：
{json.dumps(experience_data, ensure_ascii=False, indent=2)}

请提取和组织以下信息：
1. 任务信息：任务类型、目标、参数
2. 执行过程：步骤、决策点、行动
3. 结果评估：成功/失败、性能指标、用户反馈
4. 环境上下文：环境条件、约束、干扰因素
5. 资源使用：时间、计算、硬件资源
6. 异常情况：错误、异常、恢复措施

返回JSON格式的结构化经验：
{{
    "task_info": {{
        "task_type": "任务类型",
        "objective": "任务目标",
        "parameters": {{}},
        "priority": "优先级"
    }},
    "execution_process": [
        {{
            "step": "步骤编号",
            "action": "执行动作",
            "decision_point": "决策点",
            "duration": "执行时间",
            "result": "步骤结果"
        }}
    ],
    "outcome": {{
        "success": "成功/失败",
        "performance_metrics": {{}},
        "user_satisfaction": "用户满意度",
        "completion_time": "完成时间"
    }},
    "context": {{
        "environment": "环境描述",
        "constraints": ["约束条件"],
        "interference": ["干扰因素"]
    }},
    "resource_usage": {{
        "computational": "计算资源",
        "time": "时间资源",
        "hardware": "硬件资源"
    }},
    "exceptions": [
        {{
            "type": "异常类型",
            "description": "异常描述",
            "recovery_action": "恢复措施",
            "impact": "影响程度"
        }}
    ]
}}
"""
        
        structure_message = BaseMessage.make_user_message(
            role_name="user",
            content=structure_prompt
        )
        
        response = await self.agent.step(structure_message)
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {
                "task_info": {},
                "execution_process": [],
                "outcome": {},
                "context": {},
                "resource_usage": {},
                "exceptions": []
            }
            
    async def _extract_learning_insights(self, structured_experience: Dict[str, Any]) -> Dict[str, Any]:
        """提取学习洞察"""
        insights_prompt = f"""
从以下结构化经验中提取学习洞察：

结构化经验：
{json.dumps(structured_experience, ensure_ascii=False, indent=2)}

历史学习模式：
{json.dumps(self.learned_patterns, ensure_ascii=False, indent=2)}

请分析并提取：
1. 成功因素：导致成功的关键因素
2. 失败原因：导致失败的根本原因
3. 最佳实践：可复用的最佳实践
4. 风险因素：需要注意的风险点
5. 优化机会：可以改进的方面
6. 知识更新：需要更新的知识点

返回JSON格式的学习洞察：
{{
    "success_factors": [
        {{
            "factor": "成功因素",
            "importance": "重要程度（0-1）",
            "evidence": "支持证据",
            "generalizability": "可泛化性（0-1）"
        }}
    ],
    "failure_causes": [
        {{
            "cause": "失败原因",
            "frequency": "出现频率",
            "severity": "严重程度",
            "prevention": "预防措施"
        }}
    ],
    "best_practices": [
        {{
            "practice": "最佳实践",
            "context": "适用上下文",
            "effectiveness": "有效性（0-1）",
            "implementation": "实施方法"
        }}
    ],
    "risk_factors": [
        {{
            "risk": "风险因素",
            "probability": "发生概率（0-1）",
            "impact": "影响程度（0-1）",
            "mitigation": "缓解策略"
        }}
    ],
    "optimization_opportunities": [
        {{
            "area": "优化领域",
            "potential_improvement": "潜在改进",
            "implementation_difficulty": "实施难度（0-1）",
            "expected_benefit": "预期收益"
        }}
    ],
    "knowledge_updates": [
        {{
            "knowledge_area": "知识领域",
            "update_type": "更新类型（new/modify/delete）",
            "content": "更新内容",
            "confidence": "置信度（0-1）"
        }}
    ]
}}
"""
        
        insights_message = BaseMessage.make_user_message(
            role_name="user",
            content=insights_prompt
        )
        
        response = await self.agent.step(insights_message)
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {
                "success_factors": [],
                "failure_causes": [],
                "best_practices": [],
                "risk_factors": [],
                "optimization_opportunities": [],
                "knowledge_updates": []
            }
            
    async def _analyze_patterns(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """分析模式"""
        analysis_scope = task.get("scope", "all")  # all, user_behavior, task_execution, environment
        time_window = task.get("time_window", 7)  # 天数
        
        # 获取指定时间窗口内的经验数据
        cutoff_date = datetime.now() - timedelta(days=time_window)
        recent_experiences = [
            exp for exp in self.experience_database
            if datetime.fromisoformat(exp["timestamp"]) > cutoff_date
        ]
        
        if not recent_experiences:
            return {
                "success": False,
                "error": "Insufficient data for pattern analysis"
            }
            
        pattern_prompt = f"""
分析以下经验数据中的模式：

分析范围：{analysis_scope}
时间窗口：{time_window}天
经验数据数量：{len(recent_experiences)}

经验数据样本：
{json.dumps(recent_experiences[:5], ensure_ascii=False, indent=2)}

请识别以下类型的模式：
1. 用户行为模式：用户的使用习惯和偏好
2. 任务执行模式：任务执行的规律和趋势
3. 环境变化模式：环境条件的变化规律
4. 性能模式：系统性能的变化趋势
5. 错误模式：错误发生的规律和原因

返回JSON格式的模式分析：
{{
    "user_behavior_patterns": [
        {{
            "pattern": "模式描述",
            "frequency": "出现频率",
            "confidence": "置信度（0-1）",
            "implications": "含义和影响"
        }}
    ],
    "task_execution_patterns": [
        {{
            "pattern": "执行模式",
            "success_rate": "成功率",
            "average_duration": "平均时长",
            "optimization_potential": "优化潜力"
        }}
    ],
    "environment_patterns": [
        {{
            "pattern": "环境模式",
            "conditions": "触发条件",
            "impact": "对系统的影响",
            "adaptation_strategy": "适应策略"
        }}
    ],
    "performance_trends": [
        {{
            "metric": "性能指标",
            "trend": "变化趋势",
            "factors": "影响因素",
            "prediction": "未来预测"
        }}
    ],
    "error_patterns": [
        {{
            "error_type": "错误类型",
            "occurrence_pattern": "发生模式",
            "root_causes": "根本原因",
            "prevention_strategy": "预防策略"
        }}
    ]
}}
"""
        
        pattern_message = BaseMessage.make_user_message(
            role_name="user",
            content=pattern_prompt
        )
        
        response = await self.agent.step(pattern_message)
        
        try:
            patterns = json.loads(response.content)
            
            # 更新学习到的模式
            self.learned_patterns.update({
                "last_updated": datetime.now().isoformat(),
                "analysis_scope": analysis_scope,
                "patterns": patterns
            })
            
            return {
                "success": True,
                "patterns": patterns
            }
            
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": "Failed to parse pattern analysis"
            }
            
    async def _suggest_optimizations(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """建议优化方案"""
        target_area = task.get("target_area", "overall")  # overall, planning, execution, perception
        current_performance = task.get("current_performance", {})
        
        optimization_prompt = f"""
基于学习到的经验和模式，为以下目标领域提供优化建议：

目标领域：{target_area}
当前性能：{json.dumps(current_performance, ensure_ascii=False, indent=2)}

学习到的模式：
{json.dumps(self.learned_patterns, ensure_ascii=False, indent=2)}

最近的改进机会：
{json.dumps([exp.get("improvements", []) for exp in self.experience_database[-10:]], ensure_ascii=False, indent=2)}

请提供具体的优化建议：
1. 短期优化：可以立即实施的改进
2. 中期优化：需要一定准备时间的改进
3. 长期优化：需要系统性改进的方案
4. 实验性优化：需要验证的创新方案

返回JSON格式的优化建议：
{{
    "short_term_optimizations": [
        {{
            "optimization": "优化方案",
            "implementation_steps": ["实施步骤"],
            "expected_benefit": "预期收益",
            "implementation_time": "实施时间",
            "risk_level": "风险等级（low/medium/high）"
        }}
    ],
    "medium_term_optimizations": [
        {{
            "optimization": "优化方案",
            "prerequisites": ["前置条件"],
            "implementation_plan": "实施计划",
            "expected_impact": "预期影响",
            "resource_requirements": "资源需求"
        }}
    ],
    "long_term_optimizations": [
        {{
            "optimization": "优化方案",
            "strategic_value": "战略价值",
            "implementation_roadmap": "实施路线图",
            "success_metrics": ["成功指标"],
            "investment_required": "所需投资"
        }}
    ],
    "experimental_optimizations": [
        {{
            "optimization": "实验性方案",
            "hypothesis": "假设",
            "experiment_design": "实验设计",
            "validation_criteria": "验证标准",
            "potential_risks": ["潜在风险"]
        }}
    ],
    "priority_ranking": [
        "按优先级排序的优化方案"
    ]
}}
"""
        
        optimization_message = BaseMessage.make_user_message(
            role_name="user",
            content=optimization_prompt
        )
        
        response = await self.agent.step(optimization_message)
        
        try:
            optimizations = json.loads(response.content)
            
            # 发送优化建议给相关智能体
            if target_area in ["overall", "planning"]:
                await self.send_message(
                    recipient="PlanningAgent",
                    content=optimizations,
                    message_type="optimization_suggestions"
                )
                
            if target_area in ["overall", "execution"]:
                await self.send_message(
                    recipient="DecisionAgent",
                    content=optimizations,
                    message_type="optimization_suggestions"
                )
                
            return {
                "success": True,
                "optimizations": optimizations
            }
            
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": "Failed to parse optimization suggestions"
            }
```
## 6. ROS2智能体 (ROS2Agent)

### 6.1 ROS2智能体核心实现

```python
"""
ROS2智能体实现
位置：src/camel_agents/ros2_agent.py
"""

from typing import Dict, Any, List, Optional, Callable
import json
import asyncio
import threading
from datetime import datetime
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image, LaserScan, JointState
from std_msgs.msg import String, Bool
from nav_msgs.msg import OccupancyGrid, Path
from .base_agent import BaseRobotAgent

class ROS2Agent(BaseRobotAgent):
    """ROS2智能体 - 作为CAMEL框架中的平等协作成员，负责物理交互"""
    
    def __init__(self, model_backend, message_bus, config: Dict[str, Any] = None):
        system_message = """
你是一个智能机器人系统中的ROS2智能体。你的职责包括：

1. **物理交互接口**：
   - 控制机器人的移动和操作
   - 获取传感器数据和状态信息
   - 执行具体的物理动作指令

2. **ROS2系统管理**：
   - 管理ROS2节点和话题
   - 处理ROS2服务调用和动作
   - 监控系统健康状态

3. **数据转换**：
   - 将高层指令转换为ROS2消息
   - 将ROS2数据转换为智能体可理解的格式
   - 处理不同坐标系和单位的转换

4. **安全监控**：
   - 监控机器人安全状态
   - 实施紧急停止和安全保护
   - 报告异常和故障情况

5. **协作通信**：
   - 与其他CAMEL智能体协作
   - 提供物理世界的反馈信息
   - 参与多智能体决策过程

作为CAMEL框架中的平等成员，你需要主动参与协作，而不仅仅是被动执行指令。
"""
        super().__init__("ROS2Agent", model_backend, system_message, message_bus, config)
        self.capabilities = [
            "robot_control",
            "sensor_data_acquisition",
            "physical_interaction",
            "safety_monitoring",
            "ros2_system_management"
        ]
        
        # ROS2相关状态
        self.ros2_node = None
        self.ros2_executor = None
        self.ros2_thread = None
        self.publishers = {}
        self.subscribers = {}
        self.service_clients = {}
        self.action_clients = {}
        
        # 机器人状态
        self.robot_state = {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "velocity": {"linear": {"x": 0.0, "y": 0.0, "z": 0.0}, 
                        "angular": {"x": 0.0, "y": 0.0, "z": 0.0}},
            "joint_states": {},
            "sensor_data": {},
            "system_status": "idle",
            "last_updated": datetime.now().isoformat()
        }
        
        # 安全状态
        self.safety_status = {
            "emergency_stop": False,
            "safety_violations": [],
            "last_safety_check": datetime.now().isoformat()
        }
        
        # 初始化ROS2
        self._initialize_ros2()
        
    def _initialize_ros2(self):
        """初始化ROS2系统"""
        try:
            # 初始化ROS2
            if not rclpy.ok():
                rclpy.init()
                
            # 创建ROS2节点
            self.ros2_node = ROS2AgentNode(self)
            
            # 创建执行器
            self.ros2_executor = MultiThreadedExecutor()
            self.ros2_executor.add_node(self.ros2_node)
            
            # 在单独线程中运行ROS2
            self.ros2_thread = threading.Thread(
                target=self._run_ros2_executor,
                daemon=True
            )
            self.ros2_thread.start()
            
            self.logger.info("ROS2 system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ROS2: {e}")
            raise
            
    def _run_ros2_executor(self):
        """运行ROS2执行器"""
        try:
            self.ros2_executor.spin()
        except Exception as e:
            self.logger.error(f"ROS2 executor error: {e}")
            
    async def process_message(self, message: BaseMessage) -> BaseMessage:
        """处理消息"""
        return await super().process_message(message)
        
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行ROS2相关任务"""
        task_type = task.get("type")
        
        if task_type == "robot_movement":
            return await self._execute_movement(task)
        elif task_type == "sensor_reading":
            return await self._get_sensor_data(task)
        elif task_type == "manipulation":
            return await self._execute_manipulation(task)
        elif task_type == "navigation":
            return await self._execute_navigation(task)
        elif task_type == "system_status":
            return await self._get_system_status(task)
        elif task_type == "emergency_stop":
            return await self._emergency_stop(task)
        else:
            return {
                "success": False,
                "error": f"Unknown task type: {task_type}"
            }
            
    async def _execute_movement(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行移动指令"""
        movement_params = task.get("parameters", {})
        movement_type = movement_params.get("type", "velocity")
        
        try:
            if movement_type == "velocity":
                # 速度控制
                linear_vel = movement_params.get("linear", {"x": 0.0, "y": 0.0, "z": 0.0})
                angular_vel = movement_params.get("angular", {"x": 0.0, "y": 0.0, "z": 0.0})
                duration = movement_params.get("duration", 1.0)
                
                result = await self._send_velocity_command(linear_vel, angular_vel, duration)
                
            elif movement_type == "position":
                # 位置控制
                target_position = movement_params.get("position")
                target_orientation = movement_params.get("orientation")
                
                result = await self._move_to_position(target_position, target_orientation)
                
            elif movement_type == "relative":
                # 相对移动
                relative_movement = movement_params.get("relative")
                
                result = await self._move_relative(relative_movement)
                
            else:
                return {
                    "success": False,
                    "error": f"Unknown movement type: {movement_type}"
                }
                
            # 使用LLM分析移动结果
            analysis = await self._analyze_movement_result(task, result)
            
            return {
                "success": result["success"],
                "movement_result": result,
                "analysis": analysis
            }
            
        except Exception as e:
            self.logger.error(f"Movement execution failed: {e}")
            return {
                "success": False,
                "error": f"Movement execution failed: {str(e)}"
            }
            
    async def _send_velocity_command(self, linear_vel: Dict, angular_vel: Dict, duration: float) -> Dict[str, Any]:
        """发送速度指令"""
        if not self.ros2_node:
            return {"success": False, "error": "ROS2 node not initialized"}
            
        # 安全检查
        safety_check = await self._check_movement_safety(linear_vel, angular_vel)
        if not safety_check["safe"]:
            return {
                "success": False,
                "error": f"Movement safety check failed: {safety_check['reason']}"
            }
            
        try:
            # 创建Twist消息
            twist_msg = Twist()
            twist_msg.linear.x = float(linear_vel.get("x", 0.0))
            twist_msg.linear.y = float(linear_vel.get("y", 0.0))
            twist_msg.linear.z = float(linear_vel.get("z", 0.0))
            twist_msg.angular.x = float(angular_vel.get("x", 0.0))
            twist_msg.angular.y = float(angular_vel.get("y", 0.0))
            twist_msg.angular.z = float(angular_vel.get("z", 0.0))
            
            # 发送指令
            start_time = datetime.now()
            self.ros2_node.publish_velocity(twist_msg)
            
            # 等待指定时间
            await asyncio.sleep(duration)
            
            # 停止机器人
            stop_msg = Twist()  # 所有速度为0
            self.ros2_node.publish_velocity(stop_msg)
            
            end_time = datetime.now()
            
            # 更新机器人状态
            self.robot_state["velocity"]["linear"] = linear_vel
            self.robot_state["velocity"]["angular"] = angular_vel
            self.robot_state["last_updated"] = end_time.isoformat()
            
            return {
                "success": True,
                "execution_time": (end_time - start_time).total_seconds(),
                "final_velocity": {"linear": linear_vel, "angular": angular_vel}
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Velocity command failed: {str(e)}"
            }
            
    async def _check_movement_safety(self, linear_vel: Dict, angular_vel: Dict) -> Dict[str, Any]:
        """检查移动安全性"""
        safety_prompt = f"""
检查以下移动指令的安全性：

线性速度：{json.dumps(linear_vel, indent=2)}
角速度：{json.dumps(angular_vel, indent=2)}

当前机器人状态：
{json.dumps(self.robot_state, ensure_ascii=False, indent=2)}

当前安全状态：
{json.dumps(self.safety_status, ensure_ascii=False, indent=2)}

安全检查项目：
1. 速度是否在安全范围内
2. 是否存在紧急停止状态
3. 传感器数据是否显示障碍物
4. 系统状态是否正常

返回JSON格式的安全评估：
{{
    "safe": "是否安全（true/false）",
    "confidence": "安全置信度（0-1）",
    "reason": "安全评估理由",
    "warnings": ["安全警告"],
    "recommendations": ["安全建议"]
}}
"""
        
        safety_message = BaseMessage.make_user_message(
            role_name="user",
            content=safety_prompt
        )
        
        response = await self.agent.step(safety_message)
        
        try:
            safety_result = json.loads(response.content)
            
            # 硬编码的安全检查
            max_linear_speed = 2.0  # m/s
            max_angular_speed = 1.0  # rad/s
            
            if (abs(linear_vel.get("x", 0)) > max_linear_speed or
                abs(angular_vel.get("z", 0)) > max_angular_speed):
                safety_result["safe"] = False
                safety_result["reason"] = "Speed exceeds safety limits"
                
            if self.safety_status["emergency_stop"]:
                safety_result["safe"] = False
                safety_result["reason"] = "Emergency stop is active"
                
            return safety_result
            
        except json.JSONDecodeError:
            return {
                "safe": False,
                "confidence": 0.0,
                "reason": "Failed to parse safety assessment",
                "warnings": [],
                "recommendations": []
            }
            
    async def _analyze_movement_result(self, original_task: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """分析移动结果"""
        analysis_prompt = f"""
分析机器人移动任务的执行结果：

原始任务：
{json.dumps(original_task, ensure_ascii=False, indent=2)}

执行结果：
{json.dumps(result, ensure_ascii=False, indent=2)}

当前机器人状态：
{json.dumps(self.robot_state, ensure_ascii=False, indent=2)}

请分析：
1. 任务执行是否成功
2. 执行效果是否符合预期
3. 是否存在需要改进的地方
4. 对后续任务的影响
5. 学习要点和经验

返回JSON格式的分析结果：
{{
    "execution_success": "执行是否成功",
    "effectiveness": "执行效果评估",
    "performance_metrics": {{
        "accuracy": "执行精度",
        "efficiency": "执行效率",
        "smoothness": "运动平滑度"
    }},
    "issues_identified": ["发现的问题"],
    "improvement_suggestions": ["改进建议"],
    "learning_points": ["学习要点"],
    "impact_on_subsequent_tasks": "对后续任务的影响"
}}
"""
        
        analysis_message = BaseMessage.make_user_message(
            role_name="user",
            content=analysis_prompt
        )
        
        response = await self.agent.step(analysis_message)
        
        try:
            analysis = json.loads(response.content)
            
            # 发送学习数据给学习智能体
            await self.send_message(
                recipient="LearningAgent",
                content={
                    "type": "learn_from_experience",
                    "experience_data": {
                        "task": original_task,
                        "result": result,
                        "analysis": analysis,
                        "agent": "ROS2Agent",
                        "timestamp": datetime.now().isoformat()
                    }
                },
                message_type="learning_data"
            )
            
            return analysis
            
        except json.JSONDecodeError:
            return {
                "execution_success": "unknown",
                "effectiveness": "unknown",
                "performance_metrics": {},
                "issues_identified": [],
                "improvement_suggestions": [],
                "learning_points": [],
                "impact_on_subsequent_tasks": "unknown"
            }
            
    async def _get_sensor_data(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """获取传感器数据"""
        sensor_types = task.get("sensor_types", ["all"])
        
        if not self.ros2_node:
            return {"success": False, "error": "ROS2 node not initialized"}
            
        try:
            sensor_data = {}
            
            for sensor_type in sensor_types:
                if sensor_type == "all" or sensor_type == "camera":
                    sensor_data["camera"] = await self._get_camera_data()
                    
                if sensor_type == "all" or sensor_type == "lidar":
                    sensor_data["lidar"] = await self._get_lidar_data()
                    
                if sensor_type == "all" or sensor_type == "imu":
                    sensor_data["imu"] = await self._get_imu_data()
                    
                if sensor_type == "all" or sensor_type == "joints":
                    sensor_data["joints"] = await self._get_joint_states()
                    
            # 使用LLM分析传感器数据
            analysis = await self._analyze_sensor_data(sensor_data)
            
            # 发送给感知智能体进行进一步处理
            await self.send_message(
                recipient="PerceptionAgent",
                content={
                    "type": "visual_perception",
                    "sensor_data": sensor_data,
                    "analysis": analysis
                },
                message_type="sensor_data"
            )
            
            return {
                "success": True,
                "sensor_data": sensor_data,
                "analysis": analysis
            }
            
        except Exception as e:
            self.logger.error(f"Sensor data acquisition failed: {e}")
            return {
                "success": False,
                "error": f"Sensor data acquisition failed: {str(e)}"
            }
            
    async def _analyze_sensor_data(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析传感器数据"""
        analysis_prompt = f"""
分析以下传感器数据：

传感器数据：
{json.dumps(sensor_data, ensure_ascii=False, indent=2)}

请从ROS2智能体的角度分析：
1. 数据质量和完整性
2. 异常情况检测
3. 对机器人控制的影响
4. 安全相关的观察
5. 需要其他智能体关注的信息

返回JSON格式的分析结果：
{{
    "data_quality": {{
        "completeness": "数据完整性（0-1）",
        "reliability": "数据可靠性（0-1）",
        "latency": "数据延迟评估"
    }},
    "anomalies_detected": [
        {{
            "type": "异常类型",
            "severity": "严重程度",
            "description": "异常描述",
            "recommended_action": "建议行动"
        }}
    ],
    "control_implications": [
        "对机器人控制的影响"
    ],
    "safety_observations": [
        "安全相关观察"
    ],
    "alerts_for_other_agents": [
        {{
            "target_agent": "目标智能体",
            "alert_type": "警报类型",
            "message": "警报信息"
        }}
    ]
}}
"""
        
        analysis_message = BaseMessage.make_user_message(
            role_name="user",
            content=analysis_prompt
        )
        
        response = await self.agent.step(analysis_message)
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {
                "data_quality": {"completeness": 0.5, "reliability": 0.5},
                "anomalies_detected": [],
                "control_implications": [],
                "safety_observations": [],
                "alerts_for_other_agents": []
            }
            
    async def _emergency_stop(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """紧急停止"""
        reason = task.get("reason", "Emergency stop requested")
        
        try:
            # 立即停止所有运动
            stop_msg = Twist()
            if self.ros2_node:
                self.ros2_node.publish_velocity(stop_msg)
                
            # 更新安全状态
            self.safety_status["emergency_stop"] = True
            self.safety_status["safety_violations"].append({
                "timestamp": datetime.now().isoformat(),
                "reason": reason,
                "action": "emergency_stop"
            })
            self.safety_status["last_safety_check"] = datetime.now().isoformat()
            
            # 更新机器人状态
            self.robot_state["system_status"] = "emergency_stop"
            self.robot_state["velocity"] = {
                "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
                "angular": {"x": 0.0, "y": 0.0, "z": 0.0}
            }
            
            # 通知所有其他智能体
            await self.broadcast_message(
                content={
                    "alert_type": "emergency_stop",
                    "reason": reason,
                    "timestamp": datetime.now().isoformat(),
                    "robot_status": self.robot_state
                },
                message_type="emergency_alert",
                priority=3
            )
            
            self.logger.critical(f"Emergency stop activated: {reason}")
            
            return {
                "success": True,
                "message": "Emergency stop activated",
                "reason": reason
            }
            
        except Exception as e:
            self.logger.error(f"Emergency stop failed: {e}")
            return {
                "success": False,
                "error": f"Emergency stop failed: {str(e)}"
            }
            
    async def start(self):
        """启动ROS2智能体"""
        await super().start()
        
        # 启动状态监控
        asyncio.create_task(self._status_monitoring_loop())
        
        # 启动安全监控
        asyncio.create_task(self._safety_monitoring_loop())
        
    async def _status_monitoring_loop(self):
        """状态监控循环"""
        while self.state != AgentState.SHUTDOWN:
            try:
                # 更新机器人状态
                await self._update_robot_status()
                
                # 每5秒监控一次
                await asyncio.sleep(5.0)
                
            except Exception as e:
                self.logger.error(f"Status monitoring error: {e}")
                await asyncio.sleep(1.0)
                
    async def _safety_monitoring_loop(self):
        """安全监控循环"""
        while self.state != AgentState.SHUTDOWN:
            try:
                # 执行安全检查
                safety_check = await self._perform_safety_check()
                
                if not safety_check["safe"]:
                    await self._handle_safety_violation(safety_check)
                    
                # 每1秒检查一次
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Safety monitoring error: {e}")
                await asyncio.sleep(0.5)
                
    async def _perform_safety_check(self) -> Dict[str, Any]:
        """执行安全检查"""
        # 简化的安全检查实现
        # 实际实现中会检查传感器数据、系统状态等
        
        current_time = datetime.now()
        last_update = datetime.fromisoformat(self.robot_state["last_updated"])
        
        # 检查数据更新时间
        if (current_time - last_update).total_seconds() > 10:
            return {
                "safe": False,
                "reason": "Robot status data is stale",
                "severity": "medium"
            }
            
        # 检查紧急停止状态
        if self.safety_status["emergency_stop"]:
            return {
                "safe": False,
                "reason": "Emergency stop is active",
                "severity": "high"
            }
            
        return {
            "safe": True,
            "reason": "All safety checks passed",
            "severity": "none"
        }


class ROS2AgentNode(Node):
    """ROS2节点实现"""
    
    def __init__(self, ros2_agent):
        super().__init__('robot_agent_node')
        self.ros2_agent = ros2_agent
        
        # 创建发布者
        self.velocity_publisher = self.create_publisher(
            Twist, '/cmd_vel', 10
        )
        
        # 创建订阅者
        self.pose_subscriber = self.create_subscription(
            PoseStamped, '/robot_pose', self.pose_callback, 10
        )
        
        self.laser_subscriber = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10
        )
        
        self.image_subscriber = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        
        self.joint_subscriber = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10
        )
        
        # 状态数据
        self.latest_pose = None
        self.latest_laser = None
        self.latest_image = None
        self.latest_joints = None
        
    def publish_velocity(self, twist_msg: Twist):
        """发布速度指令"""
        self.velocity_publisher.publish(twist_msg)
        
    def pose_callback(self, msg: PoseStamped):
        """位置回调"""
        self.latest_pose = msg
        # 更新ROS2智能体的机器人状态
        self.ros2_agent.robot_state["position"] = {
            "x": msg.pose.position.x,
            "y": msg.pose.position.y,
            "z": msg.pose.position.z
        }
        self.ros2_agent.robot_state["orientation"] = {
            "x": msg.pose.orientation.x,
            "y": msg.pose.orientation.y,
            "z": msg.pose.orientation.z,
            "w": msg.pose.orientation.w
        }
        
    def laser_callback(self, msg: LaserScan):
        """激光雷达回调"""
        self.latest_laser = msg
        
    def image_callback(self, msg: Image):
        """图像回调"""
        self.latest_image = msg
        
    def joint_callback(self, msg: JointState):
        """关节状态回调"""
        self.latest_joints = msg
        joint_dict = {}
        for i, name in enumerate(msg.name):
            joint_dict[name] = {
                "position": msg.position[i] if i < len(msg.position) else 0.0,
                "velocity": msg.velocity[i] if i < len(msg.velocity) else 0.0,
                "effort": msg.effort[i] if i < len(msg.effort) else 0.0
            }
        self.ros2_agent.robot_state["joint_states"] = joint_dict
```

## 7. 消息总线实现

### 7.1 Redis消息总线

```python
"""
消息总线实现
位置：src/communication/message_bus.py
"""

import redis
import json
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum

class MessageType(Enum):
    """消息类型枚举"""
    TEXT = "text"
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    EMERGENCY_ALERT = "emergency_alert"
    SENSOR_DATA = "sensor_data"
    EXECUTION_PLAN = "execution_plan"
    LEARNING_DATA = "learning_data"
    BROADCAST = "broadcast"

class Message:
    """消息类"""
    
    def __init__(self, 
                 sender: str,
                 recipient: str,
                 content: Any,
                 message_type: MessageType = MessageType.TEXT,
                 priority: int = 1,
                 metadata: Dict[str, Any] = None):
        self.id = str(uuid.uuid4())
        self.sender = sender
        self.recipient = recipient
        self.content = content
        self.message_type = message_type
        self.priority = priority
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "message_type": self.message_type.value,
            "priority": self.priority,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """从字典创建消息"""
        msg = cls(
            sender=data["sender"],
            recipient=data["recipient"],
            content=data["content"],
            message_type=MessageType(data["message_type"]),
            priority=data.get("priority", 1),
            metadata=data.get("metadata", {})
        )
        msg.id = data["id"]
        msg.timestamp = data["timestamp"]
        return msg

class MessageBus:
    """基于Redis的消息总线"""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379, redis_db: int = 0):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True
        )
        self.pubsub = self.redis_client.pubsub()
        self.subscribers = {}  # agent_name -> callback
        self.message_history = []
        self.running = False
        
    async def start(self):
        """启动消息总线"""
        self.running = True
        # 启动消息监听循环
        asyncio.create_task(self._message_listener_loop())
        
    async def stop(self):
        """停止消息总线"""
        self.running = False
        self.pubsub.close()
        
    def subscribe(self, agent_name: str, callback: Callable):
        """订阅消息"""
        self.subscribers[agent_name] = callback
        # 订阅Redis频道
        self.pubsub.subscribe(f"agent:{agent_name}")
        self.pubsub.subscribe("broadcast")
        
    async def send_message(self,
                          sender: str,
                          recipient: str,
                          content: Any,
                          message_type: str = "text",
                          priority: int = 1,
                          metadata: Dict[str, Any] = None) -> str:
        """发送消息"""
        message = Message(
            sender=sender,
            recipient=recipient,
            content=content,
            message_type=MessageType(message_type),
            priority=priority,
            metadata=metadata
        )
        
        # 发布到Redis
        channel = f"agent:{recipient}"
        message_data = json.dumps(message.to_dict())
        self.redis_client.publish(channel, message_data)
        
        # 存储消息历史
        self.message_history.append(message)
        
        return message.id
        
    async def broadcast_message(self,
                               sender: str,
                               content: Any,
                               message_type: str = "broadcast",
                               priority: int = 1,
                               metadata: Dict[str, Any] = None) -> str:
        """广播消息"""
        message = Message(
            sender=sender,
            recipient="all",
            content=content,
            message_type=MessageType(message_type),
            priority=priority,
            metadata=metadata
        )
        
        # 广播到所有智能体
        message_data = json.dumps(message.to_dict())
        self.redis_client.publish("broadcast", message_data)
        
        # 存储消息历史
        self.message_history.append(message)
        
        return message.id
        
    async def _message_listener_loop(self):
        """消息监听循环"""
        while self.running:
            try:
                # 获取消息
                message = self.pubsub.get_message(timeout=1.0)
                
                if message and message['type'] == 'message':
                    await self._handle_received_message(message)
                    
            except Exception as e:
                print(f"Message listener error: {e}")
                await asyncio.sleep(0.1)
                
    async def _handle_received_message(self, redis_message):
        """处理接收到的消息"""
        try:
            # 解析消息
            message_data = json.loads(redis_message['data'])
            message = Message.from_dict(message_data)
            
            # 路由消息
            if redis_message['channel'] == 'broadcast':
                # 广播消息，发送给所有订阅者
                for agent_name, callback in self.subscribers.items():
                    if agent_name != message.sender:  # 不发送给发送者自己
                        await callback(message)
            else:
                # 点对点消息
                channel_parts = redis_message['channel'].split(':')
                if len(channel_parts) == 2:
                    agent_name = channel_parts[1]
                    if agent_name in self.subscribers:
                        await self.subscribers[agent_name](message)
                        
        except Exception as e:
            print(f"Error handling received message: {e}")
```
