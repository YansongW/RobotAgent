# -*- coding: utf-8 -*-

# 动作智能体 (Action Agent)
# 专注于动作规划、任务分解和执行逻辑的智能体实现
# 作者: RobotAgent开发团队
# 版本: 0.0.1 (Initial Release)
# 更新时间: 2025-08-21

# 导入标准库
import asyncio
import uuid
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass, asdict

# 导入项目基础组件
from .base_agent import BaseRobotAgent
from config import (
    MessageType, AgentMessage, MessagePriority, TaskMessage, ResponseMessage,
    TaskStatus
)
from src.communication.protocols import (
    CollaborationMode, StatusMessage, MemoryMessage
)
from src.communication.message_bus import get_message_bus

# 导入CAMEL框架组件
try:
    from camel.agents import ChatAgent
    from camel.messages import BaseMessage
    from camel.models import ModelFactory
    CAMEL_AVAILABLE = True
except ImportError:
    CAMEL_AVAILABLE = False
    import logging
    logging.warning("CAMEL框架未安装，使用模拟实现")


class TaskType(Enum):
    """任务类型枚举"""
    SIMPLE = "simple"  # 简单任务
    COMPLEX = "complex"  # 复杂任务
    PARALLEL = "parallel"  # 并行任务
    SEQUENTIAL = "sequential"  # 顺序任务


class ExecutionStatus(Enum):
    """执行状态枚举"""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PAUSED = "PAUSED"
    CANCELLED = "CANCELLED"


@dataclass
class SubTask:
    """子任务数据结构"""
    id: str
    description: str
    dependencies: List[str]
    tools_required: List[str]
    estimated_time: str
    status: ExecutionStatus
    priority: int = 1
    retry_count: int = 0
    max_retries: int = 3
    result: Optional[Any] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class TaskTree:
    """任务树数据结构"""
    root_task: str
    task_id: str
    subtasks: List[SubTask]
    task_type: TaskType
    total_estimated_time: str
    created_at: datetime
    status: ExecutionStatus = ExecutionStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "task_id": self.task_id,
            "root_task": self.root_task,
            "task_type": self.task_type.value,
            "status": self.status.value,
            "total_estimated_time": self.total_estimated_time,
            "created_at": self.created_at.isoformat(),
            "subtasks": [{
                "id": task.id,
                "description": task.description,
                "dependencies": task.dependencies,
                "tools_required": task.tools_required,
                "estimated_time": task.estimated_time,
                "status": task.status.value,
                "priority": task.priority,
                "retry_count": task.retry_count,
                "result": task.result,
                "error_message": task.error_message,
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None
            } for task in self.subtasks]
        }


@dataclass
class ExecutionPlan:
    """执行计划数据结构"""
    sequence: List[str]  # 顺序执行的任务ID列表
    parallel_groups: List[List[str]]  # 并行执行的任务组
    estimated_total_time: str
    optimization_strategy: str = "time_optimal"  # 优化策略


@dataclass
class ExecutionResult:
    """执行结果数据结构"""
    status: ExecutionStatus
    completed_tasks: List[str]
    failed_tasks: List[str]
    output_data: Any
    execution_time: float
    error_details: Optional[Dict[str, str]] = None
    performance_metrics: Optional[Dict[str, Any]] = None


class ActionAgent(BaseRobotAgent):
    """动作执行智能体
    
    负责复杂任务的分解、规划和执行，是系统的任务执行核心。
    主要功能包括：
    - 任务分解与建模
    - 动作序列规划
    - 工具调用与集成
    - 执行状态监控
    - 异常处理与恢复
    - 结果验证与反馈
    """

    def __init__(self, agent_id: str = "action_agent", config: Optional[Dict[str, Any]] = None):
        """初始化ActionAgent
        
        Args:
            agent_id: 智能体ID
            config: 配置参数
        """
        super().__init__(agent_id, "action", config)
        
        # 任务管理
        self.active_tasks: Dict[str, TaskTree] = {}
        self.task_history: List[TaskTree] = []
        
        # 执行状态
        self.execution_queue: List[str] = []
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # 性能监控
        self.performance_metrics = {
            "total_tasks_executed": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "average_execution_time": 0.0,
            "tool_usage_stats": {}
        }
        
        # 工具映射
        self.tool_mapping = {
            "file_operation": self._handle_file_operation,
            "api_call": self._handle_api_call,
            "data_processing": self._handle_data_processing,
            "system_command": self._handle_system_command
        }
        
        self.logger.info(f"ActionAgent {agent_id} 初始化完成")

    async def execute_task(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行任务的主入口方法
        
        Args:
            task_description: 任务描述
            context: 任务上下文
            
        Returns:
            执行结果字典
        """
        try:
            self.logger.info(f"开始执行任务: {task_description}")
            
            # 1. 任务分解
            task_tree = await self._decompose_task(task_description, context)
            
            # 2. 制定执行计划
            execution_plan = await self._create_execution_plan(task_tree)
            
            # 3. 执行任务
            execution_result = await self._execute_task_tree(task_tree, execution_plan)
            
            # 4. 生成响应
            response = {
                "task_tree": task_tree.to_dict(),
                "execution_plan": asdict(execution_plan),
                "execution_result": asdict(execution_result)
            }
            
            # 5. 更新性能指标
            await self._update_performance_metrics(execution_result)
            
            # 6. 存储到记忆系统
            await self._store_execution_memory(task_tree, execution_result)
            
            self.logger.info(f"任务执行完成: {task_description}")
            return response
            
        except Exception as e:
            self.logger.error(f"任务执行失败: {str(e)}")
            return {
                "error": str(e),
                "status": "FAILED",
                "task_description": task_description
            }

    async def _decompose_task(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> TaskTree:
        """任务分解
        
        将复杂任务分解为可执行的子任务序列
        """
        task_id = str(uuid.uuid4())
        
        # 分析任务复杂度和类型
        task_type = await self._analyze_task_type(task_description)
        
        # 生成子任务
        subtasks = await self._generate_subtasks(task_description, task_type, context)
        
        # 估算总时间
        total_time = self._estimate_total_time(subtasks)
        
        task_tree = TaskTree(
            task_id=task_id,
            root_task=task_description,
            task_type=task_type,
            subtasks=subtasks,
            total_estimated_time=total_time,
            created_at=datetime.now()
        )
        
        self.active_tasks[task_id] = task_tree
        self.logger.info(f"任务分解完成，生成 {len(subtasks)} 个子任务")
        
        return task_tree

    async def _analyze_task_type(self, task_description: str) -> TaskType:
        """分析任务类型"""
        # 简单的任务类型判断逻辑
        if "并行" in task_description or "同时" in task_description:
            return TaskType.PARALLEL
        elif "顺序" in task_description or "依次" in task_description:
            return TaskType.SEQUENTIAL
        elif len(task_description.split()) > 10:  # 复杂任务判断
            return TaskType.COMPLEX
        else:
            return TaskType.SIMPLE

    async def _generate_subtasks(self, task_description: str, task_type: TaskType, context: Optional[Dict[str, Any]]) -> List[SubTask]:
        """生成子任务列表"""
        subtasks = []
        
        # 基于任务描述生成子任务（这里是简化版本，实际应该使用更复杂的NLP分析）
        if "文件" in task_description:
            subtasks.append(SubTask(
                id=f"subtask_{len(subtasks)+1}",
                description="文件操作任务",
                dependencies=[],
                tools_required=["file_operation"],
                estimated_time="2min",
                status=ExecutionStatus.PENDING
            ))
        
        if "API" in task_description or "接口" in task_description:
            subtasks.append(SubTask(
                id=f"subtask_{len(subtasks)+1}",
                description="API调用任务",
                dependencies=[],
                tools_required=["api_call"],
                estimated_time="1min",
                status=ExecutionStatus.PENDING
            ))
        
        if "数据" in task_description or "处理" in task_description:
            subtasks.append(SubTask(
                id=f"subtask_{len(subtasks)+1}",
                description="数据处理任务",
                dependencies=[],
                tools_required=["data_processing"],
                estimated_time="3min",
                status=ExecutionStatus.PENDING
            ))
        
        # 如果没有识别到具体任务，创建一个通用任务
        if not subtasks:
            subtasks.append(SubTask(
                id="subtask_1",
                description=f"执行任务: {task_description}",
                dependencies=[],
                tools_required=["system_command"],
                estimated_time="5min",
                status=ExecutionStatus.PENDING
            ))
        
        return subtasks

    async def _create_execution_plan(self, task_tree: TaskTree) -> ExecutionPlan:
        """制定执行计划"""
        # 分析依赖关系
        sequence = []
        parallel_groups = []
        
        # 简化的执行计划生成
        independent_tasks = [task for task in task_tree.subtasks if not task.dependencies]
        dependent_tasks = [task for task in task_tree.subtasks if task.dependencies]
        
        if len(independent_tasks) > 1 and task_tree.task_type == TaskType.PARALLEL:
            parallel_groups.append([task.id for task in independent_tasks])
        else:
            sequence.extend([task.id for task in independent_tasks])
        
        sequence.extend([task.id for task in dependent_tasks])
        
        return ExecutionPlan(
            sequence=sequence,
            parallel_groups=parallel_groups,
            estimated_total_time=task_tree.total_estimated_time
        )

    async def _execute_task_tree(self, task_tree: TaskTree, execution_plan: ExecutionPlan) -> ExecutionResult:
        """执行任务树"""
        start_time = datetime.now()
        completed_tasks = []
        failed_tasks = []
        output_data = {}
        
        try:
            task_tree.status = ExecutionStatus.IN_PROGRESS
            
            # 执行并行任务组
            for parallel_group in execution_plan.parallel_groups:
                await self._execute_parallel_tasks(task_tree, parallel_group, completed_tasks, failed_tasks, output_data)
            
            # 执行顺序任务
            for task_id in execution_plan.sequence:
                await self._execute_single_task(task_tree, task_id, completed_tasks, failed_tasks, output_data)
            
            # 确定最终状态
            if failed_tasks:
                final_status = ExecutionStatus.FAILED if len(failed_tasks) == len(task_tree.subtasks) else ExecutionStatus.COMPLETED
            else:
                final_status = ExecutionStatus.COMPLETED
                
            task_tree.status = final_status
            
        except Exception as e:
            self.logger.error(f"任务执行异常: {str(e)}")
            task_tree.status = ExecutionStatus.FAILED
            final_status = ExecutionStatus.FAILED
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return ExecutionResult(
            status=final_status,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            output_data=output_data,
            execution_time=execution_time
        )

    async def _execute_parallel_tasks(self, task_tree: TaskTree, task_ids: List[str], completed: List[str], failed: List[str], output: Dict[str, Any]):
        """并行执行任务"""
        tasks = []
        for task_id in task_ids:
            task = asyncio.create_task(self._execute_single_task(task_tree, task_id, completed, failed, output))
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_single_task(self, task_tree: TaskTree, task_id: str, completed: List[str], failed: List[str], output: Dict[str, Any]):
        """执行单个子任务"""
        subtask = next((task for task in task_tree.subtasks if task.id == task_id), None)
        if not subtask:
            return
        
        try:
            subtask.status = ExecutionStatus.IN_PROGRESS
            subtask.started_at = datetime.now()
            
            # 检查依赖
            if not await self._check_dependencies(subtask, completed):
                raise Exception(f"依赖任务未完成: {subtask.dependencies}")
            
            # 执行任务
            result = await self._execute_subtask(subtask)
            
            subtask.result = result
            subtask.status = ExecutionStatus.COMPLETED
            subtask.completed_at = datetime.now()
            completed.append(task_id)
            output[task_id] = result
            
            self.logger.info(f"子任务完成: {subtask.description}")
            
        except Exception as e:
            subtask.status = ExecutionStatus.FAILED
            subtask.error_message = str(e)
            subtask.retry_count += 1
            failed.append(task_id)
            
            self.logger.error(f"子任务失败: {subtask.description}, 错误: {str(e)}")
            
            # 重试逻辑
            if subtask.retry_count < subtask.max_retries:
                self.logger.info(f"重试子任务: {subtask.description} (第{subtask.retry_count}次)")
                await asyncio.sleep(1)  # 等待1秒后重试
                await self._execute_single_task(task_tree, task_id, completed, failed, output)

    async def _check_dependencies(self, subtask: SubTask, completed: List[str]) -> bool:
        """检查任务依赖"""
        return all(dep in completed for dep in subtask.dependencies)

    async def _execute_subtask(self, subtask: SubTask) -> Any:
        """执行具体的子任务"""
        results = []
        
        for tool_name in subtask.tools_required:
            if tool_name in self.tool_mapping:
                result = await self.tool_mapping[tool_name](subtask)
                results.append(result)
            else:
                self.logger.warning(f"未知工具: {tool_name}")
        
        return results if len(results) > 1 else (results[0] if results else None)

    async def _handle_file_operation(self, subtask: SubTask) -> str:
        """处理文件操作"""
        # 模拟文件操作
        await asyncio.sleep(0.1)
        return f"文件操作完成: {subtask.description}"

    async def _handle_api_call(self, subtask: SubTask) -> str:
        """处理API调用"""
        # 模拟API调用
        await asyncio.sleep(0.2)
        return f"API调用完成: {subtask.description}"

    async def _handle_data_processing(self, subtask: SubTask) -> str:
        """处理数据处理"""
        # 模拟数据处理
        await asyncio.sleep(0.3)
        return f"数据处理完成: {subtask.description}"

    async def _handle_system_command(self, subtask: SubTask) -> str:
        """处理系统命令"""
        # 模拟系统命令执行
        await asyncio.sleep(0.1)
        return f"系统命令执行完成: {subtask.description}"

    def _estimate_total_time(self, subtasks: List[SubTask]) -> str:
        """估算总执行时间"""
        total_minutes = sum(int(task.estimated_time.replace('min', '')) for task in subtasks)
        return f"{total_minutes}min"

    async def _update_performance_metrics(self, execution_result: ExecutionResult):
        """更新性能指标"""
        self.performance_metrics["total_tasks_executed"] += 1
        
        if execution_result.status == ExecutionStatus.COMPLETED:
            self.performance_metrics["successful_tasks"] += 1
        else:
            self.performance_metrics["failed_tasks"] += 1
        
        # 更新平均执行时间
        total_tasks = self.performance_metrics["total_tasks_executed"]
        current_avg = self.performance_metrics["average_execution_time"]
        new_avg = (current_avg * (total_tasks - 1) + execution_result.execution_time) / total_tasks
        self.performance_metrics["average_execution_time"] = new_avg

    async def _store_execution_memory(self, task_tree: TaskTree, execution_result: ExecutionResult):
        """存储执行记忆到MemoryAgent"""
        try:
            memory_data = {
                "type": "task_execution",
                "task_description": task_tree.root_task,
                "execution_result": asdict(execution_result),
                "task_tree": task_tree.to_dict(),
                "timestamp": datetime.now().isoformat()
            }
            
            # 发送消息给MemoryAgent存储记忆
            memory_message = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                receiver_id="memory_agent",
                message_type=MessageType.TASK,
                content=json.dumps(memory_data),
                timestamp=datetime.now()
            )
            
            await self.send_message(memory_message)
            
        except Exception as e:
            self.logger.error(f"存储执行记忆失败: {str(e)}")

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id].to_dict()
        return None

    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if task_id in self.active_tasks:
            task_tree = self.active_tasks[task_id]
            task_tree.status = ExecutionStatus.CANCELLED
            
            # 取消正在运行的任务
            if task_id in self.running_tasks:
                self.running_tasks[task_id].cancel()
                del self.running_tasks[task_id]
            
            self.logger.info(f"任务已取消: {task_id}")
            return True
        return False

    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return self.performance_metrics.copy()

    async def _handle_collaboration_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """处理协作请求"""
        try:
            request_data = json.loads(message.content)
            request_type = request_data.get("type")
            
            if request_type == "execute_task":
                task_description = request_data.get("task_description")
                context = request_data.get("context", {})
                
                result = await self.execute_task(task_description, context)
                
                response = AgentMessage(
                    message_id=str(uuid.uuid4()),
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    message_type=MessageType.RESPONSE,
                    content=json.dumps(result),
                    timestamp=datetime.now(),
                    correlation_id=message.message_id
                )
                
                return response
                
        except Exception as e:
            self.logger.error(f"处理协作请求失败: {str(e)}")
        
        return None