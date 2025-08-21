# -*- coding: utf-8 -*-

# 智能体协调器 (Agent Coordinator)
# 负责管理和协调ChatAgent、ActionAgent、MemoryAgent之间的协作
# 作者: RobotAgent开发团队
# 版本: 0.0.1 (Initial Release)
# 更新时间: 2025-08-19

# 导入标准库
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# 导入项目基础组件
from .chat_agent import ChatAgent
from .action_agent import ActionAgent
from .memory_agent import MemoryAgent
from config import (
    MessageType, AgentMessage, MessagePriority, TaskMessage, ResponseMessage,
    create_task_message
)
from src.communication.protocols import (
    CollaborationMode, StatusMessage, CollaborationRequest,
    CollaborationResponse, MemoryMessage,
    create_collaboration_request, create_memory_message
)
from src.communication.message_bus import get_message_bus, MessageBus


class CoordinationMode(Enum):
    
    # 协调模式枚举 (Coordination Mode Enum)
    
    # 定义智能体协调器支持的四种协作模式：
    # 1. 顺序协作 - 智能体按顺序依次处理任务
    # 2. 并行协作 - 多个智能体同时处理不同子任务
    # 3. 流水线协作 - 任务在智能体间流水线传递
    # 4. 自适应协作 - 根据任务特性动态选择协作模式
    
    SEQUENTIAL = "sequential"  # 顺序协作
    PARALLEL = "parallel"     # 并行协作
    PIPELINE = "pipeline"     # 流水线协作
    ADAPTIVE = "adaptive"     # 自适应协作


@dataclass
class CoordinationTask:
    
    # 协调任务数据类 (Coordination Task Data Class)
    
    # 封装协调器处理的任务信息，包括：
    # 1. 任务标识和用户输入
    # 2. 协调模式和优先级
    # 3. 上下文信息和时间戳
    
    # 任务属性定义
    task_id: str                                    # 任务唯一标识符
    user_input: str                                 # 用户输入内容
    mode: CoordinationMode                          # 协调模式
    priority: MessagePriority = MessagePriority.MEDIUM  # 任务优先级
    context: Dict[str, Any] = None                  # 任务上下文信息
    created_at: datetime = None                     # 任务创建时间
    
    def __post_init__(self):
        # 初始化默认值
        # 确保任务创建时间和上下文信息的正确初始化
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.context is None:
            self.context = {}


class AgentCoordinator:
    """
    智能体协调器 (Agent Coordinator)
    
    负责协调ChatAgent、ActionAgent和MemoryAgent之间的协作，专注于：
    1. 统一的任务分发和路由管理
    2. 多智能体协作模式控制
    3. 结果整合和状态同步
    4. 性能监控和错误处理
    
    支持四种协作模式：顺序、并行、流水线、自适应。
    通过消息总线实现智能体间的通信协调。
    
    Attributes:
        message_bus (MessageBus): 消息总线实例
        chat_agent (ChatAgent): 对话智能体实例
        action_agent (ActionAgent): 动作智能体实例
        memory_agent (MemoryAgent): 记忆智能体实例
        active_tasks (Dict): 当前活跃任务字典
        coordination_stats (Dict): 协调性能统计信息
        
    Example:
        >>> coordinator = AgentCoordinator()
        >>> await coordinator.initialize()
        >>> result = await coordinator.process_user_input("Hello", CoordinationMode.SEQUENTIAL)
        >>> await coordinator.shutdown()
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化智能体协调器
        
        Args:
            config: 配置参数字典，包含各智能体的配置信息
                    格式: {
                        'chat_agent': {...},
                        'action_agent': {...},
                        'memory_agent': {...}
                    }
        
        Raises:
            ValueError: 当配置参数无效时
            ImportError: 当依赖组件未安装时
        """
        # 设置基础配置和日志
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化消息总线
        self.message_bus: MessageBus = get_message_bus()
        
        # 初始化智能体实例（延迟创建）
        self.chat_agent: Optional[ChatAgent] = None
        self.action_agent: Optional[ActionAgent] = None
        self.memory_agent: Optional[MemoryAgent] = None
        
        # 初始化协调状态管理
        self.is_running = False
        self.active_tasks: Dict[str, CoordinationTask] = {}
        self.task_results: Dict[str, Dict[str, Any]] = {}
        
        # 初始化性能统计
        self.coordination_stats = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'average_response_time': 0.0
        }
    
    async def initialize(self):
        """
        初始化协调器和所有智能体
        
        按顺序执行以下初始化步骤：
        1. 启动消息总线服务
        2. 创建三个智能体实例
        3. 启动所有智能体
        4. 设置运行状态
        
        Raises:
            RuntimeError: 当智能体启动失败时
            ValueError: 当配置参数无效时
        """
        try:
            self.logger.info("初始化智能体协调器")
            
            # 启动消息总线服务
            await self.message_bus.start()
            
            # 创建对话智能体实例
            self.chat_agent = ChatAgent(
                agent_id="chat_agent",
                agent_type="chat",
                config=self.config.get('chat_agent', {})
            )
            
            # 创建动作智能体实例
            self.action_agent = ActionAgent(
                agent_id="action_agent",
                agent_type="action",
                config=self.config.get('action_agent', {})
            )
            
            # 创建记忆智能体实例
            self.memory_agent = MemoryAgent(
                agent_id="memory_agent",
                agent_type="memory",
                config=self.config.get('memory_agent', {})
            )
            
            # 启动所有智能体服务
            await self.chat_agent.start()
            await self.action_agent.start()
            await self.memory_agent.start()
            
            # 设置协调器运行状态
            self.is_running = True
            self.logger.info("智能体协调器初始化完成")
            
        except Exception as e:
            # 初始化失败处理
            self.logger.error(f"协调器初始化失败: {e}")
            raise
    
    async def shutdown(self):
        """关闭协调器和所有智能体"""
        try:
            self.logger.info("关闭智能体协调器")
            self.is_running = False
            
            # 停止所有智能体
            if self.chat_agent:
                await self.chat_agent.stop()
            if self.action_agent:
                await self.action_agent.stop()
            if self.memory_agent:
                await self.memory_agent.stop()
            
            # 停止消息总线
            await self.message_bus.stop()
            
            self.logger.info("智能体协调器已关闭")
            
        except Exception as e:
            self.logger.error(f"协调器关闭失败: {e}")
    
    async def process_user_input(self, 
                               user_input: str, 
                               mode: CoordinationMode = CoordinationMode.ADAPTIVE,
                               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理用户输入
        
        Args:
            user_input: 用户输入
            mode: 协调模式
            context: 上下文信息
            
        Returns:
            处理结果
        """
        if not self.is_running:
            raise RuntimeError("协调器未启动")
        
        # 创建协调任务
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        coordination_task = CoordinationTask(
            task_id=task_id,
            user_input=user_input,
            mode=mode,
            context=context or {}
        )
        
        self.active_tasks[task_id] = coordination_task
        self.coordination_stats['total_tasks'] += 1
        
        start_time = datetime.now()
        
        try:
            # 根据协调模式处理任务
            if mode == CoordinationMode.SEQUENTIAL:
                result = await self._process_sequential(coordination_task)
            elif mode == CoordinationMode.PARALLEL:
                result = await self._process_parallel(coordination_task)
            elif mode == CoordinationMode.PIPELINE:
                result = await self._process_pipeline(coordination_task)
            else:  # ADAPTIVE
                result = await self._process_adaptive(coordination_task)
            
            # 更新统计信息
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            self._update_stats(True, response_time)
            
            # 存储结果
            self.task_results[task_id] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"任务处理失败 {task_id}: {e}")
            self._update_stats(False, 0)
            raise
        finally:
            # 清理活跃任务
            self.active_tasks.pop(task_id, None)
    
    async def _process_sequential(self, task: CoordinationTask) -> Dict[str, Any]:
        """顺序协作模式"""
        self.logger.info(f"开始顺序协作处理任务: {task.task_id}")
        
        # 1. ChatAgent 理解和分析用户输入
        chat_message = create_task_message(
            sender="coordinator",
            recipient="chat_agent",
            task_type="analyze_input",
            task_data={
                "user_input": task.user_input,
                "context": task.context
            },
            priority=task.priority
        )
        
        await self.message_bus.send_message(chat_message)
        chat_response = await self._wait_for_response("chat_agent", timeout=30)
        
        if not chat_response or chat_response.get('status') != 'success':
            raise RuntimeError("ChatAgent处理失败")
        
        # 2. 根据分析结果决定是否需要ActionAgent
        analysis_result = chat_response.get('data', {})
        if analysis_result.get('requires_action', False):
            # ActionAgent 执行动作
            action_message = create_task_message(
                sender="coordinator",
                recipient="action_agent",
                task_type="execute_action",
                task_data={
                    "action_plan": analysis_result.get('action_plan'),
                    "context": task.context
                },
                priority=task.priority
            )
            
            await self.message_bus.send_message(action_message)
            action_response = await self._wait_for_response("action_agent", timeout=60)
            
            if action_response and action_response.get('status') == 'success':
                analysis_result['action_result'] = action_response.get('data')
        
        # 3. MemoryAgent 存储交互记忆
        memory_message = create_memory_message(
            sender="coordinator",
            recipient="memory_agent",
            operation="store",
            memory_data={
                "user_input": task.user_input,
                "analysis_result": analysis_result,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        await self.message_bus.send_message(memory_message)
        
        # 4. ChatAgent 生成最终响应
        response_message = create_task_message(
            sender="coordinator",
            recipient="chat_agent",
            task_type="generate_response",
            task_data={
                "analysis_result": analysis_result,
                "context": task.context
            },
            priority=task.priority
        )
        
        await self.message_bus.send_message(response_message)
        final_response = await self._wait_for_response("chat_agent", timeout=30)
        
        return {
            'status': 'success',
            'mode': 'sequential',
            'task_id': task.task_id,
            'response': final_response.get('data', {}).get('response', ''),
            'analysis': analysis_result,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _process_parallel(self, task: CoordinationTask) -> Dict[str, Any]:
        """并行协作模式"""
        self.logger.info(f"开始并行协作处理任务: {task.task_id}")
        
        # 并行发送任务给多个智能体
        tasks = []
        
        # ChatAgent 分析
        chat_task = asyncio.create_task(
            self._send_and_wait(
                "chat_agent",
                create_task_message(
                    sender="coordinator",
                    recipient="chat_agent",
                    task_type="analyze_input",
                    task_data={"user_input": task.user_input, "context": task.context},
                    priority=task.priority
                ),
                timeout=30
            )
        )
        tasks.append(('chat', chat_task))
        
        # MemoryAgent 检索相关记忆
        memory_task = asyncio.create_task(
            self._send_and_wait(
                "memory_agent",
                create_memory_message(
                    sender="coordinator",
                    recipient="memory_agent",
                    operation="retrieve",
                    memory_data={"query": task.user_input}
                ),
                timeout=20
            )
        )
        tasks.append(('memory', memory_task))
        
        # 等待所有任务完成
        results = {}
        for name, task_coro in tasks:
            try:
                result = await task_coro
                results[name] = result
            except Exception as e:
                self.logger.error(f"{name}任务失败: {e}")
                results[name] = {'status': 'error', 'error': str(e)}
        
        return {
            'status': 'success',
            'mode': 'parallel',
            'task_id': task.task_id,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _process_pipeline(self, task: CoordinationTask) -> Dict[str, Any]:
        """流水线协作模式"""
        self.logger.info(f"开始流水线协作处理任务: {task.task_id}")
        
        # 流水线处理：Memory -> Chat -> Action -> Chat
        pipeline_data = {
            'user_input': task.user_input,
            'context': task.context
        }
        
        # 阶段1: MemoryAgent 检索相关记忆
        memory_result = await self._send_and_wait(
            "memory_agent",
            create_memory_message(
                sender="coordinator",
                recipient="memory_agent",
                operation="retrieve",
                memory_data={"query": task.user_input}
            ),
            timeout=20
        )
        
        if memory_result and memory_result.get('status') == 'success':
            pipeline_data['retrieved_memories'] = memory_result.get('data', {})
        
        # 阶段2: ChatAgent 分析（包含记忆信息）
        chat_result = await self._send_and_wait(
            "chat_agent",
            create_task_message(
                sender="coordinator",
                recipient="chat_agent",
                task_type="analyze_with_memory",
                task_data=pipeline_data,
                priority=task.priority
            ),
            timeout=30
        )
        
        if chat_result and chat_result.get('status') == 'success':
            pipeline_data['analysis'] = chat_result.get('data', {})
        
        # 阶段3: ActionAgent 执行（如果需要）
        if pipeline_data.get('analysis', {}).get('requires_action', False):
            action_result = await self._send_and_wait(
                "action_agent",
                create_task_message(
                    sender="coordinator",
                    recipient="action_agent",
                    task_type="execute_pipeline_action",
                    task_data=pipeline_data,
                    priority=task.priority
                ),
                timeout=60
            )
            
            if action_result and action_result.get('status') == 'success':
                pipeline_data['action_result'] = action_result.get('data', {})
        
        # 阶段4: ChatAgent 生成最终响应
        final_result = await self._send_and_wait(
            "chat_agent",
            create_task_message(
                sender="coordinator",
                recipient="chat_agent",
                task_type="generate_final_response",
                task_data=pipeline_data,
                priority=task.priority
            ),
            timeout=30
        )
        
        return {
            'status': 'success',
            'mode': 'pipeline',
            'task_id': task.task_id,
            'response': final_result.get('data', {}).get('response', ''),
            'pipeline_data': pipeline_data,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _process_adaptive(self, task: CoordinationTask) -> Dict[str, Any]:
        """自适应协作模式"""
        self.logger.info(f"开始自适应协作处理任务: {task.task_id}")
        
        # 首先让ChatAgent分析任务复杂度和需求
        analysis_result = await self._send_and_wait(
            "chat_agent",
            create_task_message(
                sender="coordinator",
                recipient="chat_agent",
                task_type="analyze_complexity",
                task_data={
                    "user_input": task.user_input,
                    "context": task.context
                },
                priority=task.priority
            ),
            timeout=20
        )
        
        if not analysis_result or analysis_result.get('status') != 'success':
            # 降级到顺序模式
            return await self._process_sequential(task)
        
        complexity = analysis_result.get('data', {}).get('complexity', 'medium')
        requires_action = analysis_result.get('data', {}).get('requires_action', False)
        requires_memory = analysis_result.get('data', {}).get('requires_memory', False)
        
        # 根据分析结果选择最适合的协作模式
        if complexity == 'low' and not requires_action:
            # 简单任务，直接ChatAgent处理
            return await self._process_simple_chat(task)
        elif complexity == 'high' or (requires_action and requires_memory):
            # 复杂任务，使用流水线模式
            return await self._process_pipeline(task)
        else:
            # 中等复杂度，使用并行模式
            return await self._process_parallel(task)
    
    async def _process_simple_chat(self, task: CoordinationTask) -> Dict[str, Any]:
        """简单聊天处理"""
        result = await self._send_and_wait(
            "chat_agent",
            create_task_message(
                sender="coordinator",
                recipient="chat_agent",
                task_type="simple_chat",
                task_data={
                    "user_input": task.user_input,
                    "context": task.context
                },
                priority=task.priority
            ),
            timeout=30
        )
        
        return {
            'status': 'success',
            'mode': 'simple_chat',
            'task_id': task.task_id,
            'response': result.get('data', {}).get('response', ''),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _send_and_wait(self, 
                           recipient: str, 
                           message: AgentMessage, 
                           timeout: float = 30) -> Optional[Dict[str, Any]]:
        """发送消息并等待响应"""
        await self.message_bus.send_message(message)
        return await self._wait_for_response(recipient, timeout)
    
    async def _wait_for_response(self, 
                               agent_id: str, 
                               timeout: float = 30) -> Optional[Dict[str, Any]]:
        """等待智能体响应"""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < timeout:
            try:
                # 从消息总线接收响应
                response = await self.message_bus.receive_message(
                    "coordinator", 
                    timeout=1.0
                )
                
                if response and response.sender == agent_id:
                    if isinstance(response, ResponseMessage):
                        return {
                            'status': response.status,
                            'data': response.data,
                            'error': response.error
                        }
                    elif hasattr(response, 'content'):
                        return {
                            'status': 'success',
                            'data': response.content
                        }
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"等待响应时出错: {e}")
                break
        
        self.logger.warning(f"等待 {agent_id} 响应超时")
        return None
    
    def _update_stats(self, success: bool, response_time: float):
        """更新统计信息"""
        if success:
            self.coordination_stats['successful_tasks'] += 1
        else:
            self.coordination_stats['failed_tasks'] += 1
        
        # 更新平均响应时间
        total_successful = self.coordination_stats['successful_tasks']
        if total_successful > 0:
            current_avg = self.coordination_stats['average_response_time']
            new_avg = ((current_avg * (total_successful - 1)) + response_time) / total_successful
            self.coordination_stats['average_response_time'] = new_avg
    
    def get_stats(self) -> Dict[str, Any]:
        """获取协调器统计信息"""
        return {
            'coordination_stats': self.coordination_stats.copy(),
            'active_tasks': len(self.active_tasks),
            'message_bus_stats': self.message_bus.get_stats() if self.message_bus else {},
            'agents_status': {
                'chat_agent': self.chat_agent.get_state() if self.chat_agent else 'not_initialized',
                'action_agent': self.action_agent.get_state() if self.action_agent else 'not_initialized',
                'memory_agent': self.memory_agent.get_state() if self.memory_agent else 'not_initialized'
            }
        }


# 全局协调器实例
_coordinator_instance: Optional[AgentCoordinator] = None


def get_coordinator(config: Dict[str, Any] = None) -> AgentCoordinator:
    """获取全局协调器实例"""
    global _coordinator_instance
    if _coordinator_instance is None:
        _coordinator_instance = AgentCoordinator(config)
    return _coordinator_instance


async def initialize_coordinator(config: Dict[str, Any] = None) -> AgentCoordinator:
    """初始化全局协调器"""
    coordinator = get_coordinator(config)
    await coordinator.initialize()
    return coordinator


async def shutdown_coordinator():
    """关闭全局协调器"""
    global _coordinator_instance
    if _coordinator_instance:
        await _coordinator_instance.shutdown()
        _coordinator_instance = None