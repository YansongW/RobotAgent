# -*- coding: utf-8 -*-

# 智能体消息总线 (Message Bus)
# 负责智能体间消息传递、路由和协调的核心通信组件
# 作者: RobotAgent开发团队
# 版本: 0.0.1 (Initial Release)
# 更新时间: 2025-08-21

# 导入标准库
import asyncio
import uuid
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Set
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import weakref

# 导入项目基础组件
from config import MessageType, MessagePriority, AgentMessage
from src.communication.protocols import CollaborationRequest, CollaborationResponse


class MessageBusStatus(Enum):
    """消息总线状态枚举"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class MessageRoute:
    """消息路由信息"""
    sender_id: str
    receiver_id: str
    message_type: MessageType
    route_priority: int = 1
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class MessageStats:
    """消息统计信息"""
    total_sent: int = 0
    total_received: int = 0
    total_failed: int = 0
    total_queued: int = 0
    average_latency_ms: float = 0.0
    peak_queue_size: int = 0
    active_connections: int = 0


class MessageBus:
    """智能体消息总线
    
    负责智能体间的消息传递、路由和协调，是整个系统的通信中枢。
    主要功能包括：
    - 消息路由和分发
    - 消息队列管理
    - 智能体注册和发现
    - 消息持久化和重试
    - 性能监控和统计
    - 消息过滤和转换
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化消息总线
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 总线状态
        self.status = MessageBusStatus.STOPPED
        self._shutdown_event = asyncio.Event()
        
        # 智能体注册表
        self.registered_agents: Dict[str, weakref.ReferenceType] = {}
        self.agent_metadata: Dict[str, Dict[str, Any]] = {}
        
        # 消息队列（每个智能体一个队列）
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.priority_queues: Dict[str, Dict[MessagePriority, asyncio.Queue]] = {}
        
        # 消息路由表
        self.routing_table: Dict[str, List[MessageRoute]] = defaultdict(list)
        self.subscription_table: Dict[MessageType, Set[str]] = defaultdict(set)
        
        # 消息处理器
        self.message_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.middleware_stack: List[Callable] = []
        
        # 消息历史和统计
        self.message_history: deque = deque(maxlen=self.config.get("max_history", 1000))
        self.message_stats = MessageStats()
        self.latency_samples: deque = deque(maxlen=100)
        
        # 配置参数
        self.max_queue_size = self.config.get("max_queue_size", 1000)
        self.message_timeout = self.config.get("message_timeout", 30.0)
        self.retry_attempts = self.config.get("retry_attempts", 3)
        self.enable_persistence = self.config.get("enable_persistence", False)
        
        # 后台任务
        self._background_tasks: List[asyncio.Task] = []
        
        self.logger.info("MessageBus 初始化完成")

    async def start(self):
        """启动消息总线"""
        if self.status != MessageBusStatus.STOPPED:
            self.logger.warning("消息总线已经在运行中")
            return
        
        self.status = MessageBusStatus.STARTING
        self.logger.info("正在启动消息总线...")
        
        try:
            # 启动后台任务
            await self._start_background_tasks()
            
            self.status = MessageBusStatus.RUNNING
            self.logger.info("消息总线启动成功")
            
        except Exception as e:
            self.status = MessageBusStatus.ERROR
            self.logger.error(f"消息总线启动失败: {str(e)}")
            raise

    async def stop(self):
        """停止消息总线"""
        if self.status == MessageBusStatus.STOPPED:
            return
        
        self.status = MessageBusStatus.STOPPING
        self.logger.info("正在停止消息总线...")
        
        try:
            # 设置关闭事件
            self._shutdown_event.set()
            
            # 停止后台任务
            await self._stop_background_tasks()
            
            # 清空消息队列
            await self._flush_all_queues()
            
            self.status = MessageBusStatus.STOPPED
            self.logger.info("消息总线已停止")
            
        except Exception as e:
            self.status = MessageBusStatus.ERROR
            self.logger.error(f"消息总线停止失败: {str(e)}")
            raise

    async def register_agent(self, agent_id: str, agent_ref: Any, 
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """注册智能体
        
        Args:
            agent_id: 智能体ID
            agent_ref: 智能体引用
            metadata: 智能体元数据
            
        Returns:
            是否注册成功
        """
        try:
            if agent_id in self.registered_agents:
                self.logger.warning(f"智能体 {agent_id} 已经注册")
                return False
            
            # 创建弱引用
            weak_ref = weakref.ref(agent_ref, lambda ref: self._on_agent_cleanup(agent_id))
            self.registered_agents[agent_id] = weak_ref
            
            # 存储元数据
            self.agent_metadata[agent_id] = metadata or {}
            
            # 创建消息队列
            await self._create_agent_queues(agent_id)
            
            # 更新统计
            self.message_stats.active_connections += 1
            
            self.logger.info(f"智能体 {agent_id} 注册成功")
            return True
            
        except Exception as e:
            self.logger.error(f"智能体注册失败: {str(e)}")
            return False

    async def unregister_agent(self, agent_id: str) -> bool:
        """注销智能体
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            是否注销成功
        """
        try:
            if agent_id not in self.registered_agents:
                self.logger.warning(f"智能体 {agent_id} 未注册")
                return False
            
            # 移除注册信息
            del self.registered_agents[agent_id]
            del self.agent_metadata[agent_id]
            
            # 清理消息队列
            await self._cleanup_agent_queues(agent_id)
            
            # 清理路由表
            self._cleanup_routing_table(agent_id)
            
            # 清理订阅表
            self._cleanup_subscription_table(agent_id)
            
            # 更新统计
            self.message_stats.active_connections -= 1
            
            self.logger.info(f"智能体 {agent_id} 注销成功")
            return True
            
        except Exception as e:
            self.logger.error(f"智能体注销失败: {str(e)}")
            return False

    async def send_message(self, message: AgentMessage) -> bool:
        """发送消息
        
        Args:
            message: 要发送的消息
            
        Returns:
            是否发送成功
        """
        try:
            start_time = datetime.now()
            
            # 验证消息
            if not await self._validate_message(message):
                return False
            
            # 应用中间件
            processed_message = await self._apply_middleware(message)
            if not processed_message:
                return False
            
            # 路由消息
            success = await self._route_message(processed_message)
            
            # 记录统计信息
            latency = (datetime.now() - start_time).total_seconds() * 1000
            await self._update_message_stats(success, latency)
            
            # 记录消息历史
            self._record_message_history(processed_message, success)
            
            return success
            
        except Exception as e:
            self.logger.error(f"消息发送失败: {str(e)}")
            self.message_stats.total_failed += 1
            return False

    async def broadcast_message(self, message: AgentMessage, 
                              exclude_agents: Optional[List[str]] = None) -> int:
        """广播消息
        
        Args:
            message: 要广播的消息
            exclude_agents: 排除的智能体列表
            
        Returns:
            成功发送的智能体数量
        """
        exclude_agents = exclude_agents or []
        success_count = 0
        
        for agent_id in self.registered_agents.keys():
            if agent_id not in exclude_agents and agent_id != message.sender_id:
                # 创建副本消息
                broadcast_message = AgentMessage(
                    message_id=str(uuid.uuid4()),
                    sender_id=message.sender_id,
                    receiver_id=agent_id,
                    message_type=message.message_type,
                    content=message.content,
                    timestamp=message.timestamp,
                    priority=message.priority,
                    metadata=message.metadata,
                    correlation_id=message.correlation_id
                )
                
                if await self.send_message(broadcast_message):
                    success_count += 1
        
        self.logger.info(f"广播消息完成，成功发送给 {success_count} 个智能体")
        return success_count

    async def subscribe(self, agent_id: str, message_types: List[MessageType]) -> bool:
        """订阅消息类型
        
        Args:
            agent_id: 智能体ID
            message_types: 要订阅的消息类型列表
            
        Returns:
            是否订阅成功
        """
        try:
            if agent_id not in self.registered_agents:
                self.logger.error(f"智能体 {agent_id} 未注册")
                return False
            
            for message_type in message_types:
                self.subscription_table[message_type].add(agent_id)
            
            self.logger.info(f"智能体 {agent_id} 订阅消息类型: {[mt.value for mt in message_types]}")
            return True
            
        except Exception as e:
            self.logger.error(f"消息订阅失败: {str(e)}")
            return False

    async def unsubscribe(self, agent_id: str, message_types: List[MessageType]) -> bool:
        """取消订阅消息类型
        
        Args:
            agent_id: 智能体ID
            message_types: 要取消订阅的消息类型列表
            
        Returns:
            是否取消订阅成功
        """
        try:
            for message_type in message_types:
                self.subscription_table[message_type].discard(agent_id)
            
            self.logger.info(f"智能体 {agent_id} 取消订阅消息类型: {[mt.value for mt in message_types]}")
            return True
            
        except Exception as e:
            self.logger.error(f"取消订阅失败: {str(e)}")
            return False

    async def receive_message(self, agent_id: str, timeout: Optional[float] = None) -> Optional[AgentMessage]:
        """接收消息
        
        Args:
            agent_id: 智能体ID
            timeout: 超时时间（秒）
            
        Returns:
            接收到的消息，如果超时则返回None
        """
        try:
            if agent_id not in self.message_queues:
                self.logger.error(f"智能体 {agent_id} 的消息队列不存在")
                return None
            
            # 优先处理高优先级消息
            for priority_value in [6, 5, 4, 3, 2, 1]:  # CRITICAL到LOW的优先级值
                priority_queue = self.priority_queues[agent_id].get(priority_value)
                if priority_queue and not priority_queue.empty():
                    message = await asyncio.wait_for(priority_queue.get(), timeout=timeout or self.message_timeout)
                    self.message_stats.total_received += 1
                    return message
            
            # 如果没有优先级消息，从普通队列获取
            message = await asyncio.wait_for(
                self.message_queues[agent_id].get(), 
                timeout=timeout or self.message_timeout
            )
            
            self.message_stats.total_received += 1
            return message
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            self.logger.error(f"接收消息失败: {str(e)}")
            return None

    async def add_middleware(self, middleware: Callable) -> bool:
        """添加中间件
        
        Args:
            middleware: 中间件函数
            
        Returns:
            是否添加成功
        """
        try:
            self.middleware_stack.append(middleware)
            self.logger.info(f"中间件添加成功: {middleware.__name__}")
            return True
        except Exception as e:
            self.logger.error(f"中间件添加失败: {str(e)}")
            return False

    async def add_message_handler(self, agent_id: str, handler: Callable) -> bool:
        """添加消息处理器
        
        Args:
            agent_id: 智能体ID
            handler: 消息处理器函数
            
        Returns:
            是否添加成功
        """
        try:
            self.message_handlers[agent_id].append(handler)
            self.logger.info(f"消息处理器添加成功: {agent_id} -> {handler.__name__}")
            return True
        except Exception as e:
            self.logger.error(f"消息处理器添加失败: {str(e)}")
            return False

    async def get_agent_list(self) -> List[str]:
        """获取已注册的智能体列表
        
        Returns:
            智能体ID列表
        """
        return list(self.registered_agents.keys())

    async def get_agent_metadata(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """获取智能体元数据
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            智能体元数据
        """
        return self.agent_metadata.get(agent_id)

    async def get_message_stats(self) -> MessageStats:
        """获取消息统计信息
        
        Returns:
            消息统计信息
        """
        # 更新当前队列大小
        total_queued = sum(queue.qsize() for queue in self.message_queues.values())
        self.message_stats.total_queued = total_queued
        
        return self.message_stats

    async def get_queue_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """获取智能体队列状态
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            队列状态信息
        """
        if agent_id not in self.message_queues:
            return None
        
        return {
            "agent_id": agent_id,
            "queue_size": self.message_queues[agent_id].qsize(),
            "priority_queues": {
                priority: queue.qsize() 
                for priority, queue in self.priority_queues[agent_id].items()
            },
            "max_queue_size": self.max_queue_size
        }

    # 私有方法
    
    async def _start_background_tasks(self):
        """启动后台任务"""
        # 消息处理任务
        message_processor = asyncio.create_task(self._message_processor_loop())
        self._background_tasks.append(message_processor)
        
        # 统计更新任务
        stats_updater = asyncio.create_task(self._stats_updater_loop())
        self._background_tasks.append(stats_updater)
        
        # 队列监控任务
        queue_monitor = asyncio.create_task(self._queue_monitor_loop())
        self._background_tasks.append(queue_monitor)

    async def _stop_background_tasks(self):
        """停止后台任务"""
        for task in self._background_tasks:
            task.cancel()
        
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

    async def _create_agent_queues(self, agent_id: str):
        """创建智能体消息队列"""
        self.message_queues[agent_id] = asyncio.Queue(maxsize=self.max_queue_size)
        
        # 创建优先级队列
        self.priority_queues[agent_id] = {
            priority.value: asyncio.Queue(maxsize=self.max_queue_size // 4)
            for priority in MessagePriority
        }

    async def _cleanup_agent_queues(self, agent_id: str):
        """清理智能体消息队列"""
        if agent_id in self.message_queues:
            # 清空队列
            while not self.message_queues[agent_id].empty():
                try:
                    self.message_queues[agent_id].get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            del self.message_queues[agent_id]
        
        if agent_id in self.priority_queues:
            del self.priority_queues[agent_id]

    def _cleanup_routing_table(self, agent_id: str):
        """清理路由表"""
        # 移除相关路由
        routes_to_remove = []
        for key, routes in self.routing_table.items():
            routes_to_remove.extend([
                route for route in routes 
                if route.sender_id == agent_id or route.receiver_id == agent_id
            ])
        
        for route in routes_to_remove:
            for routes in self.routing_table.values():
                if route in routes:
                    routes.remove(route)

    def _cleanup_subscription_table(self, agent_id: str):
        """清理订阅表"""
        for message_type, subscribers in self.subscription_table.items():
            subscribers.discard(agent_id)

    async def _validate_message(self, message: AgentMessage) -> bool:
        """验证消息"""
        if not message.message_id or not message.sender_id:
            self.logger.error("消息缺少必要字段")
            return False
        
        if message.receiver_id and message.receiver_id not in self.registered_agents:
            self.logger.error(f"接收者 {message.receiver_id} 未注册")
            return False
        
        return True

    async def _apply_middleware(self, message: AgentMessage) -> Optional[AgentMessage]:
        """应用中间件"""
        current_message = message
        
        for middleware in self.middleware_stack:
            try:
                current_message = await middleware(current_message)
                if not current_message:
                    return None
            except Exception as e:
                self.logger.error(f"中间件处理失败: {str(e)}")
                return None
        
        return current_message

    async def _route_message(self, message: AgentMessage) -> bool:
        """路由消息"""
        try:
            # 点对点消息
            if message.receiver_id:
                return await self._deliver_message(message.receiver_id, message)
            
            # 订阅模式消息
            subscribers = self.subscription_table.get(message.message_type, set())
            if not subscribers:
                self.logger.warning(f"消息类型 {message.message_type.value} 没有订阅者")
                return False
            
            success_count = 0
            for subscriber_id in subscribers:
                if subscriber_id != message.sender_id:  # 不发送给自己
                    if await self._deliver_message(subscriber_id, message):
                        success_count += 1
            
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"消息路由失败: {str(e)}")
            return False

    async def _deliver_message(self, agent_id: str, message: AgentMessage) -> bool:
        """投递消息到智能体"""
        try:
            if agent_id not in self.message_queues:
                self.logger.error(f"智能体 {agent_id} 的消息队列不存在")
                return False
            
            # 根据优先级选择队列
            if message.priority and message.priority != MessagePriority.MEDIUM:
                priority_queue = self.priority_queues[agent_id].get(message.priority)
                if priority_queue and not priority_queue.full():
                    await priority_queue.put(message)
                    return True
            
            # 使用普通队列
            if not self.message_queues[agent_id].full():
                await self.message_queues[agent_id].put(message)
                self.message_stats.total_sent += 1
                return True
            else:
                self.logger.warning(f"智能体 {agent_id} 的消息队列已满")
                return False
                
        except Exception as e:
            self.logger.error(f"消息投递失败: {str(e)}")
            return False

    async def _flush_all_queues(self):
        """清空所有消息队列"""
        for agent_id in list(self.message_queues.keys()):
            await self._cleanup_agent_queues(agent_id)

    async def _update_message_stats(self, success: bool, latency_ms: float):
        """更新消息统计"""
        if success:
            self.message_stats.total_sent += 1
        else:
            self.message_stats.total_failed += 1
        
        # 更新延迟统计
        self.latency_samples.append(latency_ms)
        if self.latency_samples:
            self.message_stats.average_latency_ms = sum(self.latency_samples) / len(self.latency_samples)

    def _record_message_history(self, message: AgentMessage, success: bool):
        """记录消息历史"""
        history_entry = {
            "message_id": message.message_id,
            "sender_id": message.sender_id,
            "receiver_id": message.receiver_id,
            "message_type": message.message_type.value,
            "timestamp": message.timestamp.isoformat(),
            "success": success
        }
        
        self.message_history.append(history_entry)

    def _on_agent_cleanup(self, agent_id: str):
        """智能体清理回调"""
        try:
            # 检查是否有运行中的事件循环
            loop = asyncio.get_running_loop()
            if loop and not loop.is_closed():
                asyncio.create_task(self.unregister_agent(agent_id))
            else:
                # 如果没有运行中的事件循环，直接同步清理
                self._sync_unregister_agent(agent_id)
        except RuntimeError:
            # 没有运行中的事件循环，直接同步清理
            self._sync_unregister_agent(agent_id)
    
    def _sync_unregister_agent(self, agent_id: str):
        """同步方式注销智能体"""
        if agent_id in self.registered_agents:
            del self.registered_agents[agent_id]
        if agent_id in self.agent_metadata:
            del self.agent_metadata[agent_id]
        if agent_id in self.message_queues:
            del self.message_queues[agent_id]
        if agent_id in self.message_handlers:
            del self.message_handlers[agent_id]

    async def _message_processor_loop(self):
        """消息处理循环"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(0.1)  # 避免过度占用CPU
                
                # 处理消息处理器
                for agent_id, handlers in self.message_handlers.items():
                    if agent_id in self.message_queues:
                        queue = self.message_queues[agent_id]
                        if not queue.empty():
                            try:
                                message = queue.get_nowait()
                                for handler in handlers:
                                    await handler(message)
                            except asyncio.QueueEmpty:
                                continue
                            except Exception as e:
                                self.logger.error(f"消息处理器异常: {str(e)}")
                
            except Exception as e:
                self.logger.error(f"消息处理循环异常: {str(e)}")

    async def _stats_updater_loop(self):
        """统计更新循环"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(5)  # 每5秒更新一次
                
                # 更新峰值队列大小
                current_max = max(
                    (queue.qsize() for queue in self.message_queues.values()),
                    default=0
                )
                
                if current_max > self.message_stats.peak_queue_size:
                    self.message_stats.peak_queue_size = current_max
                
            except Exception as e:
                self.logger.error(f"统计更新循环异常: {str(e)}")

    async def _queue_monitor_loop(self):
        """队列监控循环"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(10)  # 每10秒检查一次
                
                # 检查队列健康状态
                for agent_id, queue in self.message_queues.items():
                    queue_size = queue.qsize()
                    
                    # 队列接近满载警告
                    if queue_size > self.max_queue_size * 0.8:
                        self.logger.warning(
                            f"智能体 {agent_id} 的消息队列接近满载: {queue_size}/{self.max_queue_size}"
                        )
                    
                    # 清理无效的智能体引用
                    if agent_id in self.registered_agents:
                        agent_ref = self.registered_agents[agent_id]()
                        if agent_ref is None:
                            self.logger.info(f"检测到智能体 {agent_id} 已被垃圾回收，开始清理")
                            await self.unregister_agent(agent_id)
                
            except Exception as e:
                self.logger.error(f"队列监控循环异常: {str(e)}")


# 全局消息总线实例
_global_message_bus: Optional[MessageBus] = None


def get_message_bus() -> MessageBus:
    """获取全局消息总线实例"""
    global _global_message_bus
    if _global_message_bus is None:
        _global_message_bus = MessageBus()
    return _global_message_bus


async def initialize_message_bus(config: Optional[Dict[str, Any]] = None) -> MessageBus:
    """初始化全局消息总线"""
    global _global_message_bus
    _global_message_bus = MessageBus(config)
    await _global_message_bus.start()
    return _global_message_bus


async def shutdown_message_bus():
    """关闭全局消息总线"""
    global _global_message_bus
    if _global_message_bus:
        await _global_message_bus.stop()
        _global_message_bus = None