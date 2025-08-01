import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import deque
from utils.config import Config
from utils.logger import CustomLogger
from models.message_models import QueueMessage, MessageType, Priority

class MessageQueue:
    """基于内存的消息队列服务（简化版本，避免Redis依赖问题）"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = CustomLogger("MessageQueue")
        self.is_connected = True  # 内存队列始终可用
        
        # 内存队列
        self.memory_queue = deque()
        self.ros2_queue = deque()
        self.status_queue = deque()
        
        # 队列锁
        self.memory_lock = asyncio.Lock()
        self.ros2_lock = asyncio.Lock()
        self.status_lock = asyncio.Lock()
        
        # 统计信息
        self.messages_sent = 0
        self.messages_received = 0
        
        self.logger.info("内存消息队列已初始化")
        
    async def connect(self):
        """连接到消息队列（内存版本无需实际连接）"""
        self.is_connected = True
        self.logger.info("消息队列连接成功（内存模式）")
    
    async def disconnect(self):
        """断开连接（内存版本无需实际断开）"""
        self.is_connected = False
        self.logger.info("消息队列已断开（内存模式）")
    
    async def send_to_memory_agent(self, message: QueueMessage) -> bool:
        """发送消息到记忆Agent队列"""
        return await self._send_message("memory", message)
    
    async def send_to_ros2_agent(self, message: QueueMessage) -> bool:
        """发送消息到ROS2 Agent队列"""
        return await self._send_message("ros2", message)
    
    async def send_status_update(self, message: QueueMessage) -> bool:
        """发送状态更新消息"""
        return await self._send_message("status", message)
    
    async def send_to_queue(self, queue_name: str, message: QueueMessage) -> bool:
        """发送消息到指定队列"""
        return await self._send_message(queue_name, message)
    
    async def _send_message(self, queue_name: str, message: QueueMessage) -> bool:
        """发送消息到指定队列"""
        try:
            # 获取对应的队列和锁
            queue, lock = self._get_queue_and_lock(queue_name)
            
            async with lock:
                if message.priority == Priority.HIGH:
                    # 高优先级消息插入队列头部
                    queue.appendleft(message)
                else:
                    # 普通优先级消息插入队列尾部
                    queue.append(message)
            
            self.messages_sent += 1
            
            self.logger.info(
                f"消息已发送到队列 {queue_name}",
                extra={
                    "message_id": message.message_id,
                    "message_type": message.message_type.value,
                    "priority": message.priority.value,
                    "queue": queue_name
                }
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                f"发送消息到队列 {queue_name} 失败: {str(e)}",
                extra={
                    "message_id": message.message_id,
                    "error": str(e)
                }
            )
            return False
    
    async def receive_from_queue(self, queue_name: str, timeout: int = 10) -> Optional[QueueMessage]:
        """从指定队列接收消息"""
        try:
            # 获取对应的队列和锁
            queue, lock = self._get_queue_and_lock(queue_name)
            
            # 等待消息或超时
            start_time = asyncio.get_event_loop().time()
            
            while True:
                async with lock:
                    if queue:
                        message = queue.popleft()
                        self.messages_received += 1
                        
                        self.logger.info(
                            f"从队列 {queue_name} 接收到消息",
                            extra={
                                "message_id": message.message_id,
                                "message_type": message.message_type.value,
                                "queue": queue_name
                            }
                        )
                        
                        return message
                
                # 检查超时
                if asyncio.get_event_loop().time() - start_time > timeout:
                    return None
                
                # 短暂等待
                await asyncio.sleep(0.1)
            
        except Exception as e:
            self.logger.error(
                f"从队列 {queue_name} 接收消息失败: {str(e)}",
                extra={"error": str(e)}
            )
            return None
    
    def _get_queue_and_lock(self, queue_name: str):
        """获取队列和对应的锁"""
        if queue_name == "memory":
            return self.memory_queue, self.memory_lock
        elif queue_name == "ros2":
            return self.ros2_queue, self.ros2_lock
        elif queue_name == "status":
            return self.status_queue, self.status_lock
        else:
            # 默认使用status队列
            return self.status_queue, self.status_lock
    
    async def get_queue_length(self, queue_name: str) -> int:
        """获取队列长度"""
        try:
            queue, lock = self._get_queue_and_lock(queue_name)
            
            async with lock:
                return len(queue)
            
        except Exception as e:
            self.logger.error(f"获取队列 {queue_name} 长度失败: {str(e)}")
            return 0
    
    async def clear_queue(self, queue_name: str) -> bool:
        """清空指定队列"""
        try:
            queue, lock = self._get_queue_and_lock(queue_name)
            
            async with lock:
                queue.clear()
            
            self.logger.info(f"已清空队列 {queue_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"清空队列 {queue_name} 失败: {str(e)}")
            return False
    
    async def get_all_queue_stats(self) -> Dict[str, Any]:
        """获取所有队列的统计信息"""
        try:
            stats = {
                "memory_queue_length": await self.get_queue_length("memory"),
                "ros2_queue_length": await self.get_queue_length("ros2"),
                "status_queue_length": await self.get_queue_length("status"),
                "messages_sent": self.messages_sent,
                "messages_received": self.messages_received,
                "connection_status": "connected" if self.is_connected else "disconnected"
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"获取队列统计信息失败: {str(e)}")
            return {
                "error": str(e),
                "connection_status": "error"
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            stats = await self.get_all_queue_stats()
            
            return {
                "status": "healthy" if self.is_connected else "unhealthy",
                "queue_stats": stats,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    # 为了兼容性，添加一些属性
    @property
    def memory_queue_name(self):
        return "memory"
    
    @property
    def ros2_queue_name(self):
        return "ros2"
    
    @property
    def status_queue_name(self):
        return "status"