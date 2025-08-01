import aioredis
import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from utils.config import Config
from utils.logger import CustomLogger
from models.message_models import QueueMessage, MessageType, Priority

class MessageQueue:
    """基于Redis的消息队列服务"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = CustomLogger("MessageQueue")
        self.redis: Optional[aioredis.Redis] = None
        self.is_connected = False
        
        # 队列名称
        self.memory_queue = "memory_agent_queue"
        self.ros2_queue = "ros2_agent_queue"
        self.status_queue = "status_queue"
        
        # 统计信息
        self.messages_sent = 0
        self.messages_received = 0
        
    async def connect(self):
        """连接到Redis"""
        try:
            self.redis = aioredis.from_url(
                f"redis://{self.config.redis.host}:{self.config.redis.port}",
                password=self.config.redis.password,
                db=self.config.redis.db,
                decode_responses=True,
                socket_connect_timeout=self.config.redis.timeout,
                socket_timeout=self.config.redis.timeout
            )
            
            # 测试连接
            await self.redis.ping()
            self.is_connected = True
            
            self.logger.info("成功连接到Redis消息队列")
            
        except Exception as e:
            self.is_connected = False
            self.logger.error(f"连接Redis失败: {str(e)}")
            raise
    
    async def disconnect(self):
        """断开Redis连接"""
        if self.redis:
            await self.redis.close()
            self.is_connected = False
            self.logger.info("已断开Redis连接")
    
    async def send_to_memory_agent(self, message: QueueMessage) -> bool:
        """发送消息到记忆Agent队列"""
        return await self._send_message(self.memory_queue, message)
    
    async def send_to_ros2_agent(self, message: QueueMessage) -> bool:
        """发送消息到ROS2 Agent队列"""
        return await self._send_message(self.ros2_queue, message)
    
    async def send_status_update(self, message: QueueMessage) -> bool:
        """发送状态更新消息"""
        return await self._send_message(self.status_queue, message)
    
    async def _send_message(self, queue_name: str, message: QueueMessage) -> bool:
        """发送消息到指定队列"""
        try:
            if not self.is_connected:
                await self.connect()
            
            # 序列化消息
            message_data = message.dict()
            message_json = json.dumps(message_data, default=str, ensure_ascii=False)
            
            # 根据优先级选择队列操作
            if message.priority == Priority.HIGH:
                # 高优先级消息插入队列头部
                await self.redis.lpush(queue_name, message_json)
            else:
                # 普通优先级消息插入队列尾部
                await self.redis.rpush(queue_name, message_json)
            
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
            if not self.is_connected:
                await self.connect()
            
            # 阻塞式接收消息
            result = await self.redis.blpop(queue_name, timeout=timeout)
            
            if result:
                queue, message_json = result
                message_data = json.loads(message_json)
                
                # 重建QueueMessage对象
                message = QueueMessage(**message_data)
                
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
            
            return None
            
        except Exception as e:
            self.logger.error(
                f"从队列 {queue_name} 接收消息失败: {str(e)}",
                extra={"error": str(e)}
            )
            return None
    
    async def get_queue_length(self, queue_name: str) -> int:
        """获取队列长度"""
        try:
            if not self.is_connected:
                await self.connect()
            
            return await self.redis.llen(queue_name)
            
        except Exception as e:
            self.logger.error(f"获取队列 {queue_name} 长度失败: {str(e)}")
            return 0
    
    async def clear_queue(self, queue_name: str) -> bool:
        """清空指定队列"""
        try:
            if not self.is_connected:
                await self.connect()
            
            await self.redis.delete(queue_name)
            
            self.logger.info(f"已清空队列 {queue_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"清空队列 {queue_name} 失败: {str(e)}")
            return False
    
    async def get_all_queue_stats(self) -> Dict[str, Any]:
        """获取所有队列的统计信息"""
        try:
            if not self.is_connected:
                await self.connect()
            
            stats = {
                "memory_queue_length": await self.get_queue_length(self.memory_queue),
                "ros2_queue_length": await self.get_queue_length(self.ros2_queue),
                "status_queue_length": await self.get_queue_length(self.status_queue),
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
            if not self.is_connected:
                await self.connect()
            
            # 测试Redis连接
            start_time = datetime.now()
            await self.redis.ping()
            latency = (datetime.now() - start_time).total_seconds()
            
            # 获取Redis信息
            info = await self.redis.info()
            
            return {
                "status": "healthy",
                "latency": latency,
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory": info.get("used_memory_human"),
                "queue_stats": await self.get_all_queue_stats()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "connection_status": self.is_connected
            }
    
    async def publish_event(self, channel: str, event_data: Dict[str, Any]) -> bool:
        """发布事件到Redis频道（用于实时通知）"""
        try:
            if not self.is_connected:
                await self.connect()
            
            event_json = json.dumps(event_data, default=str, ensure_ascii=False)
            await self.redis.publish(channel, event_json)
            
            self.logger.info(
                f"事件已发布到频道 {channel}",
                extra={"channel": channel, "event_type": event_data.get("type")}
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"发布事件到频道 {channel} 失败: {str(e)}")
            return False
    
    async def subscribe_to_events(self, channels: List[str], callback):
        """订阅Redis频道事件"""
        try:
            if not self.is_connected:
                await self.connect()
            
            pubsub = self.redis.pubsub()
            await pubsub.subscribe(*channels)
            
            self.logger.info(f"已订阅频道: {channels}")
            
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        event_data = json.loads(message["data"])
                        await callback(message["channel"], event_data)
                    except Exception as e:
                        self.logger.error(f"处理订阅事件失败: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"订阅频道失败: {str(e)}")
            raise