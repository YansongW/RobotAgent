"""
记忆服务
实现对话历史的本地化保存和管理，支持记忆文件导入prompt的二次复用
"""

import json
import logging
import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)

class MemoryService:
    """记忆服务类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化记忆服务
        
        Args:
            config: 记忆配置
        """
        self.config = config
        self.max_history = config.get("max_history", 100)
        
        # 设置文件路径
        self.memory_dir = Path(__file__).parent.parent / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        self.conversation_file = self.memory_dir / "conversation_history.json"
        self.session_file = self.memory_dir / "session_data.json"
        self.context_file = self.memory_dir / "context_cache.json"
        
        # 初始化数据结构
        self.conversations = self._load_conversations()
        self.session_data = self._load_session_data()
        self.context_cache = self._load_context_cache()
        
        logger.info("记忆服务初始化完成")
    
    def _load_conversations(self) -> Dict[str, List[Dict]]:
        """加载对话历史"""
        try:
            if self.conversation_file.exists():
                with open(self.conversation_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"加载对话历史失败: {e}")
            return {}
    
    def _load_session_data(self) -> Dict[str, Any]:
        """加载会话数据"""
        try:
            if self.session_file.exists():
                with open(self.session_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"加载会话数据失败: {e}")
            return {}
    
    def _load_context_cache(self) -> Dict[str, Any]:
        """加载上下文缓存"""
        try:
            if self.context_file.exists():
                with open(self.context_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"加载上下文缓存失败: {e}")
            return {}
    
    async def save_conversation(
        self,
        conversation_id: str,
        user_message: str,
        bot_response: str,
        ros2_command: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ):
        """
        保存对话记录
        
        Args:
            conversation_id: 对话ID
            user_message: 用户消息
            bot_response: 机器人回复
            ros2_command: ROS2命令
            metadata: 元数据
        """
        try:
            timestamp = datetime.now().isoformat()
            
            # 创建对话记录
            conversation_record = {
                "id": str(uuid.uuid4()),
                "timestamp": timestamp,
                "user_message": user_message,
                "bot_response": bot_response,
                "ros2_command": ros2_command,
                "metadata": metadata or {}
            }
            
            # 添加到对话历史
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = []
            
            self.conversations[conversation_id].append(conversation_record)
            
            # 限制历史记录长度
            if len(self.conversations[conversation_id]) > self.max_history:
                self.conversations[conversation_id] = self.conversations[conversation_id][-self.max_history:]
            
            # 保存到文件
            await self._save_conversations()
            
            logger.info(f"对话记录已保存: {conversation_id}")
            
        except Exception as e:
            logger.error(f"保存对话记录失败: {e}")
    
    async def get_conversation_history(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        获取对话历史
        
        Args:
            conversation_id: 对话ID
            limit: 限制返回的记录数量
            
        Returns:
            对话历史列表
        """
        try:
            history = self.conversations.get(conversation_id, [])
            
            if limit:
                history = history[-limit:]
            
            # 转换为标准格式
            formatted_history = []
            for record in history:
                formatted_history.extend([
                    {
                        "role": "user",
                        "content": record["user_message"],
                        "timestamp": record["timestamp"]
                    },
                    {
                        "role": "assistant",
                        "content": record["bot_response"],
                        "timestamp": record["timestamp"]
                    }
                ])
            
            return formatted_history
            
        except Exception as e:
            logger.error(f"获取对话历史失败: {e}")
            return []
    
    async def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """
        获取对话摘要
        
        Args:
            conversation_id: 对话ID
            
        Returns:
            对话摘要信息
        """
        try:
            history = self.conversations.get(conversation_id, [])
            
            if not history:
                return {
                    "conversation_id": conversation_id,
                    "message_count": 0,
                    "start_time": None,
                    "last_time": None,
                    "ros2_commands_count": 0
                }
            
            ros2_commands = [r for r in history if r.get("ros2_command")]
            
            return {
                "conversation_id": conversation_id,
                "message_count": len(history),
                "start_time": history[0]["timestamp"],
                "last_time": history[-1]["timestamp"],
                "ros2_commands_count": len(ros2_commands),
                "recent_topics": self._extract_recent_topics(history)
            }
            
        except Exception as e:
            logger.error(f"获取对话摘要失败: {e}")
            return {}
    
    def _extract_recent_topics(self, history: List[Dict]) -> List[str]:
        """提取最近的话题"""
        topics = []
        for record in history[-5:]:  # 最近5条记录
            user_msg = record.get("user_message", "")
            if len(user_msg) > 10:  # 过滤太短的消息
                topics.append(user_msg[:50])  # 截取前50个字符
        return topics
    
    async def save_context_for_reuse(
        self,
        context_id: str,
        context_data: Dict[str, Any],
        tags: List[str] = None
    ):
        """
        保存上下文以供复用
        
        Args:
            context_id: 上下文ID
            context_data: 上下文数据
            tags: 标签列表
        """
        try:
            timestamp = datetime.now().isoformat()
            
            context_record = {
                "id": context_id,
                "timestamp": timestamp,
                "data": context_data,
                "tags": tags or [],
                "usage_count": 0
            }
            
            self.context_cache[context_id] = context_record
            await self._save_context_cache()
            
            logger.info(f"上下文已保存: {context_id}")
            
        except Exception as e:
            logger.error(f"保存上下文失败: {e}")
    
    async def get_context_for_reuse(
        self,
        context_id: str = None,
        tags: List[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        获取可复用的上下文
        
        Args:
            context_id: 指定的上下文ID
            tags: 标签过滤
            
        Returns:
            上下文数据
        """
        try:
            if context_id and context_id in self.context_cache:
                context = self.context_cache[context_id]
                context["usage_count"] += 1
                await self._save_context_cache()
                return context["data"]
            
            if tags:
                # 根据标签查找匹配的上下文
                for ctx_id, context in self.context_cache.items():
                    if any(tag in context.get("tags", []) for tag in tags):
                        context["usage_count"] += 1
                        await self._save_context_cache()
                        return context["data"]
            
            return None
            
        except Exception as e:
            logger.error(f"获取上下文失败: {e}")
            return None
    
    async def import_prompt_context(self, prompt_file: str) -> Optional[Dict[str, Any]]:
        """
        从prompt文件导入上下文
        
        Args:
            prompt_file: prompt文件路径
            
        Returns:
            导入的上下文数据
        """
        try:
            prompts_dir = Path(__file__).parent.parent / "prompts"
            file_path = prompts_dir / prompt_file
            
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    prompt_data = json.load(f)
                
                # 创建上下文ID
                context_id = f"prompt_{prompt_file}_{datetime.now().strftime('%Y%m%d')}"
                
                # 保存为可复用上下文
                await self.save_context_for_reuse(
                    context_id=context_id,
                    context_data=prompt_data,
                    tags=["prompt", "imported"]
                )
                
                logger.info(f"Prompt上下文已导入: {prompt_file}")
                return prompt_data
            else:
                logger.warning(f"Prompt文件不存在: {prompt_file}")
                return None
                
        except Exception as e:
            logger.error(f"导入prompt上下文失败: {e}")
            return None
    
    async def cleanup_old_data(self, days: int = 30):
        """
        清理旧数据
        
        Args:
            days: 保留天数
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            cutoff_str = cutoff_date.isoformat()
            
            # 清理旧的对话记录
            for conv_id in list(self.conversations.keys()):
                history = self.conversations[conv_id]
                filtered_history = [
                    record for record in history
                    if record.get("timestamp", "") > cutoff_str
                ]
                
                if filtered_history:
                    self.conversations[conv_id] = filtered_history
                else:
                    del self.conversations[conv_id]
            
            # 清理旧的上下文缓存
            for ctx_id in list(self.context_cache.keys()):
                context = self.context_cache[ctx_id]
                if context.get("timestamp", "") < cutoff_str:
                    del self.context_cache[ctx_id]
            
            # 保存清理后的数据
            await self._save_conversations()
            await self._save_context_cache()
            
            logger.info(f"已清理{days}天前的旧数据")
            
        except Exception as e:
            logger.error(f"清理旧数据失败: {e}")
    
    async def _save_conversations(self):
        """保存对话历史到文件"""
        try:
            with open(self.conversation_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversations, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存对话历史失败: {e}")
    
    async def _save_session_data(self):
        """保存会话数据到文件"""
        try:
            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(self.session_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存会话数据失败: {e}")
    
    async def _save_context_cache(self):
        """保存上下文缓存到文件"""
        try:
            with open(self.context_file, 'w', encoding='utf-8') as f:
                json.dump(self.context_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存上下文缓存失败: {e}")
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        try:
            total_conversations = len(self.conversations)
            total_messages = sum(len(history) for history in self.conversations.values())
            total_contexts = len(self.context_cache)
            
            return {
                "total_conversations": total_conversations,
                "total_messages": total_messages,
                "total_contexts": total_contexts,
                "memory_file_size": self.conversation_file.stat().st_size if self.conversation_file.exists() else 0,
                "context_file_size": self.context_file.stat().st_size if self.context_file.exists() else 0
            }
            
        except Exception as e:
            logger.error(f"获取记忆统计失败: {e}")
            return {}