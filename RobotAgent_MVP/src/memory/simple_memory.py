# -*- coding: utf-8 -*-

# 简化记忆系统 (Simple Memory System)
# 为MVP版本提供基础记忆功能和存储管理
# 作者: RobotAgent开发团队
# 版本: 0.0.1 (Initial Release)
# 更新时间: 2025-08-21

# 导入标准库
import json
import uuid
import pickle
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import logging
from pathlib import Path

# 导入项目基础组件
from ..communication.protocols import MessageType, AgentMessage


@dataclass
class SimpleMemoryItem:
    """简化记忆项数据结构"""
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    memory_type: str = "general"  # general, task, conversation, knowledge
    importance: float = 0.5  # 0.0 - 1.0
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimpleMemoryItem':
        """从字典创建实例"""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'last_accessed' in data and isinstance(data['last_accessed'], str):
            data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        return cls(**data)
    
    def update_access(self):
        """更新访问信息"""
        self.last_accessed = datetime.now()
        self.access_count += 1


class SimpleMemorySystem:
    """简化记忆系统"""
    
    def __init__(self, storage_path: Optional[str] = None, max_memories: int = 10000):
        """初始化简化记忆系统
        
        Args:
            storage_path: 存储路径
            max_memories: 最大记忆数量
        """
        self.max_memories = max_memories
        self.storage_path = Path(storage_path) if storage_path else Path("./memory_storage")
        self.storage_path.mkdir(exist_ok=True)
        
        # 内存存储
        self.memories: Dict[str, SimpleMemoryItem] = {}
        self.type_index: Dict[str, List[str]] = defaultdict(list)  # type -> memory_ids
        self.tag_index: Dict[str, List[str]] = defaultdict(list)   # tag -> memory_ids
        self.recent_memories: deque = deque(maxlen=100)  # 最近访问的记忆
        
        self.logger = logging.getLogger(__name__)
        
        # 加载已存储的记忆
        self._load_memories()
    
    def store_memory(self, content: str, memory_type: str = "general",
                    importance: float = 0.5, tags: Optional[List[str]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """存储记忆
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型
            importance: 重要性 (0.0-1.0)
            tags: 标签列表
            metadata: 元数据
            
        Returns:
            记忆ID
        """
        memory_item = SimpleMemoryItem(
            content=content,
            memory_type=memory_type,
            importance=max(0.0, min(1.0, importance)),
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # 存储到内存
        self.memories[memory_item.memory_id] = memory_item
        
        # 更新索引
        self.type_index[memory_type].append(memory_item.memory_id)
        for tag in memory_item.tags:
            self.tag_index[tag].append(memory_item.memory_id)
        
        # 添加到最近记忆
        self.recent_memories.append(memory_item.memory_id)
        
        # 检查记忆数量限制
        self._cleanup_old_memories()
        
        # 持久化存储
        self._save_memory(memory_item)
        
        self.logger.info(f"存储记忆: {memory_item.memory_id[:8]}... 类型: {memory_type}")
        return memory_item.memory_id
    
    def retrieve_memory(self, memory_id: str) -> Optional[SimpleMemoryItem]:
        """检索指定记忆"""
        memory_item = self.memories.get(memory_id)
        if memory_item:
            memory_item.update_access()
            self.recent_memories.append(memory_id)
            return memory_item
        return None
    
    def search_memories(self, query: str = "", memory_type: Optional[str] = None,
                       tags: Optional[List[str]] = None, limit: int = 10,
                       min_importance: float = 0.0) -> List[SimpleMemoryItem]:
        """搜索记忆
        
        Args:
            query: 搜索关键词
            memory_type: 记忆类型过滤
            tags: 标签过滤
            limit: 返回数量限制
            min_importance: 最小重要性
            
        Returns:
            匹配的记忆列表
        """
        results = []
        
        for memory_item in self.memories.values():
            # 重要性过滤
            if memory_item.importance < min_importance:
                continue
            
            # 类型过滤
            if memory_type and memory_item.memory_type != memory_type:
                continue
            
            # 标签过滤
            if tags and not any(tag in memory_item.tags for tag in tags):
                continue
            
            # 关键词搜索
            if query:
                query_lower = query.lower()
                if (query_lower not in memory_item.content.lower() and
                    not any(query_lower in tag.lower() for tag in memory_item.tags)):
                    continue
            
            results.append(memory_item)
            
            # 更新访问信息
            memory_item.update_access()
        
        # 按重要性和最近访问时间排序
        results.sort(key=lambda x: (x.importance, x.last_accessed), reverse=True)
        
        return results[:limit]
    
    def get_memories_by_type(self, memory_type: str, limit: int = 10) -> List[SimpleMemoryItem]:
        """按类型获取记忆"""
        memory_ids = self.type_index.get(memory_type, [])
        memories = []
        
        for memory_id in memory_ids:
            if memory_id in self.memories:
                memory_item = self.memories[memory_id]
                memory_item.update_access()
                memories.append(memory_item)
        
        # 按重要性和时间排序
        memories.sort(key=lambda x: (x.importance, x.created_at), reverse=True)
        return memories[:limit]
    
    def get_memories_by_tags(self, tags: List[str], limit: int = 10) -> List[SimpleMemoryItem]:
        """按标签获取记忆"""
        memory_ids = set()
        for tag in tags:
            memory_ids.update(self.tag_index.get(tag, []))
        
        memories = []
        for memory_id in memory_ids:
            if memory_id in self.memories:
                memory_item = self.memories[memory_id]
                memory_item.update_access()
                memories.append(memory_item)
        
        # 按重要性和时间排序
        memories.sort(key=lambda x: (x.importance, x.created_at), reverse=True)
        return memories[:limit]
    
    def get_recent_memories(self, limit: int = 10) -> List[SimpleMemoryItem]:
        """获取最近的记忆"""
        recent_ids = list(self.recent_memories)[-limit:]
        recent_ids.reverse()  # 最新的在前
        
        memories = []
        for memory_id in recent_ids:
            if memory_id in self.memories:
                memories.append(self.memories[memory_id])
        
        return memories
    
    def update_memory(self, memory_id: str, content: Optional[str] = None,
                     importance: Optional[float] = None, tags: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """更新记忆"""
        memory_item = self.memories.get(memory_id)
        if not memory_item:
            return False
        
        # 更新内容
        if content is not None:
            memory_item.content = content
        
        if importance is not None:
            memory_item.importance = max(0.0, min(1.0, importance))
        
        if tags is not None:
            # 从旧标签索引中移除
            for old_tag in memory_item.tags:
                if memory_id in self.tag_index[old_tag]:
                    self.tag_index[old_tag].remove(memory_id)
            
            # 更新标签
            memory_item.tags = tags
            
            # 添加到新标签索引
            for tag in tags:
                self.tag_index[tag].append(memory_id)
        
        if metadata is not None:
            memory_item.metadata.update(metadata)
        
        # 更新访问信息
        memory_item.update_access()
        
        # 持久化存储
        self._save_memory(memory_item)
        
        self.logger.info(f"更新记忆: {memory_id[:8]}...")
        return True
    
    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        memory_item = self.memories.get(memory_id)
        if not memory_item:
            return False
        
        # 从索引中移除
        memory_type = memory_item.memory_type
        if memory_id in self.type_index[memory_type]:
            self.type_index[memory_type].remove(memory_id)
        
        for tag in memory_item.tags:
            if memory_id in self.tag_index[tag]:
                self.tag_index[tag].remove(memory_id)
        
        # 从最近记忆中移除
        if memory_id in self.recent_memories:
            temp_recent = deque()
            for mid in self.recent_memories:
                if mid != memory_id:
                    temp_recent.append(mid)
            self.recent_memories = temp_recent
        
        # 从内存中删除
        del self.memories[memory_id]
        
        # 删除持久化文件
        self._delete_memory_file(memory_id)
        
        self.logger.info(f"删除记忆: {memory_id[:8]}...")
        return True
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        type_counts = {}
        for memory_type, memory_ids in self.type_index.items():
            type_counts[memory_type] = len(memory_ids)
        
        tag_counts = {}
        for tag, memory_ids in self.tag_index.items():
            tag_counts[tag] = len(memory_ids)
        
        # 计算平均重要性
        total_importance = sum(m.importance for m in self.memories.values())
        avg_importance = total_importance / len(self.memories) if self.memories else 0
        
        return {
            'total_memories': len(self.memories),
            'max_memories': self.max_memories,
            'memory_types': type_counts,
            'tag_counts': tag_counts,
            'average_importance': avg_importance,
            'recent_memories_count': len(self.recent_memories)
        }
    
    def export_memories(self, file_path: str) -> bool:
        """导出记忆到文件"""
        try:
            export_data = {
                'memories': [memory.to_dict() for memory in self.memories.values()],
                'export_time': datetime.now().isoformat(),
                'total_count': len(self.memories)
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"导出 {len(self.memories)} 个记忆到: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"导出记忆失败: {e}")
            return False
    
    def import_memories(self, file_path: str) -> bool:
        """从文件导入记忆"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            imported_count = 0
            for memory_data in import_data.get('memories', []):
                memory_item = SimpleMemoryItem.from_dict(memory_data)
                
                # 避免ID冲突
                if memory_item.memory_id in self.memories:
                    memory_item.memory_id = str(uuid.uuid4())
                
                # 存储记忆
                self.memories[memory_item.memory_id] = memory_item
                
                # 更新索引
                self.type_index[memory_item.memory_type].append(memory_item.memory_id)
                for tag in memory_item.tags:
                    self.tag_index[tag].append(memory_item.memory_id)
                
                imported_count += 1
            
            self.logger.info(f"导入 {imported_count} 个记忆从: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"导入记忆失败: {e}")
            return False
    
    def clear_memories(self, memory_type: Optional[str] = None) -> int:
        """清除记忆"""
        if memory_type:
            # 清除指定类型的记忆
            memory_ids = self.type_index.get(memory_type, []).copy()
            cleared_count = 0
            
            for memory_id in memory_ids:
                if self.delete_memory(memory_id):
                    cleared_count += 1
            
            self.logger.info(f"清除 {cleared_count} 个 {memory_type} 类型的记忆")
            return cleared_count
        else:
            # 清除所有记忆
            cleared_count = len(self.memories)
            self.memories.clear()
            self.type_index.clear()
            self.tag_index.clear()
            self.recent_memories.clear()
            
            # 清除存储文件
            self._clear_storage()
            
            self.logger.info(f"清除所有 {cleared_count} 个记忆")
            return cleared_count
    
    def _cleanup_old_memories(self):
        """清理旧记忆"""
        if len(self.memories) <= self.max_memories:
            return
        
        # 按重要性和最后访问时间排序，移除最不重要且最久未访问的记忆
        sorted_memories = sorted(
            self.memories.items(),
            key=lambda x: (x[1].importance, x[1].last_accessed)
        )
        
        memories_to_remove = len(self.memories) - self.max_memories
        for i in range(memories_to_remove):
            memory_id, _ = sorted_memories[i]
            self.delete_memory(memory_id)
    
    def _save_memory(self, memory_item: SimpleMemoryItem):
        """保存记忆到文件"""
        try:
            file_path = self.storage_path / f"{memory_item.memory_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(memory_item.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存记忆失败: {e}")
    
    def _load_memories(self):
        """加载已存储的记忆"""
        if not self.storage_path.exists():
            return
        
        loaded_count = 0
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    memory_data = json.load(f)
                
                memory_item = SimpleMemoryItem.from_dict(memory_data)
                self.memories[memory_item.memory_id] = memory_item
                
                # 重建索引
                self.type_index[memory_item.memory_type].append(memory_item.memory_id)
                for tag in memory_item.tags:
                    self.tag_index[tag].append(memory_item.memory_id)
                
                loaded_count += 1
            except Exception as e:
                self.logger.error(f"加载记忆文件失败 {file_path}: {e}")
        
        if loaded_count > 0:
            self.logger.info(f"加载了 {loaded_count} 个记忆")
    
    def _delete_memory_file(self, memory_id: str):
        """删除记忆文件"""
        try:
            file_path = self.storage_path / f"{memory_id}.json"
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            self.logger.error(f"删除记忆文件失败: {e}")
    
    def _clear_storage(self):
        """清除存储目录"""
        try:
            for file_path in self.storage_path.glob("*.json"):
                file_path.unlink()
        except Exception as e:
            self.logger.error(f"清除存储目录失败: {e}")