# -*- coding: utf-8 -*-

# 记忆智能体 (Memory Agent)
# 专注于多层记忆管理、知识存储和检索的智能体实现
# 作者: RobotAgent开发团队
# 版本: 0.0.1 (Initial Release)
# 更新时间: 2025-08-21

# 导入标准库
import asyncio
import uuid
import json
import hashlib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
from pathlib import Path

# 导入项目基础组件
from .base_agent import BaseRobotAgent
from config import (
    MessageType, AgentMessage, MessagePriority, TaskMessage, ResponseMessage,
    TaskStatus, MemoryType, MemoryPriority, MemoryItem
)
from src.communication.protocols import (
    CollaborationMode, StatusMessage, MemoryMessage
)
from src.communication.message_bus import get_message_bus

# 导入记忆管理组件
from ..memory.graph_storage import GraphStorage
from ..memory.knowledge_retriever import KnowledgeRetriever, RetrievalMode
from ..memory import embedding_model

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


class MemoryType(Enum):
    """记忆类型枚举"""
    WORKING = "working"  # 工作记忆
    SHORT_TERM = "short_term"  # 短期记忆
    LONG_TERM = "long_term"  # 长期记忆
    EPISODIC = "episodic"  # 情节记忆
    SEMANTIC = "semantic"  # 语义记忆
    PROCEDURAL = "procedural"  # 程序记忆


class MemoryPriority(Enum):
    """记忆优先级枚举"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class MemoryItem:
    """记忆项数据结构"""
    id: str
    content: Any
    memory_type: MemoryType
    priority: MemoryPriority
    tags: List[str]
    source_agent: str
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    decay_factor: float = 1.0
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    related_memories: List[str] = None
    
    def __post_init__(self):
        if self.related_memories is None:
            self.related_memories = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "priority": self.priority.value,
            "tags": self.tags,
            "source_agent": self.source_agent,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "decay_factor": self.decay_factor,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "related_memories": self.related_memories
        }


@dataclass
class KnowledgeNode:
    """知识图谱节点"""
    id: str
    label: str
    node_type: str
    properties: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass
class KnowledgeEdge:
    """知识图谱边"""
    id: str
    source_id: str
    target_id: str
    relation_type: str
    properties: Dict[str, Any]
    weight: float = 1.0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class SearchQuery:
    """搜索查询结构"""
    query_text: str
    memory_types: List[MemoryType] = None
    tags: List[str] = None
    time_range: Tuple[datetime, datetime] = None
    similarity_threshold: float = 0.7
    max_results: int = 10
    include_related: bool = True


@dataclass
class SearchResult:
    """搜索结果结构"""
    memory_item: MemoryItem
    similarity_score: float
    relevance_score: float
    context_matches: List[str] = None


class MemoryAgent(BaseRobotAgent):
    """记忆管理智能体
    
    负责系统的记忆存储、检索和管理，是智能体系统的知识中心。
    主要功能包括：
    - 多层记忆存储（工作记忆、短期记忆、长期记忆）
    - 向量化记忆检索
    - 知识图谱构建与推理
    - 记忆整合与遗忘机制
    - 学习模式识别
    - 上下文关联分析
    """

    def __init__(self, agent_id: str = "memory_agent", config: Optional[Dict[str, Any]] = None):
        """初始化MemoryAgent
        
        Args:
            agent_id: 智能体ID
            config: 配置参数
        """
        super().__init__(agent_id, config)
        
        # 记忆存储
        self.working_memory: Dict[str, MemoryItem] = {}  # 工作记忆（容量有限）
        self.short_term_memory: Dict[str, MemoryItem] = {}  # 短期记忆
        self.long_term_memory: Dict[str, MemoryItem] = {}  # 长期记忆
        
        # 记忆索引
        self.memory_index: Dict[str, MemoryItem] = {}  # 全局记忆索引
        self.tag_index: Dict[str, List[str]] = defaultdict(list)  # 标签索引
        self.time_index: Dict[str, List[str]] = defaultdict(list)  # 时间索引
        
        # 知识图谱
        self.knowledge_nodes: Dict[str, KnowledgeNode] = {}
        self.knowledge_edges: Dict[str, KnowledgeEdge] = {}
        self.adjacency_list: Dict[str, List[str]] = defaultdict(list)
        
        # 向量存储（简化版本，实际应使用专业向量数据库）
        self.embeddings_cache: Dict[str, List[float]] = {}
        
        # 配置参数
        self.config = config or {}
        self.working_memory_capacity = self.config.get("working_memory_capacity", 20)
        self.short_term_retention_hours = self.config.get("short_term_retention_hours", 24)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.7)
        self.decay_rate = self.config.get("decay_rate", 0.1)
        
        # 知识图谱和向量检索组件
        storage_path = self.config.get("storage_path", "data/memory")
        self.graph_storage = GraphStorage(storage_path)
        self.knowledge_retriever = KnowledgeRetriever(self.graph_storage)
        
        # 对话历史管理
        self.conversation_history: List[Dict[str, Any]] = []
        self.max_conversation_history = self.config.get("max_conversation_history", 100)
        
        # 统计信息
        self.memory_stats = {
            "total_memories": 0,
            "working_memory_count": 0,
            "short_term_count": 0,
            "long_term_count": 0,
            "search_queries": 0,
            "successful_retrievals": 0,
            "knowledge_nodes": 0,
            "knowledge_edges": 0,
            "knowledge_extractions": 0
        }
        
        # 启动后台任务
        self._background_tasks = []
        self._start_background_tasks()
        
        self.logger.info(f"MemoryAgent {agent_id} 初始化完成")

    def _start_background_tasks(self):
        """启动后台任务"""
        # 记忆整合任务
        consolidation_task = asyncio.create_task(self._memory_consolidation_loop())
        self._background_tasks.append(consolidation_task)
        
        # 记忆衰减任务
        decay_task = asyncio.create_task(self._memory_decay_loop())
        self._background_tasks.append(decay_task)

    async def store_memory(self, content: Any, memory_type: MemoryType, 
                          source_agent: str, tags: List[str] = None, 
                          priority: MemoryPriority = MemoryPriority.MEDIUM,
                          metadata: Dict[str, Any] = None) -> str:
        """存储记忆
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型
            source_agent: 来源智能体
            tags: 标签列表
            priority: 优先级
            metadata: 元数据
            
        Returns:
            记忆ID
        """
        try:
            memory_id = self._generate_memory_id(content)
            
            # 创建记忆项
            memory_item = MemoryItem(
                id=memory_id,
                content=content,
                memory_type=memory_type,
                priority=priority,
                tags=tags or [],
                source_agent=source_agent,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                metadata=metadata or {}
            )
            
            # 生成向量嵌入
            memory_item.embedding = await self._generate_embedding(content)
            
            # 存储到相应的记忆层
            await self._store_to_memory_layer(memory_item)
            
            # 更新索引
            await self._update_indices(memory_item)
            
            # 更新知识图谱
            await self._update_knowledge_graph(memory_item)
            
            # 提取并存储知识到图谱
            await self._extract_and_store_knowledge(memory_item)
            
            # 更新统计信息
            self._update_memory_stats(memory_type)
            
            self.logger.info(f"记忆存储成功: {memory_id}")
            return memory_id
            
        except Exception as e:
            self.logger.error(f"记忆存储失败: {str(e)}")
            raise

    async def retrieve_memory(self, query: SearchQuery) -> List[SearchResult]:
        """检索记忆
        
        Args:
            query: 搜索查询
            
        Returns:
            搜索结果列表
        """
        try:
            self.memory_stats["search_queries"] += 1
            
            # 使用知识检索器进行多模式检索
            kg_results = await self._retrieve_from_knowledge_graph(query)
            
            # 传统记忆检索（作为补充）
            traditional_results = await self._traditional_memory_search(query)
            
            # 合并和去重结果
            combined_results = await self._merge_search_results(kg_results, traditional_results)
            
            # 更新访问记录
            await self._update_access_records(combined_results)
            
            self.memory_stats["successful_retrievals"] += len(combined_results)
            
            self.logger.info(f"记忆检索完成，返回 {len(combined_results)} 个结果")
            return combined_results[:query.max_results]
            
        except Exception as e:
            self.logger.error(f"记忆检索失败: {str(e)}")
            return []

    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """更新记忆
        
        Args:
            memory_id: 记忆ID
            updates: 更新内容
            
        Returns:
            是否更新成功
        """
        try:
            memory_item = self.memory_index.get(memory_id)
            if not memory_item:
                return False
            
            # 更新记忆项
            for key, value in updates.items():
                if hasattr(memory_item, key):
                    setattr(memory_item, key, value)
            
            memory_item.last_accessed = datetime.now()
            
            # 如果内容更新，重新生成嵌入
            if 'content' in updates:
                memory_item.embedding = await self._generate_embedding(memory_item.content)
            
            self.logger.info(f"记忆更新成功: {memory_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"记忆更新失败: {str(e)}")
            return False

    async def delete_memory(self, memory_id: str) -> bool:
        """删除记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            是否删除成功
        """
        try:
            memory_item = self.memory_index.get(memory_id)
            if not memory_item:
                return False
            
            # 从各层记忆中删除
            memory_type = memory_item.memory_type
            if memory_type == MemoryType.WORKING:
                self.working_memory.pop(memory_id, None)
            elif memory_type == MemoryType.SHORT_TERM:
                self.short_term_memory.pop(memory_id, None)
            elif memory_type == MemoryType.LONG_TERM:
                self.long_term_memory.pop(memory_id, None)
            
            # 从索引中删除
            self.memory_index.pop(memory_id, None)
            
            # 更新标签索引
            for tag in memory_item.tags:
                if memory_id in self.tag_index[tag]:
                    self.tag_index[tag].remove(memory_id)
            
            self.logger.info(f"记忆删除成功: {memory_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"记忆删除失败: {str(e)}")
            return False

    async def get_memory_summary(self) -> Dict[str, Any]:
        """获取记忆系统摘要
        
        Returns:
            记忆系统统计信息
        """
        # 获取图存储统计信息
        graph_stats = await self.graph_storage.get_statistics() if hasattr(self, 'graph_storage') else {}
        retriever_stats = await self.knowledge_retriever.get_statistics() if hasattr(self, 'knowledge_retriever') else {}
        
        return {
            "statistics": self.memory_stats.copy(),
            "memory_distribution": {
                "working_memory": len(self.working_memory),
                "short_term_memory": len(self.short_term_memory),
                "long_term_memory": len(self.long_term_memory)
            },
            "knowledge_graph": {
                "nodes": len(self.knowledge_nodes),
                "edges": len(self.knowledge_edges)
            },
            "graph_storage_stats": graph_stats,
            "retriever_stats": retriever_stats,
            "conversation_history_length": len(getattr(self, 'conversation_history', [])),
            "top_tags": self._get_top_tags(10),
            "recent_activity": await self._get_recent_activity()
        }

    def _generate_memory_id(self, content: Any) -> str:
        """生成记忆ID"""
        content_str = json.dumps(content, sort_keys=True, default=str)
        hash_obj = hashlib.md5(content_str.encode())
        return f"mem_{hash_obj.hexdigest()[:12]}"

    async def _generate_embedding(self, content: Any) -> List[float]:
        """生成内容的向量嵌入（简化版本）"""
        # 这里是简化的嵌入生成，实际应使用专业的嵌入模型
        content_str = str(content).lower()
        
        # 简单的词频向量（实际应使用BERT、OpenAI等模型）
        words = content_str.split()
        vocab_size = 1000  # 简化的词汇表大小
        embedding = [0.0] * vocab_size
        
        for word in words:
            # 简单的哈希映射到向量维度
            index = hash(word) % vocab_size
            embedding[index] += 1.0
        
        # 归一化
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding

    async def _store_to_memory_layer(self, memory_item: MemoryItem):
        """存储到相应的记忆层"""
        memory_type = memory_item.memory_type
        memory_id = memory_item.id
        
        if memory_type == MemoryType.WORKING:
            # 工作记忆容量限制
            if len(self.working_memory) >= self.working_memory_capacity:
                await self._evict_working_memory()
            self.working_memory[memory_id] = memory_item
            
        elif memory_type == MemoryType.SHORT_TERM:
            self.short_term_memory[memory_id] = memory_item
            
        elif memory_type == MemoryType.LONG_TERM:
            self.long_term_memory[memory_id] = memory_item
        
        # 添加到全局索引
        self.memory_index[memory_id] = memory_item

    async def _evict_working_memory(self):
        """工作记忆淘汰策略"""
        # 基于LRU策略淘汰最少使用的记忆
        if not self.working_memory:
            return
        
        # 找到最少访问的记忆
        oldest_memory = min(
            self.working_memory.values(),
            key=lambda x: (x.last_accessed, x.access_count)
        )
        
        # 将其转移到短期记忆
        oldest_memory.memory_type = MemoryType.SHORT_TERM
        self.short_term_memory[oldest_memory.id] = oldest_memory
        del self.working_memory[oldest_memory.id]
        
        self.logger.debug(f"工作记忆淘汰: {oldest_memory.id}")

    async def _update_indices(self, memory_item: MemoryItem):
        """更新索引"""
        memory_id = memory_item.id
        
        # 更新标签索引
        for tag in memory_item.tags:
            self.tag_index[tag].append(memory_id)
        
        # 更新时间索引
        date_key = memory_item.created_at.strftime("%Y-%m-%d")
        self.time_index[date_key].append(memory_id)

    async def _update_knowledge_graph(self, memory_item: MemoryItem):
        """更新知识图谱"""
        try:
            # 创建知识节点
            node_id = f"node_{memory_item.id}"
            knowledge_node = KnowledgeNode(
                id=node_id,
                label=str(memory_item.content)[:100],  # 截取前100字符作为标签
                node_type=memory_item.memory_type.value,
                properties={
                    "memory_id": memory_item.id,
                    "tags": memory_item.tags,
                    "priority": memory_item.priority.value,
                    "source_agent": memory_item.source_agent
                },
                created_at=memory_item.created_at,
                updated_at=datetime.now()
            )
            
            self.knowledge_nodes[node_id] = knowledge_node
            self.memory_stats["knowledge_nodes"] += 1
            
            # 基于标签创建关联边
            await self._create_knowledge_edges(memory_item, node_id)
            
        except Exception as e:
            self.logger.error(f"知识图谱更新失败: {str(e)}")

    async def _create_knowledge_edges(self, memory_item: MemoryItem, node_id: str):
        """创建知识图谱边"""
        # 查找具有相同标签的其他节点
        for tag in memory_item.tags:
            related_memory_ids = self.tag_index.get(tag, [])
            
            for related_id in related_memory_ids:
                if related_id != memory_item.id:
                    related_node_id = f"node_{related_id}"
                    
                    if related_node_id in self.knowledge_nodes:
                        # 创建关联边
                        edge_id = f"edge_{node_id}_{related_node_id}"
                        
                        if edge_id not in self.knowledge_edges:
                            knowledge_edge = KnowledgeEdge(
                                id=edge_id,
                                source_id=node_id,
                                target_id=related_node_id,
                                relation_type="tag_similarity",
                                properties={"shared_tag": tag},
                                weight=1.0
                            )
                            
                            self.knowledge_edges[edge_id] = knowledge_edge
                            self.adjacency_list[node_id].append(related_node_id)
                            self.adjacency_list[related_node_id].append(node_id)
                            self.memory_stats["knowledge_edges"] += 1

    async def _multi_layer_search(self, query: SearchQuery, query_embedding: List[float]) -> List[MemoryItem]:
        """多层记忆搜索"""
        candidates = []
        
        # 搜索各层记忆
        memory_layers = {
            MemoryType.WORKING: self.working_memory,
            MemoryType.SHORT_TERM: self.short_term_memory,
            MemoryType.LONG_TERM: self.long_term_memory
        }
        
        for memory_type, memory_layer in memory_layers.items():
            if query.memory_types is None or memory_type in query.memory_types:
                layer_candidates = await self._search_memory_layer(
                    memory_layer, query, query_embedding
                )
                candidates.extend(layer_candidates)
        
        return candidates

    async def _search_memory_layer(self, memory_layer: Dict[str, MemoryItem], 
                                  query: SearchQuery, query_embedding: List[float]) -> List[MemoryItem]:
        """搜索单层记忆"""
        candidates = []
        
        for memory_item in memory_layer.values():
            # 标签过滤
            if query.tags and not any(tag in memory_item.tags for tag in query.tags):
                continue
            
            # 时间范围过滤
            if query.time_range:
                start_time, end_time = query.time_range
                if not (start_time <= memory_item.created_at <= end_time):
                    continue
            
            # 向量相似度计算
            if memory_item.embedding:
                similarity = self._calculate_cosine_similarity(
                    query_embedding, memory_item.embedding
                )
                
                if similarity >= query.similarity_threshold:
                    candidates.append(memory_item)
        
        return candidates

    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    async def _rank_search_results(self, candidates: List[MemoryItem], 
                                  query_embedding: List[float], 
                                  query: SearchQuery) -> List[SearchResult]:
        """对搜索结果进行排序"""
        results = []
        
        for memory_item in candidates:
            # 计算相似度分数
            similarity_score = self._calculate_cosine_similarity(
                query_embedding, memory_item.embedding or []
            )
            
            # 计算相关性分数（综合多个因素）
            relevance_score = self._calculate_relevance_score(memory_item, query)
            
            search_result = SearchResult(
                memory_item=memory_item,
                similarity_score=similarity_score,
                relevance_score=relevance_score
            )
            
            results.append(search_result)
        
        # 按相关性分数排序
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results

    def _calculate_relevance_score(self, memory_item: MemoryItem, query: SearchQuery) -> float:
        """计算相关性分数"""
        score = 0.0
        
        # 基础相似度分数
        if memory_item.embedding:
            query_embedding = asyncio.create_task(self._generate_embedding(query.query_text))
            # 这里简化处理，实际应该异步计算
            score += 0.4  # 占40%权重
        
        # 优先级权重
        priority_weight = memory_item.priority.value / 4.0  # 归一化到0-1
        score += priority_weight * 0.2  # 占20%权重
        
        # 访问频率权重
        access_weight = min(memory_item.access_count / 10.0, 1.0)  # 归一化
        score += access_weight * 0.2  # 占20%权重
        
        # 时间新鲜度权重
        time_diff = (datetime.now() - memory_item.created_at).total_seconds()
        freshness = max(0, 1 - time_diff / (7 * 24 * 3600))  # 7天内的新鲜度
        score += freshness * 0.2  # 占20%权重
        
        return score

    async def _update_access_records(self, results: List[SearchResult]):
        """更新访问记录"""
        for result in results:
            memory_item = result.memory_item
            memory_item.last_accessed = datetime.now()
            memory_item.access_count += 1

    async def _expand_with_related_memories(self, results: List[SearchResult]) -> List[SearchResult]:
        """扩展相关记忆"""
        expanded_results = results.copy()
        
        for result in results[:5]:  # 只对前5个结果扩展
            memory_item = result.memory_item
            
            # 通过知识图谱查找相关记忆
            node_id = f"node_{memory_item.id}"
            if node_id in self.adjacency_list:
                related_node_ids = self.adjacency_list[node_id][:3]  # 最多3个相关
                
                for related_node_id in related_node_ids:
                    related_memory_id = related_node_id.replace("node_", "")
                    related_memory = self.memory_index.get(related_memory_id)
                    
                    if related_memory and not any(r.memory_item.id == related_memory_id for r in expanded_results):
                        related_result = SearchResult(
                            memory_item=related_memory,
                            similarity_score=result.similarity_score * 0.8,  # 降低相关记忆的分数
                            relevance_score=result.relevance_score * 0.8
                        )
                        expanded_results.append(related_result)
        
        return expanded_results

    def _update_memory_stats(self, memory_type: MemoryType):
        """更新记忆统计信息"""
        self.memory_stats["total_memories"] += 1
        
        if memory_type == MemoryType.WORKING:
            self.memory_stats["working_memory_count"] += 1
        elif memory_type == MemoryType.SHORT_TERM:
            self.memory_stats["short_term_count"] += 1
        elif memory_type == MemoryType.LONG_TERM:
            self.memory_stats["long_term_count"] += 1

    def _get_top_tags(self, limit: int) -> List[Tuple[str, int]]:
        """获取最常用的标签"""
        tag_counts = {tag: len(memory_ids) for tag, memory_ids in self.tag_index.items()}
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_tags[:limit]

    async def _get_recent_activity(self) -> List[Dict[str, Any]]:
        """获取最近活动"""
        recent_memories = sorted(
            self.memory_index.values(),
            key=lambda x: x.last_accessed,
            reverse=True
        )[:10]
        
        return [{
            "memory_id": mem.id,
            "content_preview": str(mem.content)[:100],
            "last_accessed": mem.last_accessed.isoformat(),
            "access_count": mem.access_count
        } for mem in recent_memories]

    async def _memory_consolidation_loop(self):
        """记忆整合循环（后台任务）"""
        while True:
            try:
                await asyncio.sleep(3600)  # 每小时执行一次
                await self._consolidate_memories()
            except Exception as e:
                self.logger.error(f"记忆整合任务异常: {str(e)}")

    async def _memory_decay_loop(self):
        """记忆衰减循环（后台任务）"""
        while True:
            try:
                await asyncio.sleep(1800)  # 每30分钟执行一次
                await self._apply_memory_decay()
            except Exception as e:
                self.logger.error(f"记忆衰减任务异常: {str(e)}")

    async def _consolidate_memories(self):
        """记忆整合处理"""
        # 将重要的短期记忆转移到长期记忆
        consolidation_candidates = []
        
        for memory_item in self.short_term_memory.values():
            # 整合条件：高优先级、高访问频率、或存在时间超过阈值
            age_hours = (datetime.now() - memory_item.created_at).total_seconds() / 3600
            
            if (memory_item.priority.value >= MemoryPriority.HIGH.value or 
                memory_item.access_count >= 5 or 
                age_hours >= self.short_term_retention_hours):
                consolidation_candidates.append(memory_item)
        
        # 执行整合
        for memory_item in consolidation_candidates:
            memory_item.memory_type = MemoryType.LONG_TERM
            self.long_term_memory[memory_item.id] = memory_item
            del self.short_term_memory[memory_item.id]
            
            self.logger.debug(f"记忆整合: {memory_item.id} -> 长期记忆")

    async def _apply_memory_decay(self):
        """应用记忆衰减"""
        current_time = datetime.now()
        
        for memory_item in list(self.short_term_memory.values()):
            # 计算衰减
            age_hours = (current_time - memory_item.last_accessed).total_seconds() / 3600
            decay_factor = max(0.1, 1.0 - (age_hours * self.decay_rate / 24))
            memory_item.decay_factor = decay_factor
            
            # 如果衰减过度且不重要，考虑删除
            if (decay_factor < 0.3 and 
                memory_item.priority.value < MemoryPriority.HIGH.value and 
                memory_item.access_count < 2):
                await self.delete_memory(memory_item.id)
                self.logger.debug(f"记忆衰减删除: {memory_item.id}")

    async def _handle_collaboration_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """处理协作请求"""
        try:
            request_data = json.loads(message.content)
            request_type = request_data.get("type")
            
            if request_type == "store_memory":
                result = await self._handle_store_request(request_data)
            elif request_type == "retrieve_memory":
                result = await self._handle_retrieve_request(request_data)
            elif request_type == "get_summary":
                result = await self.get_memory_summary()
            else:
                result = {"error": f"未知请求类型: {request_type}"}
            
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

    async def _handle_store_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理存储请求"""
        try:
            memory_id = await self.store_memory(
                content=request_data.get("content"),
                memory_type=MemoryType(request_data.get("memory_type", "short_term")),
                source_agent=request_data.get("source_agent"),
                tags=request_data.get("tags", []),
                priority=MemoryPriority(request_data.get("priority", 2)),
                metadata=request_data.get("metadata", {})
            )
            
            return {"success": True, "memory_id": memory_id}
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_retrieve_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理检索请求"""
        try:
            query = SearchQuery(
                query_text=request_data.get("query_text"),
                memory_types=[MemoryType(t) for t in request_data.get("memory_types", [])],
                tags=request_data.get("tags"),
                similarity_threshold=request_data.get("similarity_threshold", 0.7),
                max_results=request_data.get("max_results", 10)
            )
            
            results = await self.retrieve_memory(query)
            
            return {
                "success": True,
                "results": [{
                    "memory_item": result.memory_item.to_dict(),
                    "similarity_score": result.similarity_score,
                    "relevance_score": result.relevance_score
                } for result in results]
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _extract_and_store_knowledge(self, memory_item: MemoryItem):
        """从记忆项中提取知识并存储到图谱"""
        try:
            content_str = str(memory_item.content)
            
            # 提取实体
            entities = await self._extract_entities_advanced(content_str)
            
            # 提取关系
            relationships = await self._extract_relationships_advanced(content_str, entities)
            
            # 存储到图谱
            for entity in entities:
                await self.graph_storage.add_entity(
                    entity_id=entity["id"],
                    entity_type=entity["type"],
                    properties=entity["properties"]
                )
                
                # 为重要实体创建文档向量存储
                if entity["importance"] > 0.7:
                    await self.graph_storage.add_entity_document(
                        entity["id"], content_str, memory_item.metadata or {}
                    )
            
            for relationship in relationships:
                await self.graph_storage.add_relationship(
                    source_id=relationship["source"],
                    target_id=relationship["target"],
                    relation_type=relationship["type"],
                    properties=relationship["properties"]
                )
            
            self.memory_stats["knowledge_extractions"] += 1
            
        except Exception as e:
            self.logger.error(f"知识提取失败: {str(e)}")
    
    async def _extract_entities_advanced(self, content: str) -> List[Dict[str, Any]]:
        """高级实体提取"""
        entities = []
        
        # 简化的实体提取逻辑（实际应使用NER模型）
        words = content.split()
        for i, word in enumerate(words):
            if len(word) > 3 and word.isalpha():
                entity = {
                    "id": f"entity_{hash(word) % 10000}",
                    "type": "concept",
                    "properties": {
                        "name": word,
                        "context": " ".join(words[max(0, i-2):i+3]),
                        "position": i
                    },
                    "importance": min(1.0, len(word) / 10.0)
                }
                entities.append(entity)
        
        return entities[:10]  # 限制数量
    
    async def _extract_relationships_advanced(self, content: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """高级关系提取"""
        relationships = []
        
        # 简化的关系提取逻辑
        for i in range(len(entities) - 1):
            for j in range(i + 1, min(i + 3, len(entities))):
                relationship = {
                    "source": entities[i]["id"],
                    "target": entities[j]["id"],
                    "type": "co_occurrence",
                    "properties": {
                        "distance": j - i,
                        "context": content[:200]
                    }
                }
                relationships.append(relationship)
        
        return relationships
    
    async def _retrieve_from_knowledge_graph(self, query: SearchQuery) -> List[SearchResult]:
        """从知识图谱检索记忆"""
        try:
            # 使用不同的检索模式
            results = []
            
            # 快速检索
            quick_results = await self.knowledge_retriever.retrieve(
                query.query_text, RetrievalMode.QUICK, max_results=query.max_results // 2
            )
            
            # 联想检索
            associate_results = await self.knowledge_retriever.retrieve(
                query.query_text, RetrievalMode.ASSOCIATE, max_results=query.max_results // 2
            )
            
            # 转换为SearchResult格式
            for item in quick_results + associate_results:
                if item.memory_id in self.memory_index:
                    memory_item = self.memory_index[item.memory_id]
                    search_result = SearchResult(
                        memory_item=memory_item,
                        similarity_score=item.similarity_score,
                        relevance_score=item.relevance_score
                    )
                    results.append(search_result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"知识图谱检索失败: {str(e)}")
            return []
    
    async def _traditional_memory_search(self, query: SearchQuery) -> List[SearchResult]:
        """传统记忆搜索（原有逻辑）"""
        try:
            # 生成查询向量
            query_embedding = await self._generate_embedding(query.query_text)
            
            # 多层检索
            candidates = await self._multi_layer_search(query, query_embedding)
            
            # 相似度计算和排序
            results = await self._rank_search_results(candidates, query_embedding, query)
            
            return results
            
        except Exception as e:
            self.logger.error(f"传统记忆搜索失败: {str(e)}")
            return []
    
    async def _merge_search_results(self, kg_results: List[SearchResult], 
                                   traditional_results: List[SearchResult]) -> List[SearchResult]:
        """合并搜索结果并去重"""
        seen_ids = set()
        merged_results = []
        
        # 优先使用知识图谱结果
        for result in kg_results:
            if result.memory_item.id not in seen_ids:
                seen_ids.add(result.memory_item.id)
                merged_results.append(result)
        
        # 添加传统搜索结果
        for result in traditional_results:
            if result.memory_item.id not in seen_ids:
                seen_ids.add(result.memory_item.id)
                merged_results.append(result)
        
        # 按相关性分数排序
        merged_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return merged_results
    
    async def get_relevant_memories_for_prompt(self, query: str, max_memories: int = 5) -> str:
        """获取相关记忆并格式化为提示词"""
        try:
            search_query = SearchQuery(
                query_text=query,
                max_results=max_memories,
                similarity_threshold=0.6
            )
            
            results = await self.retrieve_memory(search_query)
            
            if not results:
                return "暂无相关记忆。"
            
            # 格式化为提示词
            formatted_memories = []
            for i, result in enumerate(results, 1):
                memory_content = str(result.memory_item.content)[:200]
                formatted_memory = f"{i}. {memory_content}... (相关度: {result.relevance_score:.2f})"
                formatted_memories.append(formatted_memory)
            
            return "\n".join(formatted_memories)
            
        except Exception as e:
            self.logger.error(f"获取相关记忆失败: {str(e)}")
            return "记忆检索出现错误。"
    
    async def add_conversation_turn(self, user_input: str, agent_response: str, agent_id: str):
        """添加对话轮次并管理对话历史"""
        try:
            conversation_turn = {
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "agent_response": agent_response,
                "agent_id": agent_id
            }
            
            self.conversation_history.append(conversation_turn)
            
            # 限制对话历史长度
            if len(self.conversation_history) > self.max_conversation_history:
                self.conversation_history = self.conversation_history[-self.max_conversation_history:]
            
            # 异步提取对话知识
            asyncio.create_task(self._extract_conversation_knowledge(conversation_turn))
            
        except Exception as e:
            self.logger.error(f"添加对话轮次失败: {str(e)}")
    
    async def _extract_conversation_knowledge(self, conversation_turn: Dict[str, Any]):
        """从对话中提取知识"""
        try:
            # 存储用户输入记忆
            await self.store_memory(
                content=conversation_turn["user_input"],
                memory_type=MemoryType.SHORT_TERM,
                source_agent="user",
                tags=["conversation", "user_input"],
                priority=MemoryPriority.MEDIUM,
                metadata={
                    "timestamp": conversation_turn["timestamp"],
                    "conversation_turn": True
                }
            )
            
            # 存储智能体响应记忆
            await self.store_memory(
                content=conversation_turn["agent_response"],
                memory_type=MemoryType.SHORT_TERM,
                source_agent=conversation_turn["agent_id"],
                tags=["conversation", "agent_response"],
                priority=MemoryPriority.MEDIUM,
                metadata={
                    "timestamp": conversation_turn["timestamp"],
                    "conversation_turn": True
                }
            )
            
        except Exception as e:
            self.logger.error(f"对话知识提取失败: {str(e)}")
    
    async def save_memory_data(self):
        """保存记忆数据"""
        try:
            await self.graph_storage.save()
            self.logger.info("记忆数据保存成功")
        except Exception as e:
            self.logger.error(f"记忆数据保存失败: {str(e)}")
    
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行记忆相关任务"""
        task_type = task_data.get("type")
        
        if task_type == "store_memory":
            return await self._handle_store_request(task_data)
        elif task_type == "retrieve_memory":
            return await self._handle_retrieve_request(task_data)
        elif task_type == "get_summary":
            return {"success": True, "summary": await self.get_memory_summary()}
        else:
            return {
                "success": False,
                "error": f"Unknown task type: {task_type}"
            }
    
    async def cleanup(self):
        """清理资源"""
        # 保存记忆数据
        await self.save_memory_data()
        
        # 清理图存储和检索缓存
        if hasattr(self, 'graph_storage') and self.graph_storage:
            # GraphStorage清理方法（如果存在）
            if hasattr(self.graph_storage, 'cleanup'):
                try:
                    cleanup_result = self.graph_storage.cleanup()
                    if hasattr(cleanup_result, '__await__'):
                        await cleanup_result
                except Exception as e:
                    self.logger.warning(f"图存储清理失败: {e}")
        
        if hasattr(self, 'knowledge_retriever') and self.knowledge_retriever:
            # KnowledgeRetriever清理方法（如果存在）
            if hasattr(self.knowledge_retriever, 'cleanup'):
                try:
                    cleanup_result = self.knowledge_retriever.cleanup()
                    if hasattr(cleanup_result, '__await__'):
                        await cleanup_result
                except Exception as e:
                    self.logger.warning(f"知识检索器清理失败: {e}")
        
        # 取消后台任务
        for task in self._background_tasks:
            task.cancel()
        
        # 等待任务完成
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self.logger.info("MemoryAgent 资源清理完成")