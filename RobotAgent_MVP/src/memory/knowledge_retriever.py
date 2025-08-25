# -*- coding: utf-8 -*-

# 知识检索器 (Knowledge Retriever)
# 实现基于向量相似度的知识检索功能，支持多种检索模式
# 作者: RobotAgent开发团队
# 版本: 0.0.2 (Bug Fix Release)
# 更新时间: 2025年08月25日

import time
from typing import List, Dict, Tuple, Optional, Set, Any
from enum import Enum
from collections import defaultdict
from datetime import datetime, timedelta

from .graph_storage import GraphStorage
from .embedding_model import embedding_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RetrievalMode(Enum):
    """检索模式枚举"""
    QUICK = "quick"          # 快速检索
    ASSOCIATIVE = "associative"  # 联想检索
    RELATIONAL = "relational"    # 关系检索
    COMPREHENSIVE = "comprehensive"  # 综合检索


class MemoryItem:
    
    # 记忆项数据结构 (Memory Item Data Structure)
    
    # 表示单个记忆项的数据结构，包含内容、实体ID、元数据等信息。
    # 主要功能包括：
    # 1. 记忆内容存储
    # 2. 实体关联管理
    # 3. 元数据维护
    # 4. 相关性评分
    
    # 继承自object，实现了记忆项的基础数据结构。
    
    def __init__(self, content: str, entity_id: Optional[str] = None, 
                 metadata: Optional[Dict] = None, score: float = 0.0):
        self.content = content
        self.entity_id = entity_id
        self.metadata = metadata or {}
        self.score = score
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'content': self.content,
            'entity_id': self.entity_id,
            'metadata': self.metadata,
            'score': self.score,
            'timestamp': self.timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        return f"MemoryItem(content='{self.content[:50]}...', score={self.score:.3f})"


class KnowledgeRetriever:
    
    # 知识检索器 (Knowledge Retriever)
    
    # 实现基于向量相似度的知识检索功能，支持多种检索模式。
    # 为对话系统提供相关记忆和知识的智能检索服务，主要功能包括：
    # 1. 快速向量检索
    # 2. 联想式检索
    # 3. 关系网络检索
    # 4. 记忆相关性排序
    
    # 继承自object，实现了完整的知识检索功能。
    #
    
    def __init__(self, graph_storage: GraphStorage, cache_size: int = 1000):
        """初始化知识检索器
        
        Args:
            graph_storage: 图存储实例
            cache_size: 缓存大小
        """
        self.graph_storage = graph_storage
        self.cache_size = cache_size
        
        # 检索缓存
        self.retrieval_cache: Dict[str, Tuple[List[MemoryItem], float]] = {}
        self.cache_access_times: Dict[str, float] = {}
        
        # 检索统计
        self.retrieval_stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'avg_retrieval_time': 0.0
        }
    
    def retrieve_memories(self, query: str, mode: RetrievalMode = RetrievalMode.COMPREHENSIVE,
                         max_results: int = 5, score_threshold: float = 0.3,
                         use_cache: bool = True) -> List[MemoryItem]:
        """检索相关记忆
        
        Args:
            query: 查询文本
            mode: 检索模式
            max_results: 最大结果数量
            score_threshold: 分数阈值
            use_cache: 是否使用缓存
            
        Returns:
            相关记忆列表
        """
        start_time = time.time()
        self.retrieval_stats['total_queries'] += 1
        
        # 生成缓存键
        cache_key = f"{query}_{mode.value}_{max_results}_{score_threshold}"
        
        # 检查缓存
        if use_cache and cache_key in self.retrieval_cache:
            cached_results, cache_time = self.retrieval_cache[cache_key]
            # 缓存有效期为5分钟
            if time.time() - cache_time < 300:
                self.retrieval_stats['cache_hits'] += 1
                self.cache_access_times[cache_key] = time.time()
                logger.debug(f"缓存命中: {cache_key}")
                return cached_results[:max_results]
        
        # 执行检索
        try:
            if mode == RetrievalMode.QUICK:
                results = self._quick_retrieval(query, max_results, score_threshold)
            elif mode == RetrievalMode.ASSOCIATIVE:
                results = self._associative_retrieval(query, max_results, score_threshold)
            elif mode == RetrievalMode.RELATIONAL:
                results = self._relational_retrieval(query, max_results, score_threshold)
            else:  # COMPREHENSIVE
                results = self._comprehensive_retrieval(query, max_results, score_threshold)
            
            # 更新缓存
            if use_cache:
                self._update_cache(cache_key, results)
            
            # 更新统计信息
            retrieval_time = time.time() - start_time
            self._update_stats(retrieval_time)
            
            logger.debug(f"检索完成: 查询='{query}', 模式={mode.value}, 结果数={len(results)}, 耗时={retrieval_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []
    
    def _quick_retrieval(self, query: str, max_results: int, 
                        score_threshold: float) -> List[MemoryItem]:
        """快速检索模式
        
        基于实体嵌入的快速相似度搜索
        """
        results = []
        
        # 搜索相似实体
        similar_entities = self.graph_storage.search_similar_entities(
            query, k=max_results * 2, threshold=score_threshold
        )
        
        for entity_id, similarity in similar_entities:
            entity_data = self.graph_storage.get_entity(entity_id)
            if entity_data:
                content = self._entity_to_content(entity_id, entity_data)
                memory_item = MemoryItem(
                    content=content,
                    entity_id=entity_id,
                    metadata={'similarity': similarity, 'type': 'entity'},
                    score=similarity
                )
                results.append(memory_item)
        
        return self._rank_and_filter_results(results, max_results)
    
    def _associative_retrieval(self, query: str, max_results: int,
                              score_threshold: float) -> List[MemoryItem]:
        """联想检索模式
        
        基于实体关系网络的联想式检索
        """
        results = []
        
        # 首先进行快速检索获取种子实体
        seed_entities = self.graph_storage.search_similar_entities(
            query, k=3, threshold=score_threshold
        )
        
        processed_entities = set()
        
        for entity_id, base_score in seed_entities:
            if entity_id in processed_entities:
                continue
            
            processed_entities.add(entity_id)
            
            # 添加种子实体
            entity_data = self.graph_storage.get_entity(entity_id)
            if entity_data:
                content = self._entity_to_content(entity_id, entity_data)
                memory_item = MemoryItem(
                    content=content,
                    entity_id=entity_id,
                    metadata={'base_score': base_score, 'type': 'seed_entity'},
                    score=base_score
                )
                results.append(memory_item)
            
            # 获取相关实体
            relationships = self.graph_storage.get_relationships(entity_id)
            
            for rel in relationships:
                related_entity_id = rel['target'] if rel['source'] == entity_id else rel['source']
                
                if related_entity_id in processed_entities:
                    continue
                
                processed_entities.add(related_entity_id)
                
                # 计算关联分数
                relation_weight = self._get_relation_weight(rel['relation_type'])
                associated_score = base_score * relation_weight * 0.8  # 关联衰减
                
                if associated_score >= score_threshold:
                    related_entity_data = self.graph_storage.get_entity(related_entity_id)
                    if related_entity_data:
                        content = self._entity_to_content(related_entity_id, related_entity_data)
                        memory_item = MemoryItem(
                            content=content,
                            entity_id=related_entity_id,
                            metadata={
                                'base_score': base_score,
                                'relation_type': rel['relation_type'],
                                'relation_weight': relation_weight,
                                'type': 'associated_entity'
                            },
                            score=associated_score
                        )
                        results.append(memory_item)
        
        return self._rank_and_filter_results(results, max_results)
    
    def _relational_retrieval(self, query: str, max_results: int,
                             score_threshold: float) -> List[MemoryItem]:
        """关系检索模式
        
        重点关注实体间的关系信息
        """
        results = []
        
        # 搜索相关实体
        relevant_entities = self.graph_storage.search_similar_entities(
            query, k=max_results, threshold=score_threshold
        )
        
        for entity_id, similarity in relevant_entities:
            # 获取实体的所有关系
            relationships = self.graph_storage.get_relationships(entity_id)
            
            # 为每个关系创建记忆项
            for rel in relationships:
                relation_content = self._relationship_to_content(rel)
                
                memory_item = MemoryItem(
                    content=relation_content,
                    entity_id=entity_id,
                    metadata={
                        'similarity': similarity,
                        'relation_type': rel['relation_type'],
                        'source': rel['source'],
                        'target': rel['target'],
                        'type': 'relationship'
                    },
                    score=similarity * 0.9  # 关系信息稍微降权
                )
                results.append(memory_item)
        
        return self._rank_and_filter_results(results, max_results)
    
    def _comprehensive_retrieval(self, query: str, max_results: int,
                                score_threshold: float) -> List[MemoryItem]:
        """综合检索模式
        
        结合多种检索策略的综合检索
        """
        all_results = []
        
        # 快速检索 (权重: 0.4)
        quick_results = self._quick_retrieval(query, max_results, score_threshold)
        for item in quick_results:
            item.score *= 0.4
            item.metadata['retrieval_type'] = 'quick'
            all_results.append(item)
        
        # 联想检索 (权重: 0.4)
        associative_results = self._associative_retrieval(query, max_results, score_threshold)
        for item in associative_results:
            item.score *= 0.4
            item.metadata['retrieval_type'] = 'associative'
            all_results.append(item)
        
        # 关系检索 (权重: 0.2)
        relational_results = self._relational_retrieval(query, max_results, score_threshold)
        for item in relational_results:
            item.score *= 0.2
            item.metadata['retrieval_type'] = 'relational'
            all_results.append(item)
        
        # 去重和合并
        merged_results = self._merge_duplicate_results(all_results)
        
        return self._rank_and_filter_results(merged_results, max_results)
    
    def _entity_to_content(self, entity_id: str, entity_data: Dict) -> str:
        """将实体转换为内容文本"""
        content_parts = [f"实体: {entity_id}"]
        
        for key, value in entity_data.items():
            if key not in ['created_at', 'updated_at'] and value:
                content_parts.append(f"{key}: {value}")
        
        return "; ".join(content_parts)
    
    def _relationship_to_content(self, relationship: Dict) -> str:
        """将关系转换为内容文本"""
        source = relationship.get('source', '')
        target = relationship.get('target', '')
        relation_type = relationship.get('relation_type', '')
        
        content = f"关系: {source} {relation_type} {target}"
        
        # 添加其他属性
        for key, value in relationship.items():
            if key not in ['source', 'target', 'relation_type', 'created_at'] and value:
                content += f"; {key}: {value}"
        
        return content
    
    def _get_relation_weight(self, relation_type: str) -> float:
        """获取关系类型的权重"""
        weights = {
            'related_to': 0.8,
            'part_of': 0.9,
            'similar_to': 0.7,
            'caused_by': 0.8,
            'leads_to': 0.8,
            'mentioned_with': 0.6,
            'default': 0.5
        }
        return weights.get(relation_type, weights['default'])
    
    def _rank_and_filter_results(self, results: List[MemoryItem], 
                                max_results: int) -> List[MemoryItem]:
        """对结果进行排序和过滤"""
        # 按分数排序
        results.sort(key=lambda x: x.score, reverse=True)
        
        # 返回前max_results个结果
        return results[:max_results]
    
    def _merge_duplicate_results(self, results: List[MemoryItem]) -> List[MemoryItem]:
        """合并重复的结果"""
        merged = {}
        
        for item in results:
            key = f"{item.entity_id}_{item.content[:50]}"
            
            if key in merged:
                # 合并分数（取最高分）
                if item.score > merged[key].score:
                    merged[key] = item
            else:
                merged[key] = item
        
        return list(merged.values())
    
    def _update_cache(self, cache_key: str, results: List[MemoryItem]) -> None:
        """更新检索缓存"""
        # 清理过期缓存
        if len(self.retrieval_cache) >= self.cache_size:
            self._cleanup_cache()
        
        self.retrieval_cache[cache_key] = (results.copy(), time.time())
        self.cache_access_times[cache_key] = time.time()
    
    def _cleanup_cache(self) -> None:
        """清理过期缓存"""
        current_time = time.time()
        
        # 移除5分钟前的缓存
        expired_keys = [
            key for key, (_, cache_time) in self.retrieval_cache.items()
            if current_time - cache_time > 300
        ]
        
        for key in expired_keys:
            del self.retrieval_cache[key]
            if key in self.cache_access_times:
                del self.cache_access_times[key]
        
        # 如果还是太多，移除最久未访问的
        if len(self.retrieval_cache) >= self.cache_size:
            sorted_keys = sorted(
                self.cache_access_times.items(),
                key=lambda x: x[1]
            )
            
            keys_to_remove = sorted_keys[:len(sorted_keys) // 2]
            for key, _ in keys_to_remove:
                if key in self.retrieval_cache:
                    del self.retrieval_cache[key]
                del self.cache_access_times[key]
    
    def _update_stats(self, retrieval_time: float) -> None:
        """更新统计信息"""
        total_queries = self.retrieval_stats['total_queries']
        current_avg = self.retrieval_stats['avg_retrieval_time']
        
        # 计算新的平均检索时间
        new_avg = (current_avg * (total_queries - 1) + retrieval_time) / total_queries
        self.retrieval_stats['avg_retrieval_time'] = new_avg
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        cache_hit_rate = 0.0
        if self.retrieval_stats['total_queries'] > 0:
            cache_hit_rate = self.retrieval_stats['cache_hits'] / self.retrieval_stats['total_queries']
        
        return {
            **self.retrieval_stats,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.retrieval_cache)
        }
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self.retrieval_cache.clear()
        self.cache_access_times.clear()
        logger.info("检索缓存已清空")
    
    def format_memories_for_prompt(self, memories: List[MemoryItem], 
                                  max_length: int = 1000) -> str:
        """格式化记忆用于提示词
        
        Args:
            memories: 记忆列表
            max_length: 最大长度限制
            
        Returns:
            格式化的记忆文本
        """
        if not memories:
            return "暂无相关记忆。"
        
        formatted_parts = []
        current_length = 0
        
        for i, memory in enumerate(memories, 1):
            memory_text = f"{i}. {memory.content} (相关度: {memory.score:.2f})"
            
            if current_length + len(memory_text) > max_length:
                break
            
            formatted_parts.append(memory_text)
            current_length += len(memory_text)
        
        return "\n".join(formatted_parts)