# -*- coding: utf-8 -*-

# 知识图谱存储管理器 (Knowledge Graph Storage Manager)
# 负责知识图谱的持久化存储和向量检索功能
# 作者: RobotAgent开发团队
# 版本: 0.0.1 (Initial Release)
# 更新时间: 2025-08-21

import os
import json
import pickle
from typing import Dict, List, Set, Optional, Tuple, Any, Union
from collections import defaultdict
import numpy as np
import networkx as nx
from datetime import datetime

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from .embedding_model import embedding_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    
    # 向量存储器 (Vector Store)
    
    # 简单的向量存储实现，提供基础的向量管理功能。
    # 主要功能包括：
    # 1. 向量数据存储
    # 2. 相似度搜索
    # 3. 元数据管理
    # 4. 向量索引维护
    
    # 继承自object，实现了基础的向量存储功能。
    #
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict] = []
        self.ids: List[str] = []
    
    def add_vectors(self, vectors: np.ndarray, metadatas: List[Dict], ids: List[str]) -> None:
        """添加向量"""
        for i, vector in enumerate(vectors):
            self.vectors.append(vector)
            self.metadata.append(metadatas[i] if i < len(metadatas) else {})
            self.ids.append(ids[i] if i < len(ids) else str(len(self.vectors)))
    
    def similarity_search(self, query_vector: np.ndarray, k: int = 5, 
                         score_threshold: float = 0.0) -> List[Tuple[Dict, float]]:
        """相似度搜索"""
        if not self.vectors:
            return []
        
        # 计算余弦相似度
        similarities = []
        query_norm = np.linalg.norm(query_vector)
        
        for i, vector in enumerate(self.vectors):
            vector_norm = np.linalg.norm(vector)
            if query_norm > 0 and vector_norm > 0:
                similarity = np.dot(query_vector, vector) / (query_norm * vector_norm)
            else:
                similarity = 0.0
            
            if similarity >= score_threshold:
                similarities.append((i, similarity))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前k个结果
        results = []
        for i, similarity in similarities[:k]:
            result_metadata = self.metadata[i].copy()
            result_metadata['id'] = self.ids[i]
            results.append((result_metadata, similarity))
        
        return results
    
    def save(self, filepath: str) -> None:
        """保存向量存储"""
        data = {
            'vectors': [v.tolist() for v in self.vectors],
            'metadata': self.metadata,
            'ids': self.ids,
            'dimension': self.dimension
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str) -> None:
        """加载向量存储"""
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.vectors = [np.array(v) for v in data.get('vectors', [])]
            self.metadata = data.get('metadata', [])
            self.ids = data.get('ids', [])
            self.dimension = data.get('dimension', 384)


class GraphStorage:
    
    # 知识图谱存储管理器 (Knowledge Graph Storage Manager)
    
    # 负责知识图谱的持久化存储和向量检索功能。
    # 支持实体关系图、向量存储、社区发现等核心功能，主要功能包括：
    # 1. 图结构存储与加载
    # 2. 向量存储管理
    # 3. 实体别名管理
    # 4. 社区发现与存储
    
    # 继承自object，实现了完整的知识图谱存储管理功能。
    #
    
    def __init__(self, storage_path: str):
        """初始化存储管理器
        
        Args:
            storage_path: 存储根目录路径
        """
        self.storage_path = storage_path
        self.graph_file = os.path.join(storage_path, "graph.json")
        self.embeddings_file = os.path.join(storage_path, "embeddings.json")
        self.communities_file = os.path.join(storage_path, "communities.json")
        self.vectors_dir = os.path.join(storage_path, "vectors")
        
        # 核心数据结构
        self.graph = nx.MultiDiGraph()
        self.entity_embeddings: Dict[str, np.ndarray] = {}
        self.entity_aliases: Dict[str, Set[str]] = defaultdict(set)
        self.alias_to_main_id: Dict[str, str] = {}
        self.communities: Dict[int, Dict] = {}
        
        # 向量存储
        self.vector_stores: Dict[str, VectorStore] = {}
        self.global_vector_store: Optional[VectorStore] = None
        
        # 变更追踪
        self.modified_entities: Set[str] = set()
        self.modified_communities: Set[int] = set()
        
        # 初始化存储结构
        self._init_storage()
        self.load()
    
    def _init_storage(self) -> None:
        """初始化存储目录结构"""
        os.makedirs(self.storage_path, exist_ok=True)
        os.makedirs(self.vectors_dir, exist_ok=True)
    
    def add_entity(self, entity_id: str, properties: Dict[str, Any], 
                   aliases: Optional[List[str]] = None) -> None:
        """添加实体
        
        Args:
            entity_id: 实体ID
            properties: 实体属性
            aliases: 实体别名列表
        """
        # 添加到图中
        self.graph.add_node(entity_id, **properties)
        
        # 处理别名
        if aliases:
            self.entity_aliases[entity_id].update(aliases)
            for alias in aliases:
                self.alias_to_main_id[alias] = entity_id
        
        # 生成实体嵌入
        entity_text = self._entity_to_text(entity_id, properties)
        embedding = embedding_model.encode(entity_text)
        self.entity_embeddings[entity_id] = embedding[0] if len(embedding.shape) > 1 else embedding
        
        # 标记为已修改
        self.modified_entities.add(entity_id)
        
        logger.debug(f"添加实体: {entity_id}")
    
    def add_relationship(self, source_id: str, target_id: str, 
                        relation_type: str, properties: Optional[Dict] = None) -> None:
        """添加关系
        
        Args:
            source_id: 源实体ID
            target_id: 目标实体ID
            relation_type: 关系类型
            properties: 关系属性
        """
        if properties is None:
            properties = {}
        
        properties['relation_type'] = relation_type
        properties['created_at'] = datetime.now().isoformat()
        
        self.graph.add_edge(source_id, target_id, **properties)
        
        # 标记相关实体为已修改
        self.modified_entities.add(source_id)
        self.modified_entities.add(target_id)
        
        logger.debug(f"添加关系: {source_id} -> {target_id} ({relation_type})")
    
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """获取实体信息"""
        if entity_id in self.graph.nodes:
            return dict(self.graph.nodes[entity_id])
        return None
    
    def get_relationships(self, entity_id: str, direction: str = 'both') -> List[Dict]:
        """获取实体的关系
        
        Args:
            entity_id: 实体ID
            direction: 关系方向 ('in', 'out', 'both')
        """
        relationships = []
        
        if direction in ['out', 'both']:
            for target in self.graph.successors(entity_id):
                for edge_data in self.graph[entity_id][target].values():
                    relationships.append({
                        'source': entity_id,
                        'target': target,
                        'direction': 'out',
                        **edge_data
                    })
        
        if direction in ['in', 'both']:
            for source in self.graph.predecessors(entity_id):
                for edge_data in self.graph[source][entity_id].values():
                    relationships.append({
                        'source': source,
                        'target': entity_id,
                        'direction': 'in',
                        **edge_data
                    })
        
        return relationships
    
    def search_similar_entities(self, query_text: str, k: int = 5, 
                               threshold: float = 0.5) -> List[Tuple[str, float]]:
        """搜索相似实体
        
        Args:
            query_text: 查询文本
            k: 返回结果数量
            threshold: 相似度阈值
        """
        if not self.entity_embeddings:
            return []
        
        # 生成查询向量
        query_embedding = embedding_model.encode(query_text)
        if len(query_embedding.shape) > 1:
            query_embedding = query_embedding[0]
        
        # 计算相似度
        similarities = []
        for entity_id, entity_embedding in self.entity_embeddings.items():
            similarity = self._cosine_similarity(query_embedding, entity_embedding)
            if similarity >= threshold:
                similarities.append((entity_id, similarity))
        
        # 排序并返回前k个
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def create_entity_vector_store(self, entity_id: str, documents: List[str]) -> None:
        """为实体创建向量存储
        
        Args:
            entity_id: 实体ID
            documents: 相关文档列表
        """
        if not documents:
            return
        
        # 创建向量存储
        vector_store = VectorStore()
        
        # 生成文档向量
        embeddings = embedding_model.encode(documents)
        
        # 准备元数据
        metadatas = []
        ids = []
        for i, doc in enumerate(documents):
            metadatas.append({
                'entity_id': entity_id,
                'document': doc,
                'doc_index': i
            })
            ids.append(f"{entity_id}_doc_{i}")
        
        # 添加到向量存储
        vector_store.add_vectors(embeddings, metadatas, ids)
        
        # 保存向量存储
        self.vector_stores[entity_id] = vector_store
        vector_file = os.path.join(self.vectors_dir, f"{entity_id}.json")
        vector_store.save(vector_file)
        
        logger.debug(f"为实体 {entity_id} 创建向量存储，包含 {len(documents)} 个文档")
    
    def search_entity_documents(self, entity_id: str, query: str, 
                               k: int = 5) -> List[Tuple[str, float]]:
        """搜索实体相关文档
        
        Args:
            entity_id: 实体ID
            query: 查询文本
            k: 返回结果数量
        """
        if entity_id not in self.vector_stores:
            return []
        
        query_embedding = embedding_model.encode(query)
        if len(query_embedding.shape) > 1:
            query_embedding = query_embedding[0]
        
        results = self.vector_stores[entity_id].similarity_search(query_embedding, k)
        
        return [(result[0]['document'], result[1]) for result in results]
    
    def _entity_to_text(self, entity_id: str, properties: Dict[str, Any]) -> str:
        """将实体转换为文本表示"""
        text_parts = [entity_id]
        
        for key, value in properties.items():
            if isinstance(value, (str, int, float)):
                text_parts.append(f"{key}: {value}")
        
        return " ".join(text_parts)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def save(self) -> None:
        """保存所有数据"""
        try:
            # 保存图结构
            graph_data = {
                'graph': nx.node_link_data(self.graph),
                'aliases': {k: list(v) for k, v in self.entity_aliases.items()},
                'alias_to_main_id': self.alias_to_main_id,
                'modified_at': datetime.now().isoformat()
            }
            
            with open(self.graph_file, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)
            
            # 保存实体嵌入
            embeddings_data = {}
            for entity_id, embedding in self.entity_embeddings.items():
                embeddings_data[entity_id] = embedding.tolist()
            
            with open(self.embeddings_file, 'w', encoding='utf-8') as f:
                json.dump(embeddings_data, f, ensure_ascii=False, indent=2)
            
            # 保存社区信息
            with open(self.communities_file, 'w', encoding='utf-8') as f:
                json.dump(self.communities, f, ensure_ascii=False, indent=2)
            
            # 清除修改标记
            self.modified_entities.clear()
            self.modified_communities.clear()
            
            logger.info("知识图谱数据保存成功")
            
        except Exception as e:
            logger.error(f"保存知识图谱数据失败: {e}")
    
    def load(self) -> None:
        """加载所有数据"""
        try:
            # 加载图结构
            if os.path.exists(self.graph_file):
                with open(self.graph_file, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                
                self.graph = nx.node_link_graph(graph_data['graph'], 
                                               directed=True, 
                                               multigraph=True)
                
                # 加载别名信息
                aliases_data = graph_data.get('aliases', {})
                for entity_id, aliases in aliases_data.items():
                    self.entity_aliases[entity_id] = set(aliases)
                
                self.alias_to_main_id = graph_data.get('alias_to_main_id', {})
            
            # 加载实体嵌入
            if os.path.exists(self.embeddings_file):
                with open(self.embeddings_file, 'r', encoding='utf-8') as f:
                    embeddings_data = json.load(f)
                
                for entity_id, embedding in embeddings_data.items():
                    self.entity_embeddings[entity_id] = np.array(embedding)
            
            # 加载社区信息
            if os.path.exists(self.communities_file):
                with open(self.communities_file, 'r', encoding='utf-8') as f:
                    self.communities = json.load(f)
            
            # 加载向量存储
            if os.path.exists(self.vectors_dir):
                for filename in os.listdir(self.vectors_dir):
                    if filename.endswith('.json'):
                        entity_id = filename[:-5]  # 移除.json后缀
                        vector_store = VectorStore()
                        vector_file = os.path.join(self.vectors_dir, filename)
                        vector_store.load(vector_file)
                        self.vector_stores[entity_id] = vector_store
            
            logger.info(f"知识图谱数据加载成功，包含 {len(self.graph.nodes)} 个实体")
            
        except Exception as e:
            logger.error(f"加载知识图谱数据失败: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        return {
            'entities_count': len(self.graph.nodes),
            'relationships_count': len(self.graph.edges),
            'communities_count': len(self.communities),
            'vector_stores_count': len(self.vector_stores),
            'aliases_count': sum(len(aliases) for aliases in self.entity_aliases.values())
        }
    
    def cleanup(self) -> None:
        """清理资源"""
        self.save()
        self.graph.clear()
        self.entity_embeddings.clear()
        self.entity_aliases.clear()
        self.alias_to_main_id.clear()
        self.vector_stores.clear()
        logger.info("图存储资源已清理")