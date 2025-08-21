# -*- coding: utf-8 -*-

# 嵌入模型管理器 (Embedding Model Manager)
# 提供统一的文本嵌入服务，支持多种嵌入模型的加载和使用
# 作者: RobotAgent开发团队
# 版本: 0.0.1 (Initial Release)
# 更新时间: 2025-08-21

import os
import logging
from typing import List, Optional, Union
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available, using fallback embedding")

from ..utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    
    # 嵌入模型管理器 (Embedding Model Manager)
    
    # 提供统一的文本嵌入接口，支持多种后端模型。
    # 采用单例模式确保资源的高效利用，主要功能包括：
    # 1. 文本向量化处理
    # 2. 模型单例管理
    # 3. 设备自适应配置
    # 4. 多种嵌入模型支持
    
    # 继承自object，实现了单例模式的嵌入模型管理器。
    #
    
    _instance: Optional['EmbeddingModel'] = None
    _model: Optional[Union[SentenceTransformer, object]] = None
    _model_name: str = "all-MiniLM-L6-v2"
    _device: str = "cpu"
    
    def __new__(cls) -> 'EmbeddingModel':
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化嵌入模型"""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._load_model()
    
    def _load_model(self) -> None:
        """加载嵌入模型"""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.info(f"正在加载嵌入模型: {self._model_name}")
                self._model = SentenceTransformer(self._model_name, device=self._device)
                logger.info("嵌入模型加载成功")
            else:
                logger.warning("使用简单的词向量作为后备方案")
                self._model = self._create_fallback_model()
        except Exception as e:
            logger.error(f"加载嵌入模型失败: {e}")
            self._model = self._create_fallback_model()
    
    def _create_fallback_model(self) -> object:
        """创建后备嵌入模型"""
        class FallbackEmbedding:
            def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
                if isinstance(texts, str):
                    texts = [texts]
                # 简单的基于字符的向量化
                embeddings = []
                for text in texts:
                    # 使用字符频率作为简单的向量表示
                    vector = np.zeros(384)  # 固定维度
                    for i, char in enumerate(text[:384]):
                        vector[i] = ord(char) / 1000.0
                    embeddings.append(vector)
                return np.array(embeddings)
        
        return FallbackEmbedding()
    
    def encode(self, texts: Union[str, List[str]], 
               normalize_embeddings: bool = True,
               batch_size: int = 32) -> np.ndarray:
        """编码文本为向量
        
        Args:
            texts: 待编码的文本或文本列表
            normalize_embeddings: 是否归一化向量
            batch_size: 批处理大小
            
        Returns:
            文本向量数组
        """
        try:
            if self._model is None:
                self._load_model()
            
            if isinstance(texts, str):
                texts = [texts]
            
            # 分批处理大量文本
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                if SENTENCE_TRANSFORMERS_AVAILABLE and isinstance(self._model, SentenceTransformer):
                    batch_embeddings = self._model.encode(
                        batch_texts,
                        normalize_embeddings=normalize_embeddings,
                        show_progress_bar=False
                    )
                else:
                    batch_embeddings = self._model.encode(batch_texts)
                    if normalize_embeddings:
                        norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                        batch_embeddings = batch_embeddings / (norms + 1e-8)
                
                all_embeddings.append(batch_embeddings)
            
            return np.vstack(all_embeddings)
            
        except Exception as e:
            logger.error(f"文本编码失败: {e}")
            # 返回零向量作为后备
            if isinstance(texts, str):
                texts = [texts]
            return np.zeros((len(texts), 384))
    
    def get_embedding_dim(self) -> int:
        """获取嵌入向量维度"""
        if SENTENCE_TRANSFORMERS_AVAILABLE and isinstance(self._model, SentenceTransformer):
            return self._model.get_sentence_embedding_dimension()
        return 384  # 后备模型的固定维度
    
    def set_device(self, device: str) -> None:
        """设置计算设备
        
        Args:
            device: 设备名称 ('cpu', 'cuda', 'mps')
        """
        self._device = device
        if self._model is not None and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._model.to(device)
                logger.info(f"嵌入模型已切换到设备: {device}")
            except Exception as e:
                logger.warning(f"切换设备失败: {e}")
    
    def cleanup(self) -> None:
        """清理模型资源"""
        if self._model is not None:
            del self._model
            self._model = None
        logger.info("嵌入模型资源已清理")
    
    @classmethod
    def get_instance(cls) -> 'EmbeddingModel':
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# 全局实例
embedding_model = EmbeddingModel.get_instance()