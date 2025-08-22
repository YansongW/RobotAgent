# -*- coding: utf-8 -*-

# 嵌入模型管理器 (Embedding Model Manager)
# 提供统一的文本嵌入服务，支持多种嵌入模型的加载和使用
# 作者: RobotAgent开发团队
# 版本: 0.0.1 (Initial Release)
# 更新时间: 2025-08-21

import os
import logging
from typing import List, Optional, Union, Dict, Any
import numpy as np
import requests
import json
import base64
from io import BytesIO
from PIL import Image

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available, using fallback embedding")

from src.utils.logger import get_logger
from src.utils.config_loader import config_loader

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
    _use_multimodal: bool = False
    _api_config: Optional[Dict[str, Any]] = None
    
    def __new__(cls) -> 'EmbeddingModel':
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化嵌入模型"""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._load_config()
            self._load_model()
    
    def _load_config(self) -> None:
        """加载配置"""
        try:
            api_config = config_loader.load_api_config()
            volcengine_config = api_config.get('volcengine', {})
            self._api_config = volcengine_config.get('embedding', {})
            
            if self._api_config and self._api_config.get('api_key'):
                self._use_multimodal = True
                logger.info("检测到多模态向量模型配置，将使用火山方舟多模态向量模型")
            else:
                logger.info("未检测到多模态向量模型配置，将使用本地模型")
                
        except Exception as e:
            logger.warning(f"加载配置失败，使用默认配置: {e}")
            self._use_multimodal = False
    
    def _load_model(self) -> None:
        """加载嵌入模型"""
        try:
            if self._use_multimodal:
                logger.info("使用火山方舟多模态向量模型")
                # 多模态模型不需要预加载，使用API调用
                self._model = None
            elif SENTENCE_TRANSFORMERS_AVAILABLE:
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
    
    def _call_multimodal_api(self, input_data: List[Dict[str, Any]]) -> np.ndarray:
        """调用多模态向量模型API
        
        Args:
            input_data: 输入数据列表，支持文本和图像
            
        Returns:
            向量数组
        """
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_config['api_key']}"
            }
            
            payload = {
                "model": self._api_config.get('model', 'doubao-embedding-vision-250615'),
                "input": input_data
            }
            
            response = requests.post(
                self._api_config.get('base_url', 'https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal'),
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.debug(f"API响应: {result}")
                
                # 检查响应格式
                if 'data' in result and isinstance(result['data'], list):
                    embeddings = [item['embedding'] for item in result['data']]
                    return np.array(embeddings)
                elif 'data' in result and 'embedding' in result['data']:
                    # 火山方舟API格式：单个embedding对象
                    return np.array([result['data']['embedding']])
                elif 'embedding' in result:
                    # 单个嵌入向量的情况
                    return np.array([result['embedding']])
                else:
                    logger.error(f"未知的API响应格式: {result}")
                    raise Exception(f"未知的API响应格式")
            else:
                logger.error(f"API调用失败: {response.status_code} - {response.text}")
                raise Exception(f"API调用失败: {response.status_code}")
                
        except Exception as e:
            logger.error(f"多模态API调用失败: {e}")
            raise
    
    def _prepare_multimodal_input(self, inputs: Union[str, List[Union[str, Dict[str, Any]]]]) -> List[Dict[str, Any]]:
        """准备多模态输入数据
        
        Args:
            inputs: 输入数据，可以是文本字符串、文本列表或包含图像的字典列表
            
        Returns:
            格式化的输入数据列表
        """
        if isinstance(inputs, str):
            return [{"type": "text", "text": inputs}]
        
        formatted_inputs = []
        for item in inputs:
            if isinstance(item, str):
                formatted_inputs.append({"type": "text", "text": item})
            elif isinstance(item, dict):
                if item.get('type') == 'image_url':
                    formatted_inputs.append(item)
                elif item.get('type') == 'image_path':
                    # 将本地图像路径转换为base64
                    image_path = item.get('path')
                    if os.path.exists(image_path):
                        with open(image_path, 'rb') as f:
                            image_data = base64.b64encode(f.read()).decode('utf-8')
                        formatted_inputs.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        })
                    else:
                        logger.warning(f"图像文件不存在: {image_path}")
                elif 'text' in item:
                    formatted_inputs.append({"type": "text", "text": item['text']})
        
        return formatted_inputs
    
    def encode(self, inputs: Union[str, List[Union[str, Dict[str, Any]]]], 
               normalize_embeddings: bool = True,
               batch_size: int = 32) -> np.ndarray:
        """编码文本或多模态数据为向量
        
        Args:
            inputs: 待编码的输入数据，可以是:
                   - 文本字符串
                   - 文本字符串列表
                   - 包含文本和图像的字典列表
            normalize_embeddings: 是否归一化向量
            batch_size: 批处理大小
            
        Returns:
            向量数组
        """
        try:
            # 如果使用多模态API
            if self._use_multimodal and self._api_config:
                formatted_inputs = self._prepare_multimodal_input(inputs)
                return self._call_multimodal_api(formatted_inputs)
            
            # 使用本地模型处理纯文本
            if self._model is None:
                self._load_model()
            
            # 将输入转换为文本列表
            if isinstance(inputs, str):
                texts = [inputs]
            elif isinstance(inputs, list):
                texts = []
                for item in inputs:
                    if isinstance(item, str):
                        texts.append(item)
                    elif isinstance(item, dict) and 'text' in item:
                        texts.append(item['text'])
                    else:
                        logger.warning(f"跳过不支持的输入类型: {type(item)}")
            else:
                texts = [str(inputs)]
            
            if not texts:
                logger.warning("没有有效的文本输入")
                return np.zeros((1, 384))
            
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
            logger.error(f"编码失败: {e}")
            # 返回零向量作为后备
            if isinstance(inputs, str):
                return np.zeros((1, 384))
            elif isinstance(inputs, list):
                return np.zeros((len(inputs), 384))
            else:
                return np.zeros((1, 384))
    
    def get_embedding_dim(self) -> int:
        """获取嵌入向量维度"""
        if self._use_multimodal and self._api_config:
            return self._api_config.get('dimension', 1024)  # 多模态模型默认维度
        elif SENTENCE_TRANSFORMERS_AVAILABLE and isinstance(self._model, SentenceTransformer):
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