#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 配置加载工具
# 用于安全地加载API配置和其他系统配置

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigLoader:
    # 配置加载器
    
    def __init__(self, config_dir: Optional[Path] = None):
        # 初始化配置加载器
        # 
        # Args:
        #     config_dir: 配置文件目录，默认为项目根目录下的config文件夹
        if config_dir is None:
            # 获取项目根目录
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            self.config_dir = project_root / "config"
        else:
            self.config_dir = config_dir
    
    def load_api_config(self) -> Dict[str, Any]:
        # 加载API配置
        # 
        # Returns:
        #     API配置字典
        config_file = self.config_dir / "api_config.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"API配置文件不存在: {config_file}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise RuntimeError(f"加载API配置失败: {e}")
    
    def get_volcengine_config(self) -> Dict[str, Any]:
        # 获取火山方舟API配置
        # 
        # Returns:
        #     火山方舟配置字典
        api_config = self.load_api_config()
        volcengine_config = api_config.get('volcengine', {})
        
        if not volcengine_config:
            raise ValueError("未找到火山方舟API配置")
        
        # 检查必需的配置项
        required_keys = ['api_key', 'base_url', 'default_model']
        for key in required_keys:
            if key not in volcengine_config:
                raise ValueError(f"火山方舟配置缺少必需项: {key}")
        
        return volcengine_config
    
    def get_api_key(self, service: str) -> str:
        # 获取指定服务的API密钥
        # 
        # Args:
        #     service: 服务名称 (如 'volcengine', 'openai')
        #     
        # Returns:
        #     API密钥
        api_config = self.load_api_config()
        service_config = api_config.get(service, {})
        
        if not service_config:
            raise ValueError(f"未找到服务配置: {service}")
        
        api_key = service_config.get('api_key')
        if not api_key:
            raise ValueError(f"未找到{service}的API密钥")
        
        return api_key
    
    def load_system_config(self) -> Dict[str, Any]:
        # 加载系统配置
        # 
        # Returns:
        #     系统配置字典
        config_file = self.config_dir / "system_config.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"系统配置文件不存在: {config_file}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise RuntimeError(f"加载系统配置失败: {e}")

# 全局配置加载器实例
config_loader = ConfigLoader()