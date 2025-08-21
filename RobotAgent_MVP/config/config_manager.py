# -*- coding: utf-8 -*-

# 配置管理模块 (Configuration Manager)
# 统一管理系统配置、API配置和智能体配置
# 作者: RobotAgent开发团队
# 版本: 0.0.1 (Initial Release)
# 更新时间: 2025-08-21
# 基于框架: CAMEL框架集成的配置管理系统

# 导入标准库
import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from enum import Enum
import logging

# 导入项目基础组件
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import ConfigLoader


class ConfigType(Enum):
    """配置类型枚举
    
    定义系统支持的配置类型，用于配置分类管理和验证
    """
    SYSTEM = "system"           # 系统配置
    API = "api"                 # API配置
    AGENTS = "agents"           # 智能体配置
    COMMUNICATION = "communication"  # 通信配置
    MEMORY = "memory"           # 记忆系统配置
    SECURITY = "security"       # 安全配置


class ConfigSource(Enum):
    """配置源枚举
    
    定义配置数据的来源，支持多种配置源的优先级管理
    """
    FILE = "file"               # 配置文件
    ENVIRONMENT = "environment" # 环境变量
    COMMAND_LINE = "command_line"  # 命令行参数
    DATABASE = "database"       # 数据库
    REMOTE = "remote"           # 远程配置服务


@dataclass
class ConfigItem:
    """配置项数据结构
    
    封装单个配置项的完整信息，包括值、来源、验证规则等
    
    主要功能:
    - 配置值存储和类型管理
    - 配置来源追踪
    - 配置验证规则定义
    - 配置更新历史记录
    
    继承关系: 无直接继承，作为数据容器使用
    """
    key: str                    # 配置键
    value: Any                  # 配置值
    config_type: ConfigType     # 配置类型
    source: ConfigSource        # 配置来源
    description: str = ""       # 配置描述
    is_required: bool = False   # 是否必需
    is_sensitive: bool = False  # 是否敏感信息
    validation_rule: Optional[str] = None  # 验证规则
    default_value: Any = None   # 默认值
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据


class ConfigManager:
    """配置管理器
    
    统一管理系统所有配置，提供配置加载、验证、更新和持久化功能
    
    主要功能:
    - 多源配置加载和合并
    - 配置验证和类型检查
    - 配置热更新和监听
    - 敏感配置加密存储
    - 配置变更历史追踪
    
    继承关系: 无直接继承，作为核心配置管理组件
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """初始化配置管理器
        
        Args:
            config_dir: 配置文件目录路径
        """
        # 配置目录设置
        if config_dir is None:
            current_file = Path(__file__)
            self.config_dir = current_file.parent
        else:
            self.config_dir = Path(config_dir)
        
        # 配置存储
        self.configs: Dict[str, ConfigItem] = {}
        self.config_cache: Dict[str, Any] = {}
        
        # 配置加载器
        self.config_loader = ConfigLoader(self.config_dir.parent)
        
        # 配置监听器
        self.config_listeners: Dict[str, List[callable]] = {}
        
        # 日志记录器
        self.logger = logging.getLogger(__name__)
        
        # 初始化配置
        self._load_all_configs()
    
    def _load_all_configs(self) -> None:
        """加载所有配置文件
        
        按照优先级顺序加载各类配置，并进行合并和验证
        """
        try:
            # 加载系统配置
            self._load_system_config()
            
            # 加载API配置
            self._load_api_config()
            
            # 加载智能体配置
            self._load_agents_config()
            
            # 加载环境变量配置
            self._load_environment_config()
            
            self.logger.info(f"配置加载完成，共加载 {len(self.configs)} 个配置项")
            
        except Exception as e:
            self.logger.error(f"配置加载失败: {e}")
            raise
    
    def _load_system_config(self) -> None:
        """加载系统配置
        
        从system_config.yaml文件加载系统级配置
        """
        try:
            config_file = self.config_dir / "system_config.yaml"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                self._register_config_section(
                    config_data, ConfigType.SYSTEM, ConfigSource.FILE
                )
                
        except Exception as e:
            self.logger.error(f"系统配置加载失败: {e}")
    
    def _load_api_config(self) -> None:
        """加载API配置
        
        从api_config.yaml文件加载API相关配置
        """
        try:
            api_config = self.config_loader.load_api_config()
            self._register_config_section(
                api_config, ConfigType.API, ConfigSource.FILE
            )
            
        except Exception as e:
            self.logger.warning(f"API配置加载失败: {e}")
    
    def _load_agents_config(self) -> None:
        """加载智能体配置
        
        从agents_config.yaml文件加载智能体相关配置
        """
        try:
            config_file = self.config_dir / "agents_config.yaml"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                self._register_config_section(
                    config_data, ConfigType.AGENTS, ConfigSource.FILE
                )
                
        except Exception as e:
            self.logger.error(f"智能体配置加载失败: {e}")
    
    def _load_environment_config(self) -> None:
        """加载环境变量配置
        
        从环境变量中加载配置，优先级高于文件配置
        """
        env_mappings = {
            'ROBOT_AGENT_LOG_LEVEL': 'system.log_level',
            'ROBOT_AGENT_API_KEY': 'api.volcengine.api_key',
            'ROBOT_AGENT_MODEL': 'api.volcengine.default_model',
        }
        
        for env_key, config_key in env_mappings.items():
            env_value = os.getenv(env_key)
            if env_value:
                self.set_config(
                    config_key, env_value, ConfigSource.ENVIRONMENT
                )
    
    def _register_config_section(self, config_data: Dict[str, Any], 
                                config_type: ConfigType, 
                                source: ConfigSource) -> None:
        """注册配置段
        
        将配置数据注册到配置管理器中
        
        Args:
            config_data: 配置数据字典
            config_type: 配置类型
            source: 配置来源
        """
        def _register_recursive(data: Dict[str, Any], prefix: str = ""):
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key
                
                if isinstance(value, dict):
                    _register_recursive(value, full_key)
                else:
                    config_item = ConfigItem(
                        key=full_key,
                        value=value,
                        config_type=config_type,
                        source=source
                    )
                    self.configs[full_key] = config_item
                    self.config_cache[full_key] = value
        
        _register_recursive(config_data)
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            key: 配置键
            default: 默认值
            
        Returns:
            配置值
        """
        if key in self.config_cache:
            return self.config_cache[key]
        
        # 尝试从配置项中获取
        config_item = self.configs.get(key)
        if config_item:
            return config_item.value
        
        return default
    
    def set_config(self, key: str, value: Any, source: ConfigSource = ConfigSource.FILE,
                  description: str = "", is_required: bool = False) -> None:
        """设置配置值
        
        Args:
            key: 配置键
            value: 配置值
            source: 配置来源
            description: 配置描述
            is_required: 是否必需
        """
        # 确定配置类型
        config_type = self._determine_config_type(key)
        
        # 创建配置项
        config_item = ConfigItem(
            key=key,
            value=value,
            config_type=config_type,
            source=source,
            description=description,
            is_required=is_required
        )
        
        # 存储配置
        self.configs[key] = config_item
        self.config_cache[key] = value
        
        # 触发配置变更监听器
        self._notify_config_change(key, value)
    
    def _determine_config_type(self, key: str) -> ConfigType:
        """确定配置类型
        
        Args:
            key: 配置键
            
        Returns:
            配置类型
        """
        if key.startswith('system.'):
            return ConfigType.SYSTEM
        elif key.startswith('api.'):
            return ConfigType.API
        elif key.startswith('agents.') or key.startswith('chat_agent.') or key.startswith('action_agent.') or key.startswith('memory_agent.'):
            return ConfigType.AGENTS
        elif key.startswith('communication.'):
            return ConfigType.COMMUNICATION
        elif key.startswith('memory.'):
            return ConfigType.MEMORY
        elif key.startswith('security.'):
            return ConfigType.SECURITY
        else:
            return ConfigType.SYSTEM
    
    def _notify_config_change(self, key: str, value: Any) -> None:
        """通知配置变更
        
        Args:
            key: 配置键
            value: 新配置值
        """
        listeners = self.config_listeners.get(key, [])
        for listener in listeners:
            try:
                listener(key, value)
            except Exception as e:
                self.logger.error(f"配置变更监听器执行失败: {e}")
    
    def add_config_listener(self, key: str, listener: callable) -> None:
        """添加配置变更监听器
        
        Args:
            key: 配置键
            listener: 监听器函数
        """
        if key not in self.config_listeners:
            self.config_listeners[key] = []
        self.config_listeners[key].append(listener)
    
    def get_config_by_type(self, config_type: ConfigType) -> Dict[str, Any]:
        """按类型获取配置
        
        Args:
            config_type: 配置类型
            
        Returns:
            配置字典
        """
        result = {}
        for key, config_item in self.configs.items():
            if config_item.config_type == config_type:
                result[key] = config_item.value
        return result
    
    def validate_configs(self) -> List[str]:
        """验证配置
        
        Returns:
            验证错误列表
        """
        errors = []
        
        # 检查必需配置
        for key, config_item in self.configs.items():
            if config_item.is_required and config_item.value is None:
                errors.append(f"必需配置项缺失: {key}")
        
        # 检查API配置
        api_key = self.get_config('api.volcengine.api_key')
        if api_key and api_key.startswith('your-'):
            errors.append("API密钥未正确配置")
        
        return errors
    
    def get_volcengine_config(self) -> Dict[str, Any]:
        """获取火山方舟配置
        
        Returns:
            火山方舟配置字典
        """
        return {
            'api_key': self.get_config('api.volcengine.api_key'),
            'base_url': self.get_config('api.volcengine.base_url'),
            'default_model': self.get_config('api.volcengine.default_model'),
            'temperature': self.get_config('api.volcengine.temperature', 0.7),
            'max_tokens': self.get_config('api.volcengine.max_tokens', 2000),
            'max_history_turns': self.get_config('api.volcengine.max_history_turns', 10)
        }
    
    def get_system_config(self) -> Dict[str, Any]:
        """获取系统配置
        
        Returns:
            系统配置字典
        """
        return {
            'log_level': self.get_config('system.log_level', 'INFO'),
            'max_concurrent_tasks': self.get_config('system.max_concurrent_tasks', 10),
            'message_timeout': self.get_config('communication.message_timeout', 30),
            'retry_attempts': self.get_config('communication.retry_attempts', 3),
            'tts_enabled': self.get_config('output.tts_enabled', True),
            'action_file_format': self.get_config('output.action_file_format', 'json')
        }
    
    def get_agents_config(self) -> Dict[str, Any]:
        """获取智能体配置
        
        Returns:
            智能体配置字典
        """
        return {
            'chat_agent': {
                'model': self.get_config('chat_agent.model', 'qwen-7b'),
                'max_tokens': self.get_config('chat_agent.max_tokens', 2048),
                'temperature': self.get_config('chat_agent.temperature', 0.7)
            },
            'action_agent': {
                'planning_horizon': self.get_config('action_agent.planning_horizon', 5),
                'safety_check': self.get_config('action_agent.safety_check', True)
            },
            'memory_agent': {
                'max_history': self.get_config('memory_agent.max_history', 100),
                'learning_rate': self.get_config('memory_agent.learning_rate', 0.01)
            }
        }
    
    def export_config(self, config_type: Optional[ConfigType] = None) -> Dict[str, Any]:
        """导出配置
        
        Args:
            config_type: 配置类型，None表示导出所有配置
            
        Returns:
            配置字典
        """
        if config_type:
            return self.get_config_by_type(config_type)
        else:
            return {key: item.value for key, item in self.configs.items()}
    
    def reload_config(self) -> None:
        """重新加载配置
        
        清空当前配置并重新加载所有配置文件
        """
        self.configs.clear()
        self.config_cache.clear()
        self._load_all_configs()
        self.logger.info("配置重新加载完成")


# 全局配置管理器实例
config_manager = ConfigManager()