import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

class Config:
    """配置管理类"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            # 默认配置文件路径 - 指向项目根目录的config文件夹
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                "config", 
                "config.yaml"
            )
        
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 环境变量覆盖
            self._override_with_env(config)
            return config
            
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件未找到: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件格式错误: {e}")
    
    def _override_with_env(self, config: Dict[str, Any]):
        """使用环境变量覆盖配置"""
        # Qwen API Key
        if os.getenv("QWEN_API_KEY"):
            config["qwen"]["api_key"] = os.getenv("QWEN_API_KEY")
        
        # Redis配置
        if os.getenv("REDIS_HOST"):
            config["redis"]["host"] = os.getenv("REDIS_HOST")
        if os.getenv("REDIS_PORT"):
            config["redis"]["port"] = int(os.getenv("REDIS_PORT"))
        if os.getenv("REDIS_PASSWORD"):
            config["redis"]["password"] = os.getenv("REDIS_PASSWORD")
        
        # 服务器配置
        if os.getenv("SERVER_HOST"):
            config["server"]["host"] = os.getenv("SERVER_HOST")
        if os.getenv("SERVER_PORT"):
            config["server"]["port"] = int(os.getenv("SERVER_PORT"))
        
        # 调试模式
        if os.getenv("DEBUG"):
            config["system"]["debug"] = os.getenv("DEBUG").lower() == "true"
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点号分隔的嵌套键"""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """获取配置段"""
        return self.get(section, {})
    
    @property
    def system(self) -> Dict[str, Any]:
        return self.get_section("system")
    
    @property
    def server(self) -> Dict[str, Any]:
        return self.get_section("server")
    
    @property
    def qwen(self) -> Dict[str, Any]:
        return self.get_section("qwen")
    
    @property
    def redis(self) -> Dict[str, Any]:
        return self.get_section("redis")
    
    @property
    def message_queue(self) -> Dict[str, Any]:
        return self.get_section("message_queue")
    
    @property
    def ros2(self) -> Dict[str, Any]:
        return self.get_section("ros2")
    
    @property
    def memory_agent(self) -> Dict[str, Any]:
        return self.get_section("memory_agent")
    
    @property
    def logging(self) -> Dict[str, Any]:
        return self.get_section("logging")
    
    @property
    def monitoring(self) -> Dict[str, Any]:
        return self.get_section("monitoring")
    
    @property
    def frontend(self) -> Dict[str, Any]:
        return self.get_section("frontend")

# 全局配置实例
config = Config()