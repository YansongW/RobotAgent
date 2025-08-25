# -*- coding: utf-8 -*-

# 配置加载器 (Configuration Loader)
# 统一的配置文件加载和管理功能
# 作者: RobotAgent开发团队
# 版本: 0.0.1 (Initial Release)
# 更新时间: 2025-08-21

# 导入标准库
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

# 导入环境变量加载器
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    logging.warning("python-dotenv未安装，无法从.env文件加载环境变量")

# 导入项目基础组件
try:
    from cryptography.fernet import Fernet
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False
    logging.warning("加密库未安装，敏感配置将以明文存储")


class ConfigLoader:
    """
    配置加载器 (Configuration Loader)
    
    安全地加载和管理各类配置文件，支持多种配置格式和加密存储。
    
    主要功能:
    - 多格式配置文件加载（YAML、JSON）
    - API密钥安全管理和加密存储
    - 配置文件验证和错误处理
    - 环境变量配置覆盖
    - 配置缓存和热重载
    
    继承关系: 无直接继承，作为工具类使用
    
    Attributes:
        config_dir (Path): 配置文件目录路径
        config_cache (Dict): 配置缓存字典
        encryption_key (bytes): 配置加密密钥
        
    Example:
        >>> loader = ConfigLoader()
        >>> api_config = loader.load_api_config()
        >>> volcengine_config = loader.get_volcengine_config()
        
    Note:
        配置文件应放置在项目根目录的config文件夹中。
        敏感配置建议使用环境变量或加密存储。
        
    See Also:
        ConfigManager: 高级配置管理器
        SystemConfig: 系统配置数据结构
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        初始化配置加载器
        
        设置配置文件目录路径，初始化缓存和加密组件。
        如果未指定配置目录，将自动定位到项目根目录下的config文件夹。
        
        Args:
            config_dir: 配置文件目录路径，默认为项目根目录下的config文件夹
            
        Raises:
            FileNotFoundError: 当配置目录不存在时
            PermissionError: 当没有配置目录访问权限时
        """
        # 配置目录路径设置
        if config_dir is None:
            # 获取项目根目录
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            self.config_dir = project_root / "config"
        else:
            self.config_dir = Path(config_dir)
        
        # 验证配置目录存在性
        if not self.config_dir.exists():
            raise FileNotFoundError(f"配置目录不存在: {self.config_dir}")
        
        # 初始化配置缓存
        self.config_cache: Dict[str, Any] = {}
        
        # 初始化日志记录器
        self.logger = logging.getLogger(__name__)
        
        # 加载环境变量文件
        self._load_env_file()
        
        # 初始化加密组件
        self._init_encryption()

    def _load_env_file(self) -> None:
        """
        加载.env文件中的环境变量
        
        从项目根目录查找.env文件并加载其中的环境变量。
        如果文件不存在或加载失败，会记录警告但不会中断程序运行。
        """
        if not DOTENV_AVAILABLE:
            return
            
        try:
            # 获取项目根目录
            project_root = self.config_dir.parent
            env_file = project_root / ".env"
            
            if env_file.exists():
                load_dotenv(env_file)
                self.logger.info(f"已加载环境变量文件: {env_file}")
            else:
                self.logger.info("未找到.env文件，将使用系统环境变量")
                
        except Exception as e:
            self.logger.warning(f"加载.env文件失败: {e}")

    def _init_encryption(self) -> None:
        """
        初始化加密组件
        
        设置配置加密密钥，用于敏感配置的加密存储。
        如果加密库不可用，将记录警告信息。
        """
        if ENCRYPTION_AVAILABLE:
            # 尝试从环境变量获取加密密钥
            key_env = os.getenv('CONFIG_ENCRYPTION_KEY')
            if key_env:
                self.encryption_key = key_env.encode()
            else:
                # 生成新的加密密钥（仅用于开发环境）
                self.encryption_key = Fernet.generate_key()
                self.logger.warning("使用临时加密密钥，生产环境请设置CONFIG_ENCRYPTION_KEY环境变量")
        else:
            self.encryption_key = None
    
    def load_api_config(self) -> Dict[str, Any]:
        """
        加载API配置
        
        从api_config.yaml文件加载API相关配置，支持缓存和环境变量覆盖。
        
        Returns:
            API配置字典，包含各种API服务的配置信息
            
        Raises:
            FileNotFoundError: 当API配置文件不存在时
            RuntimeError: 当配置文件格式错误或加载失败时
        """
        # 检查缓存
        cache_key = "api_config"
        if cache_key in self.config_cache:
            return self.config_cache[cache_key]
        
        config_file = self.config_dir / "api_config.yaml"
        
        if not config_file.exists():
            # 尝试从模板文件创建
            template_file = self.config_dir.parent / "api_config.yaml.template"
            if template_file.exists():
                self.logger.warning(f"API配置文件不存在，请根据模板创建: {template_file}")
            raise FileNotFoundError(f"API配置文件不存在: {config_file}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 环境变量覆盖
            config = self._apply_env_overrides(config, 'API')
            
            # 缓存配置
            self.config_cache[cache_key] = config
            
            self.logger.info("API配置加载成功")
            return config
            
        except yaml.YAMLError as e:
            raise RuntimeError(f"API配置文件格式错误: {e}")
        except Exception as e:
            raise RuntimeError(f"加载API配置失败: {e}")
    
    def _apply_env_overrides(self, config: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        """
        应用环境变量覆盖
        
        使用环境变量覆盖配置文件中的值，支持嵌套配置和占位符替换。
        环境变量格式: {PREFIX}_{SECTION}_{KEY}
        占位符格式: ${ENV_VAR_NAME}
        
        Args:
            config: 原始配置字典
            prefix: 环境变量前缀
            
        Returns:
            应用环境变量覆盖后的配置字典
        """
        # 首先替换配置中的环境变量占位符
        config = self._replace_env_placeholders(config)
        
        # 然后应用环境变量覆盖
        for key, value in os.environ.items():
            if key.startswith(f"{prefix}_"):
                # 解析环境变量键
                config_path = key[len(prefix)+1:].lower().split('_')
                
                # 应用到配置中
                current = config
                for path_part in config_path[:-1]:
                    if path_part not in current:
                        current[path_part] = {}
                    current = current[path_part]
                
                current[config_path[-1]] = value
                self.logger.debug(f"环境变量覆盖: {key} -> {config_path}")
        
        return config
    
    def _replace_env_placeholders(self, config: Any) -> Any:
        """
        递归替换配置中的环境变量占位符
        
        Args:
            config: 配置值（可能是字典、列表或字符串）
            
        Returns:
            替换占位符后的配置值
        """
        if isinstance(config, dict):
            return {key: self._replace_env_placeholders(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._replace_env_placeholders(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            # 提取环境变量名
            env_var = config[2:-1]
            env_value = os.getenv(env_var)
            if env_value is not None:
                return env_value
            else:
                self.logger.warning(f"环境变量 {env_var} 未设置，保持原值: {config}")
                return config
        else:
            return config
    
    def get_volcengine_config(self) -> Dict[str, Any]:
        """
        获取火山方舟API配置
        
        从API配置中提取火山方舟相关配置，并进行必要的验证。
        
        Returns:
            火山方舟配置字典，包含API密钥、基础URL、默认模型等信息
            
        Raises:
            ValueError: 当配置缺失或无效时
        """
        api_config = self.load_api_config()
        volcengine_config = api_config.get('volcengine', {})
        
        if not volcengine_config:
            raise ValueError("未找到火山方舟API配置")
        
        # 检查必需的配置项
        required_keys = ['api_key', 'base_url', 'default_model']
        for key in required_keys:
            if key not in volcengine_config:
                raise ValueError(f"火山方舟配置缺少必需项: {key}")
        
        # 验证配置值
        api_key = volcengine_config['api_key']
        if not api_key or api_key.startswith('${'):
            raise ValueError("火山方舟API密钥未正确配置，请设置VOLCENGINE_API_KEY环境变量")
        
        if not volcengine_config['base_url'].startswith(('http://', 'https://')):
            raise ValueError("火山方舟基础URL格式无效")
        
        self.logger.info("火山方舟配置验证成功")
        return volcengine_config
    
    def get_api_key(self, service: str) -> str:
        """
        获取指定服务的API密钥
        
        从API配置中安全地获取指定服务的API密钥，支持加密存储。
        
        Args:
            service: 服务名称 (如 'volcengine', 'openai', 'anthropic')
            
        Returns:
            解密后的API密钥字符串
            
        Raises:
            ValueError: 当服务配置不存在或API密钥缺失时
        """
        api_config = self.load_api_config()
        service_config = api_config.get(service, {})
        
        if not service_config:
            raise ValueError(f"未找到服务配置: {service}")
        
        api_key = service_config.get('api_key')
        if not api_key:
            raise ValueError(f"未找到{service}的API密钥")
        
        # 如果API密钥是加密的，进行解密
        if isinstance(api_key, str) and api_key.startswith('encrypted:'):
            api_key = self._decrypt_value(api_key[10:])  # 移除'encrypted:'前缀
        
        self.logger.debug(f"成功获取{service}的API密钥")
        return api_key
    
    def _decrypt_value(self, encrypted_value: str) -> str:
        """
        解密配置值
        
        使用配置的加密密钥解密敏感配置值。
        
        Args:
            encrypted_value: 加密后的配置值
            
        Returns:
            解密后的原始值
            
        Raises:
            RuntimeError: 当解密失败时
        """
        if not ENCRYPTION_AVAILABLE or not self.encryption_key:
            raise RuntimeError("加密功能不可用，无法解密配置值")
        
        try:
            fernet = Fernet(self.encryption_key)
            decrypted_bytes = fernet.decrypt(encrypted_value.encode())
            return decrypted_bytes.decode()
        except Exception as e:
            raise RuntimeError(f"配置值解密失败: {e}")
    
    def load_system_config(self) -> Dict[str, Any]:
        """
        加载系统配置
        
        从system_config.yaml文件加载系统级配置，支持缓存和验证。
        
        Returns:
            系统配置字典，包含日志级别、并发设置等系统参数
            
        Raises:
            FileNotFoundError: 当系统配置文件不存在时
            RuntimeError: 当配置文件格式错误或加载失败时
        """
        # 检查缓存
        cache_key = "system_config"
        if cache_key in self.config_cache:
            return self.config_cache[cache_key]
        
        config_file = self.config_dir / "system_config.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"系统配置文件不存在: {config_file}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 环境变量覆盖
            config = self._apply_env_overrides(config, 'SYSTEM')
            
            # 配置验证
            self._validate_system_config(config)
            
            # 缓存配置
            self.config_cache[cache_key] = config
            
            self.logger.info("系统配置加载成功")
            return config
            
        except yaml.YAMLError as e:
            raise RuntimeError(f"系统配置文件格式错误: {e}")
        except Exception as e:
            raise RuntimeError(f"加载系统配置失败: {e}")
    
    def _validate_system_config(self, config: Dict[str, Any]) -> None:
        """
        验证系统配置
        
        检查系统配置的完整性和有效性。
        
        Args:
            config: 系统配置字典
            
        Raises:
            ValueError: 当配置无效时
        """
        # 验证必需的配置节
        required_sections = ['system', 'communication', 'output']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"系统配置缺少必需节: {section}")
        
        # 验证日志级别
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        log_level = config.get('system', {}).get('log_level')
        if log_level and log_level not in valid_log_levels:
            raise ValueError(f"无效的日志级别: {log_level}")
        
        # 验证并发任务数
        max_tasks = config.get('system', {}).get('max_concurrent_tasks')
        if max_tasks and (not isinstance(max_tasks, int) or max_tasks <= 0):
            raise ValueError("最大并发任务数必须是正整数")
    
    def clear_cache(self) -> None:
        """
        清空配置缓存
        
        清空所有缓存的配置，强制重新加载配置文件。
        """
        self.config_cache.clear()
        self.logger.info("配置缓存已清空")


# 全局配置加载器实例
# 提供便捷的全局访问接口
_config_loader_instance = None

def get_config_loader() -> ConfigLoader:
    """获取全局配置加载器实例（延迟初始化）"""
    global _config_loader_instance
    if _config_loader_instance is None:
        _config_loader_instance = ConfigLoader()
    return _config_loader_instance

# 为了向后兼容，保留config_loader属性
class _ConfigLoaderProxy:
    def __getattr__(self, name):
        return getattr(get_config_loader(), name)

config_loader = _ConfigLoaderProxy()