# -*- coding: utf-8 -*-

# 日志管理器 (Logger Manager)
# 统一的日志记录和管理功能
# 作者: RobotAgent开发团队
# 版本: 0.0.1 (Initial Release)
# 更新时间: 2024-01-20
# 基于: BaseRobotAgent v0.0.1

import logging
import logging.handlers
import sys
import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import threading
from functools import wraps


class SensitiveDataFilter(logging.Filter):
    """
    敏感数据过滤器
    
    过滤日志中的敏感信息，如API密钥、密码等。
    """
    
    def __init__(self):
        super().__init__()
        # 敏感信息关键词
        self.sensitive_keywords = [
            'password', 'passwd', 'pwd', 'secret', 'token', 'key',
            'api_key', 'access_token', 'refresh_token', 'auth',
            'credential', 'private', 'confidential'
        ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        过滤敏感信息
        
        Args:
            record: 日志记录
            
        Returns:
            是否允许记录该日志
        """
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            # 检查是否包含敏感信息
            msg_lower = record.msg.lower()
            for keyword in self.sensitive_keywords:
                if keyword in msg_lower:
                    # 替换敏感信息
                    record.msg = self._mask_sensitive_data(record.msg)
                    break
        
        return True
    
    def _mask_sensitive_data(self, message: str) -> str:
        """
        遮蔽敏感数据
        
        Args:
            message: 原始消息
            
        Returns:
            遮蔽后的消息
        """
        # 简单的遮蔽策略，可以根据需要扩展
        import re
        
        # 遮蔽类似 "key=value" 的模式
        pattern = r'((?:api_)?(?:key|token|password|secret)\s*[=:]\s*)([^\s,}\]]+)'
        return re.sub(pattern, r'\1***', message, flags=re.IGNORECASE)


class PerformanceLogger:
    """
    性能日志记录器
    
    记录函数执行时间和性能统计信息。
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.stats: Dict[str, List[float]] = {}
        self.lock = threading.Lock()
    
    def log_execution_time(self, func_name: str, execution_time: float) -> None:
        """
        记录函数执行时间
        
        Args:
            func_name: 函数名称
            execution_time: 执行时间（秒）
        """
        with self.lock:
            if func_name not in self.stats:
                self.stats[func_name] = []
            self.stats[func_name].append(execution_time)
        
        self.logger.debug(f"函数 {func_name} 执行时间: {execution_time:.4f}秒")
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """
        获取性能统计摘要
        
        Returns:
            性能统计信息
        """
        summary = {}
        with self.lock:
            for func_name, times in self.stats.items():
                if times:
                    summary[func_name] = {
                        'count': len(times),
                        'total': sum(times),
                        'average': sum(times) / len(times),
                        'min': min(times),
                        'max': max(times)
                    }
        return summary


class StructuredFormatter(logging.Formatter):
    """
    结构化日志格式化器
    
    将日志记录格式化为JSON格式，便于日志分析和处理。
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日志记录
        
        Args:
            record: 日志记录
            
        Returns:
            格式化后的JSON字符串
        """
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # 添加异常信息
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # 添加额外字段
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, ensure_ascii=False)


def get_logger(
    name: str, 
    level: str = "INFO", 
    log_file: Optional[str] = None,
    structured: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_performance: bool = False
) -> logging.Logger:
    """
    获取配置好的日志记录器
    
    创建并配置一个功能完整的日志记录器，支持多种输出格式和高级功能。
    
    Args:
        name: 日志记录器名称，通常使用模块名
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 可选的日志文件路径，支持日志轮转
        structured: 是否使用结构化JSON格式
        max_bytes: 日志文件最大大小（字节）
        backup_count: 保留的备份文件数量
        enable_performance: 是否启用性能监控
        
    Returns:
        配置好的日志记录器实例
        
    Raises:
        ValueError: 当日志级别无效时
        OSError: 当无法创建日志文件时
    """
    # 验证日志级别
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if level.upper() not in valid_levels:
        raise ValueError(f"无效的日志级别: {level}，有效值: {valid_levels}")
    
    logger = logging.getLogger(name)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 设置日志级别
    logger.setLevel(getattr(logging, level.upper()))
    
    # 添加敏感数据过滤器
    sensitive_filter = SensitiveDataFilter()
    
    # 选择格式化器
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(sensitive_filter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定了日志文件）
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 使用轮转文件处理器
            file_handler = logging.handlers.RotatingFileHandler(
                log_path, 
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            file_handler.addFilter(sensitive_filter)
            logger.addHandler(file_handler)
            
        except OSError as e:
            raise OSError(f"无法创建日志文件 {log_file}: {e}")
    
    # 防止日志向上传播，避免重复输出
    logger.propagate = False
    
    # 添加性能监控（如果启用）
    if enable_performance:
        perf_logger = PerformanceLogger(logger)
        logger.performance = perf_logger
    
    logger.info(f"日志记录器 '{name}' 初始化完成，级别: {level}")
    return logger


def performance_monitor(logger: Optional[logging.Logger] = None):
    """
    性能监控装饰器
    
    监控函数执行时间并记录到日志。
    
    Args:
        logger: 日志记录器，如果为None则使用默认记录器
        
    Returns:
        装饰器函数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                log = logger or logging.getLogger(func.__module__)
                
                if hasattr(log, 'performance'):
                    log.performance.log_execution_time(func.__name__, execution_time)
                else:
                    log.debug(f"函数 {func.__name__} 执行时间: {execution_time:.4f}秒")
        
        return wrapper
    return decorator


def setup_root_logger(level: str = "INFO", log_dir: str = "logs") -> None:
    """
    设置根日志记录器
    
    为整个应用程序设置统一的日志配置。
    
    Args:
        level: 根日志级别
        log_dir: 日志文件目录
    """
    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 配置根记录器
    root_logger = get_logger(
        "RobotAgent",
        level=level,
        log_file=str(log_path / "robotagent.log"),
        enable_performance=True
    )
    
    # 设置为根记录器
    logging.root = root_logger