import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
import json
import time
from utils.config import config

class PerformanceLogger:
    """性能日志记录器"""
    
    def __init__(self):
        self.start_times = {}
        self.performance_log = self._setup_performance_logger()
    
    def _setup_performance_logger(self):
        """设置性能日志记录器"""
        logger = logging.getLogger('performance')
        logger.setLevel(logging.INFO)
        
        # 创建日志目录
        log_dir = Path(config.get('logging.file_path', './logs'))
        log_dir.mkdir(exist_ok=True)
        
        # 性能日志文件处理器
        perf_handler = RotatingFileHandler(
            log_dir / 'performance.log',
            maxBytes=50*1024*1024,  # 50MB
            backupCount=5,
            encoding='utf-8'
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        perf_handler.setFormatter(formatter)
        logger.addHandler(perf_handler)
        
        return logger
    
    def start_timer(self, operation_id: str):
        """开始计时"""
        self.start_times[operation_id] = time.time()
    
    def end_timer(self, operation_id: str, additional_data: dict = None):
        """结束计时并记录"""
        if operation_id in self.start_times:
            duration = time.time() - self.start_times[operation_id]
            del self.start_times[operation_id]
            
            log_data = {
                'operation': operation_id,
                'duration': round(duration, 4),
                'timestamp': datetime.now().isoformat()
            }
            
            if additional_data:
                log_data.update(additional_data)
            
            self.performance_log.info(json.dumps(log_data, ensure_ascii=False))
            return duration
        return None

class CustomLogger:
    """自定义日志记录器"""
    
    def __init__(self, name: str, log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config.get('logging.level', 'INFO')))
        
        # 避免重复添加处理器
        if not self.logger.handlers:
            self._setup_handlers(log_file)
    
    def _setup_handlers(self, log_file: Optional[str] = None):
        """设置日志处理器"""
        # 创建日志目录
        log_dir = Path(config.get('logging.file_path', './logs'))
        log_dir.mkdir(exist_ok=True)
        
        # 格式化器
        formatter = logging.Formatter(
            config.get('logging.format', 
                      '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 文件处理器
        if log_file is None:
            log_file = f"{self.logger.name}.log"
        
        file_handler = RotatingFileHandler(
            log_dir / log_file,
            maxBytes=int(config.get('logging.max_file_size', '50MB').replace('MB', '')) * 1024 * 1024,
            backupCount=config.get('logging.backup_count', 5),
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        self.logger.critical(message, extra=kwargs)
    
    def log_request(self, method: str, url: str, status_code: int, 
                   duration: float, user_id: str = None):
        """记录HTTP请求日志"""
        log_data = {
            'type': 'http_request',
            'method': method,
            'url': url,
            'status_code': status_code,
            'duration': round(duration, 4),
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        }
        self.info(json.dumps(log_data, ensure_ascii=False))
    
    def log_error(self, error: Exception, context: dict = None):
        """记录错误日志"""
        log_data = {
            'type': 'error',
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat()
        }
        
        if context:
            log_data['context'] = context
        
        self.error(json.dumps(log_data, ensure_ascii=False))

def setup_logging():
    """设置全局日志配置"""
    # 创建日志目录
    log_dir = Path(config.get('logging.file_path', './logs'))
    log_dir.mkdir(exist_ok=True)
    
    # 设置根日志级别
    logging.getLogger().setLevel(
        getattr(logging, config.get('logging.level', 'INFO'))
    )

# 全局实例
performance_logger = PerformanceLogger()

def get_logger(name: str, log_file: Optional[str] = None) -> CustomLogger:
    """获取日志记录器实例"""
    return CustomLogger(name, log_file)