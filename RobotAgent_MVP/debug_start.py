#!/usr/bin/env python3
"""
RobotAgent MVP 调试启动脚本
用于调试启动问题
"""

import sys
import os
import traceback

# 添加backend路径
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

def check_dependencies():
    """检查依赖"""
    print("检查Python依赖...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'pydantic', 'redis', 'aioredis', 
        'httpx', 'yaml', 'jinja2', 'websockets', 'psutil'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} - 缺失")
    
    if missing_packages:
        print(f"\n缺失的包: {', '.join(missing_packages)}")
        print("请运行: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_config():
    """检查配置文件"""
    print("\n检查配置文件...")
    
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
    if os.path.exists(config_path):
        print(f"✓ 配置文件存在: {config_path}")
        return True
    else:
        print(f"✗ 配置文件不存在: {config_path}")
        return False

def check_directories():
    """检查必要目录"""
    print("\n检查目录结构...")
    
    required_dirs = [
        'logs',
        'memory_records', 
        'data',
        'backend',
        'config',
        'frontend'
    ]
    
    base_dir = os.path.dirname(__file__)
    for dir_name in required_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path):
            print(f"✓ {dir_name}")
        else:
            print(f"✗ {dir_name} - 创建中...")
            os.makedirs(dir_path, exist_ok=True)
            print(f"  已创建: {dir_path}")

def test_imports():
    """测试导入"""
    print("\n测试模块导入...")
    
    try:
        from utils.config import Config
        print("✓ Config")
    except Exception as e:
        print(f"✗ Config: {e}")
        return False
    
    try:
        from utils.logger import CustomLogger
        print("✓ CustomLogger")
    except Exception as e:
        print(f"✗ CustomLogger: {e}")
        return False
    
    try:
        from services.qwen_service import QwenService
        print("✓ QwenService")
    except Exception as e:
        print(f"✗ QwenService: {e}")
        return False
    
    try:
        from services.message_queue import MessageQueue
        print("✓ MessageQueue")
    except Exception as e:
        print(f"✗ MessageQueue: {e}")
        return False
    
    return True

def main():
    """主函数"""
    print("=== RobotAgent MVP 启动诊断 ===\n")
    
    # 检查依赖
    if not check_dependencies():
        return False
    
    # 检查配置
    if not check_config():
        return False
    
    # 检查目录
    check_directories()
    
    # 测试导入
    if not test_imports():
        return False
    
    print("\n=== 启动应用 ===")
    
    try:
        # 导入应用
        from app import app
        import uvicorn
        
        print("启动FastAPI应用...")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False
        )
        
    except Exception as e:
        print(f"启动失败: {e}")
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)