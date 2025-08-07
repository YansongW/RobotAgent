"""
ROS2动作库管理服务
管理ROS2动作库的加载、更新和查询
"""

import json
import logging
import os
import importlib.util
import inspect
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)

class ROS2Service:
    """ROS2动作库管理服务"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化ROS2服务
        
        Args:
            config: ROS2配置
        """
        self.config = config
        
        # 设置动作库目录
        self.actions_dir = Path(__file__).parent.parent / "ros2_actions"
        self.actions_dir.mkdir(parents=True, exist_ok=True)
        
        # 动作库缓存文件
        self.actions_cache_file = self.actions_dir / "actions_cache.json"
        
        # 动作库字典
        self.actions_dict = {}
        
        # 加载动作库
        self._load_actions()
        
        logger.info("ROS2动作库服务初始化完成")
    
    def _load_actions(self):
        """加载所有ROS2动作"""
        try:
            # 清空现有动作库
            self.actions_dict = {}
            
            # 扫描动作库目录
            for py_file in self.actions_dir.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                
                try:
                    self._load_action_from_file(py_file)
                except Exception as e:
                    logger.error(f"加载动作文件失败 {py_file}: {e}")
            
            # 保存到缓存文件
            self._save_actions_cache()
            
            logger.info(f"已加载 {len(self.actions_dict)} 个ROS2动作")
            
        except Exception as e:
            logger.error(f"加载ROS2动作库失败: {e}")
    
    def _load_action_from_file(self, file_path: Path):
        """
        从文件加载动作
        
        Args:
            file_path: Python文件路径
        """
        try:
            # 动态导入模块
            spec = importlib.util.spec_from_file_location(
                file_path.stem, file_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 查找动作函数
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) and hasattr(obj, '__ros2_action__'):
                    action_info = obj.__ros2_action__
                    
                    # 构建动作字典条目
                    action_entry = {
                        "name": action_info.get("name", name),
                        "description": action_info.get("description", ""),
                        "category": action_info.get("category", "general"),
                        "parameters": action_info.get("parameters", {}),
                        "examples": action_info.get("examples", []),
                        "function": obj,
                        "file_path": str(file_path),
                        "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    }
                    
                    self.actions_dict[name] = action_entry
                    logger.debug(f"加载动作: {name}")
            
        except Exception as e:
            logger.error(f"加载动作文件失败 {file_path}: {e}")
    
    def _save_actions_cache(self):
        """保存动作库缓存"""
        try:
            # 准备可序列化的数据
            cache_data = {}
            for action_name, action_info in self.actions_dict.items():
                cache_data[action_name] = {
                    "name": action_info["name"],
                    "description": action_info["description"],
                    "category": action_info["category"],
                    "parameters": action_info["parameters"],
                    "examples": action_info["examples"],
                    "file_path": action_info["file_path"],
                    "last_modified": action_info["last_modified"]
                }
            
            with open(self.actions_cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"保存动作库缓存失败: {e}")
    
    async def reload_actions(self):
        """重新加载动作库"""
        try:
            self._load_actions()
            logger.info("ROS2动作库已重新加载")
            return True
        except Exception as e:
            logger.error(f"重新加载动作库失败: {e}")
            return False
    
    async def get_all_actions(self) -> Dict[str, Any]:
        """
        获取所有动作信息
        
        Returns:
            所有动作的字典
        """
        try:
            result = {}
            for action_name, action_info in self.actions_dict.items():
                result[action_name] = {
                    "name": action_info["name"],
                    "description": action_info["description"],
                    "category": action_info["category"],
                    "parameters": action_info["parameters"],
                    "examples": action_info["examples"]
                }
            return result
        except Exception as e:
            logger.error(f"获取动作库失败: {e}")
            return {}
    
    async def get_actions_by_category(self, category: str) -> Dict[str, Any]:
        """
        根据类别获取动作
        
        Args:
            category: 动作类别
            
        Returns:
            指定类别的动作字典
        """
        try:
            result = {}
            for action_name, action_info in self.actions_dict.items():
                if action_info["category"] == category:
                    result[action_name] = {
                        "name": action_info["name"],
                        "description": action_info["description"],
                        "parameters": action_info["parameters"],
                        "examples": action_info["examples"]
                    }
            return result
        except Exception as e:
            logger.error(f"获取类别动作失败: {e}")
            return {}
    
    async def search_actions(self, keyword: str) -> Dict[str, Any]:
        """
        搜索动作
        
        Args:
            keyword: 搜索关键词
            
        Returns:
            匹配的动作字典
        """
        try:
            result = {}
            keyword_lower = keyword.lower()
            
            for action_name, action_info in self.actions_dict.items():
                # 在名称、描述中搜索
                if (keyword_lower in action_info["name"].lower() or
                    keyword_lower in action_info["description"].lower()):
                    result[action_name] = {
                        "name": action_info["name"],
                        "description": action_info["description"],
                        "category": action_info["category"],
                        "parameters": action_info["parameters"],
                        "examples": action_info["examples"]
                    }
            
            return result
        except Exception as e:
            logger.error(f"搜索动作失败: {e}")
            return {}
    
    async def execute_action(
        self,
        action_name: str,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        执行动作
        
        Args:
            action_name: 动作名称
            parameters: 动作参数
            
        Returns:
            执行结果
        """
        try:
            if action_name not in self.actions_dict:
                return {
                    "success": False,
                    "error": f"动作不存在: {action_name}",
                    "available_actions": list(self.actions_dict.keys())
                }
            
            action_info = self.actions_dict[action_name]
            action_function = action_info["function"]
            
            # 准备参数
            params = parameters or {}
            
            # 执行动作
            if inspect.iscoroutinefunction(action_function):
                result = await action_function(**params)
            else:
                result = action_function(**params)
            
            return {
                "success": True,
                "action_name": action_name,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"执行动作失败 {action_name}: {e}")
            return {
                "success": False,
                "action_name": action_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_action_info(self, action_name: str) -> Optional[Dict[str, Any]]:
        """
        获取指定动作的详细信息
        
        Args:
            action_name: 动作名称
            
        Returns:
            动作信息
        """
        try:
            if action_name in self.actions_dict:
                action_info = self.actions_dict[action_name]
                return {
                    "name": action_info["name"],
                    "description": action_info["description"],
                    "category": action_info["category"],
                    "parameters": action_info["parameters"],
                    "examples": action_info["examples"],
                    "file_path": action_info["file_path"],
                    "last_modified": action_info["last_modified"]
                }
            return None
        except Exception as e:
            logger.error(f"获取动作信息失败: {e}")
            return None
    
    async def get_categories(self) -> List[str]:
        """
        获取所有动作类别
        
        Returns:
            类别列表
        """
        try:
            categories = set()
            for action_info in self.actions_dict.values():
                categories.add(action_info["category"])
            return sorted(list(categories))
        except Exception as e:
            logger.error(f"获取动作类别失败: {e}")
            return []
    
    async def validate_action_file(self, file_path: str) -> Dict[str, Any]:
        """
        验证动作文件格式
        
        Args:
            file_path: 文件路径
            
        Returns:
            验证结果
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return {
                    "valid": False,
                    "error": "文件不存在"
                }
            
            if file_path.suffix != ".py":
                return {
                    "valid": False,
                    "error": "文件必须是Python文件"
                }
            
            # 尝试加载文件
            spec = importlib.util.spec_from_file_location(
                file_path.stem, file_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 检查是否包含ROS2动作
            actions_found = []
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) and hasattr(obj, '__ros2_action__'):
                    actions_found.append(name)
            
            if not actions_found:
                return {
                    "valid": False,
                    "error": "文件中未找到ROS2动作函数"
                }
            
            return {
                "valid": True,
                "actions_found": actions_found,
                "message": f"找到 {len(actions_found)} 个动作"
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"验证失败: {str(e)}"
            }
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        获取动作库统计信息
        
        Returns:
            统计信息
        """
        try:
            categories = await self.get_categories()
            category_counts = {}
            
            for category in categories:
                actions = await self.get_actions_by_category(category)
                category_counts[category] = len(actions)
            
            return {
                "total_actions": len(self.actions_dict),
                "total_categories": len(categories),
                "category_counts": category_counts,
                "actions_dir": str(self.actions_dir),
                "cache_file_exists": self.actions_cache_file.exists()
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}
    
    async def get_actions_library(self) -> Dict[str, Any]:
        """
        获取完整的动作库信息
        
        Returns:
            动作库信息字典
        """
        try:
            return {
                "actions": self.actions_dict,
                "categories": await self.get_categories(),
                "stats": await self.get_stats()
            }
        except Exception as e:
            logger.error(f"获取动作库失败: {e}")
            return {}
    
    def get_available_actions(self) -> Dict[str, Any]:
        """
        获取可用动作列表（同步方法）
        
        Returns:
            可用动作字典
        """
        try:
            result = {}
            for action_name, action_info in self.actions_dict.items():
                result[action_name] = {
                    "name": action_info["name"],
                    "description": action_info["description"],
                    "category": action_info["category"],
                    "parameters": action_info["parameters"]
                }
            return result
        except Exception as e:
            logger.error(f"获取可用动作失败: {e}")
            return {}