# -*- coding: utf-8 -*-

# 动作文件生成器 (Action File Generator)
# 负责生成机器人动作指令文件和动作序列管理
# 作者: RobotAgent开发团队
# 版本: 0.0.1 (Initial Release)
# 更新时间: 2025-08-21

# 导入标准库
import os
import json
import uuid
import yaml
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
import math

# 导入项目基础组件
from config import MessageType, AgentMessage


class ActionType(Enum):
    """动作类型"""
    MOVE = "move"                   # 移动动作
    ROTATE = "rotate"               # 旋转动作
    GRAB = "grab"                   # 抓取动作
    RELEASE = "release"             # 释放动作
    GESTURE = "gesture"             # 手势动作
    EXPRESSION = "expression"       # 表情动作
    SPEECH = "speech"               # 语音动作
    WAIT = "wait"                   # 等待动作
    SEQUENCE = "sequence"           # 动作序列
    PARALLEL = "parallel"           # 并行动作
    CONDITIONAL = "conditional"     # 条件动作
    LOOP = "loop"                   # 循环动作


class ActionFormat(Enum):
    """动作文件格式"""
    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    CSV = "csv"
    CUSTOM = "custom"


class CoordinateSystem(Enum):
    """坐标系类型"""
    CARTESIAN = "cartesian"         # 笛卡尔坐标系
    POLAR = "polar"                 # 极坐标系
    JOINT = "joint"                 # 关节坐标系
    RELATIVE = "relative"           # 相对坐标系


@dataclass
class Position:
    """位置数据结构"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    coordinate_system: CoordinateSystem = CoordinateSystem.CARTESIAN
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'coordinate_system': self.coordinate_system.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        return cls(
            x=data.get('x', 0.0),
            y=data.get('y', 0.0),
            z=data.get('z', 0.0),
            coordinate_system=CoordinateSystem(data.get('coordinate_system', 'cartesian'))
        )


@dataclass
class Orientation:
    """方向数据结构"""
    roll: float = 0.0   # 翻滚角
    pitch: float = 0.0  # 俯仰角
    yaw: float = 0.0    # 偏航角
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'roll': self.roll,
            'pitch': self.pitch,
            'yaw': self.yaw
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Orientation':
        return cls(
            roll=data.get('roll', 0.0),
            pitch=data.get('pitch', 0.0),
            yaw=data.get('yaw', 0.0)
        )


@dataclass
class ActionParameter:
    """动作参数"""
    name: str
    value: Any
    unit: Optional[str] = None
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActionParameter':
        return cls(
            name=data['name'],
            value=data['value'],
            unit=data.get('unit'),
            description=data.get('description')
        )


@dataclass
class ActionCommand:
    """动作指令数据结构"""
    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action_type: ActionType = ActionType.MOVE
    name: str = ""
    description: str = ""
    
    # 位置和方向
    position: Optional[Position] = None
    orientation: Optional[Orientation] = None
    target_position: Optional[Position] = None
    target_orientation: Optional[Orientation] = None
    
    # 动作参数
    parameters: List[ActionParameter] = field(default_factory=list)
    
    # 时间控制
    duration: float = 1.0  # 动作持续时间（秒）
    delay: float = 0.0     # 延迟时间（秒）
    speed: float = 1.0     # 动作速度倍率
    
    # 条件和循环
    condition: Optional[str] = None  # 执行条件
    loop_count: int = 1              # 循环次数
    
    # 子动作（用于序列和并行动作）
    sub_actions: List['ActionCommand'] = field(default_factory=list)
    
    # 元数据
    priority: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_parameter(self, name: str, value: Any, unit: Optional[str] = None,
                     description: Optional[str] = None):
        """添加动作参数"""
        param = ActionParameter(name=name, value=value, unit=unit, description=description)
        self.parameters.append(param)
    
    def get_parameter(self, name: str) -> Optional[ActionParameter]:
        """获取动作参数"""
        for param in self.parameters:
            if param.name == name:
                return param
        return None
    
    def add_sub_action(self, action: 'ActionCommand'):
        """添加子动作"""
        self.sub_actions.append(action)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'action_id': self.action_id,
            'action_type': self.action_type.value,
            'name': self.name,
            'description': self.description,
            'position': self.position.to_dict() if self.position else None,
            'orientation': self.orientation.to_dict() if self.orientation else None,
            'target_position': self.target_position.to_dict() if self.target_position else None,
            'target_orientation': self.target_orientation.to_dict() if self.target_orientation else None,
            'parameters': [p.to_dict() for p in self.parameters],
            'duration': self.duration,
            'delay': self.delay,
            'speed': self.speed,
            'condition': self.condition,
            'loop_count': self.loop_count,
            'sub_actions': [a.to_dict() for a in self.sub_actions],
            'priority': self.priority,
            'tags': self.tags,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActionCommand':
        """从字典创建"""
        action = cls(
            action_id=data.get('action_id', str(uuid.uuid4())),
            action_type=ActionType(data.get('action_type', 'move')),
            name=data.get('name', ''),
            description=data.get('description', ''),
            duration=data.get('duration', 1.0),
            delay=data.get('delay', 0.0),
            speed=data.get('speed', 1.0),
            condition=data.get('condition'),
            loop_count=data.get('loop_count', 1),
            priority=data.get('priority', 0),
            tags=data.get('tags', []),
            metadata=data.get('metadata', {})
        )
        
        # 解析位置和方向
        if data.get('position'):
            action.position = Position.from_dict(data['position'])
        if data.get('orientation'):
            action.orientation = Orientation.from_dict(data['orientation'])
        if data.get('target_position'):
            action.target_position = Position.from_dict(data['target_position'])
        if data.get('target_orientation'):
            action.target_orientation = Orientation.from_dict(data['target_orientation'])
        
        # 解析参数
        for param_data in data.get('parameters', []):
            action.parameters.append(ActionParameter.from_dict(param_data))
        
        # 解析子动作
        for sub_data in data.get('sub_actions', []):
            action.sub_actions.append(ActionCommand.from_dict(sub_data))
        
        # 解析时间
        if data.get('created_at'):
            action.created_at = datetime.fromisoformat(data['created_at'])
        
        return action


@dataclass
class ActionSequence:
    """动作序列"""
    sequence_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    actions: List[ActionCommand] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_action(self, action: ActionCommand):
        """添加动作"""
        self.actions.append(action)
    
    def get_total_duration(self) -> float:
        """获取总持续时间"""
        total = 0.0
        for action in self.actions:
            total += action.delay + action.duration
        return total
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'sequence_id': self.sequence_id,
            'name': self.name,
            'description': self.description,
            'actions': [a.to_dict() for a in self.actions],
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActionSequence':
        """从字典创建"""
        sequence = cls(
            sequence_id=data.get('sequence_id', str(uuid.uuid4())),
            name=data.get('name', ''),
            description=data.get('description', ''),
            metadata=data.get('metadata', {})
        )
        
        # 解析动作
        for action_data in data.get('actions', []):
            sequence.actions.append(ActionCommand.from_dict(action_data))
        
        # 解析时间
        if data.get('created_at'):
            sequence.created_at = datetime.fromisoformat(data['created_at'])
        
        return sequence


class ActionFileGenerator:
    """动作文件生成器"""
    
    def __init__(self, output_dir: Optional[str] = None,
                 default_format: ActionFormat = ActionFormat.JSON):
        """初始化动作文件生成器
        
        Args:
            output_dir: 输出目录
            default_format: 默认文件格式
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./action_files")
        self.output_dir.mkdir(exist_ok=True)
        
        self.default_format = default_format
        self.logger = logging.getLogger(__name__)
        
        # 动作模板库
        self.action_templates: Dict[str, ActionCommand] = {}
        self.sequence_templates: Dict[str, ActionSequence] = {}
        
        # 初始化基础动作模板
        self._initialize_templates()
    
    def _initialize_templates(self):
        """初始化基础动作模板"""
        # 基础移动动作
        move_template = ActionCommand(
            action_type=ActionType.MOVE,
            name="基础移动",
            description="机器人基础移动动作",
            duration=2.0
        )
        move_template.add_parameter("max_speed", 1.0, "m/s", "最大移动速度")
        move_template.add_parameter("acceleration", 0.5, "m/s²", "加速度")
        self.action_templates["move_basic"] = move_template
        
        # 基础旋转动作
        rotate_template = ActionCommand(
            action_type=ActionType.ROTATE,
            name="基础旋转",
            description="机器人基础旋转动作",
            duration=1.5
        )
        rotate_template.add_parameter("max_angular_speed", 90.0, "deg/s", "最大角速度")
        self.action_templates["rotate_basic"] = rotate_template
        
        # 抓取动作
        grab_template = ActionCommand(
            action_type=ActionType.GRAB,
            name="抓取物体",
            description="机器人抓取物体动作",
            duration=1.0
        )
        grab_template.add_parameter("grip_force", 50.0, "N", "抓取力度")
        grab_template.add_parameter("approach_speed", 0.1, "m/s", "接近速度")
        self.action_templates["grab_basic"] = grab_template
        
        # 手势动作
        gesture_template = ActionCommand(
            action_type=ActionType.GESTURE,
            name="挥手",
            description="友好挥手手势",
            duration=2.0
        )
        gesture_template.add_parameter("amplitude", 30.0, "deg", "挥手幅度")
        gesture_template.add_parameter("frequency", 1.0, "Hz", "挥手频率")
        self.action_templates["wave_hand"] = gesture_template
        
        self.logger.info("初始化动作模板完成")
    
    def create_action(self, action_type: ActionType, name: str = "",
                     description: str = "", **kwargs) -> ActionCommand:
        """创建动作指令
        
        Args:
            action_type: 动作类型
            name: 动作名称
            description: 动作描述
            **kwargs: 其他参数
            
        Returns:
            动作指令对象
        """
        action = ActionCommand(
            action_type=action_type,
            name=name or f"{action_type.value}_action",
            description=description or f"{action_type.value}动作"
        )
        
        # 设置参数
        for key, value in kwargs.items():
            if hasattr(action, key):
                setattr(action, key, value)
            else:
                action.add_parameter(key, value)
        
        return action
    
    def create_move_action(self, target_x: float, target_y: float, target_z: float = 0.0,
                          duration: float = 2.0, speed: float = 1.0) -> ActionCommand:
        """创建移动动作"""
        action = self.create_action(
            ActionType.MOVE,
            name="移动到目标位置",
            description=f"移动到坐标 ({target_x}, {target_y}, {target_z})",
            duration=duration,
            speed=speed
        )
        
        action.target_position = Position(target_x, target_y, target_z)
        return action
    
    def create_rotate_action(self, target_yaw: float, duration: float = 1.5,
                           speed: float = 1.0) -> ActionCommand:
        """创建旋转动作"""
        action = self.create_action(
            ActionType.ROTATE,
            name="旋转到目标角度",
            description=f"旋转到角度 {target_yaw}°",
            duration=duration,
            speed=speed
        )
        
        action.target_orientation = Orientation(yaw=target_yaw)
        return action
    
    def create_grab_action(self, target_x: float, target_y: float, target_z: float,
                          grip_force: float = 50.0, duration: float = 1.0) -> ActionCommand:
        """创建抓取动作"""
        action = self.create_action(
            ActionType.GRAB,
            name="抓取物体",
            description=f"抓取位于 ({target_x}, {target_y}, {target_z}) 的物体",
            duration=duration
        )
        
        action.target_position = Position(target_x, target_y, target_z)
        action.add_parameter("grip_force", grip_force, "N", "抓取力度")
        return action
    
    def create_sequence(self, name: str, description: str = "",
                       actions: Optional[List[ActionCommand]] = None) -> ActionSequence:
        """创建动作序列"""
        sequence = ActionSequence(
            name=name,
            description=description or f"{name}动作序列"
        )
        
        if actions:
            for action in actions:
                sequence.add_action(action)
        
        return sequence
    
    def save_action(self, action: ActionCommand, filename: Optional[str] = None,
                   format: Optional[ActionFormat] = None) -> Optional[str]:
        """保存单个动作到文件
        
        Args:
            action: 动作指令
            filename: 文件名
            format: 文件格式
            
        Returns:
            保存的文件路径
        """
        try:
            format = format or self.default_format
            
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"action_{action.action_type.value}_{timestamp}.{format.value}"
            
            file_path = self.output_dir / filename
            data = action.to_dict()
            
            if format == ActionFormat.JSON:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            elif format == ActionFormat.YAML:
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            
            elif format == ActionFormat.XML:
                self._save_action_xml(data, file_path)
            
            else:
                self.logger.error(f"不支持的文件格式: {format}")
                return None
            
            self.logger.info(f"保存动作文件: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"保存动作文件失败: {e}")
            return None
    
    def save_sequence(self, sequence: ActionSequence, filename: Optional[str] = None,
                     format: Optional[ActionFormat] = None) -> Optional[str]:
        """保存动作序列到文件"""
        try:
            format = format or self.default_format
            
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"sequence_{sequence.name}_{timestamp}.{format.value}"
            
            file_path = self.output_dir / filename
            data = sequence.to_dict()
            
            if format == ActionFormat.JSON:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            elif format == ActionFormat.YAML:
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            
            elif format == ActionFormat.XML:
                self._save_sequence_xml(data, file_path)
            
            else:
                self.logger.error(f"不支持的文件格式: {format}")
                return None
            
            self.logger.info(f"保存动作序列文件: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"保存动作序列文件失败: {e}")
            return None
    
    def load_action(self, file_path: str) -> Optional[ActionCommand]:
        """从文件加载动作"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                self.logger.error(f"文件不存在: {file_path}")
                return None
            
            format = ActionFormat(file_path.suffix[1:])  # 去掉点号
            
            if format == ActionFormat.JSON:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            elif format == ActionFormat.YAML:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
            
            elif format == ActionFormat.XML:
                data = self._load_action_xml(file_path)
            
            else:
                self.logger.error(f"不支持的文件格式: {format}")
                return None
            
            action = ActionCommand.from_dict(data)
            self.logger.info(f"加载动作文件: {file_path}")
            return action
            
        except Exception as e:
            self.logger.error(f"加载动作文件失败: {e}")
            return None
    
    def load_sequence(self, file_path: str) -> Optional[ActionSequence]:
        """从文件加载动作序列"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                self.logger.error(f"文件不存在: {file_path}")
                return None
            
            format = ActionFormat(file_path.suffix[1:])  # 去掉点号
            
            if format == ActionFormat.JSON:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            elif format == ActionFormat.YAML:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
            
            elif format == ActionFormat.XML:
                data = self._load_sequence_xml(file_path)
            
            else:
                self.logger.error(f"不支持的文件格式: {format}")
                return None
            
            sequence = ActionSequence.from_dict(data)
            self.logger.info(f"加载动作序列文件: {file_path}")
            return sequence
            
        except Exception as e:
            self.logger.error(f"加载动作序列文件失败: {e}")
            return None
    
    def _save_action_xml(self, data: Dict[str, Any], file_path: Path):
        """保存动作为XML格式"""
        root = ET.Element("action")
        
        for key, value in data.items():
            if value is not None:
                elem = ET.SubElement(root, key)
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        sub_elem = ET.SubElement(elem, sub_key)
                        sub_elem.text = str(sub_value)
                elif isinstance(value, list):
                    for item in value:
                        item_elem = ET.SubElement(elem, "item")
                        if isinstance(item, dict):
                            for sub_key, sub_value in item.items():
                                sub_elem = ET.SubElement(item_elem, sub_key)
                                sub_elem.text = str(sub_value)
                        else:
                            item_elem.text = str(item)
                else:
                    elem.text = str(value)
        
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding='utf-8', xml_declaration=True)
    
    def _save_sequence_xml(self, data: Dict[str, Any], file_path: Path):
        """保存动作序列为XML格式"""
        root = ET.Element("action_sequence")
        
        for key, value in data.items():
            if value is not None:
                if key == "actions":
                    actions_elem = ET.SubElement(root, "actions")
                    for action_data in value:
                        action_elem = ET.SubElement(actions_elem, "action")
                        for action_key, action_value in action_data.items():
                            if action_value is not None:
                                elem = ET.SubElement(action_elem, action_key)
                                elem.text = str(action_value)
                else:
                    elem = ET.SubElement(root, key)
                    elem.text = str(value)
        
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding='utf-8', xml_declaration=True)
    
    def _load_action_xml(self, file_path: Path) -> Dict[str, Any]:
        """从XML文件加载动作"""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        data = {}
        for elem in root:
            if len(elem) > 0:
                if elem.tag == "parameters":
                    data[elem.tag] = []
                    for item in elem:
                        item_data = {}
                        for sub_elem in item:
                            item_data[sub_elem.tag] = sub_elem.text
                        data[elem.tag].append(item_data)
                else:
                    data[elem.tag] = {}
                    for sub_elem in elem:
                        data[elem.tag][sub_elem.tag] = sub_elem.text
            else:
                data[elem.tag] = elem.text
        
        return data
    
    def _load_sequence_xml(self, file_path: Path) -> Dict[str, Any]:
        """从XML文件加载动作序列"""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        data = {}
        for elem in root:
            if elem.tag == "actions":
                data[elem.tag] = []
                for action_elem in elem:
                    action_data = {}
                    for action_child in action_elem:
                        action_data[action_child.tag] = action_child.text
                    data[elem.tag].append(action_data)
            else:
                data[elem.tag] = elem.text
        
        return data
    
    def get_template(self, template_name: str) -> Optional[ActionCommand]:
        """获取动作模板"""
        return self.action_templates.get(template_name)
    
    def add_template(self, template_name: str, action: ActionCommand):
        """添加动作模板"""
        self.action_templates[template_name] = action
        self.logger.info(f"添加动作模板: {template_name}")
    
    def list_templates(self) -> List[str]:
        """列出所有模板"""
        return list(self.action_templates.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        action_files = list(self.output_dir.glob("action_*"))
        sequence_files = list(self.output_dir.glob("sequence_*"))
        
        return {
            'output_directory': str(self.output_dir),
            'default_format': self.default_format.value,
            'template_count': len(self.action_templates),
            'action_file_count': len(action_files),
            'sequence_file_count': len(sequence_files),
            'total_files': len(action_files) + len(sequence_files)
        }