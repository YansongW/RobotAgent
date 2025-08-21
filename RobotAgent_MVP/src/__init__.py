# RobotAgent MVP - 基于CAMEL的三智能体架构
# Chat Agent + Action Agent + Memory Agent

# 添加配置模块路径
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))