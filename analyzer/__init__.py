"""
微博热点分析器
Weibo Hot Topic Analyzer

模块说明：
- summary: 使用 AI 进行两阶段分析（微观事实提取 + 宏观关联分析）
- settings: 配置文件，包含 API 密钥、模型参数、文件路径等

Module Description:
- summary: Two-stage AI analysis (Micro-fact extraction + Macro correlation analysis)
- settings: Configuration file including API key, model parameters, file paths, etc.
"""

from .summary import main
from . import settings

__all__ = ['main', 'settings']
__version__ = '1.0.0'
