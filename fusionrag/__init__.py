"""FusionRAG - 图增强检索生成 (GraphRAG) 知识库问答系统。"""

from .config import FusionRAGConfig
from .core import FusionRAG
from .query import QueryParam, QueryResult

__version__ = "0.1.0"

__all__ = ["FusionRAG", "FusionRAGConfig", "QueryParam", "QueryResult", "__version__"]
