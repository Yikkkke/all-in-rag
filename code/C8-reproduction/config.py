"""
RAG系统配置文件
"""
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class RAGconfig:
    """配置类"""
    # 数据路径配置
    data_path: str = "../../data/C8/cook"
    index_save_path: str = "./vector_index"
    # 模型配置
    embedding_model_name:str = "BAAI/bge-small-zh-v1.5"
    llm_model_name:str = "kimi-k2-0711-preview"
    # 检索配置
    top_k:int = 3
    # 生成配置
    temperature: float = 0.1
    max_tokens:int = 2048


    def __post_init__(self):
        """初始化后的处理"""
        pass

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGconfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'data_path': self.data_path,
            'index_save_path': self.index_save_path,
            'embedding_model_name': self.embedding_model_name,
            'llm_model_name': self.llm_model_name,
            'top_k': self.top_k,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }

# 默认配置实例
DEFAULT_CONFIG = RAGconfig()