import logging
from typing import List
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class IndexConstructionModule:
    """索引构建模块 - 负责向量化和索引构建"""
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", index_save_path: str = "./vector_index"):
        """
        初始化索引构建模块

        Args:
            model_name: 嵌入模型名称
            index_save_path: 索引保存路径
        """
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.embeddings = None
        self.vectorstore = None
        self.setup_embeddings()
        
    def setup_embeddings(self):
        """初始化嵌入模型"""
        logger.info(f"正在初始化嵌入模型: {self.model_name}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        logger.info("嵌入模型初始化完成")

    def build_vector_index(self, chunks: List[Document]) -> FAISS:
        """
        构建向量索引
        
        Args:
            chunks: 文档块列表
            
        Returns:
            FAISS向量存储对象
        """
        logger.info("正在构建FAISS向量索引...")
        
        if not chunks:
            raise ValueError("文档块列表不能为空")
        
        # 构建FAISS向量存储
        try:
            self.vectorstore = FAISS.from_documents(
                documents = chunks,
                embedding = self.embeddings
            )
        except Exception as e:
            logger.error(f"构建向量索引时出错: {e}")
            raise e

        logger.info("FAISS向量索引构建完成， 包含 {len(chunks)} 个向量")
        return self.vectorstore
    
    def add_documents(self, new_chunks: List[Document]):
        """
        向现有向量索引中添加新文档块
        
        Args:
            new_chunks: 新文档块列表
        """
        logger.info("正在向向量索引中添加新文档块...")
        
        if self.vectorstore is None:
            raise ValueError("向量存储未构建，无法添加文档")
        
        if not new_chunks:
            logger.warning("新文档块列表为空，未添加任何文档")
            return
        
        logger.info(f"正在添加 {len(new_chunks)} 个新文档到索引...")
        self.vectorstore.add_documents(new_chunks)        
        logger.info(f"新文档添加完成，当前索引包含 {self.vectorstore.index.ntotal} 个向量")



    def save_vector_index(self):
        """保存向量索引到指定路径"""
        if self.vectorstore is None:
            raise ValueError("向量存储未构建，无法保存")
        
        save_path = Path(self.index_save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.vectorstore.save_local(str(save_path))
        logger.info(f"向量索引已保存到: {self.index_save_path}")


    def load_vector_index(self) -> FAISS:
        """
        从配置的路径加载向量索引

        Returns:
            加载的向量存储对象，如果加载失败返回None
        """
        if not self.embeddings:
            self.setup_embeddings()

        load_path = Path(self.index_save_path)
        if not load_path.exists():
            logger.warning(f"索引路径不存在: {self.index_save_path}，无法加载向量索引")
            return None
        
        try:
            logger.info(f"正在从路径加载向量索引: {self.index_save_path}")
            self.vectorstore = FAISS.load_local(
                self.index_save_path, 
                self.embeddings,
                allow_dangerous_deserialization=True)# 只能在受本人生成的向量索引文件里使用
            logger.info("向量索引加载完成")
            return self.vectorstore
        except Exception as e:
            logger.warning(f"加载向量索引失败: {e}，将构建新索引")
            return None
    

    def similarity_search(self, query:str, k:int=5) -> List[Document]:
        """
        在向量索引中进行相似度搜索
        
        Args:
            query: 查询字符串
            k: 返回的相似文档数量
            
        Returns:
            相似文档列表
        """
        if self.vectorstore is None:
            raise ValueError("向量存储未构建，无法进行相似度搜索")
        
        return self.vectorstore.similarity_search(query, k=k)

