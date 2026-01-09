"""
检索优化模块
"""

import logging
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class RetrievalOptimizationModule:
    """检索优化模块 - 负责混合检索和过滤"""
    
    def __init__(self, vectorstore: FAISS, chunks: List[Document]):
        """
        初始化检索优化模块
        
        Args:
            vectorstore: FAISS向量存储
            chunks: 文档块列表
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.setup_retrievers()

    def setup_retrievers(self):
        """设置向量检索器和BM25检索器
        
        向量检索器：相似性检索
        BM25检索器：基于词频的检索，关键词检索
        """
        logger.info("正在设置检索器...")

        # 向量检索器
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # BM25检索器
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,  # 文档块列表,BM25检索器需要原始文本来统计词频
            k=5
        )

        logger.info("检索器设置完成")
    
    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        混合检索 - 结合向量检索(密集检索)和BM25检索（关键词的稀疏检索），使用RRF重排

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            检索到的文档列表
        """
        # 分别获取向量检索和BM25检索结果
        vector_docs = self.vector_retriever.get_relevant_documents(query)
        bm25_docs = self.bm25_retriever.get_relevant_documents(query)
        # 使用RRF重排结果
        combined_results = self._rrf_rerank(vector_docs, bm25_docs, top_k)
        return combined_results[:top_k]
    

    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], k: int) -> List[Document]:   
        """
        使用RRF (Reciprocal Rank Fusion) 算法重排文档

        Args:
            vector_docs: 向量检索结果
            bm25_docs: BM25检索结果
            k: RRF参数，用于平滑排名

        Returns:
            重排后的文档列表
        """
        doc_scores = {} # dict: doc_id -> score
        doc_objects = {} #dict: doc_id -> Document
        
        # 处理向量检索结果
        for rank, doc in enumerate(vector_docs):
            # 使用文档内容的哈希作为唯一标识
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc
            # RRF得分计算1 / (k + rank)
            score = 1.0 / (rank + k + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
            logger.debug(f"向量检索 - 文档{rank+1}: RRF分数 = {score:.4f}")
        
        # 处理BM25检索结果
        for rank, doc in enumerate(bm25_docs):
           # 使用文档内容的哈希作为唯一标识
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc

            # RRF得分计算1 / (k + rank)
            score = 1.0 / (rank + k + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
            logger.debug(f"BM25检索 - 文档{rank+1}: RRF分数 = {score:.4f}")


        # 根据累积得分排序
        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # 构建最终结果列表，并把RRF分数加载回Document对象
        final_docs = []
        for doc_id, score in ranked_docs:
            doc = doc_objects[doc_id]
            # 将RRF分数作为元数据添加到文档中
            doc.metadata["rrf_score"] = score
            final_docs.append(doc)
        logger.info(f"RRF重排完成: 向量检索{len(vector_docs)}个文档, BM25检索{len(bm25_docs)}个文档, 合并后{len(final_docs)}个文档")
        return final_docs


    def metadata_filtered_search(self, query: str, metadata_filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
        """
        基于元数据过滤的检索

        Args:
            query: 查询文本
            metadata_filters: 元数据过滤条件字典
            top_k: 返回结果数量

        Returns:
            检索到的文档列表
        """
        # 先进行混合检索
        initial_docs = self.hybrid_search(query, top_k=top_k*3)  # 多取一些以便过滤后仍有足够结果

        # 应用元数据过滤
        filtered_docs = []
        for doc in initial_docs:
            match = True
            for key, value in metadata_filters.items():
                if key in doc.metadata:
                    # 对于metadata_filters的字典values，支持列表和单值匹配
                    if isinstance(value, list):
                        if doc.metadata[key] not in value:
                            match = False
                            break
                    else:
                        if doc.metadata[key] != value:
                            match = False
                            break
                else:
                    match = False
                    break
            if match:
                filtered_docs.append(doc)
                if len(filtered_docs) >= top_k:
                    break
        logger.info(f"应用元数据过滤后，返回 {len(filtered_docs)} 个文档")
        return filtered_docs