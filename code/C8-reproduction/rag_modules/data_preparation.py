import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from pathlib import Path
import uuid

from config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

class DataPreparation:
    """数据准备模块 - 负责数据加载、清洗和预处理"""
    # 统一维护的分类与难度配置，供外部复用，避免关键词重复定义
    CATEGORY_MAPPING = {
        'meat_dish': '荤菜',
        'vegetable_dish': '素菜',
        'soup': '汤品',
        'dessert': '甜品',
        'breakfast': '早餐',
        'staple': '主食',
        'aquatic': '水产',
        'condiment': '调料',
        'drink': '饮品'
    }
    CATEGORY_LABELS = list(set(CATEGORY_MAPPING.values()))
    DIFFICULTY_LABELS = ['非常简单', '简单', '中等', '困难', '非常困难']
 

    def __init__(self, data_path:str):
        """
        Args:
            data_path: 数据文件夹路径
        """
        self.data_path = data_path
        self.docs: List[Document] = [] # 父文档（一个个完整的菜谱文档）
        self.chunks: List[Document] = [] # 子文档（按标题分割的小块）
        self.parent_child_map: Dict[str, str] = {} # 子文档ID到父文档ID的映射


    def load_data(self) -> List[Document]:
        """
        加载数据文件夹中的所有文档
        Args:
            data_path: 数据文件夹路径
        Returns:
            文档列表
        """
        logger.info(f"Loading data from {self.data_path}")

        documents = []
        for filename in Path(self.data_path).rglob('*.md'):
            try:
                # 读取markdown文件内容
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                # 获取markdown文件的相对路径
                try:
                    data_root = Path(self.data_path).resolve()
                    relative_path:str = Path(filename).resolve().relative_to(data_root).as_posix() # str类型
                    # logger.info(f"Processing file: root path is {data_root}, relative path is {relative_path}")
                except Exception as e:
                    relative_path:str = Path(filename).as_posix()
                    logger.warning(f"Could not determine relative path for {filename}: {e}")
                # 给父文档创建唯一ID：ID由文档相对路径相关hash生成
                parent_doc_id = hashlib.md5(relative_path.encode('utf-8')).hexdigest()
                # 创建Document对象
                doc = Document(
                    page_content = content,
                    metadata={
                        'doc_type': 'parent',
                        'source': str(filename),
                        'parent_id': parent_doc_id  # 标志是父文档
                    }
                )
                documents.append(doc)

            except Exception as e:
                logger.error(f"Error reading file {filename}: {e}")

        # 增加文档的元数据信息量
        for doc in documents:
            self._enhance_metadata(doc)


        self.documents = documents
        logger.info(f"Successfully loaded {len(documents)} documents from {self.data_path}")
        return documents


    def _enhance_metadata(self, doc: Document):
        """
        增强文档的元数据信息，添加字段： 种类、菜名、难度
        Args:
            doc: Document对象
        """
        file_path = Path(doc.metadata['source'])
        path_parts = file_path.parts  # 获取路径各部分，即路径上所有文件夹的名

        # 从文件路径中提取菜品分类
        doc.metadata['category'] = '其他' # 默认分类
        for key,value in self.CATEGORY_MAPPING.items():
            if key in path_parts:
                doc.metadata['category'] = value
                break
        
        # 从文件名中提取菜名（去掉扩展名）
        doc.metadata['dish_name'] = file_path.stem

        # 从文件内容中提取难度等级
        content = doc.page_content
        if "★★★★★" in content:
            doc.metadata['difficulty'] = '非常困难'
        elif "★★★★" in content:
            doc.metadata['difficulty'] = '困难'
        elif "★★★" in content:
            doc.metadata['difficulty'] = '中等'
        elif "★★" in content:
            doc.metadata['difficulty'] = '简单'
        elif "★" in content:   
            doc.metadata['difficulty'] = '非常简单'
        else:
            doc.metadata['difficulty'] = '未知'

    @classmethod
    def get_supported_categories(cls) -> List[str]:
        """对外提供支持的分类标签列表"""
        return cls.CATEGORY_LABELS

    @classmethod
    def get_supported_difficulties(cls) -> List[str]:
        """对外提供支持的难度标签列表"""
        return cls.DIFFICULTY_LABELS
    '''使用示例
    test3 = DataPreparation.get_supported_difficulties()
    print(test3)
    output：['非常简单', '简单', '中等', '困难', '非常困难']
    '''




    def chunk_documents(self) -> List[Document]:
        """
        Markdown结构感知分块

        Returns:
            分块后的文档列表
        """
        logger.info("正在进行Markdown结构感知分块...")

        if not self.documents:
            raise ValueError("没有加载的文档，无法进行分块。请先调用load_data()方法。")
        
        chunks = self._markdown_header_split()

        for i,chunk in enumerate(chunks):
            if 'chunk_id' not in chunk.metadata:
                # 为未分块的文档生成chunk_id
                chunk.metadata['chunk_id'] = str(uuid.uuid4())
                logger.warning(f"为未分块文档生成chunk_id: {chunk}")
            chunk.metadata['batch_index'] = i  # 全局块索引
            chunk.metadata['chunk_size'] = len(chunk.page_content)  # 块大小（字符数）
        
        self.chunks = chunks
        logger.info(f"分块完成，生成 {len(chunks)} 个文档块")
        return chunks


    def _markdown_header_split(self) -> List[Document]:
        """
        使用Markdown标题分割器进行结构化分割

        Returns:
            按标题结构分割的文档列表
        """
        headers_to_split_on = [
            ("#", "主标题"),      # 菜品名称
            ("##", "二级标题"),   # 必备原料、计算、操作等
            # ("###", "三级标题")   # 简易版本、复杂版本等
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )

        all_chunks = []
        for doc in self.documents:
            try:
                # 检查文档内容是否包含markdown标题

                # 对每个文档进行Markdown分割
                md_chunks = markdown_splitter.split_text(doc.page_content)
                logger.debug(f"文档 {doc.metadata.get('dish_name', '未知')} 分割成 {len(md_chunks)} 个chunk")

                # 如果没有分割成功，说明文档可能没有标题结构
                if len(md_chunks) == 0:
                    logger.warning(f"文档 {doc.metadata.get('source', '未知')} 未检测到Markdown标题结构，保留为单一chunk")
                
                # 为每个chunk创建Document对象，并继承父文档的元数据
                parent_id = doc.metadata['parent_id']
                for i,chunk in enumerate(md_chunks):
                    # chunk: Document类型
                    chunk_id = str(uuid.uuid4())  # 为每个chunk生成唯一ID
                    # 合并原文档元数据和新的标题元数据
                    chunk.metadata.update(doc.metadata)
                    chunk.metadata.update({
                        'doc_type': 'child',
                        "chunk_id": chunk_id,
                        "parent_id": parent_id,
                        "chunk_index": i,
                        })
                    # 建立父子映射关系
                    self.parent_child_map[chunk_id] = parent_id
                all_chunks.extend(md_chunks)

            except Exception as e:
                logger.warning(f"文档 {doc.metadata.get('source', '未知')} Markdown分割失败: {e}")
                # Markdown分割失败的情况下，保留整个原文档作为一个chunk                
                all_chunks.append(doc)  # 保留原文档

        logger.info(f"Markdown结构分割完成，生成 {len(all_chunks)} 个结构化块")
        return all_chunks
    



    def filter_documents_by_category(self, category: str) -> List[Document]:
        """
        根据分类过滤文档

        Args:
            category: 目标分类标签

        Returns:
            过滤后的文档列表
        """
        if category not in self.CATEGORY_LABELS:
            raise ValueError(f"不支持的分类标签: {category}")

        return [doc for doc in self.documents if doc.metadata.get('category') == category]

    def filter_documents_by_difficulty(self, difficulty: str) -> List[Document]:
        """
        按难度过滤文档
        
        Args:
            difficulty: 难度等级
            
        Returns:
            过滤后的文档列表
        """
        return [doc for doc in self.documents if doc.metadata.get('difficulty') == difficulty]

    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据统计信息

        Returns:
            统计信息字典
        """
        if not self.documents:
            return {}

        categories = {}
        difficulties = {}

        for doc in self.documents:
            # 统计分类
            category = doc.metadata.get('category', '未知')
            categories[category] = categories.get(category, 0) + 1

            # 统计难度
            difficulty = doc.metadata.get('difficulty', '未知')
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1

        return {
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks),
            'categories': categories,
            'difficulties': difficulties,
            'avg_chunk_size': sum(chunk.metadata.get('chunk_size', 0) for chunk in self.chunks) / len(self.chunks) if self.chunks else 0
        }


    
    def export_metadata(self, output_path: str):
        """
        导出元数据到JSON文件
        
        Args:
            output_path: 输出文件路径
        """
        import json
        
        metadata_list = []
        for doc in self.documents:
            metadata_list.append({
                'source': doc.metadata.get('source'),
                'dish_name': doc.metadata.get('dish_name'),
                'category': doc.metadata.get('category'),
                'difficulty': doc.metadata.get('difficulty'),
                'content_length': len(doc.page_content)
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)
        
        logger.info(f"元数据已导出到: {output_path}")


    def get_parent_document(self, child_chunks: List[Document]) -> List[Document]:
        """
        根据子文档列表获取对应的父文档列表（智能去重）
        
        Args:
            child_chunks: 子文档列表
            
        Returns:
            父文档列表（去重，按相关性排序）
        """
        # 统计每个父文档被匹配的次数（作为相关性指标）
        parent_relevance = {}
        parent_docs_map = {}

        # 收集每个父文档的id和相关性次数
        for chunk in child_chunks:
            child_id = chunk.metadata.get('chunk_id')
            parent_id = chunk.metadata.get('parent_id')
            if parent_id:
                parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1
                # 存储父文档对象（避免重复查找）: parent_docs_map-> {parent_id: Document}
                if parent_id not in parent_docs_map:
                    parent_doc = next((doc for doc in self.documents if doc.metadata['parent_id'] == parent_id), None)
                    if parent_doc:
                        parent_docs_map[parent_id] = parent_doc
        # 按相关性对父文档进行排序（次数多的排前面）
        sorted_parent_ids = sorted(parent_relevance.keys(), key=lambda x: parent_relevance[x], reverse=True)

        # 构建按相关性排序后的父文档列表
        parent_docs = []
        for parent_id in sorted_parent_ids:
            if parent_id in parent_docs_map:
                parent_docs.append(parent_docs_map[parent_id])


        # 收集父文档名称和相关性信息用于日志
        parent_info = []
        for doc in parent_docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            parent_id = doc.metadata.get('parent_id')
            relevance_count = parent_relevance.get(parent_id, 0)
            parent_info.append(f"{dish_name}({relevance_count}块)")

        logger.info(f"从 {len(child_chunks)} 个子块中找到 {len(parent_docs)} 个去重父文档: {', '.join(parent_info)}")
        
        return parent_docs

"""测试"""
if __name__ == "__main__":
    config = DEFAULT_CONFIG
    data_path = config.data_path
    test = DataPreparation(data_path)
    test.load_data()

    test.chunk_documents()
    test.get_parent_document(test.chunks)