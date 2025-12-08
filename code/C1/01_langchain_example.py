import os
# hugging face镜像设置，如果国内环境无法使用启用该设置
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

load_dotenv()

markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"

# 加载本地markdown文件
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()

# 文本分块
text_splitter = RecursiveCharacterTextSplitter()
chunks = text_splitter.split_documents(docs)

# 中文嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
  
# 构建向量存储
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)

# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:"""
                                          )

# 配置大语言模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=4096,
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 用户查询
question = "文中举了哪些例子？"

# 在向量存储中查询相关文档
retrieved_docs = vectorstore.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

try:
    answer = llm.invoke(prompt.format(question=question, context=docs_content))
    print(answer)
except Exception as e:
    err_str = str(e)
    # 针对常见的余额/计费错误给出可操作提示
    if "Insufficient Balance" in err_str or "402" in err_str or "insufficient" in err_str.lower():
        print("调用 API 失败：检测到余额/计费问题（Insufficient Balance / 402）。")
        print("解决建议：")
        print("- 检查并充值你的 OpenAI / DeepSeek 账户余额。")
        print("- 检查环境变量是否正确设置：例如 `DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY`。")
        print("- 如不想/不能使用外部付费 API，可改用本地 HuggingFace 模型作为回退。")
        print("（原始错误信息：{}）".format(err_str))
    else:
        print("调用 LLM 时发生错误：", err_str)
