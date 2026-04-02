"""
LangChain RAG Pipeline 实现
基于 learn.txt 文本文件的检索增强生成示例
使用 LangChain 组件实现
"""

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ============================================================
# 1. 配置参数
# ============================================================

# CHUNK_SIZE = 500
# CHUNK_OVERLAP = 50
CHUNK_SIZE = 1000 
CHUNK_OVERLAP = 100 
TOP_K = 3
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DB_PATH = "./chroma_db_langchain1000"
MODEL_NAME = "openai/gpt-5.4"


# ============================================================
# 2. LangChain RAG 类
# ============================================================

class LangChainRAG:
    def __init__(self, db_path: str = DB_PATH):
        """初始化 LangChain RAG 系统"""
        print(f"🔄 加载 Embedding 模型: {EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        print(f"🔄 初始化向量数据库: {db_path}")
        self.db_path = db_path
        self.vectorstore = None
        self.rag_chain = None

        print("🔄 初始化 LLM (通过 OpenRouter)")
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        )

        print("✅ 初始化完成")

    def index_document(self, file_path: str) -> int:
        """
        索引文档：加载、分块、向量化、存储
        """
        # 1. 加载文档
        print(f"📄 加载文档: {file_path}")
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        print(f"📄 加载完成: {len(documents)} 个文档")

        # 2. 分块
        print(f"📦 分块中... (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"📦 分块完成: {len(chunks)} 个块")

        # 3. 创建向量存储
        print("🔄 生成向量并存储...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        print(f"✅ 索引完成，总块数: {len(chunks)}")

        # 4. 创建 RAG Chain
        self._create_rag_chain()

        return len(chunks)

    def load_existing_index(self) -> bool:
        """
        加载已有的向量存储
        """
        if os.path.exists(self.db_path):
            print(f"🔄 加载已有向量存储: {self.db_path}")
            self.vectorstore = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )
            self._create_rag_chain()
            print("✅ 加载完成")
            return True
        return False

    def _create_rag_chain(self):
        """创建 RAG Chain"""
        # 定义 Prompt 模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", """基于以下上下文回答问题。如果上下文中没有相关信息，请说明。

上下文：
{context}"""),
            ("human", "{input}")
        ])

        # 创建文档处理 chain
        document_chain = create_stuff_documents_chain(self.llm, prompt)

        # 创建检索器
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": TOP_K}
        )

        # 创建 RAG chain
        self.rag_chain = create_retrieval_chain(retriever, document_chain)

    def query(self, query_text: str) -> dict:
        """
        执行 RAG 查询
        """
        if self.rag_chain is None:
            raise ValueError("请先索引文档或加载已有索引")

        # 调用 RAG chain
        response = self.rag_chain.invoke({"input": query_text})

        return {
            "query": query_text,
            "answer": response["answer"],
            "contexts": [doc.page_content for doc in response["context"]],
            "sources": [doc.metadata.get("source", "unknown") for doc in response["context"]]
        }

    def clear(self):
        """清空向量库"""
        import shutil
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
            print("🗑️ 向量库已清空")


# ============================================================
# 3. 主程序
# ============================================================

def main():
    # 初始化 RAG
    rag = LangChainRAG()

    # 尝试加载已有索引，否则重新索引
    doc_path = os.path.join(os.path.dirname(__file__), "chatWithNavar-2.md")

    if not rag.load_existing_index():
        if os.path.exists(doc_path):
            rag.index_document(doc_path)
        else:
            print(f"❌ 文件不存在: {doc_path}")
            return

    # 测试查询
    print("\n" + "="*60)
    print("🔍 测试查询")
    print("="*60)

    # queries = [
    #     "Gabriel Petersson 的学习方法是什么？",
    #     "什么是递归式知识填充？",
    #     "Unknown Unknowns 是什么意思？",
    #     "四象限学习框架有哪些？"
    # ]

    # for q in queries:
    #     print(f"\n📌 问题: {q}")
    #     print("-" * 40)
    #     result = rag.query(q)

    #     for i, ctx in enumerate(result["contexts"]):
    #         print(f"\n[相关块 {i+1}]")
    #         print(ctx[:200] + "..." if len(ctx) > 200 else ctx)

    #     print(f"\n🤖 AI 回答: {result['answer'][:300]}...")
    #     print("\n" + "="*60)

    # 交互式查询
    print("\n🤖 进入交互模式 (输入 'quit' 退出)")
    print("-" * 40)

    while True:
        try:
            user_query = input("\n请输入问题: ").strip()
            if user_query.lower() == 'quit':
                break
            if not user_query:
                continue

            result = rag.query(user_query)

            print(f"\n📚 检索到 {len(result['contexts'])} 个相关块:")
            print("-" * 40)

            for i, ctx in enumerate(result["contexts"]):
                print(f"\n[块 {i+1}]")
                print(ctx[:300] + "..." if len(ctx) > 300 else ctx)

            print("\n🤖 AI 回答:")
            print("-" * 40)
            print(result["answer"])

        except KeyboardInterrupt:
            print("\n\n👋 再见!")
            break


if __name__ == "__main__":
    main()
