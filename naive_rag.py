"""
Naive RAG Pipeline 实现
基于 learn.txt 文本文件的简单检索增强生成示例

RAG (Retrieval-Augmented Generation) 检索增强生成：
1. 将文档分块并向量化存储到向量数据库
2. 用户提问时，先检索相关文档块
3. 将检索到的上下文和问题一起交给大模型生成回答

核心流程：文档处理 → 向量化 → 存储 → 检索 → 生成
"""

import os
from typing import List, Tuple
from dotenv import load_dotenv

# 向量化和向量存储
from sentence_transformers import SentenceTransformer  # HuggingFace 的句子嵌入模型，用于将文本转换为向量
import chromadb  # 轻量级向量数据库，用于存储和检索向量

# 大模型接口
from openrouter import OpenRouter  # OpenRouter API 客户端，用于访问各种大语言模型

# 加载环境变量（从 .env 文件读取 API 密钥等配置）
load_dotenv()

# ============================================================
# 1. 默认配置参数
# ============================================================

# 文本分块默认配置
DEFAULT_CHUNK_SIZE = 1000      # 每个文本块的字符数（控制分块大小，影响检索精度）
DEFAULT_CHUNK_OVERLAP = 100    # 相邻块之间的重叠字符数（避免信息被切断，保持上下文连贯性）

# 检索默认配置
DEFAULT_TOP_K = 3             # 检索时返回最相关的 K 个文档块（数量影响上下文长度和信息量）

# Embedding 模型默认配置
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 句子变换器模型，将文本转为 384 维向量
                                              # 英文场景推荐，中文可换用 "paraphrase-multilingual-MiniLM-L12-v2"

# 向量数据库默认配置
DEFAULT_DB_PATH = "./chroma_dbnaive1"           # ChromaDB 数据持久化存储路径
DEFAULT_COLLECTION_NAME = "learn_documents1"  # 文档集合名称，用于分类管理不同来源的文档

# ============================================================
# 2. 文本分块函数
# ============================================================

import nltk
from nltk.tokenize import sent_tokenize

# 下载必要的nltk资源（只需要运行一次）
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    将文本分割成指定大小的块，支持块间重叠，保持句子完整性
    
    参数:
        text: 要分块的文本
        chunk_size: 每个文本块的最大字符数
        chunk_overlap: 相邻块之间的重叠字符数
        
    返回:
        List[str]: 分块后的文本列表
    """
    if not text:
        return []
    
    # 首先将文本分割成句子
    sentences = sent_tokenize(text)
    if not sentences:
        return [text]
    
    chunks = []
    current_chunk = []
    current_chunk_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # 如果句子本身就超过了块大小，特殊处理
        if sentence_length > chunk_size:
            # 保存当前块
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_chunk_length = 0
            
            # 将超长句子分割成更小的块
            start = 0
            while start < sentence_length:
                end = min(start + chunk_size, sentence_length)
                # 尽量在标点处分割
                if end < sentence_length:
                    # 查找最后一个标点符号
                    punctuation = ['.', '!', '?', ';', '。', '！', '？', '；']
                    last_punc = -1
                    for p in punctuation:
                        pos = sentence.rfind(p, start, end)
                        if pos > last_punc:
                            last_punc = pos
                    if last_punc != -1:
                        end = last_punc + 1
                
                chunks.append(sentence[start:end])
                # 处理重叠
                if end < sentence_length:
                    start = max(end - chunk_overlap, start + 1)
                else:
                    start = end
            continue
        
        # 计算添加当前句子后的总长度（包括空格）
        new_length = current_chunk_length + sentence_length + (1 if current_chunk else 0)
        
        # 如果添加后超过限制，完成当前块并开始新块
        if new_length > chunk_size:
            # 保存当前块
            chunks.append(' '.join(current_chunk))
            
            # 处理重叠：从当前块末尾开始取句子，直到达到重叠大小
            overlap_chars = 0
            overlap_sentences = []
            
            # 从后往前遍历当前块的句子
            for sent in reversed(current_chunk):
                sent_len_with_space = len(sent) + 1  # +1 for space
                if overlap_chars + sent_len_with_space > chunk_overlap:
                    break
                overlap_chars += sent_len_with_space
                overlap_sentences.insert(0, sent)
            
            # 开始新块，包含重叠句子
            current_chunk = overlap_sentences
            current_chunk_length = sum(len(s) + 1 for s in current_chunk) - 1 if current_chunk else 0
        
        # 添加当前句子到块
        current_chunk.append(sentence)
        current_chunk_length = current_chunk_length + sentence_length + (1 if current_chunk_length > 0 else 0)
    
    # 添加最后一个块
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


# ============================================================
# 3. 向量数据库操作
# ============================================================

class NaiveRAG:
    def __init__(self, 
                 db_path: str = DEFAULT_DB_PATH,
                 collection_name: str = DEFAULT_COLLECTION_NAME,
                 embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
                 top_k: int = DEFAULT_TOP_K):
        """
        初始化 RAG 系统
        
        参数:
            db_path: 向量数据库存储路径
            collection_name: 文档集合名称
            embedding_model: 用于生成向量的模型名称
            chunk_size: 文本分块大小
            chunk_overlap: 文本分块重叠大小
            top_k: 默认检索返回的文档块数量
        """
        # 存储配置参数
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        # 初始化模型
        print(f"🔄 加载 Embedding 模型: {embedding_model}")
        self.embed_model = SentenceTransformer(embedding_model)

        # 初始化向量数据库
        print(f"🔄 初始化向量数据库: {db_path}")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(collection_name)
        print(f"✅ 初始化完成，当前文档块数: {self.collection.count()}")

    def index_document(self, file_path: str, replace_existing: bool = False) -> int:
        """
        索引文档：读取、分块、向量化、存储
        
        参数:
            file_path: 要索引的文档路径
            replace_existing: 如果文档已存在，是否替换现有索引
            
        返回:
            int: 索引的文档块数
            
        异常:
            FileNotFoundError: 当文件不存在时
            IOError: 当文件读取失败时
            Exception: 当索引过程中发生其他错误时
        """
        # 输入验证
        if not file_path:
            raise ValueError("文件路径不能为空")
        
        # 读取文档
        print(f"📄 读取文档: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ 文件不存在: {file_path}")
        except IOError as e:
            raise IOError(f"❌ 文件读取失败: {str(e)}")

        # 检查是否已存在该文档的索引
        doc_name = os.path.basename(file_path)
        existing_count = 0
        try:
            # 查询是否存在该文档的索引
            existing_results = self.collection.get(where={"source": doc_name})
            existing_count = len(existing_results.get("ids", []))
        except Exception:
            # 如果查询失败，假设文档不存在
            existing_count = 0
        
        if existing_count > 0:
            if replace_existing:
                print(f"🔄 文档 '{doc_name}' 已存在 {existing_count} 个块，正在替换...")
                # 删除现有索引
                try:
                    self.collection.delete(where={"source": doc_name})
                    print("🗑️  已删除现有索引")
                except Exception as e:
                    raise Exception(f"❌ 删除现有索引失败: {str(e)}")
            else:
                print(f"⚠️  文档 '{doc_name}' 已存在 {existing_count} 个块，跳过索引")
                return 0

        # 分块
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        print(f"📦 分块完成: {len(chunks)} 个块")

        # 生成 embedding
        print("🔄 生成向量...")
        try:
            embeddings = self.embed_model.encode(chunks).tolist()
        except Exception as e:
            raise Exception(f"❌ 向量生成失败: {str(e)}")

        # 生成 ID
        ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]

        # 存入向量库
        try:
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                ids=ids,
                metadatas=[{"source": doc_name} for _ in chunks]
            )
        except Exception as e:
            raise Exception(f"❌ 向量存储失败: {str(e)}")

        print(f"✅ 索引完成，总块数: {self.collection.count()}")
        return len(chunks)

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        """
        检索相关文档块
        返回: [(文档块, 相似度分数), ...]
        
        参数:
            query: 查询文本
            top_k: 返回的相关文档块数量（默认为初始化时设置的值）
            
        返回:
            List[Tuple[str, float]]: 文档块和相似度分数的列表
            
        异常:
            ValueError: 当查询为空或 top_k 无效时
            Exception: 当检索过程中发生错误时
        """
        # 输入验证
        if not query:
            raise ValueError("查询文本不能为空")
            
        # 使用默认值如果top_k为None
        if top_k is None:
            top_k = self.top_k
            
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        
        try:
            # 查询向量化
            query_embedding = self.embed_model.encode([query]).tolist()

            # 检索
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                include=["documents", "distances"]
            )

            # 整理结果
            retrieved = []
            for doc, dist in zip(results["documents"][0], results["distances"][0]):
                retrieved.append((doc, dist))

            return retrieved
        except Exception as e:
            raise Exception(f"❌ 检索失败: {str(e)}")

    def build_context(self, query: str, top_k: int = None) -> str:
        """
        构建上下文字符串
        
        参数:
            query: 查询文本
            top_k: 使用的相关文档块数量（默认为初始化时设置的值）
            
        返回:
            str: 构建好的上下文字符串
            
        异常:
            ValueError: 当查询为空时
            Exception: 当构建上下文过程中发生错误时
        """
        # 输入验证
        if not query:
            raise ValueError("查询文本不能为空")
        
        # 使用默认值如果top_k为None
        if top_k is None:
            top_k = self.top_k
        
        try:
            results = self.retrieve(query, top_k)
            context = "\n\n---\n\n".join([doc for doc, _ in results])
            return context
        except Exception as e:
            raise Exception(f"❌ 上下文构建失败: {str(e)}")

    def query(self, query: str, top_k: int = None) -> dict:
        """
        执行完整查询：检索 + 返回结果
        
        参数:
            query: 查询文本
            top_k: 返回的相关文档块数量（默认为初始化时设置的值）
            
        返回:
            dict: 查询结果，包含查询文本、上下文、距离和上下文文本
            
        异常:
            ValueError: 当查询为空时
            Exception: 当查询过程中发生错误时
        """
        # 输入验证
        if not query:
            raise ValueError("查询文本不能为空")
        
        # 使用默认值如果top_k为None
        if top_k is None:
            top_k = self.top_k
        
        try:
            results = self.retrieve(query, top_k)

            return {
                "query": query,
                "contexts": [doc for doc, _ in results],
                "distances": [dist for _, dist in results],
                "context_text": "\n\n---\n\n".join([doc for doc, _ in results])
            }
        except Exception as e:
            raise Exception(f"❌ 查询失败: {str(e)}")

    def clear(self):
        """
        清空向量库
        
        异常:
            Exception: 当清空向量库过程中发生错误时
        """
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(self.collection_name)
            print("🗑️ 向量库已清空")
        except Exception as e:
            raise Exception(f"❌ 清空向量库失败: {str(e)}")


# ============================================================
# 4. 简单的 LLM 接口（可选）
# ============================================================

def generate_answer_with_context(query: str, context: str, model: str = "openai/gpt-5.4") -> str:
    """
    使用上下文生成回答
    通过 OpenRouter 调用大模型
    
    参数:
        query: 用户的问题
        context: 用于回答问题的上下文信息
        model: 使用的大语言模型名称（默认为 openai/gpt-5.4）
        
    返回:
        str: 大模型生成的回答
        
    异常:
        ValueError: 当 OPENROUTER_API_KEY 环境变量未设置时
        Exception: 当调用大模型时发生错误时
    """
    # 检查 API 密钥是否存在
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("请设置 OPENROUTER_API_KEY 环境变量")
    
    # 构建提示词模板
    prompt = f"""基于以下上下文回答问题。如果上下文中没有相关信息，请说明。

上下文：
{context}

问题：{query}

回答："""

    with OpenRouter(api_key=api_key) as client:
        response = client.chat.send(
            model=model,  # 使用函数参数传入的模型
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content


# ============================================================
# 5. 主程序
# ============================================================

def main():
    """
    主程序入口，演示 NaiveRAG 的基本功能
    包括：初始化、索引文档、测试查询和交互式查询
    """
    try:
        # 初始化 RAG
        rag = NaiveRAG()

        # 索引文档
        doc_path = os.path.join(os.path.dirname(__file__), "chatWithNavar-2.md")
        if os.path.exists(doc_path):
            rag.index_document(doc_path)
        else:
            print(f"❌ 文件不存在: {doc_path}")
            return
    except Exception as e:
        print(f"❌ 初始化或索引失败: {str(e)}")
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
    #     try:
    #         print(f"\n📌 问题: {q}")
    #         print("-" * 40)
    #         result = rag.query(q, top_k=2)

    #         for i, (ctx, dist) in enumerate(zip(result["contexts"], result["distances"])):
    #             print(f"\n[相关块 {i+1}] (距离: {dist:.4f})")
    #             print(ctx[:200] + "..." if len(ctx) > 200 else ctx)

    #         print("\n" + "="*60)
    #     except Exception as e:
    #         print(f"❌ 查询失败: {str(e)}")
    #         print("\n" + "="*60)

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

            result = rag.query(user_query, top_k=3)

            print(f"\n📚 检索到 {len(result['contexts'])} 个相关块:")
            print("-" * 40)

            for i, (ctx, dist) in enumerate(zip(result["contexts"], result["distances"])):
                print(f"\n[块 {i+1}] (相似度距离: {dist:.4f})")
                print(ctx[:300] + "..." if len(ctx) > 300 else ctx)

            # 调用大模型生成回答
            print("\n🤖 AI 回答:")
            print("-" * 40)
            answer = generate_answer_with_context(user_query, result["context_text"])
            print(answer)

        except KeyboardInterrupt:
            print("\n\n👋 再见!")
            break
        except Exception as e:
            print(f"❌ 交互查询失败: {str(e)}")
            print("请检查您的输入或系统配置后重试")


if __name__ == "__main__":
    main()
