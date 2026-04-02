# rag_learn

RAG (Retrieval-Augmented Generation) 学习项目 - 实现文档检索与问答系统

## 项目结构

```
.
├── naive_rag.py           # 原生 RAG 实现（从零开始）
├── langchain_rag.py       # LangChain RAG 实现（使用框架）
├── chatWithNavar-2.md     # 知识库：Naval Ravikant 访谈录
├── .gitignore
└── CLAUDE.md
```

## 实现对比

| 特性 | naive_rag.py | langchain_rag.py |
|------|--------------|------------------|
| Embedding | sentence-transformers | HuggingFaceEmbeddings |
| 向量库 | chromadb | Chroma |
| 文本分块 | 自定义句子感知分块 | RecursiveCharacterTextSplitter |
| LLM | OpenRouter | ChatOpenAI (via OpenRouter) |
| 代码量 | ~500 行 | ~200 行 |

## 环境配置

### 1. 安装依赖

```bash
pip install sentence-transformers chromadb openrouter python-dotenv
pip install langchain langchain-community langchain-core nltk
```

### 2. 配置 API Key

创建 `.env` 文件：

```bash
OPENROUTER_API_KEY=your_api_key_here
```

## 使用方法

### 运行 Naive RAG

```bash
python naive_rag.py
```

### 运行 LangChain RAG

```bash
python langchain_rag.py
```

### 交互式查询

两个脚本都支持交互式查询模式，输入问题后按回车获取答案，输入 `quit` 退出。

## 配置参数

可在脚本中调整以下参数：

- `CHUNK_SIZE`: 文本分块大小（默认 1000 字符）
- `CHUNK_OVERLAP`: 分块重叠大小（默认 100 字符）
- `TOP_K`: 检索返回的相关块数量（默认 3）
- `MODEL_NAME`: 使用的 LLM 模型（默认 openai/gpt-5.4）
- `DB_PATH`: 向量数据库存储路径

## 知识库

当前使用的知识库是 `chatWithNavar-2.md`，包含 Naval Ravikant 与 Chris Williamson 的深度访谈，涵盖：

- 幸福与成功
- 决策与自由
- 人生选择
- 智慧与哲学
- 财富与育儿

## 参考资料

- [RAG 原始论文](https://arxiv.org/abs/2005.11401)
- [LangChain 文档](https://python.langchain.com/)
- [ChromaDB 文档](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
