# ========== 前置兼容配置（解决联网超时+低版本SQLite报错） ==========
import os
# HuggingFace国内镜像，避免连接超时
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 修复Python3.8 sqlite3版本过低问题
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ========== 步骤1：构建并持久化向量库 ==========
def build_vector_database():
    # 1. 加载csv问答数据
    df = pd.read_csv('data/campus_data.csv')

    # 2. 初始化本地中文嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh"
    )

    # 3. 提取文本与元数据
    texts = df['answer'].tolist()
    metadatas = df[['id', 'category', 'question']].to_dict('records')

    # 4. 创建并持久化向量库
    vector_db = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory="./vector_db"
    )
    vector_db.persist()
    print(f"✅ 向量库构建完成，已存入{len(texts)}条记录\n")
    return vector_db, embeddings

# ========== 步骤2：向量检索测试函数 ==========
def search_knowledge(vector_db, query):
    results = vector_db.similarity_search(query, k=3)
    print(f"🔍 查询问题：{query}")
    print("-" * 50)
    for r in results:
        print(f"📋 元数据: {r.metadata}")
        print(f"📄 问答内容: {r.page_content}\n")
    return results

# ========== 程序入口：先建库，再执行检索测试 ==========
if __name__ == "__main__":
    # 构建向量库
    db, embed = build_vector_database()
    # 执行检索测试（实训要求：查询发烧请假相关）
    search_knowledge(db, "我发烧了怎么办？")