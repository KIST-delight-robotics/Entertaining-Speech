import sqlite3
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
import pandas as pd

DB_PATH = "./mp3_database.db"
VECTOR_DB_PATH = "./faiss_index"

# SQLite에서 데이터 로드
def load_data_from_db():
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT file_name, id FROM mp3_files"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# FAISS 벡터 DB 생성
def initialize_faiss(db_path, vector_db_path):
    embeddings = OllamaEmbeddings(model="llama3.1")
    faiss_index = FAISS()

    # SQLite에서 파일명 로드
    filenames = load_data_from_db()
    for _, row in filenames.iterrows():
        embedding = embeddings.embed_query(row['file_name'])
        faiss_index.add_texts([row['file_name']], embeddings=[embedding], metadatas=[{"id": row['id']}])

    # FAISS 데이터베이스 저장
    os.makedirs(vector_db_path, exist_ok=True)  # 디렉토리 생성
    faiss_index.save_local(vector_db_path)
    print(f"FAISS index saved at {vector_db_path}")

# FAISS 인덱스 존재 여부 확인
def faiss_index_exists(vector_db_path):
    index_file = os.path.join(vector_db_path, "index")
    meta_file = os.path.join(vector_db_path, "index.pkl")
    return os.path.exists(index_file) and os.path.exists(meta_file)

# 초기화 실행
if not faiss_index_exists(VECTOR_DB_PATH):
    print("FAISS index does not exist. Creating a new one...")
    initialize_faiss(DB_PATH, VECTOR_DB_PATH)
else:
    print("FAISS index already exists.")
