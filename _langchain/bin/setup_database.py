import os
import sqlite3
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

DB_PATH = "./mp3_database.db"
VECTOR_DB_PATH = "./faiss_index"
FOLDER_PATH = "./mp3_files"

# Embedding 모델 초기화
embeddings = OllamaEmbeddings(model="llama3.1")

# 데이터베이스 설정
def setup_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS mp3_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT,
        file_path TEXT
    )
    ''')
    conn.commit()
    conn.close()

# SQLite에 mp3 파일 추가
def populate_sqlite_from_folder(db_path, folder_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for file in os.listdir(folder_path):
        if file.endswith('.mp3'):
            file_name = os.path.basename(file)
            file_path = os.path.join(folder_path, file)

            # 중복 확인 후 삽입
            cursor.execute("SELECT COUNT(*) FROM mp3_files WHERE file_name = ?", (file_name,))
            if cursor.fetchone()[0] == 0:
                cursor.execute("INSERT INTO mp3_files (file_name, file_path) VALUES (?, ?)", (file_name, file_path))
                print(f"Added to SQLite: {file_name}")
    
    conn.commit()
    conn.close()

# FAISS 벡터 DB 재생성
def rebuild_faiss_index(db_path, vector_db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT file_name, file_path FROM mp3_files")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("No data found in SQLite database to build FAISS index.")
        return

    print("Building new FAISS index...")
    file_names = [row[0] for row in rows]
    file_paths = [{"path": row[1]} for row in rows]

    # FAISS 인덱스 생성
    faiss_index = FAISS.from_texts(file_names, embedding=embeddings, metadatas=file_paths)
    os.makedirs(vector_db_path, exist_ok=True)
    faiss_index.save_local(vector_db_path)
    print(f"FAISS index saved at: {vector_db_path}")

# 메인 실행
if __name__ == "__main__":
    # 1. 데이터베이스 설정
    setup_database(DB_PATH)
    print("SQLite 데이터베이스 설정 완료.")

    # 2. 폴더 내 mp3 파일들을 SQLite에 추가
    print("폴더 내 mp3 파일을 SQLite에 추가 중...")
    populate_sqlite_from_folder(DB_PATH, FOLDER_PATH)

    # 3. FAISS 인덱스 재생성
    print("FAISS 인덱스를 재생성 중...")
    rebuild_faiss_index(DB_PATH, VECTOR_DB_PATH)

    print("모든 작업이 완료되었습니다.")
