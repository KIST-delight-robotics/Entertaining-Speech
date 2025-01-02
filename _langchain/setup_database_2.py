import os
import time
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# SBERT 모델 초기화
sbert_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# FAISS 인덱스 파일 경로
faiss_index_file = "faiss_index.bin"
faiss_index = None

# 데이터베이스 설정
db_path = './mp3_database.db'
dir_path = './mp3_database'

# 벡터 정규화 함수
def normalize_vector(vector):
    return vector / np.linalg.norm(vector)

# FAISS 인덱스 로드 또는 생성
def load_faiss_index():
    if os.path.exists(faiss_index_file):
        index = faiss.read_index(faiss_index_file)
        if isinstance(index, faiss.IndexIDMap):
            print("Loaded existing FAISS index.")
            return index
        else:
            raise ValueError("Existing FAISS index is not wrapped with IndexIDMap.")
    else:
        base_index = faiss.IndexFlatIP(384)  # SBERT 벡터 차원(384), 내적 기반 검색
        print("Created a new FAISS index.")
        return faiss.IndexIDMap(base_index)

def save_faiss_index(index):
    faiss.write_index(index, faiss_index_file)
    print(f"FAISS index saved to {faiss_index_file}.")

# SQLite 초기화
def setup_database():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS mp3_files (
        id INTEGER PRIMARY KEY,
        file_name TEXT,
        file_path TEXT,
        faiss_index INTEGER UNIQUE
    )
    ''')
    conn.commit()
    conn.close()

# 데이터 삽입 및 동기화
def insert_data(mp3_file):
    global faiss_index
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 파일명 및 경로 설정
    file_name = os.path.splitext(mp3_file)[0]
    file_path = os.path.join(dir_path, mp3_file)

    # SBERT 임베딩 생성 및 정규화
    embedding = sbert_model.encode(file_name).astype("float32")
    normalized_embedding = normalize_vector(embedding)

    # SQLite에 데이터 삽입
    cursor.execute('''
    INSERT INTO mp3_files (file_name, file_path) 
    VALUES (?, ?)
    ''', (file_name, file_path))
    sqlite_id = cursor.lastrowid

    # FAISS에 벡터 추가
    embedding = np.array([normalized_embedding], dtype="float32")
    ids = np.array([sqlite_id], dtype="int64")
    faiss_index.add_with_ids(embedding, ids)

    # SQLite에 FAISS 인덱스 저장
    cursor.execute('''
    UPDATE mp3_files SET faiss_index = ? WHERE id = ?
    ''', (sqlite_id, sqlite_id))

    conn.commit()
    conn.close()

    # FAISS 인덱스 저장
    save_faiss_index(faiss_index)
    print(f"Added '{file_name}' with ID {sqlite_id} to SQLite and FAISS.")

# 데이터 삭제
def delete_data(mp3_file):
    global faiss_index
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # SQLite에서 데이터 가져오기
    cursor.execute('SELECT id FROM mp3_files WHERE file_name = ?', (os.path.splitext(mp3_file)[0],))
    result = cursor.fetchone()

    if result:
        sqlite_id = result[0]

        # SQLite에서 데이터 삭제
        cursor.execute('DELETE FROM mp3_files WHERE id = ?', (sqlite_id,))
        print(f"Deleted '{mp3_file}' with ID {sqlite_id} from SQLite.")

        # FAISS에서 벡터 삭제
        id_array = np.array([sqlite_id], dtype="int64")
        faiss_index.remove_ids(id_array)
        print(f"Removed vector with ID {sqlite_id} from FAISS.")

        # FAISS 인덱스 저장
        save_faiss_index(faiss_index)

    conn.commit()
    conn.close()

# FAISS 검색 기능
def search_faiss(query, top_k=15):
    query_embedding = normalize_vector(sbert_model.encode(query).astype("float32"))
    distances, indices = faiss_index.search(np.array([query_embedding]), top_k)
    return indices[0], distances[0]

# MP3 파일 처리 핸들러
class MP3Handler(FileSystemEventHandler):
    def __init__(self, db_path, dir_path):
        self.db_path = db_path
        self.dir_path = dir_path

    def on_created(self, event):
        if event.src_path.endswith('.mp3'):
            mp3_file = os.path.basename(event.src_path)
            insert_data(mp3_file)

    def on_deleted(self, event):
        if event.src_path.endswith('.mp3'):
            mp3_file = os.path.basename(event.src_path)
            delete_data(mp3_file)

# 디렉토리 모니터링 시작
def start_watching():
    event_handler = MP3Handler(db_path, dir_path)
    observer = Observer()
    observer.schedule(event_handler, path=dir_path, recursive=False)
    observer.start()
    print(f"Watching for new mp3 files in: {dir_path}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    # SQLite 및 FAISS 초기화
    setup_database()
    faiss_index = load_faiss_index()

    print("\n=== FAISS Index Content ===")
    print(f"Total vectors in FAISS index: {faiss_index.ntotal}")

    # 디렉토리 모니터링 시작
    start_watching()
