import os
import re
import sqlite3
import openai
from konlpy.tag import Okt
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from sentence_transformers import SentenceTransformer
import pickle

# GPT API Key 설정
openai.api_key = "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"  

# SBERT 모델 로드
sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


# 데이터베이스 연결 및 테이블 확인 후 필요 시 생성
def setup_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 데이터베이스 연결 및 테이블 확인 후 필요 시 생성
def setup_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 테이블이 없으면 생성
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS mp3_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT,
        file_path TEXT,
        embedding BLOB
    )
    ''')

    # 'embedding' 컬럼 확인 및 추가
    cursor.execute("PRAGMA table_info(mp3_files);")
    columns = [col[1] for col in cursor.fetchall()]
    if 'embedding' not in columns:
        cursor.execute("ALTER TABLE mp3_files ADD COLUMN embedding BLOB")
        print("'embedding' 컬럼이 추가되었습니다.")

    conn.commit()
    conn.close()

# mp3 파일을 데이터베이스에 추가하는 함수
def insert_mp3_to_db(db_path, mp3_file, folder_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    file_name = os.path.splitext(mp3_file)[0]  # 파일명에서 확장자 제거
    file_path = os.path.join(folder_path, mp3_file)

    # 데이터베이스에 삽입
    cursor.execute('''
    INSERT INTO mp3_files (file_name, file_path) 
    VALUES (?, ?,)
    ''', (mp3_file, file_path))

    conn.commit()
    conn.close()

# mp3 파일을 업데이트하는 함수 (키워드가 null인 경우만)
def update_mp3_to_db(db_path, mp3_file, folder_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    file_name = os.path.splitext(mp3_file)[0]  # 파일명에서 확장자 제거
    file_path = os.path.join(folder_path, mp3_file)

    conn.commit()
    conn.close()

# mp3 파일을 데이터베이스에서 삭제하는 함수
def delete_mp3_from_db(db_path, mp3_file):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 파일명에서 확장자 제거
    file_name = os.path.splitext(mp3_file)[0]

    # 데이터베이스에서 해당 파일 삭제
    cursor.execute('''
    DELETE FROM mp3_files WHERE file_name = ?
    ''', (file_name,))

    conn.commit()
    conn.close()

# # 디렉토리 내 모든 기존 mp3 파일을 처리하는 함수
# def process_existing_files(db_path, folder_path):
#     for mp3_file in os.listdir(folder_path):
#         if mp3_file.endswith('.mp3'):
#             print(f"Processing existing file: {mp3_file}")
#             insert_mp3_to_db(db_path, mp3_file, folder_path)
#             update_mp3_to_db(db_path, mp3_file, folder_path)

# 임베딩 계산 및 데이터베이스 업데이트
def update_embeddings_in_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 임베딩이 없는 데이터만 선택
    cursor.execute("SELECT id, file_name FROM mp3_files WHERE embedding IS NULL")
    rows = cursor.fetchall()

    print(f"{len(rows)}개의 파일에 대해 임베딩을 계산합니다...")

    for row in rows:
        file_id, file_name = row
        # SBERT로 임베딩 계산
        embedding = sbert_model.encode(file_name)
        # 임베딩을 바이너리 형태로 직렬화하여 저장
        embedding_blob = pickle.dumps(embedding)

        # 데이터베이스 업데이트
        cursor.execute("UPDATE mp3_files SET embedding = ? WHERE id = ?", (embedding_blob, file_id))

    conn.commit()
    conn.close()
    print("임베딩 계산 및 업데이트 완료!")

# mp3 파일을 데이터베이스에 추가하는 함수 (okt_keywords, gpt_keywords, embedding 포함)
def insert_mp3_to_db(db_path, mp3_file, folder_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    file_name = os.path.splitext(mp3_file)[0]  # 파일명에서 확장자 제거
    file_path = os.path.join(folder_path, mp3_file)

    # 키워드 및 임베딩 추출
    
    embedding = sbert_model.encode(file_name)
    embedding_blob = pickle.dumps(embedding)  # 바이너리 형태로 직렬화

    # 데이터베이스에 삽입
    cursor.execute('''
    INSERT INTO mp3_files (file_name, file_path, okt_keywords, gpt_keywords, embedding) 
    VALUES (?, ?, ?, ?, ?)
    ''', (mp3_file, file_path, okt_keywords, gpt_keywords, embedding_blob))

    conn.commit()
    conn.close()


# 저장된 임베딩 로드 및 유사도 계산
def search_with_embeddings(db_path, query, top_k=5):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 모든 임베딩 로드
    cursor.execute("SELECT file_name, embedding FROM mp3_files")
    rows = cursor.fetchall()

    file_names = []
    embeddings = []

    for row in rows:
        file_name = row[0]
        embedding_blob = row[1]
        if embedding_blob is not None:
            embedding = pickle.loads(embedding_blob)  # 바이너리를 Numpy 배열로 복원
            file_names.append(file_name)
            embeddings.append(embedding)

    conn.close()

    # 입력 질문 임베딩 계산
    query_embedding = sbert_model.encode(query).reshape(1, -1)

    # 코사인 유사도 계산
    similarities = cosine_similarity(query_embedding, np.vstack(embeddings))[0]

    # 유사도가 높은 순으로 정렬
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_results = [(file_names[i], similarities[i]) for i in top_indices]

    return top_results


# 파일 시스템 이벤트 핸들러
class MP3Handler(FileSystemEventHandler):
    def __init__(self, db_path, folder_path):
        self.db_path = db_path
        self.folder_path = folder_path
    
    def on_created(self, event):
        if event.src_path.endswith('.mp3'):  # mp3 파일만 처리
            mp3_file = os.path.basename(event.src_path)
            insert_mp3_to_db(self.db_path, mp3_file, self.folder_path)
            print(f"{mp3_file} has been added to the database with okt_keywords, gpt_keywords, and embedding.")

    def on_deleted(self, event):
        # mp3 파일이 삭제되면 데이터베이스에서도 해당 파일 삭제
        if event.src_path.endswith('.mp3'):
            mp3_file = os.path.basename(event.src_path)
            delete_mp3_from_db(self.db_path, mp3_file)
            print(f"{mp3_file} has been deleted from the database.")

# 자동화 시작 함수
def start_watching(folder_path, db_path):
    event_handler = MP3Handler(db_path, folder_path)
    observer = Observer()
    observer.schedule(event_handler, path=folder_path, recursive=False)
    observer.start()
    print(f"Watching for new mp3 files in: {folder_path}")

    try:
        while True:
            time.sleep(1)  # 계속 모니터링
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    
if __name__ == "__main__":
    db_path = './mp3_database.db'
    folder_path = './mp3_database'

    # 데이터베이스 설정
    setup_database(db_path)

    # 임베딩 업데이트
    update_embeddings_in_db(db_path)

    # 폴더 모니터링 시작
    start_watching(folder_path, db_path)

    # 검색 테스트
    user_query = "오늘 날씨와 관련된 정보는 뭐가 있을까?"
    results = search_with_embeddings(db_path, user_query)
    print("검색 결과:")
    for file_name, similarity in results:
        print(f"파일명: {file_name}, 유사도: {similarity:.4f}")