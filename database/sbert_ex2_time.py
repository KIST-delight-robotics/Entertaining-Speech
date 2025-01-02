import time
from sentence_transformers import SentenceTransformer, util
import sqlite3

# SBERT 모델 로드
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # 2등 성능 모델

# 텍스트 파일에서 A의 질문 읽는 함수
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# 데이터베이스에서 파일명 가져오는 함수
def get_filenames_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT file_name FROM mp3_files")
    filenames = [row[0] for row in cursor.fetchall()]
    conn.close()
    return filenames

# A의 질문과 파일명 비교
def find_best_match_from_question(question, db_path):
    # 데이터베이스에서 파일명 가져오기
    start_time = time.time()  # 데이터베이스 검색 시작 시간
    filenames = get_filenames_from_db(db_path)
    db_time = time.time() - start_time  # 데이터베이스 검색 소요 시간 계산

    # 질문과 파일명을 벡터로 변환 및 유사도 계산
    start_time = time.time()  # SBERT 검색 시작 시간
    question_embedding = model.encode(question, convert_to_tensor=True)
    filename_embeddings = model.encode(filenames, convert_to_tensor=True)
    similarities = util.cos_sim(question_embedding, filename_embeddings)
    sbert_time = time.time() - start_time  # SBERT 검색 소요 시간 계산

    # 가장 높은 유사도 파일명 찾기
    best_match_idx = similarities.argmax()
    best_match_filename = filenames[best_match_idx]
    best_similarity_score = similarities[0][best_match_idx].item()

    return best_match_filename, best_similarity_score, db_time, sbert_time

# 메인 실행
text_file_path = "./requestA.txt"
start_time = time.time()  # 전체 실행 시작 시간

question_text = read_text_file(text_file_path)
print(f"A's Question: {question_text}")

db_path = './mp3_database.db'
best_match, similarity_score, db_time, sbert_time = find_best_match_from_question(question_text, db_path)

total_time = time.time() - start_time  # 전체 실행 소요 시간 계산

# 결과 출력
print(f"Best match filename: {best_match}, Similarity score: {similarity_score}")
print(f"Database Search Time: {db_time:.2f} seconds")
print(f"SBERT Search Time: {sbert_time:.2f} seconds")
print(f"Total Execution Time: {total_time:.2f} seconds")
