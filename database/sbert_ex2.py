#---------------------------------------------------------------------------------------------
#A의 질문만 보고, 기존 대답 리스트 중에서 적절한 대답을 직접 선택
#---------------------------------------------------------------------------------------------


from sentence_transformers import SentenceTransformer, util
import sqlite3

# SBERT 모델 로드
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')  # 2등 성능 모델

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
    filenames = get_filenames_from_db(db_path)
    
    # 질문과 파일명을 벡터로 변환
    question_embedding = model.encode(question, convert_to_tensor=True)
    filename_embeddings = model.encode(filenames, convert_to_tensor=True)
    
    # 유사도 계산
    similarities = util.cos_sim(question_embedding, filename_embeddings)
    
    # 가장 높은 유사도 파일명 찾기
    best_match_idx = similarities.argmax()
    best_match_filename = filenames[best_match_idx]
    best_similarity_score = similarities[0][best_match_idx].item()
    
    return best_match_filename, best_similarity_score

# 파일에서 질문 읽기
text_file_path = "./requestA.txt"
question_text = read_text_file(text_file_path)
print(f"A's Question: {question_text}")

# 데이터베이스 경로
db_path = './mp3_database.db'

# 가장 적합한 파일명 찾기
best_match, similarity_score = find_best_match_from_question(question_text, db_path)
print(f"Best match filename: {best_match}, Similarity score: {similarity_score}")
