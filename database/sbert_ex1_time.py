import time
from sentence_transformers import SentenceTransformer, util
import sqlite3
import openai

# SBERT 모델 로드
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # 2등 성능 모델

# OpenAI API 키 설정
OPENAI_API_KEY = "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"

openai.api_key = OPENAI_API_KEY

# 텍스트 파일에서 내용을 읽는 함수
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# GPT에게 텍스트 파일 내용을 보내고 응답을 받는 함수
def get_gpt_response(file_path):
    user_input = read_text_file(file_path)
    start_time = time.time()  # GPT 응답 시작 시간 측정
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "상황을 제시하면 그에 맞는 적절하고 자연스러운 대답을 하면 돼. A와 B는 친구야. 예를 들면, A : 오늘 눈이 올까? B : '눈이 오면 좋겠다.' 이런식으로 B의 대답은 A의 질문에 적절하게 이루어질거야. 단, B의 대답에서 명사형 키워드를 추출하기 적합하게 대답해줘. 반문은 하지 않아도 되고, 질문 키워드가 들어가게 대답해줘."},
            {"role": "assistant", "content": "user가 A의 질문을 제시할거야."},
            {"role": "user", "content": user_input}
        ]
    )
    gpt_time = time.time() - start_time  # GPT 응답 소요 시간 계산
    return response['choices'][0]['message']['content'], gpt_time

# 데이터베이스에서 파일명 가져오는 함수
def get_filenames_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT file_name FROM mp3_files")
    filenames = [row[0] for row in cursor.fetchall()]
    conn.close()
    return filenames

# GPT 응답과 파일명 비교
def find_best_match(gpt_response, db_path):
    # 데이터베이스에서 파일명 가져오기
    start_time = time.time()  # 데이터베이스 검색 시작 시간 측정
    filenames = get_filenames_from_db(db_path)
    db_time = time.time() - start_time  # 데이터베이스 검색 소요 시간 계산

    # GPT 응답과 파일명을 벡터로 변환
    start_time = time.time()  # SBERT 비교 시작 시간 측정
    gpt_embedding = model.encode(gpt_response, convert_to_tensor=True)
    filename_embeddings = model.encode(filenames, convert_to_tensor=True)
    similarities = util.cos_sim(gpt_embedding, filename_embeddings)
    sbert_time = time.time() - start_time  # SBERT 비교 소요 시간 계산

    # 가장 높은 유사도 파일명 찾기
    best_match_idx = similarities.argmax()
    best_match_filename = filenames[best_match_idx]
    best_similarity_score = similarities[0][best_match_idx].item()

    return best_match_filename, best_similarity_score, db_time, sbert_time

# 메인 실행
text_file_path = "./requestA.txt"
start_time = time.time()  # 전체 실행 시작 시간 측정

gpt_response_text, gpt_time = get_gpt_response(text_file_path)
print(f"GPT Response: {gpt_response_text}")

db_path = './mp3_database.db'
best_match, similarity_score, db_time, sbert_time = find_best_match(gpt_response_text, db_path)

total_time = time.time() - start_time  # 전체 실행 소요 시간 계산

# 결과 출력
print(f"Best match filename: {best_match}, Similarity score: {similarity_score}")
print(f"GPT Response Time: {gpt_time:.2f} seconds")
print(f"Database Search Time: {db_time:.2f} seconds")
print(f"SBERT Comparison Time: {sbert_time:.2f} seconds")
print(f"Total Execution Time: {total_time:.2f} seconds")
