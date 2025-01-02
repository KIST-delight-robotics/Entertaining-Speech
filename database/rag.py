import sqlite3
import openai
import numpy as np
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from bert_score import score
import pandas as pd

# OpenAI API 설정
openai.api_key = ""  

# Sentence-BERT 모델 로드
sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# SQLite 데이터베이스 경로
db_path = "./mp3_database.db"

# SQLite에서 데이터 로드
def load_data_from_db():
    conn = sqlite3.connect(db_path)
    query = "SELECT file_name FROM mp3_files"  # 'mp3_files' 테이블에서 file_name 열 읽기
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# SBERT로 문장 임베딩 생성
def get_sbert_embedding(text):
    return sbert_model.encode(text)

# 두 임베딩 간 코사인 유사도 계산
def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))

# 코사인 유사도로 상위 후보 필터링
def filter_top_candidates(df, query, top_k=15):
    query_embedding = get_sbert_embedding(query)
    # 파일명을 임베딩으로 변환
    df['embedding'] = df['file_name'].apply(get_sbert_embedding)
    # 코사인 유사도 계산
    df['cosine_similarity'] = df['embedding'].apply(lambda x: cos_sim(query_embedding, x))
    # 상위 후보 필터링
    top_candidates = df.sort_values(by="cosine_similarity", ascending=False).head(top_k)
    return top_candidates

# GPT로 상위 후보 평가
def evaluate_with_gpt(question, candidates):
    candidate_list = "\n".join([f"{i+1}. {row['file_name']}" for i, row in candidates.iterrows()])
    prompt = f"""질문: '{question}'의 대답으로 가장 적합한 파일명을 선택하세요. 후보 리스트는 다음과 같습니다:\n{candidate_list}\n\n가장 적합한 파일명을 3개 선택하고 (적절한 순으로 나열) 이유를 간략히 설명해주세요."""

    messages = [
        {"role": "system", "content": "다음 질문과 파일명 간의 대화의 문맥적 연결을 평가합니다."},
        {"role": "user", "content": prompt}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages
    )
    best_match = response['choices'][0]['message']['content']
    return best_match

# 메인 함수
def main(question):
    # Step 1: 데이터베이스 로드
    df = load_data_from_db()

    # Step 2: SBERT로 코사인 유사도 계산 및 상위 후보 필터링
    top_candidates = filter_top_candidates(df, question, top_k=15)

    # Step 3: GPT로 최종 후보 평가
    gpt_result = evaluate_with_gpt(question, top_candidates)
    
    print("GPT가 선택한 파일명:")
    print(gpt_result)
    return gpt_result

# 실행
if __name__ == "__main__":
    print("user request: ")
    user_question = "점심 먹었어?"
    print(user_question)
    result = main(user_question)