import os
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm
import faiss
import openai
import pandas as pd

# OpenAI API 설정
openai.api_key =  

# SBERT 모델 초기화
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# SQLite 및 FAISS 설정
db_path = "./mp3_database.db"
faiss_index_file = "faiss_index.bin"

# FAISS 인덱스 로드
def load_faiss_index():
    if os.path.exists(faiss_index_file):
        index = faiss.read_index(faiss_index_file)
        return faiss.IndexIDMap(index)  # ID 매핑을 지원하도록 래핑
    else:
        raise FileNotFoundError("FAISS 인덱스 파일이 존재하지 않습니다.")

# SQLite에서 파일명 및 ID 로드
def load_metadata_from_db():
    conn = sqlite3.connect(db_path)
    query = "SELECT id, file_name FROM mp3_files"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# SBERT로 문장 임베딩 생성
def get_sbert_embedding(text):
    embedding = sbert_model.encode(text)
    return embedding / norm(embedding)  # 정규화하여 코사인 유사도 기반 검색 가능

# 코사인 유사도 계산
def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))

# 모든 벡터와 코사인 유사도 계산
def compute_cosine_similarity(query_embedding, faiss_index, metadata):
    results = []
    for idx in range(faiss_index.ntotal):
        stored_embedding = faiss_index.reconstruct(idx)  # FAISS에서 벡터 복원
        cosine_similarity = cos_sim(query_embedding, stored_embedding)
        file_name = metadata.loc[metadata['id'] == idx, 'file_name'].values[0]
        results.append({"file_name": file_name, "cosine_similarity": cosine_similarity})
    
    # 코사인 유사도 기준 정렬
    results = sorted(results, key=lambda x: x["cosine_similarity"], reverse=True)
    return results

# GPT로 최종 평가
def evaluate_with_gpt(question, candidates, top_n=3):
    candidate_list = "\n".join([f"{i+1}. {candidate['file_name']} (cos_sim: {candidate['cosine_similarity']:.4f})" for i, candidate in enumerate(candidates)])
    prompt = f"""질문: '{question}'의 대답으로 가장 적합한 파일명을 선택하세요. 후보 리스트는 다음과 같습니다:\n{candidate_list}\n\n가장 적합한 파일명을 {top_n}개 선택하고 (적절한 순으로 나열) 이유를 간략히 설명해주세요."""

    messages = [
        {"role": "system", "content": "다음 질문과 파일명 간의 대화의 문맥적 연결을 평가합니다."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"OpenAI API 호출 중 오류 발생: {e}")
        return "GPT 평가 실패"

# 메인 함수
def main(question):
    # Step 1: 데이터베이스 및 FAISS 인덱스 로드
    faiss_index = load_faiss_index()
    metadata = load_metadata_from_db()

    # Step 2: 사용자 질문을 SBERT 임베딩으로 변환
    query_embedding = get_sbert_embedding(question)

    # Step 3: 모든 벡터와 코사인 유사도 계산
    results = compute_cosine_similarity(query_embedding, faiss_index, metadata)
    print("\nSBERT 코사인 유사도로 정렬된 상위 후보:")
    for result in results[:10]:  # 상위 10개 출력
        print(f"파일명: {result['file_name']}, 코사인 유사도: {result['cosine_similarity']:.4f}")

    # Step 4: GPT로 최종 후보 평가
    gpt_result = evaluate_with_gpt(question, results[:5])  # 상위 5개 후보 전달
    print("\nGPT가 선택한 파일명:")
    print(gpt_result)

# 실행
if __name__ == "__main__":
    print("사용자 요청: ")
    user_question = "점심 먹었어?"
    print(user_question)
    main(user_question)
