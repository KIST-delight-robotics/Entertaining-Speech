import os
import time
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
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
        if isinstance(index, faiss.IndexIDMap):
            print("FAISS 인덱스 로드 성공.")
            return index
        else:
            # ID 매핑이 없다면 래핑
            print("FAISS 인덱스를 IndexIDMap으로 래핑합니다.")
            return faiss.IndexIDMap(index)
    else:
        raise FileNotFoundError("FAISS 인덱스 파일이 존재하지 않습니다.")

# SQLite에서 파일명 및 ID 로드
def load_metadata_from_db():
    conn = sqlite3.connect(db_path)
    query = "SELECT id, file_name FROM mp3_files"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# SBERT로 문장 임베딩 생성 및 정규화
def get_sbert_embedding(text):
    embedding = sbert_model.encode(text).astype("float32")
    return embedding / np.linalg.norm(embedding)  # 정규화하여 코사인 유사도 기반 검색

# GPT로 최종 평가
def evaluate_with_gpt(question, candidates, top_n=1):
    candidate_list = "\n".join([f"{i+1}. {candidate['file_name']} (cos_sim: {candidate['cosine_similarity']:.4f})" for i, candidate in enumerate(candidates)])
    prompt = f"""질문: '{question}'에 대답으로 대화 행간상 가장 적합한 MP3 파일을 선택하세요. 후보 리스트는 다음과 같습니다:\n{candidate_list}\n\n가장 적합한 파일명을 {top_n}개 선택하고 (적절한 순으로 나열) 이유를 간략히 설명해주세요."""

    messages = [
        {"role": "system", "content": "다음 질문과 파일명 간의 흐름이 자연스럽도록 대화 문맥적 연결을 평가합니다."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
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

    # Step 3: FAISS를 이용한 유사도 검색
    top_k = 100  # 상위 몇 개를 검색할지 설정
    distances, indices = faiss_index.search(np.array([query_embedding]), top_k)
    
    # FAISS가 inner product를 기준으로 정렬하므로, 코사인 유사도와 동일
    candidates = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx == -1:
            continue  # 유효하지 않은 인덱스
        file_name = metadata.loc[metadata['id'] == idx, 'file_name'].values
        if len(file_name) > 0:
            candidates.append({"file_name": file_name[0], "cosine_similarity": distance})
    
    # print("\nSBERT 코사인 유사도로 정렬된 상위 후보:")
    # for result in candidates:
    #     print(f"파일명: {result['file_name']}, 코사인 유사도: {result['cosine_similarity']:.4f}")

    # Step 4: GPT로 최종 후보 평가
    gpt_result = evaluate_with_gpt(question, candidates, top_n=1)  # 상위 3개 후보 전달
    print("\nGPT가 선택한 파일명:")
    print(gpt_result)

# 실행
if __name__ == "__main__":
    start_time = time.time()  # 시작 시간 기록
    user_question = "어디서 왔어?"
    print(f"사용자 질문: {user_question}")
    main(user_question)
    total_time = time.time() - start_time  # 소요 시간 계산
    print(f"소요 시간 : {total_time}")
    
