import os
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import openai
import pandas as pd
from numpy.linalg import norm
import torch
from functools import lru_cache
import asyncio
import time  # 시간 측정을 위한 모듈 추가

# OpenAI API 설정
openai.api_key = "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"  

# SBERT 모델 초기화 (GPU 사용 가능 시)
device = "cuda" if torch.cuda.is_available() else "cpu"
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

# SQLite 및 FAISS 설정
db_path = "./mp3_database.db"
faiss_index_file = "faiss_index.bin"

# FAISS 인덱스 로드
def load_faiss_index():
    start_time = time.time()
    if os.path.exists(faiss_index_file):
        index = faiss.read_index(faiss_index_file)
        if isinstance(index, faiss.IndexIDMap):
            print("FAISS 인덱스 로드 성공.")
            faiss_index = index
        else:
            print("FAISS 인덱스를 IndexIDMap으로 래핑합니다.")
            faiss_index = faiss.IndexIDMap(index)
    else:
        # IVF 인덱스 생성 (예: 100개의 클러스터 사용)
        quantizer = faiss.IndexFlatIP(384)
        index = faiss.IndexIVFFlat(quantizer, 384, 100, faiss.METRIC_INNER_PRODUCT)
        faiss_index = faiss.IndexIDMap(index)
        # 실제 데이터로 트레이닝 필요
        dummy_data = np.random.random((1000, 384)).astype('float32')
        faiss_index.train(dummy_data)
        print("IVF FAISS 인덱스 생성.")
    end_time = time.time()
    print(f"FAISS 인덱스 로드 시간: {end_time - start_time:.4f}초")
    return faiss_index

# SQLite에서 파일명 및 ID 로드 (딕셔너리 사용)
def load_metadata_from_db():
    start_time = time.time()
    conn = sqlite3.connect(db_path)
    query = "SELECT id, file_name FROM mp3_files"
    cursor = conn.cursor()
    cursor.execute(query)
    metadata = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    end_time = time.time()
    print(f"메타데이터 로드 시간: {end_time - start_time:.4f}초")
    return metadata

# SBERT로 문장 임베딩 생성 및 정규화 (캐싱)
@lru_cache(maxsize=1000)
def get_sbert_embedding(text):
    start_time = time.time()
    embedding = sbert_model.encode(text).astype("float32")
    normalized_embedding = embedding / norm(embedding)  # 정규화하여 코사인 유사도 기반 검색
    end_time = time.time()
    print(f"SBERT 임베딩 생성 시간: {end_time - start_time:.4f}초")
    return normalized_embedding

# GPT로 최종 평가 (비동기)
async def evaluate_with_gpt_async(question, candidates, top_n=1):
    start_time = time.time()
    candidate_list = "\n".join([f"{i+1}. {candidate['file_name']} (cos_sim: {candidate['cosine_similarity']:.4f})" for i, candidate in enumerate(candidates)])
    prompt = f"""질문: '{question}'에 대답으로 가장 적합한 MP3 파일을 선택하세요. 후보 리스트는 다음과 같습니다:\n{candidate_list}\n\n가장 적합한 파일명을 {top_n}개 선택하고 (적절한 순으로 나열) 이유를 간략히 설명해주세요. """
    
    messages = [
        {"role": "system", "content": "코사인유사도를 절대적인 기준으로 사용하지 않고, 다음 질문과 파일명 간의 대화상의 문맥적 연결을 평가합니다."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o",
            messages=messages
        )
        gpt_response = response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"OpenAI API 호출 중 오류 발생: {e}")
        gpt_response = "GPT 평가 실패"
    end_time = time.time()
    print(f"GPT 평가 시간: {end_time - start_time:.4f}초")
    return gpt_response

# 메인 함수
async def main(question):
    # Step 2: 사용자 질문을 SBERT 임베딩으로 변환
    query_embedding = get_sbert_embedding(question).reshape(1, -1)
    
    # Step 3: FAISS를 이용한 유사도 검색
    search_start_time = time.time()
    top_k = 100  # 상위 몇 개를 검색할지 설정
    distances, indices = faiss_index.search(query_embedding, top_k)
    search_end_time = time.time()
    print(f"FAISS 검색 시간: {search_end_time - search_start_time:.4f}초")
    
    # 후보 리스트 생성
    candidates = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx == -1:
            continue  # 유효하지 않은 인덱스
        file_name = metadata.get(idx, "Unknown")
        candidates.append({"file_name": file_name, "cosine_similarity": distance})
    
    # print("\nSBERT 코사인 유사도로 정렬된 상위 후보:")
    # for result in candidates:
    #     print(f"파일명: {result['file_name']}, 코사인 유사도: {result['cosine_similarity']:.4f}")
    
    # Step 4: GPT로 최종 후보 평가 (비동기)
    gpt_result = await evaluate_with_gpt_async(question, candidates, top_n=1)  # 상위 3개 후보 전달
    print("\nGPT가 선택한 파일명:")
    print(gpt_result)

# 전역 FAISS 인덱스 및 메타데이터 로드
faiss_index = load_faiss_index()
metadata = load_metadata_from_db()

# 실행
if __name__ == "__main__":
    user_question = "어디서 왔어?"
    print(f"사용자 질문: {user_question}")
    asyncio.run(main(user_question))
