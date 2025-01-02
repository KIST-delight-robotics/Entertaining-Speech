import sqlite3
import numpy as np
import pandas as pd
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate


# OpenAI API 설정
openai_api_key = "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"
llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)

# SBERT 모델 로드
sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Cross-Encoder 모델 로드
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')



# SQLite 데이터베이스 경로
db_path = "./mp3_database.db"

# SQLite에서 데이터 로드
def load_data_from_db():
    conn = sqlite3.connect(db_path)
    query = "SELECT file_name FROM mp3_files"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df['file_name'].tolist()

# SBERT로 문장 임베딩 생성 (GPU에서 CPU로 복사 후 변환)
def get_sbert_embedding(text):
    # SBERT로 텍스트를 임베딩 (GPU 텐서 생성)
    tensor = sbert_model.encode(text, convert_to_tensor=True)
    
    # 텐서를 CPU로 복사 후 NumPy로 변환
    return tensor.cpu().numpy()

# FAISS 인덱스 생성
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index



# 4. 벡터 저장소 구축 (Vector Database)
vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)
# 4-1. 쿼리 저장소 검색을 위한 retriever 생성
retriever = vector_store.as_retriever()

# 데이터 준비
answers = load_data_from_db()
answer_embeddings = np.array([get_sbert_embedding(answer) for answer in answers]).astype('float32')
index = create_faiss_index(answer_embeddings)

# FAISS를 사용한 검색
def search_with_faiss(query, top_k=5):
    query_embedding = get_sbert_embedding(query).reshape(1, -1).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    top_answers = [(answers[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return top_answers


# Cross-Encoder로 상위 후보 재평가
def rerank_with_cross_encoder(query, candidates):
    pairs = [[query, candidate[0]] for candidate in candidates]
    scores = cross_encoder.predict(pairs)  # 질문-대답 쌍에 대해 점수 계산
    ranked_candidates = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [candidate for candidate, _ in ranked_candidates]

# GPT를 사용한 최종 후보 평가
def evaluate_with_gpt(question, candidates):
    candidates_text = "\n".join([f"{i+1}. {candidate[0]} (distance: {candidate[1]:.4f})" for i, candidate in enumerate(candidates)])
    prompt = PromptTemplate(
        input_variables=["question", "candidates"],
        template="""
        질문: '{question}'
        후보 대답 리스트:
        {candidates}
        \n위 질문에 가장 적합한 대답을 선택하고 그 이유를 간단히 설명하세요.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run({"question": question, "candidates": candidates_text})
    return response

# 메인 함수
def main(question):
    # Step 1: FAISS로 상위 후보 검색
    top_candidates = search_with_faiss(question, top_k=30)  # FAISS로 후보 필터링
    
    # Step 2: Cross-Encoder로 재평가
    ranked_candidates = rerank_with_cross_encoder(question, top_candidates)
    
    # Step 3: GPT로 최종 평가
    final_answer = evaluate_with_gpt(question, ranked_candidates[:10])  # 상위 5개만 GPT 평가
    
    print("GPT가 선택한 대답:")
    print(final_answer)
    return final_answer


# 실행
if __name__ == "__main__":
    print("user request:")
    user_question = "콜라 있어?"
    print(user_question)
    result = main(user_question)
