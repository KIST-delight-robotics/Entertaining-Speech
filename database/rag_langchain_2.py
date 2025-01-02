import sqlite3
from concurrent.futures import ThreadPoolExecutor

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from sentence_transformers import SentenceTransformer, util
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

# SBERT 모델 로드
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# 문서 임베딩
corpus_embeddings = sbert_model.encode(corpus, convert_to_tensor=True)

# 쿼리 임베딩
query_embedding = sbert_model.encode(query, convert_to_tensor=True)

# SQLite 데이터베이스 연결
conn = sqlite3.connect("mp3_database.sqlite")
cursor = conn.cursor()

# mp3 파일 메타데이터 가져오기
cursor.execute("SELECT id, title FROM mp3_files")
rows = cursor.fetchall()

# 메타데이터를 결합하여 임베딩 생성
embeddings = OllamaEmbeddings(model="llama3.1")
documents = [{"id": row[0], "content": f"{row[1]} {row[2]}"} for row in rows]
faiss_index = FAISS.from_documents(
    documents=[doc["content"] for doc in documents],
    embedding=embeddings,
)

# 로컬에 벡터 데이터베이스 저장
faiss_index.save_local("faiss_index")

# 멀티스레드로 메타데이터 임베딩 생성
def embed_document(doc):
    return embeddings.embed_query(doc["content"])

with ThreadPoolExecutor(max_workers=8) as executor:
    embeddings_list = list(executor.map(embed_document, documents))

# FAISS에 병렬 처리된 임베딩 추가
faiss_index = FAISS()
for doc, embedding in zip(documents, embeddings_list):
    faiss_index.add_texts([doc["content"]], embeddings=[embedding])

# SQLite에 임베딩 저장
for doc, embedding in zip(documents, embeddings_list):
    cursor.execute(
        "UPDATE mp3_files SET embedding = ? WHERE id = ?",
        (embedding.tobytes(), doc["id"]),
    )
conn.commit()

# 임베딩 로드 시
cursor.execute("SELECT id, embedding FROM mp3_files WHERE embedding IS NOT NULL")
rows = cursor.fetchall()
for row in rows:
    faiss_index.add_texts([row[0]], embeddings=[np.frombuffer(row[1])])


def retrieve_with_ondemand_embedding(query):
    # 질문에 맞는 메타데이터를 SQLite에서 검색
    cursor.execute(
        "SELECT id, title, artist FROM mp3_files WHERE title LIKE ? OR artist LIKE ?",
        (f"%{query}%", f"%{query}%"),
    )
    rows = cursor.fetchall()

    # 검색된 메타데이터만 임베딩
    documents = [{"id": row[0], "content": f"{row[1]} {row[2]}"} for row in rows]
    embeddings_list = [embeddings.embed_query(doc["content"]) for doc in documents]

    # 임베딩으로 FAISS 검색
    faiss_index = FAISS()
    for doc, embedding in zip(documents, embeddings_list):
        faiss_index.add_texts([doc["content"]], embeddings=[embedding])
    
    return faiss_index.similarity_search(query)

# SQLite에서 임베딩 데이터 가져오기
cursor.execute("SELECT id, embedding FROM mp3_files WHERE embedding IS NOT NULL")
rows = cursor.fetchall()

# FAISS에 추가
faiss_index = FAISS()
for row in rows:
    embedding = np.frombuffer(row[1])
    faiss_index.add_texts([row[0]], embeddings=[embedding])



# 상위 후보군 검색
top_k = 10
hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
candidates = [corpus[hit['corpus_id']] for hit in hits[0]]


# Ollama 모델 초기화
ollama_embeddings = OllamaEmbeddings(model="llama3.1")

# 후보군 임베딩 생성
candidate_embeddings = [ollama_embeddings.embed_query(candidate) for candidate in candidates]

# 쿼리 임베딩 생성
query_embedding_ollama = ollama_embeddings.embed_query(query)

# 유사도 계산 (코사인 유사도)
import numpy as np
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Ollama 기반 정밀 유사도 계산
ollama_scores = [cosine_similarity(query_embedding_ollama, candidate_emb) for candidate_emb in candidate_embeddings]

# 유사도 기준 상위 후보군 정렬
sorted_candidates = [x for _, x in sorted(zip(ollama_scores, candidates), reverse=True)]


# GPT-4 프롬프트 생성
context = "\n\n".join(sorted_candidates[:3])  # 상위 3개 후보군 사용
prompt = f"""
당신은 질문-답변(Question-Answering)을 수행하는 AI 어시스턴트입니다.
질문: {query}
문맥: {context}
답변:
"""

# GPT-4 모델 초기화
gpt4_model = Ollama(model="gpt-4.0")

# GPT-4를 통한 답변 생성
response = gpt4_model.generate(prompt)
print("최종 답변:", response)