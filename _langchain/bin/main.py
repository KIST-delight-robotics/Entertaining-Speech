import os
import sqlite3
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
#from langchain.memory import ChatMessageHistory, SQLiteChatMessageHistory
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

DB_PATH = "./mp3_database.db"
VECTOR_DB_PATH = "./faiss_index"
FOLDER_PATH = "./mp3_files"

# Embedding 및 GPT 모델 초기화
embeddings = OllamaEmbeddings(model="llama3.1")
llm = Ollama(model="llama3.1", base_url="http://localhost:11434")

# # SQLite 데이터베이스 설정
# def setup_database():
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
#     cursor.execute('''
#     CREATE TABLE IF NOT EXISTS mp3_files (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         file_name TEXT,
#         file_path TEXT
#     )
#     ''')
#     conn.commit()
#     conn.close()

# # SQLite에 mp3 파일 추가
# def populate_sqlite(folder_path):
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()

#     for file in os.listdir(folder_path):
#         if file.endswith('.mp3'):
#             file_name = os.path.basename(file)
#             file_path = os.path.join(folder_path, file)
#             cursor.execute("INSERT OR IGNORE INTO mp3_files (file_name, file_path) VALUES (?, ?)", (file_name, file_path))
#             print(f"Added: {file_name}")
    
#     conn.commit()
#     conn.close()

# # FAISS 벡터 인덱스 생성
# def build_faiss_index():
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
#     cursor.execute("SELECT file_name FROM mp3_files")
#     rows = cursor.fetchall()
#     conn.close()

#     if not rows:
#         print("No data found in SQLite to build FAISS index.")
#         return None

#     file_names = [row[0] for row in rows]
#     faiss_index = FAISS.from_texts(file_names, embeddings)
#     faiss_index.save_local(VECTOR_DB_PATH)
#     print("FAISS index has been saved.")
#     return faiss_index

# FAISS 로드
def load_faiss_index():
    return FAISS.load_local(VECTOR_DB_PATH, embeddings)


# 대화 히스토리 관리
#def get_chat_history():
#    return SQLiteChatMessageHistory(session_id="mp3_search_session", database_path="./chat_history.db")

# 사용자 질문을 기반으로 GPT가 추천하는 체인
def gpt_chain(llm):
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""당신은 mp3 파일을 추천하는 AI 어시스턴트입니다.
사용자의 질문에 기반하여 아래의 검색된 결과(context) 중에서 가장 적합한 파일을 추천하세요.
만약 관련 정보가 없으면 '적절한 파일을 찾을 수 없습니다'라고 답변하세요.

질문: {question}
검색된 결과:
{context}

추천 파일:"""
    )
    return LLMChain(llm=llm, prompt=prompt)

def search_and_recommend(question, faiss_index, llm):
    # FAISS를 통해 상위 후보 검색
    results = faiss_index.similarity_search(question, k=5)
    context = "\n".join([result.page_content for result in results])

    # GPT 체인을 통해 최적의 결과 추천
    chain = gpt_chain(llm)
    response = chain.run(question=question, context=context)
    return response


if __name__ == "__main__":
    # 1. 데이터베이스 설정
    #setup_database()
    #populate_sqlite(FOLDER_PATH)

    # 2. FAISS 인덱스 생성 또는 로드
    if not os.path.exists(VECTOR_DB_PATH):
        print("FAISS 인덱스를 생성 중...")
        faiss_index = build_faiss_index()
    else:
        print("FAISS 인덱스를 불러오는 중...")
        faiss_index = load_faiss_index()

    # 3. 대화 기록 및 사용자 질문 처리
    chat_history = get_chat_history()
    print("질문을 입력하세요 (종료하려면 'exit'):")

    while True:
        user_input = input(">> ")
        if user_input.lower() == "exit":
            print("프로그램을 종료합니다.")
            break

        # 질문 처리 및 파일 추천
        recommendation = search_and_recommend(user_input, faiss_index, llm)
        print(f"추천 결과: {recommendation}")
        
        # 대화 기록 업데이트
        chat_history.add_user_message(user_input)
        chat_history.add_ai_message(recommendation)
