import sqlite3
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import SQLiteChatMessageHistory
from langchain.chains import ConversationChain
import pandas as pd

VECTOR_DB_PATH = "./faiss_index"
DB_PATH = "./mp3_database.db"
DB_MEMORY_PATH = "./chat_memory.db"

# SQLite 데이터 로드
def load_data_from_db():
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT file_name, id FROM mp3_files"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# FAISS Retriever 초기화
def initialize_retriever(vector_db_path):
    embeddings = OllamaEmbeddings(model="llama3.1")
    vector_store = FAISS.load_local(vector_db_path, embeddings)
    retriever = vector_store.as_retriever()
    return retriever

# SQLite Memory 초기화
def initialize_memory():
    history = SQLiteChatMessageHistory(database_path=DB_MEMORY_PATH)
    memory = ConversationBufferMemory(chat_memory=history, return_messages=True)
    return memory

# LangChain 체인 생성
def create_chain(memory):
    prompt_template = PromptTemplate.from_template(
        """당신은 SQLite 데이터베이스에서 mp3 파일명을 검색하고 사용자 질문에 답변하는 AI 어시스턴트입니다.
        ---
        # Question: 
        {question}
        # Context (Retrieved from Memory or FAISS): 
        {context}
        # Answer:
        """
    )
    llm = Ollama(model="gpt-4o", temperature=0)
    chain = ConversationChain(llm=llm, memory=memory, prompt=prompt_template)
    return chain

# 질문-답변 처리
def answer_query_with_memory(chain, retriever, memory, question):
    # FAISS에서 검색
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # 메모리에 저장
    memory.save_context({"user_input": question}, {"context": context})

    # 질문과 문맥을 결합하여 LangChain 실행
    response = chain.run(question=question, context=context)

    # 결과 반환
    return response

# 메인 실행 함수
def main():
    # Step 1: SQLite Memory 초기화
    memory = initialize_memory()

    # Step 2: FAISS Retriever 초기화
    retriever = initialize_retriever(VECTOR_DB_PATH)

    # Step 3: LangChain Chain 생성
    chain = create_chain(memory)

    # Step 4: 사용자 질문 처리
    user_question = input("질문을 입력하세요: ")
    result = answer_query_with_memory(chain, retriever, memory, user_question)

    print("\nGPT-4 응답:")
    print(result)

# 실행
if __name__ == "__main__":
    main()
