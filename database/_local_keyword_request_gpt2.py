import requests
import openai
import time
import re


# OpenAI API 키 설정
openai.api_key = "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"  

# 텍스트 파일에서 내용을 읽는 함수
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# GPT에게 텍스트 파일 내용을 보내고 응답과 키워드를 동시에 받는 함수
def get_gpt_response_and_extract_keywords(file_path):
    start_time = time.time()  # 시작 시간 기록
    user_input = read_text_file(file_path)
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "A와 B는 친구야. 대화에 장난스럽고, 밝은 말투를 반영해줘. A의 질문에 적합한 B의 대답을 제공하고, 대답 후 두 종류의 키워드를 추출해줘: 1) 문장에서 사용된 주요 명사(L Keywords), 2) 상황 설명에 적합한 5개의 주요 키워드(S Keywords). 명확한 형식으로 응답해줘. 예: L Keywords: 명사1, 명사2, 명사3 (개행) S Keywords: 상황1, 상황2, 상황3."},
            {"role": "user", "content": user_input}
        ]
    )
    gpt_response = response['choices'][0]['message']['content']
    
    # 정규식으로 키워드 추출
    l_keywords_match = re.search(r"L Keywords:\s*(.+)", gpt_response)
    s_keywords_match = re.search(r"S Keywords:\s*(.+)", gpt_response)

    # 키워드 추출 및 기본값 설정
    l_keywords = l_keywords_match.group(1).strip() if l_keywords_match else "No L Keywords Found"
    s_keywords = s_keywords_match.group(1).strip() if s_keywords_match else "No S Keywords Found"

    gpt_time = time.time() - start_time  # 소요 시간 계산
    return gpt_response, l_keywords, s_keywords, gpt_time

# PC의 API 서버로 키워드를 전송하는 함수
def send_keywords_to_server(l_keywords, s_keywords, gpt_response_text):
    start_time = time.time()  # 요청 시작 시간 기록
    server_url = 'http://127.0.0.1:5000/search'

    try:
        # GET 요청 전송
        response = requests.get(server_url, params={
            'l_keywords': l_keywords,
            's_keywords': s_keywords,
            'gpt_response_text': gpt_response_text
        })
        server_time = time.time() - start_time  # JSON 수신 후 시간 계산

        if response.status_code == 200:
            mp3_files = response.json().get('files', [])
            print(f"검색된 mp3 파일: {mp3_files}")
        else:
            print(f"Error: {response.status_code}, {response.text}")
    except Exception as e:
        server_time = time.time() - start_time  # 예외가 발생해도 소요 시간 기록
        print(f"Error during server request: {e}")

    print(f"Server Request Time: {server_time:.4f} seconds")  # 요청 시간 출력
    return server_time

# 메인 실행
if __name__ == '__main__':
    text_file_path = "./requestA.txt"

    # GPT 응답 생성 및 키워드 추출
    gpt_response_text, l_keywords, s_keywords, gpt_time = get_gpt_response_and_extract_keywords(text_file_path)
    print(f"GPT Response: {gpt_response_text}")
    #print(f"L Keywords: {l_keywords}")
    #print(f"S Keywords: {s_keywords}")

    # 키워드로 서버 호출
    server_time = send_keywords_to_server(l_keywords, s_keywords, gpt_response_text)

    # 총 소요 시간 계산
    total_time = gpt_time + server_time

    # 각 단계의 시간 출력
    print("\n--- Timing ---")
    print(f"GPT Response and Extract Time: {gpt_time:.2f} seconds")
    print(f"Server Request Time: {server_time:.2f} seconds")
    print(f"Total Execution Time: {total_time:.2f} seconds")
