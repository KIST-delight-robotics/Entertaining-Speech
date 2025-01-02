# gpt 이용 GPT response(B)에서 형태소 키워드 추출

import requests
import openai
import re
import time

# OpenAI API 키 설정
openai.api_key = "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"  

# 텍스트 파일에서 내용을 읽는 함수
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# GPT에게 텍스트 파일 내용을 보내고 응답을 받는 함수
def get_gpt_response(file_path):
    start_time = time.time()  # 시작 시간 기록
    user_input = read_text_file(file_path)
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "상황을 제시하면 그에 맞는 적절한 대답을 하면 돼. A와 B는 친구야. 예를 들면, A : 오늘 눈이 올까? B : '눈이 오면 좋겠다.' 이런식으로 B의 대답은 A의 질문에 적절하게 이루어질거야. 단, B의 대답에서 명사형 키워드를 추출하기 적합하게 대답해줘. 반문은 하지 않아도 되고, 질문 키워드가 들어가게 대답해줘. 친한 두 사람이 대화하는 느낌이 들게 대답해줘."},
            {"role": "assistant", "content": "user가 A의 질문을 제시할거야."},
            {"role": "user", "content": user_input}
        ]
    )
    gpt_response = response['choices'][0]['message']['content']
    gpt_time = time.time() - start_time  # 소요 시간 계산
    return gpt_response, gpt_time

# GPT 응답에서 키워드를 추출하는 함수 (모든 명사 및 동사 어근 추출)
def extract_l_keywords(response_text):
    start_time = time.time()  # 시작 시간 기록
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "문장에서 유의미한 부분을 명사형으로 추출해줘."},
            {"role": "user", "content": response_text}
        ]
    )
    l_keywords = response['choices'][0]['message']['content']
    l_keywords_time = time.time() - start_time  # 소요 시간 계산
    return l_keywords, l_keywords_time


# GPT에서 상황 키워드를 5개만 추출하는 함수
def extract_s_keywords(response_text):
    start_time = time.time()  # 시작 시간 기록
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "문장에서 상황을 설명하는 주요 명사형 키워드 5개를 콤마로 구분해서 추출해줘."},
            {"role": "user", "content": response_text}
        ]
    )
    s_keywords = response['choices'][0]['message']['content']
    s_keywords_time = time.time() - start_time  # 소요 시간 계산
    return s_keywords, s_keywords_time

# PC의 API 서버로 키워드를 전송하는 함수
def send_keywords_to_server(okt_keywords, main_keywords, gpt_response_text):
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

    # GPT 응답 생성
    gpt_response_text, gpt_time = get_gpt_response(text_file_path)
    print(f"GPT Response: {gpt_response_text}")

    # 형태소 분석 키워드 추출
    l_keywords, l_keywords_time = extract_l_keywords(gpt_response_text)
    print(f"l Keywords: {l_keywords}")

    # GPT 주요 키워드 추출
    s_keywords, s_keywords_time = extract_s_keywords(gpt_response_text)
    print(f"s Keywords: {s_keywords}")

    # 키워드로 서버 호출
    server_time = send_keywords_to_server(l_keywords, s_keywords, gpt_response_text)

    # 총 소요 시간 계산
    total_time = gpt_time + l_keywords_time + s_keywords_time + server_time

    # 각 단계의 시간 출력
    print("\n--- Timing ---")
    print(f"GPT Response Time: {gpt_time:.2f} seconds")
    print(f"L Keyword Extraction Time: {l_keywords_time:.2f} seconds")
    print(f"S Keyword Extraction Time: {s_keywords_time:.2f} seconds")
    print(f"Server Request Time: {server_time:.2f} seconds")
    print(f"Total Execution Time: {total_time:.2f} seconds")
