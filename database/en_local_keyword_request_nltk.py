# GPT response(B)에서 형태소 키워드 추출
import os
import requests
import openai
import re
import time
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# OpenAI API 키 설정
openai.api_key = "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"  

nltk.data.path.append('/home/delight/nltk_data')

nltk.download('punkt', download_dir='/home/delight/nltk_data')
nltk.download('averaged_perceptron_tagger', download_dir='/home/delight/nltk_data')


def translate_korean_to_english(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"])
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 형태소 키워드 추출
def extract_l_keywords(text):
    start_time = time.time()  # 시작 시간 기록
    try:
        valid_pos = ['NN', 'NNS', 'NNP', 'NNPS',  # 명사
                     'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # 동사
                     'JJ', 'JJR', 'JJS',  # 형용사
                     'RB', 'RBR', 'RBS']  # 부사

        print(f"Extracting keywords from text: {text}")
        words = word_tokenize(text)  # 단어 토큰화
        tagged = pos_tag(words)  # 품사 태깅
        print(f"POS tagging: {tagged}")  # 디버깅 메시지

        # valid_pos 리스트에 포함된 품사만 필터링
        l_keywords = [word for word, tag in tagged if tag in valid_pos]
        print(f"Extracted keywords: {l_keywords}")
        l_keyword_time = time.time() - start_time  # 소요 시간 계산
        return ', '.join(l_keywords), l_keyword_time

    except Exception as e:
        print(f"Error during keyword extraction: {e}")
        return ""

# GPT 상황 키워드 추출
def extract_s_keywords(response_text):
    start_time = time.time()  # 시작 시간 기록
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Extract 5 main noun-form English keywords that infer the situation, separated by commas."},
                {"role": "user", "content": response_text}
            ]
        )
        s_keywords = response['choices'][0]['message']['content']
        s_keywords_time = time.time() - start_time  # 소요 시간 계산
        return s_keywords, s_keywords_time

    except Exception as e:
        print(f"GPT API 호출 중 오류 발생: {e}")
        return ""  # 기본값 반환

def get_gpt_response(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return "", 0

    start_time = time.time()
    with open(file_path, 'r', encoding='utf-8') as file:
        user_input = file.read()

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Provide a response appropriate to the given situation. A and B are friends. For example: A: 'Will it snow today?' B: 'I hope it snows.' In this case, B's response should be relevant to A's question. Additionally, B's response should be in English and phrased in a way that makes it suitable for extracting noun keywords. B does not need to ask questions back, and the response should include keywords related to the question."},
                {"role": "user", "content": user_input}
            ]
        )
        gpt_response = response['choices'][0]['message']['content']
    except Exception as e:
        print(f"GPT API 호출 중 오류 발생: {e}")
        return "", time.time() - start_time

    return gpt_response, time.time() - start_time

def send_keywords_to_server(l_keywords, s_keywords):
    start_time = time.time()
    server_url = 'http://127.0.0.1:5000/search'

    try:
        response = requests.get(server_url, params={
            'l_keywords': l_keywords,
            's_keywords': s_keywords
        })
        server_time = time.time() - start_time

        if response.status_code == 200:
            mp3_files = response.json().get('files', [])
            print(f"검색된 mp3 파일: {mp3_files}")
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return 0
    except Exception as e:
        server_time = time.time() - start_time
        print(f"Error during server request: {e}")
        return 0

    print(f"Server Request Time: {server_time:.4f} seconds")
    return server_time

def main():
    text_file_path = "./requestA.txt"

    # GPT 응답 생성
    gpt_response_text, gpt_time = get_gpt_response(text_file_path)
    if not gpt_response_text:
        print("Error: Failed to generate GPT response.")
        return

    print(f"GPT Response: {gpt_response_text}")

    # 형태소 키워드 추출
    l_keywords, l_keyword_time = extract_l_keywords(gpt_response_text)
    print(f"l Keywords (nltk 이용): {l_keywords}")

    # GPT 주요 키워드 추출
    s_keywords, s_keywords_time = extract_s_keywords(gpt_response_text)
    print(f"s Keywords: {s_keywords}")

    # 키워드로 서버 호출
    server_time = send_keywords_to_server(l_keywords, s_keywords)

    # 총 소요 시간 계산
    total_time = gpt_time + server_time
    print("\n--- Timing ---")
    print(f"GPT Response Time: {gpt_time:.2f} seconds")
    print(f"L Keyword Extraction Time: {l_keyword_time:.2f} seconds")
    print(f"S Keyword Extraction Time: {s_keywords_time:.2f} seconds")
    print(f"Server Request Time: {server_time:.2f} seconds")
    print(f"Total Execution Time: {total_time:.2f} seconds")

if __name__ == '__main__':
    main()