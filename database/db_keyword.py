import os
import re
import sqlite3
import openai
from konlpy.tag import Okt

# GPT API Key 설정
openai.api_key = "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"    # 여기에 GPT API 키를 입력하세요.

# 형태소 분석기 객체 생성
okt = Okt()

# 텍스트 파일에서 okt_keywords를 추출하는 함수 (모든 명사 및 동사 어근 추출)
def extract_okt_keywords(text):
    # 형태소 분석 결과 중 명사와 동사의 어간만 추출
    okt_keywords = []
    morphs = okt.pos(text, stem=True)  # 형태소 분석, 동사 어간 추출
    for word, tag in morphs:
        if tag in ['Noun', 'Verb']:  # 명사와 동사(어간만)
            okt_keywords.append(word)

    # 영어 단어 추출 (정규 표현식 사용)
    english_words = re.findall(r'[a-zA-Z]+', text)
    okt_keywords.extend(english_words)
        
    return ', '.join(okt_keywords)  # 쉼표로 구분하여 반환

# GPT에게 텍스트 내용을 보내고 gpt_keywords를 추출하는 함수
def ask_gpt_with_text(user_input):
    try:
        # GPT API 호출
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "상황을 제시하면 그에 맞는 적절한 대답을 하면 돼. A와 B는 친구야. 예를 들면, A : 오늘 눈이 올까? B : '내일 아침에 하얀 눈이 쌓여 있었음 해요.' 이런식으로 B의 대답은 기존의 음악이나 영화, 드라마의 한 문장이야. 예시를 고려해서 유추된 A의 문장 속 상황 키워드(명사형) 5개만 ,로 구분해서 제시해줘."},
                {"role": "assistant", "content": "user가 B의 대답을 제시할거야."},
                {"role": "user", "content": user_input}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"GPT API 호출 중 오류 발생: {e}")
        return None  # API 호출에 실패한 경우

# 텍스트 파일 처리 함수
def process_text_file(file_path):
    if not os.path.exists(file_path):
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return

    # 파일 내용 읽기
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    print("\n--- 텍스트 파일 내용 ---")
    print(content)

    # okt_keywords 추출
    okt_keywords = extract_okt_keywords(content)
    print("\n--- 형태소 분석 키워드 (okt_keywords) ---")
    print(okt_keywords)

    # gpt_keywords 추출
    gpt_keywords = ask_gpt_with_text(content)
    print("\n--- GPT 키워드 (gpt_keywords) ---")
    print(gpt_keywords)

# 실행 경로 설정
text_file_path = "./requestA.txt"

# 텍스트 파일 처리
process_text_file(text_file_path)
