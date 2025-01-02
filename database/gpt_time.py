import time
import openai

# OpenAI API 키 설정
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY

# GPT에게 텍스트 파일 내용을 보내고 응답을 받는 함수 (응답 시간 측정 포함)
def get_gpt_response_with_timing(file_path):
    # 텍스트 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as file:
        user_input = file.read()

    # API 호출 전 시간 기록
    start_time = time.time()

    # OpenAI GPT 호출
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "상황을 제시하면 그에 맞는 적절하고 자연스러운 대답을 하면 돼. A와 B는 친구야. 예를 들면, A : 오늘 눈이 올까? B : '눈이 오면 좋겠다.' 이런식으로 B의 대답은 A의 질문에 적절하게 이루어질거야. 단, B의 대답에서 명사형 키워드를 추출하기 적합하게 대답해줘. 반문은 하지 않아도 되고, 질문 키워드가 들어가게 대답해줘."},
            {"role": "assistant", "content": "user가 A의 질문을 제시할거야."},
            {"role": "user", "content": user_input}
        ]
    )

    # API 호출 후 시간 기록
    end_time = time.time()

    # 응답 시간 계산
    response_time = end_time - start_time

    return response['choices'][0]['message']['content'], response_time

# 테스트할 텍스트 파일 경로
text_file_path = "./requestA.txt"

# GPT 응답 및 시간 측정
gpt_response_text, gpt_response_time = get_gpt_response_with_timing(text_file_path)

# 결과 출력
print(f"GPT Response: {gpt_response_text}")
print(f"API Response Time: {gpt_response_time:.2f} seconds")
