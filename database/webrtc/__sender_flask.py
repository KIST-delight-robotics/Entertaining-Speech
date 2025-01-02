import socketio
import asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription
import openai
import re
from konlpy.tag import Okt

# Signaling 서버와 WebRTC 피어 연결 생성
sio = socketio.AsyncClient()
pc = RTCPeerConnection()
okt = Okt()

# Signaling 서버와 WebRTC 피어 연결 생성
sio = socketio.AsyncClient()
pc = RTCPeerConnection()
okt = Okt()

# OpenAI API 키 설정
OPENAI_API_KEY = "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"  
openai.api_key = OPENAI_API_KEY

# 텍스트 파일에서 내용 읽기
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# GPT에게 텍스트 파일 내용을 보내고 응답 받기
async def ask_gpt_with_text_file(file_path):
    user_input = read_text_file(file_path)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "상황을 제시하면 그에 맞는 적절한 대답을 하면 돼. A와 B는 친구야. 예를 들면, A : 오늘 눈이 올까? B : '눈이 오면 좋겠다.' 이런식으로 B의 대답은 A의 질문에 적절하게 이루어질거야. 예시를 고려해서 제시된 A의 문장 속 B의 대답을 제시해줘. 단, B의 대답에서 명사형 키워드를 추출하기 적합하게 대답해줘. 반문은 하지 않아도 되고, 질문 키워드가 들어가게 대답해줘."},
            {"role": "assistant", "content": "user가 A의 질문을 제시할거야."},
            {"role": "user", "content": user_input}
        ]
    )
    return response['choices'][0]['message']['content']

# GPT 응답에서 OKT 키워드 추출
def extract_okt_keywords(response):
    keywords = []
    morphs = okt.pos(response, stem=True)
    for word, tag in morphs:
        if tag in ['Noun', 'Verb']:
            keywords.append(word)
    return ', '.join(keywords)

# GPT 응답에서 상황 키워드 추출
def extract_gpt_keywords(response):
    gpt_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "문장에서 상황을 나타내는 주요 명사형 키워드 5개만 콤마로 구분해서 추출해줘."},
            {"role": "user", "content": response}
        ]
    )
    return gpt_response['choices'][0]['message']['content']

# Signaling 서버 연결
@sio.event
async def connect():
    print("Connected to signaling server")
    await sio.emit('join_room', {'room': 'my_room'})  # 특정 방에 참여

# 데이터 채널 추가
data_channel = pc.createDataChannel("chat")

# Offer 전송
async def send_offer():
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    await sio.emit('offer', {'room': 'my_room', 'offer': offer.sdp})

# 주기적으로 OKT 및 GPT 키워드를 전송
async def send_keywords(okt_keywords, gpt_keywords):
    await sio.emit('keywords', {'room': 'my_room', 'okt_keywords': okt_keywords, 'gpt_keywords': gpt_keywords})

# 메인 로직
async def main():
    await sio.connect('http://localhost:3000')
    await send_offer()  # 최초 Offer 전송

    # 텍스트 파일로부터 GPT 응답 및 키워드 추출
    text_file_path = "./requestA.txt"
    gpt_response = await ask_gpt_with_text_file(text_file_path)
    okt_keywords = extract_okt_keywords(gpt_response)
    gpt_keywords = extract_gpt_keywords(gpt_response)
    
    print(f"GPT Response: {gpt_response}")
    print(f"OKT Keywords: {okt_keywords}")
    print(f"GPT Keywords: {gpt_keywords}")

    # 주기적으로 두 키워드를 전송
    while True:
        await send_keywords(okt_keywords, gpt_keywords)
        await asyncio.sleep(5)

    # 세션 종료
    await sio.disconnect()
    await pc.close()

asyncio.run(main())
