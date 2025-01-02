import asyncio
import re
import requests
import av
from aiortc import RTCPeerConnection, RTCSessionDescription, AudioStreamTrack
from konlpy.tag import Okt
import openai
import json

# OpenAI API 키 설정
openai.api_key = "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"  

# 텍스트 파일에서 내용을 읽는 함수
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# GPT에게 텍스트 파일 내용을 보내고 응답을 받는 함수
def ask_gpt_with_text_file(file_path):
    user_input = read_text_file(file_path)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "상황을 제시하면 그에 맞는 적절한 대답을 하면 돼..."},
            {"role": "assistant", "content": "user가 A의 질문을 제시할거야."},
            {"role": "user", "content": user_input}
        ]
    )
    return response['choices'][0]['message']['content']

# 형태소 분석기 객체 생성
okt = Okt()

# GPT 응답에서 키워드를 추출하는 함수
def extract_keywords(response):
    keywords = []
    morphs = okt.pos(response, stem=True)
    for word, tag in morphs:
        if tag in ['Noun', 'Verb']:
            keywords.append(word)
    english_words = re.findall(r'[a-zA-Z]+', response)
    keywords.extend(english_words)
    return ', '.join(keywords)

# GPT에서 상황 키워드를 5개만 추출하는 함수
def extract_gpt_keywords(response):
    gpt_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "문장에서 상황을 나타내는 주요 명사형 키워드 5개만 콤마로 구분해서 추출해줘."},
            {"role": "user", "content": response}
        ]
    )
    return gpt_response['choices'][0]['message']['content']

# PC의 API 서버로 키워드를 전송하여 mp3 파일 검색 요청
def search_mp3_files(okt_keywords, gpt_keywords):
    PC_IP = 'http://127.0.0.1:5000/search'
    response = requests.get(PC_IP, params={'keyword': okt_keywords, 'gpt_keyword': gpt_keywords})
    
    if response.status_code == 200:
        mp3_files = response.json().get('files', [])
        if mp3_files:
            # 가장 일치도가 높은 mp3 파일을 선택
            selected_file = mp3_files[0]
            print(f"선택된 mp3 파일: {selected_file}")
            return selected_file['file_path']
        else:
            print("해당 키워드로 mp3 파일을 찾을 수 없습니다.")
    else:
        print(f"Error: {response.status_code}, {response.text}")
    return None

class AudioFileTrack(AudioStreamTrack):
    def __init__(self, file_path):
        super().__init__()  # AudioStreamTrack 초기화
        self.container = av.open(file_path)
        self.audio_stream = self.container.streams.audio[0]
        self.frame_generator = self.container.decode(self.audio_stream)

    async def recv(self):
        frame = next(self.frame_generator, None)
        if frame is None:
            raise av.error.EOFError("End of audio file reached")
        return frame.to_audio()

async def start_webrtc_streaming(selected_file):
    pc = RTCPeerConnection()  # configuration 인자 제거
    audio_track = AudioFileTrack(selected_file)
    pc.addTrack(audio_track)

    # WebRTC Offer 생성 및 설정
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    # PC의 Signaling 서버에 Offer 전송 및 Answer 수신
    response = requests.post("http://127.0.0.1:5000/offer", json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
    
    # 서버 응답 내용 확인 및 예외 처리 추가
    try:
        print("서버 응답:", response.text)  # 서버에서 어떤 응답이 오는지 확인
        answer = response.json()
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
        print("WebRTC 연결이 성공적으로 설정되었습니다.")
    except json.JSONDecodeError:
        print("JSONDecodeError: 서버에서 유효한 JSON 응답을 받지 못했습니다.")
        print("서버 응답 내용:", response.text)

# 통합 실행 함수
def main():
    text_file_path = "./requestA.txt"
    gpt_response = ask_gpt_with_text_file(text_file_path)
    print(f"GPT Response: {gpt_response}")
    okt_keywords = extract_keywords(gpt_response)
    gpt_keywords = extract_gpt_keywords(gpt_response)
    print(f"OKT Keywords: {okt_keywords}")
    print(f"GPT Keywords: {gpt_keywords}")

    selected_file = search_mp3_files(okt_keywords, gpt_keywords)
    if selected_file:
        asyncio.run(start_webrtc_streaming(selected_file))
    else:
        print("WebRTC 연결을 설정할 mp3 파일을 찾을 수 없습니다.")

if __name__ == "__main__":
    main()