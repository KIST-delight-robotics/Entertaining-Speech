

from __future__ import annotations

# ────────────────────────────────────────────────────────────────
# Std / third‑party imports
# ────────────────────────────────────────────────────────────────
from typing import Optional
import os, threading, time, queue, random, asyncio, wave
from datetime import datetime
import numpy as np
import torch
import pyaudio
import webrtcvad
import soundfile as sf
import tempfile
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from google.cloud import speech
from dotenv import load_dotenv
import pygame
import rclpy
from rclpy.node import Node
import simpleaudio as sa
from pydub import AudioSegment
from pydub.playback import play
import json
from std_msgs.msg import String, Float32
import librosa
import librosa.display
from scipy import ndimage 
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from scipy import signal
import matplotlib.pyplot as plt
import webbrowser
from openpyxl import Workbook
from datetime import datetime
import csv

# 기존 imports에 추가
import requests  # TTS API 호출용





load_dotenv("/home/nvidia/ros2_ws/src/.env")



# ──────────────────────────────────────────────────────────────────────────────
# 실시간 단어 전송
# ──────────────────────────────────────────────────────────────────────────────
class RealtimeWordProcessor:
    def __init__(self):
        self.master_transcript = ""  # 전체 문장 누적
        self.sent_phrases = []  # 이미 전송된 구문들
        self.last_sent_word_count = 0  # 마지막으로 전송한 단어 개수
        self.debounce_time = 0.05  # 300ms 디바운싱
        self.last_process_time = 0
        
    def process_transcript(self, transcript, is_final=False):
        """새로운 방식: 전체 문장 기준으로 증분 처리"""
        current_time = time.time()
        
        # 디바운싱
        if current_time - self.last_process_time < self.debounce_time and not is_final:
            return []
        
        self.last_process_time = current_time
        
        # 전체 문장 업데이트 (더 긴 문장으로 대체)
        if len(transcript) > len(self.master_transcript):
            self.master_transcript = transcript
        
        # 단어 분할
        words = self.master_transcript.strip().split()
        new_phrases = []
        
        # 새로운 단어들만 처리
        if len(words) > self.last_sent_word_count:
            start_idx = self.last_sent_word_count
            new_words = words[start_idx:]
            
            # 두 단어씩 묶어서 구문 생성
            for i in range(0, len(new_words), 2):
                if i + 1 < len(new_words):
                    phrase = f"{new_words[i]} {new_words[i+1]}"
                else:
                    phrase = new_words[i]  # 마지막 단어
                
                new_phrases.append(phrase)
                self.last_sent_word_count += 2 if i + 1 < len(new_words) else 1
        
        # Final 결과일 때 남은 단어 처리
        if is_final and len(words) > self.last_sent_word_count:
            remaining_words = words[self.last_sent_word_count:]
            if remaining_words:
                new_phrases.append(' '.join(remaining_words))
        
        return new_phrases
    
    def reset(self):
        """새 문장 시작시 초기화"""
        self.master_transcript = ""
        self.sent_phrases = []
        self.last_sent_word_count = 0









# ──────────────────────────────────────────────────────────────────────────────
# UserQuestion 노드
# ──────────────────────────────────────────────────────────────────────────────

class UserQuestion(Node):
    def __init__(self):
        super().__init__("UserQuestion")
        self.get_logger().info("UserQuestion Node started")

        # Google Cloud STT
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/nvidia/ros2_ws/my-service-account.json"
        self.client = speech.SpeechClient()


        # 🆕 신뢰성 높은 QoS 설정
        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )


        # ROS 2 인터페이스
        stt_group = ReentrantCallbackGroup()
        self.publisher_ = self.create_publisher(String, "user_question", 10)
        # 🆕 트리거 상태 퍼블리시용 추가
        self.trigger_status_pub = self.create_publisher(String, "/trigger_status", 10)
        self.gif_status_pub = self.create_publisher(String, "/gif_status", 10)
        #self.gif_status_pub = self.create_publisher(String, "/gif_status", reliable_qos)
        

        # self.create_subscription(
        #     String, "processing_done", self.processing_done_callback, 10
        # )
        self.processing_subscription = self.create_subscription(String, "processing_done", self.processing_done_callback, 10)
        self.music_status_subscription = self.create_subscription(String, "music_status", self.music_status_callback, 10)
        # effect_stop 토픽 구독자 추가
        self.effect_stop_subscription = self.create_subscription(String, "effect_stop", self.effect_stop_callback, 10)


        self.spectrum_frame_counter = 0
        self.spectrum_skip_rate = 2  # 2프레임마다 1번만 처리


        # 상태 변수
        self.audio_stream = queue.Queue()
        self.audio_buffer = []  

        self.processing = False  
        self.music_playing = False  
        self.last_published_text = ""  
        self.stt_restart_time = time.time()  
        self.partial_transcript = ""  
        self.trigger_detected = False  

        self.last_speech_time = time.time()
        self.is_sound_playing = False
        
       

        # ✅ 강제 퍼블리시 방지를 위한 플래그 추가
        self.force_published = False 
        self.transcribing = False  # ✅ STT 중복 실행 방지용
        self.ignore_stt = False  # 🔇 효과음 재생 중 STT 무시

        self.waiting_for_input_after_music = False  # 음악 종료 후 최초 입력 대기 플래그
        self.timer_30s = None  # 30초 타이머 초기화

        # PyAudio 세팅 (16 kHz mono)
        self.p = pyaudio.PyAudio()
        # self.stream = self.p.open(
        #     format=pyaudio.paInt16,
        #     channels=1,
        #     rate=16000,
        #     input=True,
        #     frames_per_buffer=1024,
        #     stream_callback=self.audio_callback,
        # )
        # # STT 스레드 시작
        # threading.Thread(target=self.transcribe_streaming, daemon=True).start()
        self.device_index = 24
        
        self.stream = None

        # 마이크 스트리밍 시작
        self.visualizer_pub = self.create_publisher(String, "/audio_visualizer", 10)
        


        self.visualizer_queue = queue.Queue(maxsize=100)

        threading.Thread(target=self.visualizer_worker, daemon=True).start()

        self.is_speaking = False  # STT 인식 중인지 여부
        self.current_speaker_id = 1  # 최초 화자 id 1로 시작
        # 음성 강조 스펙트럼 시각화를 위한 변수 추가
        self.baseline_spectrum = None
        self.spectrum_history = []
        self.history_size = 50
        self.sample_rate = 16000
        
        # 주파수 계산을 위한 변수
        self.fft_size = 1024
        self.freqs = np.fft.fftfreq(self.fft_size, 1/self.sample_rate)[:self.fft_size//2]

        # 🆕 현재 각도 저장 및 고정 각도 퍼블리시용
        self.current_direction = 0.0
        self.fixed_direction_pub = self.create_publisher(Float32, "/fixed_direction", 10)
        
        # 🆕 실시간 각도 구독
        self.direction_sub = self.create_subscription(
            Float32, 
            '/sound_direction_angle', 
            self.direction_callback, 
            10
        )

        self.waiting_spectrum_pub = self.create_publisher(String, "/waiting_spectrum", 10)
        self.waiting_image_pub = self.create_publisher(String, "/waiting_image", 10)
        # 기존 플래그들 다음에 추가
        self.waiting_sequence_running = False


        # 이미지 표시 상태 추적용 플래그 추가
        self.waiting_image_displayed = False
        self.current_waiting_image_path = ""

        # 스펙트럼 평균화를 위한 변수들 (기존 변수들과 함께 추가)
        self.spectrum_buffer = []  # 5개의 스펙트럼을 저장할 버퍼
        self.spectrum_count = 0    # 현재 누적된 스펙트럼 개수



            



        # 🆕 실시간 단어 처리 추가
        self.word_processor = RealtimeWordProcessor()
        self.realtime_words_pub = self.create_publisher(String, "/realtime_words", 10)


        # 🆕 질문 확인 TTS 파일 저장 경로 추가
        self.question_confirm_path = "/home/nvidia/ros2_ws/src/pkg_mic/pkg_mic/question_confirm.mp3"
            


        self.start_audio_stream()



        

    def text2speech_question_confirm(self, text):
        """
        ElevenLabs TTS 호출하여 질문 확인 음성 생성
        """
        api_key = "sk_fdb1ba8706bb125cb308ae613f58105e23e26a89d127a4cd"
        # voice_id = "59zWnTQLbwyr94bFbcUe"
        voice_id = "2oCsvoTtWZkaDZUSExSz"
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg"
        }

        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.25,
                "speed": 0.9
            },
            "apply_text_normalization": "on"
        }

        try:
            # 🕐 TTS 생성 시간 측정 시작
            start_time = time.time()


            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                with open(self.question_confirm_path, "wb") as f:
                    f.write(response.content)
                # 🕐 TTS 생성 완료 시간 계산
                generation_time = time.time() - start_time    
                self.get_logger().info(f"🟢 질문 확인 TTS 생성 성공 → {self.question_confirm_path}")
                self.get_logger().info(f"⏱️ TTS 생성 시간: {generation_time:.3f}초")
                self.save_log(f"🟢 질문 확인 TTS 생성 성공")
                return True
            else:
                self.get_logger().error(f"🔴 TTS 오류 발생: {response.status_code}\n{response.text}")
                self.save_log(f"🔴 TTS 오류 발생: {response.status_code}")
                self.get_logger().info(f"⏱️ TTS 시도 시간: {generation_time:.3f}초")
                return False
        except Exception as e:
            self.get_logger().error(f"🔴 TTS 호출 실패: {e}")
            self.save_log(f"🔴 TTS 호출 실패: {e}")
            self.get_logger().info(f"⏱️ TTS 시도 시간: {generation_time:.3f}초")
            return False

    def extract_question_from_published_text(self, published_text):
        """
        published_text에서 순수 질문만 추출
        형태: "speaker001|질문내용" → "질문내용"
        """
        try:
            if "|" in published_text:
                _, question = published_text.split("|", 1)
                return question.strip()
            else:
                return published_text.strip()
        except Exception as e:
            self.get_logger().error(f"질문 추출 실패: {e}")
            return published_text


















    # ── Google STT -----------------------------------------------------------

    
    def processing_done_callback(self, msg):
        """ ✅ 오류 해결: 이 함수가 누락되어 있었음 """
        self.get_logger().info("Processing completed. Resuming recognition.")
        self.processing = False
        self.last_published_text = ""  
        self.force_restart_stt()


    def publish_trigger_status(self):
            """🆕 기존 trigger_detected 플래그 상태를 프론트엔드로 전송"""
            status = "triggered" if self.trigger_detected else "waiting"
            msg = String()
            msg.data = status
            self.trigger_status_pub.publish(msg)
            self.get_logger().info(f"Trigger status published: {status}")



    def music_status_callback(self, msg):
        """ 음악 상태에 따라 STT 동작 제어 """
        if msg.data == "music_playing":
            self.get_logger().info("Music is playing. Muting STT output.")
            self.music_playing = True
            self.audio_stream.queue.clear()
            self.audio_buffer = []
            self.partial_transcript = ""
            self.stop_audio_stream()

        elif msg.data == "music_done":
            self.get_logger().info("Music playback finished. Resuming STT output.")
            self.music_playing = False
            self.word_processor.reset() 
            self.publish_realtime_phrase("")   # ← True 플래그

            # 음악 종료 후 입력 대기 플래그 활성화
            self.trigger_detected = True
            self.waiting_for_input_after_music = True
            self.partial_transcript = ""
            # 🆕 트리거 상태 퍼블리시 (대기 상태)
            self.publish_trigger_status()

            # 마이크 입력 다시 시작 및 STT 재개
            self.start_audio_stream()
            threading.Thread(target=self.transcribe_streaming, daemon=True).start()

            # 음악 종료 후 30초 타이머 시작
            self.start_30s_timer()
            # 무음 모니터링은 최초 입력이 들어올 때 시작



    def start_30s_timer(self):
        """음악 종료 후 30초 타이머 시작 함수 추가"""
        if self.timer_30s is not None and self.timer_30s.is_alive():
            self.timer_30s.cancel()

        self.get_logger().info("⏳ 음악 종료 후 30초 타이머 시작")
        self.timer_30s = threading.Timer(30, self.timer_30s_expired)
        self.timer_30s.start()


    def timer_30s_expired(self):
        self.get_logger().info("⏱️ 음악 종료 후 30초 동안 추가 입력 없음. trigger 상태 초기화")
        self.trigger_detected = False
        self.waiting_for_input_after_music = False
        self.partial_transcript = ""
        self.current_speaker_id += 1  # 새로운 화자 id 할당
        self.get_logger().info(f"새로운 speaker_id 할당: {self.current_speaker_id}")
        # 🆕 trigger_detected 상태 전송
        self.publish_trigger_status()



    def start_audio_stream(self):
        """ 마이크 입력을 Google STT API로 실시간 전송 """
        self.get_logger().info('Starting microphone stream (continuous)...')
     
        #self.stop_audio_stream()

        try:
            self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,  # ✅ PulseAudio에서는 1 채널을 지원할 가능성이 높음
            rate=16000,
            input=True,
            frames_per_buffer=1024,
            input_device_index=None,  # ✅ PulseAudio의 기본 입력 장치를 사용
            stream_callback=self.audio_callback
        )


            time.sleep(0.5)  
            #self.transcribe_streaming()  # ✅ 누락된 함수 호출 (아래에 정의)
            threading.Thread(target=self.transcribe_streaming, daemon=True).start()
        except Exception as e:
            self.get_logger().error(f"Failed to start microphone stream: {e}")
            self.get_logger().info("Retrying microphone stream in 1 second...")
            time.sleep(1)
            self.start_audio_stream()

    def stop_audio_stream(self):
        """ ✅ 마이크 입력 스트리밍 중지 함수 추가 """
        if self.stream is not None:
            self.get_logger().info("Stopping microphone stream...")
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None



    def transcribe_streaming(self):
        """ Google STT API를 사용하여 실시간 음성 인식 """
        if self.transcribing:
            self.get_logger().info("STT already running, skipping duplicate start.")
            return

        self.transcribing = True
        self.get_logger().info("Starting transcribe_streaming...")

        def request_gen():
            while True:
                data = self.audio_stream.get() 
                if data is None:
                    break
                yield speech.StreamingRecognizeRequest(audio_content=data)

        # 1) 화자 분할 설정
        diar_cfg = speech.SpeakerDiarizationConfig(
            enable_speaker_diarization=True,
            min_speaker_count=2,
            max_speaker_count=2,
        )

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="ko-KR",
            model='telephony',
            enable_automatic_punctuation = True
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config, interim_results=True
        )
        try:
            self.stt_restart_time = time.time()
            responses = self.client.streaming_recognize(
                streaming_config, request_gen()
            )
            self.process_responses(responses)
        except Exception as e:
            self.get_logger().error(f"STT error: {e}")
            self.force_restart_stt()
        finally:
            self.transcribing = False  # ✅ 항상 플래그 초기화
            



    def audio_callback(self, in_data, frame_count, time_info, status):
        # 1) 시각화용 큐에 즉시 저장 (blocking 없이)
        try:
            self.visualizer_queue.put_nowait(in_data)
        except queue.Full:
            pass

      
    
        # 3) STT 큐 등 기존 로직
        if not (self.music_playing or self.ignore_stt):
            self.audio_stream.put(in_data)
            if self.trigger_detected:
                self.audio_buffer.append(in_data)

        return None, pyaudio.paContinue


    def visualizer_worker(self):
        while True:
            in_data = self.visualizer_queue.get()
            self.publish_audio_visualizer(in_data)


  



    # def publish_audio_visualizer(self, in_data):

    #     samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
        
 
    #     # 1. 기본 FFT 계산
    #     fft = np.fft.fft(samples)
    #     spectrum = np.abs(fft[:len(fft)//2])
  

    #     # Mp3Player.py와 동일한 정규화
    #     spectrum = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum
        
    #     # Mp3Player.py와 동일한 JSON 구조로 발송
    #     msg = String()
    #     msg.data = json.dumps({"spectrum": spectrum.tolist()})
    #     self.visualizer_pub.publish(msg)


    # # DC 오프셋 제거
    # def publish_audio_visualizer(self, in_data):
    #     # 🆕 0.5초 간격 출력 제어
    #     current_time = time.time()
    #     if not hasattr(self, 'last_print_time'):
    #         self.last_print_time = 0
        
    #     should_print = current_time - self.last_print_time >= 0.5

    #     samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
    #     # 🆕 DC 오프셋 제거 - 평균값 빼기
    #     samples = samples - np.mean(samples)

  
    #     window = np.hanning(len(samples))
    #     windowed_data = samples * window

    #     # 1. 기본 FFT 계산
    #     fft = np.fft.fft(windowed_data)

    #     spectrum = np.abs(fft[:len(fft)//2])

    #     # Mp3Player.py와 동일한 정규화
    #     # spectrum = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum
        



    #     # 🆕 0.5초에 한 번씩만 스펙트럼 값 10개 출력
    #     if should_print:
    #         self.last_print_time = current_time
    #         print("=== 스펙트럼 값 (처음 10개) ===")
    #         for i in range(min(10, len(spectrum))):
    #             print(f"[{i}] {spectrum[i]:.6f}")
    #         print("==============================")
            
    #     # Mp3Player.py와 동일한 JSON 구조로 발송
    #     msg = String()
    #     msg.data = json.dumps({"spectrum": spectrum.tolist()})
    #     self.visualizer_pub.publish(msg)




    #DC 오프셋 제거 + 5회 평균화
    def publish_audio_visualizer(self, in_data):
        # 🆕 0.5초 간격 출력 제어
        current_time = time.time()
        if not hasattr(self, 'last_print_time'):
            self.last_print_time = 0
        
        should_print = current_time - self.last_print_time >= 0.5

        samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
        # 🆕 DC 오프셋 제거 - 평균값 빼기
        samples = samples - np.mean(samples)

        window = np.hanning(len(samples))
        windowed_data = samples * window

        # 1. 기본 FFT 계산
        fft = np.fft.fft(windowed_data)
        spectrum = np.abs(fft[:len(fft)//2])
        # spectrum = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum

        # Mp3Player.py와 동일한 JSON 구조로 평균 스펙트럼 발송
        msg = String()
        msg.data = json.dumps({"spectrum": spectrum.tolist()})
        self.visualizer_pub.publish(msg)
  
   #   spectrum = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum
        

        # log_spectrum = np.log10(spectrum + 1)
        # log_spectrum /= np.log10(np.max(spectrum) + 1)  # 정규화
         

        # # 🆕 스펙트럼 버퍼에 추가
        # self.spectrum_buffer.append(spectrum)
        # self.spectrum_count += 1

        # # 🆕 5개가 모이면 평균 계산 후 전송
        # if self.spectrum_count >= 2:
        #     # 평균 스펙트럼 계산
        #     avg_spectrum = np.mean(self.spectrum_buffer, axis=0)
        #     # 로그 스케일링 적용 (값이 0~수천까지 나올 수 있으므로)
           
        #     # # 🆕 0.5초에 한 번씩만 스펙트럼 값 10개 출력 (디버깅용)
        #     # if should_print:
        #     #     self.last_print_time = current_time
        #     #     print("=== 평균 스펙트럼 값 (처음 10개) ===")
        #     #     for i in range(min(10, len(avg_spectrum))):
        #     #         print(f"[{i}] {avg_spectrum[i]:.6f}")
        #     #         print(f"최소값: {np.min(avg_spectrum):.6f}")
        #     #         print(f"최대값: {np.max(avg_spectrum):.6f}")
        #     #     print("==============================")
            
        #     # Mp3Player.py와 동일한 JSON 구조로 평균 스펙트럼 발송
        #     msg = String()
        #     msg.data = json.dumps({"spectrum": avg_spectrum.tolist()})
        #     self.visualizer_pub.publish(msg)
            
        #     # 🆕 버퍼 초기화
        #     self.spectrum_buffer = []
        #     self.spectrum_count = 0
        
        # # 개별 스펙트럼은 더 이상 전송하지 않음













    def process_responses(self, responses):
        silence_threshold = 3  # 3초 무음 시 퍼블리시
        for resp in responses:
            for result in resp.results:
                txt = result.alternatives[0].transcript.strip()
                is_final = result.is_final

                if self.ignore_stt:
                    self.get_logger().info(f"[무시됨] 효과음 재생 중 transcript: {txt}")
                    continue

                if txt:
                    self.is_speaking = True  # 말하고 있음 (텍스트 인식됨)
                    self.last_speech_time = time.time()
                    self.silence_seconds = 0


                    # 🆕 실시간 단어 처리 (is_final 플래그 추가)
                    if self.trigger_detected:
                        new_phrases = self.word_processor.process_transcript(txt, is_final)
                        for phrase in new_phrases:
                            self.publish_realtime_phrase(phrase)
                    
                        




                    # 음악 종료 후 최초 음성 입력이 들어왔을 때만 무음 감지 시작
                    if self.waiting_for_input_after_music:
                        self.waiting_for_input_after_music = False  # 최초 입력 감지 완료
                        self.get_logger().info("🎤 음악 종료 후 최초 입력 감지됨. 무음 체크 시작.")
                        self.start_silence_monitoring()
                else:
                    self.is_speaking = False  # 말 안 하고 있음 (텍스트 없음)

                self.get_logger().info(f'Transcript: {txt} (Final: {is_final})')

                # ── 1) trigger 감지 시 ──
                if not self.trigger_detected:
                    if "안녕!" in txt:
                        split_text = txt.split("안녕!", 1)
                        if len(split_text) > 1:
                            self.partial_transcript = split_text[1].strip()
                            self.get_logger().info(f"Trigger detected. Capturing transcript: {self.partial_transcript}")


                            # 🆕 단어 처리기 초기화
                            self.word_processor.reset()



                            self.play_effect_sound_trigger()
                            self.trigger_detected = True
                            self.publish_realtime_phrase("")   # ← True 플래그
                            # 🆕 트리거 감지 상태 전송
                            self.publish_trigger_status()
                            self.audio_buffer = []  # 본 질문 음성 버퍼링 시작
                            
                            
                            self.start_silence_monitoring()
                        continue

                # ── 2) trigger 이후 본 질문 저장 ──
                elif self.trigger_detected:
                    if "안녕!" in txt:
                        split_text = txt.split("안녕!", 1)
                        if len(split_text) > 1:
                            self.partial_transcript = split_text[1].strip()
                    else:
                        self.partial_transcript = txt

                # ── 3) 무음 3초 후 퍼블리시 시점 ──
                if is_final and self.partial_transcript.strip():
                    try:
                        if self.waiting_sequence_running:
                            self.get_logger().info("이미 대기 시퀀스가 실행 중입니다.")
                            return
                        
                        self.publish_transcription(self.partial_transcript)
                        self.save_audio_clip()
                        return
                    except Exception as e:
                        self.get_logger().error(f"Speaker identification error: {e}")
                        self.publish_transcription(self.partial_transcript)
                        self.save_audio_clip()
                        return

            if not self.waiting_for_input_after_music:
                self.start_silence_monitoring()


    def publish_realtime_phrase(self, phrase, is_final=False):
        """실시간 단어 구문을 프론트엔드로 전송"""
        phrase_data = {
            "type": "word_phrase",
            "phrase": phrase,
            "timestamp": time.time(),
            "speaker_id": self.current_speaker_id,
            "is_final": is_final          # ★ 추가
        }
        
        msg = String()
        msg.data = json.dumps(phrase_data)
        self.realtime_words_pub.publish(msg)
        
        self.get_logger().info(f"📝 실시간 구문 전송: '{phrase}'")




    def start_silence_monitoring(self):
        """무음 상태에서 1초마다 경과 시간을 출력하는 스레드 실행"""
        
        if hasattr(self, 'silence_monitoring_thread') and self.silence_monitoring_thread.is_alive():
            return  # 이미 실행 중이면 중복 실행 방지
        
        self.silence_monitoring_thread = threading.Thread(target=self.monitor_silence,args=(3,), daemon=True)
        self.silence_monitoring_thread.start()


   

    def monitor_silence(self, silence_threshold):
        """ 3초 이상 무음 상태가 지속되면 강제 Publish 또는 상태 초기화 """
        self.silence_seconds = 0  # 무음 지속 시간 초기화
        self.after_prompt = False  # 종료음 후 무음 감지 상태 초기화

        while self.trigger_detected:
            # 🔥 오디오 재생 중일 때 무음 감지 시작 방지
            if self.is_sound_playing:
                time.sleep(0.1)
                continue

            elapsed_silence = time.time() - self.last_speech_time

            # 1초마다 로그 출력
            if elapsed_silence >= self.silence_seconds + 1:
                self.silence_seconds += 1
                self.get_logger().info(f"무음성 {self.silence_seconds}초 경과 (무음 감지 중)")

            # 무음 시간이 임계값을 초과했을 때
            if elapsed_silence >= silence_threshold:
                # 🔥 이미 퍼블리시된 경우 종료음 실행 방지
                if self.force_published:
                    self.get_logger().info("이미 퍼블리시된 텍스트이므로 종료음 생략")
                    self.force_published = False  # 플래그 리셋
                    break

                # 🔥 무음 시간 동안 텍스트가 있는지 최종 확인
                if self.partial_transcript.strip():
                    self.get_logger().info(f"무음성 3초 경과 전 텍스트 감지: {self.partial_transcript}")
                    self.publish_transcription(self.partial_transcript)
                    self.last_published_text = self.partial_transcript
                    self.partial_transcript = ""
                    self.trigger_detected = False
                    self.get_logger().info("무음 감지 중지: 퍼블리시 완료")
                    break

                # 종료음 재생 전이면
                if not self.after_prompt:
                    self.get_logger().info("무음성 3초 경과 (초기 체크): 종료음 재생 후 추가 무음 체크 시작")
                    self.play_effect_sound_requestion()  # 종료음 재생

                    # 종료음 후에도 무음 체크를 위해 시간 갱신
                    self.last_speech_time = time.time()

                    # 상태 전환
                    self.after_prompt = True
                    self.silence_seconds = 0  # 무음 카운터 초기화
                    continue  # 추가 무음 체크 계속

                # 종료음 후 3초 무음 상태 확인
                else:
                    if not self.partial_transcript.strip():
                        self.get_logger().info(f"종료음 후 추가 무음 {self.silence_seconds}초 경과 (음성 없음)")
                        self.get_logger().info("추가 음성이 없으므로 초기 상태로 복귀")
                        self.trigger_detected = False
                        self.partial_transcript = ""
                        self.after_prompt = False  # 상태 초기화
                        break
                    else:
                        self.get_logger().info(f"종료음 후 추가 무음 {self.silence_seconds}초 경과 (음성 감지)")
                        self.get_logger().info("종료음 재생 후 3초 경과로 인해 강제 publish")
                        self.publish_transcription(self.partial_transcript)
                        self.last_published_text = self.partial_transcript
                        self.partial_transcript = ""
                        self.after_prompt = False  # 상태 초기화
                        break

            time.sleep(0.1)







    def play_effect_sound_requestion(self):
        """ 랜덤으로 요청 음성(MP3)을 재생하며, 재생 중 텍스트 입력을 무시 """
        # 효과음 파일이 저장된 디렉토리 경로
        effects_dir = "/home/nvidia/ros2_ws/src/pkg_mic/pkg_mic/_tts_requestion"

        # 디렉토리에서 MP3 파일 목록 가져오기
        mp3_files = [f for f in os.listdir(effects_dir) if f.endswith(".mp3")]

        if not mp3_files:
            self.get_logger().error("No MP3 files found in the requestion directory.")
            return

        try:
            self.ignore_stt = True  # 🔇 STT 입력 무시 시작
            self.audio_buffer = []
            self.partial_transcript = ""
            self.audio_stream.queue.clear()

            # 랜덤으로 하나의 MP3 파일 선택
            selected_file = random.choice(mp3_files)
            selected_path = os.path.join(effects_dir, selected_file)

            self.get_logger().info(f"Playing sound: {selected_file}")

            # 🔥 효과음 재생 중 상태 설정
            self.is_sound_playing = True

            # ✅ 버퍼 초기화 (효과음 재생 중 텍스트 무시)
            

            # pygame을 사용하여 MP3 파일 재생
            pygame.mixer.init()
            pygame.mixer.music.load(selected_path)
            pygame.mixer.music.play()

            # 재생이 끝날 때까지 대기
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            self.ignore_stt = False  # ✅ 재생 완료 후 STT 다시 허용

            # 🔥 효과음 재생 완료
            self.is_sound_playing = False
            self.start_silence_monitoring()

        except Exception as e:
            self.get_logger().error(f"Failed to play effect sound: {e}")
            # 🔥 비상상황: 플래그 해제
            self.is_sound_playing = False




    def play_effect_sound_trigger(self):
        """효과음 파일을 재생하며, 재생 중 텍스트 입력을 무시"""
        # 효과음 파일이 저장된 디렉토리 경로
        effects_dir = "/home/nvidia/ros2_ws/src/pkg_mic/pkg_mic/_tts_trigger"

        # 디렉토리에서 MP3 파일 목록 가져오기
        mp3_files = [f for f in os.listdir(effects_dir) if f.endswith(".mp3")]

        try:
            self.ignore_stt = True  # 🔇 STT 입력 무시 시작
            # ✅ 버퍼 초기화 (효과음 재생 중 텍스트 무시)
            self.audio_buffer = []
            self.partial_transcript = ""
            self.audio_stream.queue.clear()
            # 랜덤으로 하나의 MP3 파일 선택
            selected_file = random.choice(mp3_files)
            selected_path = os.path.join(effects_dir, selected_file)

            self.get_logger().info(f"Playing sound: {selected_file}")

            # 🔥 효과음 재생 중 상태 설정
            self.is_sound_playing = True

            

            # pygame을 사용하여 MP3 파일 재생
            pygame.mixer.init()
            pygame.mixer.music.load(selected_path)
            pygame.mixer.music.play()

            # 🔥 재생이 끝날 때까지 대기
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            # 🔥 효과음 재생 완료 후에 무음 감지 시작
            self.is_sound_playing = False
            
            

            self.get_logger().info("효과음 재생 완료 후 무음 감지 초기화")
            self.last_speech_time = time.time()  # 🔥 무음 시간 초기화
            self.start_silence_monitoring()
            self.ignore_stt = False  # ✅ 재생 완료 후 STT 다시 허용

        except Exception as e:
            self.get_logger().error(f"Failed to play effect sound: {e}")
            # 🔥 비상상황: 플래그 해제
            self.is_sound_playing = False

  


    # #원본 play effect
    # def play_effect_sound_waiting_1(self):
    #     """대기 효과음 1을 재생하며 스펙트럼 시각화 (Mp3Player.py 방식과 동일)"""
    #     effects_dir = "/home/nvidia/ros2_ws/src/pkg_mic/pkg_mic/_tts_waiting1"
        
    #     if not os.path.exists(effects_dir):
    #         self.get_logger().error(f"Directory not found: {effects_dir}")
    #         return
            
    #     mp3_files = [f for f in os.listdir(effects_dir) if f.endswith(".mp3")]

    #     if not mp3_files:
    #         self.get_logger().error("No MP3 files found in waiting_1 directory.")
    #         return

    #     try:
    #         # 랜덤으로 하나의 MP3 파일 선택
    #         selected_file = random.choice(mp3_files)
    #         selected_path = os.path.join(effects_dir, selected_file)

    #         self.get_logger().info(f"Playing waiting sound 1: {selected_file}")

    #         # pydub을 사용하여 MP3 로드 (Mp3Player.py와 동일)
    #         sound = AudioSegment.from_file(selected_path, format="mp3")
            
    #         # 정규화 (Mp3Player.py와 동일)
    #         target_dBFS = -14.0
    #         change_in_dBFS = target_dBFS - sound.dBFS
    #         sound = sound.apply_gain(change_in_dBFS)
            
    #         # 임시 WAV로 변환
    #         temp_wav = "/tmp/waiting_audio.wav"
    #         sound.export(temp_wav, format="wav")

    #         # Mp3Player.py와 동일한 방식으로 스펙트럼과 재생 병렬 처리
    #         self.waiting_publish_and_play(temp_wav)
    #         time.sleep(1)
        

    #         self.get_logger().info("Waiting sound 1 playback finished")

    #     except Exception as e:
    #         self.get_logger().error(f"Failed to play waiting sound 1: {e}")





    def play_effect_sound_waiting_1(self):
        """
        질문 확인 TTS 생성 및 재생 (기존 대기음 대신)
        """
        try:
            # 1. 발행된 질문 텍스트 확인
            if not hasattr(self, 'last_published_text') or not self.last_published_text:
                self.get_logger().warning("발행된 질문이 없어 기본 대기음을 재생합니다.")
                self.play_default_waiting_sound()
                return

            # 2. 순수 질문 텍스트 추출 (speaker### 부분 제거)
            question_text = self.extract_question_from_published_text(self.last_published_text)
            
            # 3. 질문 확인 문구 생성
            confirm_text = f"{question_text} 라고 물어본거지?"
            self.get_logger().info(f"🎤 질문 확인 TTS 생성: {confirm_text}")
            self.save_log(f"🎤 질문 확인 TTS 생성: {confirm_text}")

            # 4. TTS 생성
            if not self.text2speech_question_confirm(confirm_text):
                self.get_logger().warning("TTS 생성 실패로 기본 대기음을 재생합니다.")
                self.play_default_waiting_sound()
                return

            # 5. 생성된 TTS 파일 재생 (기존 스펙트럼 방식 유지)
            if os.path.exists(self.question_confirm_path):
                # pydub을 사용하여 MP3 로드
                sound = AudioSegment.from_file(self.question_confirm_path, format="mp3")
                
                # 정규화 (기존 방식과 동일)
                target_dBFS = -14.0
                change_in_dBFS = target_dBFS - sound.dBFS
                sound = sound.apply_gain(change_in_dBFS)
                
                # 임시 WAV로 변환
                temp_wav = "/tmp/question_confirm_audio.wav"
                sound.export(temp_wav, format="wav")

                # 기존과 동일한 방식으로 스펙트럼과 재생 병렬 처리
                self.waiting_publish_and_play(temp_wav)
                # time.sleep(1)

                self.get_logger().info("질문 확인 TTS 재생 완료")
                self.save_log("질문 확인 TTS 재생 완료")
            else:
                self.get_logger().error(f"TTS 파일이 생성되지 않음: {self.question_confirm_path}")
                self.play_default_waiting_sound()

        except Exception as e:
            self.get_logger().error(f"질문 확인 TTS 처리 실패: {e}")
            self.save_log(f"질문 확인 TTS 처리 실패: {e}")
            self.play_default_waiting_sound()

    def play_default_waiting_sound(self):
        """
        TTS 생성 실패 시 기본 대기음 재생 (백업용)
        """
        try:
            effects_dir = "/home/nvidia/ros2_ws/src/pkg_mic/pkg_mic/_tts_waiting1"
            
            if not os.path.exists(effects_dir):
                self.get_logger().error(f"기본 대기음 디렉토리 없음: {effects_dir}")
                return
                
            mp3_files = [f for f in os.listdir(effects_dir) if f.endswith(".mp3")]

            if not mp3_files:
                self.get_logger().error("기본 대기음 파일이 없습니다.")
                return

            # 랜덤으로 하나의 MP3 파일 선택
            selected_file = random.choice(mp3_files)
            selected_path = os.path.join(effects_dir, selected_file)

            self.get_logger().info(f"기본 대기음 재생: {selected_file}")

            # 기존 방식과 동일하게 재생
            sound = AudioSegment.from_file(selected_path, format="mp3")
            target_dBFS = -14.0
            change_in_dBFS = target_dBFS - sound.dBFS
            sound = sound.apply_gain(change_in_dBFS)
            
            temp_wav = "/tmp/default_waiting_audio.wav"
            sound.export(temp_wav, format="wav")

            self.waiting_publish_and_play(temp_wav)
            # time.sleep(1)

        except Exception as e:
            self.get_logger().error(f"기본 대기음 재생 실패: {e}")







    def effect_stop_callback(self, msg):
        """effect_stop 토픽 수신 시 대기 이미지 숨김"""
        if msg.data == "effect_stop" and self.waiting_image_displayed:
            self.get_logger().info("🛑 effect_stop 토픽 수신: 대기 이미지 숨김")
            
            # 이미지 숨김 처리
            hide_msg = String()
            hide_msg.data = ""
            self.waiting_image_pub.publish(hide_msg)
                        
            # 상태 플래그 초기화
            self.waiting_image_displayed = False
            self.current_waiting_image_path = ""
            
            self.get_logger().info("대기 이미지 표시 완료")









    def play_effect_sound_waiting_2(self):
        """대기 효과음 재생 및 effect_stop 토픽까지 이미지 표시"""
        # 효과음 파일이 저장된 디렉토리 경로
        effects_dir = "/home/nvidia/ros2_ws/src/pkg_mic/pkg_mic/_tts_waiting2"

        # 디렉토리에서 MP3 파일 목록 가져오기
        mp3_files = [f for f in os.listdir(effects_dir) if f.endswith(".mp3")]

        if not mp3_files:
            self.get_logger().info("No MP3 files found in the effects directory.")
            return

        # 랜덤으로 하나의 MP3 파일 선택
        selected_file = random.choice(mp3_files)
        selected_path = os.path.join(effects_dir, selected_file)

        self.get_logger().info(f"Playing sound: {selected_file}")

        # pygame을 사용하여 MP3 파일 재생
        pygame.mixer.init()
        pygame.mixer.music.load(selected_path)
        pygame.mixer.music.play()
        
        # 대기 이미지 표시 (effect_stop 토픽까지 유지)
        self.display_waiting_image_until_stop()






    def display_waiting_image_until_stop(self):
        """대기 이미지를 랜덤으로 표시 (effect_stop 토픽까지 유지)"""
        image_dir = "/home/nvidia/ros2_ws/emotion-face-react/public/waiting"
        
        if not os.path.exists(image_dir):
            self.get_logger().error(f"Directory not found: {image_dir}")
            return
        
        try:
            # .jpeg 파일 목록 가져오기
            jpeg_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpeg')]
            
            if not jpeg_files:
                self.get_logger().error("No JPEG files found in waiting_img directory.")
                return

            # 랜덤으로 하나의 이미지 선택
            selected_file = random.choice(jpeg_files)
            image_path = f"/waiting/{selected_file}"  # public 폴더 기준 경로
            
            self.get_logger().info(f"Displaying waiting image: {selected_file} (until effect_stop)")
            
            # 이미지 경로 전송
            msg = String()
            msg.data = image_path
            self.waiting_image_pub.publish(msg)
            
            # 상태 플래그 설정
            self.waiting_image_displayed = True
            self.current_waiting_image_path = image_path
            
            self.get_logger().info("대기 이미지 표시 시작 (effect_stop 토픽 대기중)")

        except Exception as e:
            self.get_logger().error(f"Failed to display waiting image: {e}")
            self.waiting_image_displayed = False



   

    def direction_callback(self, msg):
        """실시간 각도 업데이트"""
        self.current_direction = msg.data

    def publish_transcription(self, text: str):


        # 🆕 중복 실행 방지
        if self.waiting_sequence_running:
            self.get_logger().info("대기 시퀀스가 이미 실행 중입니다. 중복 실행을 방지합니다.")
            return


        if text.strip():
            if self.timer_30s and self.timer_30s.is_alive():
                self.timer_30s.cancel()  # ✅ 퍼블리시 후 타이머 종료

            self.force_published = True


            # 🆕 2단계: searching 상태 신호 전송 (기존 gif_status_pub 활용)
            status_msg = String()
            status_msg.data = "searching"  # gif_status 대신 searching으로 통일
            self.gif_status_pub.publish(status_msg)
            self.get_logger().info("📊 UserQuestion에서 searching 상태 전송")
            # time.sleep(0.5)


            # 🆕 질문 퍼블리시와 동시에 현재 각도를 고정 각도로 전송
            fixed_msg = Float32()
            fixed_msg.data = self.current_direction
            self.fixed_direction_pub.publish(fixed_msg)
            self.get_logger().info(f"🔒 고정 각도 설정: {self.current_direction}도")
            

            msg = String()
            msg.data = f"speaker{self.current_speaker_id:03d}|{text}"
            self.publisher_.publish(msg)
            self.last_published_text = msg.data

            self.get_logger().info(f'Transcription published: "{msg.data}"')
            self.save_log(f'Transcription published: "{msg.data}"')
            self.partial_transcript = ""  # ✅ 퍼블리시 후 즉시 초기화
            self.trigger_detected = False  # ✅ 퍼블리시 후 trigger 상태 초기화
            self.waiting_for_input_after_music = False  # ✅ 입력 대기 상태 해제
            self.publish_realtime_phrase(text, is_final=True)
            # time.sleep(2)
            #self.play_effect_sound_rag()




            # 🆕 플래그 설정 후 실행
            self.waiting_sequence_running = True
            threading.Thread(target=self.execute_waiting_sequence, daemon=False).start()    


  

    

    # def play_effect_sound_rag(self):
    #     # 효과음 파일이 저장된 디렉토리 경로
    #     effects_dir = "/home/nvidia/ros2_ws/src/pkg_mic/pkg_mic/_tts_rag"

    #     # 디렉토리에서 MP3 파일 목록 가져오기
    #     mp3_files = [f for f in os.listdir(effects_dir) if f.endswith(".mp3")]

    #     if not mp3_files:
    #         self.get_logger().info("No MP3 files found in the effects directory.")
    #         return

    #     # 랜덤으로 하나의 MP3 파일 선택
    #     selected_file = random.choice(mp3_files)
    #     selected_path = os.path.join(effects_dir, selected_file)

    #     self.get_logger().info(f"Playing sound: {selected_file}")

    #     # pygame을 사용하여 MP3 파일 재생
    #     pygame.mixer.init()
    #     pygame.mixer.music.load(selected_path)
    #     pygame.mixer.music.play()
        
    #     # 재생이 끝날 때까지 대기
    #     while pygame.mixer.music.get_busy():
    #         pygame.time.Clock().tick(10)




    def execute_waiting_sequence(self):
        """새로운 대기 효과들을 순차 실행 (중복 방지 포함)"""
        try:
            # 🆕 중복 실행 체크
            if not self.waiting_sequence_running:
                self.get_logger().info("대기 시퀀스가 이미 완료되었습니다.")
                return
                
            self.get_logger().info("대기 시퀀스 시작")
            
            # 첫 번째 대기 효과 (스펙트럼 시각화)
            self.play_effect_sound_waiting_1()
            
            # 두 번째 대기 효과 (이미지 표시)  
            self.play_effect_sound_waiting_2()
            
            self.get_logger().info("대기 시퀀스 완료")
            
        except Exception as e:
            self.get_logger().error(f"대기 효과 실행 중 오류: {e}")
        finally:
            # 🆕 플래그 해제
            self.waiting_sequence_running = False








    def waiting_publish_and_play(self, wav_path):
        """Mp3Player.py의 publish_and_play와 동일한 방식"""
        import wave
        
        wf = wave.open(wav_path, 'rb')
        chunk_size = 2024

        def publish_spectrum():
            data = wf.readframes(chunk_size)
            while data:
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                if wf.getnchannels() == 2:
                    samples = samples.reshape((-1, 2)).mean(axis=1)
            
                

                fft = np.fft.fft(samples)
                spectrum = np.abs(fft[:len(fft)//2])
                #spectrum = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum
                msg = String()
                msg.data = json.dumps({"spectrum": spectrum.tolist()})
                self.waiting_spectrum_pub.publish(msg)
                data = wf.readframes(chunk_size)
                time.sleep(chunk_size / wf.getframerate())







        spectrum_thread = threading.Thread(target=publish_spectrum)
        spectrum_thread.start()

        # 시스템 명령어로 재생 (Mp3Player.py와 동일)
        os.system(f"aplay {wav_path}")
        spectrum_thread.join()
        wf.close()


            



    def save_audio_clip(self):
        """ "안녕!" 이후의 오디오를 WAV 파일로 저장 """
        if not self.audio_buffer:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/home/nvidia/ros2_ws/audio_files/{timestamp}.wav"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b''.join(self.audio_buffer))

        self.get_logger().info(f"Saved audio: {filename}")
        self.save_log(f"Saved audio: {filename}")
        self.audio_buffer = []  
        
        
  

    def force_restart_stt(self):
        self.get_logger().info("Forcing STT restart...")

        # ✅ STT 세션 종료 표시
        self.transcribing = False

        # ✅ 세션 강제 중지
        self.stop_audio_stream()

        # ✅ 대기 시간 조금 여유롭게
        # time.sleep(2.5)

        # ✅ 입력 스트림 재시작
        self.start_audio_stream()

        # ✅ STT 재시작 – 쓰레드로 안전하게 분리
        threading.Thread(target=self.transcribe_streaming, daemon=True).start()


    def save_log(self, message):
        """ 로그를 파일에 저장 """
        log_file_path = "/home/nvidia/ros2_ws/_logs/UserQuestion_log.txt"
        log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(log_message)


    

# ──────────────────────────────────────────────────────────────────────────────
# 메인 루프
# ──────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = UserQuestion()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    async def spin():
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            await asyncio.sleep(0.1)

  

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(spin())
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
