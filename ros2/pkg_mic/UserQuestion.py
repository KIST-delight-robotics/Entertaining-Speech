
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from google.cloud import speech
import pyaudio
import queue
import wave
import time
import simpleaudio as sa
import threading
import random
from pydub import AudioSegment
from pydub.playback import play
from datetime import datetime
import pygame


class UserQuestion(Node):
    def __init__(self):
        super().__init__('UserQuestion')
        self.get_logger().info('UserQuestion Node has started')

        # Google Cloud 인증 설정
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/nvidia/ros2_ws/my-service-account.json"
        
        self.client = speech.SpeechClient()
        
        # 퍼블리셔 설정 (ROS2 토픽 "user_question")
        self.publisher_ = self.create_publisher(String, "user_question", 10)

        # ✅ 구독 설정 (오류 수정: 함수 추가)
        self.processing_subscription = self.create_subscription(String, "processing_done", self.processing_done_callback, 10)
        self.music_status_subscription = self.create_subscription(String, "music_status", self.music_status_callback, 10)

        # 오디오 스트리밍 관련 설정
        self.audio_stream = queue.Queue()
        self.audio_buffer = []  
        self.processing = False  
        self.music_playing = False  
        self.last_published_text = ""  
        self.stt_restart_time = time.time()  
        self.partial_transcript = ""  
        self.trigger_detected = False  
        self.last_speech_time = None  
        # ✅ 강제 퍼블리시 방지를 위한 플래그 추가
        self.force_published = False 
        self.transcribing = False  # ✅ STT 중복 실행 방지용
        self.ignore_stt = False  # 🔇 효과음 재생 중 STT 무시

        self.waiting_for_input_after_music = False  # 음악 종료 후 최초 입력 대기 플래그
        self.timer_30s = None  # 30초 타이머 초기화





        # PyAudio 설정
        self.p = pyaudio.PyAudio()
        self.device_index = 24
        
        self.stream = None

        # 마이크 스트리밍 시작
        self.start_audio_stream()

    def processing_done_callback(self, msg):
        """ ✅ 오류 해결: 이 함수가 누락되어 있었음 """
        self.get_logger().info("Processing completed. Resuming recognition.")
        self.processing = False
        self.last_published_text = ""  
        self.force_restart_stt()


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

            # 음악 종료 후 입력 대기 플래그 활성화
            self.trigger_detected = True
            self.waiting_for_input_after_music = True
            self.partial_transcript = ""

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



    def start_audio_stream(self):
        """ 마이크 입력을 Google STT API로 실시간 전송 """
        self.get_logger().info('Starting microphone stream (continuous)...')
     
        #self.stop_audio_stream()

        try:
            self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,  # ✅ PulseAudio에서는 1 채널을 지원할 가능성이 높음
            rate=44100,
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

        def request_generator():
            while True:
                chunk = self.audio_stream.get()
                if chunk is None:
                    break
                yield speech.StreamingRecognizeRequest(audio_content=chunk)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code="ko-KR",
            model='telephony'
        )

        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True
        )

        try:
            self.stt_restart_time = time.time()
            responses = self.client.streaming_recognize(streaming_config, request_generator())
            self.process_responses(responses)
        except Exception as e:
            self.get_logger().error(f"Error in streaming STT: {e}")
            self.force_restart_stt()
        finally:
            self.transcribing = False  # ✅ 항상 플래그 초기화


    def audio_callback(self, in_data, frame_count, time_info, status):
        if self.music_playing or self.ignore_stt:
            if self.trigger_detected:
                self.audio_buffer.append(in_data)  # ✅ 트리거 상태에서는 녹음은 계속해야 함
            return None, pyaudio.paContinue


        """ 마이크로 입력된 데이터를 큐에 추가 """
        self.audio_stream.put(in_data)

        

        # ✅ "안녕" 감지 후 음성 데이터를 버퍼에 저장
        if self.trigger_detected:
            self.audio_buffer.append(in_data)

        return None, pyaudio.paContinue
    


    def process_responses(self, responses):
        silence_threshold = 3  # 3초 무음 시 퍼블리시

        for response in responses:
            for result in response.results:
                transcript = result.alternatives[0].transcript.strip()
                is_final = result.is_final

                if self.ignore_stt:
                    self.get_logger().info(f"[무시됨] 효과음 재생 중 transcript: {transcript}")
                    continue

                if transcript:
                    self.last_speech_time = time.time()
                    self.silence_seconds = 0

                    # 음악 종료 후 최초 음성 입력이 들어왔을 때만 무음 감지 시작
                    if self.waiting_for_input_after_music:
                        self.waiting_for_input_after_music = False  # 최초 입력 감지 완료
                        self.get_logger().info("🎤 음악 종료 후 최초 음성 입력 감지됨. 무음 체크 시작.")
                        self.start_silence_monitoring()

                self.get_logger().info(f'Transcript: {transcript} (Final: {is_final})')

                if not self.trigger_detected:
                    if "안녕" in transcript:
                        split_text = transcript.split("안녕", 1)
                        if len(split_text) > 1:
                            self.partial_transcript = split_text[1].strip()
                            self.get_logger().info(f"Trigger detected. Capturing transcript: {self.partial_transcript}")
                            self.play_effect_sound()
                            self.trigger_detected = True
                            self.audio_buffer = []
                            self.start_silence_monitoring()
                    continue

                elif self.trigger_detected:
                    if "안녕" in transcript:
                        split_text = transcript.split("안녕", 1)
                        if len(split_text) > 1:
                            self.partial_transcript = split_text[1].strip()
                    else:
                        self.partial_transcript = transcript

                if is_final and self.partial_transcript.strip():
                    self.publish_transcription(self.partial_transcript)
                    self.save_audio_clip()
                    # 이곳에서는 trigger_detected와 partial_transcript 초기화 제거
                    return

            if not self.waiting_for_input_after_music:
                self.start_silence_monitoring()



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
                    self.play_effect_sound_prompt()  # 종료음 재생

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







    def play_effect_sound_prompt(self):
        """ 랜덤으로 요청 음성(MP3)을 재생하며, 재생 중 텍스트 입력을 무시 """
        # 효과음 파일이 저장된 디렉토리 경로
        effects_dir = "/home/nvidia/ros2_ws/src/pkg_mic/pkg_mic/requestion"

        # 디렉토리에서 MP3 파일 목록 가져오기
        mp3_files = [f for f in os.listdir(effects_dir) if f.endswith(".wav")]

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




    def play_effect_sound(self):
        """효과음 파일을 재생하며, 재생 중 텍스트 입력을 무시"""
        # 효과음 파일이 저장된 디렉토리 경로
        effects_dir = "/home/nvidia/ros2_ws/src/pkg_mic/pkg_mic/trigger_sound"

        # 디렉토리에서 MP3 파일 목록 가져오기
        mp3_files = [f for f in os.listdir(effects_dir) if f.endswith(".wav")]

        try:
            self.ignore_stt = True  # 🔇 STT 입력 무시 시작
            # ✅ 버퍼 초기화 (효과음 재생 중 텍스트 무시)
            self.audio_buffer = []
            self.partial_transcript = ""
            self.audio_stream.queue.clear()
            # 랜덤으로 하나의 MP3 파일 선택
            selected_file = random.choice(mp3_files)
            selected_path = os.path.join(effects_dir, selected_file)

            print(f"Playing sound: {selected_file}")

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



    def publish_transcription(self, transcript):
        """ STT 결과를 퍼블리시 """
        if transcript.strip():
            if self.timer_30s and self.timer_30s.is_alive():
                self.timer_30s.cancel()  # ✅ 퍼블리시 후 타이머 종료

            self.force_published = True

            msg = String()
            msg.data = transcript.strip()
            self.publisher_.publish(msg)
            self.last_published_text = transcript.strip()

            self.get_logger().info(f'Transcription published: "{transcript.strip()}"')
            self.save_log(f'Transcription published: "{transcript.strip()}"')
            self.play_effect_sound_robot()

            self.partial_transcript = ""  # ✅ 퍼블리시 후 즉시 초기화
            self.trigger_detected = False  # ✅ 퍼블리시 후 trigger 상태 초기화
            self.waiting_for_input_after_music = False  # ✅ 입력 대기 상태 해제



    def play_effect_sound_robot(self):
        # 효과음 파일이 저장된 디렉토리 경로
        effects_dir = "/home/nvidia/ros2_ws/src/pkg_mic/pkg_mic/effects"

        # 디렉토리에서 MP3 파일 목록 가져오기
        mp3_files = [f for f in os.listdir(effects_dir) if f.endswith(".mp3")]

        if not mp3_files:
            print("No MP3 files found in the effects directory.")
            return

        # 랜덤으로 하나의 MP3 파일 선택
        selected_file = random.choice(mp3_files)
        selected_path = os.path.join(effects_dir, selected_file)

        print(f"Playing sound: {selected_file}")

        # pygame을 사용하여 MP3 파일 재생
        pygame.mixer.init()
        pygame.mixer.music.load(selected_path)
        pygame.mixer.music.play()
        
        # 재생이 끝날 때까지 대기
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)



    def save_audio_clip(self):
        """ "안녕" 이후의 오디오를 WAV 파일로 저장 """
        if not self.audio_buffer:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/home/nvidia/ros2_ws/src/pkg_mic/google_audio/{timestamp}.wav"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(44100)
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
        time.sleep(2.5)

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


def main(args=None):
    rclpy.init(args=args)
    node = UserQuestion()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
