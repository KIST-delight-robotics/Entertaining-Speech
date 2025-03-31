import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from google.cloud import speech
import pyaudio
import queue
import wave
import time
from datetime import datetime
import simpleaudio as sa
import threading
import random
from pydub import AudioSegment
from pydub.playback import play
from datetime import datetime


class UserQuestion(Node):
    def __init__(self):
        super().__init__('UserQuestion')
        self.get_logger().info('UserQuestion Node has started')

        # Google Cloud 인증 설정
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/delight/bumblebee_ws/my-service-account.json"
        
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

        # PyAudio 설정
        self.p = pyaudio.PyAudio()
        self.device_index = 2
        #123456789
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
            self.audio_buffer = []  # ✅ 기존 버퍼 삭제
            self.partial_transcript = ""  # ✅ 기존에 감지된 텍스트 삭제


             # ✅ 마이크 입력 완전 중단
            self.stop_audio_stream() 

        elif msg.data == "music_done":
            self.get_logger().info("Music playback finished. Resuming STT output.")
            self.music_playing = False

            # ✅ STT 재시작
            self.start_audio_stream()  # 마이크 입력 다시 시작
            self.transcribe_streaming()  # STT 재개

    # def get_supported_channels(self, device_index):
    #     """ 🎙️ 사용 가능한 입력 채널을 자동으로 감지 """
    #     try:
    #         dev_info = self.p.get_device_info_by_index(device_index)
    #         max_channels = dev_info["maxInputChannels"]  # ✅ 지원 가능한 최대 채널 확인
    #         if max_channels >= 2:
    #             self.get_logger().info(f"🎧 마이크가 {max_channels}채널을 지원합니다. channels=2 설정")
    #             return 2  # 🎯 스테레오(2채널) 사용
    #         else:
    #             self.get_logger().info(f"🎙️ 마이크가 {max_channels}채널을 지원합니다. channels=1 설정")
    #             return 1  # 🎯 모노(1채널) 사용
    #     except Exception as e:
    #         self.get_logger().error(f"❌ PyAudio 장치 정보 가져오기 실패: {e}")
    #         return 1  # 기본적으로 1채널 사용
            
    def start_audio_stream(self):
        """ 마이크 입력을 Google STT API로 실시간 전송 """
        self.get_logger().info('Starting microphone stream (continuous)...')

        self.stop_audio_stream()

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
            self.transcribe_streaming()  # ✅ 누락된 함수 호출 (아래에 정의)

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
        """ ✅ Google STT API를 사용하여 실시간 음성 인식 """
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

    def audio_callback(self, in_data, frame_count, time_info, status):
        # ✅ 음악이 재생 중이면 마이크 입력 무시
        if self.music_playing:
            return None, pyaudio.paContinue


        """ 마이크로 입력된 데이터를 큐에 추가 """
        self.audio_stream.put(in_data)

        # ✅ "안녕" 감지 후 음성 데이터를 버퍼에 저장
        if self.trigger_detected:
            self.audio_buffer.append(in_data)

        return None, pyaudio.paContinue
    


    def process_responses(self, responses):
        """ 음성 인식 결과 처리 (안녕 이후 문장만 퍼블리시) """
        silence_threshold = 3  # 3초 동안 무음 시 처리
        start_time = time.time()

        # ✅ 음악 재생 중이면 STT 자체를 수행하지 않음
        if self.music_playing:
            self.get_logger().info("Music is playing. STT is disabled.")
            return

        for response in responses:
            for result in response.results:
                transcript = result.alternatives[0].transcript.strip()
                is_final = result.is_final  

                if transcript:
                    self.last_speech_time = time.time()  # ✅ 음성 감지 시 시간 갱신
                    self.silence_seconds = 0  # ✅ 무음 카운트 리셋

                self.get_logger().info(f'Transcript: {transcript} (Final: {is_final})')

                if self.music_playing:
                    self.get_logger().info("Ignoring STT output since music is playing.")
                    continue

                if not self.trigger_detected:
                    if "안녕" in transcript:
                        split_text = transcript.split("안녕", 1)
                        if len(split_text) > 1:
                            self.partial_transcript = split_text[1].strip()  

                            self.get_logger().info(f"Trigger detected. Capturing transcript: {self.partial_transcript}")
                            self.play_effect_sound()  # ✅ 효과음 실행
                            self.trigger_detected = True  
                            self.audio_buffer = []  

                            # ✅ 트리거 감지 후 무음 모니터링 스레드 실행
                            self.start_silence_monitoring()

                    continue  

                elif self.trigger_detected:
                    if "안녕" in transcript:  
                        split_text = transcript.split("안녕", 1)
                        if len(split_text) > 1:
                            self.partial_transcript = split_text[1].strip()  
                    else:
                        self.partial_transcript = transcript  

                if is_final:
                    if not self.partial_transcript.strip():
                        continue
                    
                    
                    self.publish_transcription(self.partial_transcript)
                    self.save_audio_clip()  
                    self.trigger_detected = False  
                    self.partial_transcript = ""  

                    end_time = time.time()
                    trigger_detected_time = end_time - start_time  
                    self.get_logger().info(f"trigger_detected_time: {trigger_detected_time:.4f} seconds")
            
                    return
                
            end_time = time.time()
            STT_time = end_time - start_time  
            self.get_logger().info(f"STT time: {STT_time :.4f} seconds")
            # ✅ 음성이 감지되었어도 무음 감지를 다시 시작해야 함
            self.start_silence_monitoring()



    def start_silence_monitoring(self):
        """무음 상태에서 1초마다 경과 시간을 출력하는 스레드 실행"""
        if hasattr(self, 'silence_monitoring_thread') and self.silence_monitoring_thread.is_alive():
            return  # 이미 실행 중이면 중복 실행 방지
        
        self.silence_monitoring_thread = threading.Thread(target=self.monitor_silence,args=(3,), daemon=True)
        self.silence_monitoring_thread.start()



    def monitor_silence(self, silence_threshold):
        """ 3초 이상 무음 상태가 지속되면 강제 Publish 또는 음성 안내 """
        self.silence_seconds = 0  # 무음 지속 시간 초기화

        while self.trigger_detected:
            elapsed_silence = time.time() - self.last_speech_time

            if elapsed_silence >= self.silence_seconds + 1:  # 1초마다 로그 출력
                self.silence_seconds += 1
                self.get_logger().info(f"무음성 {self.silence_seconds}초 경과")

            if elapsed_silence >= silence_threshold:
                # ✅ 이미 퍼블리시된 경우 강제 퍼블리시 방지
                if self.force_published:
                    self.get_logger().info("이미 퍼블리시된 텍스트이므로 강제 퍼블리시 생략")
                    self.force_published = False  # ✅ 플래그 리셋
                    break

                if self.partial_transcript.strip() and self.last_published_text != self.partial_transcript:
                    self.get_logger().info("무음성 3초 경과로 인해 강제 publish")
                    self.publish_transcription(self.partial_transcript)
                    self.last_published_text = self.partial_transcript  # ✅ 중복 방지
                    self.partial_transcript = ""  # ✅ 이전 텍스트 초기화
                else:
                    self.get_logger().info("무음성 3초 경과")
                    self.get_logger().info("말씀하세요! (WAV파일 실행)")
                    self.play_effect_sound_prompt()

                    # ✅ 트리거 상태 유지
                    self.get_logger().info("트리거 감지 상태 유지")
                    self.start_silence_monitoring()  # ✅ 무음 감지를 다시 시작

                break  # ✅ 반복문 종료 후 다시 시작될 수 있도록 설정

            time.sleep(0.1)  # 100ms 단위로 체크하여 정확한 1초 간격 유지



    def play_effect_sound_prompt(self):
        """ ✅ '말씀해주세요.wav' 실행 (사용자에게 말하기 요청) """
        effect_file = "/home/delight/bumblebee_ws/src/pkg_mic/pkg_mic/말씀해주세요.wav"

        try:
            wave_obj = sa.WaveObject.from_wave_file(effect_file)  # ✅ WAV 파일 불러오기
            play_obj = wave_obj.play()  # ✅ 재생 시작
            play_obj.wait_done()  # ✅ 완료될 때까지 대기
        except Exception as e:
            self.get_logger().error(f"Failed to play effect sound: {e}")


    def play_effect_sound(self):
        """ ✅ '멍_편집완료.wav' 실행 (완료 후 음성 녹음 시작) """
        effect_file = "/home/delight/bumblebee_ws/src/pkg_mic/pkg_mic/멍_편집완료.wav"
    
        try:
            wave_obj = sa.WaveObject.from_wave_file(effect_file)  # ✅ WAV 파일 불러오기
            play_obj = wave_obj.play()  # ✅ 재생 시작
            play_obj.wait_done()  # ✅ 완료될 때까지 대기
        except Exception as e:
            self.get_logger().error(f"Failed to play effect sound: {e}")




    def publish_transcription(self, transcript):
        """ STT 결과를 퍼블리시 """
        if transcript.strip():
            # ✅ 강제 퍼블리시 방지를 위한 플래그 설정
            self.force_published = True  
            start_time = time.time()

            msg = String()
            msg.data = transcript.strip()
            self.publisher_.publish(msg)
            self.last_published_text = transcript.strip()  # ✅ 중복 방지용 저장

            end_time = time.time()
            transcription_published_time = end_time - start_time

            self.get_logger().info(f'Transcription published: "{transcript.strip()}"')
            self.save_log(f'Transcription published: "{transcript.strip()}"')
            self.get_logger().info(f"Transcription published time: {transcription_published_time:.4f} seconds")
            self.play_effect_sound_robot()


    def play_effect_sound_robot(self):
        # """ ✅ 'dj스크래치.wav' 실행 (완료 후 음성 녹음 시작) """
        # effect_file = "/home/delight/bumblebee_ws/src/pkg_mic/pkg_mic/dj스크레치.wav"
    
        # try:
        #     wave_obj = sa.WaveObject.from_wave_file(effect_file)  # ✅ WAV 파일 불러오기
        #     play_obj = wave_obj.play()  # ✅ 재생 시작
        #     play_obj.wait_done()  # ✅ 완료될 때까지 대기
        # except Exception as e:
        #     self.get_logger().error(f"Failed to play effect sound: {e}")

        """ ✅ 효과음 디렉토리에서 랜덤 MP3/WAV 효과음 재생 """
        effect_dir = "/home/delight/bumblebee_ws/src/pkg_mic/pkg_mic/effects"  # ✅ 효과음이 저장된 디렉토리
        try:
            # ✅ 효과음 디렉토리에서 mp3 또는 wav 파일만 필터링하여 리스트 생성
            effect_files = [f for f in os.listdir(effect_dir) if f.endswith(('.mp3', '.wav'))]

            if not effect_files:
                self.get_logger().error("❌ 효과음 디렉토리에 MP3 또는 WAV 파일이 없습니다.")
                return

            # ✅ 랜덤으로 하나 선택
            random_effect = random.choice(effect_files)
            effect_path = os.path.join(effect_dir, random_effect)

            # ✅ 효과음 파일 로드 및 재생
            self.get_logger().info(f"🎵 랜덤 효과음 재생: {random_effect}")
            self.save_log(f"🎵 랜덤 효과음 재생: {random_effect}")
            audio = AudioSegment.from_file(effect_path)
            play(audio)

            # ✅ 효과음 길이 확인 후 Fade Out 적용 (마지막 500ms)
            fade_duration = min(500, len(audio))  
            audio = audio.fade_out(fade_duration)

        except Exception as e:
            self.get_logger().error(f"❌ 효과음 재생 중 오류 발생: {e}")


    def save_audio_clip(self):
        """ "안녕" 이후의 오디오를 WAV 파일로 저장 """
        if not self.audio_buffer:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/home/delight/bumblebee_ws/src/pkg_mic/audio_files/{timestamp}.wav"
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
        """ ✅ STT 강제 재시작 (마이크 스트리밍 완전 종료 후 다시 시작) """
        self.get_logger().info("Forcing STT restart...")

        # ✅ STT 세션 강제 종료
        self.stop_audio_stream()
        time.sleep(2)  # ✅ 완전히 닫힐 시간을 확보

        # ✅ 마이크 스트리밍 재시작
        self.start_audio_stream()

        # ✅ STT 스트리밍 강제 재개
        time.sleep(1)  # 마이크 안정화 대기
        self.transcribe_streaming()
    
    def save_log(self, message):
        """ 로그를 파일에 저장 """
        log_file_path = "/home/delight/bumblebee_ws/_logs/UserQuestion_log.txt"
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