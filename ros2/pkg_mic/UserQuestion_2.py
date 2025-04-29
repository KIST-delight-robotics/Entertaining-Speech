
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
import nemo.collections.asr as nemo_asr  # ★ NeMo

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from google.cloud import speech
from dotenv import load_dotenv
import pygame

load_dotenv("/home/nvidia/ros2_ws/src/.env")

class NeMoEmbedder:
    """Wraps NVIDIA NeMo speaker verification model → 256‑d embeddings."""

    def __init__(self, device: str):
        self.device = device
        # 전화품질(telephony) ECAPA‑TDNN – 256‑d
        self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            #"speakerverification_speakernet"
            #"ecapa_tdnn"
            "titanet_large"
        ).to(device)
        self.model.eval()

    def __call__(self, pcm16: np.ndarray, sr: int = 16000) -> np.ndarray:
        # Normalize PCM16 to float32 in range [-1.0, 1.0]
        audio_float32 = pcm16.astype(np.float32) / 32768.0

        # Write to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav:
            sf.write(tmp_wav.name, audio_float32, sr)
            tmp_wav.flush()

            # Extract embedding
            with torch.inference_mode():
                emb = self.model.get_embedding(tmp_wav.name)  # (1, 256)

        emb = emb.squeeze(0).cpu().numpy()
        return emb / (np.linalg.norm(emb) + 1e-9)  # (256,)

class RealTimeDiarizer:
    """NeMo ECAPA + NumPy cosine search with sticky IDs."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        window_dur: float = 2.0,
        hop_dur: float = 1.0,
        thr_same: float = 0.85,
        thr_new: float = 0.40,
        switch_needed: int = 3,
        update_alpha: float = 0.02,
        vad_level: int = 2,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = NeMoEmbedder(self.device)

        self.sr = sample_rate
        self.win = int(window_dur * sample_rate)
        self.hop = int(hop_dur * sample_rate)
        self.vad = webrtcvad.Vad(vad_level) if vad_level >= 0 else None

        # Speaker DB
        self.cents: list[np.ndarray] = []
        self.ids: list[int] = []
        self.next_id = 1

        # Params
        self.TS = thr_same
        self.TN = thr_new
        self.switch_needed = switch_needed
        self.update_alpha = update_alpha

        # Runtime state
        self.active_id: Optional[int] = None
        self.cand_id: Optional[int] = None
        self.cand_cnt = 0
        self.buf = np.zeros((0,), dtype=np.int16)
        self.lock = threading.Lock()

    # ── helpers ────────────────────────────────────────────────
    def _has_voice(self, pcm: bytes) -> bool:
        if not self.vad:
            return True
        flen = int(0.03 * self.sr) * 2
        if len(pcm) < flen:
            return False
        step = (len(pcm) - flen) // 2 or flen
        return any(self.vad.is_speech(pcm[i : i + flen], self.sr) for i in (0, step, 2 * step))

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def _update_centroid(self, idx: int, emb: np.ndarray):
        self.cents[idx] = (
            (1 - self.update_alpha) * self.cents[idx] + self.update_alpha * emb
        )
        self.cents[idx] /= np.linalg.norm(self.cents[idx]) + 1e-9

    # ── streaming API ──────────────────────────────────────────
    def accept_audio(self, pcm: bytes) -> Optional[int]:
        pcm16 = np.frombuffer(pcm, dtype=np.int16)
        self.buf = np.concatenate([self.buf, pcm16])
        if len(self.buf) < self.win:
            return self.active_id
        frame = self.buf[: self.win]
        self.buf = self.buf[self.hop :]
        if self.vad and not self._has_voice(frame.tobytes()):
            return self.active_id

        emb = self.embedder(frame)

        with self.lock:
            if not self.cents:
                self.cents.append(emb.copy()); self.ids.append(self.next_id)
                self.active_id = self.next_id; self.next_id += 1
                print(f"[DIAR] first speaker → ID {self.active_id}")
                return self.active_id
            sims = [self._cos(emb, c) for c in self.cents]
            best_idx = int(np.argmax(sims)); best_sim = sims[best_idx]
            best_id = self.ids[best_idx]
            print(f"[DIAR] sim={best_sim:.3f} best_id={best_id} active={self.active_id} cand={self.cand_id}/{self.cand_cnt}")
            if best_sim >= self.TS:
                self._update_centroid(best_idx, emb)
                if best_id == self.active_id:
                    self.cand_id = None; self.cand_cnt = 0
                else:
                    if self.cand_id == best_id:
                        self.cand_cnt += 1
                    else:
                        self.cand_id = best_id; self.cand_cnt = 1
                    if self.cand_cnt >= self.switch_needed:
                        print(f"[DIAR] >>> ID switch {self.active_id} → {best_id}")
                        self.active_id = best_id; self.cand_id = None; self.cand_cnt = 0
                return self.active_id
            if best_sim < self.TN:
                return self.active_id  # ignore low‑sim
            return self.active_id      # ambiguous

    # ── clip identification ───────────────────────────────────
    def identify_speaker(self, pcm: bytes) -> Optional[int]:
        emb = self.embedder(np.frombuffer(pcm, dtype=np.int16))
        with self.lock:
            sims = [self._cos(emb, c) for c in self.cents]
            if not sims:
                self.cents.append(emb.copy()); self.ids.append(self.next_id)
                cid = self.next_id; self.next_id += 1; self.active_id = cid
                return cid
            best_idx = int(np.argmax(sims)); best_sim = sims[best_idx]
            best_id = self.ids[best_idx]
            if best_sim >= self.TS:
                self.active_id = best_id; return best_id
            if best_sim < self.TN:
                self.cents.append(emb.copy()); self.ids.append(self.next_id)
                cid = self.next_id; self.next_id += 1; self.active_id = cid
                return cid
            self.active_id = best_id; return best_id

            
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

        # ROS 2 인터페이스
        stt_group = ReentrantCallbackGroup()
        self.publisher_ = self.create_publisher(String, "user_question", 10)

        # self.create_subscription(
        #     String, "processing_done", self.processing_done_callback, 10
        # )
        self.processing_subscription = self.create_subscription(String, "processing_done", self.processing_done_callback, 10)
        self.music_status_subscription = self.create_subscription(String, "music_status", self.music_status_callback, 10)



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
        
        # 화자 식별기
        self.spkr = RealTimeDiarizer(
            device="cpu",      # or "cpu"
            thr_same=0.72,      # 같은 화자 판정
            thr_new=0.30,       # 새 화자 판정
            switch_needed=3     # 동일 후보 3프레임(≈2.25 s) 연속일 때만 ID 스위치
        )
        self.current_speaker_id = 0

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
        self.start_audio_stream()
        self.visualizer_pub = self.create_publisher(String, "audio_visualizer", 10)

    # ── Google STT -----------------------------------------------------------

    
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
        self.timer_30s = threading.Timer(5, self.timer_30s_expired)
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
            diarization_config=diar_cfg,
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

        # 1) 화자 식별
        spk = self.spkr.accept_audio(in_data)
        if spk is not None:
            self.current_speaker_id = spk

        # 2) STT 큐에 넣기 (음악 재생 중이거나 ignore 플래그일 땐 제외)
        if not (self.music_playing or self.ignore_stt):
            self.audio_stream.put(in_data)
            if self.trigger_detected:
                self.audio_buffer.append(in_data)

        return None, pyaudio.paContinue


        # ✅ "안녕" 감지 후 음성 데이터를 버퍼에 저장
        if self.trigger_detected:
            self.audio_buffer.append(in_data)
            # === 실시간 오디오 시각화 데이터 publish ===
            self.publish_audio_visualizer(in_data)

        return None, pyaudio.paContinue


    def publish_audio_visualizer(self, in_data):
        samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
        # FFT (스펙트럼)
        fft = np.fft.fft(samples)
        spectrum = np.abs(fft[:len(fft)//2])
        spectrum = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum
        data = {
            "spectrum": spectrum.tolist()
        }
        msg = String()
        msg.data = json.dumps(data)
        self.visualizer_pub.publish(msg)

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
                    self.last_speech_time = time.time()
                    self.silence_seconds = 0

                    # 음악 종료 후 최초 음성 입력이 들어왔을 때만 무음 감지 시작
                    if self.waiting_for_input_after_music:
                        self.waiting_for_input_after_music = False  # 최초 입력 감지 완료
                        self.get_logger().info("🎤 음악 종료 후 최초 입력 감지됨. 무음 체크 시작.")
                        self.start_silence_monitoring()

                self.get_logger().info(f'Transcript: {txt} (Final: {is_final})')

                # ── 1) trigger 감지 시 ──
                if not self.trigger_detected:
                    if "안녕" in txt:
                        split_text = txt.split("안녕", 1)
                        if len(split_text) > 1:
                            self.partial_transcript = split_text[1].strip()
                            self.get_logger().info(f"Trigger detected. Capturing transcript: {self.partial_transcript}")
                            self.play_effect_sound()
                            self.trigger_detected = True
                            self.audio_buffer = []  # 본 질문 음성 버퍼링 시작
                            self.current_speaker_id = 0  # 임시 speaker_id
                            self.start_silence_monitoring()
                        continue

                # ── 2) trigger 이후 본 질문 저장 ──
                elif self.trigger_detected:
                    if "안녕" in txt:
                        split_text = txt.split("안녕", 1)
                        if len(split_text) > 1:
                            self.partial_transcript = split_text[1].strip()
                    else:
                        self.partial_transcript = txt

                # ── 3) 무음 3초 후 퍼블리시 시점 ──
                if is_final and self.partial_transcript.strip():
                    # 본 질문 음성 → 화자 식별 진행 (identify_speaker로 변경)
                    try:
                        spk_audio = b''.join(self.audio_buffer)
                        self.get_logger().info(f"🔍 spk_audio length: {len(spk_audio)} bytes")  # ✅ 길이 확인

                        speaker_id = self.spkr.identify_speaker(spk_audio)  # 🔥 identify_speaker 사용
                        if speaker_id is not None:
                            self.current_speaker_id = speaker_id
                            self.get_logger().info(f"🎙️ Identified speaker_id: {speaker_id}")
                        else:
                            self.current_speaker_id = 0  # fallback
                            self.get_logger().info("❌ Speaker identification failed, fallback to 0")

                        # 퍼블리시
                        self.publish_transcription(self.partial_transcript)
                        self.save_audio_clip()

                        # 화자 식별기 버퍼 초기화
                        self.spkr.buffer = np.zeros((0,), dtype="int16")  # 🔥 추가
                        return
                    except Exception as e:
                        self.get_logger().error(f"Speaker identification error: {e}")
                        # 퍼블리시는 진행
                        self.publish_transcription(self.partial_transcript)
                        self.save_audio_clip()
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




    def play_effect_sound(self):
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

    # ── Audio callback -------------------------------------------------------

    # audio_q: queue.Queue = queue.Queue()

    # def audio_callback(self, in_data, frame_count, time_info, status):
    #     # 화자 식별 업데이트
    #     for spk, _ in self.spkr.accept_audio(in_data):
    #         self.current_speaker_id = spk
    #     # STT 스트림에 푸시
    #     self.audio_q.put(in_data)
    #     return None, pyaudio.paContinue

    # ── ROS 퍼블리시 ---------------------------------------------------------

    def publish_transcription(self, text: str):
        # msg = String()
        # msg.data = f"speaker{self.current_speaker_id:03d}|{text}"
        # self.publisher_.publish(msg)
        # self.get_logger().info(f"🗣 {msg.data}")



        if text.strip():
            if self.timer_30s and self.timer_30s.is_alive():
                self.timer_30s.cancel()  # ✅ 퍼블리시 후 타이머 종료

            self.force_published = True

            msg = String()
            msg.data = f"speaker{self.current_speaker_id:03d}|{text}"
            self.publisher_.publish(msg)
            self.last_published_text = msg.data

            self.get_logger().info(f'Transcription published: "{msg.data}"')
            self.save_log(f'Transcription published: "{msg.data}"')
            time.sleep(2)
            self.play_effect_sound_rag()

            self.partial_transcript = ""  # ✅ 퍼블리시 후 즉시 초기화
            self.trigger_detected = False  # ✅ 퍼블리시 후 trigger 상태 초기화
            self.waiting_for_input_after_music = False  # ✅ 입력 대기 상태 해제

    def play_effect_sound_rag(self):
        # 효과음 파일이 저장된 디렉토리 경로
        effects_dir = "/home/nvidia/ros2_ws/src/pkg_mic/pkg_mic/_tts_rag"

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
        
        # 재생이 끝날 때까지 대기
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)



    def save_audio_clip(self):
        """ "안녕" 이후의 오디오를 WAV 파일로 저장 """
        if not self.audio_buffer:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/home/nvidia/ros2_ws/audio_files/{timestamp}.wav"
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
