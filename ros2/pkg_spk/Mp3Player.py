
#동적자막(google-stt 이용) + 원본 텍스트 기반 자막 생성


import os
import requests
import threading
from datetime import datetime
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pydub import AudioSegment
from pydub.playback import play
import asyncio
import numpy as np
import json
import time
import websockets
import random  

import wave
import pyaudio
import csv
from google.cloud import speech
import io
import os
import re


class Mp3Player(Node):
    def __init__(self):
        super().__init__("Mp3Player")

        # 파일 경로
        self.file_path = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/mp3_database_plus"
        self.reply_path = "/home/nvidia/ros2_ws/src/pkg_spk/pkg_spk/reply.mp3"
        self.api_key = "sk_fdb1ba8706bb125cb308ae613f58105e23e26a89d127a4cd"


        # # 구독: 추천된 MP3
        # self.subscription_ = self.create_subscription(
        #     String,
        #     "recommended_mp3",
        #     self.mp3_callback,
        #     10
        # )


        # 퍼블리시: 음악 재생 상태
        self.publisher_ = self.create_publisher(String, "music_status", 10)
        self.amplitude_publisher_ = self.create_publisher(String, "audio_amplitude", 10)
        self.is_playing = False  # 재생 중 여부 플래그



        self.current_image_path = None
        self.image_subscription_ = self.create_subscription(
            String, "recommended_image", self.image_callback, 10
        )
        self.image_publisher_ = self.create_publisher(String, "current_music_image", 10)
        # 기존 publisher들 다음에 추가
        self.mp3_waiting_spectrum_pub = self.create_publisher(String, "/mp3_waiting_spectrum", 10)


        # 🆕 TTS 요청 구독 추가
        self.tts_subscription = self.create_subscription(
            String,
            "tts_request",
            self.tts_request_callback,
            10
        )
        
        # 🆕 TTS 완료 상태 퍼블리시 추가
        self.tts_status_publisher = self.create_publisher(String, "tts_status", 10)

        # TTS 재생 요청 구독
        self.tts_play_subscription = self.create_subscription(
            String,
            "tts_play_request",
            self.tts_play_request_callback,
            10
        )


        # 🆕 TTS 전용 스펙트럼 퍼블리셔 추가
        self.tts_spectrum_publisher = self.create_publisher(String, "/tts_spectrum", 10)



        # 🆕 TTS 자막 데이터 퍼블리셔 추가
        self.tts_subtitle_publisher = self.create_publisher(String, "/tts_subtitle", 10)
        
        # Google Cloud 인증 설정
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/nvidia/ros2_ws/my-service-account.json'

        # 🆕 STT 재시작 신호 퍼블리셔 추가 (music_done 대신)
        self.stt_restart_publisher = self.create_publisher(String, "stt_restart", 10)



# ──────────────────────────────────────────────────────────────────────────────
# 단어별 타임스탬프 추출 함수 추가
# ──────────────────────────────────────────────────────────────────────────────



    def extract_word_timestamps(self, audio_path, original_text):
        """
        Google Cloud Speech-to-Text API를 사용하여 단어별 타임스탬프 추출
        """
        try:
            client = speech.SpeechClient()
            
            with io.open(audio_path, "rb") as audio_file:
                content = audio_file.read()
            
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="ko-KR",
                enable_word_time_offsets=True,
                enable_word_confidence=True,
                model="default",
            )
            
            self.get_logger().info("🗣️ Google STT API 요청 시작...")
            response = client.recognize(config=config, audio=audio)
            
            word_timestamps = []
            
            for result in response.results:
                alternative = result.alternatives[0]
                stt_text = alternative.transcript
                self.get_logger().info(f"🗣️ STT 인식 결과: '{stt_text}'")
                self.get_logger().info(f"🔍 원본 텍스트: '{original_text}'")
                
                for word_info in alternative.words:
                    word = word_info.word
                    start_time = word_info.start_time.total_seconds()
                    end_time = word_info.end_time.total_seconds()
                    confidence = word_info.confidence
                    
                    word_timestamps.append({
                        "word": word,
                        "start": round(start_time, 3),
                        "end": round(end_time, 3),
                        "confidence": round(confidence, 3)
                    })
            
            self.get_logger().info(f"🎯 STT 타임스탬프 추출 완료: {len(word_timestamps)}개 단어")
            return word_timestamps
            
        except Exception as e:
            self.get_logger().error(f"❌ STT 타임스탬프 추출 실패: {e}")
            return []







    

# ──────────────────────────────────────────────────────────────────────────────
# 대기효과음3 재생 및 스펙트럼 퍼블리시
# ──────────────────────────────────────────────────────────────────────────────



    def get_random_waiting_file(self):
        """
        waiting_3 디렉토리에서 랜덤한 MP3 파일 경로 반환
        """
        waiting_dir = "/home/nvidia/ros2_ws/src/pkg_mic/pkg_mic/_tts_waiting3"
        
        try:
            if not os.path.exists(waiting_dir):
                self.get_logger().warning(f"Waiting 디렉토리가 존재하지 않습니다: {waiting_dir}")
                return None
                
            # mp3 파일만 필터링
            mp3_files = [f for f in os.listdir(waiting_dir) if f.lower().endswith('.mp3')]
            
            if not mp3_files:
                self.get_logger().warning(f"Waiting 디렉토리에 MP3 파일이 없습니다: {waiting_dir}")
                return None
                
            # 랜덤 선택
            selected_file = random.choice(mp3_files)
            full_path = os.path.join(waiting_dir, selected_file)
            
            self.get_logger().info(f"🎲 랜덤 waiting 파일 선택: {selected_file}")
            self.save_log(f"🎲 랜덤 waiting 파일 선택: {selected_file}")
            
            return full_path
            
        except Exception as e:
            error_msg = f"❌ Waiting 파일 선택 중 오류: {e}"
            self.get_logger().error(error_msg)
            self.save_log(error_msg)
            return None




    def play_waiting_with_spectrum(self, file_path):
        """waiting 파일을 스펙트럼 시각화와 함께 재생 (UserQuestion.py와 동일한 방식)"""
        try:
            # pydub으로 MP3 로드 및 정규화 (UserQuestion.py와 동일)
            sound = AudioSegment.from_file(file_path, format="mp3")
            
            # # 정규화 (UserQuestion.py와 동일)
            # target_dBFS = -14.0
            # change_in_dBFS = target_dBFS - sound.dBFS
            # sound = sound.apply_gain(change_in_dBFS)
            
            # 임시 WAV로 변환
            temp_wav = "/tmp/mp3_waiting_audio.wav"
            sound.export(temp_wav, format="wav")

            self.get_logger().info(f"🎵 Mp3Player Waiting 파일 스펙트럼 재생: {file_path}")
            self.save_log(f"🎵 Mp3Player Waiting 파일 스펙트럼 재생: {file_path}")

            # UserQuestion.py와 동일한 방식으로 스펙트럼과 재생 병렬 처리
            self.mp3_waiting_publish_and_play(temp_wav)

            time.sleep(0.5)

            self.get_logger().info("🎵 Mp3Player Waiting 스펙트럼 재생 완료")
            self.save_log("🎵 Mp3Player Waiting 스펙트럼 재생 완료")

        except Exception as e:
            error_msg = f"❌ Mp3Player Waiting 스펙트럼 재생 실패: {e}"
            self.get_logger().error(error_msg)
            self.save_log(error_msg)




    def mp3_waiting_publish_and_play(self, wav_path):
        """Mp3Player.py 전용 waiting 스펙트럼 시각화 (UserQuestion.py 방식과 동일)"""

        
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
                self.mp3_waiting_spectrum_pub.publish(msg)  # 별도 토픽 사용
                data = wf.readframes(chunk_size)
                time.sleep(chunk_size / wf.getframerate())


        spectrum_thread = threading.Thread(target=publish_spectrum)
        spectrum_thread.start()

        # 시스템 명령어로 재생 (기존 방식과 동일)
        os.system(f"aplay {wav_path}")
        spectrum_thread.join()


        




 

# ──────────────────────────────────────────────────────────────────────────────
# 음악 재생 및 이미지 퍼블리시
# ──────────────────────────────────────────────────────────────────────────────


    def image_callback(self, msg):
        """이미지 파일명 수신 및 저장"""
        if msg.data and msg.data.strip():
            self.current_image_path = f"/images/{msg.data}"  # 웹 경로로 변환
            # 즉시 퍼블리시 (사전 로딩 효과)
            img_msg = String()
            img_msg.data = self.current_image_path
            self.image_publisher_.publish(img_msg)
            self.get_logger().info(f"이미지 수신: {self.current_image_path}")
        else:
            self.current_image_path = None
            self.get_logger().info("이미지 수신: 없음")




    def match_target_amplitude(self, sound, target_dBFS):
        """
        주어진 오디오를 타깃 dBFS로 정규화
        """
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)




    def play_mp3(self, file_path):
        try:
            sound = AudioSegment.from_file(file_path, format="mp3")
            sound = self.match_target_amplitude(sound, -14.0)
            
            # 임시 WAV로 변환 후 저장
            temp_wav = "/tmp/temp_audio.wav"
            sound.export(temp_wav, format="wav")

            # 스펙트럼과 재생 병렬로 실행
            playback_thread = threading.Thread(target=self.publish_and_play, args=(temp_wav,))
            playback_thread.start()
            playback_thread.join()

        except Exception as e:
            self.get_logger().error(f"❌ MP3 재생 실패: {file_path} → {e}")
            self.save_log(f"❌ MP3 재생 실패: {file_path} → {e}")



    def publish_and_play(self, wav_path):
        wf = wave.open(wav_path, 'rb')
        chunk_size = 2024


        # 2) CSV 파일 한 번 열어두기 (append 모드)
        csv_file = open('spectrum.csv', 'a', newline='')
        csv_writer = csv.writer(csv_file)



        def publish_spectrum():
            data = wf.readframes(chunk_size)
            while data:
                samples = np.frombuffer(data, dtype=np.int16)
                if wf.getnchannels() == 2:
                    samples = samples.reshape((-1, 2)).mean(axis=1)
                fft = np.fft.fft(samples)
                spectrum = np.abs(fft[:len(fft)//2])
                #spectrum = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum


                #────────────────────────────────────────────────
                # # 7) CSV에 한 열로 실시간 저장
                # #    - spectrum 값들
                # for mag in spectrum:
                #     csv_writer.writerow([mag])
                # #    - 이번 청크의 평균·최소·최대 (한 번 저장할 때 마지막에)
                # mean_val = spectrum.mean()
                # min_val  = spectrum.min()
                # max_val  = spectrum.max()
                # csv_writer.writerow([f"최소값: {min_val:.6f}"])
                # csv_writer.writerow([f"최대값: {max_val:.6f}"])
                # csv_writer.writerow([f"평균값: {mean_val:.6f}"])

                # #    - 파일에 바로 반영
                # csv_file.flush()

              
                msg = String()
                msg.data = json.dumps({"spectrum": spectrum.tolist()})
                self.amplitude_publisher_.publish(msg)
                data = wf.readframes(chunk_size)
                time.sleep(chunk_size / wf.getframerate())

        spectrum_thread = threading.Thread(target=publish_spectrum)
        spectrum_thread.start()

        # 시스템 명령어 aplay로 재생 (즉각적인 출력)
        os.system(f"aplay {wav_path}")  # 🔥실제 출력장치로 변경(카드2)
        spectrum_thread.join()
        csv_file.close()
        wf.close()


     

# ──────────────────────────────────────────────────────────────────────────────
# TTS 재생 및 스펙트럼 퍼블리시
# ──────────────────────────────────────────────────────────────────────────────



    def text2speech(self, text):
        """
        ElevenLabs TTS 호출 → reply.mp3 저장 → 원본 텍스트 기반 자막 생성
        """
        api_key = "sk_fdb1ba8706bb125cb308ae613f58105e23e26a89d127a4cd"
        voice_id = "2oCsvoTtWZkaDZUSExSz"
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json; charset=utf-8",  # UTF-8 명시
            "Accept": "audio/mpeg"
        }

        # 🆕 텍스트 전처리
        cleaned_text = text.strip()
        cleaned_text = ' '.join(cleaned_text.split())
        
        # 🆕 디버깅 로그
        self.get_logger().info(f"🗣️ TTS 원본 텍스트: '{cleaned_text}'")

        data = {
            "text": cleaned_text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.95,
                "similarity_boost": 0.6,
                "style": 0.4,
                "speed": 0.8
            },
            "apply_text_normalization": "on"
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(data, ensure_ascii=False).encode('utf-8')
            )
            
            if response.status_code == 200:
                with open(self.reply_path, "wb") as f:
                    f.write(response.content)
                
                file_size = os.path.getsize(self.reply_path)
                self.get_logger().info(f"🟢 음성 변환 성공 → {self.reply_path} ({file_size} bytes)")
                
                # 🆕 WAV 변환 (STT API용)
                sound = AudioSegment.from_file(self.reply_path, format="mp3")
                wav_path = "/tmp/tts_for_stt.wav"
                sound = sound.set_frame_rate(16000).set_channels(1)
                sound.export(wav_path, format="wav")
                
                # 🆕 STT 타임스탬프 추출
                stt_timestamps = self.extract_word_timestamps(wav_path, cleaned_text)
                
                # 🔑 핵심: 원본 텍스트로 덮어쓰기
                corrected_timestamps = self.merge_original_with_stt_timestamps(
                    original_text=cleaned_text,
                    stt_timestamps=stt_timestamps
                )
                
                # 🆕 수정된 자막 데이터 퍼블리시
                if corrected_timestamps:
                    subtitle_data = {
                        "original_text": cleaned_text,
                        "words": corrected_timestamps,
                        "total_duration": corrected_timestamps[-1]["end"] if corrected_timestamps else 0
                    }
                    
                    msg = String()
                    msg.data = json.dumps(subtitle_data, ensure_ascii=False)
                    self.tts_subtitle_publisher.publish(msg)
                    self.get_logger().info(f"📝 수정된 자막 데이터 퍼블리시: {len(corrected_timestamps)}개 단어")
                else:
                    self.get_logger().warning("⚠️ 자막 수정 실패 - 기본 자막 사용")
                    self._publish_fallback_subtitle(cleaned_text)
                    
            else:
                self.get_logger().error(f"🔴 TTS 오류: {response.status_code}")
                self.get_logger().error(f"응답: {response.text}")
        except Exception as e:
            self.get_logger().error(f"🔴 TTS 호출 실패: {e}")




    def align_words_with_dynamic_programming(self, original_words, stt_timestamps):
        """
        동적 계획법과 음성학적 유사도를 활용한 정교한 단어 정렬
        """
        from difflib import SequenceMatcher
        import re
        
        # 전처리: 구두점 제거 및 정규화
        def normalize_word(word):
            return re.sub(r'[^\w]', '', word).lower()
        
        orig_normalized = [normalize_word(w) for w in original_words]
        stt_words = [ts["word"].lower() for ts in stt_timestamps]
        
        # 🆕 SequenceMatcher를 사용한 최적 정렬
        matcher = SequenceMatcher(None, orig_normalized, stt_words)
        opcodes = matcher.get_opcodes()
        
        aligned_results = []
        current_stt_time = 0.0
        
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'equal':
                # 정확히 매칭된 단어들
                for k in range(i2 - i1):
                    orig_idx = i1 + k
                    stt_idx = j1 + k
                    
                    aligned_results.append({
                        "word": original_words[orig_idx],
                        "start": stt_timestamps[stt_idx]["start"],
                        "end": stt_timestamps[stt_idx]["end"],
                        "confidence": stt_timestamps[stt_idx]["confidence"],
                        "match_type": "exact"
                    })
                    current_stt_time = stt_timestamps[stt_idx]["end"]
            
            elif tag == 'delete':
                # STT가 놓친 원본 단어들 - 시간 보간
                self._interpolate_missing_words(
                    original_words[i1:i2], aligned_results, current_stt_time
                )
            
            elif tag == 'insert':
                # STT에만 있는 단어들 - 시간만 진행
                if j2 <= len(stt_timestamps):
                    current_stt_time = stt_timestamps[j2-1]["end"]
            
            elif tag == 'replace':
                # 다른 단어들 - 음성학적 유사도로 매핑
                self._map_mismatched_words(
                    original_words[i1:i2], stt_timestamps[j1:j2], aligned_results
                )
        
        return aligned_results

    def _interpolate_missing_words(self, missing_words, aligned_results, start_time):
        """
        누락된 단어들의 타이밍 보간
        """
        base_duration = 0.4  # 기본 단어 지속시간
        
        for i, word in enumerate(missing_words):
            # 단어 길이에 따른 지속시간 조정
            word_length = len(re.sub(r'[^\w]', '', word))
            duration = max(0.2, min(0.8, word_length * 0.1))
            
            word_start = start_time + (i * base_duration)
            word_end = word_start + duration
            
            aligned_results.append({
                "word": word,
                "start": word_start,
                "end": word_end,
                "confidence": 0.3,  # 낮은 신뢰도
                "match_type": "interpolated"
            })

    def _map_mismatched_words(self, orig_words, stt_words, aligned_results):
        """
        불일치 단어들의 음성학적 유사도 기반 매핑
        """
        if not stt_words:
            return
            
        # 전체 STT 시간 범위
        total_duration = stt_words[-1]["end"] - stt_words[0]["start"]
        time_per_word = total_duration / len(orig_words)
        
        for i, orig_word in enumerate(orig_words):
            word_start = stt_words[0]["start"] + (i * time_per_word)
            word_end = word_start + time_per_word
            
            # 🆕 STT 단어와의 유사도 계산
            best_confidence = 0.0
            for stt_word in stt_words:
                similarity = self._calculate_phonetic_similarity(
                    orig_word, stt_word["word"]
                )
                best_confidence = max(best_confidence, similarity)
            
            aligned_results.append({
                "word": orig_word,
                "start": word_start,
                "end": word_end,
                "confidence": best_confidence,
                "match_type": "phonetic_match"
            })

    def _calculate_phonetic_similarity(self, word1, word2):
        """
        음성학적 유사도 계산 (한글 특화)
        """
        from difflib import SequenceMatcher
        
        # 1. 편집 거리 기반 유사도
        similarity = SequenceMatcher(None, word1.lower(), word2.lower()).ratio()
        
        # 2. 한글 자모 분해 유사도 (선택적)
        # 여기서는 단순화하여 편집 거리만 사용
        
        return similarity * 0.8  # 신뢰도 조정
    

    def smooth_timestamps(self, timestamps):
        """
        타이밍 스무딩 및 겹침 해결
        """
        if not timestamps:
            return timestamps
        
        smoothed = []
        
        for i, ts in enumerate(timestamps):
            current = ts.copy()
            
            # 이전 단어와 겹침 방지
            if i > 0:
                prev_end = smoothed[-1]["end"]
                if current["start"] < prev_end:
                    current["start"] = prev_end + 0.05  # 50ms 간격
            
            # 너무 짧은 지속시간 조정
            min_duration = 0.15  # 최소 150ms
            if (current["end"] - current["start"]) < min_duration:
                current["end"] = current["start"] + min_duration
            
            # 다음 단어와의 간격 조정
            if i < len(timestamps) - 1:
                next_start = timestamps[i + 1]["start"]
                if current["end"] > next_start:
                    # 시간을 균등 분할
                    mid_time = (current["start"] + next_start) / 2
                    current["end"] = mid_time
            
            smoothed.append(current)
        
        return smoothed






    def merge_original_with_stt_timestamps(self, original_text, stt_timestamps):
        """
        🆕 개선된 단어 정렬 및 타이밍 매핑
        """
        try:
            original_words = original_text.split()
            
            if not stt_timestamps:
                return self._create_default_timestamps(original_words)
            
            # 🆕 지능적 정렬 수행
            aligned_timestamps = self.align_words_with_dynamic_programming(
                original_words, stt_timestamps
            )
            
            # 🆕 타이밍 후처리
            smoothed_timestamps = self.smooth_timestamps(aligned_timestamps)
            
            # 디버깅 로그
            self.get_logger().info(f"🎯 정렬 결과: {len(original_words)}개 원본 → {len(smoothed_timestamps)}개 자막")
            
            for i, ts in enumerate(smoothed_timestamps[:5]):  # 처음 5개만 로깅
                self.get_logger().info(f"  [{i}] '{ts['word']}': {ts['start']:.2f}s-{ts['end']:.2f}s (신뢰도: {ts['confidence']:.2f}, 매칭: {ts['match_type']})")
            
            return smoothed_timestamps
            
        except Exception as e:
            self.get_logger().error(f"❌ 고급 정렬 실패: {e}")
            return stt_timestamps  # 실패시 원본 STT 결과 반환







    def _distribute_extra_words(self, original_words, stt_timestamps):
        """
        원본 단어 수가 STT보다 많을 때 시간을 분할하여 배치 (구두점 포함)
        """
        corrected_timestamps = []
        
        if not stt_timestamps:
            return self._create_default_timestamps(original_words)
        
        # 🆕 구두점이 있는 단어는 조금 더 짧은 시간 할당
        total_duration = stt_timestamps[-1]["end"] - stt_timestamps[0]["start"]
        
        # 🆕 단어별 가중치 계산 (구두점 있는 단어는 더 짧게)
        word_weights = []
        for word in original_words:
            import re
            clean_length = len(re.sub(r'[^\w]', '', word))
            if clean_length == 0:  # 구두점만 있는 경우
                weight = 0.1
            else:
                weight = max(0.3, clean_length * 0.2)  # 최소 0.3초, 글자당 0.2초
            word_weights.append(weight)
        
        total_weight = sum(word_weights)
        start_time = stt_timestamps[0]["start"]
        
        for i, (word, weight) in enumerate(zip(original_words, word_weights)):
            duration = (weight / total_weight) * total_duration
            word_start = start_time
            word_end = word_start + duration
            
            corrected_timestamps.append({
                "word": word,  # 🎯 구두점 포함
                "start": round(word_start, 3),
                "end": round(word_end, 3),
                "confidence": 0.9
            })
            
            start_time = word_end  # 다음 단어의 시작점
        
        return corrected_timestamps

    def _create_default_timestamps(self, words):
        """
        STT 실패시 기본 타임스탬프 생성 (구두점 포함)
        """
        timestamps = []
        current_time = 0.0
        
        for word in words:
            import re
            # 🆕 구두점 포함 단어의 길이에 따라 시간 조정
            clean_length = len(re.sub(r'[^\w]', '', word))
            
            if clean_length == 0:  # 구두점만 있는 경우 (예: ".")
                duration = 0.1
            elif len(word) <= 2:  # 짧은 단어
                duration = 0.3
            elif len(word) <= 5:  # 중간 길이 단어
                duration = 0.5
            else:  # 긴 단어
                duration = 0.7
            
            start_time = current_time
            end_time = start_time + duration
            
            timestamps.append({
                "word": word,  # 🎯 구두점 포함
                "start": round(start_time, 3),
                "end": round(end_time, 3),
                "confidence": 1.0
            })
            
            current_time = end_time
        
        return timestamps


    def _publish_fallback_subtitle(self, text):
        """
        폴백 자막 데이터 퍼블리시
        """
        fallback_data = {
            "original_text": text,
            "words": [{"word": text, "start": 0, "end": 3, "confidence": 1.0}],
            "total_duration": 3
        }
        
        msg = String()
        msg.data = json.dumps(fallback_data, ensure_ascii=False)
        self.tts_subtitle_publisher.publish(msg)







# ──────────────────────────────────────────────────────────────────────────────
# 재생 상태 및 로그 상태 저장
# ──────────────────────────────────────────────────────────────────────────────


    def publish_music_status(self, status):
        """
        음악 재생 상태 퍼블리시
        """
        msg = String()
        msg.data = status
        self.publisher_.publish(msg)
        self.get_logger().info(f"📡 음악 상태: {status}")
        self.save_log(f"📡 음악 상태: {status}")

    def save_log(self, message):
        """
        로그 파일에 저장
        """
        log_file_path = "/home/nvidia/ros2_ws/_logs/Mp3Player_log.txt"
        log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(log_message)
# ──────────────────────────────────────────────────────────────────────────────
# reply TTS 코드
# ──────────────────────────────────────────────────────────────────────────────
    def tts_request_callback(self, msg):
        """TTS 요청 수신 및 처리"""
        try:
            reply_text = msg.data
            self.get_logger().info(f"🗣️ TTS 요청 수신: {reply_text}")
            self.save_log(f"🗣️ TTS 요청 수신: {reply_text}")
            
            # TTS 생성 시작 신호
            self.publish_tts_status("tts_generating")
            
            # TTS 생성
            self.text2speech(reply_text)
            
            # TTS 준비 완료 신호
            self.publish_tts_status("tts_ready")
            
        except Exception as e:
            error_msg = f"❌ TTS 생성 중 오류: {e}"
            self.get_logger().error(error_msg)
            self.save_log(error_msg)
            self.publish_tts_status("tts_error")



    def play_tts_audio(self):
        """TTS 오디오 재생 (App.jsx에서 요청시) - 스펙트럼 포함"""
        try:
            self.get_logger().info("🎵 TTS 오디오 재생 시작")
            self.save_log("🎵 TTS 오디오 재생 시작")
            
            # 재생 시작 신호
            self.publish_tts_status("tts_playing")
            
            # 🆕 TTS 전용 스펙트럼과 함께 재생
            self.play_tts_with_spectrum(self.reply_path)
            
            # 재생 완료 신호
            self.publish_tts_status("tts_done")
            
            self.get_logger().info("🎵 TTS 오디오 재생 완료")
            self.save_log("🎵 TTS 오디오 재생 완료")
            
        except Exception as e:
            error_msg = f"❌ TTS 재생 중 오류: {e}"
            self.get_logger().error(error_msg)
            self.save_log(error_msg)
            self.publish_tts_status("tts_error")

    def play_tts_with_spectrum(self, file_path):
        """TTS 전용 전체 음량 기반 스펙트럼과 함께 재생"""
        try:
            sound = AudioSegment.from_file(file_path, format="mp3")
            sound = self.match_target_amplitude(sound, -14.0)
            
            # 임시 WAV로 변환 후 저장
            temp_wav = "/tmp/tts_audio.wav"
            sound.export(temp_wav, format="wav")

            # TTS 전용 스펙트럼과 재생 병렬로 실행
            playback_thread = threading.Thread(target=self.tts_publish_and_play, args=(temp_wav,))
            playback_thread.start()
            playback_thread.join()

        except Exception as e:
            self.get_logger().error(f"❌ TTS 스펙트럼 재생 실패: {file_path} → {e}")
            self.save_log(f"❌ TTS 스펙트럼 재생 실패: {file_path} → {e}")

    # def tts_publish_and_play(self, wav_path):
    #     """TTS 전용 전체 음량 기반 스펙트럼 퍼블리시 및 재생"""
    #     wf = wave.open(wav_path, 'rb')
    #     chunk_size = 2024

    #     def publish_tts_volume():
    #         data = wf.readframes(chunk_size)
    #         while data:
    #             samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    #             if wf.getnchannels() == 2:
    #                 samples = samples.reshape((-1, 2)).mean(axis=1)
                
    #             # 🆕 전체 음량 계산 (RMS - Root Mean Square)
    #             rms = np.sqrt(np.mean(samples**2))
                
    #             # 🆕 FFT로 주파수 분석 (참고용, 전체 에너지만 사용)
    #             fft = np.fft.fft(samples)
    #             magnitude_spectrum = np.abs(fft[:len(fft)//2])
                
    #             # 🆕 전체 에너지 합계 (모든 주파수 대역의 에너지 합)
    #             total_energy = np.sum(magnitude_spectrum)
                
    #             # 🆕 정규화된 전체 음량 (0~1 범위)
    #             normalized_volume = min(1.0, rms / 32768.0 * 10)  # int16 최대값으로 정규화
    #             normalized_energy = min(1.0, total_energy / 1000000)  # 적절한 범위로 정규화

    #             # 🆕 TTS 전용 데이터 (RMS와 전체 에너지를 모두 전송)
    #             tts_data = {
    #                 "volume": float(normalized_volume),
    #                 "energy": float(normalized_energy),
    #                 "rms": float(rms)
    #             }

    #             msg = String()
    #             msg.data = json.dumps(tts_data)
    #             self.tts_spectrum_publisher.publish(msg)
                
    #             data = wf.readframes(chunk_size)
    #             time.sleep(chunk_size / wf.getframerate())

    #     spectrum_thread = threading.Thread(target=publish_tts_volume)
    #     spectrum_thread.start()

    #     # 시스템 명령어로 재생
    #     os.system(f"aplay {wav_path}")
    #     spectrum_thread.join()
    #     wf.close()





    def tts_publish_and_play(self, wav_path):
        """TTS 재생 시간 정보 퍼블리시 및 재생"""
        wf = wave.open(wav_path, 'rb')
        chunk_size = 2024
        frame_rate = wf.getframerate()
        start_time = time.time()

        def publish_tts_time():
            data = wf.readframes(chunk_size)
            while data:
                # 🆕 현재 재생 시간 계산
                current_time = round(time.time() - start_time, 3)
                
                # 🆕 재생 시간 정보 전송 (RMS 대신)
                time_data = {
                    "current_time": current_time,
                    "status": "playing",
                    "timestamp": time.time()
                }

                msg = String()
                msg.data = json.dumps(time_data)
                self.tts_spectrum_publisher.publish(msg)  # 기존 퍼블리셔 재활용
                
                data = wf.readframes(chunk_size)
                time.sleep(chunk_size / frame_rate)
            
            # 🆕 재생 완료 신호
            final_data = {
                "current_time": current_time,
                "status": "finished",
                "timestamp": time.time()
            }
            msg = String()
            msg.data = json.dumps(final_data)
            self.tts_spectrum_publisher.publish(msg)

        time_thread = threading.Thread(target=publish_tts_time)
        time_thread.start()

        # 시스템 명령어로 재생
        os.system(f"aplay {wav_path}")
        time_thread.join()
        wf.close()








    def publish_tts_status(self, status):
        """TTS 상태 퍼블리시"""
        msg = String()
        msg.data = status
        self.tts_status_publisher.publish(msg)
        self.get_logger().info(f"📡 TTS 상태: {status}")
        self.save_log(f"📡 TTS 상태: {status}")


        # # 🆕 TTS 완료 시 UserQuestion에게 STT 재시작 신호 전송
        # if status == "tts_done":
        #     restart_msg = String()
        #     restart_msg.data = "restart_stt_after_tts"
        #     self.stt_restart_publisher.publish(restart_msg)
        #     self.get_logger().info("📡 TTS 완료 - UserQuestion STT 재시작 신호 전송")
        #     self.save_log("📡 TTS 완료 - UserQuestion STT 재시작 신호 전송")

            


    


    def tts_play_request_callback(self, msg):
        """TTS 재생 요청 처리"""
        if msg.data == "play_tts":
            self.play_tts_audio()



def main(args=None):
    rclpy.init(args=args)
    node = Mp3Player()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
