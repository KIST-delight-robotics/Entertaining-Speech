
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

class Mp3Player(Node):
    def __init__(self):
        super().__init__("Mp3Player")

        # 파일 경로
        self.file_path = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/mp3_database_plus"
        self.reply_path = "/home/nvidia/ros2_ws/src/pkg_spk/pkg_spk/reply.mp3"
        self.api_key = "sk_fdb1ba8706bb125cb308ae613f58105e23e26a89d127a4cd"


        # 구독: 추천된 MP3
        self.subscription_ = self.create_subscription(
            String,
            "recommended_mp3",
            self.mp3_callback,
            10
        )


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




# ──────────────────────────────────────────────────────────────────────────────
# Mp3Recommender에서 수신된 음악 및 TTS 재생
# ──────────────────────────────────────────────────────────────────────────────


    def mp3_callback(self, msg):
        """
        수신된 추천 MP3 (key=value;key=value 형태) 파싱 → 음악 + TTS 재생
        """
        try:
            result_dict = {}
            for pair in msg.data.split(";"):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    result_dict[k.strip()] = v.strip()

            file_path = result_dict.get("file_name", "")
            reply_text = result_dict.get("reply", "")

            if not file_path:
                self.get_logger().warn("파일 경로가 비어 있습니다.")
                return

            # 전체 경로가 아니면 조립
            if not os.path.isabs(file_path):
                file_path = os.path.join(self.file_path, file_path)

            self.get_logger().info(f"🎵 추천 MP3: {file_path}")
            self.get_logger().info(f"💬 Assistant 응답: {reply_text}")
            self.save_log(f"🎵 추천 MP3: {file_path}")
            self.save_log(f"💬 Assistant 응답: {reply_text}")

            # 🎯 TTS 스레드 실행
            tts_thread = threading.Thread(
                target=self.text2speech, args=(reply_text,)
            )
            tts_thread.start()


            # === 🎲 Mp3Player Waiting MP3 랜덤 재생 (스펙트럼 시각화 적용) ===
            waiting_file = self.get_random_waiting_file()
            if waiting_file:
                self.get_logger().info("🎵 Mp3Player Waiting MP3 스펙트럼 재생 시작")
                self.save_log("🎵 Mp3Player Waiting MP3 스펙트럼 재생 시작")
                self.publish_music_status("mp3_waiting_playing")  # 상태명 변경으로 구분
                self.play_waiting_with_spectrum(waiting_file)  # 새로운 함수 사용
            else:
                self.get_logger().warning("⚠️ Waiting 파일을 찾을 수 없어 생략합니다")
                self.save_log("⚠️ Waiting 파일을 찾을 수 없어 생략합니다")




            if self.current_image_path:
                img_msg = String()
                img_msg.data = self.current_image_path
                self.image_publisher_.publish(img_msg)
                self.get_logger().info(f"🖼️ 음악 재생 시작 - 이미지 표시: {self.current_image_path}")
            
            # === 음악 재생 ===
            self.publish_music_status("music_playing")
            self.play_mp3(file_path)
            
            # === 음악 재생 끝 - 이미지 숨김 ===
            empty_img_msg = String()
            empty_img_msg.data = ""
            self.image_publisher_.publish(empty_img_msg)
            self.get_logger().info("🖼️ 음악 재생 끝 - 이미지 숨김")
            
            # === TTS 재생 ===
            tts_thread.join()
            self.play_mp3(self.reply_path)
            self.publish_music_status("music_done")


        except Exception as e:
            error_msg = f"❌ MP3 재생 중 오류 발생: {e}"
            self.get_logger().error(error_msg)
            self.save_log(error_msg)




    

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
        ElevenLabs TTS 호출 → reply.mp3 저장
        """
        api_key = "sk_fdb1ba8706bb125cb308ae613f58105e23e26a89d127a4cd"
        voice_id = "59zWnTQLbwyr94bFbcUe"
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
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                with open(self.reply_path, "wb") as f:
                    f.write(response.content)
                print(f"🟢 음성 변환 성공 → {self.reply_path}")
            else:
                print(f"🔴 TTS 오류 발생: {response.status_code}\n{response.text}")
        except Exception as e:
            print(f"🔴 TTS 호출 실패: {e}")



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
