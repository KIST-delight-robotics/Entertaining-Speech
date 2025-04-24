import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pydub import AudioSegment
from pydub.playback import play
import os
import json
import random
from datetime import datetime
import unicodedata
import re
import asyncio
import websockets
import numpy as np  # 위에 추가
import time

class Mp3Player(Node):
    def __init__(self):
        super().__init__("Mp3Player")
        self.file_path = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/mp3_database"
        # self.file_path = "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/movie_database"

        self.effect_dir = "/home/nvidia/ros2_ws/src/pkg_mic/pkg_mic/effects"

        # MP3 추천 파일 구독
        self.subscription_ = self.create_subscription(
            String,
            "recommended_mp3",
            self.mp3_callback,
            10
        )

        # 퍼블리셔 추가 (음악 재생 상태 전송)
        self.publisher_ = self.create_publisher(String, "music_status", 10)
        self.amplitude_publisher_ = self.create_publisher(String, "audio_amplitude", 10)
        self.is_playing = False  # 재생 중 여부 플래그


    def mp3_callback(self, msg):
        """수신된 JSON 추천 MP3 데이터를 파싱하여 리스트로 변환 후 재생"""
        if self.is_playing:
            self.get_logger().warn("이미 음악이 재생 중입니다. 새 요청을 무시합니다.")
            return
        self.is_playing = True




        try:
            # ✅ JSON 없이 Key=Value 문자열을 파싱
            recommended_files = []
            pairs = msg.data.split(";")
            
            for pair in pairs:
                key_value = pair.split("=")
                if len(key_value) == 2:
                    recommended_files.append(key_value[1].strip())

            self.get_logger().info(f"🎵 수신된 추천 MP3 파일들: {recommended_files}")
            self.save_log(f"🎵 수신된 추천 MP3 파일들: {recommended_files}")
            

            # MP3 재생 시작 전 'music_playing' 퍼블리시
            self.publish_music_status("music_playing")

            # MP3 파일들을 순차적으로 재생
            self.play_mp3_list(recommended_files)

            # MP3 재생이 끝나면 'music_done' 퍼블리시
            self.publish_music_status("music_done")

        except Exception as e:
            self.get_logger().error(f"❌ MP3 재생 중 오류 발생: {e}")
            # ✅ 로그 저장
            self.save_log(f"❌ MP3 재생 중 오류 발생: {e}")
        finally:
            self.is_playing = False  # 재생 완료 후 플래그 해제



    def publish_audio_spectrum(self, audio_segment):
        chunk_size_ms = 50
        total_duration = len(audio_segment)
        for i in range(0, total_duration, chunk_size_ms):
            chunk = audio_segment[i:i + chunk_size_ms]
            samples = np.array(chunk.get_array_of_samples())
            if chunk.channels == 2:
                samples = samples.reshape((-1, 2))
                samples = samples.mean(axis=1)
            # FFT (주파수 스펙트럼)
            fft = np.fft.fft(samples)
            spectrum = np.abs(fft[:len(fft)//2])  # 양수 주파수만
            spectrum = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum
            data = {
                "timestamp": i,
                "spectrum": spectrum.tolist()
            }
            msg = String()
            msg.data = json.dumps(data)
            self.amplitude_publisher_.publish(msg)
            time.sleep(chunk_size_ms / 1000.0)



  
    def play_mp3_list(self, files):
        """MP3 파일들을 로드하여 순차적으로 재생"""
        effect_dir = "/home/nvidia/ros2_ws/src/pkg_mic/pkg_mic/effects"  # ✅ 효과음 디렉토리 경로

        try:
            final_audio = AudioSegment.silent(duration=200)
            # ✅ 효과음 디렉토리에서 mp3 또는 wav 파일만 필터링하여 리스트 생성
            effect_files = [f for f in os.listdir(effect_dir) if f.endswith(('.mp3', '.wav'))]

            for i, file_name in enumerate(files):
                file_name = file_name.strip()

                # 절대 경로가 아니면 기본 디렉토리를 추가
                if not os.path.isabs(file_name):
                    file_name = os.path.join(self.file_path, file_name)

                if not os.path.exists(file_name):
                    self.get_logger().error(f"MP3 파일 '{file_name}'이(가) 존재하지 않습니다.")
                    self.save_log(f"MP3 파일 '{file_name}'이(가) 존재하지 않습니다.")
                    continue

                # MP3 파일 로드 및 처리
                audio = AudioSegment.from_file(file_name, format="mp3")
                duration = len(audio)

                #볼륨 정규화 (다른 곡과 음량 차이가 클 경우)
                target_db = -20.0  # 기준 음량 (dBFS)
                change_in_dB = target_db - audio.dBFS
                audio = audio.apply_gain(change_in_dB)  # 음량 정규화

                # ✅ 랜덤 효과음 선택 후 로드
                random_effect = random.choice(effect_files)
                effect_sound_path = os.path.join(effect_dir, random_effect)
                effect_sound = AudioSegment.from_file(effect_sound_path, format="mp3")  # ✅ 효과음 로드

                self.get_logger().info(f"🎵 선택된 랜덤 효과음: {random_effect}")
                self.save_log(f"🎵 선택된 랜덤 효과음: {random_effect}")
                
                # 첫 번째 파일은 바로 추가
                if i == 0:
                    final_audio += audio
                else:
                    # 파일과 파일 사이에 효과음을 추가
                    final_audio += effect_sound + audio

                self.get_logger().info(f"파일 재생 준비 완료: {file_name}, 길이: {duration}ms.")
                self.save_log(f"파일 재생 준비 완료: {file_name}, 길이: {duration}ms.")

            self.get_logger().info("MP3 재생 시작...")
            self.save_log("MP3 재생 시작...")
            # play(final_audio)

            # play(final_audio) 대신 export + aplay
            temp_wav_path = "/tmp/final_audio.wav"
            final_audio.export(temp_wav_path, format="wav")
            #os.system(f"aplay --device=default {temp_wav_path}")

            # ✅ WebSocket 전송 & 재생을 병렬로 실행
            asyncio.run(self.parallel_play_and_stream(final_audio, temp_wav_path))





            self.get_logger().info("MP3 재생 완료.")
            self.save_log("MP3 재생 완료.")


        except Exception as e:
            self.get_logger().error(f"MP3 파일 처리 중 오류 발생: {e}")
            self.save_log(f"MP3 파일 처리 중 오류 발생: {e}")


    async def parallel_play_and_stream(self, audio_segment, wav_path):
        await asyncio.gather(
            asyncio.to_thread(self.publish_audio_spectrum, audio_segment),
            asyncio.to_thread(os.system, f"aplay --device=default {wav_path}")
        )


    def publish_music_status(self, status):
        """음악 재생 상태를 퍼블리시"""
        msg = String()
        msg.data = status
        self.publisher_.publish(msg)
        self.get_logger().info(f"Published music status: {status}")
        self.save_log(f"Published music status: {status}")

    def save_log(self, message):
        """ 로그를 파일에 저장 """
        log_file_path = "/home/nvidia/ros2_ws/_logs/Mp3Player_log.txt"
        log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(log_message)



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
