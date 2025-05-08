
#통합
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

import wave
import pyaudio

class Mp3Player(Node):
    def __init__(self):
        super().__init__("Mp3Player")

        # 파일 경로
        self.file_path = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/mp3_database_plus"
        self.reply_path = "/home/nvidia/ros2_ws/src/pkg_spk/pkg_spk/reply.mp3"
        self.api_key = "sk_fdb1ba8706bb125cb308ae613f58105e23e26a89d127a4cd"
        self.voice_id = "dtu2KmDq4zRNfRVuhajI"

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

            # 🎵 음악 먼저 재생
            self.publish_music_status("music_playing")
            self.play_mp3(file_path)

            # 🎧 TTS 재생
            tts_thread.join()
            self.play_mp3(self.reply_path)
            self.publish_music_status("music_done")

        except Exception as e:
            error_msg = f"❌ MP3 재생 중 오류 발생: {e}"
            self.get_logger().error(error_msg)
            self.save_log(error_msg)

    # def play_mp3(self, file_path):
    #     """
    #     단일 MP3 파일 재생 (정규화 포함)
    #     """
    #     try:
    #         sound = AudioSegment.from_file(file_path, format="mp3")
    #         sound = self.match_target_amplitude(sound, -14.0)
    #         play(sound)
    #     except Exception as e:
    #         self.get_logger().error(f"❌ MP3 재생 실패: {file_path} → {e}")
    #         self.save_log(f"❌ MP3 재생 실패: {file_path} → {e}")



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

    # def play_mp3(self, file_path):
    #     """
    #     단일 MP3 파일 재생 (정규화 포함) - aplay 사용
    #     """
    #     try:
    #         sound = AudioSegment.from_file(file_path, format="mp3")
    #         self.publish_audio_spectrum(sound)
    #         sound = self.match_target_amplitude(sound, -14.0)

    #         # 임시 wav 파일로 저장
    #         temp_wav = "/tmp/temp_audio.wav"
    #         sound.export(temp_wav, format="wav")

    #         # # 시스템 명령어로 재생
    #         os.system(f"aplay {temp_wav}")
            

    #     except Exception as e:
    #         self.get_logger().error(f"❌ MP3 재생 실패: {file_path} → {e}")
    #         self.save_log(f"❌ MP3 재생 실패: {file_path} → {e}")


    # def play_mp3(self, file_path):
    #     try:
    #         sound = AudioSegment.from_file(file_path, format="mp3")
    #         sound = self.match_target_amplitude(sound, -14.0)
    #         temp_wav = "/tmp/temp_audio.wav"
    #         sound.export(temp_wav, format="wav")
    #         # os.system(f"aplay -D hw3,0 {temp_wav}")



    #         wf = wave.open(temp_wav, 'rb')
    #         p = pyaudio.PyAudio()
    #         chunk_size = 1024

    #         stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
    #                         channels=wf.getnchannels(),
    #                         rate=wf.getframerate(),
    #                         output=True,
    #                         output_device_index=26
    #                         )

    #         data = wf.readframes(chunk_size)
    #         while data:
    #             stream.write(data)
    #             # 스펙트럼 퍼블리시
    #             samples = np.frombuffer(data, dtype=np.int16)
    #             if wf.getnchannels() == 2:
    #                 samples = samples.reshape((-1, 2)).mean(axis=1)
    #             fft = np.fft.fft(samples)
    #             spectrum = np.abs(fft[:len(fft)//2])
    #             spectrum = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum
    #             msg = String()
    #             msg.data = json.dumps({"spectrum": spectrum.tolist()})
    #             self.amplitude_publisher_.publish(msg)
    #             data = wf.readframes(chunk_size)

    #         stream.stop_stream()
    #         stream.close()
    #         p.terminate()
    #         wf.close()
    #     except Exception as e:
    #         self.get_logger().error(f"❌ MP3 재생 실패: {file_path} → {e}")
    #         self.save_log(f"❌ MP3 재생 실패: {file_path} → {e}")

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
        chunk_size = 1024

        def publish_spectrum():
            data = wf.readframes(chunk_size)
            while data:
                samples = np.frombuffer(data, dtype=np.int16)
                if wf.getnchannels() == 2:
                    samples = samples.reshape((-1, 2)).mean(axis=1)
                fft = np.fft.fft(samples)
                spectrum = np.abs(fft[:len(fft)//2])
                spectrum = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum
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
        wf.close()



    def text2speech(self, text):
        """
        ElevenLabs TTS 호출 → reply.mp3 저장
        """
        api_key = "sk_fdb1ba8706bb125cb308ae613f58105e23e26a89d127a4cd"
        voice_id = "dtu2KmDq4zRNfRVuhajI"
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
                "stability": 0.94,
                "similarity_boost": 0.96,
                "style": 0.10
            }
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

    def match_target_amplitude(self, sound, target_dBFS):
        """
        주어진 오디오를 타깃 dBFS로 정규화
        """
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)

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
