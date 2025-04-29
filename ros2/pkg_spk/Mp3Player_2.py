import os
import requests
import threading
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pydub import AudioSegment
from pydub.playback import play


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


    def play_mp3(self, file_path):
        """
        단일 MP3 파일 재생 (정규화 포함) - aplay 사용
        """
        try:
            sound = AudioSegment.from_file(file_path, format="mp3")
            sound = self.match_target_amplitude(sound, -14.0)

            # 임시 wav 파일로 저장
            temp_wav = "/tmp/temp_audio.wav"
            sound.export(temp_wav, format="wav")

            # 시스템 명령어로 재생
            os.system(f"aplay {temp_wav}")
            
        except Exception as e:
            self.get_logger().error(f"❌ MP3 재생 실패: {file_path} → {e}")
            self.save_log(f"❌ MP3 재생 실패: {file_path} → {e}")


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
