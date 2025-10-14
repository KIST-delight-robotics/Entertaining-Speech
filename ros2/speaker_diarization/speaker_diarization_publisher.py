
import rclpy
from rclpy.node import Node
from speaker_diarization_msgs.msg import SpeakerInfo
import subprocess
import threading
import signal
import sys
import os
import time
import selectors


class SpeakerDiarizationPublisher(Node):
    def __init__(self):
        super().__init__('speaker_diarization_publisher')
        self.publisher_ = self.create_publisher(SpeakerInfo, 'speaker_diarization', 10)
        self.process = None
        self.output_thread = None
        self.running = True

        signal.signal(signal.SIGINT, self.signal_handler)

        self.start_diart_client()
        self.get_logger().info('Speaker Diarization Publisher 시작됨')
        self.get_logger().info('화자 분리 결과를 /speaker_diarization 토픽으로 퍼블리시합니다')

    def signal_handler(self, signum, frame):
        self.get_logger().info('프로그램 종료 중...')
        self.running = False
        if self.process:
            self.process.terminate()
        sys.exit(0)

    def start_diart_client(self):
        cmd = [
            'diart.client', 'microphone', 
            '--host', 'localhost',
            '--port', '7007',
       
       
        ]
        try:
            self.get_logger().info(f'diart 클라이언트 시작: {" ".join(cmd)}')
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            env['PYTHONIOENCODING'] = 'utf-8'
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
               
                text=True,         # 텍스트 모드
                bufsize=1,         # 라인 버퍼링
                stdin=subprocess.DEVNULL,
                env=env,
                preexec_fn=os.setsid
            )
            self.output_thread = threading.Thread(target=self.publish_output_realtime, daemon=True)
            self.output_thread.start()
            self.get_logger().info('diart 클라이언트 프로세스가 시작되었습니다')
        except Exception as e:
            self.get_logger().error(f'diart 클라이언트 시작 실패: {e}')





    def publish_output_realtime(self):
        sel = selectors.DefaultSelector()
        sel.register(self.process.stdout, selectors.EVENT_READ)
        sel.register(self.process.stderr, selectors.EVENT_READ)

        while self.running:
            if self.process.poll() is not None:
                self.get_logger().warn('diart 프로세스가 종료되었습니다')
                break

            for key, _ in sel.select(timeout=0.1):
                data = key.fileobj.readline()
                if not data:
                    continue
                txt = data.strip()
                if not txt:
                    continue
                if key.fileobj is self.process.stderr:
                    self.get_logger().error(f'diart stderr: {txt}')
                else:
                    self.publish_line(txt)




    def publish_line(self, line):
        self.get_logger().info(f'🔄 diart 출력: {line}')
        try:
            msg = SpeakerInfo()
            msg.message_type = 'RAW'
            msg.host_port = ''
            msg.channel_id = 0
            msg.timestamp = 0.0
            msg.duration = 0.0
            msg.speaker_id = ''
            msg.metadata = [line]
            self.publisher_.publish(msg)
        except Exception as e:
            self.get_logger().error(f'퍼블리시 실패: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = SpeakerDiarizationPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
