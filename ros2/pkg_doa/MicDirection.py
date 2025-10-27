#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import sys
import os
import usb.core
import usb.util
import time
# from tuning import Tuning
from pkg_doa.usb_4_mic_array.tuning import Tuning
import usb.core
import usb.util
import time

# # tuning 모듈 경로 추가 (import 전에 먼저 실행)
# current_dir = os.path.dirname(os.path.abspath(__file__))
# usb_4_mic_array_path = os.path.join(current_dir, 'usb_4_mic_array')
# sys.path.append(usb_4_mic_array_path)

# # 경로 추가 후에 import
# try:
#     from tuning import Tuning
#     print("tuning module imported successfully")
# except ImportError as e:
#     print(f"Failed to import tuning module: {e}")

class MicDirection(Node):
    def __init__(self):
        super().__init__('MicDirection')
        
        # ReSpeaker 디바이스 찾기
        self.dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
        
        if self.dev:
            try:
                self.mic_tuning = Tuning(self.dev)
                self.get_logger().info("ReSpeaker device found and initialized")
                
                # Publisher 생성
                self.angle_publisher = self.create_publisher(Float32, '/sound_direction_angle', 10)
                
                # 타이머 생성 (1초 간격)
                self.timer = self.create_timer(1.0, self.publish_direction)
                
                self.get_logger().info("MicDirection Node started - Publishing to /sound_direction_angle")
            except Exception as e:
                self.get_logger().error(f"Failed to initialize tuning: {e}")
        else:
            self.get_logger().error("ReSpeaker device not found!")
    
    def publish_direction(self):
        if self.dev and hasattr(self, 'mic_tuning'):
            try:
                # 방향 정보 읽기
                direction = self.mic_tuning.direction
                
                # Float32 메시지 생성
                angle_msg = Float32()
                angle_msg.data = float(direction)
                
                # 퍼블리시
                self.angle_publisher.publish(angle_msg)
                
                # 로그 출력
                self.get_logger().info(f"Sound direction: {direction} degrees")
                
            except Exception as e:
                self.get_logger().error(f"Error reading direction: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    node = MicDirection()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down MicDirection Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
