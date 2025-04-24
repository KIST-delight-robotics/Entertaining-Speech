import asyncio
import websockets
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

connected_clients = set()

class AudioBridge(Node):
    def __init__(self):
        super().__init__('audio_bridge')
        self.subscription = self.create_subscription(
            String,
            'audio_amplitude',
            self.listener_callback,
            10
        )
        self.loop = asyncio.get_event_loop()
        self.queue = asyncio.Queue()

    def listener_callback(self, msg):
        self.loop.call_soon_threadsafe(self.queue.put_nowait, msg.data)

async def ws_handler(websocket, _path):
    print(f"[✅ 연결됨] {websocket.remote_address}")
    connected_clients.add(websocket)
    try:
        while True:
            msg = await websocket.recv()  # "ping" 수신도 받기
            print(f"[📥 클라이언트로부터 수신]: {msg}")
    except websockets.exceptions.ConnectionClosed:
        print(f"[❌ 연결 종료] {websocket.remote_address}")
    finally:
        connected_clients.remove(websocket)


async def ws_publisher(node):
    while rclpy.ok():
        data = await node.queue.get()
        if connected_clients:
            await asyncio.gather(*[client.send(data) for client in connected_clients])

async def main():
    rclpy.init()
    node = AudioBridge()

    server = await websockets.serve(ws_handler, '0.0.0.0', 8765)
    print("🎧 WebSocket 서버 실행 중 (ws://0.0.0.0:8765)")

    # rclpy와 WebSocket 송신을 함께 처리
    loop = asyncio.get_running_loop()
    loop.create_task(ws_publisher(node))

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("서버 종료됨")
    finally:
        server.close()
        await server.wait_closed()
        rclpy.shutdown()

if __name__ == '__main__':
    asyncio.run(main())
