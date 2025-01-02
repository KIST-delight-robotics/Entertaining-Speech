import socketio
import sqlite3
import asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription
from flask import Flask, request, jsonify

# Flask 서버 초기화
app = Flask(__name__)
sio = socketio.AsyncClient()
pc = RTCPeerConnection()

# SQLite 데이터베이스 경로
db_path = './mp3_database.db'

# Signaling 서버 연결
@sio.event
async def connect():
    print("Connected to signaling server")
    await sio.emit('join_room', {'room': 'my_room'})  # 특정 방에 참여

# Offer 수신 및 Answer 전송
@sio.on('offer')
async def on_offer(data):
    offer = RTCSessionDescription(sdp=data, type='offer')
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    await sio.emit('answer', {'room': 'my_room', 'answer': answer.sdp})

# 두 키워드를 수신하고 DB에서 검색
@sio.on('keywords')
async def on_keywords(data):
    okt_keywords = data['okt_keywords']
    gpt_keywords = data['gpt_keywords']
    print(f"Received OKT Keywords: {okt_keywords}")
    print(f"Received GPT Keywords: {gpt_keywords}")

    # SQLite DB에서 okt_keywords와 gpt_keywords로 검색
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT file_name, file_path 
        FROM mp3_files 
        WHERE okt_keywords LIKE ? AND gpt_keywords LIKE ?
    """, ('%' + okt_keywords + '%', '%' + gpt_keywords + '%'))
    
    rows = cursor.fetchall()
    conn.close()

    # 검색된 mp3 파일 경로 출력
    if rows:
        print(f"Found mp3 files: {[row[1] for row in rows]}")
    else:
        print("No matching mp3 files found")

# Flask API 서버 - mp3 파일 검색
@app.route('/search', methods=['GET'])
def search_mp3():
    keyword = request.args.get('keyword')
    if not keyword:
        return jsonify({"error": "Keyword is required"}), 400

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT file_name, file_path FROM mp3_files WHERE okt_keywords LIKE ?", ('%' + keyword + '%',))
    results = [{"file_name": row[0], "file_path": row[1]} for row in cursor.fetchall()]
    conn.close()

    if results:
        return jsonify({"files": results}), 200
    else:
        return jsonify({"message": "No mp3 files found"}), 404

# Signaling 서버와 Flask 서버 동시 실행
async def main():
    await sio.connect('http://localhost:3000')
    app.run(host='0.0.0.0', port=5000)

    # 세션 종료
    await sio.disconnect()
    await pc.close()

# asyncio로 비동기 Flask와 signaling 통신 동시 실행
asyncio.run(main())
