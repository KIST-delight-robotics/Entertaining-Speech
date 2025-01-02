from flask import Flask, request, jsonify
from aiortc import RTCPeerConnection, RTCSessionDescription
import sqlite3

app = Flask(__name__)
db_path = './mp3_database.db'  # SQLite 데이터베이스 경로
stun_servers = [{"urls": "stun:stun.l.google.com:19302"}]  # STUN 서버 설정

# 키워드로 mp3 파일을 검색하는 API 엔드포인트
@app.route('/search', methods=['GET'])
def search_mp3():
    keywords = request.args.get('keyword')
    gpt_keywords = request.args.get('gpt_keyword')
    
    if not keywords and not gpt_keywords:
        return jsonify({"error": "At least one of 'keyword' or 'gpt_keyword' is required"}), 400

    keywords_list = keywords.split(',') if keywords else []
    gpt_keywords_list = gpt_keywords.split(',') if gpt_keywords else []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    results = []
    # 모든 키워드에 대해 검색하여 일치도 계산
    for keyword in keywords_list + gpt_keywords_list:
        cursor.execute("""
            SELECT file_name, file_path, keywords, gpt_keywords 
            FROM mp3_files 
            WHERE keywords LIKE ? OR gpt_keywords LIKE ?
        """, ('%' + keyword.strip() + '%', '%' + keyword.strip() + '%'))
        
        rows = cursor.fetchall()
        for row in rows:
            file_name, file_path, file_keywords, file_gpt_keywords = row

            # `keywords_list`와 `gpt_keywords_list`와의 일치도 계산
            file_keywords_list = file_keywords.split(',')
            file_gpt_keywords_list = file_gpt_keywords.split(',')

            matched_keywords = set(keywords_list) & set(file_keywords_list)
            matched_gpt_keywords = set(gpt_keywords_list) & set(file_gpt_keywords_list)

            match_keywords_count = len(matched_keywords)
            match_gpt_keywords_count = len(matched_gpt_keywords)
            total_match_count = match_keywords_count + match_gpt_keywords_count

            results.append({
                'file_name': file_name,
                'file_path': file_path,
                'keywords_match_count': match_keywords_count,
                'gpt_keywords_match_count': match_gpt_keywords_count,
                'total_match_count': total_match_count
            })

    conn.close()

    # `total_match_count`로 정렬하여 반환
    if results:
        results = sorted(results, key=lambda x: x['total_match_count'], reverse=True)
        return jsonify({"files": results}), 200
    else:
        return jsonify({"message": "No mp3 files found"}), 404

# WebRTC Offer를 받아 Answer 생성 및 반환
@app.route("/offer", methods=["POST"])
async def offer():
    data = request.json
    offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])

    pc = RTCPeerConnection(configuration={"iceServers": stun_servers})
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return jsonify({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)