#jetson search 수정(flask)

from flask import Flask, request, jsonify  # Flask, request 객체와 JSON 응답을 반환하기 위한 모듈 가져오기
import sqlite3  # SQLite와 상호작용하기 위한 모듈

app = Flask(__name__)  # Flask 애플리케이션 생성

# SQLite 데이터베이스 파일 경로 설정
db_path = './mp3_database.db'

# 키워드로 mp3 파일을 검색하는 API 엔드포인트
@app.route('/search', methods=['GET'])
def search_mp3():
    # 클라이언트로부터 GET 요청으로 받은 'keyword' 매개변수 가져오기
    keywords = request.args.get('keyword')
    
    # 키워드가 없으면 오류 응답을 보냄
    if not keywords:
        return jsonify({"error": "Keyword is required"}), 400
    
    # 쉼표로 구분된 키워드를 리스트로 변환
    keywords_list = keywords.split(',')

    # 데이터베이스 연결
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 각 mp3 파일에서 'keywords' 및 'gpt_keywords' 열의 일치하는 키워드 수를 계산
    results = []
    for keyword in keywords_list:
        # 'keywords' 열과 'gpt_keywords' 열에서 키워드를 검색
        cursor.execute("""
            SELECT file_name, file_path, keywords, gpt_keywords 
            FROM mp3_files 
            WHERE keywords LIKE ? OR gpt_keywords LIKE ?
        """, ('%' + keyword.strip() + '%', '%' + keyword.strip() + '%',))
        
        rows = cursor.fetchall()

        # 검색된 파일들의 일치하는 키워드 수 계산 및 결과 저장
        for row in rows:
            file_name, file_path, file_keywords, file_gpt_keywords = row
            
            # 키워드와 GPT 키워드에서 각각 일치하는 키워드 계산
            file_keywords_list = file_keywords.split(',')
            file_gpt_keywords_list = file_gpt_keywords.split(',')

            # 키워드 간 일치하는 개수
            match_keywords_count = len(set(keywords_list) & set(file_keywords_list))
            match_gpt_keywords_count = len(set(keywords_list) & set(file_gpt_keywords_list))

            # 총 일치하는 키워드 수 계산 (둘을 합산)
            total_match_count = match_keywords_count + match_gpt_keywords_count

            results.append({
                'file_name': file_name,
                'file_path': file_path,
                'keywords_match_count': match_keywords_count,
                'gpt_keywords_match_count': match_gpt_keywords_count,
                'total_match_count': total_match_count
            })
    
    conn.close()

    # 매칭된 파일들을 'total_match_count' (키워드 + GPT 키워드)로 정렬하여 반환
    if results:
        results = sorted(results, key=lambda x: x['total_match_count'], reverse=True)
        return jsonify({"files": results}), 200
    else:
        return jsonify({"message": "No mp3 files found"}), 404

if __name__ == '__main__':
    # Flask 서버 실행 (0.0.0.0은 외부 요청도 수락)
    app.run(host='0.0.0.0', port=5000)
