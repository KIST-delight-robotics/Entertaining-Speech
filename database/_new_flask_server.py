from flask import Flask, request, jsonify
import sqlite3
import time

app = Flask(__name__)

# SQLite 데이터베이스 파일 경로 설정
db_path = './mp3_database.db'

# 키워드로 mp3 파일을 검색하는 API 엔드포인트
@app.route('/search', methods=['GET'])
def search_mp3_files():
    try:
        # GET 요청으로 받은 'l_keywords' 및 's_keywords' 매개변수 가져오기
        l_keywords_param = request.args.get('l_keywords')
        s_keywords_param = request.args.get('s_keywords')
        
        if not l_keywords_param or not s_keywords_param:
            return jsonify({"error": "Both l_keywords and s_keywords are required"}), 400
        
        # 검색 시작 시간 기록
        search_start_time = time.time()

        # 쉼표로 구분된 키워드를 리스트로 변환
        l_keywords_list = [kw.strip() for kw in l_keywords_param.split(',') if kw.strip()]
        s_keywords_list = [kw.strip() for kw in s_keywords_param.split(',') if kw.strip()]
        
        # 데이터베이스 연결
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        search_results = []
        keyword_times = []  # 각 키워드 검색 소요 시간을 저장

        print("Starting keyword searches...", flush=True)

        # 'okt_keywords' 열에서 l_keywords_list에 있는 키워드 검색
        for l_keyword in l_keywords_list:
            # 각 l_keywords 키워드 검색 시간 측정
            keyword_start_time = time.time()
            cursor.execute("""
                SELECT file_name, file_path, okt_keywords, gpt_keywords 
                FROM mp3_files 
                WHERE okt_keywords LIKE ?
            """, ('%' + l_keyword + '%',))
            
            rows = cursor.fetchall()
            keyword_end_time = time.time()
            keyword_time = keyword_end_time - keyword_start_time
            keyword_times.append({'keyword': l_keyword, 'search_time': f"{keyword_time:.4f} seconds"})
            print(f"l Keyword '{l_keyword}' search time: {keyword_time:.4f} seconds", flush=True)

            for row in rows:
                file_name, file_path, file_okt_keywords, file_gpt_keywords = row
                file_okt_keywords_list = [kw.strip() for kw in file_okt_keywords.split(',') if kw.strip()]
                
                # 매칭된 l_keywords들만 선택
                matched_okt_keywords = list(set(l_keywords_list) & set(file_okt_keywords_list))
                
                # 'gpt_keywords' 열에서 s_keywords_list에 있는 키워드 검색
                matched_s_keywords = []
                s_keyword_match_count = 0

                if file_gpt_keywords:
                    # 각 s_keywords 키워드 검색 시간 측정
                    s_keyword_start_time = time.time()
                    file_gpt_keywords_list = [kw.strip() for kw in file_gpt_keywords.split(',') if kw.strip()]
                    matched_s_keywords = list(set(s_keywords_list) & set(file_gpt_keywords_list))
                    s_keyword_match_count = len(matched_s_keywords)
                    s_keyword_end_time = time.time()
                    s_keyword_time = s_keyword_end_time - s_keyword_start_time
                    keyword_times.append({'keyword': "GPT_MATCH", 'search_time': f"{s_keyword_time:.4f} seconds"})
                    print(f"s Keyword matching search time: {s_keyword_time:.4f} seconds", flush=True)

                # 총 일치도 계산
                okt_keyword_match_count = len(matched_okt_keywords)
                total_match_count = okt_keyword_match_count + s_keyword_match_count

                if total_match_count > 0:
                    search_results.append({
                        'file_name': file_name,
                        'file_path': file_path,
                        'l_keywords_match_count': okt_keyword_match_count,
                        's_keywords_match_count': s_keyword_match_count,
                        'total_match_count': total_match_count,
                        'matched_l_keywords': matched_okt_keywords,
                        'matched_s_keywords': matched_s_keywords
                    })

        conn.close()

        # 검색 끝 시간 기록
        search_end_time = time.time()

        # 검색 시간 계산
        total_search_time = search_end_time - search_start_time

        # 터미널에 총 검색 시간 출력
        print(f"Total search time: {total_search_time:.2f} seconds", flush=True)

        # total_match_count로 정렬하여 반환
        if search_results:
            search_results = sorted(search_results, key=lambda x: x['total_match_count'], reverse=True)
            return jsonify({
                "files": search_results,
                "timing": {
                    "keyword_search_times": keyword_times,
                    "total_search_time": f"{total_search_time:.2f} seconds"
                }
            }), 200
        else:
            return jsonify({
                "message": "No mp3 files found",
                "timing": {
                    "keyword_search_times": keyword_times,
                    "total_search_time": f"{total_search_time:.2f} seconds"
                }
            }), 404

    except Exception as e:
        print(f"Unexpected error: {e}", flush=True)
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == '__main__':
    # Flask 서버 실행 (0.0.0.0은 외부 요청도 수락)
    app.run(host='0.0.0.0', port=5000)
