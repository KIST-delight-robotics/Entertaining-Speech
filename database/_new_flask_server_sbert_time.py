from flask import Flask, request, jsonify
import sqlite3
from sentence_transformers import SentenceTransformer, util
import time

app = Flask(__name__)

# SQLite 데이터베이스 파일 경로 설정
db_path = './mp3_database.db'

# SBERT 모델 로드
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 키워드로 mp3 파일을 검색하는 API 엔드포인트
@app.route('/search', methods=['GET'])
def search_mp3_files():
    try:
        start_time = time.time()  # 총 검색 시간 측정 시작

        # GET 요청으로 받은 'okt_keywords', 'gpt_keywords', 'gpt_response_text' 매개변수 가져오기
        okt_keywords_param = request.args.get('okt_keywords')
        gpt_keywords_param = request.args.get('gpt_keywords')
        gpt_response_text = request.args.get('gpt_response_text')
        
        if not okt_keywords_param or not gpt_keywords_param or not gpt_response_text:
            return jsonify({"error": "Both okt_keywords, gpt_keywords, and gpt_response_text are required"}), 400
        
        # 쉼표로 구분된 키워드를 리스트로 변환
        okt_keywords_list = okt_keywords_param.split(',')
        gpt_keywords_list = gpt_keywords_param.split(',')
        
        # 데이터베이스 연결 및 검색
        db_start_time = time.time()  # 데이터베이스 검색 시작 시간
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        filtered_results = []
        
        # 'okt_keywords' 열에서 okt_keywords_list에 있는 키워드 검색
        for okt_keyword in okt_keywords_list:
            cursor.execute("""
                SELECT file_name, file_path, okt_keywords, gpt_keywords 
                FROM mp3_files 
                WHERE okt_keywords LIKE ?
            """, ('%' + okt_keyword.strip() + '%',))
            
            rows = cursor.fetchall()
            for row in rows:
                file_name, file_path, file_okt_keywords, file_gpt_keywords = row
                file_okt_keywords_list = file_okt_keywords.split(',')
                
                # 매칭된 okt 키워드들만 선택
                matched_okt_keywords = list(set(okt_keywords_list) & set(file_okt_keywords_list))
                
                # 'gpt_keywords' 열에서 gpt_keywords_list에 있는 키워드 검색
                matched_gpt_keywords = []
                gpt_keyword_match_count = 0
                if file_gpt_keywords:
                    file_gpt_keywords_list = file_gpt_keywords.split(',')
                    matched_gpt_keywords = list(set(gpt_keywords_list) & set(file_gpt_keywords_list))
                    gpt_keyword_match_count = len(matched_gpt_keywords)

                # 총 일치도 계산
                okt_keyword_match_count = len(matched_okt_keywords)
                total_match_count = okt_keyword_match_count + gpt_keyword_match_count

                # 매치 개수가 1 이상인 경우만 필터링
                if total_match_count > 1:
                    filtered_results.append({
                        'file_name': file_name,
                        'file_path': file_path,
                        'okt_keywords_match_count': okt_keyword_match_count,
                        'gpt_keywords_match_count': gpt_keyword_match_count,
                        'total_match_count': total_match_count,
                        'matched_okt_keywords': matched_okt_keywords,
                        'matched_gpt_keywords': matched_gpt_keywords
                    })
        
        conn.close()
        db_time = time.time() - db_start_time  # 데이터베이스 검색 시간 계산
        
        # SBERT를 사용하여 문맥 유사도 비교
        if filtered_results:
            sbert_start_time = time.time()  # SBERT 검색 시작 시간
            gpt_embedding = model.encode(gpt_response_text, convert_to_tensor=True)
            
            for result in filtered_results:
                file_embedding = model.encode(result['file_name'], convert_to_tensor=True)
                similarity_score = util.cos_sim(gpt_embedding, file_embedding).item()
                result['sbert_similarity'] = similarity_score

            # SBERT 유사도 기준으로 정렬
            filtered_results = sorted(filtered_results, key=lambda x: x['sbert_similarity'], reverse=True)[:10]
            sbert_time = time.time() - sbert_start_time  # SBERT 검색 시간 계산

            total_time = time.time() - start_time  # 총 검색 시간 계산
            
            return jsonify({
                "files": filtered_results,
                "timing": {
                    "db_search_time": f"{db_time:.2f} seconds",
                    "sbert_search_time": f"{sbert_time:.2f} seconds",
                    "total_search_time": f"{total_time:.2f} seconds"
                }
            }), 200
        else:
            total_time = time.time() - start_time  # 총 검색 시간 계산
            return jsonify({
                "message": "No mp3 files found",
                "timing": {
                    "db_search_time": f"{db_time:.2f} seconds",
                    "sbert_search_time": "0.00 seconds",
                    "total_search_time": f"{total_time:.2f} seconds"
                }
            }), 404

    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == '__main__':
    # Flask 서버 실행 (0.0.0.0은 외부 요청도 수락)
    app.run(host='0.0.0.0', port=6000)
