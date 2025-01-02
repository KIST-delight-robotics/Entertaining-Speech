import os
import re
import sqlite3
import openai
from konlpy.tag import Okt
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time


# GPT API Key 설정
openai.api_key = "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"  


# 데이터베이스 연결 및 테이블 확인 후 필요 시 생성
def setup_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 테이블이 없으면 생성
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS mp3_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT,
        file_path TEXT,
        okt_keywords TEXT,
        gpt_keywords TEXT
    )
    ''')

    # 'gpt_keywords' 열이 있는지 확인하고, 없으면 추가
    cursor.execute("PRAGMA table_info(mp3_files);")
    columns = [col[1] for col in cursor.fetchall()]
    if 'gpt_keywords' not in columns:
        cursor.execute("ALTER TABLE mp3_files ADD COLUMN gpt_keywords TEXT")
        print("'gpt_keywords' 열이 추가되었습니다.")

    conn.commit()
    conn.close()

# 형태소 분석기 객체 생성
okt = Okt()

# 파일명에서 okt_keywords를 추출하는 함수 (모든 명사 및 동사 어근 추출)
def extract_okt_keywords(file_name):
    # 형태소 분석 결과 중 명사와 동사의 어간만 추출
    okt_keywords = []
    morphs = okt.pos(file_name, stem=True)  # 형태소 분석, 동사 어간 추출
    for word, tag in morphs:
        if tag in ['Noun', 'Verb']:  # 명사와 동사(어간만)
            okt_keywords.append(word)

    # 영어 단어 추출 (정규 표현식 사용)
    english_words = re.findall(r'[a-zA-Z]+', file_name)
    okt_keywords.extend(english_words)
        
    return ', '.join(okt_keywords)  # 쉼표로 구분하여 반환

# GPT에게 텍스트 파일 내용을 보내고 gpt_keywords를 추출하는 함수
def ask_gpt_with_text_file(user_input):
    try:
        # GPT API 호출
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "상황을 제시하면 그에 맞는 적절한 대답을 하면 돼. A와 B는 친구야. 예를 들면, A : 오늘 눈이 올까? B : '내일 아침에 하얀 눈이 쌓여 있었음 해요.' 이런식으로 B의 대답은 기존의 음악이나 영화, 드라마의 한 문장이야. 예시를 고려해서 유추된 A의 문장 속 상황 키워드(명사형) 5개만 ,로 구분해서 제시해줘."},
                {"role": "assistant", "content": "user가 B의 대답을 제시할거야."},
                {"role": "user", "content": user_input}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"GPT API 호출 중 오류 발생: {e}")
        return None  # API 호출에 실패한 경우

# mp3 파일을 데이터베이스에 추가하는 함수 (okt_keywords 및 gpt_keywords 포함)
def insert_mp3_to_db(db_path, mp3_file, folder_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    file_name = os.path.splitext(mp3_file)[0]  # 파일명에서 확장자 제거
    file_path = os.path.join(folder_path, mp3_file)

    # 형태소 분석을 통한 okt_keywords 추출
    okt_keywords = extract_okt_keywords(file_name)
    # GPT API 호출하여 gpt_keywords 추출
    gpt_keywords = ask_gpt_with_text_file(file_name)

    # 데이터베이스에 삽입
    cursor.execute('''
    INSERT INTO mp3_files (file_name, file_path, okt_keywords, gpt_keywords) 
    VALUES (?, ?, ?, ?)
    ''', (mp3_file, file_path, okt_keywords, gpt_keywords))

    conn.commit()
    conn.close()

# mp3 파일을 업데이트하는 함수 (키워드가 null인 경우만)
def update_mp3_to_db(db_path, mp3_file, folder_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    file_name = os.path.splitext(mp3_file)[0]  # 파일명에서 확장자 제거
    file_path = os.path.join(folder_path, mp3_file)

    # 키워드 추출
    okt_keywords = extract_okt_keywords(file_name)
    gpt_keywords = ask_gpt_with_text_file(file_name)

    # okt_keywords와 gpt_keywords가 NULL인 경우에만 업데이트
    cursor.execute('''
    UPDATE mp3_files
    SET okt_keywords = ?, gpt_keywords = ?
    WHERE file_name = ? AND (okt_keywords IS NULL OR gpt_keywords IS NULL)
    ''', (okt_keywords, gpt_keywords, mp3_file))

    conn.commit()
    conn.close()

# mp3 파일을 데이터베이스에서 삭제하는 함수
def delete_mp3_from_db(db_path, mp3_file):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 파일명에서 확장자 제거
    file_name = os.path.splitext(mp3_file)[0]

    # 데이터베이스에서 해당 파일 삭제
    cursor.execute('''
    DELETE FROM mp3_files WHERE file_name = ?
    ''', (file_name,))

    conn.commit()
    conn.close()

# # 디렉토리 내 모든 기존 mp3 파일을 처리하는 함수
# def process_existing_files(db_path, folder_path):
#     for mp3_file in os.listdir(folder_path):
#         if mp3_file.endswith('.mp3'):
#             print(f"Processing existing file: {mp3_file}")
#             insert_mp3_to_db(db_path, mp3_file, folder_path)
#             update_mp3_to_db(db_path, mp3_file, folder_path)

# 파일 시스템 이벤트 핸들러
class MP3Handler(FileSystemEventHandler):
    def __init__(self, db_path, folder_path):
        self.db_path = db_path
        self.folder_path = folder_path
    
    def on_created(self, event):
        # 새로운 파일이 생성되었을 때만 동작 (mp3 파일만 처리)
        if event.src_path.endswith('.mp3'):
            mp3_file = os.path.basename(event.src_path)
            insert_mp3_to_db(self.db_path, mp3_file, self.folder_path)
            print(f"{mp3_file} has been added to the database with okt_keywords and gpt_keywords.")
            update_mp3_to_db(self.db_path, mp3_file, self.folder_path)
            print(f"{mp3_file}의 okt_keywords와 gpt_keywords가 업데이트되었습니다.")

    def on_deleted(self, event):
        # mp3 파일이 삭제되면 데이터베이스에서도 해당 파일 삭제
        if event.src_path.endswith('.mp3'):
            mp3_file = os.path.basename(event.src_path)
            delete_mp3_from_db(self.db_path, mp3_file)
            print(f"{mp3_file} has been deleted from the database.")

# 자동화 시작 함수
def start_watching(folder_path, db_path):
    event_handler = MP3Handler(db_path, folder_path)
    observer = Observer()
    observer.schedule(event_handler, path=folder_path, recursive=False)
    observer.start()
    print(f"Watching for new mp3 files in: {folder_path}")

    try:
        while True:
            time.sleep(1)  # 계속 모니터링
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# 설정
db_path = './mp3_database.db'  # 이 파일은 실행할 때 자동 생성됨
folder_path = './mp3_database'  # mp3 파일이 있는 폴더 경로

# 데이터베이스 설정
setup_database(db_path)

# # 기존 mp3 파일 처리
# process_existing_files(db_path, folder_path)

# 폴더 모니터링 시작 (자동으로 mp3 파일 추가 감지)
start_watching(folder_path, db_path)
