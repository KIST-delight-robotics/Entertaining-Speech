import os
import re
import sqlite3
from konlpy.tag import Okt
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

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
        keywords TEXT
    )
    ''')
    conn.commit()
    conn.close()

# 형태소 분석기 객체 생성
okt = Okt()

# 파일명에서 키워드를 추출하는 함수 (모든 명사 및 동사 어근 추출)
def extract_keywords(file_name):
    # 형태소 분석 결과 중 명사와 동사의 어간만 추출
    keywords = []
    morphs = okt.pos(file_name, stem=True)  # 형태소 분석, 동사 어간 추출
    for word, tag in morphs:
        if tag in ['Noun', 'Verb']:  # 명사와 동사(어간만)
            keywords.append(word)

    # 영어 단어 추출 (정규 표현식 사용)
    english_words = re.findall(r'[a-zA-Z]+', file_name)
    keywords.extend(english_words)

    return keywords

# mp3 파일을 데이터베이스에 추가하는 함수 (모든 키워드 포함)
def insert_mp3_to_db(db_path, mp3_file, folder_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    file_name = os.path.splitext(mp3_file)[0]  # 파일명에서 확장자 제거
    file_path = os.path.join(folder_path, mp3_file)
    
    # 파일명에서 키워드 추출
    keywords = extract_keywords(file_name)
    
    # 쉼표로 구분하여 모든 키워드를 문자열로 저장
    keywords_str = ', '.join(keywords)
    
    # 데이터베이스에 삽입
    cursor.execute('''
    INSERT INTO mp3_files (file_name, file_path, keywords) 
    VALUES (?, ?, ?)''', 
    (mp3_file, file_path, keywords_str))

    conn.commit()
    conn.close()

# mp3 파일을 업데이트하는 함수 (키워드가 null인 경우만)
def update_keywords_to_db(db_path, mp3_file, folder_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    file_name = os.path.splitext(mp3_file)[0]  # 파일명에서 확장자 제거
    file_path = os.path.join(folder_path, mp3_file)

    # 파일명에서 키워드 추출
    keywords = extract_keywords(file_name)
    
    # 쉼표로 구분하여 모든 키워드를 문자열로 저장
    keywords_str = ', '.join(keywords)

    # keywords가 NULL인 경우에만 업데이트
    cursor.execute('''
    UPDATE mp3_files
    SET keywords = ?
    WHERE file_name = ? AND keywords IS NULL
    ''', (keywords_str, mp3_file))

    conn.commit()
    conn.close()

# 디렉토리 내 모든 기존 mp3 파일을 처리하는 함수
def process_existing_files(db_path, folder_path):
    for mp3_file in os.listdir(folder_path):
        if mp3_file.endswith('.mp3'):
            print(f"Processing existing file: {mp3_file}")
            insert_mp3_to_db(db_path, mp3_file, folder_path)
            update_keywords_to_db(db_path, mp3_file, folder_path)

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
            print(f"{mp3_file} has been added to the database with keywords.")
            update_keywords_to_db(self.db_path, mp3_file, self.folder_path)
            print(f"{mp3_file}의 keywords가 업데이트되었습니다.")

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

# 기존 mp3 파일 처리
process_existing_files(db_path, folder_path)

# 폴더 모니터링 시작 (자동으로 mp3 파일 추가 감지)
start_watching(folder_path, db_path)
