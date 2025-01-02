import os
import re
import sqlite3
import openai
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# GPT API Key 설정
openai.api_key = "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"  

nltk.data.path.append('/home/delight/nltk_data')

nltk.download('punkt', download_dir='/home/delight/nltk_data')
nltk.download('averaged_perceptron_tagger', download_dir='/home/delight/nltk_data')

# SeamlessM4T 모델 설정
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ko-en")

# 데이터베이스 연결 및 테이블 확인 후 필요 시 생성
def setup_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS mp3_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        translated_name TEXT,
        file_path TEXT,
        l_keywords TEXT,
        s_keywords TEXT
    )
    ''')
    conn.commit()
    conn.close()


def translate_korean_to_english(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs["input_ids"])
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def extract_l_keywords(text):
    try:
        valid_pos = ['NN', 'NNS', 'NNP', 'NNPS',  # 명사
                     'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # 동사
                     'JJ', 'JJR', 'JJS',  # 형용사
                     'RB', 'RBR', 'RBS']  # 부사

        print(f"Extracting keywords from text: {text}")
        words = word_tokenize(text)  # 단어 토큰화
        tagged = pos_tag(words)  # 품사 태깅
        print(f"POS tagging: {tagged}")  # 디버깅 메시지

        # valid_pos 리스트에 포함된 품사만 필터링
        l_keywords = [word for word, tag in tagged if tag in valid_pos]
        print(f"Extracted keywords: {l_keywords}")

        return ', '.join(l_keywords)
    except Exception as e:
        print(f"Error during keyword extraction: {e}")
        return ""

def extract_s_keywords(user_input):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "상황을 제시하면 그에 맞는 적절한 대답을 하면 돼. A와 B는 친구야. 예를 들면, A : 오늘 눈이 올까? B : '내일 아침에 하얀 눈이 쌓여 있었음 해요.' 이런식으로 B의 대답은 기존의 음악이나 영화, 드라마의 한 문장이야. 예시를 고려해서 유추된 A의 문장 속 상황 키워드(명사형) 5개만 ,로 구분해서 영어로 제시해줘."},
                {"role": "assistant", "content": "user가 B의 대답을 제시할거야."},
                {"role": "user", "content": user_input}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"GPT API 호출 중 오류 발생: {e}")
        return ""  # 기본값 반환

def process_file(file_name):
    translated_name = translate_korean_to_english(file_name)
    l_keywords = extract_l_keywords(translated_name)
    s_keywords = extract_s_keywords(translated_name)
    return translated_name, l_keywords, s_keywords

def insert_mp3_to_db(db_path, mp3_file, folder_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    file_name = os.path.splitext(mp3_file)[0]
    file_path = os.path.join(folder_path, mp3_file)

    translated_name, l_keywords, s_keywords = process_file(file_name)

    cursor.execute('''
    INSERT INTO mp3_files (translated_name, file_path, l_keywords, s_keywords) 
    VALUES (?, ?, ?, ?)
    ''', (translated_name, file_path, l_keywords, s_keywords))

    conn.commit()
    conn.close()

def update_mp3_to_db(db_path, mp3_file, folder_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    file_name = os.path.splitext(mp3_file)[0]
    file_path = os.path.join(folder_path, mp3_file)
    translated_name = translate_korean_to_english(file_name)
    l_keywords = extract_l_keywords(translated_name)
    s_keywords = extract_s_keywords(translated_name)

    cursor.execute('''
    UPDATE mp3_files
    SET file_path = ?, l_keywords = ?, s_keywords = ?
    WHERE translated_name = ?
    ''', (file_path, l_keywords, s_keywords, translated_name))

    conn.commit()
    conn.close()

def delete_mp3_from_db(db_path, mp3_file):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    file_name = os.path.splitext(mp3_file)[0]
    translated_name = translate_korean_to_english(file_name)

    cursor.execute('''
    DELETE FROM mp3_files WHERE translated_name = ?
    ''', (translated_name,))

    conn.commit()
    conn.close()

class MP3Handler(FileSystemEventHandler):
    def __init__(self, db_path, folder_path):
        self.db_path = db_path
        self.folder_path = folder_path

    def on_created(self, event):
        if event.src_path.endswith('.mp3'):
            mp3_file = os.path.basename(event.src_path)
            insert_mp3_to_db(self.db_path, mp3_file, self.folder_path)
            print(f"{mp3_file} has been added to the database.")

    def on_deleted(self, event):
        if event.src_path.endswith('.mp3'):
            mp3_file = os.path.basename(event.src_path)
            delete_mp3_from_db(self.db_path, mp3_file)
            print(f"{mp3_file} has been deleted from the database.")

def start_watching(folder_path, db_path):
    os.makedirs(folder_path, exist_ok=True)
    event_handler = MP3Handler(db_path, folder_path)
    observer = Observer()
    observer.schedule(event_handler, path=folder_path, recursive=False)
    observer.start()
    print(f"Watching for new mp3 files in: {folder_path}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

db_path = './en_mp3_database.db'
folder_path = './en_mp3_database'

setup_database(db_path)
start_watching(folder_path, db_path)