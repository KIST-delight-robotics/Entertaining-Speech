import os
import sqlite3
import openai

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
        keywords TEXT,
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

# 텍스트 파일에서 내용을 읽는 함수
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# GPT에게 텍스트 파일 내용을 보내고 응답을 받는 함수
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

# GPT 키워드를 추출하여 데이터베이스에 저장하는 함수
def insert_gpt_keywords_to_db(db_path, mp3_file, folder_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    file_name = os.path.splitext(mp3_file)[0]  # 파일명에서 확장자 제거
    file_path = os.path.join(folder_path, mp3_file)

    # GPT API를 호출해 키워드 추출
    gpt_keywords = ask_gpt_with_text_file(file_name)

    # 데이터베이스에 키워드 저장
    cursor.execute('''
    UPDATE mp3_files 
    SET gpt_keywords = ? 
    WHERE file_name = ?
    ''', (gpt_keywords, mp3_file))

    conn.commit()
    conn.close()

# mp3 파일을 데이터베이스에 추가하는 함수 (GPT 키워드 포함)
def insert_mp3_to_db_with_gpt(db_path, mp3_file, folder_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    file_name = os.path.splitext(mp3_file)[0]  # 파일명에서 확장자 제거
    file_path = os.path.join(folder_path, mp3_file)

    # GPT API 호출하여 키워드 추출
    gpt_keywords = ask_gpt_with_text_file(file_name)


    # GPT에서 받은 키워드 출력 (테스트용)
    print(f"{mp3_file}의 GPT 키워드:", gpt_keywords)

    # 데이터베이스에 삽입 (GPT 키워드 포함)
    cursor.execute('''
    INSERT INTO mp3_files (file_name, file_path, keywords, gpt_keywords) 
    VALUES (?, ?, ?, ?)
    ''', (mp3_file, file_path, None, gpt_keywords))

    conn.commit()
    conn.close()


# GPT 키워드를 추출하여 데이터베이스를 업데이트하는 함수
def update_gpt_keywords_to_db(db_path, mp3_file, folder_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    file_name = os.path.splitext(mp3_file)[0]  # 파일명에서 확장자 제거
    file_path = os.path.join(folder_path, mp3_file)

    # GPT API 호출하여 키워드 추출
    gpt_keywords = ask_gpt_with_text_file(file_name)

    # GPT 키워드가 없으면 기본값을 설정
    if gpt_keywords is None:
        gpt_keywords = '키워드 없음'

    # gpt_keywords가 NULL인 경우에만 업데이트
    cursor.execute('''
    UPDATE mp3_files
    SET gpt_keywords = ?
    WHERE file_name = ? AND gpt_keywords IS NULL
    ''', (gpt_keywords, mp3_file))

    conn.commit()
    conn.close()


# mp3 파일들을 읽고 GPT 키워드 추출 후 데이터베이스에 저장하는 함수
def process_mp3_files_with_gpt(db_path, folder_path):
    for mp3_file in os.listdir(folder_path):
        if mp3_file.endswith('.mp3'):
            insert_mp3_to_db_with_gpt(db_path, mp3_file, folder_path)
            print(f"{mp3_file} has been added to the database with GPT keywords.")
            update_gpt_keywords_to_db(db_path, mp3_file, folder_path)
            print(f"{mp3_file}의 gpt_keywords가 업데이트되었습니다.")

# 설정
db_path = './mp3_database.db'  # 이 파일은 실행할 때 자동 생성됨
folder_path = './mp3_database'  # mp3 파일이 있는 폴더 경로

# 데이터베이스 설정 (테이블 생성 등)
setup_database(db_path)

# mp3 파일 처리 및 GPT 키워드 추출
process_mp3_files_with_gpt(db_path, folder_path)
