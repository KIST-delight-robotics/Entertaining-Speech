import pandas as pd
import sqlite3

# SQLite DB 연결 및 테이블에서 데이터 불러오기
conn = sqlite3.connect('mp3_database.db')
df = pd.read_sql_query("SELECT * FROM mp3_files", conn)

# 데이터프레임을 CSV 파일로 저장
df.to_csv('output_file.csv', index=False)

# 연결 종료
conn.close()