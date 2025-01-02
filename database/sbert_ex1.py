#---------------------------------------------------------------------------------------------
#A의 질문에 대해 GPT를 통해 B의 대답을 추출하고, 그 대답과 대답 리스트를 비교해서 적절한 대답을 선택
#---------------------------------------------------------------------------------------------

from sentence_transformers import SentenceTransformer, util
import sqlite3
import openai

#별로인 이유 : 일치도가 낮은 문장을 높은 스코어로 도출해서.
# 2,3등의 경우 일치도가 높은 문장이 없어서 뽑아내지 못하는 것으로 보임. 그래도 어느정도 문맥에 맞는 문장 뽑아냄.

# sbert 모델 로드
#model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') #3등 .. 그런데 좀 별로
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2') #2등
#model = SentenceTransformer('intfloat/multilingual-e5-large') #1등 그런데 좀 느림
#model = SentenceTransformer('xlm-r-100langs-bert-base-nli-stsb-mean-tokens') #별로임
#model = SentenceTransformer('quora-distilbert-multilingual') #별로임
#model = SentenceTransformer('LaBSE') #그냥저냥
#model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1') #별로임
#model = SentenceTransformer('distiluse-base-multilingual-cased-v1') #별로임
#model = SentenceTransformer('distiluse-base-multilingual-cased-v2') #별로임
#model = SentenceTransformer('distilbert-multilingual-nli-stsb-quora-ranking') #별로임
#model = SentenceTransformer('intfloat/multilingual-e5-base') #별로임
#model = SentenceTransformer('intfloat/multilingual-e5-small') #별로임
#model = SentenceTransformer('stsb-xlm-r-multilingual') #별로임
# model = SentenceTransformer('') #
# model = SentenceTransformer('') #
# model = SentenceTransformer('') #
# model = SentenceTransformer('') #
# model = SentenceTransformer('') #
# model = SentenceTransformer('') #


# OpenAI API 키 설정
OPENAI_API_KEY = "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"

openai.api_key = OPENAI_API_KEY

# 텍스트 파일에서 내용을 읽는 함수
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# GPT에게 텍스트 파일 내용을 보내고 응답을 받는 함수
def get_gpt_response(file_path):
    user_input = read_text_file(file_path)
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "상황을 제시하면 그에 맞는 적절하고 자연스러운 대답을 하면 돼. A와 B는 친구야. 예를 들면, A : 오늘 눈이 올까? B : '눈이 오면 좋겠다.' 이런식으로 B의 대답은 A의 질문에 적절하게 이루어질거야. 단, B의 대답에서 명사형 키워드를 추출하기 적합하게 대답해줘. 반문은 하지 않아도 되고, 질문 키워드가 들어가게 대답해줘."},
            {"role": "assistant", "content": "user가 A의 질문을 제시할거야."},
            {"role": "user", "content": user_input}
        ]
    )
    return response['choices'][0]['message']['content']



# 데이터베이스에서 파일명 가져오는 함수
def get_filenames_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT file_name FROM mp3_files")
    filenames = [row[0] for row in cursor.fetchall()]
    conn.close()
    return filenames

# GPT 응답과 파일명 비교
def find_best_match(gpt_response, db_path):
    # 데이터베이스에서 파일명 가져오기
    filenames = get_filenames_from_db(db_path)


    
    # GPT 응답과 파일명을 벡터로 변환
    gpt_embedding = model.encode(gpt_response, convert_to_tensor=True)
    filename_embeddings = model.encode(filenames, convert_to_tensor=True)
    
    # 유사도 계산
    similarities = util.cos_sim(gpt_embedding, filename_embeddings)
    
    # 가장 높은 유사도 파일명 찾기
    best_match_idx = similarities.argmax()
    best_match_filename = filenames[best_match_idx]
    best_similarity_score = similarities[0][best_match_idx].item()
    
    return best_match_filename, best_similarity_score

text_file_path = "./requestA.txt"

gpt_response_text = get_gpt_response(text_file_path)
print(f"GPT Response: {gpt_response_text}")

db_path = './mp3_database.db'
best_match, similarity_score = find_best_match(gpt_response_text, db_path)
print(f"Best match filename: {best_match}, Similarity score: {similarity_score}")
