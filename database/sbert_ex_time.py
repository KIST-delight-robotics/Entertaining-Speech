import time
from sentence_transformers import SentenceTransformer, util
import sqlite3
import openai

# 모델 리스트
model_names = [
    'paraphrase-multilingual-mpnet-base-v2',  # 2등  #이거
    'intfloat/multilingual-e5-large',         # 1등 (느림)  #이거
    'quora-distilbert-multilingual',          # 별로임
    'paraphrase-xlm-r-multilingual-v1',        # 별로임
    'paraphrase-multilingual-MiniLM-L12-v2',   #3등 .. 그런데 좀 별로
    'xlm-r-100langs-bert-base-nli-stsb-mean-tokens', #별로임
    'LaBSE', #그냥저냥
    'paraphrase-xlm-r-multilingual-v1', #별로임
    'distiluse-base-multilingual-cased-v1', #별로임
    'distiluse-base-multilingual-cased-v2', #별로임
    'distilbert-multilingual-nli-stsb-quora-ranking', #별로임
    'intfloat/multilingual-e5-base', #별로임
    'intfloat/multilingual-e5-small', #별로임
    'stsb-xlm-r-multilingual', #별로임

]

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
        model="gpt-4",
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
def find_best_match(gpt_response, model, db_path):
    # 데이터베이스에서 파일명 가져오기
    filenames = get_filenames_from_db(db_path)
    
    # GPT 응답과 파일명을 벡터로 변환
    start_time = time.time()
    gpt_embedding = model.encode(gpt_response, convert_to_tensor=True)
    filename_embeddings = model.encode(filenames, convert_to_tensor=True)
    inference_time = time.time() - start_time
    
    # 유사도 계산
    similarities = util.cos_sim(gpt_embedding, filename_embeddings)
    
    # 가장 높은 유사도 파일명 찾기
    best_match_idx = similarities.argmax()
    best_match_filename = filenames[best_match_idx]
    best_similarity_score = similarities[0][best_match_idx].item()
    
    return best_match_filename, best_similarity_score, inference_time

# 테스트할 텍스트 파일 경로
text_file_path = "./requestA.txt"
gpt_response_text = get_gpt_response(text_file_path)

# 데이터베이스 경로
db_path = './mp3_database.db'

# 모델별로 성능과 속도 측정
results = []
for model_name in model_names:
    model = SentenceTransformer(model_name)
    print(f"Testing model: {model_name}")
    
    # 매칭 수행
    best_match, similarity_score, inference_time = find_best_match(gpt_response_text, model, db_path)
    
    # 결과 저장
    results.append({
        'model': model_name,
        'best_match': best_match,
        'similarity_score': similarity_score,
        'inference_time': inference_time
    })

# 결과 출력
for result in results:
    print(f"Model: {result['model']}")
    print(f"Best match filename: {result['best_match']}, Similarity score: {result['similarity_score']:.2f}, Inference time: {result['inference_time']:.2f} seconds")
    print('-' * 50)
 