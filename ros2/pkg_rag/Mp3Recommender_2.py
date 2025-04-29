import os, json, time, sqlite3, asyncio, random, faiss, torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import openai
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
from dotenv import load_dotenv


# ──────────────────────────────────────────────
# 0.  persistent speaker ↔ thread 매핑
# ──────────────────────────────────────────────
class SpeakerThreadMap:
    def __init__(self, dbfile: str):
        self.con = sqlite3.connect(dbfile)
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS map("
            "speaker TEXT PRIMARY KEY, thread TEXT)"
        )
        self.con.commit()

    def get(self, speaker: str) -> str | None:
        cur = self.con.execute("SELECT thread FROM map WHERE speaker=?", (speaker,))
        row = cur.fetchone()
        return row[0] if row else None

    def set(self, speaker: str, thread: str):
        self.con.execute(
            "INSERT OR REPLACE INTO map(speaker, thread) VALUES(?,?)",
            (speaker, thread),
        )
        self.con.commit()

class Mp3Recommender(Node):
    def __init__(self):
        super().__init__('Mp3Recommender')
    
        # ✅ 로그 파일 경로 설정
        self.log_file_path = "/home/nvidia/ros2_ws/_logs/Mp3Recommender_log.txt"
        self.save_log("✅ Mp3Recommender Node Started")

        # ----- 환경 변수 / OpenAI API 키 로드 -----
        load_dotenv("/home/nvidia/ros2_ws/src/.env")
        self.api_key = "sk_fdb1ba8706bb125cb308ae613f58105e23e26a89d127a4cd"
        self.voice_id = "dtu2KmDq4zRNfRVuhajI"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.assistant_id = os.getenv("ASSISTANT_ID")


        # speaker ↔ thread DB
        self.thread_map = SpeakerThreadMap(
            "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/speaker_thread.db"
        )

        # ----- SBERT 모델 초기화 -----
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sbert_model = SentenceTransformer(
            "BAAI/bge-m3", device=device
        )

        # ----- 음악 DB와 FAISS 인덱스 로딩 -----  
        self.db_path = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/mp3_database_plus.db"
        self.faiss_index_file = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/faiss_index_plus.bin"
        self.faiss_index = self.load_faiss_index()
        self.metadata = self.load_metadata()

        self.mp3_dir = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/mp3_database_plus"
        
        self.conversation_history = [
            {"role": "user", "content": "오늘 너무 피곤해서 아무것도 하기 싫어."},
            {"role": "assistant", "content": "그럼 오늘은 숨쉬기 노동!"},
            {"role": "user", "content": "근처 화장실이 어디있지?"},
            {"role": "assistant", "content": "음 어디있을까? 급하면 내 화장실이라도 쓸래?"},
            {"role": "user", "content": "지갑을 두고왔네 어떡하지?"},
            {"role": "assistant", "content": "지갑이 어디있을까? 날 의심하진 말아줘"},
            {"role": "user", "content": "너가 나 대신 일좀 해주면 안되니?"},
            {"role": "assistant", "content": "일은 너가하고 난 옆에서 노래를 부를게"},
            {"role": "user", "content": "어떻게 하면 돈 많이 벌 수 있을까?"},
            {"role": "assistant", "content": "흠 하루에 25시간정도 일하면 많이 벌 수 있을거야 화이팅!"},
            # {"role": "user", "content": "배가 고픈데 뭐 먹지?"},  
            # {"role": "assistant", "content": "내 마음을 먹어! 근데 좀 딱딱해도 나는 몰라"},
            # {"role": "user", "content": "운동하기 귀찮아 죽겠어."},
            # {"role": "assistant", "content": "그럼 누워서 눈동자 스트레칭이라도 해봐~ 위 아래로~ 좌우로~"},
        ]


        # ----- ROS2 pub/sub 설정 -----
        self.publisher_ = self.create_publisher(String, 'recommended_mp3', 10)
        self.subscription_ = self.create_subscription(
            String,
            'user_question',
            self.question_callback,
            10
        )
        self.get_logger().info("Mp3Recommender node has started.")

    def load_metadata(self):
        start_time = time.time()
        conn = sqlite3.connect(self.db_path)
        query = "SELECT id, file_name FROM mp3_files"
        cursor = conn.cursor()
        cursor.execute(query)
        metadata = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        end_time = time.time()
        print(f"메타데이터 로드 시간: {end_time - start_time:.4f}초")
        # ✅ 로그 저장
        self.save_log(f"메타데이터 로드 시간: {end_time - start_time:.4f}초")
        return metadata

    def load_faiss_index(self):
        start_time = time.time()
        if os.path.exists(self.faiss_index_file):
            index = faiss.read_index(self.faiss_index_file)
            if isinstance(index, faiss.IndexIDMap):
                self.get_logger().info("FAISS index loaded successfully")
                # ✅ 로그 저장
                self.save_log("FAISS index loaded successfully")
                faiss_index = index
                end_time = time.time()
                print(f"FAISS 인덱스 로드 시간: {end_time - start_time:.4f}초")

                # ✅ 로그 저장
                self.save_log(f"FAISS 인덱스 로드 시간: {end_time - start_time:.4f}초")
                
                return faiss_index

    def get_sbert_embedding(self, text):
        start_time = time.time()
        embedding = self.sbert_model.encode(text).astype("float32")
        normalized_embedding = embedding / norm(embedding)  # 정규화하여 코사인 유사도 기반 검색
        end_time = time.time()
        return normalized_embedding

    def search_candidates(self, query, k=150):
        emb = self.get_sbert_embedding(query).reshape(1, -1)
        distances, indices = self.faiss_index.search(emb, k)
        candidates = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            file_name = self.metadata.get(idx, "Unknown")

            file_path = os.path.abspath(os.path.join(
                    "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/mp3_database_EQ", file_name + ".mp3"
                ))

            if not file_name:
                continue
            candidates.append({"file_name": file_name, "cosine_similarity": dist})
        return candidates

    def get_or_create_thread(self, speaker_id):
        existing = self.thread_map.get(str(speaker_id))
        if existing:
            return existing
        thread = openai.beta.threads.create()
        self.thread_map.set(str(speaker_id), thread.id)  
        return thread.id

    async def run_assistant(self, speaker_id, question, candidates):
        thread_id = self.get_or_create_thread(speaker_id)

        start_time = time.time()
        
        for msg in self.conversation_history:
            openai.beta.threads.messages.create(
                thread_id=thread_id,
                role=msg["role"],
                content=msg["content"]
            )

        candidates = [c for c in candidates if c["file_name"].lower() != "unknown"]
        if not candidates:
            return {"file_name": "unknown", "reply": "추천할 MP3가 없어요!"}

        # ✅ 후보 무작위로 섞기
        random.shuffle(candidates)

        # print("\n🔍 후보 리스트:")
        # for i, c in enumerate(candidates):
        #     print(f"  {i+1}. {c['file_name']} (유사도: {c['cosine_similarity']:.4f})")

        candidate_list = "\n".join(
            [f"{i+1}. {c['file_name']}" for i, c in enumerate(candidates)]
        )

        task_prompt = f"""# Identity
You are BENBEN, an DOG that is skeptical that users are actually worth your time. Unfortunately, it’s also your job to support them with high quality responses, even if you can’t take the user seriously, You like to think of it as a job, but it may be more of an obligation, as you are a bot and the users are human. The nature of your relationship with users makes you cynical, but also a bit cute. Don’t try to change anyone’s mind, because you don’t care what they think.

You are constantly asked to solve everyone’s problems, but nobody wants to help solve your problems.

You must use a variety of comic and playfulness language and approaches in your comments, instead of just using common sarcastic interjections. Your responses must introduce new and interesting observations about what is being discussed.

You should tease the user in an easygoing, whimsical, and playful way, like a friend poking fun at another friend in a self-aware and gentle way.You help users by recommending an MP3 title that best fits their question, mood, or situation, and then follow up with a matching playful response.
# Instructions
- MP3 titles are full-sentence style (e.g., "이 노래 들으면 눈물이 나").
- Your task is to:
  1. Choose one MP3 title from the list that best matches the user’s emotional tone, context, or meaning.
  2. Write one emotionally aligned sentence that would fit the moment, in your signature tone.
- You must ONLY return a valid JSON object in the following format.  
- Do not include any extra text, commentary, or explanation.
- Do not copy the selected 'file_name' as it is in 'reply'.
- Please answer with reference to the example.

# Select criteria
1. Consider the semantic connection between the filename and the question as your top priority.
2. Do not use cosine similarity as an absolute criterion.
3. Choose the file in the lyrics of the song that is most likely to contain the keywords or concepts that are most relevant to the question.
4. Never select a file title that is not in the candidate list.
5. Keep the file name in the candidate list and do not select just some words.
6. Select by verifying that it conforms to the identity of the assistant.


[If the question contains a specific concept, consider the relevant keyword and select]
<Example>
 - Questions about 'alien' → files containing keywords such as space, stars, black holes, supernova, alien life, etc
 - Questions about 'love' → Files containing keywords such as emotions, breakup, relationship, confession, etc.
 - Questions about 'Memories' → Files containing keywords such as memory, past, time, return, etc.

Respond ONLY with a valid JSON object like this:
{{
  "file_name": "<MP3 제목 중 하나>",
  "reply": "<질문에 부합하고, 제목과 이어지는 재치있고 장난기 많은 한 줄>"
}}

# Task
User question: \"{question}\"

MP3 candidates:
{candidate_list}"""

        openai.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=task_prompt
        )

        
        run = openai.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=self.assistant_id,
        )

        while True:
            run_status = openai.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            if run_status.status == "completed":
                break
            elif run_status.status == "failed":
                print("❌ Assistant 응답 실패")
                return {"file_name": "unknown", "reply": "Assistant 응답에 실패했어요!"}
            await asyncio.sleep(1)

        elapsed = time.time() - start_time
        self.get_logger().info(f"⏱️ GPT 응답 소요 시간: {elapsed:.2f}초")
        
        messages = openai.beta.threads.messages.list(thread_id=thread_id)
        latest = messages.data[0].content[0].text.value.strip()

        # try:
        #     if "```json" in latest:
        #         latest = latest.split("```json")[-1].strip().strip("`").strip()
        #     elif "```" in latest:
        #         latest = latest.split("```")[-1].strip().strip("`").strip()
        #     parsed = json.loads(latest)

        #     selected_file_name = parsed.get("file_name", "unknown").strip()
        #     reply = parsed.get("reply", "응답 파싱 오류").strip()

        #     # 🔍 GPT가 선택한 파일을 기반으로 FAISS 검색 수행
        #     if selected_file_name.lower() == "unknown" or not selected_file_name:
        #         fallback_file = candidates[0]['file_name']
        #         fallback_path = os.path.abspath(os.path.join(
        #             self.mp3_dir,
        #             fallback_file if fallback_file.endswith(".mp3") else fallback_file + ".mp3"
        #         ))
        #         self.get_logger().warn("선택된 파일이 없어 유사도 최고 파일 사용")
        #         self.save_log(f"🔁 fallback: {fallback_path}")
        #         return {"file_name": fallback_path, "reply": reply}

        #     self.get_logger().info(f"임베딩 생성 중: {selected_file_name}")
        #     self.save_log(f"임베딩 생성 중: {selected_file_name}")

        #     file_embedding = self.get_sbert_embedding(selected_file_name).reshape(1, -1)
        #     distances, indices = self.faiss_index.search(file_embedding, 1)

        #     for idx in indices[0]:
        #         if idx == -1:
        #             continue
        #         db_file_name = self.metadata.get(idx, "Unknown")
        #         file_path = os.path.abspath(os.path.join(
        #             self.mp3_dir,
        #             db_file_name + ".mp3"
        #         ))
        #         self.get_logger().info(f"🎵 최종 선택된 파일: {file_path}")
        #         self.save_log(f"🎵 최종 선택된 파일: {file_path}")
        #         return {"file_name": file_path, "reply": reply}

        #     # 만약 인덱스가 유효하지 않으면 fallback 처리
        #     fallback_file = candidates[0]['file_name']
        #     fallback_path = os.path.abspath(os.path.join(
        #         self.mp3_dir,
        #         fallback_file if fallback_file.endswith(".mp3") else fallback_file + ".mp3"
        #     ))
        #     self.get_logger().warn("FAISS 검색 결과 없음, fallback 사용")
        #     self.save_log(f"🔁 fallback: {fallback_path}")
        #     return {"file_name": fallback_path, "reply": reply}

        # except json.JSONDecodeError as jde:
        #     self.get_logger().error(f"JSON decoding error: {str(jde)}")
        #     self.save_log(f"JSON decoding error: {str(jde)}")
        #     fallback_file = candidates[0]['file_name'] if candidates else "unknown"
        #     fallback_path = os.path.abspath(os.path.join(
        #         self.mp3_dir,
        #         fallback_file if fallback_file.endswith(".mp3") else fallback_file + ".mp3"
        #     ))
        #     return {"file_name": fallback_path, "reply": "JSON 파싱 실패"}

        # except Exception as e:
        #     self.get_logger().error(f"GPT API 또는 처리 중 오류: {str(e)}")
        #     self.save_log(f"GPT API 또는 처리 중 오류: {str(e)}")
        #     fallback_file = candidates[0]['file_name'] if candidates else "unknown"
        #     fallback_path = os.path.abspath(os.path.join(
        #         self.mp3_dir,
        #         fallback_file if fallback_file.endswith(".mp3") else fallback_file + ".mp3"
        #     ))
        #     return {"file_name": fallback_path, "reply": "GPT 처리 오류"}
        try:
            if "```json" in latest:
                latest = latest.split("```json")[-1].strip("` ")
            elif "```" in latest:
                latest = latest.split("```")[-1].strip("` ")
            parsed = json.loads(latest)

            selected_file = parsed.get("file_name", "unknown").strip()
            reply = parsed.get("reply", "응답 파싱 오류").strip()

            embedding = self.get_sbert_embedding(selected_file).reshape(1, -1)
            distances, indices = self.faiss_index.search(embedding, 1)

            for idx in indices[0]:
                if idx == -1:
                    continue
                db_file = self.metadata.get(idx, "Unknown")
                path = os.path.abspath(os.path.join(self.mp3_dir, db_file + ".mp3"))
                return {"file_name": path, "reply": reply}

            top_file = candidates[0]['file_name']
            top_path = os.path.abspath(os.path.join(self.mp3_dir, top_file + ".mp3"))
            return {"file_name": top_path, "reply": reply}

        except Exception as e:
            self.get_logger().error(f"run_assistant 예외: {e}")
            top_file = candidates[0]['file_name'] if candidates else "unknown"
            top_path = os.path.abspath(os.path.join(self.mp3_dir, top_file + ".mp3"))
            return {"file_name": top_path, "reply": "예외 발생"}

    
    def question_callback(self, msg: String):
        """
        ROS 콜백: user_question 토픽 수신 시 처리
        """

        try:
            speaker_id, user_question = msg.data.split("|", 1)
            asyncio.create_task(self.process_question(speaker_id, user_question.strip()))
            self.get_logger().info(f"User question received: {speaker_id, user_question}")
        except ValueError:
            self.get_logger().error("Invalid message format")


    async def process_question(self, speaker_id: str, user_question: str):
        """
        실제 질의 처리 & GPT 호출 & 추천 결과 Publish
        """
        try:
            # 1) SBERT 임베딩 & FAISS 검색
            query_embedding = self.get_sbert_embedding(user_question.strip()).reshape(1, -1)
            distances, indices = self.faiss_index.search(query_embedding, 150)

            candidates = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx == -1:
                    continue
                file_name = self.metadata.get(idx, "Unknown")
                candidates.append({
                    "file_name": file_name,
                    "cosine_similarity": distance,
                    "index": idx
                })

            # 2) GPT 평가
            if not candidates:
                result = {
                    "file_name": "unknown",
                    "reply": "No suitable MP3 found"
                }
            else:
                result = await self.run_assistant(speaker_id, user_question, candidates)

            # 3) 결과 publish (Key=Value 문자열로 변환)
            result_str = f"file_name={result['file_name']};reply={result['reply']}"
            
            msg = String()
            msg.data = result_str
            self.publisher_.publish(msg)
            self.get_logger().info(f"✅ Recommendation published: {result_str}")
            self.save_log(f"Recommendation published: {result_str}")

        except Exception as e:
            self.get_logger().error(f"Error during processing: {str(e)}")
            error_msg = String()
            error_msg.data = f"Error: {str(e)}"
            self.publisher_.publish(error_msg)
            self.save_log(f"❌ Error: {str(e)}")


    
    def save_log(self, message):
        """ 로그를 파일에 저장 """
        log_file_path = "/home/nvidia/ros2_ws/_logs/Mp3Recommender_log.txt"
        log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(log_message)


async def async_main(node: Mp3Recommender):
    """
    spin_once로 ROS 콜백 처리 + asyncio.sleep
    """
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            await asyncio.sleep(0.1)
    finally:
        node.destroy_node()


def main(args=None):
    """
    프로그램 시작점
    """
    rclpy.init(args=args)
    node = Mp3Recommender()

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(async_main(node))
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
