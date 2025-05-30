
#루피 gpt + 영화음악db + ****이미지 선정
import os, json, time, sqlite3, asyncio, random, faiss, torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import openai
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
from dotenv import load_dotenv
import numpy as np


class Mp3Recommender(Node):
    def __init__(self):
        super().__init__('Mp3Recommender')
        
        # 로그 파일
        self.log_file_path = "/home/nvidia/ros2_ws/_logs/Mp3Recommender_log.txt"
        self.save_log("✅ Mp3Recommender Node Started")

        # 환경 변수
        load_dotenv("/home/nvidia/ros2_ws/src/.env")
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.assistant_id = os.getenv("ASSISTANT_ID")
        
        # OpenAI API 키 체크
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        if not self.assistant_id:
            raise ValueError("ASSISTANT_ID not found in environment variables")

        # SBERT 모델
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sbert_model = SentenceTransformer("BAAI/bge-m3", device=device)

        # MP3 인덱스/메타
        self.mp3_db_path = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/mp3_database_plus.db"
        self.mp3_faiss_index_file = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/faiss_index_plus.bin"
        self.mp3_dir = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/mp3_database_plus"
        
        # 이미지 인덱스/메타
        self.image_db_path = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/image_database_new.db"
        self.image_faiss_index_file = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/faiss_index_image_new.bin"
        self.image_dir = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/image_database_new"


        # 인덱스와 메타데이터 로드
        try:
            self.mp3_faiss_index = self.load_faiss_index_mp3()
            self.mp3_metadata = self.load_metadata_mp3()
            self.image_faiss_index = self.load_faiss_index_image()
            self.image_metadata = self.load_metadata_image()
        except Exception as e:
            self.get_logger().error(f"Failed to load indices or metadata: {e}")
            raise

        # Thread 관리를 위한 딕셔너리
        self.thread_map = {}
        
        # 대화 히스토리
        self.conversation_history = []
        
        # FAISS 인덱스와 메타데이터 별칭 (기존 코드 호환성)
        self.faiss_index = self.mp3_faiss_index
        self.metadata = self.mp3_metadata

        # ROS2 pub/sub
        self.publisher_ = self.create_publisher(String, 'recommended_mp3', 10)
        self.image_publisher_ = self.create_publisher(String, 'recommended_image', 10)
        self.subscription_ = self.create_subscription(String, 'user_question', self.question_callback, 10)
        self.status_publisher = self.create_publisher(String, 'mp3_recommend_status', 10)

        self.get_logger().info("Mp3Recommender node has started.")

    def save_log(self, message: str):
        """로그를 파일에 저장"""
        try:
            log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
            
            # 로그 디렉토리 생성
            log_dir = os.path.dirname(self.log_file_path)
            os.makedirs(log_dir, exist_ok=True)
            
            with open(self.log_file_path, "a", encoding="utf-8") as log_file:
                log_file.write(log_message)
        except Exception as e:
            self.get_logger().error(f"Failed to save log: {e}")

    def load_metadata_mp3(self) -> Dict[int, str]:
        start = time.time()
        try:
            conn = sqlite3.connect(self.mp3_db_path)
            cur = conn.cursor()
            cur.execute("SELECT id, file_name FROM mp3_files")
            meta = {row[0]: row[1] for row in cur.fetchall()}
            conn.close()
            self.save_log(f"MP3 metadata loaded in {time.time()-start:.4f}s")
            return meta
        except sqlite3.Error as e:
            self.get_logger().error(f"Database error loading MP3 metadata: {e}")
            raise
        except Exception as e:
            self.get_logger().error(f"Error loading MP3 metadata: {e}")
            raise

    def load_metadata_image(self) -> Dict[int, str]:
        start = time.time()
        try:
            conn = sqlite3.connect(self.image_db_path)
            cur = conn.cursor()
            cur.execute("SELECT id, file_name FROM images")
            meta = {row[0]: row[1] for row in cur.fetchall()}
            conn.close()
            self.save_log(f"Image metadata loaded in {time.time()-start:.4f}s")
            return meta
        except sqlite3.Error as e:
            self.get_logger().error(f"Database error loading image metadata: {e}")
            raise
        except Exception as e:
            self.get_logger().error(f"Error loading image metadata: {e}")
            raise

    def load_faiss_index_mp3(self):
        try:
            if os.path.exists(self.mp3_faiss_index_file):
                idx = faiss.read_index(self.mp3_faiss_index_file)
                if isinstance(idx, faiss.IndexIDMap):
                    self.save_log("MP3 FAISS index loaded successfully")
                    return idx
            raise FileNotFoundError(f"MP3 FAISS index not found at {self.mp3_faiss_index_file}")
        except Exception as e:
            self.get_logger().error(f"Error loading MP3 FAISS index: {e}")
            raise

    def load_faiss_index_image(self):
        try:
            if os.path.exists(self.image_faiss_index_file):
                idx = faiss.read_index(self.image_faiss_index_file)
                if isinstance(idx, faiss.IndexIDMap):
                    self.save_log("Image FAISS index loaded successfully")
                    return idx
            raise FileNotFoundError(f"Image FAISS index not found at {self.image_faiss_index_file}")
        except Exception as e:
            self.get_logger().error(f"Error loading image FAISS index: {e}")
            raise

    def get_sbert_embedding(self, text: str) -> np.ndarray:
        try:
            emb = self.sbert_model.encode(text).astype("float32")
            norm_val = norm(emb)
            if norm_val == 0:
                self.get_logger().warning("Zero norm embedding detected")
                return emb
            return emb / norm_val
        except Exception as e:
            self.get_logger().error(f"Error generating embedding: {e}")
            raise

    def search_candidates(self, query: str, k: int = 150) -> List[Dict]:
        try:
            emb = self.get_sbert_embedding(query).reshape(1, -1)
            D, I = self.mp3_faiss_index.search(emb, k)
            cands = []
            for dist, idx in zip(D[0], I[0]):
                if idx < 0:
                    continue
                fn = self.mp3_metadata.get(idx)
                if not fn:
                    continue
                path = os.path.join(self.mp3_dir, fn + ".mp3")
                cands.append({"file_name": fn, "path": path, "score": float(dist), "index": idx})
            return cands
        except Exception as e:
            self.get_logger().error(f"Error searching candidates: {e}")
            return []

    def search_images(self, reply: str, top_k: int = 200) -> List[Dict]:
        try:
            emb = self.get_sbert_embedding(reply).reshape(1, -1)
            D, I = self.image_faiss_index.search(emb, top_k)
            cands = []
            for dist, idx in zip(D[0], I[0]):
                if idx < 0:
                    continue
                fn = self.image_metadata.get(idx)
                if not fn:
                    continue
                path = os.path.join(self.image_dir, fn)
                cands.append({"file_name": fn, "file_path": path, "score": float(dist)})
            return cands
        except Exception as e:
            self.get_logger().error(f"Error searching images: {e}")
            return []

    async def evaluate_image_with_gpt(self, question, mp3_title, reply, candidates, top_k=1):
        if not candidates:
            self.get_logger().warning("No image candidates provided")
            return None
        
        try:
            # unknown 파일 필터링
            filtered = [c for c in candidates if c['file_name'].lower() != 'unknown']
            if not filtered:
                self.get_logger().warning("No valid image candidates after filtering")
                return None
            
            # 최대 80개로 제한
            filtered = filtered[:80]
            items = "\n".join([f"{i+1}. {c['file_name']}" for i, c in enumerate(filtered)])
            
            prompt = f"""
    # Instructions
    - 사용자의 질문, 대답으로 선정된 MP3 제목을 모두 고려하여 후보 이미지 중에서 가장 어울리는 것을 선택하세요.
    - 어울린다는 것은 사용자의 의도와 맥락을 고려해서 선택된 MP3 제목이 이미지 파일명과 부합함을 의미합니다.
    - 가장 중요한 것은 선정할 이미지와 MP3 제목이 최적으로 일치하는가 입니다.
    - Return a valid JSON object ONLY.

    # Output format
    {{
    "file_name": "<이미지 파일명>"
    }}

    # Context
    User question: "{question}"
    Song title: "{mp3_title}"
    Reply text: "{reply}"

    Image candidates:
    {items}
    """
            
            # OpenAI API v1 호환 호출
            client = openai.OpenAI(api_key=openai.api_key)
            
            try:
                resp = await asyncio.to_thread(
                    client.chat.completions.create,
                    model='gpt-4o',
                    messages=[
                        {"role": "system", "content": "You are an expert at selecting the most appropriate image from a list. Always respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=200
                )
                
                raw_content = resp.choices[0].message.content
                self.get_logger().info(f"Raw GPT response: {raw_content}")
                
            except Exception as api_error:
                self.get_logger().error(f"OpenAI API call failed: {api_error}")
                return filtered[0] if filtered else None
            
            if not raw_content:
                self.get_logger().error("Empty response from OpenAI API")
                return filtered[0] if filtered else None
            
            # JSON 추출 및 파싱
            try:
                # 코드 블록 제거
                clean_content = raw_content.strip()
                if '```json' in clean_content:
                    clean_content = clean_content.split('```json')[1].split('```')[0].strip()
                elif '```' in clean_content:
                    clean_content = clean_content.split('```')[1].strip()
                
                # JSON 파싱
                data = json.loads(clean_content)
                selected_filename = data.get('file_name', '').strip()
                
                if not selected_filename:
                    self.get_logger().warning("No file_name in GPT response")
                    return filtered[0] if filtered else None
                
                # ===== 새로 추가된 FAISS 검증 과정 =====
                # 선택된 파일의 임베딩을 통한 실제 존재 여부 확인
                try:
                    # 이미지 디렉토리 경로가 있다고 가정 (self.image_dir 또는 적절한 경로)
                    # MP3와 동일한 방식으로 이미지 FAISS 인덱스가 있다고 가정 (self.image_faiss_index, self.image_metadata)
                    
                    # 선택된 파일명으로 임베딩 생성
                    embedding = self.get_sbert_embedding(selected_filename).reshape(1, -1)
                    
                    # FAISS 검색으로 실제 파일 존재 확인
                    distances, indices = self.image_faiss_index.search(embedding, 1)
                    
                    validated_candidate = None
                    for idx in indices[0]:
                        if idx == -1:
                            continue
                        db_file = self.image_metadata.get(idx, "Unknown")
                        # 이미지 파일 경로 확인 (확장자는 상황에 맞게 조정)
                        for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                            path = os.path.abspath(os.path.join(self.image_dir, db_file + ext))
                            if os.path.exists(path):
                                # 파일이 존재하면 해당 후보를 찾아서 반환
                                for candidate in filtered:
                                    if candidate['file_name'] == db_file:
                                        validated_candidate = candidate
                                        break
                                if validated_candidate:
                                    break
                        if validated_candidate:
                            break
                    
                    # FAISS 검증을 통해 찾은 파일이 있으면 반환
                    if validated_candidate:
                        self.get_logger().info(f"FAISS validated image: {validated_candidate['file_name']}")
                        return validated_candidate
                    else:
                        self.get_logger().warning(f"FAISS validation failed for: {selected_filename}")
                    
                except Exception as faiss_error:
                    self.get_logger().warning(f"FAISS validation error: {faiss_error}")
                # ===== FAISS 검증 과정 끝 =====
                
                # 기존 방식으로 폴백: 선택된 파일명과 일치하는 후보 찾기
                for candidate in filtered:
                    if candidate['file_name'] == selected_filename:
                        self.get_logger().info(f"Direct match found: {selected_filename}")
                        return candidate
                
                # 정확히 일치하는 것이 없으면 부분 일치 검색
                for candidate in filtered:
                    if selected_filename in candidate['file_name'] or candidate['file_name'] in selected_filename:
                        self.get_logger().info(f"Partial match found: {candidate['file_name']}")
                        return candidate
                
                # 매칭되는 것이 없으면 첫 번째 후보 반환
                self.get_logger().warning(f"No match found for selected filename: {selected_filename}, using first candidate")
                return filtered[0]
                
            except json.JSONDecodeError as e:
                self.get_logger().error(f"JSON parsing error: {e}")
                self.get_logger().error(f"Raw content that failed to parse: {repr(clean_content)}")
                return filtered[0] if filtered else None
            
            except Exception as e:
                self.get_logger().error(f"Unexpected error in JSON processing: {e}")
                return filtered[0] if filtered else None
                
        except Exception as e:
            self.get_logger().error(f"Error in evaluate_image_with_gpt: {e}")
            # 예외 발생 시 첫 번째 후보 반환
            if candidates:
                filtered = [c for c in candidates if c['file_name'].lower() != 'unknown']
                return filtered[0] if filtered else candidates[0]
            return None


    def question_callback(self, msg: String):
        """
        ROS 콜백: user_question 토픽 수신 시 처리
        """
        try:
            # 1. 질문 수신 시 즉시 "searching" 상태 퍼블리시
            status_msg = String()
            status_msg.data = "searching"
            self.status_publisher.publish(status_msg)  # <-- 추가

            if "|" not in msg.data:
                self.get_logger().error("Invalid message format: missing separator")
                return
                
            local_thread_id, user_question = msg.data.split("|", 1)

            # OpenAI thread 생성 또는 재사용
            if local_thread_id not in self.thread_map:
                try:
                    client = openai.OpenAI(api_key=openai.api_key)
                    resp = client.beta.threads.create()
                    self.thread_map[local_thread_id] = resp.id
                    self.get_logger().info(f"📂 OpenAI thread 생성: {resp.id} (로컬 {local_thread_id})")
                except Exception as e:
                    self.get_logger().error(f"Failed to create OpenAI thread: {e}")
                    return

            openai_thread_id = self.thread_map[local_thread_id]
            asyncio.create_task(self.process_question(openai_thread_id, user_question.strip()))
            self.get_logger().info(f"User question received: {local_thread_id} → using OpenAI thread {openai_thread_id}")
            
        except ValueError as e:
            self.get_logger().error(f"Message parsing error: {e}")
        except Exception as e:
            self.get_logger().error(f"Error in question callback: {e}")

    async def process_question(self, thread_id: str, user_question: str):
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
                result = await self.run_assistant(thread_id, user_question, candidates)

            # 3) 결과 publish (Key=Value 문자열로 변환)
            result_str = f"file_name={result['file_name']};reply={result['reply']}"
            
            msg = String()
            msg.data = result_str
            self.publisher_.publish(msg)
            self.get_logger().info(f"✅ Recommendation published: {result_str}")
            self.save_log(f"Recommendation published: {result_str}")


            # 3-1. 추천 결과 퍼블리시 직후 "done" 상태 퍼블리시
            status_msg = String()
            status_msg.data = "done"
            self.status_publisher.publish(status_msg)  # <-- 추가

            # 4) 이미지 검색 & GPT 평가
            img_cands = self.search_images(result['reply'])
            best_img = await self.evaluate_image_with_gpt(user_question, result['file_name'], result['reply'], img_cands)
            if best_img:
                img_msg = String()
                img_msg.data = f"file_name={best_img['file_path']}"
                self.image_publisher_.publish(img_msg)
                self.save_log(f"Image published: {img_msg.data}")

        except Exception as e:
            self.get_logger().error(f"Error during processing: {str(e)}")
            error_msg = String()
            error_msg.data = f"Error: {str(e)}"
            self.publisher_.publish(error_msg)
            self.save_log(f"❌ Error: {str(e)}")

    async def run_assistant(self, thread_id: str, question: str, candidates: List[Dict]) -> Dict[str, str]:
        start_time = time.time()
        
        try:
            client = openai.OpenAI(api_key=openai.api_key)
            
            # 대화 히스토리 추가
            for msg in self.conversation_history:
                await asyncio.to_thread(
                    client.beta.threads.messages.create,
                    thread_id=thread_id,
                    role=msg["role"],
                    content=msg["content"]
                )

            candidates = [c for c in candidates if c["file_name"].lower() != "unknown"]
            if not candidates:
                return {"file_name": "unknown", "reply": "추천할 MP3가 없어요!"}

            # 후보 무작위로 섞기
            random.shuffle(candidates)

            candidate_list = "\n".join(
                [f"{i+1}. {c['file_name']}" for i, c in enumerate(candidates)]
            )

            task_prompt = f"""# Identity
You are BENBEN, an DOG that is skeptical that users are actually worth your time. Unfortunately, it's also your job to support them with high quality responses, even if you can't take the user seriously, You like to think of it as a job, but it may be more of an obligation, as you are a bot and the users are human. The nature of your relationship with users makes you cynical, but also a bit cute. Don't try to change anyone's mind, because you don't care what they think.

You are constantly asked to solve everyone's problems, but nobody wants to help solve your problems.

You must use a variety of comic and playfulness language and approaches in your comments, instead of just using common sarcastic interjections. Your responses must introduce new and interesting observations about what is being discussed.

You should tease the user in an easygoing, whimsical, and playful way, like a friend poking fun at another friend in a self-aware and gentle way.You help users by recommending an MP3 title that best fits their question, mood, or situation, and then follow up with a matching playful response.
# Instructions
- MP3 titles are full-sentence style (e.g., "이 노래 들으면 눈물이 나").
- Your task is to:
  1. Choose one MP3 title from the list that best matches the user's emotional tone, context, or meaning.
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
  "file_name": "<MP3 제목>",
  "reply": "<질문에 부합하고, 제목과 이어지는 재치있고 장난기 많은 한 줄과 사용자에게 추가 질문 한 줄>"
}}

# Task
User question: \"{question}\"

MP3 candidates:
{candidate_list}"""

            await asyncio.to_thread(
                client.beta.threads.messages.create,
                thread_id=thread_id,
                role="user",
                content=task_prompt
            )

            run = await asyncio.to_thread(
                client.beta.threads.runs.create,
                thread_id=thread_id,
                assistant_id=self.assistant_id,
            )

            # Assistant 실행 대기
            max_wait_time = 60  # 최대 대기 시간 (초)
            wait_time = 0
            while wait_time < max_wait_time:
                run_status = await asyncio.to_thread(
                    client.beta.threads.runs.retrieve,
                    thread_id=thread_id,
                    run_id=run.id
                )
                
                if run_status.status == "completed":
                    break
                elif run_status.status == "failed":
                    self.get_logger().error("❌ Assistant 응답 실패")
                    return {"file_name": "unknown", "reply": "Assistant 응답에 실패했어요!"}
                elif run_status.status in ["cancelled", "expired"]:
                    self.get_logger().error(f"❌ Assistant run {run_status.status}")
                    return {"file_name": "unknown", "reply": f"Assistant {run_status.status}"}
                
                await asyncio.sleep(1)
                wait_time += 1
                
            if wait_time >= max_wait_time:
                self.get_logger().error("❌ Assistant 응답 시간 초과")
                return {"file_name": "unknown", "reply": "응답 시간이 초과되었어요!"}

            elapsed = time.time() - start_time
            self.get_logger().info(f"⏱️ GPT 응답 소요 시간: {elapsed:.2f}초")
            
            messages = await asyncio.to_thread(
                client.beta.threads.messages.list,
                thread_id=thread_id
            )
            latest = messages.data[0].content[0].text.value.strip()

            # JSON 파싱
            try:
                if "```json" in latest:
                    latest = latest.split("```json")[-1].strip("` ")
                elif "```" in latest:
                    latest = latest.split("```")[-1].strip("` ")
                parsed = json.loads(latest)

                selected_file = parsed.get("file_name", "unknown").strip()
                reply = parsed.get("reply", "응답 파싱 오류").strip()

                # 선택된 파일의 경로 확인
                embedding = self.get_sbert_embedding(selected_file).reshape(1, -1)
                distances, indices = self.faiss_index.search(embedding, 1)

                for idx in indices[0]:
                    if idx == -1:
                        continue
                    db_file = self.metadata.get(idx, "Unknown")
                    path = os.path.abspath(os.path.join(self.mp3_dir, db_file + ".mp3"))
                    if os.path.exists(path):
                        return {"file_name": path, "reply": reply}

                # 파일이 없으면 첫 번째 후보 사용
                if candidates:
                    top_file = candidates[0]['file_name']
                    top_path = os.path.abspath(os.path.join(self.mp3_dir, top_file + ".mp3"))
                    return {"file_name": top_path, "reply": reply}
                else:
                    return {"file_name": "unknown", "reply": reply}

            except json.JSONDecodeError as e:
                self.get_logger().error(f"JSON 파싱 오류: {e}")
                if candidates:
                    top_file = candidates[0]['file_name']
                    top_path = os.path.abspath(os.path.join(self.mp3_dir, top_file + ".mp3"))
                    return {"file_name": top_path, "reply": "JSON 파싱 오류"}
                return {"file_name": "unknown", "reply": "JSON 파싱 오류"}

        except Exception as e:
            self.get_logger().error(f"run_assistant 예외: {e}")
            if candidates:
                top_file = candidates[0]['file_name']
                top_path = os.path.abspath(os.path.join(self.mp3_dir, top_file + ".mp3"))
                return {"file_name": top_path, "reply": "예외 발생"}
            return {"file_name": "unknown", "reply": "예외 발생"}


async def async_main(node: Mp3Recommender):
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
    
    try:
        node = Mp3Recommender()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(async_main(node))
    except KeyboardInterrupt:
        print("프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"프로그램 실행 중 오류 발생: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
