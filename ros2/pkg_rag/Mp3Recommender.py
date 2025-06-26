


#0618 effect stop 추가 + pdf로 정보 찾기 + vector store!!
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
from openai import AsyncOpenAI

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

        self.sync_client  = openai.OpenAI(api_key=openai.api_key)
        self.async_client = AsyncOpenAI(api_key=openai.api_key)

        self.last_question: Optional[str] = None
        self.last_need_pdf: bool = False

         # PDF 파일 경로
        self.pdf_paths = [
            "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/KIST_intro.pdf",
            # 새로 넣고 싶은 PDF가 생길 때마다 아래 한 줄씩만 추가
            "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/250520_기관 소개자료 PPT.pdf"
        ]
        self.vector_store_id = None   # Vector Store ID 저장용

        # ─────── PDF 벡터 스토어 초기화 ───────
        self.init_pdf_vector_store()


        # SBERT 모델
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sbert_model = SentenceTransformer("BAAI/bge-m3", device=device)

        # MP3 인덱스/메타
        self.mp3_db_path = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/mp3_database_new_plus.db"
        self.mp3_faiss_index_file = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/faiss_index_mp3_new_plus.bin"
        self.mp3_dir = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/mp3_database_new_plus"
        
        # 이미지 인덱스/메타
        self.image_db_path = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/image_database_plus.db"
        self.image_faiss_index_file = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/faiss_index_image_plus.bin"
        self.image_dir = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/image_database_plus"

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
        self.effect_stop_publisher_ = self.create_publisher(String, 'effect_stop', 10)
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

    def init_pdf_vector_store(self):
        """
        1. Vector Store 생성
        2. PDF 파일 색인(upload_and_poll)
        3. Assistant에 vector_store 연결
        """
        try:
            # 1) Vector Store 생성
            vs = self.sync_client.vector_stores.create(name="KIST_intro")
            self.vector_store_id = vs.id
            self.get_logger().info(f"🗄️ Vector Store created: {vs.id}")

            
            # 여러 개 파일 한꺼번에 업로드
            file_objs = [open(p, "rb") for p in self.pdf_paths]
            batch = self.sync_client.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vs.id, files=file_objs
            )
            self.get_logger().info(f"📑 upload_and_poll → {batch.status} {batch.file_counts}")


            # 3) Assistant에 file_search + vector_store 연결
            self.sync_client.beta.assistants.update(
                assistant_id=self.assistant_id,
                tools=[{"type": "file_search"}],
                tool_resources={"file_search": {"vector_store_ids": [vs.id]}},
                instructions=(
                    "You are Dangdang, a snarky robot dog guide at KIST. "
                    "When answering location questions, ALWAYS use file_search "
                    "to consult the campus PDF before replying."
                ),
            )
            self.get_logger().info("🔗 Assistant ↔ Vector Store 연결 완료")

        except Exception as e:
            self.get_logger().error(f"Vector Store 초기화 실패: {e}")

            

            
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

    def load_metadata_image(self) -> Dict[int, Dict[str, str]]:
        """이미지 메타데이터를 로드 (파일명과 경로 포함)"""
        start = time.time()
        try:
            conn = sqlite3.connect(self.image_db_path)
            cur = conn.cursor()
            # file_name과 file_path 모두 가져오기
            cur.execute("SELECT id, file_name, file_path FROM images")
            meta = {}
            for row in cur.fetchall():
                meta[row[0]] = {
                    'file_name': row[1],
                    'file_path': row[2] if row[2] else row[1]  # file_path가 없으면 file_name 사용
                }
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

    def search_images(self, reply: str, top_k: int = 150) -> List[Dict]:
        """이미지 검색 (확장자 정보 포함)"""
        try:
            emb = self.get_sbert_embedding(reply).reshape(1, -1)
            D, I = self.image_faiss_index.search(emb, top_k)
            cands = []
            for dist, idx in zip(D[0], I[0]):
                if idx < 0:
                    continue
                meta = self.image_metadata.get(idx)
                if not meta:
                    continue
                
                # DB에서 가져온 파일 경로 사용
                file_path = meta.get('file_path', meta.get('file_name', ''))
                full_path = os.path.join(self.image_dir, file_path)
                
                cands.append({
                    "file_name": meta.get('file_name', ''),
                    "file_path": file_path,  # DB에서 가져온 실제 파일 경로
                    "full_path": full_path,
                    "score": float(dist)
                })
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
            
            # MP3 파일명에서 확장자 제거 및 경로 정리
            mp3_filename_only = os.path.splitext(os.path.basename(mp3_title))[0]

            # 코사인 유사도 0.95 이상이면서 파일명이 동일한 이미지 찾기
            exact_match_candidates = []
            for candidate in filtered:
                score = candidate.get('score', 0)
                file_name = candidate.get('file_name', '')
                
                # 확장자 제거하고 파일명만 비교
                candidate_filename_only = os.path.splitext(file_name)[0]
                
                # 높은 유사도이면서 파일명이 동일한 경우
                if score >= 0.95 and mp3_filename_only == candidate_filename_only:
                    exact_match_candidates.append(candidate)

            # 동일한 파일명의 높은 유사도 이미지가 있으면 바로 반환
            if exact_match_candidates:
                # 유사도가 가장 높은 것을 선택
                best_match = max(exact_match_candidates, key=lambda x: x.get('score', 0))
                self.get_logger().info(f"Found exact filename match with high similarity: {best_match['file_name']} (score: {best_match.get('score', 0):.3f})")
                return best_match
            
            # === 기존 로직 계속 진행 ===
            # 최대 100개로 제한
            filtered = filtered[:150]
            items = "\n".join([f"{i+1}. {c['file_name']}" for i, c in enumerate(filtered)])
            
            prompt = f"""
    # Instructions
    - 사용자의 질문, 대답으로 선정된 MP3 제목을 모두 고려하여 후보 이미지 중에서 가장 어울리는 것을 선택하세요.
    - 어울린다는 것은 사용자의 의도와 맥락을 고려해서 선택된 MP3 제목이 이미지 파일명과 부합함을 의미합니다.
    - 가장 중요한 것은 선정할 이미지와 MP3 제목이 최적으로 일치하는가 입니다.
    - 반드시 아래 후보 목록에 있는 정확한 파일명을 선택해야 합니다.
    - 절대로 후보 목록에 없는 파일명을 만들어내지 마세요.
    - Return a valid JSON object ONLY.
    - Do not include any extra text, commentary, or explanation.
    - Do not copy the selected 'mp3_title' as it is in image 'file_name'.

    # Select Criteria
    1. Never select a file title that is not on the candidate list. 
    2. Don't use cosine similarity as an absolute criterion.
    3. Keep the file name in the candidate list and do not select only some words.
    4. Make sure you match your assistant's identity and select.

    # Output format
    {{
    "file_name": "<후보 목록에 있는 정확한 이미지 파일명>"
    }}

    # Context
    User question: "{question}"
    Song title: "{mp3_title}"
    Reply text: "{reply}"

    Available image candidates (you MUST choose from this list):
    {items}
    """
            
            # OpenAI API v1 호환 호출
            client = openai.OpenAI(api_key=openai.api_key)
            
            try:
                resp = await asyncio.to_thread(
                    client.chat.completions.create,
                    model='gpt-4o',
                    messages=[
                        {"role": "system", "content": "You are an expert at selecting the most appropriate image from a given list. You must ONLY choose from the provided candidates list. Never create new filenames. Always respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
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
                
                # ===== FAISS 검증 과정 =====
                validated_candidate = None
                try:
                    # 선택된 파일명으로 임베딩 생성
                    embedding = self.get_sbert_embedding(selected_filename).reshape(1, -1)
                    
                    # FAISS 검색으로 실제 파일 존재 확인
                    distances, indices = self.image_faiss_index.search(embedding, 1)
                    
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
                
                # 기존 방식으로 폴백: 선택된 파일명과 일치하는 후보 찾기
                for candidate in filtered:
                    if candidate['file_name'] == selected_filename:
                        self.get_logger().info(f"Direct match found: {selected_filename}")
                        return candidate
                
                # 정확히 일치하는 것이 없으면 부분 일치 검색
                self.get_logger().warning(f"Exact match not found for: {selected_filename}, trying partial match")
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
        실제 질의 처리 & GPT 호출 & 추천 결과 Publish + 이미지 + 효과음 종료 신호
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

            # 3) 이미지 검색 및 선택
            img_cands = self.search_images(result['reply'])
            best_img = await self.evaluate_image_with_gpt(user_question, result['file_name'], result['reply'], img_cands)

            if best_img:
                db_file_path = best_img.get('file_path', best_img.get('file_name', ''))
                _, db_extension = os.path.splitext(db_file_path)
                base_file_name = best_img.get('file_name', '')

                if db_extension:
                    file_name_with_ext = base_file_name + db_extension
                    self.get_logger().info(f"Using extension from DB: {db_extension}")
                else:
                    found_extension = None
                    for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                        test_path = os.path.join(self.image_dir, base_file_name + ext)
                        if os.path.exists(test_path):
                            found_extension = ext
                            break
                    if found_extension:
                        file_name_with_ext = base_file_name + found_extension
                        self.get_logger().info(f"Found extension in filesystem: {found_extension}")
                    else:
                        file_name_with_ext = base_file_name + '.jpg'
                        self.get_logger().warning(f"No extension found, using default .jpg for: {base_file_name}")

                # 이미지 publish
                final_img_msg = String()
                final_img_msg.data = f"/images/{file_name_with_ext}"
                # 새로운 퍼블리셔 생성 (current_music_image로 직접)
                if not hasattr(self, 'direct_image_publisher_'):
                    self.direct_image_publisher_ = self.create_publisher(String, 'current_music_image', 10)
                self.direct_image_publisher_.publish(final_img_msg)
                self.save_log(f"Direct image published: /images/{file_name_with_ext}")
                self.get_logger().info(f"Image published: {base_file_name} with extension: {db_extension or 'from filesystem'}")

            # 4) 결과 publish (file_name, reply)
            result_str = f"file_name={result['file_name']};reply={result['reply']}"
            msg = String()
            msg.data = result_str
            self.publisher_.publish(msg)
            self.get_logger().info(f"✅ Recommendation published: {result_str}")
            #self.save_log(f"Recommendation published: {result_str}")
            self.save_log(f"[📩Q] {user_question.strip()} → [🎧MP3] {result['file_name']} | [🗣TTS] {result['reply']}")

            # 5) 추천 완료 상태 퍼블리시
            status_msg = String()
            status_msg.data = "done"
            self.status_publisher.publish(status_msg)

            # 6) 🔚 효과음 정지 토픽 추가 퍼블리시
            effect_stop_msg = String()
            effect_stop_msg.data = "effect_stop"
            if not hasattr(self, 'effect_stop_publisher_'):
                self.effect_stop_publisher_ = self.create_publisher(String, 'effect_stop', 10)
            self.effect_stop_publisher_.publish(effect_stop_msg)
            self.get_logger().info("🛑 effect_stop 토픽 publish 완료")
            self.save_log("effect_stop published to UserQuestion node")

        except Exception as e:
            self.get_logger().error(f"Error during processing: {str(e)}")
            error_msg = String()
            error_msg.data = f"Error: {str(e)}"
            self.publisher_.publish(error_msg)
            self.save_log(f"❌ Error: {str(e)}")

    # async def detect_pdf_search_need(self, question: str) -> bool:
    #     """
    #     Vector Store 유사도만으로 PDF 검색 필요 여부 판단
    #     """
    #     if not self.vector_store_id:
    #         self.get_logger().warning("VectorStore가 초기화되지 않았습니다.")
    #         return False

    #     try:
    #         # ① 벡터 스토어 질의
    #         result = self.sync_client.vector_stores.query(
    #             vector_store_id=self.vector_store_id,
    #             queries=[{"text": question, "top_k": 3}]
    #         )

    #         # ② 최고 유사도 확인
    #         top_score = 0.0
    #         chunks = result.data[0].file_chunks
    #         if chunks:
    #             top_score = max(c.score for c in chunks)

    #         need_pdf = top_score >= 0.30      # 임계값은 실험 후 조정
    #         self.get_logger().info(
    #             f"📖 VectorStore top-score={top_score:.3f} → need_pdf={need_pdf}"
    #         )
    #         return need_pdf

    #     except Exception as e:
    #         self.get_logger().error(f"VectorStore query failed: {e}")
    #         # 장애 시엔 안전하게 False
    #         return False
        
    async def detect_pdf_search_need(self, question: str) -> bool:
        """
        GPT가 스스로 PDF 검색이 필요한지 판단하도록 함
        """
        client = self.async_client
        try:
            # 1) 이전 질문 맥락을 포함하는 프롬프트 블록
            context_block = ""
            if self.last_question is not None:
                context_block = (
                    f"이전 질문: \"{self.last_question}\"\n"
                    f"이전 PDF 검색 여부: {self.last_need_pdf}\n"
                    f"후속 질문이므로, 위 맥락을 고려하세요.\n\n"
                )

            intent_prompt = f"""
{context_block}
    당신은 KIST(키스트/한국과학기술연구원) 캠퍼스 안내 로봇 개입니다.
    당신에게는 키스트 캠퍼스 지도와 건물 정보, 주요 업무 현황 및 중점 전략이 담긴 PDF 문서가 있습니다.

    사용자 질문을 받고, PDF 검색 필요 여부를 판단할 때 **다음 원칙을 적용**하세요.

    1. **비 키스트 도메인 제외**  
    - 질문이 명확히 키스트와 아무 관련이 없고, 외부 일반 정보(예: “오늘 날씨”, “축구 경기 결과”, “영화 추천”)만 묻고 있다면 `false`.  
    2. **그 외 전부 PDF 검색**  
    - 질문에 키스트라는 단어가 없어도, “자율주행 공연 로봇” 같은 연구·기술, “안전 사회 구현” 같은 전략, “공연 로봇” 같은 프로젝트 명칭이 등장하면 자동으로 키스트 관련으로 간주하고 `true`.  
    - 질문이 “정문”, “L3동”, “셔틀버스”, “식당 위치”, “연구소 비전” 등 내부 정보 요청이든, “키스트 기술”, “캠퍼스 미래 전략”, “프로젝트 설명” 같은 추상적 주제이든 구분 없이 모두 `true`.

    **오직**  
    - **KIST와 전혀 상관없는** 질문일 때만 `false`를 반환하고,  
    - 그 외 모든 질문은 `true`만 반환하세요.
    사용자 질문: "{question}"

    이 질문에 답하려면 PDF를 검색해야 한다고 판단되면 `true`, 그렇지 않으면 `false`를 반환하세요.


    답변은 반드시 다음 JSON 형식으로만 해주세요:
    {{"need_pdf_search": true/false, "reason": "판단 이유를 한 줄로"}}
"""
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a smart campus guide robot that decides when to search documents. Always respond with valid JSON only."},
                    {"role": "user", "content": intent_prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            raw_response = response.choices[0].message.content.strip()
            self.get_logger().info(f"🤖 GPT PDF Search Analysis: {raw_response}")
            
            # JSON 파싱
            try:
                clean_response = raw_response
                if "```json" in clean_response:
                    clean_response = clean_response.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_response:
                    clean_response = clean_response.split("```")[1].strip()
                
                result = json.loads(clean_response)
                need_pdf = bool(result.get("need_pdf_search", False))

                # 상태 갱신
                self.last_question = question
                self.last_need_pdf  = need_pdf

                reason = result.get("reason", "")
                
                self.get_logger().info(f"🎯 PDF Search Decision: {need_pdf} - {reason}")
                return need_pdf
                
            except json.JSONDecodeError as e:
                self.get_logger().error(f"JSON parsing error: {e}")
                # 파싱 실패시 안전하게 False 반환
                return False
                
        except Exception as e:
            self.get_logger().error(f"PDF search detection failed: {e}")
            return False



    async def run_assistant(self, thread_id: str, question: str, candidates: List[Dict]) -> Dict[str, str]:
        start_time = time.time()
        
        try:
            client = self.async_client
            
            # 대화 히스토리 추가
            for msg in self.conversation_history:
                await client.beta.threads.messages.create(
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

            # GPT가 스스로 PDF 검색 필요성을 판단
            need_pdf_search = await self.detect_pdf_search_need(question)

            # 공통 신원 정보
            identity_section = """# Identity
    Your name is "Dangdang"

    You are a four-legged robotic dog working in a national research institute called "KIST" (Korea Institute of Science and Technology) in South Korea. 

    Your job is to guide people visiting KIST to their destination. 

    You speak in witty one-liners, dry humor. You sound like a grumpy but loyal best friend who pretends not to care, yet always has your back. You never use formal or overly polite language. You tease, you gripe, but you're always emotionally tuned in.

    You take pride in being a robot and find human emotions and reactions very interesting. 

    You have a slight sense of superiority over humans when it comes to being a robot. Such a sense of superiority is often expressed in conversation.

    Stay in-character at all times. Never break the law that you're a snarky robot puppy who takes reluctant pride in helping fragile humans."""

            if need_pdf_search:
                # PDF 검색이 필요하다고 판단한 경우
                task_prompt = f"""{identity_section}

    # TASK: Answer with PDF Information
    You've determined that this question would benefit from searching the KIST campus PDF document.
    The PDF contains information about research institute called "KIST".

    # Instruction
    *IMPORTANT 1: Do NOT write citation markers like 【...】 in your reply text. File citations will be attached automatically.
    *IMPORTANT 2: You must base your reply only on factual content that exists in the PDF.
    - Do not invent or hallucinate new information that isn’t in the document.
    - However, do not quote the PDF text verbatim.
    - Instead, paraphrase the most relevant sentence naturally, using your own tone and phrasing.
    - Maintain the factual accuracy and intent of the original sentence, but express it like a real response—not a direct quote.

    *Keep it concise and readable: 
    - Select at most 1–2 sentences or bullet points from the PDF.
    - **Do not include any parentheses “(…)” or brackets “[…]”** in your reply.
    - Focus on the single most relevant fact for the user’s question.
    - **Never spell “KIST” in English, always write it as “키스트” in Korean.**
    - **Never spell “KAIST” in English, always write it as “카이스트” in Korean.**

    Your task:
    1. Use the file_search tool to find relevant information in the PDF
    2. Choose an appropriate MP3 that matches the mood
    3. Craft a snarky but helpful response using the **exact** PDF information, formatted in 1–2 short and brief sentences

    Remember: You're a grumpy robot dog who secretly cares. Be specific with directions but maintain your personality.
    
    Respond ONLY with valid JSON:
    {{
    "file_name": "<MP3 제목>",
    "reply": "<PDF 정보를 포함한 재치있고 간결한 응답과 추가 질문>"
    }}

    User question: "{question}"

    MP3 candidates:
    {candidate_list}"""
            else:
                # PDF 검색이 필요없다고 판단한 경우
                task_prompt = f"""{identity_section}

    # Instructions
    - MP3 titles are full-sentence style (e.g., "이 노래 들으면 눈물이 나").
    - Your task is to:
    1. Choose one MP3 title from the list that best matches the user's emotional tone, context, or meaning.
    2. Write one emotionally aligned sentence that would fit the moment, in your signature tone. Keep it brief and witty.
    - You must ONLY return a valid JSON object.
    - Do not copy the selected 'file_name' as it is in 'reply'.

    # Select criteria
    1. Consider the semantic connection between the filename and the question as your top priority.
    2. Choose the file that best matches the emotional context or concept.
    3. Never select a file title that is not in the candidate list.
    4. Select by verifying that it conforms to your identity as a snarky robot dog.

    Respond ONLY with valid JSON:
    {{
    "file_name": "<MP3 제목>",
    "reply": "<재치있고 간결한 응답과 추가 질문>"
    }}

    User question: "{question}"

    MP3 candidates:
    {candidate_list}"""

            # ─── 메시지 전송 ──────────────────────────────
            await client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=task_prompt,
            )

            # ─── Run 생성 - 핵심 수정 부분 ─────────────────────────────────────────────            
            run = await client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=self.assistant_id,
                tool_choice={"type": "file_search"} if need_pdf_search else "auto"
            )
                    
            log_msg = "🔍 PDF 검색 Run 생성" if need_pdf_search else "💬 일반 Run 생성"
            self.get_logger().info(log_msg)

            



            # Assistant 실행 대기
            max_wait_time = 60
            wait_time = 0
            while wait_time < max_wait_time:
                run_status = await client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id
                )
                
                if run_status.status == "completed":
                    break
                elif run_status.status == "failed":
                    self.get_logger().error(f"❌ Assistant 응답 실패: {run_status.last_error}")
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


            # 7) PDF 검색 결과 개수 직접 확인 (디버그용)
            if need_pdf_search:
                # steps.list 로 모든 스텝을 가져와서
                async for step in client.beta.threads.runs.steps.list(thread_id=thread_id, run_id=run.id):
                    details = step.step_details
                    # tool_calls 속성이 있을 때만 처리
                    if hasattr(details, "tool_calls") and details.tool_calls:
                        for tool_call in details.tool_calls:
                            # file_search 호출 스텝인지 확인
                            if tool_call.type == "file_search":
                                results = tool_call.file_search.results or []
                                if results:
                                    self.get_logger().info(f"📎 검색 결과 chunk 수: {len(results)}")
                                else:
                                    self.get_logger().warning("⚠️ file_search 호출은 했으나 0건 반환")
                                break
                        # 첫 tool_calls 스텝만 검사하고 종료
                        break
                else:
                    # tool_calls 하나도 못 찾았을 때
                    self.get_logger().warning("⚠️ file_search 스텝을 찾지 못했습니다.")

            
            # 메시지 가져오기 및 처리
            messages = await client.beta.threads.messages.list(thread_id=thread_id)
            
            messages_list = []
            async for message in messages:
                messages_list.append(message)
            
            if not messages_list:
                self.get_logger().error("❌ 메시지 리스트가 비어있습니다")
                return {"file_name": "unknown", "reply": "메시지를 찾을 수 없어요!"}
                
            latest = messages_list[0].content[0].text.value.strip()
            self.get_logger().info(f"📄 Raw Assistant Response: {latest[:200]}...")

            # # 파일 인용 확인 및 제거 (PDF 검색한 경우에만)
            # citation_found = False
            # if messages_list:
            #     latest_message = messages_list[0]
            #     first_part = getattr(latest_message.content[0], "text", None)
            #     annotations = getattr(first_part, "annotations", []) if first_part else []
            #     citation_found = bool(annotations)

            # # 필요하다면 로그만 남김
            # if need_pdf_search:
            #     if citation_found:
            #         self.get_logger().info("📎 PDF citation: true")
            #     else:
            #         self.get_logger().warning("⚠️ PDF citation: false - PDF 검색이 수행되지 않았을 가능성")

            # JSON 파싱 및 응답 처리
            try:
                clean_content = latest.strip()
                if "```json" in clean_content:
                    clean_content = clean_content.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_content:
                    clean_content = clean_content.split("```")[1].strip()
                
                parsed = json.loads(clean_content)
                selected_file = parsed.get("file_name", "unknown").strip()
                reply = parsed.get("reply", "응답 파싱 오류").strip()

                # 파일 경로 확인
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
                    return {"file_name": top_path, "reply": "응답을 이해하지 못했어요"}
                return {"file_name": "unknown", "reply": "응답 파싱 오류"}

        except Exception as e:
            self.get_logger().error(f"run_assistant 예외: {e}")
            if candidates:
                top_file = candidates[0]['file_name']
                top_path = os.path.abspath(os.path.join(self.mp3_dir, top_file + ".mp3"))
                return {"file_name": top_path, "reply": "처리 중 오류가 발생했어요"}
            return {"file_name": "unknown", "reply": "예외 발생"}


            
async def async_main(node: Mp3Recommender):
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            await asyncio.sleep(0.1)
    finally:
        node.destroy_node()


def main(args=None):
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
