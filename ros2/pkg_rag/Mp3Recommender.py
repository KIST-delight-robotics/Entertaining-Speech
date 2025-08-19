
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

import re

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

        # # mp3 인덱스/메타
        # self.mp3_db_path = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/mp3_database_new_plus.db"
        # self.mp3_faiss_index_file = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/faiss_index_mp3_new_plus.bin"
        # self.mp3_dir = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/mp3_database_new_plus"
        

        # mp3 인덱스/메타
        self.mp4_db_path = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/mp4_database_mp4.db"
        self.mp4_faiss_index_file = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/faiss_index_mp4.bin"
        self.mp4_dir = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/mp4_database(특수문자제외ver2)"
        



        # # 이미지 인덱스/메타
        # self.image_db_path = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/image_database_plus.db"
        # self.image_faiss_index_file = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/faiss_index_image_plus.bin"
        # self.image_dir = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/image_database_plus"

        # 인덱스와 메타데이터 로드
        try:
            self.mp4_faiss_index = self.load_faiss_index_mp4()
            self.mp4_metadata = self.load_metadata_mp4()
            # self.image_faiss_index = self.load_faiss_index_image()
            # self.image_metadata = self.load_metadata_image()
        except Exception as e:
            self.get_logger().error(f"Failed to load indices or metadata: {e}")
            raise

        # Thread 관리를 위한 딕셔너리
        self.thread_map = {}
        
        # 대화 히스토리
        self.conversation_history = []
        
        # FAISS 인덱스와 메타데이터 별칭 (기존 코드 호환성)
        self.faiss_index = self.mp4_faiss_index
        self.metadata = self.mp4_metadata

        # ROS2 pub/sub
        # self.publisher_ = self.create_publisher(String, 'recommended_mp3', 10)
        self.publisher_ = self.create_publisher(String, 'recommended_mp4', 10)
        # 🆕 TTS 요청용 퍼블리시 추가
        self.tts_publisher = self.create_publisher(String, 'tts_request', 10)
        # self.image_publisher_ = self.create_publisher(String, 'recommended_image', 10)
        self.subscription_ = self.create_subscription(String, 'user_question', self.question_callback, 10)
        self.status_publisher = self.create_publisher(String, 'mp4_recommend_status', 10)
        self.effect_stop_publisher_ = self.create_publisher(String, 'effect_stop', 10)
        self.get_logger().info("mp4Recommender node has started.")

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

            
            # # 여러 개 파일 한꺼번에 업로드
            # file_objs = [open(p, "rb") for p in self.pdf_paths]
            # batch = self.sync_client.vector_stores.file_batches.upload_and_poll(
            #     vector_store_id=vs.id, files=file_objs
            # )
            # self.get_logger().info(f"📑 upload_and_poll → {batch.status} {batch.file_counts}")


            # 2) 한 파일씩 업로드 후 poll
            for pdf in self.pdf_paths:
                with open(pdf, "rb") as f:
                    self.get_logger().info(f"⬆️ '{os.path.basename(pdf)}' 업로드 시작")
                    batch = self.sync_client.vector_stores.file_batches.upload_and_poll(
                        vector_store_id=vs.id,
                        files=[f],          # 리스트지만 한 파일만
                        poll_interval=2.0,  # 초
                        timeout=300         # 5 분까지 기다림
                    )
                    self.get_logger().info(f"✅ {pdf} → {batch.status}")


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

            

            
    def load_metadata_mp4(self) -> Dict[int, str]:
        start = time.time()
        try:
            conn = sqlite3.connect(self.mp4_db_path)
            cur = conn.cursor()
            cur.execute("SELECT id, file_name FROM mp4_files")
            meta = {row[0]: row[1] for row in cur.fetchall()}
            conn.close()
            self.save_log(f"mp4 metadata loaded in {time.time()-start:.4f}s")
            return meta
        except sqlite3.Error as e:
            self.get_logger().error(f"Database error loading mp4 metadata: {e}")
            raise
        except Exception as e:
            self.get_logger().error(f"Error loading mp4 metadata: {e}")
            raise

 

    def load_faiss_index_mp4(self):
        try:
            if os.path.exists(self.mp4_faiss_index_file):
                idx = faiss.read_index(self.mp4_faiss_index_file)
                if isinstance(idx, faiss.IndexIDMap):
                    self.save_log("mp4 FAISS index loaded successfully")
                    return idx
            raise FileNotFoundError(f"mp4 FAISS index not found at {self.mp4_faiss_index_file}")
        except Exception as e:
            self.get_logger().error(f"Error loading mp4 FAISS index: {e}")
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

    def search_candidates(self, query: str, k: int = 5) -> List[Dict]:
        try:
            emb = self.get_sbert_embedding(query).reshape(1, -1)
            D, I = self.mp4_faiss_index.search(emb, k)
            cands = []
            for dist, idx in zip(D[0], I[0]):
                if idx < 0:
                    continue
                fn = self.mp4_metadata.get(idx)
                if not fn:
                    continue
                path = os.path.join(self.mp4_dir, fn + ".mp4")
                cands.append({"file_name": fn, "path": path, "score": float(dist), "index": idx})
            return cands
        except Exception as e:
            self.get_logger().error(f"Error searching candidates: {e}")
            return []




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
        실제 질의 처리 & GPT 호출 & 추천 결과 Publish + TTS 요청
        """
        try:
            # 1) SBERT 임베딩 & FAISS 검색
            query_embedding = self.get_sbert_embedding(user_question.strip()).reshape(1, -1)
            distances, indices = self.faiss_index.search(query_embedding, 3)

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
                    "reply": "No suitable mp4 found"
                }
            else:
                result = await self.run_assistant(thread_id, user_question, candidates)

            # 4) 결과 publish (file_name, reply)
           
            result_str = f"file_name={result['file_name']};reply={result['reply']}"
            msg = String()
            msg.data = result_str
            self.publisher_.publish(msg)


            # 🆕 TTS 요청 별도 퍼블리시
            if result['reply'] and result['reply'] != "No suitable mp4 found":
                tts_msg = String()
                tts_msg.data = result['reply']
                self.tts_publisher.publish(tts_msg)
                self.get_logger().info(f"🗣️ TTS 요청 전송: {result['reply']}")

            self.get_logger().info(f"✅ Recommendation published: {result_str}")
            self.save_log(f"[📩Q] {user_question.strip()} → [🎧mp4] {result['file_name']} | [🗣TTS] {result['reply']}")


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
    {{"need_pdf_search": true/false}}
"""
            

            # {{"need_pdf_search": true/false, "reason": "판단 이유를 한 줄로"}}
            response = await client.chat.completions.create(
                model="gpt-4.1-nano",
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

                # reason = result.get("reason", "")
                
                # self.get_logger().info(f"🎯 PDF Search Decision: {need_pdf} - {reason}")
                self.get_logger().info(f"🎯 PDF Search Decision: {need_pdf}")
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
                return {"file_name": "unknown", "reply": "추천할 mp4가 없어요!"}

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
    2. Choose an appropriate mp4 that matches the mood
    3. Craft a snarky but helpful response using the **exact** PDF information, formatted in 1–2 short and brief sentences

    Remember: You're a grumpy robot dog who secretly cares. Be specific with directions but maintain your personality.
    
    Respond ONLY with valid JSON:
    {{
    "file_name": "<mp4 제목>",
    "reply": "<PDF 정보를 포함한 재치있고 간결한 응답과 추가 질문>"
    }}

    User question: "{question}"

    mp4 candidates:
    {candidate_list}"""
            else:
                # PDF 검색이 필요없다고 판단한 경우
                task_prompt = f"""{identity_section}
 # Instructions
    - mp4 titles are full-sentence style (e.g., "이건 너무한거 아니냐고").
    - Your task is to:
    1. Choose one mp4 title from the list that best matches the user's emotional tone, context, or meaning.
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
    "file_name": "<mp4 제목>",
    "reply": "<재치있고 간결한 응답과 추가 질문>"
    }}

    User question: "{question}"

    mp4 candidates:
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

             
                # gpt 선정파일 바로 publish - 파일 존재 체크는 로깅용으로만 사용
                selected_path = os.path.abspath(os.path.join(self.mp4_dir, selected_file + ".mp4"))
                if not os.path.exists(selected_path):
                    self.get_logger().warning(f"⚠️ Selected file does not exist: {selected_path}")

                # 항상 GPT가 선택한 파일명 + .mp4를 return
                return {"file_name": selected_file + ".mp4", "reply": reply}


               
                # # 1단계: GPT 선택 파일명을 FAISS로 재검색하여 가장 유사한 실제 파일 찾기
                # try:
                #     self.get_logger().info(f"🔍 FAISS 재검색 시작: '{selected_file}'")
                    
                #     # 안전한 임베딩 생성
                #     gpt_embedding = self.get_sbert_embedding(selected_file)
                #     if gpt_embedding is None:
                #         raise ValueError("임베딩 생성 실패")
                    
                #     # reshape을 안전하게 처리
                #     gpt_embedding_reshaped = gpt_embedding.reshape(1, -1).astype('float32')
                    
                #     # FAISS 검색
                #     distances, indices = self.faiss_index.search(gpt_embedding_reshaped, 1)
                    
                #     if len(indices) > 0 and len(indices[0]) > 0 and indices != -1:
                #         # FAISS에서 찾은 가장 유사한 파일
                #         verified_idx = int(indices)  # numpy.int64를 int로 변환
                #         verified_file = self.metadata.get(verified_idx, "Unknown")
                #         verified_path = os.path.abspath(os.path.join(self.mp4_dir, verified_file + ".mp4"))
                        
                #         if os.path.exists(verified_path):
                #             # GPT 선택과 FAISS 검증 결과 비교 로깅
                #             similarity_score = float(distances[0][0])  # numpy.float32를 float로 변환
                #             if selected_file != verified_file:
                #                 self.get_logger().info(f"🔄 GPT선택: '{selected_file}' → FAISS검증: '{verified_file}' (유사도: {similarity_score:.4f})")
                #             else:
                #                 self.get_logger().info(f"✅ GPT 선택 파일명 검증 성공: '{selected_file}'")
                            
                #             return {"file_name": verified_path, "reply": reply}
                #         else:
                #             self.get_logger().warning(f"⚠️ FAISS 검증 파일이 존재하지 않음: {verified_path}")
                #     else:
                #         self.get_logger().warning("⚠️ FAISS 검색에서 유효한 결과를 찾지 못함")
                    
                # except Exception as e:
                #     self.get_logger().error(f"❌ FAISS 재검색 실패: {e}")
                #     self.get_logger().error(f"📍 오류 발생 시점: selected_file='{selected_file}'")

                # # 2단계: FAISS 검증 실패시 candidates에서 정확 매칭 시도
                # for candidate in candidates:
                #     if candidate['file_name'] == selected_file:
                #         candidate_path = os.path.abspath(os.path.join(self.mp4_dir, candidate['file_name'] + ".mp4"))
                #         if os.path.exists(candidate_path):
                #             self.get_logger().info(f"🎯 Candidates에서 정확 매칭: '{selected_file}'")
                #             return {"file_name": candidate_path, "reply": reply}

                # # 3단계: 모든 검증 실패시 첫 번째 후보 사용 (안전장치)
                # if candidates:
                #     fallback_file = candidates[0]['file_name']
                #     fallback_path = os.path.abspath(os.path.join(self.mp4_dir, fallback_file + ".mp4"))
                #     self.get_logger().warning(f"🚨 Fallback to first candidate: '{fallback_file}'")
                #     return {"file_name": fallback_path, "reply": reply}

                # # 4단계: 모든 것이 실패한 경우
                # self.get_logger().error("❌ 모든 파일명 검증 실패")
                # return {"file_name": "unknown.mp4", "reply": reply}



            except json.JSONDecodeError as e:
                self.get_logger().error(f"JSON 파싱 오류: {e}")
                if candidates:
                    top_file = candidates[0]['file_name']
                    top_path = os.path.abspath(os.path.join(self.mp4_dir, top_file + ".mp4"))
                    return {"file_name": top_path, "reply": "응답을 이해하지 못했어요"}
                return {"file_name": "unknown", "reply": "응답 파싱 오류"}

        except Exception as e:
            self.get_logger().error(f"run_assistant 예외: {e}")
            if candidates:
                top_file = candidates[0]['file_name']
                top_path = os.path.abspath(os.path.join(self.mp4_dir, top_file + ".mp4"))
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


