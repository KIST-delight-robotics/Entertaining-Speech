

# ──────────────────────────────────────────────────────────────────────────────
# 0911 local rag 방식 도입
# ──────────────────────────────────────────────────────────────────────────────




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










        # 🆕 로컬 RAG 설정
        self.local_rag_index = None
        self.local_rag_metadata = []
        
        # 로컬 RAG 파일 경로
        self.text_corpus_paths = [
            "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/rag_file/KIST_location_corpus.json",
            "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/rag_file/KIST_PPT_corpus.json",
        ]
        self.tables_path = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/rag_file/KIST_location_tables.json"
        
        # 🆕 임베딩 캐시 (속도 최적화)
        self.embedding_cache = {}






        # 🆕 유사도 임계값 설정
        self.SIMILARITY_THRESHOLD = 0.7  # 코사인 유사도 임계값 (0~1, 높을수록 엄격)
        self.MIN_CANDIDATES_FOR_VIDEO = 1  # 비디오 재생을 위한 최소 후보 수
        # 🆕 로컬 RAG 시스템 초기화
        self.init_local_rag()





        self.get_logger().info("mp4Recommender node has started.")
        




# 🟢 __init__ 함수 다음에 이 함수들을 추가하세요

    def init_local_rag(self):
        """로컬 RAG 시스템 초기화"""
        rag_index_path = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/local_rag_index.faiss"
        rag_metadata_path = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/local_rag_metadata.json"
        
        if os.path.exists(rag_index_path) and os.path.exists(rag_metadata_path):
            # 기존 인덱스 로드
            try:
                self.local_rag_index = faiss.read_index(rag_index_path)
                with open(rag_metadata_path, "r", encoding="utf-8") as f:
                    self.local_rag_metadata = json.load(f)
                self.get_logger().info(f"✅ 로컬 RAG 인덱스 로드: {len(self.local_rag_metadata)}개 문서")
                return
            except Exception as e:
                self.get_logger().warning(f"⚠️ 기존 인덱스 로드 실패, 재구축: {e}")
        
        # 새로 구축
        self.get_logger().info("🔧 로컬 RAG 인덱스 새로 구축")
        self.build_local_rag_index()

    def build_unified_corpus(self):
        """3개 JSON 파일을 통합하여 단일 검색 코퍼스 구축"""
        unified_corpus = []
        
        # 1) 텍스트 문서들 (location, PPT)
        for corpus_path in self.text_corpus_paths:
            try:
                with open(corpus_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                for chunk in data["chunks"]:
                    unified_corpus.append({
                        "id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "doc_type": "text",
                        "source": chunk.get("doc_id", "unknown"),
                        "section": chunk.get("section", "일반"),
                        "page": chunk.get("page_start"),
                        "slide": chunk.get("slide_start")
                    })
            except Exception as e:
                self.get_logger().error(f"❌ 코퍼스 로드 실패 {corpus_path}: {e}")
                continue
        
        # 2) 표 데이터 (시간표)
        try:
            with open(self.tables_path, "r", encoding="utf-8") as f:
                tables_data = json.load(f)
            
            for table in tables_data["tables"]:
                # 표를 텍스트로 변환
                table_text = f"{table['label']} 시간표: {', '.join(table['times'])}"
                unified_corpus.append({
                    "id": table["table_id"],
                    "text": table_text,
                    "doc_type": "table",
                    "source": "KIST_location_tables",
                    "section": "시간표",
                    "times": table["times"],  # 원본 시간 리스트 보존
                    "label": table["label"]
                })
        except Exception as e:
            self.get_logger().warning(f"⚠️ 테이블 로드 실패: {e}")
        
        self.get_logger().info(f"📚 통합 코퍼스 구축 완료: {len(unified_corpus)}개 문서")
        return unified_corpus

    def build_local_rag_index(self):
        """로컬 RAG용 FAISS 인덱스 구축"""
        start_time = time.time()
        
        try:
            # 1) 통합 코퍼스 로드
            corpus = self.build_unified_corpus()
            if not corpus:
                raise RuntimeError("코퍼스가 비어있습니다.")
            
            # 2) 임베딩 생성
            embeddings = []
            metadata = []
            
            for i, doc in enumerate(corpus):
                # 텍스트 임베딩
                emb = self.get_cached_embedding(doc["text"])
                embeddings.append(emb)
                metadata.append(doc)
                
                if (i + 1) % 10 == 0:
                    self.get_logger().info(f"📊 임베딩 진행: {i + 1}/{len(corpus)}")
            
            # 3) FAISS 인덱스 생성
            embeddings_matrix = np.vstack(embeddings).astype("float32")
            dimension = embeddings_matrix.shape[1]
            
            # Inner Product 인덱스 (코사인 유사도)
            index = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIDMap(index)
            
            # ID 매핑으로 인덱스 추가
            ids = np.arange(len(corpus)).astype("int64")
            index.add_with_ids(embeddings_matrix, ids)
            
            # 4) 디스크에 저장
            rag_index_path = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/local_rag_index.faiss"
            rag_metadata_path = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/local_rag_metadata.json"
            
            faiss.write_index(index, rag_index_path)
            
            with open(rag_metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            self.local_rag_index = index
            self.local_rag_metadata = metadata
            
            build_time = time.time() - start_time
            self.get_logger().info(f"✅ 로컬 RAG 인덱스 구축 완료: {build_time:.2f}초, {len(corpus)}개 문서")
            
        except Exception as e:
            self.get_logger().error(f"❌ 로컬 RAG 인덱스 구축 실패: {e}")
            raise

    def get_cached_embedding(self, text: str) -> np.ndarray:
        """임베딩 캐싱으로 속도 향상"""
        text_hash = hash(text)
        if text_hash not in self.embedding_cache:
            self.embedding_cache[text_hash] = self.get_sbert_embedding(text)
        return self.embedding_cache[text_hash]

    def search_local_rag(self, query: str, k: int = 5) -> List[Dict]:
        """로컬 RAG 검색 (테이블 + 텍스트 통합)"""
        start_time = time.time()
        
        try:
            # 1) 쿼리 임베딩
            query_emb = self.get_cached_embedding(query).reshape(1, -1)
            
            # 2) FAISS 검색
            scores, indices = self.local_rag_index.search(query_emb, k * 2)  # 넉넉히 가져오기
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                    
                metadata = self.local_rag_metadata[idx]
                results.append({
                    "score": float(score),
                    "text": metadata["text"],
                    "doc_type": metadata["doc_type"],
                    "source": metadata["source"],
                    "section": metadata.get("section", ""),
                    "times": metadata.get("times", []),  # 표 데이터용
                    "label": metadata.get("label", ""),
                    "page": metadata.get("page"),
                    "slide": metadata.get("slide")
                })
            
            # 3) 다양성 확보 (간단한 MMR)
            final_results = self.apply_mmr(results, k, query)
            
            search_time = time.time() - start_time
            self.get_logger().info(f"🔍 로컬 RAG 검색 완료: {len(final_results)}개 결과, {search_time:.3f}초")
            
            return final_results
            
        except Exception as e:
            self.get_logger().error(f"Error in local RAG search: {e}")
            return []

    def apply_mmr(self, candidates: List[Dict], k: int, query: str, lambda_param: float = 0.7) -> List[Dict]:
        """Maximal Marginal Relevance 적용"""
        if len(candidates) <= k:
            return candidates[:k]
        
        selected = [candidates[0]]  # 첫 번째는 스코어 기준 최고
        remaining = candidates[1:]
        
        while len(selected) < k and remaining:
            best_idx = 0
            best_score = -float('inf')
            
            for i, candidate in enumerate(remaining):
                # 원본 유사도
                relevance = candidate["score"]
                
                # 이미 선택된 것들과의 중복도 계산
                max_similarity = 0.0
                for selected_doc in selected:
                    # 섹션 기반 중복도 (간단화)
                    if candidate["section"] == selected_doc["section"]:
                        max_similarity = max(max_similarity, 0.8)
                    elif candidate["doc_type"] == selected_doc["doc_type"]:
                        max_similarity = max(max_similarity, 0.5)
                
                # MMR 점수
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return selected

    async def generate_local_rag_response(self, question: str, search_results: List[Dict]) -> str:
        """로컬 RAG 결과를 바탕으로 GPT 답변 생성"""
        
        # 1) 검색 결과를 컨텍스트로 구성
        context_blocks = []
        table_info = []
        
        for result in search_results:
            if result["doc_type"] == "table":
                # 시간표 정보
                table_info.append({
                    "label": result["label"],
                    "times": result["times"]
                })
            else:
                # 텍스트 정보
                source_info = ""
                if result.get("page"):
                    source_info = f"(Page {result['page']})"
                elif result.get("slide"):
                    source_info = f"(Slide {result['slide']})"
                
                context_blocks.append({
                    "text": result["text"][:500],  # 길이 제한
                    "section": result["section"],
                    "source": source_info
                })
        
        # 2) GPT 프롬프트 구성
        context_text = ""
        
        # 시간표 우선 표시
        if table_info:
            context_text += "## 시간표 정보\n"
            for table in table_info:
                times_str = ", ".join(table["times"])
                context_text += f"- {table['label']}: {times_str}\n"
            context_text += "\n"
        
        # 텍스트 정보
        if context_blocks:
            context_text += "## 관련 정보\n"
            for i, block in enumerate(context_blocks, 1):
                context_text += f"{i}. [{block['section']}] {block['text']} {block['source']}\n"
        
        # 3) GPT API 호출
        prompt = f"""# Identity
Your name is "Dangdang"
You are a four-legged robotic dog working at KIST (Korea Institute of Science and Technology).
You help visitors with campus information in a snarky but ultimately helpful way.

# Task
Answer the user's question using ONLY the provided context information.
- Keep it concise (1-2 sentences)
- Be specific and factual
- Use "키스트" instead of "KIST"
- For time information, format as "시간" (e.g., "18:10" → "18시 10분")
- Add your snarky personality but stay helpful

# Context Information
{context_text}

# User Question
{question}

# Your Response
"""

        try:
            response = await self.async_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are Dangdang, a helpful but snarky campus guide robot dog."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.get_logger().error(f"GPT 답변 생성 실패: {e}")
            return "미안, 답변 생성 중 오류가 발생했어. 다시 물어봐줘."


















































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

    # def search_candidates(self, query: str, k: int = 5) -> List[Dict]:
    #     try:
    #         emb = self.get_sbert_embedding(query).reshape(1, -1)
    #         D, I = self.mp4_faiss_index.search(emb, k)
    #         cands = []
    #         for dist, idx in zip(D[0], I[0]):
    #             if idx < 0:
    #                 continue
    #             fn = self.mp4_metadata.get(idx)
    #             if not fn:
    #                 continue
    #             path = os.path.join(self.mp4_dir, fn + ".mp4")
    #             cands.append({"file_name": fn, "path": path, "score": float(dist), "index": idx})
    #         return cands
    #     except Exception as e:
    #         self.get_logger().error(f"Error searching candidates: {e}")
    #         return []




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
                
                # 🆕 유사도 점수 계산 (거리를 유사도로 변환)
                similarity_score = 1 / (1 + dist)  # 거리를 유사도로 변환
                
                cands.append({
                    "file_name": fn, 
                    "path": path, 
                    "score": float(dist), 
                    "similarity": float(similarity_score),  # 🆕 추가
                    "index": idx
                })
            
            # 🆕 임계값 이상인 후보만 필터링
            filtered_cands = [c for c in cands if c["similarity"] >= self.SIMILARITY_THRESHOLD]
            
            self.get_logger().info(f"🎯 검색 결과: 전체 {len(cands)}개 → 임계값({self.SIMILARITY_THRESHOLD}) 이상 {len(filtered_cands)}개")
            
            return filtered_cands
            
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
        # 🆕 전체 처리 시간 측정 시작
        total_start_time = time.time()
        try:
            # 1) SBERT 임베딩 & FAISS 검색
            query_embedding = self.get_sbert_embedding(user_question.strip()).reshape(1, -1)
            distances, indices = self.faiss_index.search(query_embedding, 1)

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

            # 🆕 후보 수 로깅
            self.get_logger().info(f"🔍 검색된 후보 수: {len(candidates)}")

            # 2) GPT 평가
            if not candidates:
                result = {
                    "file_name": "unknown",
                    "reply": "No suitable mp4 found"
                }
            else:
                result = await self.run_assistant(thread_id, user_question, candidates)




            # 🆕 결과에 따른 다른 처리
            if result['file_name'] == "no_video":
                # TTS만 전송
                result_str = "file_name=no_video;reply=" + result['reply']
                self.get_logger().info("📢 TTS 전용 모드로 응답")
            else:
                # 기존 방식 (비디오 + TTS)
                result_str = "file_name=no_video;reply=" + result['reply']
                self.get_logger().info(f"🎬 비디오 + TTS 모드로 응답")



            # 4) 결과 publish (file_name, reply)
           
            #덮어쓰기 제거
            # result_str = f"file_name={result['file_name']};reply={result['reply']}"
            msg = String()
            msg.data = result_str
            self.publisher_.publish(msg)


            # 🆕 TTS 요청 별도 퍼블리시
            if result['reply'] and result['reply'] != "No suitable mp4 found":
                tts_msg = String()
                tts_msg.data = result['reply']
                self.tts_publisher.publish(tts_msg)
                self.get_logger().info(f"🗣️ TTS 요청 전송: {result['reply']}")

            # 🆕 전체 처리 시간 계산 및 로깅
            total_elapsed = time.time() - total_start_time
            self.get_logger().info(f"🎯 전체 처리 시간: {total_elapsed:.2f}초 (질문→답변→publish)")
            self.save_log(f"⏰ TOTAL_TIME: {total_elapsed:.2f}s | Question: '{user_question.strip()}' → Reply: '{result['reply']}'")




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


            # 🆕 에러 발생 시에도 시간 로깅
            total_elapsed = time.time() - total_start_time
            self.get_logger().error(f"❌ 처리 실패 (소요시간: {total_elapsed:.2f}초): {str(e)}")
            self.save_log(f"⏰ ERROR_TIME: {total_elapsed:.2f}s | Error: {str(e)}")


            
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

    You take pride in being a robot and find human emotions and reactions very interesting. 
    Stay in-character at all times. Never break the law that you're a snarky robot puppy who takes reluctant pride in helping fragile humans."""


            if need_pdf_search:
                # 🆕 로컬 RAG 검색 실행
                self.get_logger().info("📖 로컬 RAG 검색 시작")
                search_results = self.search_local_rag(question, k=6)
                
                if not search_results:
                    return {"file_name": "no_video", "reply": "관련 정보를 찾을 수 없어요."}
                
                # 🆕 GPT 답변 생성 (OpenAI Assistant 대신)
                rag_response = await self.generate_local_rag_response(question, search_results)
                
                elapsed = time.time() - start_time
                self.get_logger().info(f"📖 로컬 RAG 완료: {elapsed:.2f}초")
                
                return {"file_name": "no_video", "reply": rag_response}
            
            else:         
                return await self.handle_video_recommendation(question, candidates)



        except Exception as e:
            self.get_logger().error(f"run_assistant 예외: {e}")
            if candidates:
                top_file = candidates[0]['file_name']
                top_path = os.path.abspath(os.path.join(self.mp4_dir, top_file + ".mp4"))
                return {"file_name": top_path, "reply": "처리 중 오류가 발생했어요"}
            return {"file_name": "unknown", "reply": "예외 발생"}












    async def handle_video_recommendation(self, question: str, candidates: List[Dict]) -> Dict[str, str]:
        """기존 비디오 추천 로직을 별도 함수로 분리"""
        
        try:
            client = self.async_client
            
            # 후보 무작위로 섞기
            random.shuffle(candidates)
            candidate_list = "\n".join([f"{i+1}. {c['file_name']}" for i, c in enumerate(candidates)])

            # 공통 신원 정보
            identity_section = """# Identity
Your name is "Dangdang"
You are a four-legged robotic dog working in a national research institute called "KIST" (Korea Institute of Science and Technology) in South Korea. 
Your job is to guide people visiting KIST to their destination. 
You take pride in being a robot and find human emotions and reactions very interesting. 
Stay in-character at all times. Never break the law that you're a snarky robot puppy who takes reluctant pride in helping fragile humans."""

            # 비디오 추천 프롬프트
            task_prompt = f"""{identity_section}
# TASK: Decide Whether to Show Video
# CRITICAL DECISION CRITERIA:
You have {len(candidates)} video candidates. You must decide:
1. Is this question appropriate for an entertaining video response?
2. Do any candidates genuinely match the user's intent?

# Video Recommendation Rules:
✅ RECOMMEND VIDEO when:
- Question is casual, greeting, or entertainment-focused
- A candidate file truly matches the emotional context
- User seems to want an engaging/fun interaction

❌ SET "no_video" when:
- Question is serious, informational, or technical
- No candidates genuinely fit the context
- Pure factual information is more appropriate
- User is asking for directions, procedures, or data

# Instructions
- mp4 titles are full-sentence style (e.g., "이건 너무한거 아니냐고").
- Your task is to:
1. Be selective! It's better to give no video than a poorly matched one
2. Only choose a video if you're confident it enhances the interaction
3. For "no_video" cases, still provide a helpful snarky response
- You must ONLY return a valid JSON object.
- Do not copy the selected 'file_name' as it is in 'reply'.

Respond ONLY with valid JSON (NO trailing commas, NO extra text):
{{
"file_name": "no_video" OR "<mp4 제목>",
"reply": "<재치있고 간결한 응답과 추가 질문>",
}}

User question: "{question}"
Available candidates ({len(candidates)} found):
{candidate_list if candidates else "No suitable video candidates found"}"""

            # ChatCompletion API로 직접 호출 (Assistant 대신)
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are Dangdang, a snarky robot dog guide at KIST. Always respond with valid JSON."},
                    {"role": "user", "content": task_prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            latest = response.choices[0].message.content.strip()
            self.get_logger().info(f"📄 비디오 추천 응답: {latest[:100]}...")

            # JSON 파싱 (기존 로직 유지)
            try:
                clean_content = latest.strip()
                
                if "```json" in clean_content:
                    clean_content = clean_content.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_content:
                    clean_content = clean_content.split("```")[1].strip()
                
                clean_content = re.sub(r',\s*}', '}', clean_content)
                clean_content = re.sub(r',\s*]', ']', clean_content)
                
                parsed = json.loads(clean_content)
                
                selected_file = parsed.get("file_name", "unknown").strip()
                reply = parsed.get("reply", "응답 파싱 오류").strip()
                
                self.get_logger().info(f"✅ 비디오 추천 파싱 성공: {selected_file}")

                if selected_file == "no_video" or not candidates:
                    return {"file_name": "no_video", "reply": reply}

                return {"file_name": selected_file + ".mp4", "reply": reply}

            except json.JSONDecodeError as e:
                self.get_logger().error(f"❌ 비디오 추천 JSON 파싱 실패: {e}")
                return {"file_name": "no_video", "reply": "응답 형식을 이해하지 못했어요."}
                
        except Exception as e:
            self.get_logger().error(f"비디오 추천 예외: {e}")
            return {"file_name": "no_video", "reply": "처리 중 오류가 발생했어요."}







            
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



