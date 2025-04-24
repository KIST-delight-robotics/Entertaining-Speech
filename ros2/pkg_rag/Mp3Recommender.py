import os
import sqlite3
import faiss
import openai
import asyncio
import torch
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from dotenv import load_dotenv
import json
import aiohttp
import time
from datetime import datetime
import numpy as np 

class Mp3Recommender(Node):
    def __init__(self):
        super().__init__('Mp3Recommender')
        self.status_pub = self.create_publisher(String, 'mp3_recommend_status', 10)

        # ✅ 로그 파일 경로 설정
        self.log_file_path = "/home/nvidia/ros2_ws/_logs/Mp3Recommender_log.txt"
        self.save_log("✅ Mp3Recommender Node Started")


        # ----- 환경 변수 / OpenAI API 키 로드 -----
        load_dotenv()
    
        openai.api_key = os.getenv('OPENAI_API_KEY')

        # ----- SBERT 모델 초기화 -----
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sbert_model = SentenceTransformer(
            "BAAI/bge-m3", device=device
        )

        # ----- 음악 DB와 FAISS 인덱스 로딩 -----  
        self.db_path = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/mp3_database.db"
        self.faiss_index_file = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/faiss_index.bin"
        self.faiss_index = self.load_faiss_index()
        self.metadata = self.load_metadata_from_db()
    
        
        # ----- ROS2 pub/sub 설정 -----
        self.publisher_ = self.create_publisher(String, 'recommended_mp3', 10)
        self.subscription_ = self.create_subscription(
            String,
            'user_question',
            self.question_callback,
            10
        )
        self.get_logger().info("Mp3Recommender node has started.")


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

    def load_metadata_from_db(self):
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

    
    def get_sbert_embedding(self, text: str):
        start_time = time.time()
        embedding = self.sbert_model.encode(text).astype("float32")
        normalized_embedding = embedding / norm(embedding)  # 정규화하여 코사인 유사도 기반 검색
        end_time = time.time()
        print(f"임베딩 생성 시간: {end_time - start_time:.4f}초")
        # ✅ 로그 저장
        self.save_log(f"임베딩 생성 시간: {end_time - start_time:.4f}초")
        return normalized_embedding

    async def evaluate_with_gpt(self, user_question: str, candidates: list):
        start_time = time.time()
        candidate_list = "\n".join([f"{i+1}. {candidate['file_name']} (cos_sim: {candidate['cosine_similarity']:.4f})" for i, candidate in enumerate(candidates)])
        #self.get_logger().info(f"Candidate list:\n{candidate_list}")  # Log the candidate list
        # ✅ 로그 저장
        self.save_log(f"Candidate list:\n{candidate_list}")  
        
        prompt = (
            f"사용자의 질문: '{user_question}'에 가장 적절한 MP3 파일명을 하나 선택하세요.\n\n"
            "### **선택 기준:**\n"
            "1. 파일명과 질문의 의미적 연결을 최우선으로 고려하시오.\n"
            "2. 코사인 유사도를 절대적인 기준으로 사용하지 마시오.\n"
            "3. 노래 가사에서 질문과 가장 연관 있는 키워드나 개념이 포함될 가능성이 높은 파일을 고릅니다.\n"
            "4. candidate list에 없는 파일 제목을 절대 선택하지 마시오.\n"
            "5. candidate list에 있는 파일명을 그대로 사용하고 일부 단어만 선택하지 마시오.\n"


            "[특정 개념이 포함된 질문일 경우, 관련된 키워드를 고려하여 선택합니다.]\n"
            "   - '외계인' 관련 질문 → 우주, 별, 블랙홀, 슈퍼노바(Supernova), 외계 생명체 등의 키워드가 포함된 파일\n"
            "   - '사랑' 관련 질문 → 감정, 이별, 연애, 고백 등의 키워드가 포함된 파일\n"
            "   - '추억' 관련 질문 → 기억, 과거, 시간, 돌아가기 등의 키워드가 포함된 파일\n\n"
            "### **후보 리스트 (코사인 유사도 높은 순):**\n"
            f"{candidate_list}\n\n"
            "이제 가장 적절한 MP3 파일명을 JSON 형식으로만 반환하세요.\n"
            "반드시 하나만 선택하고, JSON 구조는 다음과 같아야 합니다:\n\n"
            "{\n"
            '  "file_name_1": "<파일명만>",\n'
            '  "file_name_2": "<파일명만>",\n'
            '  "file_name_3": "<파일명만>"\n'
            "}\n"
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "사용자의 질문을 분석한 뒤, 후보 리스트 중에서 대답으로 가장 잘 어울리는 노래 제목(파일명)을 3개 선택하세요. "
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        try:
            # GPT API 호출
            response = await openai.ChatCompletion.acreate(
                model="gpt-4o",
                messages=messages
            )
            raw_answer = response["choices"][0]["message"]["content"].strip()
            self.get_logger().info(f"Raw GPT response: {raw_answer}")
            self.save_log(f"Raw GPT response: {raw_answer}")

            if "```json" in raw_answer:
                raw_answer = raw_answer.split("```json")[-1].strip("```").strip()

            #JSON 파싱 시도
            self.get_logger().info("Attempting to parse GPT response as JSON...")
            parsed = json.loads(raw_answer)
            # JSON 파싱 성공 로그
            self.get_logger().info(f"Parsed JSON: {parsed}")
            # ✅ 로그 저장
            self.save_log(f"Parsed JSON: {parsed}")

            # 🔹 GPT가 선택한 파일 리스트
            gpt_files = [
                parsed.get("file_name_1", "").strip(),
                parsed.get("file_name_2", "").strip(),
                parsed.get("file_name_3", "").strip()
            ]
            gpt_files = [f for f in gpt_files if f]  # 빈 문자열 제거

            end_time = time.time()
            gpt_evaluation_time = end_time - start_time  # Time taken for GPT evaluation

            # Log GPT evaluation time
            self.get_logger().info(f"GPT evaluation time: {gpt_evaluation_time:.4f} seconds")
            # ✅ 로그 저장
            self.save_log(f"GPT evaluation time: {gpt_evaluation_time:.4f} seconds")

            # GPT 선택 파일 로그
            self.get_logger().info(f"🟢 GPT selected files: {gpt_files}")
            self.save_log(f"🟢 GPT selected files: {gpt_files}")

            if not gpt_files:
                self.get_logger().warning("GPT가 선택한 파일이 없습니다.")
                if candidates:
                    return [candidates[0]['file_name']]  # 첫 번째 후보 반환
                return ["No suitable MP3 found"]
            
            # GPT가 선택한 파일 3개를 임베딩
            file_names_only = []
            for file in gpt_files:
                # 파일명만 추출 (경로 및 확장자 제거)
                base_name = os.path.basename(file)
                if base_name.endswith('.mp3'):
                    base_name = base_name[:-4]  # .mp3 확장자 제거
                file_names_only.append(base_name)
            
            final_files = []
            used_indices = set()
            
            # 각 GPT 선택 파일에 대해 FAISS 검색 수행
            for file_name in file_names_only:
                try:
                    # 파일명을 임베딩
                    self.get_logger().info(f"임베딩 생성 중: {file_name}")
                    self.save_log(f"임베딩 생성 중: {file_name}")
                    file_embedding = self.get_sbert_embedding(file_name).reshape(1, -1)
                    
                    # FAISS 검색으로 유사한 파일 찾기
                    k = min(5, len(candidates))  # 최대 5개 또는 후보 개수
                    distances, indices = self.faiss_index.search(file_embedding, k)
                    
                    # 최상위 결과 중 아직 선택되지 않은 파일 찾기
                    for i, idx in enumerate(indices[0]):
                        if idx == -1:  # 유효하지 않은 인덱스
                            continue
                        
                        # 이미 선택된 파일 건너뛰기
                        if idx in used_indices:
                            continue
                        
                        # *** 음악 파일명 가져오기
                        db_file_name = self.metadata.get(idx, "Unknown")
                        file_path = os.path.abspath(os.path.join(
                            "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/mp3_database", 
                            db_file_name + ".mp3"
                        ))

                        # # *** 영화 파일명 가져오기
                        # db_file_name = self.metadata.get(idx, "Unknown")
                        # file_path = os.path.abspath(os.path.join(
                        #     "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/movie_database", 
                        #     db_file_name + ".mp3"
                        # ))
                        
                        final_files.append(file_path)
                        used_indices.add(idx)
                        self.get_logger().info(f"FAISS 검색 결과: {db_file_name} (코사인 유사도: {distances[0][i]})")
                        self.save_log(f"FAISS 검색 결과: {db_file_name} (코사인 유사도: {distances[0][i]})")
                        break  # 첫 번째 유효한 결과만 사용
                
                except Exception as e:
                    self.get_logger().error(f"FAISS 검색 중 오류: {str(e)}")
                    self.save_log(f"FAISS 검색 중 오류: {str(e)}")
            
            # 선택된 파일이 3개보다 적으면 원래 후보에서 추가
            if len(final_files) < 3:
                self.get_logger().info(f"선택된 파일이 3개 미만: {len(final_files)}개. 후보 추가 중...")
                for candidate in candidates:
                    if len(final_files) >= 3:
                        break
                    
                    idx = candidate['index']
                    if idx not in used_indices:
                        final_files.append(candidate['file_name'])
                        used_indices.add(idx)
            
            # 최종 선택된 파일 목록 로그
            self.get_logger().info(f"🎵 최종 선택된 파일 목록: {final_files}")
            self.save_log(f"🎵 최종 선택된 파일 목록: {final_files}")
            
            return final_files[:3]  # 최대 3개 파일만 반환

        except json.JSONDecodeError as jde:
            self.get_logger().error(f"JSON decoding error: {str(jde)}")
            self.save_log(f"JSON decoding error: {str(jde)}")
            if candidates:
                return [candidates[0]['file_name']]
            return ["No suitable MP3 found"]
        
        except Exception as e:
            self.get_logger().error(f"GPT API 또는 처리 중 오류: {str(e)}")
            self.save_log(f"GPT API 또는 처리 중 오류: {str(e)}")
            if candidates:
                return [candidates[0]['file_name']]
            return ["No suitable MP3 found"]


    def question_callback(self, msg: String):
        self.status_pub.publish(String(data='searching'))

        """
        ROS 콜백: user_question 토픽 수신 시 처리
        """
        user_question = msg.data
        self.get_logger().info(f"User question received: {user_question}")
        # ✅ 로그 저장
        self.save_log(f"User question received: {user_question}")

        # 이미 메인 이벤트 루프가 돌아가므로 create_task로 비동기 함수 등록
        asyncio.create_task(self.process_question(user_question))

    async def process_question(self, user_question: str):
        """
        실제 질의 처리 & GPT 호출 & 추천 결과 Publish
        """
        try:
            # 1) SBERT 임베딩 & FAISS 검색
            query_embedding = self.get_sbert_embedding(user_question).reshape(1, -1)
            distances, indices = self.faiss_index.search(query_embedding, 150)

            candidates = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx == -1:
                    continue
                file_name = self.metadata.get(idx, "Unknown")
                
                # *** 음악 파일 경로
                file_path = os.path.abspath(os.path.join(
                    "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/mp3_database", file_name + ".mp3"
                ))

                # # *** 영화 파일 경로
                # file_path = os.path.abspath(os.path.join(
                #     "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/movie_database", file_name + ".mp3"
                # ))


                candidates.append({"file_name": file_path, "cosine_similarity": distance , "index": idx})

            # 2) GPT 평가 
            if not candidates:
                result = "No suitable MP3 found"
            else:
                result = await self.evaluate_with_gpt(user_question, candidates)

            # # 3) Publish 결과 (JSON으로 변환 후 전송)
            # result_json = json.dumps({"file_names": result})
            # self.publisher_.publish(String(data=result_json))
            # self.get_logger().info(f"Recommendation published: {result_json}")

            # 3) Publish 결과 (쉼표로 구분된 문자열 생성)
            if not isinstance(result, list):
                self.get_logger().error(f"❌ 결과가 리스트가 아닙니다: {type(result)}")

            # ✅ JSON 없이 Key=Value 문자열로 변환
            result_str = ";".join(f"file_name_{i+1}={file}" for i, file in enumerate(result))

            # ROS2 메시지에 문자열 설정
            msg = String()
            msg.data = result_str
            self.publisher_.publish(msg)
            self.get_logger().info(f"✅ Recommendation published: {result_str}")
            # ✅ 로그 저장
            self.save_log(f"Recommendation published: {result_str}")
            self.status_pub.publish(String(data='done'))

        except Exception as e:
            self.get_logger().error(f"Error during processing: {str(e)}")
            error_msg = String()
            error_msg.data = f"Error: {str(e)}"
            self.publisher_.publish(error_msg)
            # ✅ 에러 로그 저장
            self.save_log(f"❌ Error: {str(e)}")
            self.status_pub.publish(String(data='done'))

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







