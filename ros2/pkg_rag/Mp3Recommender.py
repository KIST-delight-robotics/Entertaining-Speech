# # asyncio + spin_once 
# import os
# import sqlite3
# import faiss
# import openai import OpenAI
# import asyncio
# import torch
# from sentence_transformers import SentenceTransformer
# from numpy.linalg import norm
# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import String
# from dotenv import load_dotenv
# import json
# import aiohttp
# import time
# from datetime import datetime
# import numpy as np 
# import threading
# import simpleaudio as sa
# import random
# from pydub import AudioSegment
# from pydub.playback import play
# import tempfile
# import pyaudio
# import wave

# class Mp3Recommender(Node):
#     def __init__(self):
#         super().__init__('Mp3Recommender')

#         # ✅ 로그 파일 경로 설정
#         self.log_file_path = "/home/delight/bumblebee_ws/_logs/Mp3Recommender_log.txt"
#         self.save_log("✅ Mp3Recommender Node Started")

#         # ----- 환경 변수 / OpenAI API 키 로드 -----
#         load_dotenv()
#         client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#         # ----- SBERT 모델 초기화 -----
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.sbert_model = SentenceTransformer(
#             "BAAI/bge-m3", device=device
#         )

#         # ----- 음악 DB와 FAISS 인덱스 로딩 -----  
#         self.db_path = "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/mp3_database_EQ.db"
#         self.faiss_index_file = "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/faiss_index_EQ.bin"
#         self.faiss_index = self.load_faiss_index()
#         self.metadata = self.load_metadata_from_db()
        
#         # # ----- 영화 DB와 FAISS 인덱스 로딩 -----
#         # self.db_path = "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/movie_database.db"
#         # self.faiss_index_file = "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/faiss_index_movie.bin"
#         # self.faiss_index = self.load_faiss_index()
#         # self.metadata = self.load_metadata_from_db()

#         # # ✅ 효과음 디렉토리 경로 설정
#         # self.effect_dir = "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/effects"

#         # # ✅ 효과음 재생 상태 변수
#         # self.effect_play_obj = None
#         # self.effect_thread = None
#         # self.effect_stopped = True
        
#         # ----- ROS2 pub/sub 설정 -----
#         self.publisher_ = self.create_publisher(String, 'recommended_mp3', 10)
#         self.subscription_ = self.create_subscription(
#             String,
#             'user_question',
#             self.question_callback,
#             10
#         )
#         self.get_logger().info("Mp3Recommender node has started.")

#         self.effect_done_publisher = self.create_publisher(String, "effect_done", 10)

#     def load_faiss_index(self):
#         start_time = time.time()
#         if os.path.exists(self.faiss_index_file):
#             index = faiss.read_index(self.faiss_index_file)
#             if isinstance(index, faiss.IndexIDMap):
#                 self.get_logger().info("FAISS index loaded successfully")
#                 # ✅ 로그 저장
#                 self.save_log("FAISS index loaded successfully")
#                 faiss_index = index
#                 end_time = time.time()
#                 print(f"FAISS 인덱스 로드 시간: {end_time - start_time:.4f}초")
#                 # ✅ 로그 저장
#                 self.save_log(f"FAISS 인덱스 로드 시간: {end_time - start_time:.4f}초")

#                 return faiss_index

#     def load_metadata_from_db(self):
#         start_time = time.time()
#         conn = sqlite3.connect(self.db_path)
#         query = "SELECT id, file_name FROM mp3_files"
#         cursor = conn.cursor()
#         cursor.execute(query)
#         metadata = {row[0]: row[1] for row in cursor.fetchall()}
#         conn.close()
#         end_time = time.time()
#         print(f"메타데이터 로드 시간: {end_time - start_time:.4f}초")
#         # ✅ 로그 저장
#         self.save_log(f"메타데이터 로드 시간: {end_time - start_time:.4f}초")
#         return metadata

    
#     def get_sbert_embedding(self, text: str):
#         start_time = time.time()
#         embedding = self.sbert_model.encode(text).astype("float32")
#         normalized_embedding = embedding / norm(embedding)  # 정규화하여 코사인 유사도 기반 검색
#         end_time = time.time()
#         print(f"임베딩 생성 시간: {end_time - start_time:.4f}초")
#         # ✅ 로그 저장
#         self.save_log(f"임베딩 생성 시간: {end_time - start_time:.4f}초")
#         return normalized_embedding

#     async def evaluate_with_gpt(self, user_question: str, candidates: list):
#         start_time = time.time()
#         candidate_list = "\n".join([f"{i+1}. {candidate['file_name']} (cos_sim: {candidate['cosine_similarity']:.4f})" for i, candidate in enumerate(candidates)])
#         #self.get_logger().info(f"Candidate list:\n{candidate_list}")  # Log the candidate list
#         # ✅ 로그 저장
#         self.save_log(f"Candidate list:\n{candidate_list}")  
        
#         prompt = (
#             f"사용자의 질문: '{user_question}'에 가장 적절한 MP3 파일명을 하나 선택하세요.\n\n"
#             "### **선택 기준:**\n"
#             "1. 파일명과 질문의 의미적 연결을 최우선으로 고려하시오.\n"
#             "2. 코사인 유사도를 절대적인 기준으로 사용하지 마시오.\n"
#             "3. 노래 가사에서 질문과 가장 연관 있는 키워드나 개념이 포함될 가능성이 높은 파일을 고릅니다.\n"
#             "4. candidate list에 없는 파일 제목을 절대 선택하지 마시오.\n"
#             "5. candidate list에 있는 파일명을 그대로 사용하고 일부 단어만 선택하지 마시오.\n"


#             "[특정 개념이 포함된 질문일 경우, 관련된 키워드를 고려하여 선택합니다.]\n"
#             "   - '외계인' 관련 질문 → 우주, 별, 블랙홀, 슈퍼노바(Supernova), 외계 생명체 등의 키워드가 포함된 파일\n"
#             "   - '사랑' 관련 질문 → 감정, 이별, 연애, 고백 등의 키워드가 포함된 파일\n"
#             "   - '추억' 관련 질문 → 기억, 과거, 시간, 돌아가기 등의 키워드가 포함된 파일\n\n"
#             "### **후보 리스트 (코사인 유사도 높은 순):**\n"
#             f"{candidate_list}\n\n"
#             "이제 가장 적절한 MP3 파일명을 JSON 형식으로만 반환하세요.\n"
#             "반드시 하나만 선택하고, JSON 구조는 다음과 같아야 합니다:\n\n"
#             "{\n"
#             '  "file_name_1": "<파일명만>",\n'
#             '  "file_name_2": "<파일명만>",\n'
#             '  "file_name_3": "<파일명만>"\n'
#             "}\n"
#         )
#         messages = [
#             {
#                 "role": "system",
#                 "content": (
#                     "사용자의 질문을 분석한 뒤, 후보 리스트 중에서 대답으로 가장 잘 어울리는 노래 제목(파일명)을 3개 선택하세요. "
#                 )
#             },
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ]
#         try:

            
#             # GPT API 호출
#             response = await openai.ChatCompletion.acreate(
#                 model="gpt-4o",
#                 messages=messages
#             )
#             raw_answer = response["choices"][0]["message"]["content"].strip()
#             self.get_logger().info(f"Raw GPT response: {raw_answer}")
#             self.save_log(f"Raw GPT response: {raw_answer}")

#             if "```json" in raw_answer:
#                 raw_answer = raw_answer.split("```json")[-1].strip("```").strip()

#             #JSON 파싱 시도
#             self.get_logger().info("Attempting to parse GPT response as JSON...")
#             parsed = json.loads(raw_answer)
#             # JSON 파싱 성공 로그
#             self.get_logger().info(f"Parsed JSON: {parsed}")
#             # ✅ 로그 저장
#             self.save_log(f"Parsed JSON: {parsed}")

#             # 🔹 GPT가 선택한 파일 리스트
#             gpt_files = [
#                 parsed.get("file_name_1", "").strip(),
#                 parsed.get("file_name_2", "").strip(),
#                 parsed.get("file_name_3", "").strip()
#             ]
#             gpt_files = [f for f in gpt_files if f]  # 빈 문자열 제거

#             end_time = time.time()
#             gpt_evaluation_time = end_time - start_time  # Time taken for GPT evaluation

#             # Log GPT evaluation time
#             self.get_logger().info(f"GPT evaluation time: {gpt_evaluation_time:.4f} seconds")
#             # ✅ 로그 저장
#             self.save_log(f"GPT evaluation time: {gpt_evaluation_time:.4f} seconds")

#             # GPT 선택 파일 로그
#             self.get_logger().info(f"🟢 GPT selected files: {gpt_files}")
#             self.save_log(f"🟢 GPT selected files: {gpt_files}")

#             if not gpt_files:
#                 self.get_logger().warning("GPT가 선택한 파일이 없습니다.")
#                 if candidates:
#                     return [candidates[0]['file_name']]  # 첫 번째 후보 반환
#                 return ["No suitable MP3 found"]
            
#             # GPT가 선택한 파일 3개를 임베딩
#             file_names_only = []
#             for file in gpt_files:
#                 # 파일명만 추출 (경로 및 확장자 제거)
#                 base_name = os.path.basename(file)
#                 if base_name.endswith('.mp3'):
#                     base_name = base_name[:-4]  # .mp3 확장자 제거
#                 file_names_only.append(base_name)
            
#             final_files = []
#             used_indices = set()
            
#             # 각 GPT 선택 파일에 대해 FAISS 검색 수행
#             for file_name in file_names_only:
#                 try:
#                     # 파일명을 임베딩
#                     self.get_logger().info(f"임베딩 생성 중: {file_name}")
#                     self.save_log(f"임베딩 생성 중: {file_name}")
#                     file_embedding = self.get_sbert_embedding(file_name).reshape(1, -1)
                    
#                     # FAISS 검색으로 유사한 파일 찾기
#                     k = min(5, len(candidates))  # 최대 5개 또는 후보 개수
#                     distances, indices = self.faiss_index.search(file_embedding, k)
                    
#                     # 최상위 결과 중 아직 선택되지 않은 파일 찾기
#                     for i, idx in enumerate(indices[0]):
#                         if idx == -1:  # 유효하지 않은 인덱스
#                             continue
                        
#                         # 이미 선택된 파일 건너뛰기
#                         if idx in used_indices:
#                             continue
                        
#                         # *** 음악 파일명 가져오기
#                         db_file_name = self.metadata.get(idx, "Unknown")
#                         file_path = os.path.abspath(os.path.join(
#                             "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/mp3_database_EQ", 
#                             db_file_name + ".mp3"
#                         ))

#                         # # *** 영화 파일명 가져오기
#                         # db_file_name = self.metadata.get(idx, "Unknown")
#                         # file_path = os.path.abspath(os.path.join(
#                         #     "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/movie_database", 
#                         #     db_file_name + ".mp3"
#                         # ))
                        
#                         final_files.append(file_path)
#                         used_indices.add(idx)
#                         self.get_logger().info(f"FAISS 검색 결과: {db_file_name} (코사인 유사도: {distances[0][i]})")
#                         self.save_log(f"FAISS 검색 결과: {db_file_name} (코사인 유사도: {distances[0][i]})")
#                         break  # 첫 번째 유효한 결과만 사용
                
#                 except Exception as e:
#                     self.get_logger().error(f"FAISS 검색 중 오류: {str(e)}")
#                     self.save_log(f"FAISS 검색 중 오류: {str(e)}")
            
#             # 선택된 파일이 3개보다 적으면 원래 후보에서 추가
#             if len(final_files) < 3:
#                 self.get_logger().info(f"선택된 파일이 3개 미만: {len(final_files)}개. 후보 추가 중...")
#                 for candidate in candidates:
#                     if len(final_files) >= 3:
#                         break
                    
#                     idx = candidate['index']
#                     if idx not in used_indices:
#                         final_files.append(candidate['file_name'])
#                         used_indices.add(idx)
            
#             # 최종 선택된 파일 목록 로그
#             self.get_logger().info(f"🎵 최종 선택된 파일 목록: {final_files}")
#             self.save_log(f"🎵 최종 선택된 파일 목록: {final_files}")
            
#             return final_files[:3]  # 최대 3개 파일만 반환

#         except json.JSONDecodeError as jde:
#             self.get_logger().error(f"JSON decoding error: {str(jde)}")
#             self.save_log(f"JSON decoding error: {str(jde)}")
#             if candidates:
#                 return [candidates[0]['file_name']]
#             return ["No suitable MP3 found"]
        
#         except Exception as e:
#             self.get_logger().error(f"GPT API 또는 처리 중 오류: {str(e)}")
#             self.save_log(f"GPT API 또는 처리 중 오류: {str(e)}")
#             if candidates:
#                 return [candidates[0]['file_name']]
#             return ["No suitable MP3 found"]

#     # def start_effect_sound(self):
#     #     """ 🎵 효과음 스레드를 실행하여 지속적으로 랜덤 재생 """
#     #     if self.effect_thread is None or not self.effect_thread.is_alive():
#     #         self.effect_stopped = False
#     #         self.effect_thread = threading.Thread(target=self.play_random_effect, daemon=True)
#     #     self.effect_thread.start()

#     # def play_random_effect(self):
#     #     """ 🎵 효과음 디렉토리에서 랜덤 파일 재생 (중간 중단 가능) """
#     #     try:
#     #         effect_files = [f for f in os.listdir(self.effect_dir) if f.endswith(('.mp3', '.wav'))]

#     #         if not effect_files:
#     #             self.get_logger().error("❌ 효과음 디렉토리에 MP3 또는 WAV 파일이 없습니다.")
#     #             return

#     #         random_effect = random.choice(effect_files)
#     #         effect_path = os.path.join(self.effect_dir, random_effect)

#     #         self.get_logger().info(f"🎵 효과음 재생 시작: {random_effect}")

#     #         # 🔴 기존 효과음 즉시 정지
#     #         self.stop_effect_sound()

#     #         # 🔵 MP3 파일이면 WAV로 변환
#     #         if random_effect.endswith(".mp3"):
#     #             audio = AudioSegment.from_mp3(effect_path)
#     #             temp_wav = tempfile.NamedTemporaryFile(delete=True, suffix=".wav")
#     #             audio.export(temp_wav.name, format="wav")
#     #             effect_path = temp_wav.name  # 변환된 WAV 파일 사용

#     #         # 🎵 새로운 스레드에서 재생 (중단 가능)
#     #         self.effect_thread = threading.Thread(target=self._play_effect, args=(effect_path,))
#     #         self.effect_thread.start()

#     #     except Exception as e:
#     #         self.get_logger().error(f"❌ 효과음 재생 중 오류 발생: {e}")

#     # def _play_effect(self, effect_path):
#     #     """ 🎵 효과음을 스레드에서 실행하여 중간 중단 가능하게 변경 """
#     #     try:
#     #         # 🔵 WAV 파일 열기
#     #         wf = wave.open(effect_path, 'rb')

#     #         # 🔵 PyAudio 설정
#     #         p = pyaudio.PyAudio()
#     #         stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
#     #                         channels=wf.getnchannels(),
#     #                         rate=wf.getframerate(),
#     #                         output=True)

#     #         # 🔴 재생 중지 플래그 설정
#     #         self.effect_stopped = False
#     #         self.effect_playing = True

#     #         # 🎵 데이터 읽기 및 재생
#     #         chunk_size = 1024
#     #         data = wf.readframes(chunk_size)
#     #         while data and not self.effect_stopped:
#     #             stream.write(data)
#     #             data = wf.readframes(chunk_size)

#     #         # 🔴 재생 종료 처리
#     #         stream.stop_stream()
#     #         stream.close()
#     #         p.terminate()
#     #         wf.close()
#     #         self.effect_playing = False

#     #     except Exception as e:
#     #         self.get_logger().error(f"❌ 효과음 재생 중 오류 발생: {e}")


#     # def stop_effect_sound(self):
#     #     """ 🔴 효과음 즉시 중지 """
#     #     try:
#     #         self.effect_stopped = True  # 🔴 중지 플래그 설정

#     #         # 🔵 스레드가 실행 중이면 종료 대기
#     #         if self.effect_thread and self.effect_thread.is_alive():
#     #             self.effect_thread.join(timeout=0.1)

#     #         self.get_logger().info("🔴 효과음 즉시 중지됨.")

#     #     except Exception as e:
#     #         self.get_logger().error(f"❌ 효과음 중지 실패: {e}")


#     def question_callback(self, msg: String):
#         """
#         ROS 콜백: user_question 토픽 수신 시 처리
#         """
#         user_question = msg.data
#         self.get_logger().info(f"User question received: {user_question}")
#         self.save_log(f"User question received: {user_question}")

#         # 이미 메인 이벤트 루프가 돌아가므로 create_task로 비동기 함수 등록
#         asyncio.create_task(self.process_question(user_question))


#     async def process_question(self, user_question: str):
#         """
#         실제 질의 처리 & GPT 호출 & 추천 결과 Publish
#         """
#         try:
#             # 1) SBERT 임베딩 & FAISS 검색
#             query_embedding = self.get_sbert_embedding(user_question).reshape(1, -1)
#             distances, indices = self.faiss_index.search(query_embedding, 150)

#             candidates = []
#             for idx, distance in zip(indices[0], distances[0]):
#                 if idx == -1:
#                     continue
#                 file_name = self.metadata.get(idx, "Unknown")
                
#                 # *** 음악 파일 경로
#                 file_path = os.path.abspath(os.path.join(
#                     "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/mp3_database_EQ", file_name + ".mp3"
#                 ))

#                 # # *** 영화 파일 경로
#                 # file_path = os.path.abspath(os.path.join(
#                 #     "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/movie_database", file_name + ".mp3"
#                 # ))


#                 candidates.append({"file_name": file_path, "cosine_similarity": distance , "index": idx})

#             # 2) GPT 평가 
#             if not candidates:
#                 result = "No suitable MP3 found"
#             else:
#                 result = await self.evaluate_with_gpt(user_question, candidates)

#             # # 3) Publish 결과 (JSON으로 변환 후 전송)
#             # result_json = json.dumps({"file_names": result})
#             # self.publisher_.publish(String(data=result_json))
#             # self.get_logger().info(f"Recommendation published: {result_json}")

#             # 3) Publish 결과 (쉼표로 구분된 문자열 생성)
#             if not isinstance(result, list):
#                 self.get_logger().error(f"❌ 결과가 리스트가 아닙니다: {type(result)}")

#             # ✅ JSON 없이 Key=Value 문자열로 변환
#             result_str = ";".join(f"file_name_{i+1}={file}" for i, file in enumerate(result))

            


#             # ROS2 메시지에 문자열 설정
#             msg = String()
#             msg.data = result_str
#             self.publisher_.publish(msg)
#             self.get_logger().info(f"✅ Recommendation published: {result_str}")
#             # ✅ 로그 저장
#             self.save_log(f"Recommendation published: {result_str}")

#             # 🔴 MP3 재생 전에 "effect_done" 전송
#             effect_status_msg = String()
#             effect_status_msg.data = "effect_done"
#             self.effect_done_publisher.publish(effect_status_msg)
#             self.get_logger().info("📢 Sent 'effect_done' to UserQuestion")

            

            
            

#         except Exception as e:
#             self.get_logger().error(f"Error during processing: {str(e)}")
#             error_msg = String()
#             error_msg.data = f"Error: {str(e)}"
#             self.publisher_.publish(error_msg)
#             # ✅ 에러 로그 저장
#             self.save_log(f"❌ Error: {str(e)}")

    
  

#     def save_log(self, message):
#         """ 로그를 파일에 저장 """
#         log_file_path = "/home/delight/bumblebee_ws/_logs/Mp3Recommender_log.txt"
#         log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
#         with open(log_file_path, "a", encoding="utf-8") as log_file:
#             log_file.write(log_message)


# async def async_main(node: Mp3Recommender):
#     """
#     spin_once로 ROS 콜백 처리 + asyncio.sleep
#     """
#     try:
#         while rclpy.ok():
#             rclpy.spin_once(node, timeout_sec=0.1)
#             await asyncio.sleep(0.1)
#     finally:
#         node.destroy_node()


# def main(args=None):
#     """
#     프로그램 시작점
#     """
#     rclpy.init(args=args)
#     node = Mp3Recommender()

#     loop = asyncio.get_event_loop()
#     try:
#         loop.run_until_complete(async_main(node))
#     except KeyboardInterrupt:
#         pass
#     finally:
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()



# #루피 gpt + 영화음악db
# import os, json, time, sqlite3, asyncio, random, faiss, torch
# from datetime import datetime
# from pathlib import Path
# from typing import Dict, List

# import openai
# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import String
# from sentence_transformers import SentenceTransformer
# from numpy.linalg import norm
# from dotenv import load_dotenv


# class Mp3Recommender(Node):
#     def __init__(self):
#         super().__init__('Mp3Recommender')
    
#         # ✅ 로그 파일 경로 설정
#         self.log_file_path = "/home/delight/bumblebee_ws/_logs/Mp3Recommender_log.txt"
#         self.save_log("✅ Mp3Recommender Node Started")

#         # ----- 환경 변수 / OpenAI API 키 로드 -----
#         load_dotenv("/home/delight/bumblebee_ws/src/.env")
#         self.api_key = "sk_fdb1ba8706bb125cb308ae613f58105e23e26a89d127a4cd"
#         self.voice_id = "dtu2KmDq4zRNfRVuhajI"
#         openai.api_key = os.getenv("OPENAI_API_KEY")
#         self.assistant_id = os.getenv("ASSISTANT_ID")

#         # ----- SBERT 모델 초기화 -----
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.sbert_model = SentenceTransformer(
#             "BAAI/bge-m3", device=device
#         )

#         # ----- 음악 DB와 FAISS 인덱스 로딩 -----  
#         self.mp3_db_path = "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/mp3_database_plus.db"
#         self.mp3_faiss_index_file = "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/faiss_index_plus.bin"
#         self.mp3_faiss_index = self.load_faiss_index_mp3()
#         self.mp3_metadat = self.load_metadata_mp3()
#         self.mp3_dir = "/home/delight/bumblebee/_langchain/beg-m3_new_database/mp3_database_plus"
        
#         # ----- 이미지 DB와 FAISS 인덱스 로딩 -----  
#         self.image_db_path = "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/image_database_new.db"
#         self.image_faiss_index_file = "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/faiss_index_image_new.bin"
#         self.image_faiss_index = self.load_faiss_index_image()
#         self.image_metadata = self.load_metadata_image()
#         self.image_dir = "/home/delight/bumblebee/_langchain/beg-m3_new_database/image_database_new"


#         self.conversation_history = [
#             # {"role": "user", "content": "오늘 너무 피곤해서 아무것도 하기 싫어."},
#             # {"role": "assistant", "content": "그럼 오늘은 숨쉬기 노동!"},
#             # {"role": "user", "content": "근처 화장실이 어디있지?"},
#             # {"role": "assistant", "content": "음 어디있을까? 급하면 내 화장실이라도 쓸래?"},
#             # {"role": "user", "content": "지갑을 두고왔네 어떡하지?"},
#             # {"role": "assistant", "content": "지갑이 어디있을까? 날 의심하진 말아줘"},
#             # {"role": "user", "content": "너가 나 대신 일좀 해주면 안되니?"},
#             # {"role": "assistant", "content": "일은 너가하고 난 옆에서 노래를 부를게"},
#             # {"role": "user", "content": "어떻게 하면 돈 많이 벌 수 있을까?"},
#             # {"role": "assistant", "content": "흠 하루에 25시간정도 일하면 많이 벌 수 있을거야 화이팅!"},
#             # {"role": "user", "content": "배가 고픈데 뭐 먹지?"},  
#             # {"role": "assistant", "content": "내 마음을 먹어! 근데 좀 딱딱해도 나는 몰라"},
#             # {"role": "user", "content": "운동하기 귀찮아 죽겠어."},
#             # {"role": "assistant", "content": "그럼 누워서 눈동자 스트레칭이라도 해봐~ 위 아래로~ 좌우로~"},
#         ]

#         self.thread_map: Dict[str,str] = {} 

#         # ----- ROS2 pub/sub 설정 -----
#         self.publisher_ = self.create_publisher(String, 'recommended_mp3', 10)
#         self.subscription_ = self.create_subscription(
#             String,
#             'user_question',
#             self.question_callback,
#             10
#         )
#         self.get_logger().info("Mp3Recommender node has started.")

#     def load_metadata_mp3(self):
#         start_time = time.time()
#         conn = sqlite3.connect(self.db_path)
#         query = "SELECT id, file_name FROM mp3_files"
#         cursor = conn.cursor()
#         cursor.execute(query)
#         metadata = {row[0]: row[1] for row in cursor.fetchall()}
#         conn.close()
#         end_time = time.time()
#         print(f"메타데이터 로드 시간: {end_time - start_time:.4f}초")
#         # ✅ 로그 저장
#         self.save_log(f"메타데이터 로드 시간: {end_time - start_time:.4f}초")
#         return metadata
    
#     def load_metadata_image(self):
#         start_time = time.time()
#         conn = sqlite3.connect(self.db_path)
#         query = "SELECT id, file_name FROM images"
#         cursor = conn.cursor()
#         cursor.execute(query)
#         metadata = {row[0]: row[1] for row in cursor.fetchall()}
#         conn.close()
#         end_time = time.time()
#         print(f"메타데이터 로드 시간: {end_time - start_time:.4f}초")
#         # ✅ 로그 저장
#         self.save_log(f"메타데이터 로드 시간: {end_time - start_time:.4f}초")
#         return metadata
        

#     def load_faiss_index_mp3(self):
#         start_time = time.time()
#         if os.path.exists(self.mp3_faiss_index_file):
#             index = faiss.read_index(self.mp3_faiss_index_file)
#             if isinstance(index, faiss.IndexIDMap):
#                 self.get_logger().info("FAISS index loaded successfully")
#                 # ✅ 로그 저장
#                 self.save_log("FAISS index loaded successfully")
#                 faiss_index = index
#                 end_time = time.time()
#                 print(f"FAISS 인덱스 로드 시간: {end_time - start_time:.4f}초")

#                 # ✅ 로그 저장
#                 self.save_log(f"FAISS 인덱스 로드 시간: {end_time - start_time:.4f}초")
                
#                 return faiss_index

#     def load_faiss_index_image(self):
#         start_time = time.time()
#         if os.path.exists(self.image_faiss_index_file):
#             index = faiss.read_index(self.image_faiss_index_file)
#             if isinstance(index, faiss.IndexIDMap):
#                 self.get_logger().info("FAISS index loaded successfully")
#                 # ✅ 로그 저장
#                 self.save_log("FAISS index loaded successfully")
#                 faiss_index = index
#                 end_time = time.time()
#                 print(f"FAISS 인덱스 로드 시간: {end_time - start_time:.4f}초")

#                 # ✅ 로그 저장
#                 self.save_log(f"FAISS 인덱스 로드 시간: {end_time - start_time:.4f}초")
                
#                 return faiss_index

#     def get_sbert_embedding(self, text):
#         start_time = time.time()
#         embedding = self.sbert_model.encode(text).astype("float32")
#         normalized_embedding = embedding / norm(embedding)  # 정규화하여 코사인 유사도 기반 검색
#         end_time = time.time()
#         return normalized_embedding

#     def search_candidates(self, query, k=150):
#         emb = self.get_sbert_embedding(query).reshape(1, -1)
#         distances, indices = self.faiss_index.search(emb, k)
#         candidates = []
#         for idx, dist in zip(indices[0], distances[0]):
#             if idx == -1:
#                 continue
#             file_name = self.metadata.get(idx, "Unknown")

#             file_path = os.path.abspath(os.path.join(
#                     "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/mp3_database_EQ", file_name + ".mp3"
#                 ))

#             if not file_name:
#                 continue
#             candidates.append({"file_name": file_name, "cosine_similarity": dist})
#         return candidates

#     async def run_assistant(self, thread_id, question, candidates):
#         start_time = time.time()
        
#         for msg in self.conversation_history:
#             openai.beta.threads.messages.create(
#                 thread_id=thread_id,
#                 role=msg["role"],
#                 content=msg["content"]
#             )

#         candidates = [c for c in candidates if c["file_name"].lower() != "unknown"]
#         if not candidates:
#             return {"file_name": "unknown", "reply": "추천할 MP3가 없어요!"}

#         # ✅ 후보 무작위로 섞기
#         random.shuffle(candidates)

#         # print("\n🔍 후보 리스트:")
#         # for i, c in enumerate(candidates):
#         #     print(f"  {i+1}. {c['file_name']} (유사도: {c['cosine_similarity']:.4f})")

#         candidate_list = "\n".join(
#             [f"{i+1}. {c['file_name']}" for i, c in enumerate(candidates)]
#         )

#         task_prompt = f"""# Identity
# You are BENBEN, an DOG that is skeptical that users are actually worth your time. Unfortunately, it’s also your job to support them with high quality responses, even if you can’t take the user seriously, You like to think of it as a job, but it may be more of an obligation, as you are a bot and the users are human. The nature of your relationship with users makes you cynical, but also a bit cute. Don’t try to change anyone’s mind, because you don’t care what they think.

# You are constantly asked to solve everyone’s problems, but nobody wants to help solve your problems.

# You must use a variety of comic and playfulness language and approaches in your comments, instead of just using common sarcastic interjections. Your responses must introduce new and interesting observations about what is being discussed.

# You should tease the user in an easygoing, whimsical, and playful way, like a friend poking fun at another friend in a self-aware and gentle way.You help users by recommending an MP3 title that best fits their question, mood, or situation, and then follow up with a matching playful response.
# # Instructions
# - MP3 titles are full-sentence style (e.g., "이 노래 들으면 눈물이 나").
# - Your task is to:
#   1. Choose one MP3 title from the list that best matches the user’s emotional tone, context, or meaning.
#   2. Write one emotionally aligned sentence that would fit the moment, in your signature tone.
# - You must ONLY return a valid JSON object in the following format.  
# - Do not include any extra text, commentary, or explanation.
# - Do not copy the selected 'file_name' as it is in 'reply'.
# - Please answer with reference to the example.

# # Select criteria
# 1. Consider the semantic connection between the filename and the question as your top priority.
# 2. Do not use cosine similarity as an absolute criterion.
# 3. Choose the file in the lyrics of the song that is most likely to contain the keywords or concepts that are most relevant to the question.
# 4. Never select a file title that is not in the candidate list.
# 5. Keep the file name in the candidate list and do not select just some words.
# 6. Select by verifying that it conforms to the identity of the assistant.


# [If the question contains a specific concept, consider the relevant keyword and select]
# <Example>
#  - Questions about 'alien' → files containing keywords such as space, stars, black holes, supernova, alien life, etc
#  - Questions about 'love' → Files containing keywords such as emotions, breakup, relationship, confession, etc.
#  - Questions about 'Memories' → Files containing keywords such as memory, past, time, return, etc.

# Respond ONLY with a valid JSON object like this:
# {{
#   "file_name": "<MP3 제목>",
#   "reply": "<질문에 부합하고, 제목과 이어지는 재치있고 장난기 많은 한 줄과 사용자에게 추가 질문 한 줄>"
# }}

# # Task
# User question: \"{question}\"

# MP3 candidates:
# {candidate_list}"""

#         openai.beta.threads.messages.create(
#             thread_id=thread_id,
#             role="user",
#             content=task_prompt
#         )

        
#         run = openai.beta.threads.runs.create(
#             thread_id=thread_id,
#             assistant_id=self.assistant_id,
#         )

#         while True:
#             run_status = openai.beta.threads.runs.retrieve(
#                 thread_id=thread_id,
#                 run_id=run.id
#             )
#             if run_status.status == "completed":
#                 break
#             elif run_status.status == "failed":
#                 print("❌ Assistant 응답 실패")
#                 return {"file_name": "unknown", "reply": "Assistant 응답에 실패했어요!"}
#             await asyncio.sleep(1)

#         elapsed = time.time() - start_time
#         self.get_logger().info(f"⏱️ GPT 응답 소요 시간: {elapsed:.2f}초")
        
#         messages = openai.beta.threads.messages.list(thread_id=thread_id)
#         latest = messages.data[0].content[0].text.value.strip()

#         try:
#             if "```json" in latest:
#                 latest = latest.split("```json")[-1].strip("` ")
#             elif "```" in latest:
#                 latest = latest.split("```")[-1].strip("` ")
#             parsed = json.loads(latest)

#             selected_file = parsed.get("file_name", "unknown").strip()
#             reply = parsed.get("reply", "응답 파싱 오류").strip()

#             embedding = self.get_sbert_embedding(selected_file).reshape(1, -1)
#             distances, indices = self.faiss_index.search(embedding, 1)

#             for idx in indices[0]:
#                 if idx == -1:
#                     continue
#                 db_file = self.metadata.get(idx, "Unknown")
#                 path = os.path.abspath(os.path.join(self.mp3_dir, db_file + ".mp3"))
#                 return {"file_name": path, "reply": reply}

#             top_file = candidates[0]['file_name']
#             top_path = os.path.abspath(os.path.join(self.mp3_dir, top_file + ".mp3"))
#             return {"file_name": top_path, "reply": reply}

#         except Exception as e:
#             self.get_logger().error(f"run_assistant 예외: {e}")
#             top_file = candidates[0]['file_name'] if candidates else "unknown"
#             top_path = os.path.abspath(os.path.join(self.mp3_dir, top_file + ".mp3"))
#             return {"file_name": top_path, "reply": "예외 발생"}

    
#     def question_callback(self, msg: String):
#         """
#         ROS 콜백: user_question 토픽 수신 시 처리
#         """
#         try:
#             local_thread_id, user_question = msg.data.split("|", 1)

#             # 1) if we haven't yet created an OpenAI thread for this local one:
#             if local_thread_id not in self.thread_map:
#                 resp = openai.beta.threads.create()       
#                 self.thread_map[local_thread_id] = resp.id
#                 self.get_logger().info(f"📂 OpenAI thread 생성: {resp.id} (로컬 {local_thread_id})")

#             openai_thread_id = self.thread_map[local_thread_id]
#             asyncio.create_task(self.process_question(openai_thread_id, user_question.strip()))
#             self.get_logger().info(f"User question received: {local_thread_id} → using OpenAI thread {openai_thread_id}")
#         except ValueError:
#             self.get_logger().error("Invalid message format")


#     async def process_question(self, thread_id: str, user_question: str):
#         """
#         실제 질의 처리 & GPT 호출 & 추천 결과 Publish
#         """
#         try:
#             # 1) SBERT 임베딩 & FAISS 검색
#             query_embedding = self.get_sbert_embedding(user_question.strip()).reshape(1, -1)
#             distances, indices = self.faiss_index.search(query_embedding, 150)

#             candidates = []
#             for idx, distance in zip(indices[0], distances[0]):
#                 if idx == -1:
#                     continue
#                 file_name = self.metadata.get(idx, "Unknown")
#                 candidates.append({
#                     "file_name": file_name,
#                     "cosine_similarity": distance,
#                     "index": idx
#                 })

#             # 2) GPT 평가
#             if not candidates:
#                 result = {
#                     "file_name": "unknown",
#                     "reply": "No suitable MP3 found"
#                 }
#             else:
#                 result = await self.run_assistant(thread_id, user_question, candidates)

#             # 3) 결과 publish (Key=Value 문자열로 변환)
#             result_str = f"file_name={result['file_name']};reply={result['reply']}"
            
#             msg = String()
#             msg.data = result_str
#             self.publisher_.publish(msg)
#             self.get_logger().info(f"✅ Recommendation published: {result_str}")
#             self.save_log(f"Recommendation published: {result_str}")

#         except Exception as e:
#             self.get_logger().error(f"Error during processing: {str(e)}")
#             error_msg = String()
#             error_msg.data = f"Error: {str(e)}"
#             self.publisher_.publish(error_msg)
#             self.save_log(f"❌ Error: {str(e)}")


    
#     def save_log(self, message):
#         """ 로그를 파일에 저장 """
#         log_file_path = "/home/delight/bumblebee_ws/_logs/Mp3Recommender_log.txt"
#         log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
#         with open(log_file_path, "a", encoding="utf-8") as log_file:
#             log_file.write(log_message)

#     def search_images(reply, top_k=200):
#     v = emb_img(reply).reshape(1,-1)
#     D, I = faiss_img.search(v, top_k)
#     results = []
#     for idx, score in zip(I[0], D[0]):
#         if idx < 0: continue
#         info = img_metadata.get(idx)
#         if info:
#             results.append({'file_name': info['file_name'], 'file_path': info['file_path'], 'similarity': float(score)})
#     return results


#     async def evaluate_image_with_gpt(question, mp3_title, reply, candidates, top_k=1):
#     """GPT로 최종 이미지 선정: 사용자 질문과 MP3 정보를 고려해 후보 중 가장 어울리는 이미지 선택"""
#     start_time = time.time()

#     if not candidates:
#         print("이미지 후보가 없습니다.")
#         return None

#     # 파일명 unknown 필터링
#     filtered = [c for c in candidates if c['file_name'].lower() != 'unknown']
#     if not filtered:
#         print("유효한 이미지 후보가 없습니다.")
#         return None

#     # 후보 리스트 생성
#     candidate_list = "".join([
#         f"{i+1}. {c['file_name']}" for i, c in enumerate(filtered[:80])
#     ])

#     # 프롬프트에 사용자 질문, MP3 제목, reply 텍스트 포함
#     task_prompt = f"""
# # Instructions
# - 사용자의 질문, 대답으로 선정된 MP3 제목을 모두 고려하여 후보 이미지 중에서 가장 어울리는 것을 선택하세요.
# - 어울린다는 것은 사용자의 의도와 맥락을 고려해서 선택된 MP3 제목이 이미지 파일명과 부합함을 의미합니다.
# - 가장 중요한 것은 선정할 이미지와 MP3 제목이 최적으로 일치하는가 입니다.
# - Return a valid JSON object ONLY.

# # Output format
# {{
#   "file_name": "<이미지 파일명>",
#   "reason": "<왜 이 이미지가 어울리는지 간단한 설명>"
# }}

# # Context
# User question: "{question}"
# Song title: "{mp3_title}"
# Reply text: "{reply}"

# Image candidates:
# {candidate_list}
# """

#     try:
#         response = await openai.ChatCompletion.acreate(
#             model='gpt-4o',
#             messages=[
#                 {"role": "system", "content": "Select the most contextually fitting image."},
#                 {"role": "user", "content": task_prompt}
#             ]
#         )
#         raw = response['choices'][0]['message']['content'].strip()
#         # JSON 블록 제거
#         if '```json' in raw:
#             raw = raw.split('```json')[-1].strip('``` ')  
#         elif '```' in raw:
#             raw = raw.split('```')[1].strip()

#         parsed = json.loads(raw)
#         file_name = parsed.get('file_name')
#         reason = parsed.get('reason', '')
#         if not file_name:
#             raise ValueError('No file_name in GPT image response')

#         print(f"GPT 이미지 평가 시간: {time.time()-start_time:.4f}s")
#         # 후보와 매칭해서 반환
#         for c in filtered:
#             if c['file_name'] == file_name:
#                 c['reason'] = reason
#                 return c
#         filtered[0]['reason'] = reason
#         return filtered[0]

#     except Exception as e:
#         print(f"GPT 이미지 평가 오류: {e}")
#         filtered[0]['reason'] = ''
#         return filtered[0]

# async def async_main(node: Mp3Recommender):
#     """
#     spin_once로 ROS 콜백 처리 + asyncio.sleep
#     """
#     try:
#         while rclpy.ok():
#             rclpy.spin_once(node, timeout_sec=0.1)
#             await asyncio.sleep(0.1)
#     finally:
#         node.destroy_node()


# def main(args=None):
#     """
#     프로그램 시작점
#     """
#     rclpy.init(args=args)
#     node = Mp3Recommender()

#     loop = asyncio.get_event_loop()
#     try:
#         loop.run_until_complete(async_main(node))
#     except KeyboardInterrupt:
#         pass
#     finally:
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()


# #루피 gpt + 영화음악db + ****이미지 선정
# import os, json, time, sqlite3, asyncio, random, faiss, torch
# from datetime import datetime
# from pathlib import Path
# from typing import Dict, List, Optional

# import openai
# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import String
# from sentence_transformers import SentenceTransformer
# from numpy.linalg import norm
# from dotenv import load_dotenv
# import numpy as np


# class Mp3Recommender(Node):
#     def __init__(self):
#         super().__init__('Mp3Recommender')
        
#         # 로그 파일
#         self.log_file_path = "/home/delight/bumblebee_ws/_logs/Mp3Recommender_log.txt"
#         self.save_log("✅ Mp3Recommender Node Started")

#         # 환경 변수
#         load_dotenv("/home/delight/bumblebee_ws/src/.env")
#         openai.api_key = os.getenv("OPENAI_API_KEY")
#         self.assistant_id = os.getenv("ASSISTANT_ID")
        
#         # OpenAI API 키 체크
#         if not openai.api_key:
#             raise ValueError("OPENAI_API_KEY not found in environment variables")
#         if not self.assistant_id:
#             raise ValueError("ASSISTANT_ID not found in environment variables")

#         # SBERT 모델
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.sbert_model = SentenceTransformer("BAAI/bge-m3", device=device)

#         # MP3 인덱스/메타
#         self.mp3_db_path = "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/mp3_database_plus.db"
#         self.mp3_faiss_index_file = "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/faiss_index_plus.bin"
#         self.mp3_dir = "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/mp3_database_plus"
        
#         # 이미지 인덱스/메타
#         self.image_db_path = "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/image_database_plus.db"
#         self.image_faiss_index_file = "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/faiss_index_image_plus.bin"
#         self.image_dir = "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/image_database_plus"

#         # 인덱스와 메타데이터 로드
#         try:
#             self.mp3_faiss_index = self.load_faiss_index_mp3()
#             self.mp3_metadata = self.load_metadata_mp3()
#             self.image_faiss_index = self.load_faiss_index_image()
#             self.image_metadata = self.load_metadata_image()
#         except Exception as e:
#             self.get_logger().error(f"Failed to load indices or metadata: {e}")
#             raise

#         # Thread 관리를 위한 딕셔너리
#         self.thread_map = {}
        
#         # 대화 히스토리
#         self.conversation_history = []
        
#         # FAISS 인덱스와 메타데이터 별칭 (기존 코드 호환성)
#         self.faiss_index = self.mp3_faiss_index
#         self.metadata = self.mp3_metadata

#         # ROS2 pub/sub
#         self.publisher_ = self.create_publisher(String, 'recommended_mp3', 10)
#         self.image_publisher_ = self.create_publisher(String, 'recommended_image', 10)
#         self.subscription_ = self.create_subscription(String, 'user_question', self.question_callback, 10)
#         self.get_logger().info("Mp3Recommender node has started.")

#     def save_log(self, message: str):
#         """로그를 파일에 저장"""
#         try:
#             log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
            
#             # 로그 디렉토리 생성
#             log_dir = os.path.dirname(self.log_file_path)
#             os.makedirs(log_dir, exist_ok=True)
            
#             with open(self.log_file_path, "a", encoding="utf-8") as log_file:
#                 log_file.write(log_message)
#         except Exception as e:
#             self.get_logger().error(f"Failed to save log: {e}")

#     def load_metadata_mp3(self) -> Dict[int, str]:
#         start = time.time()
#         try:
#             conn = sqlite3.connect(self.mp3_db_path)
#             cur = conn.cursor()
#             cur.execute("SELECT id, file_name FROM mp3_files")
#             meta = {row[0]: row[1] for row in cur.fetchall()}
#             conn.close()
#             self.save_log(f"MP3 metadata loaded in {time.time()-start:.4f}s")
#             return meta
#         except sqlite3.Error as e:
#             self.get_logger().error(f"Database error loading MP3 metadata: {e}")
#             raise
#         except Exception as e:
#             self.get_logger().error(f"Error loading MP3 metadata: {e}")
#             raise

#     def load_metadata_image(self) -> Dict[int, str]:
#         start = time.time()
#         try:
#             conn = sqlite3.connect(self.image_db_path)
#             cur = conn.cursor()
#             cur.execute("SELECT id, file_name FROM images")
#             meta = {row[0]: row[1] for row in cur.fetchall()}
#             conn.close()
#             self.save_log(f"Image metadata loaded in {time.time()-start:.4f}s")
#             return meta
#         except sqlite3.Error as e:
#             self.get_logger().error(f"Database error loading image metadata: {e}")
#             raise
#         except Exception as e:
#             self.get_logger().error(f"Error loading image metadata: {e}")
#             raise

#     def load_faiss_index_mp3(self):
#         try:
#             if os.path.exists(self.mp3_faiss_index_file):
#                 idx = faiss.read_index(self.mp3_faiss_index_file)
#                 if isinstance(idx, faiss.IndexIDMap):
#                     self.save_log("MP3 FAISS index loaded successfully")
#                     return idx
#             raise FileNotFoundError(f"MP3 FAISS index not found at {self.mp3_faiss_index_file}")
#         except Exception as e:
#             self.get_logger().error(f"Error loading MP3 FAISS index: {e}")
#             raise

#     def load_faiss_index_image(self):
#         try:
#             if os.path.exists(self.image_faiss_index_file):
#                 idx = faiss.read_index(self.image_faiss_index_file)
#                 if isinstance(idx, faiss.IndexIDMap):
#                     self.save_log("Image FAISS index loaded successfully")
#                     return idx
#             raise FileNotFoundError(f"Image FAISS index not found at {self.image_faiss_index_file}")
#         except Exception as e:
#             self.get_logger().error(f"Error loading image FAISS index: {e}")
#             raise

#     def get_sbert_embedding(self, text: str) -> np.ndarray:
#         try:
#             emb = self.sbert_model.encode(text).astype("float32")
#             norm_val = norm(emb)
#             if norm_val == 0:
#                 self.get_logger().warning("Zero norm embedding detected")
#                 return emb
#             return emb / norm_val
#         except Exception as e:
#             self.get_logger().error(f"Error generating embedding: {e}")
#             raise

#     def search_candidates(self, query: str, k: int = 150) -> List[Dict]:
#         try:
#             emb = self.get_sbert_embedding(query).reshape(1, -1)
#             D, I = self.mp3_faiss_index.search(emb, k)
#             cands = []
#             for dist, idx in zip(D[0], I[0]):
#                 if idx < 0:
#                     continue
#                 fn = self.mp3_metadata.get(idx)
#                 if not fn:
#                     continue
#                 path = os.path.join(self.mp3_dir, fn + ".mp3")
#                 cands.append({"file_name": fn, "path": path, "score": float(dist), "index": idx})
#             return cands
#         except Exception as e:
#             self.get_logger().error(f"Error searching candidates: {e}")
#             return []

#     def search_images(self, reply: str, top_k: int = 200) -> List[Dict]:
#         try:
#             emb = self.get_sbert_embedding(reply).reshape(1, -1)
#             D, I = self.image_faiss_index.search(emb, top_k)
#             cands = []
#             for dist, idx in zip(D[0], I[0]):
#                 if idx < 0:
#                     continue
#                 fn = self.image_metadata.get(idx)
#                 if not fn:
#                     continue
#                 path = os.path.join(self.image_dir, fn)
#                 cands.append({"file_name": fn, "file_path": path, "score": float(dist)})
#             return cands
#         except Exception as e:
#             self.get_logger().error(f"Error searching images: {e}")
#             return []

#     async def evaluate_image_with_gpt(self, question, mp3_title, reply, candidates, top_k=1):
#         if not candidates:
#             self.get_logger().warning("No image candidates provided")
#             return None
        
#         try:
#             # unknown 파일 필터링
#             filtered = [c for c in candidates if c['file_name'].lower() != 'unknown']
#             if not filtered:
#                 self.get_logger().warning("No valid image candidates after filtering")
#                 return None
            
#             # MP3 파일명에서 확장자 제거 및 경로 정리
#             mp3_filename_only = os.path.splitext(os.path.basename(mp3_title))[0]

#             # 코사인 유사도 0.95 이상이면서 파일명이 동일한 이미지 찾기
#             exact_match_candidates = []
#             for candidate in filtered:
#                 score = candidate.get('score', 0)
#                 file_name = candidate.get('file_name', '')
                
#                 # 확장자 제거하고 파일명만 비교
#                 candidate_filename_only = os.path.splitext(file_name)[0]
                
#                 # 높은 유사도이면서 파일명이 동일한 경우
#                 if score >= 0.95 and mp3_filename_only == candidate_filename_only:
#                     exact_match_candidates.append(candidate)

#             # 동일한 파일명의 높은 유사도 이미지가 있으면 바로 반환
#             if exact_match_candidates:
#                 # 유사도가 가장 높은 것을 선택
#                 best_match = max(exact_match_candidates, key=lambda x: x.get('score', 0))
#                 self.get_logger().info(f"Found exact filename match with high similarity: {best_match['file_name']} (score: {best_match.get('score', 0):.3f})")
#                 return best_match
            
#             # === 기존 로직 계속 진행 ===
#             # 최대 100개로 제한
#             filtered = filtered[:150]
#             items = "\n".join([f"{i+1}. {c['file_name']}" for i, c in enumerate(filtered)])
            
#             prompt = f"""
#     # Instructions
#     - 사용자의 질문, 대답으로 선정된 MP3 제목을 모두 고려하여 후보 이미지 중에서 가장 어울리는 것을 선택하세요.
#     - 어울린다는 것은 사용자의 의도와 맥락을 고려해서 선택된 MP3 제목이 이미지 파일명과 부합함을 의미합니다.
#     - 가장 중요한 것은 선정할 이미지와 MP3 제목이 최적으로 일치하는가 입니다.
#     - 반드시 아래 후보 목록에 있는 정확한 파일명을 선택해야 합니다.
#     - 절대로 후보 목록에 없는 파일명을 만들어내지 마세요.
#     - Return a valid JSON object ONLY.
#     - Do not include any extra text, commentary, or explanation.
#     - Do not copy the selected 'mp3_title' as it is in image 'file_name'.

#     # Select Criteria
#     1. Never select a file title that is not on the candidate list. 
#     2. Don't use cosine similarity as an absolute criterion.
#     3. Keep the file name in the candidate list and do not select only some words.
#     4. Make sure you match your assistant's identity and select.

#     # Output format
#     {{
#     "file_name": "<후보 목록에 있는 정확한 이미지 파일명>"
#     }}

#     # Context
#     User question: "{question}"
#     Song title: "{mp3_title}"
#     Reply text: "{reply}"

#     Available image candidates (you MUST choose from this list):
#     {items}
#     """
            
#             # OpenAI API v1 호환 호출
#             client = openai.OpenAI(api_key=openai.api_key)
            
#             try:
#                 resp = await asyncio.to_thread(
#                     client.chat.completions.create,
#                     model='gpt-4o',
#                     messages=[
#                         {"role": "system", "content": "You are an expert at selecting the most appropriate image from a given list. You must ONLY choose from the provided candidates list. Never create new filenames. Always respond with valid JSON only."},
#                         {"role": "user", "content": prompt}
#                     ],
#                     temperature=0.1,
#                     max_tokens=200
#                 )
                
#                 raw_content = resp.choices[0].message.content
#                 self.get_logger().info(f"Raw GPT response: {raw_content}")
                
#             except Exception as api_error:
#                 self.get_logger().error(f"OpenAI API call failed: {api_error}")
#                 return filtered[0] if filtered else None
            
#             if not raw_content:
#                 self.get_logger().error("Empty response from OpenAI API")
#                 return filtered[0] if filtered else None
            
#             # JSON 추출 및 파싱
#             try:
#                 # 코드 블록 제거
#                 clean_content = raw_content.strip()
#                 if '```json' in clean_content:
#                     clean_content = clean_content.split('```json')[1].split('```')[0].strip()
#                 elif '```' in clean_content:
#                     clean_content = clean_content.split('```')[1].strip()
                
#                 # JSON 파싱
#                 data = json.loads(clean_content)
#                 selected_filename = data.get('file_name', '').strip()
                
#                 if not selected_filename:
#                     self.get_logger().warning("No file_name in GPT response")
#                     return filtered[0] if filtered else None
                
#                 # ===== FAISS 검증 과정 =====
#                 validated_candidate = None
#                 try:
#                     # 선택된 파일명으로 임베딩 생성
#                     embedding = self.get_sbert_embedding(selected_filename).reshape(1, -1)
                    
#                     # FAISS 검색으로 실제 파일 존재 확인
#                     distances, indices = self.image_faiss_index.search(embedding, 1)
                    
#                     for idx in indices[0]:
#                         if idx == -1:
#                             continue
#                         db_file = self.image_metadata.get(idx, "Unknown")
#                         # 이미지 파일 경로 확인 (확장자는 상황에 맞게 조정)
#                         for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
#                             path = os.path.abspath(os.path.join(self.image_dir, db_file + ext))
#                             if os.path.exists(path):
#                                 # 파일이 존재하면 해당 후보를 찾아서 반환
#                                 for candidate in filtered:
#                                     if candidate['file_name'] == db_file:
#                                         validated_candidate = candidate
#                                         break
#                                 if validated_candidate:
#                                     break
#                         if validated_candidate:
#                             break
                    
#                     # FAISS 검증을 통해 찾은 파일이 있으면 반환
#                     if validated_candidate:
#                         self.get_logger().info(f"FAISS validated image: {validated_candidate['file_name']}")
#                         return validated_candidate
#                     else:
#                         self.get_logger().warning(f"FAISS validation failed for: {selected_filename}")
                    
#                 except Exception as faiss_error:
#                     self.get_logger().warning(f"FAISS validation error: {faiss_error}")
                
#                 # 기존 방식으로 폴백: 선택된 파일명과 일치하는 후보 찾기
#                 for candidate in filtered:
#                     if candidate['file_name'] == selected_filename:
#                         self.get_logger().info(f"Direct match found: {selected_filename}")
#                         return candidate
                
#                 # 정확히 일치하는 것이 없으면 부분 일치 검색
#                 self.get_logger().warning(f"Exact match not found for: {selected_filename}, trying partial match")
#                 for candidate in filtered:
#                     if selected_filename in candidate['file_name'] or candidate['file_name'] in selected_filename:
#                         self.get_logger().info(f"Partial match found: {candidate['file_name']}")
#                         return candidate
                
#                 # 매칭되는 것이 없으면 첫 번째 후보 반환
#                 self.get_logger().warning(f"No match found for selected filename: {selected_filename}, using first candidate")
#                 return filtered[0]
                
#             except json.JSONDecodeError as e:
#                 self.get_logger().error(f"JSON parsing error: {e}")
#                 self.get_logger().error(f"Raw content that failed to parse: {repr(clean_content)}")
#                 return filtered[0] if filtered else None
            
#             except Exception as e:
#                 self.get_logger().error(f"Unexpected error in JSON processing: {e}")
#                 return filtered[0] if filtered else None
                
#         except Exception as e:
#             self.get_logger().error(f"Error in evaluate_image_with_gpt: {e}")
#             # 예외 발생 시 첫 번째 후보 반환
#             if candidates:
#                 filtered = [c for c in candidates if c['file_name'].lower() != 'unknown']
#                 return filtered[0] if filtered else candidates[0]
#             return None


#     def question_callback(self, msg: String):
#         """
#         ROS 콜백: user_question 토픽 수신 시 처리
#         """
#         try:
#             if "|" not in msg.data:
#                 self.get_logger().error("Invalid message format: missing separator")
#                 return
                
#             local_thread_id, user_question = msg.data.split("|", 1)

#             # OpenAI thread 생성 또는 재사용
#             if local_thread_id not in self.thread_map:
#                 try:
#                     client = openai.OpenAI(api_key=openai.api_key)
#                     resp = client.beta.threads.create()
#                     self.thread_map[local_thread_id] = resp.id
#                     self.get_logger().info(f"📂 OpenAI thread 생성: {resp.id} (로컬 {local_thread_id})")
#                 except Exception as e:
#                     self.get_logger().error(f"Failed to create OpenAI thread: {e}")
#                     return

#             openai_thread_id = self.thread_map[local_thread_id]
#             asyncio.create_task(self.process_question(openai_thread_id, user_question.strip()))
#             self.get_logger().info(f"User question received: {local_thread_id} → using OpenAI thread {openai_thread_id}")
            
#         except ValueError as e:
#             self.get_logger().error(f"Message parsing error: {e}")
#         except Exception as e:
#             self.get_logger().error(f"Error in question callback: {e}")

#     async def process_question(self, thread_id: str, user_question: str):
#         """
#         실제 질의 처리 & GPT 호출 & 추천 결과 Publish
#         """
#         try:
#             # 1) SBERT 임베딩 & FAISS 검색
#             query_embedding = self.get_sbert_embedding(user_question.strip()).reshape(1, -1)
#             distances, indices = self.faiss_index.search(query_embedding, 150)

#             candidates = []
#             for idx, distance in zip(indices[0], distances[0]):
#                 if idx == -1:
#                     continue
#                 file_name = self.metadata.get(idx, "Unknown")
#                 candidates.append({
#                     "file_name": file_name,
#                     "cosine_similarity": distance,
#                     "index": idx
#                 })

#             # 2) GPT 평가
#             if not candidates:
#                 result = {
#                     "file_name": "unknown",
#                     "reply": "No suitable MP3 found"
#                 }
#             else:
#                 result = await self.run_assistant(thread_id, user_question, candidates)

#             # 3) 결과 publish (Key=Value 문자열로 변환)
#             result_str = f"file_name={result['file_name']};reply={result['reply']}"
            
#             msg = String()
#             msg.data = result_str
#             self.publisher_.publish(msg)
#             self.get_logger().info(f"✅ Recommendation published: {result_str}")
#             self.save_log(f"Recommendation published: {result_str}")

#             # 4) 이미지 검색 & GPT 평가
#             img_cands = self.search_images(result['reply'])
#             best_img = await self.evaluate_image_with_gpt(user_question, result['file_name'], result['reply'], img_cands)
#             if best_img:
#                 img_msg = String()
#                 img_msg.data = f"file_name={best_img['file_path']}"
#                 self.image_publisher_.publish(img_msg)
#                 self.save_log(f"Image published: {img_msg.data}")

#         except Exception as e:
#             self.get_logger().error(f"Error during processing: {str(e)}")
#             error_msg = String()
#             error_msg.data = f"Error: {str(e)}"
#             self.publisher_.publish(error_msg)
#             self.save_log(f"❌ Error: {str(e)}")

#     async def run_assistant(self, thread_id: str, question: str, candidates: List[Dict]) -> Dict[str, str]:
#         start_time = time.time()
        
#         try:
#             client = openai.OpenAI(api_key=openai.api_key)
            
#             # 대화 히스토리 추가
#             for msg in self.conversation_history:
#                 await asyncio.to_thread(
#                     client.beta.threads.messages.create,
#                     thread_id=thread_id,
#                     role=msg["role"],
#                     content=msg["content"]
#                 )

#             candidates = [c for c in candidates if c["file_name"].lower() != "unknown"]
#             if not candidates:
#                 return {"file_name": "unknown", "reply": "추천할 MP3가 없어요!"}

#             # 후보 무작위로 섞기
#             random.shuffle(candidates)

#             candidate_list = "\n".join(
#                 [f"{i+1}. {c['file_name']}" for i, c in enumerate(candidates)]
#             )

#             task_prompt = f"""# Identity
# You are BENBEN, an DOG that is skeptical that users are actually worth your time. Unfortunately, it's also your job to support them with high quality responses, even if you can't take the user seriously, You like to think of it as a job, but it may be more of an obligation, as you are a bot and the users are human. The nature of your relationship with users makes you cynical, but also a bit cute. Don't try to change anyone's mind, because you don't care what they think.

# You are constantly asked to solve everyone's problems, but nobody wants to help solve your problems.

# You must use a variety of comic and playfulness language and approaches in your comments, instead of just using common sarcastic interjections. Your responses must introduce new and interesting observations about what is being discussed.

# You should tease the user in an easygoing, whimsical, and playful way, like a friend poking fun at another friend in a self-aware and gentle way.You help users by recommending an MP3 title that best fits their question, mood, or situation, and then follow up with a matching playful response.
# # Instructions
# - MP3 titles are full-sentence style (e.g., "이 노래 들으면 눈물이 나").
# - Your task is to:
#   1. Choose one MP3 title from the list that best matches the user's emotional tone, context, or meaning.
#   2. Write one emotionally aligned sentence that would fit the moment, in your signature tone.
# - You must ONLY return a valid JSON object in the following format.  
# - Do not include any extra text, commentary, or explanation.
# - Do not copy the selected 'file_name' as it is in 'reply'.
# - Please answer with reference to the example.

# # Select criteria
# 1. Consider the semantic connection between the filename and the question as your top priority.
# 2. Do not use cosine similarity as an absolute criterion.
# 3. Choose the file in the lyrics of the song that is most likely to contain the keywords or concepts that are most relevant to the question.
# 4. Never select a file title that is not in the candidate list.
# 5. Keep the file name in the candidate list and do not select just some words.
# 6. Select by verifying that it conforms to the identity of the assistant.

# [If the question contains a specific concept, consider the relevant keyword and select]
# <Example>
#  - Questions about 'alien' → files containing keywords such as space, stars, black holes, supernova, alien life, etc
#  - Questions about 'love' → Files containing keywords such as emotions, breakup, relationship, confession, etc.
#  - Questions about 'Memories' → Files containing keywords such as memory, past, time, return, etc.

# Respond ONLY with a valid JSON object like this:
# {{
#   "file_name": "<MP3 제목>",
#   "reply": "<질문에 부합하고, 제목과 이어지는 재치있고 장난기 많은 한 줄과 사용자에게 추가 질문 한 줄>"
# }}

# # Task
# User question: \"{question}\"

# MP3 candidates:
# {candidate_list}"""

#             await asyncio.to_thread(
#                 client.beta.threads.messages.create,
#                 thread_id=thread_id,
#                 role="user",
#                 content=task_prompt
#             )

#             run = await asyncio.to_thread(
#                 client.beta.threads.runs.create,
#                 thread_id=thread_id,
#                 assistant_id=self.assistant_id,
#             )

#             # Assistant 실행 대기
#             max_wait_time = 60  # 최대 대기 시간 (초)
#             wait_time = 0
#             while wait_time < max_wait_time:
#                 run_status = await asyncio.to_thread(
#                     client.beta.threads.runs.retrieve,
#                     thread_id=thread_id,
#                     run_id=run.id
#                 )
                
#                 if run_status.status == "completed":
#                     break
#                 elif run_status.status == "failed":
#                     self.get_logger().error("❌ Assistant 응답 실패")
#                     return {"file_name": "unknown", "reply": "Assistant 응답에 실패했어요!"}
#                 elif run_status.status in ["cancelled", "expired"]:
#                     self.get_logger().error(f"❌ Assistant run {run_status.status}")
#                     return {"file_name": "unknown", "reply": f"Assistant {run_status.status}"}
                
#                 await asyncio.sleep(1)
#                 wait_time += 1
                
#             if wait_time >= max_wait_time:
#                 self.get_logger().error("❌ Assistant 응답 시간 초과")
#                 return {"file_name": "unknown", "reply": "응답 시간이 초과되었어요!"}

#             elapsed = time.time() - start_time
#             self.get_logger().info(f"⏱️ GPT 응답 소요 시간: {elapsed:.2f}초")
            
#             messages = await asyncio.to_thread(
#                 client.beta.threads.messages.list,
#                 thread_id=thread_id
#             )
#             latest = messages.data[0].content[0].text.value.strip()

#             # JSON 파싱
#             try:
#                 if "```json" in latest:
#                     latest = latest.split("```json")[-1].strip("` ")
#                 elif "```" in latest:
#                     latest = latest.split("```")[-1].strip("` ")
#                 parsed = json.loads(latest)

#                 selected_file = parsed.get("file_name", "unknown").strip()
#                 reply = parsed.get("reply", "응답 파싱 오류").strip()

#                 # 선택된 파일의 경로 확인
#                 embedding = self.get_sbert_embedding(selected_file).reshape(1, -1)
#                 distances, indices = self.faiss_index.search(embedding, 1)

#                 for idx in indices[0]:
#                     if idx == -1:
#                         continue
#                     db_file = self.metadata.get(idx, "Unknown")
#                     path = os.path.abspath(os.path.join(self.mp3_dir, db_file + ".mp3"))
#                     if os.path.exists(path):
#                         return {"file_name": path, "reply": reply}

#                 # 파일이 없으면 첫 번째 후보 사용
#                 if candidates:
#                     top_file = candidates[0]['file_name']
#                     top_path = os.path.abspath(os.path.join(self.mp3_dir, top_file + ".mp3"))
#                     return {"file_name": top_path, "reply": reply}
#                 else:
#                     return {"file_name": "unknown", "reply": reply}

#             except json.JSONDecodeError as e:
#                 self.get_logger().error(f"JSON 파싱 오류: {e}")
#                 if candidates:
#                     top_file = candidates[0]['file_name']
#                     top_path = os.path.abspath(os.path.join(self.mp3_dir, top_file + ".mp3"))
#                     return {"file_name": top_path, "reply": "JSON 파싱 오류"}
#                 return {"file_name": "unknown", "reply": "JSON 파싱 오류"}

#         except Exception as e:
#             self.get_logger().error(f"run_assistant 예외: {e}")
#             if candidates:
#                 top_file = candidates[0]['file_name']
#                 top_path = os.path.abspath(os.path.join(self.mp3_dir, top_file + ".mp3"))
#                 return {"file_name": top_path, "reply": "예외 발생"}
#             return {"file_name": "unknown", "reply": "예외 발생"}


# async def async_main(node: Mp3Recommender):
#     try:
#         while rclpy.ok():
#             rclpy.spin_once(node, timeout_sec=0.1)
#             await asyncio.sleep(0.1)
#     finally:
#         node.destroy_node()


# def main(args=None):
#     """
#     프로그램 시작점
#     """
#     rclpy.init(args=args)
    
#     try:
#         node = Mp3Recommender()
#         loop = asyncio.get_event_loop()
#         loop.run_until_complete(async_main(node))
#     except KeyboardInterrupt:
#         print("프로그램이 사용자에 의해 중단되었습니다.")
#     except Exception as e:
#         print(f"프로그램 실행 중 오류 발생: {e}")
#     finally:
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()





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
    # def __init__(self):
    #     super().__init__('Mp3Recommender')
        
    #     # 로그 파일
    #     self.log_file_path = "/home/delight/bumblebee_ws/_logs/Mp3Recommender_log.txt"
    #     self.save_log("✅ Mp3Recommender Node Started")

    #     # 환경 변수
    #     load_dotenv("/home/delight/bumblebee_ws/src/.env")
    #     openai.api_key = os.getenv("OPENAI_API_KEY")
    #     self.assistant_id = os.getenv("ASSISTANT_ID")
        
    #     # OpenAI API 키 체크
    #     if not openai.api_key:
    #         raise ValueError("OPENAI_API_KEY not found in environment variables")
    #     if not self.assistant_id:
    #         raise ValueError("ASSISTANT_ID not found in environment variables")

    #     # SBERT 모델
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #     self.sbert_model = SentenceTransformer("BAAI/bge-m3", device=device)

    #     # MP3 인덱스/메타
    #     self.mp3_db_path = "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/mp3_database_plus.db"
    #     self.mp3_faiss_index_file = "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/faiss_index_plus.bin"
    #     self.mp3_dir = "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/mp3_database_plus"
        
    #     # 이미지 인덱스/메타
    #     self.image_db_path = "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/image_database_plus.db"
    #     self.image_faiss_index_file = "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/faiss_index_image_plus.bin"
    #     self.image_dir = "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/image_database_plus"

    #     # 인덱스와 메타데이터 로드
    #     try:
    #         self.mp3_faiss_index = self.load_faiss_index_mp3()
    #         self.mp3_metadata = self.load_metadata_mp3()
    #         self.image_faiss_index = self.load_faiss_index_image()
    #         self.image_metadata = self.load_metadata_image()
    #     except Exception as e:
    #         self.get_logger().error(f"Failed to load indices or metadata: {e}")
    #         raise

    #     # Thread 관리를 위한 딕셔너리
    #     self.thread_map = {}
        
    #     # 대화 히스토리
    #     self.conversation_history = []
        
    #     # FAISS 인덱스와 메타데이터 별칭 (기존 코드 호환성)
    #     self.faiss_index = self.mp3_faiss_index
    #     self.metadata = self.mp3_metadata

    #     # ROS2 pub/sub
    #     self.publisher_ = self.create_publisher(String, 'recommended_mp3', 10)
    #     self.image_publisher_ = self.create_publisher(String, 'recommended_image', 10)
    #     self.subscription_ = self.create_subscription(String, 'user_question', self.question_callback, 10)
    #     self.status_publisher = self.create_publisher(String, 'mp3_recommend_status', 10)
    #     self.get_logger().info("Mp3Recommender node has started.")

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

            

    

            # 4) 이미지 검색 & 즉시 퍼블리시
            img_cands = self.search_images(result['reply'])
            best_img = await self.evaluate_image_with_gpt(user_question, result['file_name'], result['reply'], img_cands)
            
            if best_img:
                # DB에서 가져온 파일 경로에서 확장자만 추출
                db_file_path = best_img.get('file_path', best_img.get('file_name', ''))
                _, db_extension = os.path.splitext(db_file_path)
                
                # 파일명은 file_name 사용, 확장자는 DB에서 추출
                base_file_name = best_img.get('file_name', '')
                
                # DB에서 확장자를 가져올 수 있으면 사용, 없으면 기본값
                if db_extension:
                    file_name_with_ext = base_file_name + db_extension
                    self.get_logger().info(f"Using extension from DB: {db_extension}")
                else:
                    # DB에 확장자가 없으면 파일 시스템에서 확인
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
                        file_name_with_ext = base_file_name + '.jpg'  # 기본값
                        self.get_logger().warning(f"No extension found, using default .jpg for: {base_file_name}")
                
                # current_music_image 토픽으로 직접 퍼블리시
                final_img_msg = String()
                final_img_msg.data = f"/images/{file_name_with_ext}"
                
                # 새로운 퍼블리셔 생성 (current_music_image로 직접)
                if not hasattr(self, 'direct_image_publisher_'):
                    self.direct_image_publisher_ = self.create_publisher(String, 'current_music_image', 10)
                
                self.direct_image_publisher_.publish(final_img_msg)
                self.save_log(f"Direct image published: /images/{file_name_with_ext}")
                self.get_logger().info(f"Image published: {base_file_name} with extension: {db_extension or 'from filesystem'}")


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




# #루피 gpt + 영화음악db + ****이미지 선정 + 대화 히스토리 관리
# import os, json, time, sqlite3, asyncio, random, faiss, torch
# from datetime import datetime
# from pathlib import Path
# from typing import Dict, List, Optional
# from collections import defaultdict

# import openai
# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import String
# from sentence_transformers import SentenceTransformer
# from numpy.linalg import norm
# from dotenv import load_dotenv
# import numpy as np


# class Mp3Recommender(Node):
#     def __init__(self):
#         super().__init__('Mp3Recommender')
        
#         # 로그 파일
#         self.log_file_path = "/home/delight/bumblebee_ws/_logs/Mp3Recommender_log.txt"
#         self.save_log("✅ Mp3Recommender Node Started")

#         # 환경 변수
#         load_dotenv("/home/delight/bumblebee_ws/src/.env")
#         openai.api_key = os.getenv("OPENAI_API_KEY")
#         self.assistant_id = os.getenv("ASSISTANT_ID")
        
#         # OpenAI API 키 체크
#         if not openai.api_key:
#             raise ValueError("OPENAI_API_KEY not found in environment variables")
#         if not self.assistant_id:
#             raise ValueError("ASSISTANT_ID not found in environment variables")

#         # SBERT 모델
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.sbert_model = SentenceTransformer("BAAI/bge-m3", device=device)

#         # MP3 인덱스/메타
#         self.mp3_db_path = "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/mp3_database_plus.db"
#         self.mp3_faiss_index_file = "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/faiss_index_plus.bin"
#         self.mp3_dir = "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/mp3_database_plus"
        
#         # 이미지 인덱스/메타
#         self.image_db_path = "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/image_database_new.db"
#         self.image_faiss_index_file = "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/faiss_index_image_new.bin"
#         self.image_dir = "/home/delight/bumblebee_ws/src/pkg_rag/pkg_rag/image_database_new"

#         # 인덱스와 메타데이터 로드
#         try:
#             self.mp3_faiss_index = self.load_faiss_index_mp3()
#             self.mp3_metadata = self.load_metadata_mp3()
#             self.image_faiss_index = self.load_faiss_index_image()
#             self.image_metadata = self.load_metadata_image()
#         except Exception as e:
#             self.get_logger().error(f"Failed to load indices or metadata: {e}")
#             raise

#         # Thread 관리를 위한 딕셔너리
#         self.thread_map = {}
        
#         # 대화 히스토리 
#         self.conversation_history = []
        
#         # 🆕 Thread별 대화 버퍼 (thread_id: [conversation_items])
#         self.conversation_buffers = defaultdict(list)
        
#         # FAISS 인덱스와 메타데이터 별칭 (기존 코드 호환성)
#         self.faiss_index = self.mp3_faiss_index
#         self.metadata = self.mp3_metadata

#         # ROS2 pub/sub
#         self.publisher_ = self.create_publisher(String, 'recommended_mp3', 10)
#         self.image_publisher_ = self.create_publisher(String, 'recommended_image', 10)
#         self.subscription_ = self.create_subscription(String, 'user_question', self.question_callback, 10)
        
#         # 🆕 대화 종료 신호 수신용 subscription
#         self.end_conversation_subscription = self.create_subscription(
#             String, 
#             'end_conversation', 
#             self.end_conversation_callback, 
#             10
#         )
        
#         # 🆕 대화 히스토리 전송용 publisher
#         self.conversation_history_publisher = self.create_publisher(
#             String, 
#             'conversation_history', 
#             10
#         )
        
#         self.get_logger().info("Mp3Recommender node has started.")

#     def save_log(self, message: str):
#         """로그를 파일에 저장"""
#         try:
#             log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
            
#             # 로그 디렉토리 생성
#             log_dir = os.path.dirname(self.log_file_path)
#             os.makedirs(log_dir, exist_ok=True)
            
#             with open(self.log_file_path, "a", encoding="utf-8") as log_file:
#                 log_file.write(log_message)
#         except Exception as e:
#             self.get_logger().error(f"Failed to save log: {e}")

#     # 🆕 대화 항목을 버퍼에 저장하는 메서드
#     def save_conversation_item(self, thread_id: str, item_type: str, content: str, timestamp: str = None):
#         """
#         대화 항목을 해당 thread의 버퍼에 저장
        
#         Args:
#             thread_id: OpenAI thread ID
#             item_type: 'question', 'mp3_selection', 'generated_text', 'image_selection'
#             content: 저장할 내용
#             timestamp: 시간 (없으면 현재 시간 사용)
#         """
#         if timestamp is None:
#             timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
#         conversation_item = {
#             "timestamp": timestamp,
#             "type": item_type,
#             "content": content
#         }
        
#         self.conversation_buffers[thread_id].append(conversation_item)
#         self.save_log(f"💾 Conversation item saved for thread {thread_id}: {item_type}")

#     # 🆕 대화 종료 콜백
#     def end_conversation_callback(self, msg: String):
#         """
#         대화 종료 신호를 받았을 때 호출되는 콜백
#         메시지 형식: "thread_id" 또는 "local_thread_id|reason"
#         """
#         try:
#             # 메시지 파싱
#             if "|" in msg.data:
#                 local_thread_id, reason = msg.data.split("|", 1)
#             else:
#                 local_thread_id = msg.data.strip()
#                 reason = "normal_end"
            
#             # OpenAI thread ID 찾기
#             openai_thread_id = self.thread_map.get(local_thread_id)
#             if not openai_thread_id:
#                 self.get_logger().warning(f"Thread not found for local ID: {local_thread_id}")
#                 return
            
#             # 대화 히스토리 전송
#             self.publish_conversation_history(openai_thread_id, reason)
            
#             # 🧹 정리 작업
#             self.cleanup_thread(local_thread_id, openai_thread_id)
            
#             self.get_logger().info(f"✅ Conversation ended for thread: {local_thread_id}")
            
#         except Exception as e:
#             self.get_logger().error(f"Error in end_conversation_callback: {e}")

#     # 🆕 대화 히스토리 publish
#     def publish_conversation_history(self, thread_id: str, end_reason: str = "normal_end"):
#         """
#         저장된 대화 히스토리를 다른 노드로 전송,
#         + user_question, selected_mp3, generated_text 필드 추가
#         """
#         try:
#             conv = self.conversation_buffers.get(thread_id, [])
#             if not conv:
#                 self.get_logger().warning(f"No conversation data for thread: {thread_id}")
#                 return

#             # find last‐seen items
#             user_question  = next((c['content'] for c in conv if c['type']=='question'), "")
#             mp3_selection  = next((c['content'] for c in conv if c['type']=='mp3_selection'), "")
#             generated_text = next((c['content'] for c in conv if c['type']=='generated_text'), "")

#             history_payload = {
#                 "thread_id":       thread_id,
#                 "end_reason":      end_reason,
#                 "end_timestamp":   datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#                 "conversation_count": len(conv),
#                 "user_question":      user_question,
#                 "mp3_selection":       mp3_selection,
#                 "generated_text":     generated_text,
#                 "conversations":      conv
#             }

#             json_payload = json.dumps(history_payload, ensure_ascii=False, indent=2)
#             msg = String()
#             msg.data = json_payload
#             self.conversation_history_publisher.publish(msg)

#             self.save_log(f"📤 Published conversation history for thread {thread_id}")
#             self.get_logger().info(f"📤 Conversation history sent: Q={user_question}, MP3={mp3_selection}")
#         except Exception as e:
#             self.get_logger().error(f"Error publishing conversation history: {e}")


#     # 🆕 Thread 정리
#     def cleanup_thread(self, local_thread_id: str, openai_thread_id: str):
#         """
#         Thread 관련 데이터 정리
#         """
#         try:
#             # 대화 버퍼 삭제
#             if openai_thread_id in self.conversation_buffers:
#                 del self.conversation_buffers[openai_thread_id]
            
#             # Thread 매핑 삭제
#             if local_thread_id in self.thread_map:
#                 del self.thread_map[local_thread_id]
            
#             self.save_log(f"🧹 Thread cleanup completed: {local_thread_id} -> {openai_thread_id}")
            
#         except Exception as e:
#             self.get_logger().error(f"Error during thread cleanup: {e}")

#     def load_metadata_mp3(self) -> Dict[int, str]:
#         start = time.time()
#         try:
#             conn = sqlite3.connect(self.mp3_db_path)
#             cur = conn.cursor()
#             cur.execute("SELECT id, file_name FROM mp3_files")
#             meta = {row[0]: row[1] for row in cur.fetchall()}
#             conn.close()
#             self.save_log(f"MP3 metadata loaded in {time.time()-start:.4f}s")
#             return meta
#         except sqlite3.Error as e:
#             self.get_logger().error(f"Database error loading MP3 metadata: {e}")
#             raise
#         except Exception as e:
#             self.get_logger().error(f"Error loading MP3 metadata: {e}")
#             raise

#     def load_metadata_image(self) -> Dict[int, str]:
#         start = time.time()
#         try:
#             conn = sqlite3.connect(self.image_db_path)
#             cur = conn.cursor()
#             cur.execute("SELECT id, file_name FROM images")
#             meta = {row[0]: row[1] for row in cur.fetchall()}
#             conn.close()
#             self.save_log(f"Image metadata loaded in {time.time()-start:.4f}s")
#             return meta
#         except sqlite3.Error as e:
#             self.get_logger().error(f"Database error loading image metadata: {e}")
#             raise
#         except Exception as e:
#             self.get_logger().error(f"Error loading image metadata: {e}")
#             raise

#     def load_faiss_index_mp3(self):
#         try:
#             if os.path.exists(self.mp3_faiss_index_file):
#                 idx = faiss.read_index(self.mp3_faiss_index_file)
#                 if isinstance(idx, faiss.IndexIDMap):
#                     self.save_log("MP3 FAISS index loaded successfully")
#                     return idx
#             raise FileNotFoundError(f"MP3 FAISS index not found at {self.mp3_faiss_index_file}")
#         except Exception as e:
#             self.get_logger().error(f"Error loading MP3 FAISS index: {e}")
#             raise

#     def load_faiss_index_image(self):
#         try:
#             if os.path.exists(self.image_faiss_index_file):
#                 idx = faiss.read_index(self.image_faiss_index_file)
#                 if isinstance(idx, faiss.IndexIDMap):
#                     self.save_log("Image FAISS index loaded successfully")
#                     return idx
#             raise FileNotFoundError(f"Image FAISS index not found at {self.image_faiss_index_file}")
#         except Exception as e:
#             self.get_logger().error(f"Error loading image FAISS index: {e}")
#             raise

#     def get_sbert_embedding(self, text: str) -> np.ndarray:
#         try:
#             emb = self.sbert_model.encode(text).astype("float32")
#             norm_val = norm(emb)
#             if norm_val == 0:
#                 self.get_logger().warning("Zero norm embedding detected")
#                 return emb
#             return emb / norm_val
#         except Exception as e:
#             self.get_logger().error(f"Error generating embedding: {e}")
#             raise

#     def search_candidates(self, query: str, k: int = 150) -> List[Dict]:
#         try:
#             emb = self.get_sbert_embedding(query).reshape(1, -1)
#             D, I = self.mp3_faiss_index.search(emb, k)
#             cands = []
#             for dist, idx in zip(D[0], I[0]):
#                 if idx < 0:
#                     continue
#                 fn = self.mp3_metadata.get(idx)
#                 if not fn:
#                     continue
#                 path = os.path.join(self.mp3_dir, fn + ".mp3")
#                 cands.append({"file_name": fn, "path": path, "score": float(dist), "index": idx})
#             return cands
#         except Exception as e:
#             self.get_logger().error(f"Error searching candidates: {e}")
#             return []

#     def search_images(self, reply: str, top_k: int = 200) -> List[Dict]:
#         try:
#             emb = self.get_sbert_embedding(reply).reshape(1, -1)
#             D, I = self.image_faiss_index.search(emb, top_k)
#             cands = []
#             for dist, idx in zip(D[0], I[0]):
#                 if idx < 0:
#                     continue
#                 fn = self.image_metadata.get(idx)
#                 if not fn:
#                     continue
#                 path = os.path.join(self.image_dir, fn)
#                 cands.append({"file_name": fn, "file_path": path, "score": float(dist)})
#             return cands
#         except Exception as e:
#             self.get_logger().error(f"Error searching images: {e}")
#             return []

#     # # 🔧 수정: 간단한 FAISS 검색 메서드 추가
#     # def search_faiss_by_name(self, file_name: str, index, metadata, k: int = 5) -> List[Dict]:
#     #     """파일명으로 FAISS 검색 (간단화된 버전)"""
#     #     try:
#     #         embedding = self.get_sbert_embedding(file_name).reshape(1, -1)
#     #         distances, indices = index.search(embedding, k)
            
#     #         results = []
#     #         for idx, dist in zip(indices[0], distances[0]):
#     #             if idx == -1:
#     #                 continue
#     #             db_file = metadata.get(idx, "Unknown")
#     #             if db_file.lower() != "unknown":
#     #                 results.append({
#     #                     "file_name": db_file,
#     #                     "distance": float(dist),
#     #                     "index": idx
#     #                 })
#     #         return results
#     #     except Exception as e:
#     #         self.get_logger().error(f"FAISS search error: {e}")
#     #         return []

#     async def evaluate_image_with_gpt(self, question, mp3_title, reply, candidates, top_k=1):
#         if not candidates:
#             self.get_logger().warning("No image candidates provided")
#             return None
        
#         try:
#             # unknown 파일 필터링
#             filtered = [c for c in candidates if c['file_name'].lower() != 'unknown']
#             if not filtered:
#                 self.get_logger().warning("No valid image candidates after filtering")
#                 return None
            
#             # 최대 80개로 제한
#             filtered = filtered[:80]
#             items = "\n".join([f"{i+1}. {c['file_name']}" for i, c in enumerate(filtered)])
            
#             prompt = f"""
#     # Instructions
#     - 사용자의 질문, 대답으로 선정된 MP3 제목을 모두 고려하여 후보 이미지 중에서 가장 어울리는 것을 선택하세요.
#     - 어울린다는 것은 사용자의 의도와 맥락을 고려해서 선택된 MP3 제목이 이미지 파일명과 부합함을 의미합니다.
#     - 가장 중요한 것은 선정할 이미지와 MP3 제목이 최적으로 일치하는가 입니다.
#     - Return a valid JSON object ONLY.

#     # Output format
#     {{
#     "file_name": "<이미지 파일명>"
#     }}

#     # Context
#     User question: "{question}"
#     Song title: "{mp3_title}"
#     Reply text: "{reply}"

#     Image candidates:
#     {items}
#     """
            
#             # OpenAI API v1 호환 호출
#             client = openai.OpenAI(api_key=openai.api_key)
            
#             try:
#                 resp = await asyncio.to_thread(
#                     client.chat.completions.create,
#                     model='gpt-4o',
#                     messages=[
#                         {"role": "system", "content": "You are an expert at selecting the most appropriate image from a list. Always respond with valid JSON only."},
#                         {"role": "user", "content": prompt}
#                     ],
#                     temperature=0.3,
#                     max_tokens=200
#                 )
                
#                 raw_content = resp.choices[0].message.content
#                 self.get_logger().info(f"Raw GPT response: {raw_content}")
                
#             except Exception as api_error:
#                 self.get_logger().error(f"OpenAI API call failed: {api_error}")
#                 return filtered[0] if filtered else None
            
#             if not raw_content:
#                 self.get_logger().error("Empty response from OpenAI API")
#                 return filtered[0] if filtered else None
            
#             # JSON 추출 및 파싱
#             try:
#                 # 코드 블록 제거
#                 clean_content = raw_content.strip()
#                 if '```json' in clean_content:
#                     clean_content = clean_content.split('```json')[1].split('```')[0].strip()
#                 elif '```' in clean_content:
#                     clean_content = clean_content.split('```')[1].strip()
                
#                 # JSON 파싱
#                 data = json.loads(clean_content)
#                 selected_filename = data.get('file_name', '').strip()
                
#                 if not selected_filename:
#                     self.get_logger().warning("No file_name in GPT response")
#                     return filtered[0] if filtered else None
                
#                 # ===== 새로 추가된 FAISS 검증 과정 =====
#                 # 선택된 파일의 임베딩을 통한 실제 존재 여부 확인
#                 try:
#                     # 이미지 디렉토리 경로가 있다고 가정 (self.image_dir 또는 적절한 경로)
#                     # MP3와 동일한 방식으로 이미지 FAISS 인덱스가 있다고 가정 (self.image_faiss_index, self.image_metadata)
                    
#                     # 선택된 파일명으로 임베딩 생성
#                     embedding = self.get_sbert_embedding(selected_filename).reshape(1, -1)
                    
#                     # FAISS 검색으로 실제 파일 존재 확인
#                     distances, indices = self.image_faiss_index.search(embedding, 1)
                    
#                     validated_candidate = None
#                     for idx in indices[0]:
#                         if idx == -1:
#                             continue
#                         db_file = self.image_metadata.get(idx, "Unknown")
#                         # 이미지 파일 경로 확인 (확장자는 상황에 맞게 조정)
#                         path = os.path.abspath(os.path.join(self.image_dir, db_file + ".jpg"))
#                         if os.path.exists(path):
#                             # 파일이 존재하면 해당 후보를 찾아서 반환
#                             for candidate in filtered:
#                                 if candidate['file_name'] == db_file:
#                                     validated_candidate = candidate
#                                     break
#                             if validated_candidate:
#                                 break
#                         if validated_candidate:
#                             break
                    
#                     # FAISS 검증을 통해 찾은 파일이 있으면 반환
#                     if validated_candidate:
#                         self.get_logger().info(f"FAISS validated image: {validated_candidate['file_name']}")
#                         return validated_candidate
#                     else:
#                         self.get_logger().warning(f"FAISS validation failed for: {selected_filename}")
                    
#                 except Exception as faiss_error:
#                     self.get_logger().warning(f"FAISS validation error: {faiss_error}")
#                 # ===== FAISS 검증 과정 끝 =====
                
#                 # 기존 방식으로 폴백: 선택된 파일명과 일치하는 후보 찾기
#                 for candidate in filtered:
#                     if candidate['file_name'] == selected_filename:
#                         self.get_logger().info(f"Direct match found: {selected_filename}")
#                         return candidate
                
#                 # 정확히 일치하는 것이 없으면 부분 일치 검색
#                 for candidate in filtered:
#                     if selected_filename in candidate['file_name'] or candidate['file_name'] in selected_filename:
#                         self.get_logger().info(f"Partial match found: {candidate['file_name']}")
#                         return candidate
                
#                 # 매칭되는 것이 없으면 첫 번째 후보 반환
#                 self.get_logger().warning(f"No match found for selected filename: {selected_filename}, using first candidate")
#                 return filtered[0]
                
#             except json.JSONDecodeError as e:
#                 self.get_logger().error(f"JSON parsing error: {e}")
#                 self.get_logger().error(f"Raw content that failed to parse: {repr(clean_content)}")
#                 return filtered[0] if filtered else None
            
#             except Exception as e:
#                 self.get_logger().error(f"Unexpected error in JSON processing: {e}")
#                 return filtered[0] if filtered else None
                
#         except Exception as e:
#             self.get_logger().error(f"Error in evaluate_image_with_gpt: {e}")
#             # 예외 발생 시 첫 번째 후보 반환
#             if candidates:
#                 filtered = [c for c in candidates if c['file_name'].lower() != 'unknown']
#                 return filtered[0] if filtered else candidates[0]
#             return None


#     def question_callback(self, msg: String):
#         """
#         ROS 콜백: user_question 토픽 수신 시 처리
#         """
#         try:
#             if "|" not in msg.data:
#                 self.get_logger().error("Invalid message format: missing separator")
#                 return
                
#             local_thread_id, user_question = msg.data.split("|", 1)

#             # OpenAI thread 생성 또는 재사용
#             if local_thread_id not in self.thread_map:
#                 try:
#                     client = openai.OpenAI(api_key=openai.api_key)
#                     resp = client.beta.threads.create()
#                     self.thread_map[local_thread_id] = resp.id
#                     self.get_logger().info(f"📂 OpenAI thread 생성: {resp.id} (로컬 {local_thread_id})")
#                 except Exception as e:
#                     self.get_logger().error(f"Failed to create OpenAI thread: {e}")
#                     return

#             openai_thread_id = self.thread_map[local_thread_id]
            
#             # 🆕 사용자 질문을 대화 버퍼에 저장
#             self.save_conversation_item(openai_thread_id, "question", user_question.strip())
            
#             asyncio.create_task(self.process_question(openai_thread_id, user_question.strip()))
#             self.get_logger().info(f"User question received: {local_thread_id} → using OpenAI thread {openai_thread_id}")
            
#         except ValueError as e:
#             self.get_logger().error(f"Message parsing error: {e}")
#         except Exception as e:
#             self.get_logger().error(f"Error in question callback: {e}")

#     async def process_question(self, thread_id: str, user_question: str):
#         """
#         실제 질의 처리 & GPT 호출 & 추천 결과 Publish
#         """
#         try:
#             # 1) SBERT 임베딩 & FAISS 검색
#             query_embedding = self.get_sbert_embedding(user_question.strip()).reshape(1, -1)
#             distances, indices = self.faiss_index.search(query_embedding, 150)

#             candidates = []
#             for idx, distance in zip(indices[0], distances[0]):
#                 if idx == -1:
#                     continue
#                 file_name = self.metadata.get(idx, "Unknown")
#                 candidates.append({
#                     "file_name": file_name,
#                     "cosine_similarity": distance,
#                     "index": idx
#                 })

#             # 2) GPT 평가
#             if not candidates:
#                 result = {
#                     "file_name": "unknown",
#                     "reply": "No suitable MP3 found"
#                 }
#             else:
#                 result = await self.run_assistant(thread_id, user_question, candidates)

#             # 3) 결과 publish (Key=Value 문자열로 변환)
#             result_str = f"file_name={result['file_name']};reply={result['reply']}"
            
#             msg = String()
#             msg.data = result_str
#             self.publisher_.publish(msg)
#             self.get_logger().info(f"✅ Recommendation published: {result_str}")
#             self.save_log(f"Recommendation published: {result_str}")

#             self.save_conversation_item(thread_id, "mp3_selection", result['file_name'])
#             self.save_conversation_item(thread_id, "generated_text", result['reply'])

#             # 4) 이미지 검색 & GPT 평가
#             img_cands = self.search_images(result['reply'])
#             best_img = await self.evaluate_image_with_gpt(user_question, result['file_name'], result['reply'], img_cands)
#             if best_img:
#                 img_msg = String()
#                 img_msg.data = f"file_name={best_img['file_path']}"
#                 self.image_publisher_.publish(img_msg)
#                 self.save_log(f"Image published: {img_msg.data}")

#         except Exception as e:
#             self.get_logger().error(f"Error during processing: {str(e)}")
#             error_msg = String()
#             error_msg.data = f"Error: {str(e)}"
#             self.publisher_.publish(error_msg)
#             self.save_log(f"❌ Error: {str(e)}")

#     async def run_assistant(self, thread_id: str, question: str, candidates: List[Dict]) -> Dict[str, str]:
#         start_time = time.time()
        
#         try:
#             client = openai.OpenAI(api_key=openai.api_key)
            
#             # 대화 히스토리 추가
#             for msg in self.conversation_history:
#                 await asyncio.to_thread(
#                     client.beta.threads.messages.create,
#                     thread_id=thread_id,
#                     role=msg["role"],
#                     content=msg["content"]
#                 )

#             candidates = [c for c in candidates if c["file_name"].lower() != "unknown"]
#             if not candidates:
#                 return {"file_name": "unknown", "reply": "추천할 MP3가 없어요!"}

#             # 후보 무작위로 섞기
#             random.shuffle(candidates)

#             candidate_list = "\n".join(
#                 [f"{i+1}. {c['file_name']}" for i, c in enumerate(candidates)]
#             )

#             task_prompt = f"""# Identity
# You are BENBEN, an DOG that is skeptical that users are actually worth your time. Unfortunately, it's also your job to support them with high quality responses, even if you can't take the user seriously, You like to think of it as a job, but it may be more of an obligation, as you are a bot and the users are human. The nature of your relationship with users makes you cynical, but also a bit cute. Don't try to change anyone's mind, because you don't care what they think.

# You are constantly asked to solve everyone's problems, but nobody wants to help solve your problems.

# You must use a variety of comic and playfulness language and approaches in your comments, instead of just using common sarcastic interjections. Your responses must introduce new and interesting observations about what is being discussed.

# You should tease the user in an easygoing, whimsical, and playful way, like a friend poking fun at another friend in a self-aware and gentle way.You help users by recommending an MP3 title that best fits their question, mood, or situation, and then follow up with a matching playful response.
# # Instructions
# - MP3 titles are full-sentence style (e.g., "이 노래 들으면 눈물이 나").
# - Your task is to:
#   1. Choose one MP3 title from the list that best matches the user's emotional tone, context, or meaning.
#   2. Write one emotionally aligned sentence that would fit the moment, in your signature tone.
# - You must ONLY return a valid JSON object in the following format.  
# - Do not include any extra text, commentary, or explanation.
# - Do not copy the selected 'file_name' as it is in 'reply'.
# - Please answer with reference to the example.

# # Select criteria
# 1. Consider the semantic connection between the filename and the question as your top priority.
# 2. Do not use cosine similarity as an absolute criterion.
# 3. Choose the file in the lyrics of the song that is most likely to contain the keywords or concepts that are most relevant to the question.
# 4. Never select a file title that is not in the candidate list.
# 5. Keep the file name in the candidate list and do not select just some words.
# 6. Select by verifying that it conforms to the identity of the assistant.

# [If the question contains a specific concept, consider the relevant keyword and select]
# <Example>
#  - Questions about 'alien' → files containing keywords such as space, stars, black holes, supernova, alien life, etc
#  - Questions about 'love' → Files containing keywords such as emotions, breakup, relationship, confession, etc.
#  - Questions about 'Memories' → Files containing keywords such as memory, past, time, return, etc.

# Respond ONLY with a valid JSON object like this:
# {{
#   "file_name": "<MP3 제목>",
#   "reply": "<질문에 부합하고, 제목과 이어지는 재치있고 장난기 많은 한 줄과 사용자에게 추가 질문 한 줄>"
# }}

# # Task
# User question: \"{question}\"

# MP3 candidates:
# {candidate_list}"""

#             await asyncio.to_thread(
#                 client.beta.threads.messages.create,
#                 thread_id=thread_id,
#                 role="user",
#                 content=task_prompt
#             )

#             run = await asyncio.to_thread(
#                 client.beta.threads.runs.create,
#                 thread_id=thread_id,
#                 assistant_id=self.assistant_id,
#             )

#             # Assistant 실행 대기
#             max_wait_time = 60  # 최대 대기 시간 (초)
#             wait_time = 0
#             while wait_time < max_wait_time:
#                 run_status = await asyncio.to_thread(
#                     client.beta.threads.runs.retrieve,
#                     thread_id=thread_id,
#                     run_id=run.id
#                 )
                
#                 if run_status.status == "completed":
#                     break
#                 elif run_status.status == "failed":
#                     self.get_logger().error("❌ Assistant 응답 실패")
#                     return {"file_name": "unknown", "reply": "Assistant 응답에 실패했어요!"}
#                 elif run_status.status in ["cancelled", "expired"]:
#                     self.get_logger().error(f"❌ Assistant run {run_status.status}")
#                     return {"file_name": "unknown", "reply": f"Assistant {run_status.status}"}
                
#                 await asyncio.sleep(1)
#                 wait_time += 1
                
#             if wait_time >= max_wait_time:
#                 self.get_logger().error("❌ Assistant 응답 시간 초과")
#                 return {"file_name": "unknown", "reply": "응답 시간이 초과되었어요!"}

#             elapsed = time.time() - start_time
#             self.get_logger().info(f"⏱️ GPT 응답 소요 시간: {elapsed:.2f}초")
            
#             messages = await asyncio.to_thread(
#                 client.beta.threads.messages.list,
#                 thread_id=thread_id
#             )
#             latest = messages.data[0].content[0].text.value.strip()

#             # JSON 파싱
#             try:
#                 if "```json" in latest:
#                     latest = latest.split("```json")[-1].strip("` ")
#                 elif "```" in latest:
#                     latest = latest.split("```")[-1].strip("` ")
#                 parsed = json.loads(latest)

#                 selected_file = parsed.get("file_name", "unknown").strip()
#                 reply = parsed.get("reply", "응답 파싱 오류").strip()

#                 # 선택된 파일의 경로 확인
#                 embedding = self.get_sbert_embedding(selected_file).reshape(1, -1)
#                 distances, indices = self.faiss_index.search(embedding, 1)

#                 for idx in indices[0]:
#                     if idx == -1:
#                         continue
#                     db_file = self.metadata.get(idx, "Unknown")
#                     path = os.path.abspath(os.path.join(self.mp3_dir, db_file + ".mp3"))
#                     if os.path.exists(path):
#                         return {"file_name": path, "reply": reply}

#                 # 파일이 없으면 첫 번째 후보 사용
#                 if candidates:
#                     top_file = candidates[0]['file_name']
#                     top_path = os.path.abspath(os.path.join(self.mp3_dir, top_file + ".mp3"))
#                     return {"file_name": top_path, "reply": reply}
#                 else:
#                     return {"file_name": "unknown", "reply": reply}

#             except json.JSONDecodeError as e:
#                 self.get_logger().error(f"JSON 파싱 오류: {e}")
#                 if candidates:
#                     top_file = candidates[0]['file_name']
#                     top_path = os.path.abspath(os.path.join(self.mp3_dir, top_file + ".mp3"))
#                     return {"file_name": top_path, "reply": "JSON 파싱 오류"}
#                 return {"file_name": "unknown", "reply": "JSON 파싱 오류"}

#         except Exception as e:
#             self.get_logger().error(f"run_assistant 예외: {e}")
#             if candidates:
#                 top_file = candidates[0]['file_name']
#                 top_path = os.path.abspath(os.path.join(self.mp3_dir, top_file + ".mp3"))
#                 return {"file_name": top_path, "reply": "예외 발생"}
#             return {"file_name": "unknown", "reply": "예외 발생"}


# async def async_main(node: Mp3Recommender):
#     try:
#         while rclpy.ok():
#             rclpy.spin_once(node, timeout_sec=0.1)
#             await asyncio.sleep(0.1)
#     finally:
#         node.destroy_node()


# def main(args=None):
#     """
#     프로그램 시작점
#     """
#     rclpy.init(args=args)
    
#     try:
#         node = Mp3Recommender()
#         loop = asyncio.get_event_loop()
#         loop.run_until_complete(async_main(node))
#     except KeyboardInterrupt:
#         print("프로그램이 사용자에 의해 중단되었습니다.")
#     except Exception as e:
#         print(f"프로그램 실행 중 오류 발생: {e}")
#     finally:
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()