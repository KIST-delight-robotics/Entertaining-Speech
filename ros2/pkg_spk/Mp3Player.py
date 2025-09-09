
import os
import requests
import threading
from datetime import datetime
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pydub import AudioSegment
from pydub.playback import play
import asyncio
import numpy as np
import json
import time
import websockets
import random  

import wave
import pyaudio
import csv
from google.cloud import speech
import io
import os
import re

import torch
import whisper_timestamped as whisper

from difflib import SequenceMatcher

from kiwipiepy import Kiwi


class Mp3Player(Node):
    def __init__(self):
        super().__init__("Mp3Player")

        # 파일 경로
        self.file_path = "/home/nvidia/ros2_ws/src/pkg_rag/pkg_rag/mp3_database_plus"
        self.reply_path = "/home/nvidia/ros2_ws/src/pkg_spk/pkg_spk/reply.mp3"
        self.api_key = "sk_fdb1ba8706bb125cb308ae613f58105e23e26a89d127a4cd"


        # # 구독: 추천된 MP3
        # self.subscription_ = self.create_subscription(
        #     String,
        #     "recommended_mp3",
        #     self.mp3_callback,
        #     10
        # )


        # 퍼블리시: 음악 재생 상태
        self.publisher_ = self.create_publisher(String, "music_status", 10)
        self.amplitude_publisher_ = self.create_publisher(String, "audio_amplitude", 10)
        self.is_playing = False  # 재생 중 여부 플래그



        self.current_image_path = None
        self.image_subscription_ = self.create_subscription(
            String, "recommended_image", self.image_callback, 10
        )
        self.image_publisher_ = self.create_publisher(String, "current_music_image", 10)
        # 기존 publisher들 다음에 추가
        self.mp3_waiting_spectrum_pub = self.create_publisher(String, "/mp3_waiting_spectrum", 10)


        # 🆕 TTS 요청 구독 추가
        self.tts_subscription = self.create_subscription(
            String,
            "tts_request",
            self.tts_request_callback,
            10
        )
        
        # 🆕 TTS 완료 상태 퍼블리시 추가
        self.tts_status_publisher = self.create_publisher(String, "tts_status", 10)

        # TTS 재생 요청 구독
        self.tts_play_subscription = self.create_subscription(
            String,
            "tts_play_request",
            self.tts_play_request_callback,
            10
        )


        # 🆕 TTS 전용 스펙트럼 퍼블리셔 추가
        self.tts_spectrum_publisher = self.create_publisher(String, "/tts_spectrum", 10)



        # 🆕 TTS 자막 데이터 퍼블리셔 추가
        self.tts_subtitle_publisher = self.create_publisher(String, "/tts_subtitle", 10)
        
        # Google Cloud 인증 설정
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/nvidia/ros2_ws/my-service-account.json'

        # 🆕 STT 재시작 신호 퍼블리셔 추가 (music_done 대신)
        self.stt_restart_publisher = self.create_publisher(String, "stt_restart", 10)


         # 🆕 whisper 관련 초기화 추가
        self.whisper_model = None
        self.whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loading_lock = threading.Lock()  # 🆕 스레드 안전성
        
        # 🆕 whisper 환경 설정
        self.setup_whisper_environment()


        # 🆕 모델 사전 로딩 (비동기)
        self.preload_whisper_model()



        # 🆕 Kiwi 형태소 분석기 초기화
        self.kiwi = None
        self.mecab = None  # 호환성을 위해 유지




        # 🆕 형태소 분석기 초기화
        try:
            self.kiwi = Kiwi()
            self.get_logger().info("✅ 한국어 형태소 분석기 초기화 완료")
        except Exception as e:
            self.get_logger().error(f"❌ 형태소 분석기 초기화 실패: {e}")
            self.kiwi = None


        # 🆕 단일 단어 자막 퍼블리셔 추가
        self.single_word_publisher = self.create_publisher(String, "/single_word_subtitle", 10)
        self.pending_subtitle_data = None  # 🆕 자막 데이터 저장용


        





    def publish_single_word_subtitle(self, subtitle_data):
        """
        자막을 한 단어씩 순차적으로 퍼블리시 (다음 단어까지 현재 단어 유지)
        """
        if not subtitle_data or 'words' not in subtitle_data:
            return
        
        def subtitle_worker():
            try:
                words = subtitle_data['words']
                start_time = time.time()
                
                self.get_logger().info(f"📺 순차 단일 자막 시작: {len(words)}개 단어")
                
                for i, word_info in enumerate(words):
                    # 단어 시작 시간까지 대기
                    elapsed_time = time.time() - start_time
                    target_time = word_info['start']
                    
                    if target_time > elapsed_time:
                        sleep_duration = target_time - elapsed_time
                        time.sleep(sleep_duration)
                    
                    # 🔑 현재 단어 표시
                    current_word_data = {
                        "word": word_info['word'],
                        "start": word_info['start'],
                        "end": word_info['end'],
                        "confidence": word_info['confidence'],
                        "index": i,
                        "total": len(words),
                        "display_mode": "single_word"
                    }
                    
                    msg = String()
                    msg.data = json.dumps(current_word_data, ensure_ascii=False)
                    self.single_word_publisher.publish(msg)
                    
                    self.get_logger().info(f"📺 [{i+1}/{len(words)}] 표시: '{word_info['word']}' ({word_info['start']:.2f}s-{word_info['end']:.2f}s)")
                    
                    # 🔑 핵심 수정: 다음 단어 시작 시간까지 대기 (빈 화면 없이)
                    if i < len(words) - 1:
                        # 다음 단어가 있는 경우: 다음 단어 시작 시간까지 현재 단어 유지
                        next_word_start = words[i + 1]['start']
                        current_time_in_audio = time.time() - start_time
                        remaining_time = next_word_start - current_time_in_audio
                        
                        if remaining_time > 0:
                            time.sleep(remaining_time)
                    else:
                        # 마지막 단어인 경우: 단어의 지속 시간만큼 대기
                        word_duration = word_info['end'] - word_info['start']
                        time.sleep(word_duration)
                
                # # 🔑 모든 자막 종료
                # final_data = {
                #     "word": "",
                #     "display_mode": "finished"
                # }
                # final_msg = String()
                # final_msg.data = json.dumps(final_data, ensure_ascii=False)
                # self.single_word_publisher.publish(final_msg)
                
                # self.get_logger().info("📺 순차 단일 자막 완료")
                
            except Exception as e:
                self.get_logger().error(f"❌ 순차 자막 퍼블리시 실패: {e}")
        
        # 별도 스레드에서 실행
        subtitle_thread = threading.Thread(target=subtitle_worker, daemon=True)
        subtitle_thread.start()
















# ──────────────────────────────────────────────────────────────────────────────
# 형태소분석기
# ──────────────────────────────────────────────────────────────────────────────
    # # 🆕 명사 추출 함수
    # def extract_nouns_from_text(self, text):
    #     """
    #     Kiwi를 사용하여 텍스트에서 명사만 추출 (빠르고 정확)
    #     """
    #     if not self.kiwi:
    #         self.get_logger().warning("⚠️ Kiwi 형태소 분석기가 초기화되지 않음")
    #         return self._simple_noun_extraction(text)
        
    #     try:
    #         # Kiwi로 형태소 분석 수행
    #         result = self.kiwi.analyze(text)
            
    #         # 첫 번째 분석 결과 사용 (가장 확률이 높은 결과)
    #         if not result or not result[0] or not result[0][0]:
    #             self.get_logger().warning("⚠️ Kiwi 분석 결과가 비어있음")
    #             return self._simple_noun_extraction(text)
            
    #         tokens = result[0][0]  # 첫 번째 분석 결과의 토큰들
            
    #         # 명사만 필터링
    #         nouns = []
    #         for token in tokens:
    #             # Kiwi 태그에서 명사 확인 (NNG: 일반명사, NNP: 고유명사, NNB: 의존명사)
    #             if token.tag in ['NNG', 'NNP', 'NNB'] or token.tag.startswith('NN'):
    #                 if self._is_meaningful_noun(token.form):
    #                     nouns.append(token.form)
            
    #         # 중복 제거하되 순서 유지
    #         unique_nouns = list(dict.fromkeys(nouns))
            
    #         self.get_logger().info(f"🔤 Kiwi 추출된 명사: {unique_nouns}")
    #         return unique_nouns
            
    #     except Exception as e:
    #         self.get_logger().error(f"❌ Kiwi 명사 추출 실패: {e}")
    #         return self._simple_noun_extraction(text)




#로그추가


    def extract_nouns_from_text(self, text):
        """
        Kiwi를 사용하여 텍스트에서 명사만 추출 (빠르고 정확) - 시간 측정 포함
        """
        # 🆕 형태소 분석 시간 측정 시작
        morphology_start_time = time.time()
        self.get_logger().info(f"⏳ 형태소 분석 시작: '{text}'")
        
        if not self.kiwi:
            self.get_logger().warning("⚠️ Kiwi 형태소 분석기가 초기화되지 않음")
            fallback_result = self._simple_noun_extraction(text)
            
            # 폴백도 시간 측정
            morphology_end_time = time.time()
            morphology_duration = morphology_end_time - morphology_start_time
            self.get_logger().info(f"⌛ 폴백 형태소 분석 소요시간: {morphology_duration:.3f}초")
            self.get_logger().info(f"🔤 폴백 추출된 명사: {fallback_result}")
            return fallback_result
        
        try:
            # 🆕 Kiwi 분석 단계별 시간 측정
            kiwi_analysis_start = time.time()
            
            # Kiwi로 형태소 분석 수행
            result = self.kiwi.analyze(text)
            
            kiwi_analysis_end = time.time()
            kiwi_analysis_duration = kiwi_analysis_end - kiwi_analysis_start
            
            # 첫 번째 분석 결과 사용 (가장 확률이 높은 결과)
            if not result or not result[0] or not result[0][0]:
                self.get_logger().warning("⚠️ Kiwi 분석 결과가 비어있음")
                fallback_result = self._simple_noun_extraction(text)
                
                morphology_end_time = time.time()
                morphology_duration = morphology_end_time - morphology_start_time
                self.get_logger().info(f"⌛ 형태소 분석 총 소요시간: {morphology_duration:.3f}초 (Kiwi 분석: {kiwi_analysis_duration:.3f}초)")
                self.get_logger().info(f"🔤 폴백 추출된 명사: {fallback_result}")
                return fallback_result
            
            # 🆕 토큰 처리 시간 측정
            token_processing_start = time.time()
            
            tokens = result[0][0]  # 첫 번째 분석 결과의 토큰들
            
            # 🆕 상세한 토큰 분석 로그
            self.get_logger().info(f"📊 Kiwi 분석된 총 토큰 수: {len(tokens)}개")
            
            # 명사만 필터링
            raw_nouns = []  # 필터링 전 모든 명사
            filtered_nouns = []  # 필터링 후 의미있는 명사
            
            for token in tokens:
                # Kiwi 태그에서 명사 확인 (NNG: 일반명사, NNP: 고유명사, NNB: 의존명사)
                if token.tag in ['NNG', 'NNP', 'NNB','NR','NP','W_SERIAL','SN','SL'] or token.tag.startswith('NN'):
                    raw_nouns.append(f"{token.form}({token.tag})")  # 태그 정보 포함
                    
                    if self._is_meaningful_noun(token.form):
                        filtered_nouns.append(token.form)
            
            token_processing_end = time.time()
            token_processing_duration = token_processing_end - token_processing_start
            
            # 중복 제거하되 순서 유지
            unique_nouns = list(dict.fromkeys(filtered_nouns))
            
            # 🆕 전체 시간 계산
            morphology_end_time = time.time()
            morphology_total_duration = morphology_end_time - morphology_start_time
            
            # 🆕 상세 분석 로그
            self.get_logger().info("="*50)
            self.get_logger().info("📊 형태소 분석 상세 결과")
            self.get_logger().info(f"⌛ 총 소요시간: {morphology_total_duration:.3f}초")
            self.get_logger().info(f"  - Kiwi 분석: {kiwi_analysis_duration:.3f}초 ({kiwi_analysis_duration/morphology_total_duration*100:.1f}%)")
            self.get_logger().info(f"  - 토큰 처리: {token_processing_duration:.3f}초 ({token_processing_duration/morphology_total_duration*100:.1f}%)")
            self.get_logger().info(f"📝 원시 명사 ({len(raw_nouns)}개): {raw_nouns}")
            self.get_logger().info(f"✅ 필터링된 명사 ({len(unique_nouns)}개): {unique_nouns}")
            self.get_logger().info(f"🎯 최종 선택 명사: {unique_nouns}")
            self.get_logger().info("="*50)
            
            return unique_nouns
            
        except Exception as e:
            morphology_end_time = time.time()
            morphology_duration = morphology_end_time - morphology_start_time
            
            self.get_logger().error(f"❌ Kiwi 명사 추출 실패 (소요시간: {morphology_duration:.3f}초): {e}")
            fallback_result = self._simple_noun_extraction(text)
            self.get_logger().info(f"🔤 폴백 추출된 명사: {fallback_result}")
            return fallback_result




    def _simple_noun_extraction(self, text):
        """
        Kiwi 실패시 간단한 규칙 기반 명사 추출 (폴백 함수) - 시간 측정 포함
        """
        import re
        
        fallback_start = time.time()
        self.get_logger().warning("⚠️ Kiwi 실패 - 간단한 규칙 기반 명사 추출 사용")
        
        # 한글 2글자 이상 단어 추출
        words = re.findall(r'[가-힣]{2,}', text)
        
        # 의미있는 명사만 필터링
        nouns = []
        for word in words:
            if self._is_meaningful_noun(word):
                nouns.append(word)
        
        # 중복 제거 후 최대 8개
        result = list(dict.fromkeys(nouns))[:8]
        
        fallback_end = time.time()
        fallback_duration = fallback_end - fallback_start
        
        self.get_logger().info(f"⌛ 폴백 명사 추출 소요시간: {fallback_duration:.3f}초")
        self.get_logger().info(f"📝 규칙 기반 원시 명사: {words}")
        self.get_logger().info(f"🔤 필터링 후 명사: {result}")
        
        return result







    def _is_meaningful_noun(self, noun):
        """
        의미있는 명사인지 판별 (Kiwi용 개선 버전)
        """
        # 길이 확인
        if len(noun) < 2:
            return False
        
        # 개선된 불용어 리스트
        stopwords = {
            '것', '데', '거', '게', '걸', '곳', '때', '말', '분', '점', 
            '번', '개', '명', '사람', '이것', '그것', '저것', '여기',
            '거기', '저기', '이곳', '그곳', '저곳', '어디', '언제',
            '누구', '무엇', '어떤', '이런', '그런', '저런', '같은',
            '다른', '새로운', '오늘', '어제', '내일', '지금', '나중'
        }
        
        if noun in stopwords:
            return False
        
        # 숫자만 있는 경우 제외
        if noun.isdigit():
            return False
        
        # 특수문자만 있는 경우 제외
        if not any(char.isalnum() for char in noun):
            return False
        
        # 한글이 포함되어야 함 (한국어 명사)
        if not any('가' <= char <= '힣' for char in noun):
            return False
        
        # 너무 일반적인 단어 제외
        common_words = {'위치', '상태', '방법', '시간', '장소', '이유'}
        if noun in common_words:
            return False
        
        return True






# 🔑 핵심: 명사와 STT 타임스탬프 매핑
    def map_nouns_to_timestamps(self, nouns, stt_timestamps, original_text):
        """
        추출된 명사들을 STT 타임스탬프와 매핑하여 명사별 타임스탬프 생성
        """
        try:
            if not nouns or not stt_timestamps:
                return []
            
            noun_timestamps = []
            
            # 각 명사에 대해 STT에서 가장 적절한 타임스탬프 찾기
            for noun in nouns:
                best_match = self._find_best_timestamp_for_noun(noun, stt_timestamps, original_text)
                if best_match:
                    noun_timestamps.append({
                        'word': noun,
                        'start': best_match['start'],
                        'end': best_match['end'],
                        'confidence': best_match['confidence'],
                        'original_stt_word': best_match['original_word']
                    })
            
            # 🔑 시간순 정렬
            noun_timestamps.sort(key=lambda x: x['start'])
            
            # 🔑 핵심: 시간 분할 및 조정
            adjusted_timestamps = self._adjust_noun_timestamps(noun_timestamps)
            
            self.get_logger().info(f"🎯 명사 타임스탬프 매핑 완료: {len(adjusted_timestamps)}개")
            
            return adjusted_timestamps
            
        except Exception as e:
            self.get_logger().error(f"❌ 명사-타임스탬프 매핑 실패: {e}")
            return []

    def _find_best_timestamp_for_noun(self, noun, stt_timestamps, original_text):
        """
        명사에 가장 적합한 STT 타임스탬프 찾기
        """
        best_match = None
        best_score = 0
        
        for stt_word in stt_timestamps:
            # 1. 완전 일치 확인
            if noun == stt_word['word'].strip():
                return {
                    'start': stt_word['start'],
                    'end': stt_word['end'],
                    'confidence': stt_word['confidence'],
                    'original_word': stt_word['word']
                }
            
            # 2. 포함 관계 확인
            similarity_score = 0
            if noun in stt_word['word'] or stt_word['word'] in noun:
                similarity_score = 0.8
            else:
                # 3. 편집 거리 기반 유사도
                similarity_score = SequenceMatcher(None, noun, stt_word['word']).ratio()
            
            # 4. 원본 텍스트에서의 근접성도 고려
            text_proximity = self._calculate_text_proximity(noun, stt_word['word'], original_text)
            final_score = similarity_score * 0.7 + text_proximity * 0.3
            
            if final_score > best_score and final_score > 0.5:  # 임계값
                best_score = final_score
                best_match = {
                    'start': stt_word['start'],
                    'end': stt_word['end'],
                    'confidence': stt_word['confidence'] * final_score,
                    'original_word': stt_word['word']
                }
        
        return best_match

    def _calculate_text_proximity(self, noun, stt_word, original_text):
        """
        원본 텍스트에서 두 단어의 근접성 계산
        """
        try:
            noun_pos = original_text.find(noun)
            stt_pos = original_text.find(stt_word)
            
            if noun_pos == -1 or stt_pos == -1:
                return 0
            
            distance = abs(noun_pos - stt_pos)
            max_distance = len(original_text)
            
            return max(0, 1 - (distance / max_distance))
        except:
            return 0

    # 🔑 핵심: 시간 분할 및 조정 로직
    def _adjust_noun_timestamps(self, noun_timestamps):
        """
        명사 타임스탬프 조정 및 시간 분할
        """
        if not noun_timestamps:
            return []
        
        adjusted = []
        i = 0
        
        while i < len(noun_timestamps):
            current_noun = noun_timestamps[i]
            
            # 같은 시간대에 있는 명사들 그룹핑
            same_time_group = [current_noun]
            j = i + 1
            
            while j < len(noun_timestamps):
                next_noun = noun_timestamps[j]
                # 시간 겹침 확인 (오차 허용)
                if self._timestamps_overlap(current_noun, next_noun):
                    same_time_group.append(next_noun)
                    j += 1
                else:
                    break
            
            # 🔑 시간 분할 수행
            if len(same_time_group) > 1:
                self.get_logger().info(f"⏰ 시간 분할 필요: {len(same_time_group)}개 명사")
                divided_timestamps = self._divide_timestamp_for_multiple_nouns(same_time_group)
                adjusted.extend(divided_timestamps)
            else:
                # 단일 명사는 그대로 추가 (최소 지속시간 보장)
                single_noun = same_time_group[0]
                duration = single_noun['end'] - single_noun['start']
                if duration < 1.0:  # 최소 1초 보장
                    single_noun['end'] = single_noun['start'] + 1.0
                adjusted.append(single_noun)
            
            i = j
        
        # 최종 겹침 방지 및 간격 조정
        return self._prevent_timestamp_overlaps(adjusted)

    def _timestamps_overlap(self, ts1, ts2, tolerance=0.5):
        """
        두 타임스탬프가 겹치는지 확인 (허용 오차 포함)
        """
        return not (ts1['end'] + tolerance < ts2['start'] or ts2['end'] + tolerance < ts1['start'])

    def _divide_timestamp_for_multiple_nouns(self, noun_group):
        """
        🔑 핵심: 하나의 타임스탬프를 여러 명사로 균등 분할
        """
        if len(noun_group) <= 1:
            return noun_group
        
        # 전체 시간 범위 계산
        start_time = min(noun['start'] for noun in noun_group)
        end_time = max(noun['end'] for noun in noun_group)
        total_duration = end_time - start_time
        
        # 명사별 최소 지속시간 보장
        min_duration_per_noun = 0.8  # 각 명사당 최소 0.8초
        required_total_duration = len(noun_group) * min_duration_per_noun
        
        if total_duration < required_total_duration:
            total_duration = required_total_duration
            end_time = start_time + total_duration
        
        # 균등 분할
        duration_per_noun = total_duration / len(noun_group)
        divided_timestamps = []
        
        for i, noun in enumerate(noun_group):
            noun_start = start_time + (i * duration_per_noun)
            noun_end = noun_start + duration_per_noun
            
            divided_timestamps.append({
                'word': noun['word'],
                'start': round(noun_start, 3),
                'end': round(noun_end, 3),
                'confidence': noun['confidence'],
                'original_stt_word': noun['original_stt_word'],
                'divided': True  # 분할된 표시
            })
            
            self.get_logger().info(f"  📍 '{noun['word']}': {noun_start:.2f}s - {noun_end:.2f}s")
        
        return divided_timestamps

    def _prevent_timestamp_overlaps(self, timestamps):
        """
        타임스탬프 겹침 방지 및 간격 조정
        """
        if not timestamps:
            return timestamps
        
        # 시간순 정렬
        sorted_timestamps = sorted(timestamps, key=lambda x: x['start'])
        adjusted = []
        
        for i, current in enumerate(sorted_timestamps):
            if i == 0:
                adjusted.append(current)
                continue
            
            previous = adjusted[-1]
            
            # 겹침 방지
            if current['start'] < previous['end']:
                # 시간을 균등하게 조정
                mid_time = (previous['start'] + current['end']) / 2
                previous['end'] = round(mid_time - 0.1, 3)  # 100ms 간격
                current['start'] = round(mid_time + 0.1, 3)
            
            # 최소 지속시간 보장
            if (current['end'] - current['start']) < 0.5:
                current['end'] = current['start'] + 0.8
            
            adjusted.append(current)
        
        return adjusted






# ──────────────────────────────────────────────────────────────────────────────
# Whisper 모델 설정
# ──────────────────────────────────────────────────────────────────────────────


    def setup_whisper_environment(self):
        """whisper-timestamped GPU 환경 설정"""
        try:
            if torch.cuda.is_available():
                # cuDNN 비활성화 모드 설정
                torch.backends.cudnn.enabled = False
                os.environ['CUDNN_DISABLED'] = '1'
                os.environ['PYTORCH_DISABLE_CUDNN_CONV'] = '1'
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
                
                # GPU 메모리 정리
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                self.get_logger().info("✅ whisper GPU 환경 설정 완료")
            else:
                self.get_logger().info("⚠️ CUDA 사용 불가 - CPU 모드로 설정")
                self.whisper_device = "cpu"
                
        except Exception as e:
            self.get_logger().error(f"❌ whisper 환경 설정 실패: {e}")
            self.whisper_device = "cpu"
            


    def preload_whisper_model(self):
            """노드 시작 시 whisper 모델 사전 로딩"""
            
            def load_model_async():
                try:
                    preload_start = time.time()
                    self.get_logger().info("🚀 whisper 모델 사전 로딩 시작...")
                    
                    with self.model_loading_lock:
                        if self.whisper_model is None:
                            self.whisper_model = whisper.load_model("tiny", device=self.whisper_device)
                    
                    preload_end = time.time()
                    preload_time = preload_end - preload_start
                    
                    self.get_logger().info(f"✅ whisper 모델 사전 로딩 완료! (소요시간: {preload_time:.2f}초, device: {self.whisper_device})")
                    self.save_log(f"✅ whisper 모델 사전 로딩 완료: {preload_time:.2f}초")
                    
                    # 🆕 간단한 테스트 추론으로 모델 워밍업
                    self._warmup_model()
                    
                except Exception as e:
                    error_msg = f"❌ whisper 모델 사전 로딩 실패: {e}"
                    self.get_logger().error(error_msg)
                    self.save_log(error_msg)
                    
                    # GPU 실패 시 CPU 폴백
                    if self.whisper_device == "cuda":
                        self.get_logger().info("🔄 GPU 로딩 실패 - CPU 모드로 폴백 시도")
                        self.whisper_device = "cpu"
                        try:
                            with self.model_loading_lock:
                                self.whisper_model = whisper.load_model("tiny", device="cpu")
                            self.get_logger().info("✅ CPU 모드로 모델 로딩 성공")
                        except Exception as e2:
                            self.get_logger().error(f"❌ CPU 모드도 실패: {e2}")
                            self.whisper_model = None

            # 🆕 비동기로 모델 로딩 (노드 시작 차단 방지)
            loading_thread = threading.Thread(target=load_model_async, daemon=True)
            loading_thread.start()


    def _warmup_model(self):
            """모델 워밍업 (첫 추론 지연 방지)"""
            try:
                if self.whisper_model is not None:
                    self.get_logger().info("🔥 모델 워밍업 중...")
                    
                    # 더미 오디오로 워밍업
                    import numpy as np
                    dummy_audio = np.random.randn(8000).astype(np.float32)  # 0.5초 더미 오디오
                    
                    _ = whisper.transcribe(
                        self.whisper_model, 
                        dummy_audio, 
                        language="ko", 
                        verbose=False
                    )
                    
                    self.get_logger().info("✅ 모델 워밍업 완료")
                    
            except Exception as e:
                self.get_logger().warning(f"⚠️ 모델 워밍업 실패 (정상 동작에는 영향 없음): {e}")
            


# ──────────────────────────────────────────────────────────────────────────────
# 단어별 타임스탬프 추출 함수 추가
# ──────────────────────────────────────────────────────────────────────────────




#whisper 버전


    def extract_word_timestamps(self, audio_path, original_text):
        """
        whisper-timestamped를 사용하여 단어별 타임스탬프 추출 (사전 로딩된 모델 사용)
        """
        
        stt_start_time = time.time()
        self.get_logger().info("⏳ whisper-timestamped를 통한 동적자막 생성 시작")
        
        try:
            # 🆕 모델 준비 확인 (사전 로딩 완료 대기)
            model_ready_start = time.time()
            
            # 모델이 로딩 중이면 대기
            max_wait_time = 10.0  # 최대 10초 대기
            wait_interval = 0.1   # 100ms 간격으로 확인
            
            while self.whisper_model is None and (time.time() - model_ready_start) < max_wait_time:
                self.get_logger().info("⏳ whisper 모델 로딩 완료 대기 중...")
                time.sleep(wait_interval)
            
            # 모델이 여전히 없으면 즉시 로딩
            if self.whisper_model is None:
                self.get_logger().warning("⚠️ 사전 로딩 실패 - 즉시 로딩 시도")
                
                immediate_load_start = time.time()
                with self.model_loading_lock:
                    if self.whisper_model is None:
                        self.whisper_model = whisper.load_model("tiny", device=self.whisper_device)
                immediate_load_time = time.time() - immediate_load_start
                
                self.get_logger().info(f"🔄 즉시 로딩 완료: {immediate_load_time:.2f}초")
            else:
                self.get_logger().info("✅ 사전 로딩된 모델 사용")
            
            model_ready_time = time.time() - model_ready_start
            
            # 🆕 오디오 로딩 및 추론 (기존 로직)
            audio_start = time.time()
            self.get_logger().info("🗣️ whisper 추론 시작...")
            
            audio = whisper.load_audio(audio_path)
            
            result = whisper.transcribe(
                self.whisper_model, 
                audio, 
                language="ko",
                verbose=False,
                temperature=0
            )
            
            audio_end = time.time()
            inference_duration = audio_end - audio_start
            
            # 결과 처리 (기존과 동일)
            word_timestamps = []
            
            for segment in result.get("segments", []):
                segment_text = segment.get("text", "")
                self.get_logger().info(f"🗣️ whisper 인식 결과: '{segment_text}'")
                
                for word_info in segment.get("words", []):
                    word = word_info.get("text", "").strip()
                    start_time = word_info.get("start", 0.0)
                    end_time = word_info.get("end", 0.0)
                    confidence = word_info.get("confidence", 1.0)
                    
                    word_timestamps.append({
                        "word": word,
                        "start": round(start_time, 3),
                        "end": round(end_time, 3),
                        "confidence": round(confidence, 3)
                    })
            
            # 전체 시간 계산 및 로깅
            stt_end_time = time.time()
            stt_total_duration = stt_end_time - stt_start_time
            self._last_stt_duration = stt_total_duration
            
            self.get_logger().info(f"🎯 whisper 타임스탬프 추출 완료: {len(word_timestamps)}개 단어")
            self.get_logger().info(f"✅ 동적자막 생성 총 소요시간: {stt_total_duration:.2f}초")
            self.get_logger().info(f"   - 모델 준비 시간: {model_ready_time:.2f}초")  # 🆕 사전 로딩 시 거의 0초
            self.get_logger().info(f"   - 추론 시간: {inference_duration:.2f}초")
            self.get_logger().info(f"   - 후처리 시간: {stt_total_duration - model_ready_time - inference_duration:.2f}초")
            
            return word_timestamps
            
        except Exception as e:
            if self.whisper_device == "cuda" and "cuda" in str(e).lower():
                self.get_logger().warning(f"⚠️ GPU 모드 실패, CPU 폴백 시도: {e}")
                return self._extract_with_cpu_fallback(audio_path, original_text)
            
            stt_end_time = time.time()
            stt_total_duration = stt_end_time - stt_start_time
            
            self.get_logger().error(f"❌ whisper 타임스탬프 추출 실패: {e}")
            self.get_logger().error(f"❌ 실패까지 소요시간: {stt_total_duration:.2f}초")
            
            self._last_stt_duration = stt_total_duration
            return []

        







    def _extract_with_cpu_fallback(self, audio_path, original_text):
        """CPU 폴백 모드"""
        try:
            self.get_logger().info("🔄 CPU 폴백 모드로 재시도...")
            
            # CPU 모델 로딩
            cpu_model = whisper.load_model("tiny", device="cpu")
            audio = whisper.load_audio(audio_path)
            
            result = whisper.transcribe(
                cpu_model, audio, 
                language="ko", 
                verbose=False
            )
            
            # 결과 처리 (동일한 로직)
            word_timestamps = []
            for segment in result.get("segments", []):
                for word_info in segment.get("words", []):
                    word_timestamps.append({
                        "word": word_info.get("text", "").strip(),
                        "start": round(word_info.get("start", 0.0), 3),
                        "end": round(word_info.get("end", 0.0), 3),
                        "confidence": round(word_info.get("confidence", 0.8), 3)
                    })
            
            self.get_logger().info(f"✅ CPU 폴백 성공: {len(word_timestamps)}개 단어")
            return word_timestamps
            
        except Exception as e:
            self.get_logger().error(f"❌ CPU 폴백도 실패: {e}")
            return []


    def cleanup_whisper_model(self):
        """whisper 모델 메모리 정리"""
        if self.whisper_model is not None:
            del self.whisper_model
            self.whisper_model = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.get_logger().info("🧹 whisper 모델 메모리 정리 완료")

    def __del__(self):
        """소멸자에서 메모리 정리"""
        self.cleanup_whisper_model()




    

# ──────────────────────────────────────────────────────────────────────────────
# 대기효과음3 재생 및 스펙트럼 퍼블리시
# ──────────────────────────────────────────────────────────────────────────────



    def get_random_waiting_file(self):
        """
        waiting_3 디렉토리에서 랜덤한 MP3 파일 경로 반환
        """
        waiting_dir = "/home/nvidia/ros2_ws/src/pkg_mic/pkg_mic/_tts_waiting3"
        
        try:
            if not os.path.exists(waiting_dir):
                self.get_logger().warning(f"Waiting 디렉토리가 존재하지 않습니다: {waiting_dir}")
                return None
                
            # mp3 파일만 필터링
            mp3_files = [f for f in os.listdir(waiting_dir) if f.lower().endswith('.mp3')]
            
            if not mp3_files:
                self.get_logger().warning(f"Waiting 디렉토리에 MP3 파일이 없습니다: {waiting_dir}")
                return None
                
            # 랜덤 선택
            selected_file = random.choice(mp3_files)
            full_path = os.path.join(waiting_dir, selected_file)
            
            self.get_logger().info(f"🎲 랜덤 waiting 파일 선택: {selected_file}")
            self.save_log(f"🎲 랜덤 waiting 파일 선택: {selected_file}")
            
            return full_path
            
        except Exception as e:
            error_msg = f"❌ Waiting 파일 선택 중 오류: {e}"
            self.get_logger().error(error_msg)
            self.save_log(error_msg)
            return None




    def play_waiting_with_spectrum(self, file_path):
        """waiting 파일을 스펙트럼 시각화와 함께 재생 (UserQuestion.py와 동일한 방식)"""
        try:
            # pydub으로 MP3 로드 및 정규화 (UserQuestion.py와 동일)
            sound = AudioSegment.from_file(file_path, format="mp3")
            
            # # 정규화 (UserQuestion.py와 동일)
            # target_dBFS = -14.0
            # change_in_dBFS = target_dBFS - sound.dBFS
            # sound = sound.apply_gain(change_in_dBFS)
            
            # 임시 WAV로 변환
            temp_wav = "/tmp/mp3_waiting_audio.wav"
            sound.export(temp_wav, format="wav")

            self.get_logger().info(f"🎵 Mp3Player Waiting 파일 스펙트럼 재생: {file_path}")
            self.save_log(f"🎵 Mp3Player Waiting 파일 스펙트럼 재생: {file_path}")

            # UserQuestion.py와 동일한 방식으로 스펙트럼과 재생 병렬 처리
            self.mp3_waiting_publish_and_play(temp_wav)

            time.sleep(0.5)

            self.get_logger().info("🎵 Mp3Player Waiting 스펙트럼 재생 완료")
            self.save_log("🎵 Mp3Player Waiting 스펙트럼 재생 완료")

        except Exception as e:
            error_msg = f"❌ Mp3Player Waiting 스펙트럼 재생 실패: {e}"
            self.get_logger().error(error_msg)
            self.save_log(error_msg)




    def mp3_waiting_publish_and_play(self, wav_path):
        """Mp3Player.py 전용 waiting 스펙트럼 시각화 (UserQuestion.py 방식과 동일)"""

        
        wf = wave.open(wav_path, 'rb')
        chunk_size = 2024
        

        def publish_spectrum():
            data = wf.readframes(chunk_size)
            while data:

                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                if wf.getnchannels() == 2:
                    samples = samples.reshape((-1, 2)).mean(axis=1)
                fft = np.fft.fft(samples)
                spectrum = np.abs(fft[:len(fft)//2])
                #spectrum = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum
    




                msg = String()
                msg.data = json.dumps({"spectrum": spectrum.tolist()})
                self.mp3_waiting_spectrum_pub.publish(msg)  # 별도 토픽 사용
                data = wf.readframes(chunk_size)
                time.sleep(chunk_size / wf.getframerate())


        spectrum_thread = threading.Thread(target=publish_spectrum)
        spectrum_thread.start()

        # 시스템 명령어로 재생 (기존 방식과 동일)
        os.system(f"aplay {wav_path}")
        spectrum_thread.join()


        




 

# ──────────────────────────────────────────────────────────────────────────────
# 음악 재생 및 이미지 퍼블리시
# ──────────────────────────────────────────────────────────────────────────────


    def image_callback(self, msg):
        """이미지 파일명 수신 및 저장"""
        if msg.data and msg.data.strip():
            self.current_image_path = f"/images/{msg.data}"  # 웹 경로로 변환
            # 즉시 퍼블리시 (사전 로딩 효과)
            img_msg = String()
            img_msg.data = self.current_image_path
            self.image_publisher_.publish(img_msg)
            self.get_logger().info(f"이미지 수신: {self.current_image_path}")
        else:
            self.current_image_path = None
            self.get_logger().info("이미지 수신: 없음")




    def match_target_amplitude(self, sound, target_dBFS):
        """
        주어진 오디오를 타깃 dBFS로 정규화
        """
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)




    def play_mp3(self, file_path):
        try:
            sound = AudioSegment.from_file(file_path, format="mp3")
            sound = self.match_target_amplitude(sound, -14.0)
            
            # 임시 WAV로 변환 후 저장
            temp_wav = "/tmp/temp_audio.wav"
            sound.export(temp_wav, format="wav")

            # 스펙트럼과 재생 병렬로 실행
            playback_thread = threading.Thread(target=self.publish_and_play, args=(temp_wav,))
            playback_thread.start()
            playback_thread.join()

        except Exception as e:
            self.get_logger().error(f"❌ MP3 재생 실패: {file_path} → {e}")
            self.save_log(f"❌ MP3 재생 실패: {file_path} → {e}")



    def publish_and_play(self, wav_path):
        wf = wave.open(wav_path, 'rb')
        chunk_size = 2024


        # 2) CSV 파일 한 번 열어두기 (append 모드)
        csv_file = open('spectrum.csv', 'a', newline='')
        csv_writer = csv.writer(csv_file)



        def publish_spectrum():
            data = wf.readframes(chunk_size)
            while data:
                samples = np.frombuffer(data, dtype=np.int16)
                if wf.getnchannels() == 2:
                    samples = samples.reshape((-1, 2)).mean(axis=1)
                fft = np.fft.fft(samples)
                spectrum = np.abs(fft[:len(fft)//2])
                #spectrum = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum


                #────────────────────────────────────────────────
                # # 7) CSV에 한 열로 실시간 저장
                # #    - spectrum 값들
                # for mag in spectrum:
                #     csv_writer.writerow([mag])
                # #    - 이번 청크의 평균·최소·최대 (한 번 저장할 때 마지막에)
                # mean_val = spectrum.mean()
                # min_val  = spectrum.min()
                # max_val  = spectrum.max()
                # csv_writer.writerow([f"최소값: {min_val:.6f}"])
                # csv_writer.writerow([f"최대값: {max_val:.6f}"])
                # csv_writer.writerow([f"평균값: {mean_val:.6f}"])

                # #    - 파일에 바로 반영
                # csv_file.flush()

              
                msg = String()
                msg.data = json.dumps({"spectrum": spectrum.tolist()})
                self.amplitude_publisher_.publish(msg)
                data = wf.readframes(chunk_size)
                time.sleep(chunk_size / wf.getframerate())

        spectrum_thread = threading.Thread(target=publish_spectrum)
        spectrum_thread.start()

        # 시스템 명령어 aplay로 재생 (즉각적인 출력)
        os.system(f"aplay {wav_path}")  # 🔥실제 출력장치로 변경(카드2)
        spectrum_thread.join()
        csv_file.close()
        wf.close()


     

# ──────────────────────────────────────────────────────────────────────────────
# TTS 재생 및 스펙트럼 퍼블리시
# ──────────────────────────────────────────────────────────────────────────────



    # def text2speech(self, text):
    #     """
    #     ElevenLabs TTS 호출 → reply.mp3 저장 → 원본 텍스트 기반 자막 생성
    #     """

    #     # 🆕 전체 소요시간 측정 시작
    #     total_start_time = time.time()
    #     self.get_logger().info("⏳ TTS 전체 프로세스 시작")
        


    #     api_key = "sk_fdb1ba8706bb125cb308ae613f58105e23e26a89d127a4cd"
    #     # voice_id = "59zWnTQLbwyr94bFbcUe" #스폰지밥
    #     voice_id = "1W00IGEmNmwmsDeYy7ag" #스폰지밥
    #     url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    #     headers = {
    #         "xi-api-key": api_key,
    #         "Content-Type": "application/json; charset=utf-8",  # UTF-8 명시
    #         "Accept": "audio/mpeg"
    #     }

    #     # 🆕 텍스트 전처리
    #     cleaned_text = text.strip()
    #     cleaned_text = ' '.join(cleaned_text.split())
        
    #     # 🆕 디버깅 로그
    #     self.get_logger().info(f"🗣️ TTS 원본 텍스트: '{cleaned_text}'")

    #     # data = {
    #     #     "text": cleaned_text,
    #     #     "model_id": "eleven_multilingual_v2",
    #     #     "voice_settings": {
    #     #         "stability": 0.5,
    #     #         "similarity_boost": 0.75,
    #     #         "style": 0.25,
    #     #         "speed": 0.9
    #     #     },
    #     #     "apply_text_normalization": "on",
    #     #     "output_format": "mp3_22050_32"
    #     # }

    #     data = {
    #         "text": cleaned_text,
    #         # "model_id": "eleven_multilingual_v2",
    #         "model_id": "eleven_flash_v2_5",
    #         "voice_settings": {
    #             "stability": 1.0,
    #             "similarity_boost": 1.0,
    #             # "style": 0.25,
    #             "speed": 1.0            },
    #         "apply_text_normalization": "off"
    #     }



    #     try:

    #         # 🆕 TTS API 호출 시간 측정
    #         tts_api_start = time.time()
    #         self.get_logger().info("⏳ ElevenLabs TTS API 호출 시작")



    #         response = requests.post(
    #             url,
    #             headers=headers,
    #             data=json.dumps(data, ensure_ascii=False).encode('utf-8')
    #         )

    #         tts_api_end = time.time()
    #         tts_api_duration = tts_api_end - tts_api_start
    #         self.get_logger().info(f"✅ TTS API 호출 완료, 소요시간: {tts_api_duration:.2f}초")



            
    #         if response.status_code == 200:
    #             with open(self.reply_path, "wb") as f:
    #                 f.write(response.content)
                
    #             file_size = os.path.getsize(self.reply_path)
    #             self.get_logger().info(f"🟢 음성 변환 성공 → {self.reply_path} ({file_size} bytes)")

    #             # 🆕 WAV 변환 시간 측정
    #             wav_convert_start = time.time()
    #             self.get_logger().info("⏳ WAV 변환 시작")


                
    #             # 🆕 WAV 변환 (STT API용)
    #             sound = AudioSegment.from_file(self.reply_path, format="mp3")
    #             wav_path = "/tmp/tts_for_stt.wav"
    #             sound = sound.set_frame_rate(16000).set_channels(1)
    #             sound.export(wav_path, format="wav")
                
    #             wav_convert_end = time.time()
    #             wav_convert_duration = wav_convert_end - wav_convert_start
    #             self.get_logger().info(f"✅ WAV 변환 완료, 소요시간: {wav_convert_duration:.2f}초")

                
    #             # 🆕 STT 타임스탬프 추출
    #             stt_timestamps = self.extract_word_timestamps(wav_path, cleaned_text)

    #             # 🆕 자막 처리 시간 측정
    #             subtitle_process_start = time.time()
    #             self.get_logger().info("⏳ 자막 정렬 및 처리 시작")

                
    #             # 🔑 핵심: 원본 텍스트로 덮어쓰기
    #             corrected_timestamps = self.merge_original_with_stt_timestamps(
    #                 original_text=cleaned_text,
    #                 stt_timestamps=stt_timestamps
    #             )

    #             subtitle_process_end = time.time()
    #             subtitle_process_duration = subtitle_process_end - subtitle_process_start
    #             self.get_logger().info(f"✅ 자막 정렬 및 처리 완료, 소요시간: {subtitle_process_duration:.2f}초")
                
    #             # 🆕 자막 퍼블리시 시간 측정
    #             publish_start = time.time()


                
    #             # 🆕 수정된 자막 데이터 퍼블리시
    #             if corrected_timestamps:
    #                 subtitle_data = {
    #                     "original_text": cleaned_text,
    #                     "words": corrected_timestamps,
    #                     "total_duration": corrected_timestamps[-1]["end"] if corrected_timestamps else 0
    #                 }

    #                 # 🔑 핵심 수정: 즉시 퍼블리시하지 않고 저장만
    #                 self.pending_subtitle_data = subtitle_data


    #                 # # 🆕 순차 단일 단어 자막 시작
    #                 # self.publish_single_word_subtitle(subtitle_data)
                    
    #                 msg = String()
    #                 msg.data = json.dumps(subtitle_data, ensure_ascii=False)
    #                 self.tts_subtitle_publisher.publish(msg)
    #                 self.get_logger().info(f"📝 수정된 자막 데이터 퍼블리시: {len(corrected_timestamps)}개 단어")
    #             else:
    #                 self.get_logger().warning("⚠️ 자막 수정 실패 - 기본 자막 사용")
    #                 self._publish_fallback_subtitle(cleaned_text)
    #             publish_end = time.time()
    #             publish_duration = publish_end - publish_start
    #             self.get_logger().info(f"✅ 자막 퍼블리시 완료, 소요시간: {publish_duration:.2f}초")
                
    #             # 🆕 전체 소요시간 계산 및 로그
    #             total_end_time = time.time()
    #             total_duration = total_end_time - total_start_time
                
    #             self.get_logger().info("="*60)
    #             self.get_logger().info("📊 TTS 전체 프로세스 시간 분석")
    #             self.get_logger().info(f"  🎤 TTS API 호출:        {tts_api_duration:.2f}초 ({tts_api_duration/total_duration*100:.1f}%)")
    #             self.get_logger().info(f"  🔄 WAV 변환:            {wav_convert_duration:.2f}초 ({wav_convert_duration/total_duration*100:.1f}%)")
    #             self.get_logger().info(f"  📝 동적자막 생성:        {getattr(self, '_last_stt_duration', 0):.2f}초 ({getattr(self, '_last_stt_duration', 0)/total_duration*100:.1f}%)")
    #             self.get_logger().info(f"  🔤 형태소 분석:          {getattr(self, '_last_morphology_duration', 0):.3f}초 ({getattr(self, '_last_morphology_duration', 0)/total_duration*100:.1f}%)")  # 🆕 추가
    #             self.get_logger().info(f"  ⚙️ 자막 처리:           {subtitle_process_duration:.2f}초 ({subtitle_process_duration/total_duration*100:.1f}%)")
    #             self.get_logger().info(f"  📡 퍼블리시:            {publish_duration:.2f}초 ({publish_duration/total_duration*100:.1f}%)")
    #             self.get_logger().info(f"  🏁 전체 소요시간:        {total_duration:.2f}초")
    #             self.get_logger().info("="*60)


                    
    #         else:
    #             self.get_logger().error(f"🔴 TTS 오류: {response.status_code}")
    #             self.get_logger().error(f"응답: {response.text}")
    #     except Exception as e:
    #         self.get_logger().error(f"🔴 TTS 호출 실패: {e}")










    def text2speech(self, text):
        """
        Naver Clova Voice API 호출 → reply.mp3 저장 → 원본 텍스트 기반 자막 생성
        """

        # 🆕 전체 소요시간 측정 시작
        total_start_time = time.time()
        self.get_logger().info("⏳ TTS 전체 프로세스 시작")
        
        # Naver Clova Voice API 설정
        client_id = "fo0f88v3wl"
        client_secret = "KUa8Lcp8JAVE2EK92G0dtyn8ywWKFTH2iKOhnoaB"
        url = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
        
        headers = {
            "X-NCP-APIGW-API-KEY-ID": client_id,
            "X-NCP-APIGW-API-KEY": client_secret,
            "Content-Type": "application/x-www-form-urlencoded"
        }

        # 🆕 텍스트 전처리
        cleaned_text = text.strip()
        cleaned_text = ' '.join(cleaned_text.split())
        
        # 🆕 디버깅 로그
        self.get_logger().info(f"🗣️ TTS 원본 텍스트: '{cleaned_text}'")

        # Naver Clova Voice 설정
        data = {
            "speaker": "nsangdo",  # 음성 종류 (nara, clara, matt, shinji, meow, dinna 등)
            "volume": "0",      # 볼륨 (-5 ~ 5)
            "speed": "0",       # 속도 (-5 ~ 5)  
            "pitch": "0",       # 음높이 (-5 ~ 5)
            "format": "mp3",    # 출력 포맷 (mp3, wav, ogg)
            "text": cleaned_text
        }

        try:
            # 🆕 TTS API 호출 시간 측정
            tts_api_start = time.time()
            self.get_logger().info("⏳ Naver Clova Voice API 호출 시작")

            response = requests.post(url, headers=headers, data=data)

            tts_api_end = time.time()
            tts_api_duration = tts_api_end - tts_api_start
            self.get_logger().info(f"✅ TTS API 호출 완료, 소요시간: {tts_api_duration:.2f}초")



            
            if response.status_code == 200:
                with open(self.reply_path, "wb") as f:
                    f.write(response.content)
                
                file_size = os.path.getsize(self.reply_path)
                self.get_logger().info(f"🟢 음성 변환 성공 → {self.reply_path} ({file_size} bytes)")

                # 🆕 WAV 변환 시간 측정
                wav_convert_start = time.time()
                self.get_logger().info("⏳ WAV 변환 시작")


                
                # 🆕 WAV 변환 (STT API용)
                sound = AudioSegment.from_file(self.reply_path, format="mp3")
                wav_path = "/tmp/tts_for_stt.wav"
                sound = sound.set_frame_rate(16000).set_channels(1)
                sound.export(wav_path, format="wav")
                
                wav_convert_end = time.time()
                wav_convert_duration = wav_convert_end - wav_convert_start
                self.get_logger().info(f"✅ WAV 변환 완료, 소요시간: {wav_convert_duration:.2f}초")


                
                # 🆕 STT 타임스탬프 추출
                stt_timestamps = self.extract_word_timestamps(wav_path, cleaned_text)

                # 🆕 자막 처리 시간 측정
                subtitle_process_start = time.time()
                self.get_logger().info("⏳ 자막 정렬 및 처리 시작")

                
                # 🔑 핵심: 원본 텍스트로 덮어쓰기
                corrected_timestamps = self.merge_original_with_stt_timestamps(
                    original_text=cleaned_text,
                    stt_timestamps=stt_timestamps
                )

                subtitle_process_end = time.time()
                subtitle_process_duration = subtitle_process_end - subtitle_process_start
                self.get_logger().info(f"✅ 자막 정렬 및 처리 완료, 소요시간: {subtitle_process_duration:.2f}초")
                
                # 🆕 자막 퍼블리시 시간 측정
                publish_start = time.time()


                
                # 🆕 수정된 자막 데이터 퍼블리시
                if corrected_timestamps:
                    subtitle_data = {
                        "original_text": cleaned_text,
                        "words": corrected_timestamps,
                        "total_duration": corrected_timestamps[-1]["end"] if corrected_timestamps else 0
                    }

                    # 🔑 핵심 수정: 즉시 퍼블리시하지 않고 저장만
                    self.pending_subtitle_data = subtitle_data


                    # # 🆕 순차 단일 단어 자막 시작
                    # self.publish_single_word_subtitle(subtitle_data)
                    
                    msg = String()
                    msg.data = json.dumps(subtitle_data, ensure_ascii=False)
                    self.tts_subtitle_publisher.publish(msg)
                    self.get_logger().info(f"📝 수정된 자막 데이터 퍼블리시: {len(corrected_timestamps)}개 단어")
                else:
                    self.get_logger().warning("⚠️ 자막 수정 실패 - 기본 자막 사용")
                    self._publish_fallback_subtitle(cleaned_text)
                publish_end = time.time()
                publish_duration = publish_end - publish_start
                self.get_logger().info(f"✅ 자막 퍼블리시 완료, 소요시간: {publish_duration:.2f}초")
                
                # 🆕 전체 소요시간 계산 및 로그
                total_end_time = time.time()
                total_duration = total_end_time - total_start_time
                
                self.get_logger().info("="*60)
                self.get_logger().info("📊 TTS 전체 프로세스 시간 분석")
                self.get_logger().info(f"  🎤 TTS API 호출:        {tts_api_duration:.2f}초 ({tts_api_duration/total_duration*100:.1f}%)")
                self.get_logger().info(f"  🔄 WAV 변환:            {wav_convert_duration:.2f}초 ({wav_convert_duration/total_duration*100:.1f}%)")
                self.get_logger().info(f"  📝 동적자막 생성:        {getattr(self, '_last_stt_duration', 0):.2f}초 ({getattr(self, '_last_stt_duration', 0)/total_duration*100:.1f}%)")
                self.get_logger().info(f"  🔤 형태소 분석:          {getattr(self, '_last_morphology_duration', 0):.3f}초 ({getattr(self, '_last_morphology_duration', 0)/total_duration*100:.1f}%)")  # 🆕 추가
                self.get_logger().info(f"  ⚙️ 자막 처리:           {subtitle_process_duration:.2f}초 ({subtitle_process_duration/total_duration*100:.1f}%)")
                self.get_logger().info(f"  📡 퍼블리시:            {publish_duration:.2f}초 ({publish_duration/total_duration*100:.1f}%)")
                self.get_logger().info(f"  🏁 전체 소요시간:        {total_duration:.2f}초")
                self.get_logger().info("="*60)


                    
            else:
                self.get_logger().error(f"🔴 TTS 오류: {response.status_code}")
                self.get_logger().error(f"응답: {response.text}")
        except Exception as e:
            self.get_logger().error(f"🔴 TTS 호출 실패: {e}")




















    def align_words_with_dynamic_programming(self, original_words, stt_timestamps):
        """
        동적 계획법과 음성학적 유사도를 활용한 정교한 단어 정렬
        """
        from difflib import SequenceMatcher
        import re
        
        # 전처리: 구두점 제거 및 정규화
        def normalize_word(word):
            return re.sub(r'[^\w]', '', word).lower()
        
        orig_normalized = [normalize_word(w) for w in original_words]
        stt_words = [ts["word"].lower() for ts in stt_timestamps]
        
        # 🆕 SequenceMatcher를 사용한 최적 정렬
        matcher = SequenceMatcher(None, orig_normalized, stt_words)
        opcodes = matcher.get_opcodes()
        
        aligned_results = []
        current_stt_time = 0.0
        
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'equal':
                # 정확히 매칭된 단어들
                for k in range(i2 - i1):
                    orig_idx = i1 + k
                    stt_idx = j1 + k
                    
                    aligned_results.append({
                        "word": original_words[orig_idx],
                        "start": stt_timestamps[stt_idx]["start"],
                        "end": stt_timestamps[stt_idx]["end"],
                        "confidence": stt_timestamps[stt_idx]["confidence"],
                        "match_type": "exact"
                    })
                    current_stt_time = stt_timestamps[stt_idx]["end"]
            
            elif tag == 'delete':
                # STT가 놓친 원본 단어들 - 시간 보간
                self._interpolate_missing_words(
                    original_words[i1:i2], aligned_results, current_stt_time
                )
            
            elif tag == 'insert':
                # STT에만 있는 단어들 - 시간만 진행
                if j2 <= len(stt_timestamps):
                    current_stt_time = stt_timestamps[j2-1]["end"]
            
            elif tag == 'replace':
                # 다른 단어들 - 음성학적 유사도로 매핑
                self._map_mismatched_words(
                    original_words[i1:i2], stt_timestamps[j1:j2], aligned_results
                )
        
        return aligned_results

    def _interpolate_missing_words(self, missing_words, aligned_results, start_time):
        """
        누락된 단어들의 타이밍 보간
        """
        base_duration = 0.4  # 기본 단어 지속시간
        
        for i, word in enumerate(missing_words):
            # 단어 길이에 따른 지속시간 조정
            word_length = len(re.sub(r'[^\w]', '', word))
            duration = max(0.2, min(0.8, word_length * 0.1))
            
            word_start = start_time + (i * base_duration)
            word_end = word_start + duration
            
            aligned_results.append({
                "word": word,
                "start": word_start,
                "end": word_end,
                "confidence": 0.3,  # 낮은 신뢰도
                "match_type": "interpolated"
            })

    def _map_mismatched_words(self, orig_words, stt_words, aligned_results):
        """
        불일치 단어들의 음성학적 유사도 기반 매핑
        """
        if not stt_words:
            return
            
        # 전체 STT 시간 범위
        total_duration = stt_words[-1]["end"] - stt_words[0]["start"]
        time_per_word = total_duration / len(orig_words)
        
        for i, orig_word in enumerate(orig_words):
            word_start = stt_words[0]["start"] + (i * time_per_word)
            word_end = word_start + time_per_word
            
            # 🆕 STT 단어와의 유사도 계산
            best_confidence = 0.0
            for stt_word in stt_words:
                similarity = self._calculate_phonetic_similarity(
                    orig_word, stt_word["word"]
                )
                best_confidence = max(best_confidence, similarity)
            
            aligned_results.append({
                "word": orig_word,
                "start": word_start,
                "end": word_end,
                "confidence": best_confidence,
                "match_type": "phonetic_match"
            })

    def _calculate_phonetic_similarity(self, word1, word2):
        """
        음성학적 유사도 계산 (한글 특화)
        """
        from difflib import SequenceMatcher
        
        # 1. 편집 거리 기반 유사도
        similarity = SequenceMatcher(None, word1.lower(), word2.lower()).ratio()
        
        # 2. 한글 자모 분해 유사도 (선택적)
        # 여기서는 단순화하여 편집 거리만 사용
        
        return similarity * 0.8  # 신뢰도 조정
    

    def smooth_timestamps(self, timestamps):
        """
        타이밍 스무딩 및 겹침 해결
        """
        if not timestamps:
            return timestamps
        
        smoothed = []
        
        for i, ts in enumerate(timestamps):
            current = ts.copy()
            
            # 이전 단어와 겹침 방지
            if i > 0:
                prev_end = smoothed[-1]["end"]
                if current["start"] < prev_end:
                    current["start"] = prev_end + 0.05  # 50ms 간격
            
            # 너무 짧은 지속시간 조정
            min_duration = 0.15  # 최소 150ms
            if (current["end"] - current["start"]) < min_duration:
                current["end"] = current["start"] + min_duration
            
            # 다음 단어와의 간격 조정
            if i < len(timestamps) - 1:
                next_start = timestamps[i + 1]["start"]
                if current["end"] > next_start:
                    # 시간을 균등 분할
                    mid_time = (current["start"] + next_start) / 2
                    current["end"] = mid_time
            
            smoothed.append(current)
        
        return smoothed






    # def merge_original_with_stt_timestamps(self, original_text, stt_timestamps):
    #     """
    #     🆕 개선된 단어 정렬 및 타이밍 매핑
    #     """
    #     try:
    #         original_words = original_text.split()
            
    #         if not stt_timestamps:
    #             return self._create_default_timestamps(original_words)
            
    #         # 🆕 지능적 정렬 수행
    #         aligned_timestamps = self.align_words_with_dynamic_programming(
    #             original_words, stt_timestamps
    #         )
            
    #         # 🆕 타이밍 후처리
    #         smoothed_timestamps = self.smooth_timestamps(aligned_timestamps)
            
    #         # 디버깅 로그
    #         self.get_logger().info(f"🎯 정렬 결과: {len(original_words)}개 원본 → {len(smoothed_timestamps)}개 자막")
            
    #         for i, ts in enumerate(smoothed_timestamps[:5]):  # 처음 5개만 로깅
    #             self.get_logger().info(f"  [{i}] '{ts['word']}': {ts['start']:.2f}s-{ts['end']:.2f}s (신뢰도: {ts['confidence']:.2f}, 매칭: {ts['match_type']})")
            
    #         return smoothed_timestamps
            
    #     except Exception as e:
    #         self.get_logger().error(f"❌ 고급 정렬 실패: {e}")
    #         return stt_timestamps  # 실패시 원본 STT 결과 반환




    def merge_original_with_stt_timestamps(self, original_text, stt_timestamps):
        """
        🆕 명사 전용 자막 생성 (기존 함수 완전 교체)
        """
        try:
            # 🆕 형태소 분석 시간 측정 시작
            morphology_extraction_start = time.time()
            
            # 1. 명사 추출
            nouns = self.extract_nouns_from_text(original_text)
            
            # 🆕 형태소 분석 시간 저장 (나중에 전체 시간 분석에서 사용)
            morphology_extraction_end = time.time()
            self._last_morphology_duration = morphology_extraction_end - morphology_extraction_start
            
            if not nouns:
                self.get_logger().warning("⚠️ 추출된 명사가 없습니다. 기본 자막 생성")
                return self._create_fallback_subtitle_data(original_text)
            
            # 나머지 코드는 동일...
            # 2. STT가 없는 경우 기본 타임스탬프 생성
            if not stt_timestamps:
                return self._create_default_noun_timestamps(nouns)
            
            # 3. 명사와 STT 타임스탬프 매핑 및 분할
            noun_timestamps = self.map_nouns_to_timestamps(nouns, stt_timestamps, original_text)
            
            if not noun_timestamps:
                self.get_logger().warning("⚠️ 명사 타임스탬프 매핑 실패")
                return self._create_default_noun_timestamps(nouns)
            
            self.get_logger().info(f"✅ 명사 자막 생성 완료: {len(noun_timestamps)}개 명사")
            for i, ts in enumerate(noun_timestamps[:3]):  # 처음 3개만 로그
                divided_info = " (분할됨)" if ts.get('divided') else ""
                self.get_logger().info(f"  [{i}] '{ts['word']}': {ts['start']:.2f}s-{ts['end']:.2f}s{divided_info}")
            
            return noun_timestamps
            
        except Exception as e:
            self.get_logger().error(f"❌ 명사 자막 생성 실패: {e}")
            return self._create_fallback_subtitle_data(original_text)




    def _create_default_noun_timestamps(self, nouns):
            """
            STT 실패시 명사용 기본 타임스탬프 생성
            """
            timestamps = []
            current_time = 0.0
            
            for noun in nouns:
                duration = max(1.0, len(noun) * 0.3)  # 명사는 더 긴 지속시간
                
                timestamps.append({
                    'word': noun,
                    'start': round(current_time, 3),
                    'end': round(current_time + duration, 3),
                    'confidence': 1.0,
                    'original_stt_word': noun
                })
                
                current_time += duration + 0.5  # 명사 간 간격
            
            return timestamps

    def _create_fallback_subtitle_data(self, text):
        """
        모든 처리 실패시 폴백 데이터
        """
        return [{
            'word': '처리 중...',
            'start': 0,
            'end': 3,
            'confidence': 1.0,
            'original_stt_word': text[:20] + '...' if len(text) > 20 else text
        }]







    def _distribute_extra_words(self, original_words, stt_timestamps):
        """
        원본 단어 수가 STT보다 많을 때 시간을 분할하여 배치 (구두점 포함)
        """
        corrected_timestamps = []
        
        if not stt_timestamps:
            return self._create_default_timestamps(original_words)
        
        # 🆕 구두점이 있는 단어는 조금 더 짧은 시간 할당
        total_duration = stt_timestamps[-1]["end"] - stt_timestamps[0]["start"]
        
        # 🆕 단어별 가중치 계산 (구두점 있는 단어는 더 짧게)
        word_weights = []
        for word in original_words:
            import re
            clean_length = len(re.sub(r'[^\w]', '', word))
            if clean_length == 0:  # 구두점만 있는 경우
                weight = 0.1
            else:
                weight = max(0.3, clean_length * 0.2)  # 최소 0.3초, 글자당 0.2초
            word_weights.append(weight)
        
        total_weight = sum(word_weights)
        start_time = stt_timestamps[0]["start"]
        
        for i, (word, weight) in enumerate(zip(original_words, word_weights)):
            duration = (weight / total_weight) * total_duration
            word_start = start_time
            word_end = word_start + duration
            
            corrected_timestamps.append({
                "word": word,  # 🎯 구두점 포함
                "start": round(word_start, 3),
                "end": round(word_end, 3),
                "confidence": 0.9
            })
            
            start_time = word_end  # 다음 단어의 시작점
        
        return corrected_timestamps

    def _create_default_timestamps(self, words):
        """
        STT 실패시 기본 타임스탬프 생성 (구두점 포함)
        """
        timestamps = []
        current_time = 0.0
        
        for word in words:
            import re
            # 🆕 구두점 포함 단어의 길이에 따라 시간 조정
            clean_length = len(re.sub(r'[^\w]', '', word))
            
            if clean_length == 0:  # 구두점만 있는 경우 (예: ".")
                duration = 0.1
            elif len(word) <= 2:  # 짧은 단어
                duration = 0.3
            elif len(word) <= 5:  # 중간 길이 단어
                duration = 0.5
            else:  # 긴 단어
                duration = 0.7
            
            start_time = current_time
            end_time = start_time + duration
            
            timestamps.append({
                "word": word,  # 🎯 구두점 포함
                "start": round(start_time, 3),
                "end": round(end_time, 3),
                "confidence": 1.0
            })
            
            current_time = end_time
        
        return timestamps


    def _publish_fallback_subtitle(self, text):
        """
        폴백 자막 데이터 퍼블리시
        """
        fallback_data = {
            "original_text": text,
            "words": [{"word": text, "start": 0, "end": 3, "confidence": 1.0}],
            "total_duration": 3
        }
        
        msg = String()
        msg.data = json.dumps(fallback_data, ensure_ascii=False)
        self.tts_subtitle_publisher.publish(msg)







# ──────────────────────────────────────────────────────────────────────────────
# 재생 상태 및 로그 상태 저장
# ──────────────────────────────────────────────────────────────────────────────


    def publish_music_status(self, status):
        """
        음악 재생 상태 퍼블리시
        """
        msg = String()
        msg.data = status
        self.publisher_.publish(msg)
        self.get_logger().info(f"📡 음악 상태: {status}")
        self.save_log(f"📡 음악 상태: {status}")

    def save_log(self, message):
        """
        로그 파일에 저장
        """
        log_file_path = "/home/nvidia/ros2_ws/_logs/Mp3Player_log.txt"
        log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(log_message)
# ──────────────────────────────────────────────────────────────────────────────
# reply TTS 코드
# ──────────────────────────────────────────────────────────────────────────────
    def tts_request_callback(self, msg):
        """TTS 요청 수신 및 처리"""
        try:
            reply_text = msg.data
            self.get_logger().info(f"🗣️ TTS 요청 수신: {reply_text}")
            self.save_log(f"🗣️ TTS 요청 수신: {reply_text}")

            # 🆕 전체 TTS 처리 시간 측정 시작
            process_start_time = time.time()


            
            # TTS 생성 시작 신호
            self.publish_tts_status("tts_generating")
            
            # TTS 생성
            self.text2speech(reply_text)
            
            # TTS 준비 완료 신호
            self.publish_tts_status("tts_ready")



            # 🆕 전체 처리 시간 계산
            process_end_time = time.time()
            total_process_time = process_end_time - process_start_time
            
            self.get_logger().info("🎊 TTS 요청 처리 완료!")
            self.get_logger().info(f"📈 요청 수신부터 준비완료까지 총 소요시간: {total_process_time:.2f}초")
            self.save_log(f"📈 TTS 총 처리시간: {total_process_time:.2f}초")



            
        except Exception as e:
            error_msg = f"❌ TTS 생성 중 오류: {e}"
            self.get_logger().error(error_msg)
            self.save_log(error_msg)
            self.publish_tts_status("tts_error")



    def play_tts_audio(self):
        """TTS 오디오 재생 (App.jsx에서 요청시) - 스펙트럼 포함"""
        try:
            self.get_logger().info("🎵 TTS 오디오 재생 시작")
            self.save_log("🎵 TTS 오디오 재생 시작")
            
            # 재생 시작 신호
            self.publish_tts_status("tts_playing")

            # 🆕 TTS 재생 시작과 동시에 자막 시작
            if self.pending_subtitle_data:
                self.get_logger().info("📺 TTS 재생과 동시에 자막 시작")
                self.publish_single_word_subtitle(self.pending_subtitle_data)
                self.pending_subtitle_data = None  # 사용 후 초기화
                






            
            # 🆕 TTS 전용 스펙트럼과 함께 재생
            self.play_tts_with_spectrum(self.reply_path)
            
            # 재생 완료 신호
            self.publish_tts_status("tts_done")
            # 🔑 모든 자막 종료
            final_data = {
                   "word": "",
                    "display_mode": "finished"
                }
            final_msg = String()
            final_msg.data = json.dumps(final_data, ensure_ascii=False)
            self.single_word_publisher.publish(final_msg)
                
            self.get_logger().info("📺 순차 단일 자막 완료")
            
            self.get_logger().info("🎵 TTS 오디오 재생 완료")
            self.save_log("🎵 TTS 오디오 재생 완료")
            
        except Exception as e:
            error_msg = f"❌ TTS 재생 중 오류: {e}"
            self.get_logger().error(error_msg)
            self.save_log(error_msg)
            self.publish_tts_status("tts_error")

    def play_tts_with_spectrum(self, file_path):
        """TTS 전용 전체 음량 기반 스펙트럼과 함께 재생"""
        try:
            sound = AudioSegment.from_file(file_path, format="mp3")
            sound = self.match_target_amplitude(sound, -14.0)
            
            # 임시 WAV로 변환 후 저장
            temp_wav = "/tmp/tts_audio.wav"
            sound.export(temp_wav, format="wav")

            # TTS 전용 스펙트럼과 재생 병렬로 실행
            playback_thread = threading.Thread(target=self.tts_publish_and_play, args=(temp_wav,))
            playback_thread.start()
            playback_thread.join()

        except Exception as e:
            self.get_logger().error(f"❌ TTS 스펙트럼 재생 실패: {file_path} → {e}")
            self.save_log(f"❌ TTS 스펙트럼 재생 실패: {file_path} → {e}")

   



    def tts_publish_and_play(self, wav_path):
        """TTS 재생 시간 정보만 퍼블리시 (음량 정보 제거)"""
        wf = wave.open(wav_path, 'rb')
        chunk_size = 1024
        frame_rate = wf.getframerate()
        start_time = time.time()

        def publish_tts_time_only():
            data = wf.readframes(chunk_size)
            
            while data:
                current_time = round(time.time() - start_time, 3)
                
                # 🔑 음량 정보 완전 제거 - 시간 정보만 전송
                time_data = {
                    "current_time": current_time,
                    "status": "playing",
                    "timestamp": time.time()
                }

                msg = String()
                msg.data = json.dumps(time_data)
                self.tts_spectrum_publisher.publish(msg)
                
                data = wf.readframes(chunk_size)
                time.sleep(chunk_size / frame_rate)
            
            # 재생 완료 신호
            final_data = {
                "current_time": current_time,
                "status": "finished",
                "timestamp": time.time()
            }
            msg = String()
            msg.data = json.dumps(final_data)
            self.tts_spectrum_publisher.publish(msg)

        time_thread = threading.Thread(target=publish_tts_time_only)
        time_thread.start()
        os.system(f"aplay {wav_path}")
        time_thread.join()
        wf.close()










    def publish_tts_status(self, status):
        """TTS 상태 퍼블리시"""
        msg = String()
        msg.data = status
        self.tts_status_publisher.publish(msg)
        self.get_logger().info(f"📡 TTS 상태: {status}")
        self.save_log(f"📡 TTS 상태: {status}")


        # # 🆕 TTS 완료 시 UserQuestion에게 STT 재시작 신호 전송
        # if status == "tts_done":
        #     restart_msg = String()
        #     restart_msg.data = "restart_stt_after_tts"
        #     self.stt_restart_publisher.publish(restart_msg)
        #     self.get_logger().info("📡 TTS 완료 - UserQuestion STT 재시작 신호 전송")
        #     self.save_log("📡 TTS 완료 - UserQuestion STT 재시작 신호 전송")

            


    


    def tts_play_request_callback(self, msg):
        """TTS 재생 요청 처리"""
        if msg.data == "play_tts":
            self.play_tts_audio()



def main(args=None):
    rclpy.init(args=args)
    node = Mp3Player()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

