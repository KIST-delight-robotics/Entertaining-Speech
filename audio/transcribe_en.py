from faster_whisper import WhisperModel
import sys
import os

def seconds_to_srt_time(seconds):
    """초 단위 시간을 SRT 파일 형식의 시간 문자열로 변환합니다."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(seconds):02},{milliseconds:03}"

def process_audio_to_srt(audio_file, model, srt_folder):
    """오디오 파일을 처리하여 SRT 파일로 변환합니다."""
    base_name = os.path.splitext(os.path.basename(audio_file))[0]  # 파일 이름에서 확장자를 제거합니다.
    srt_file_name = f"{srt_folder}/{base_name}.srt"

    # 오디오 파일 변환
    segments, info = model.transcribe(audio_file, beam_size=5,language="en")

    with open(srt_file_name, "w", encoding="utf-8") as file:
        i = 0  # 세그먼트 번호를 추적하기 위한 변수
        for segment in segments:
            start_time = seconds_to_srt_time(segment.start)
            end_time = seconds_to_srt_time(segment.end)
            i += 1  # 세그먼트 번호를 증가시킵니다.
            file.write(f"{i}\n")
            file.write(f"{start_time} --> {end_time}\n")
            file.write(f"{segment.text}\n\n")
            
            sys.stdout.write(f"\r세그먼트 {i} 처리 중...")
            sys.stdout.flush()

    print(f"\n'{audio_file}' 변환 완료! 결과는 '{srt_file_name}' 파일에 저장되었습니다.")

model_size = "large-v3"
# GPU에서 FP16으로 실행
model = WhisperModel(model_size, device="CUDA", compute_type="float16")

audio_folder = "./ex"  # 현재 폴더 내의 오디오 파일들을 처리합니다.
srt_folder = "./ex"  # SRT 파일을 저장할 폴더

# SRT 폴더가 없으면 생성
if not os.path.exists(srt_folder):
    os.makedirs(srt_folder)

# 현재 폴더 내의 모든 MP3 파일을 찾아서 처리
for file in os.listdir(audio_folder):
    if file.endswith(".mp3"):
        audio_file = os.path.join(audio_folder, file)
        process_audio_to_srt(audio_file, model, srt_folder)
# python transcribe_audio.py 실행 명령어
