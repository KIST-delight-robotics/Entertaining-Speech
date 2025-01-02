import os
import re
from moviepy.editor import AudioFileClip

def parse_srt(srt_file):
    """SRT 파일에서 시작 시간, 종료 시간 및 텍스트를 추출합니다."""
    with open(srt_file, "r", encoding="utf-8") as file:
        srt_data = file.read()

    pattern = re.compile(r"(\d+)\n(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})\n(.*?)(?=\n\n|\Z)", re.DOTALL)
    segments = []

    for match in pattern.finditer(srt_data):
        start_time = int(match.group(2)) * 3600 + int(match.group(3)) * 60 + int(match.group(4)) + int(match.group(5)) / 1000
        end_time = int(match.group(6)) * 3600 + int(match.group(7)) * 60 + int(match.group(8)) + int(match.group(9)) / 1000
        text = match.group(10).strip()
        segments.append((start_time, end_time, text))

    return segments

def sanitize_filename(text):
    """파일명에 사용할 수 없는 문자를 제거합니다."""
    return re.sub(r'[\\/*?:"<>|]', "", text)

def export_audio_segments(audio_file, srt_file, output_folder):
    """SRT 세그먼트를 기반으로 오디오 파일을 잘라서 저장합니다."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    segments = parse_srt(srt_file)
    audio_clip = AudioFileClip(audio_file)
    
    max_duration = audio_clip.duration  # 오디오 파일의 최대 길이 가져오기
    used_filenames = {}

    for start_time, end_time, text in segments:
        if end_time > max_duration:
            end_time = max_duration
        
        segment_duration = end_time - start_time
        
        if segment_duration <= 1.0 or segment_duration >= 10.0:
            print(f"Segment ({start_time:.2f}-{end_time:.2f}): Skipped (Duration: {segment_duration:.2f} seconds)")
            continue
        
        segment_audio = audio_clip.subclip(start_time, end_time)
        sanitized_text = sanitize_filename(text)[:30]
        
        if sanitized_text in used_filenames:
            used_filenames[sanitized_text] += 1
            output_file = os.path.join(output_folder, f"{sanitized_text}_{used_filenames[sanitized_text]}.mp3")
        else:
            used_filenames[sanitized_text] = 1
            output_file = os.path.join(output_folder, f"{sanitized_text}.mp3")
        
        segment_audio.write_audiofile(output_file, codec='mp3')
        print(f"Segment ({start_time:.2f}-{end_time:.2f}): Saved as {output_file} (Duration: {segment_duration:.2f} seconds)")

def process_directory(directory):
    """주어진 디렉토리의 모든 mp3 및 srt 파일을 찾아서 세그먼트로 나눕니다."""
    files = os.listdir(directory)
    audio_files = [f for f in files if f.endswith('.mp3')]
    srt_files = {os.path.splitext(f)[0]: f for f in files if f.endswith('.srt')}

    for audio_file in audio_files:
        base_name = os.path.splitext(audio_file)[0]
        if base_name in srt_files:
            srt_file = srt_files[base_name]
            print(f"Processing {audio_file} with {srt_file}")
            export_audio_segments(
                os.path.join(directory, audio_file),
                os.path.join(directory, srt_file),
                os.path.join(directory, f"./2019")
            )

# 사용 예시
directory = "./ing/2019"  # mp3 및 srt 파일이 포함된 디렉토리 경로
process_directory(directory)        
