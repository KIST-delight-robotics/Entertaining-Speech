

//동적자막(google-stt 이용)

import React, { useEffect, useRef, useState } from 'react';
import ros from './ros';
import ROSLIB from 'roslib';






// 중앙 일부만 사용
function getCentralSlice(arr, ratio = 0.6) {
  const total = arr.length;
  const sliceSize = Math.floor(total * ratio);
  const start = Math.floor((total - sliceSize) / 2);
  return arr.slice(start, start + sliceSize);
}

// 다운샘플(평균)로 바 개수 줄이기
function downsampleArray(arr, targetLen) {
  const result = [];
  const binSize = Math.floor(arr.length / targetLen);
  for (let i = 0; i < targetLen; i++) {
    const start = i * binSize;
    const end = (i + 1) * binSize;
    const bin = arr.slice(start, end);
    result.push(bin.reduce((a, b) => a + b, 0) / bin.length || 0);
  }
  return result;
}

// function downsampleArray(arr, targetLen) {
//   const result = [];
//   const binSize = Math.floor(arr.length / targetLen);
//   for (let i = 0; i < targetLen; i++) {
//     const start = i * binSize;
//     const end = (i + 1) * binSize;
//     const bin = arr.slice(start, end);
//     // 🆕 빈 배열 처리 추가
//     result.push(bin.length > 0 ? Math.max(...bin) : 0);
//   }
//   return result;
// }





function SpectrumVisualizer() {

  // 🆕 비디오 관련 상태 변수들 추가
  const [currentVideo, setCurrentVideo] = useState(null);
  const [videoVisible, setVideoVisible] = useState(false);
  const [currentReply, setCurrentReply] = useState(''); // 추가: reply 텍스트
  const videoRef = useRef(null); // 비디오 ref 추가



  // 🆕 분리된 스펙트럼 상태
  const [musicSpectrum, setMusicSpectrum] = useState([]);
  const [micSpectrum, setMicSpectrum] = useState([]);

  const [musicPlaying, setMusicPlaying] = useState(false);
  //const [currentImage, setCurrentImage] = useState(null); // 이미지 상태 추가
  const canvasRef = useRef(null);
  const [recommendStatus, setRecommendStatus] = useState('done');
  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 400 });
  //const [imageVisible, setImageVisible] = useState(false);
  const [canShowSpectrum, setCanShowSpectrum] = useState(false);

  // 🆕 방향 상태 추가
  const [soundDirection, setSoundDirection] = useState(0);
  const [screenFlipped, setScreenFlipped] = useState(false);

  // 🆕 누락된 상태 변수들 추가
  const [fixedDirection, setFixedDirection] = useState(null);
  const [isDirectionFixed, setIsDirectionFixed] = useState(false);

  

// 🆕 trigger_detected 상태 추가 (기존 플래그 재사용)
  const [triggerDetected, setTriggerDetected] = useState(false);
  const [currentGif, setCurrentGif] = useState('');


  // 기존 상태 변수들 다음에 추가
  const [waitingSpectrum, setWaitingSpectrum] = useState([]);
  const [waitingImage, setWaitingImage] = useState(null);
  const [waitingImageVisible, setWaitingImageVisible] = useState(false);
  const [isWaitingAudioMode, setIsWaitingAudioMode] = useState(false);
  const [isWaitingImageMode, setIsWaitingImageMode] = useState(false);


  // Mp3Player waiting 전용 상태 추가
  const [mp3WaitingSpectrum, setMp3WaitingSpectrum] = useState([]);
  const [isMp3WaitingMode , setIsMp3WaitingMode] = useState(false);


  // 기존 상태 변수들 다음에 추가
  const [isTransitioning, setIsTransitioning] = useState(false);






  // SpectrumVisualizer 컴포넌트에 상태 추가
  const [realtimeWords, setRealtimeWords] = useState([]);
  const [currentPhrase, setCurrentPhrase] = useState('');
  const [isShowingWords, setIsShowingWords] = useState(false);

  
  const [isFinalPhrase, setIsFinalPhrase]   = useState(false);


 // 🆕 TTS 관련 상태 추가
 const [ttsStatus, setTtsStatus] = useState('idle'); // idle, generating, ready, playing, done, error
 const [showReply, setShowReply] = useState(false);



// 🆕 TTS 관련 상태 추가
const [ttsVolume, setTtsVolume] = useState(0);
const [isTtsPlaying, setIsTtsPlaying] = useState(false);


// 기존 TTS 상태들 다음에 추가
const [ttsSubtitle, setTtsSubtitle] = useState(null); // 자막 데이터
const [currentTtsTime, setCurrentTtsTime] = useState(0); // 현재 재생 시간
const [currentWordIndex, setCurrentWordIndex] = useState(-1); // 현재 재생 중인 단어 인덱스






// 🆕 TTS 자막 데이터 구독
useEffect(() => {
  const ttsSubtitleListener = new ROSLIB.Topic({
    ros: ros,
    name: '/tts_subtitle',
    messageType: 'std_msgs/String'
  });
  
  ttsSubtitleListener.subscribe((message) => {
    try {
      const subtitleData = JSON.parse(message.data);
      console.log('📝 TTS 자막 데이터 수신:', subtitleData);
      setTtsSubtitle(subtitleData);
      setCurrentWordIndex(-1); // 초기화
      setCurrentTtsTime(0); // 시간 초기화
    } catch (e) {
      console.error('TTS 자막 JSON parse error:', e);
    }
  });
  
  return () => {
    ttsSubtitleListener.unsubscribe();
  };
}, []);



// 🆕 TTS 노래방 자막 렌더링 함수
const renderTtsKaraokeSubtitle = () => {
  if (!isTtsPlaying || !ttsSubtitle || !ttsSubtitle.words || ttsSubtitle.words.length === 0) {
    return null;
  }

  return (
    <div style={{
      position: 'absolute',
      top: '0',
      left: '0',
      width: '100vw',
      height: '100vh',
      zIndex: 30,
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      alignItems: 'center',
      backgroundColor: 'rgba(26, 26, 26, 0.95)'
    }}>
      {/* 🆕 메인 자막 영역 */}
      <div style={{
        maxWidth: '90vw',
        textAlign: 'center',
        padding: '40px 20px'
      }}>
        {/* 🆕 단어별 노래방 스타일 렌더링 */}
        <div style={{
          fontSize: '3.5rem',
          fontWeight: 'bold',
          lineHeight: '1.4',
          letterSpacing: '0.05em',
          display: 'flex',
          flexWrap: 'wrap',
          justifyContent: 'center',
          alignItems: 'center',
          gap: '0.3em'
        }}>
          {ttsSubtitle.words.map((wordInfo, index) => {
            const isActive = index === currentWordIndex;
            const isPast = index < currentWordIndex;
            const isFuture = index > currentWordIndex;
            
            return (
              <span
                key={index}
                style={{
                  color: isActive ? '#FFD700' : isPast ? '#FFFFFF' : '#888888',
                  textShadow: isActive 
                    ? '0 0 30px rgba(255, 215, 0, 0.9), 3px 3px 8px rgba(0,0,0,0.8)' 
                    : isPast 
                      ? '2px 2px 6px rgba(0,0,0,0.8)' 
                      : '2px 2px 6px rgba(0,0,0,0.6)',
                  fontSize: isActive ? '4rem' : '3.5rem',
                  transform: isActive ? 'scale(1.1) translateY(-10px)' : 'scale(1) translateY(0px)',
                  transition: 'all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1)',
                  display: 'inline-block',
                  marginRight: '0.2em',
                  marginBottom: '0.1em',
                  // 🆕 활성 단어 추가 효과
                  ...(isActive && {
                    animation: 'pulse 0.8s infinite ease-in-out',
                    background: 'linear-gradient(45deg, rgba(255, 215, 0, 0.1), rgba(255, 165, 0, 0.1))',
                    padding: '5px 10px',
                    borderRadius: '8px',
                    border: '2px solid rgba(255, 215, 0, 0.3)'
                  }),
                  // 🆕 신뢰도가 낮은 단어 표시
                  ...(wordInfo.confidence < 0.8 && {
                    borderBottom: '2px dotted rgba(255, 255, 255, 0.5)'
                  })
                }}
              >
                {wordInfo.word}
              </span>
            );
          })}
        </div>
        
        {/* 🆕 진행률 바 */}
        <div style={{
          width: '80%',
          height: '6px',
          backgroundColor: 'rgba(255, 255, 255, 0.2)',
          margin: '30px auto 20px',
          borderRadius: '3px',
          overflow: 'hidden',
          boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.3)'
        }}>
          <div style={{
            width: `${ttsSubtitle.total_duration > 0 ? (currentTtsTime / ttsSubtitle.total_duration) * 100 : 0}%`,
            height: '100%',
            background: 'linear-gradient(90deg, #FFD700, #FFA500)',
            transition: 'width 0.1s linear',
            borderRadius: '3px'
          }} />
        </div>

        {/* 🆕 시간 및 단어 정보 (선택적) */}
        <div style={{
          color: '#CCCCCC',
          fontSize: '1rem',
          marginTop: '15px',
          opacity: 0.8
        }}>
          <div>
            {currentTtsTime.toFixed(1)}s / {ttsSubtitle.total_duration?.toFixed(1) || '0.0'}s
          </div>
          {currentWordIndex >= 0 && (
            <div style={{ marginTop: '5px', fontSize: '0.9rem' }}>
              현재: "{ttsSubtitle.words[currentWordIndex]?.word}" 
              ({currentWordIndex + 1}/{ttsSubtitle.words.length})
            </div>
          )}
        </div>
      </div>

      {/* 🆕 CSS 애니메이션 정의 */}
      <style>
        {`
          @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.8; }
          }
        `}
      </style>
    </div>
  );
};





// 🆕 개선된 활성 단어 찾기 로직
const findActiveWordWithTolerance = (currentTime, words) => {
  const TIMING_TOLERANCE = 0.2; // 200ms 허용 오차
  const CONFIDENCE_THRESHOLD = 0.6; // 신뢰도 임계값
  
  // 1. 정확한 시간 범위 내 단어 찾기 (신뢰도 높은 단어 우선)
  let candidates = words.map((word, index) => {
    const isInRange = currentTime >= (word.start - TIMING_TOLERANCE) && 
                     currentTime <= (word.end + TIMING_TOLERANCE);
    const distance = Math.abs(currentTime - (word.start + word.end) / 2);
    const confidence = word.confidence || 1.0;
    
    return {
      index,
      distance,
      confidence,
      isInRange,
      word
    };
  }).filter(candidate => candidate.isInRange);
  
  if (candidates.length === 0) {
    // 2. 범위 내 단어가 없으면 가장 가까운 단어 찾기
    candidates = words.map((word, index) => ({
      index,
      distance: Math.abs(currentTime - (word.start + word.end) / 2),
      confidence: word.confidence || 1.0,
      word
    }));
  }
  
  // 3. 신뢰도와 거리를 종합적으로 고려하여 최적 단어 선택
  candidates.sort((a, b) => {
    // 높은 신뢰도와 가까운 거리를 우선시
    const scoreA = (a.confidence * 2) - (a.distance * 0.5);
    const scoreB = (b.confidence * 2) - (b.distance * 0.5);
    return scoreB - scoreA;
  });
  
  const bestMatch = candidates[0];
  
  // 너무 멀거나 신뢰도가 낮으면 -1 반환
  if (bestMatch.distance > 1.0 && bestMatch.confidence < CONFIDENCE_THRESHOLD) {
    return -1;
  }
  
  return bestMatch.index;
};

// TTS 시간 구독에서 사용
useEffect(() => {
  const ttsTimeListener = new ROSLIB.Topic({
    ros: ros,
    name: '/tts_spectrum',
    messageType: 'std_msgs/String'
  });
  
  ttsTimeListener.subscribe((message) => {
    try {
      const data = JSON.parse(message.data);
      
      if (data.current_time !== undefined && data.status === 'playing') {
        setCurrentTtsTime(data.current_time);
        
        if (ttsSubtitle && ttsSubtitle.words) {
          // 🆕 개선된 단어 찾기 사용
          const activeWordIndex = findActiveWordWithTolerance(
            data.current_time,
            ttsSubtitle.words
          );
          
          if (activeWordIndex !== currentWordIndex) {
            setCurrentWordIndex(activeWordIndex);
            
            // 🆕 디버깅 정보
            if (activeWordIndex >= 0) {
              const word = ttsSubtitle.words[activeWordIndex];
              console.log(`🎯 단어 활성화: "${word.word}" (${word.start}s-${word.end}s, 신뢰도: ${word.confidence?.toFixed(2) || 'N/A'}, 매칭: ${word.match_type || 'unknown'})`);
            }
          }
        }
      }
    } catch (e) {
      console.error('TTS time JSON parse error:', e);
    }
  });
  
  return () => ttsTimeListener.unsubscribe();
}, [ttsSubtitle, currentWordIndex]);





















// 🆕 TTS 음량 스무딩 함수
const previousTtsVolumeRef = useRef(0);

const applyTtsVolumeSmoothing = (newVolume) => {
  const previous = previousTtsVolumeRef.current;
  
  let smoothed;
  if (newVolume > previous) {
    // 음성 증가: 빠른 반응 (70% 새값)
    smoothed = previous * 0.3 + newVolume * 0.7;
  } else {
    // 음성 감소: 자연스러운 감쇠 (40% 새값)
    smoothed = previous * 0.6 + newVolume * 0.4;
  }
  
  previousTtsVolumeRef.current = smoothed;
  return smoothed;
};
















// 🆕 TTS 상태 구독
useEffect(() => {
  const ttsStatusListener = new ROSLIB.Topic({
    ros: ros,
    name: '/tts_status',
    messageType: 'std_msgs/String'
  });

  ttsStatusListener.subscribe((message) => {
    console.log('🗣️ TTS 상태 변경:', message.data);
    setTtsStatus(message.data);
  });

  return () => {
    ttsStatusListener.unsubscribe();
  };
}, []);

// 🆕 TTS 재생 요청 퍼블리셔 생성
const ttsPlayPublisher = useRef(null);

useEffect(() => {
  if (!ttsPlayPublisher.current) {
    ttsPlayPublisher.current = new ROSLIB.Topic({
      ros: ros,
      name: '/tts_play_request',
      messageType: 'std_msgs/String'
    });
  }
}, []);






  

// 🆕 Mp3Recommender에서 오는 mp4 정보 구독
useEffect(() => {
  const mp4Listener = new ROSLIB.Topic({
      ros: ros,
      name: '/recommended_mp4', // Mp3Recommender에서 publish하는 토픽
      messageType: 'std_msgs/String'
  });

  mp4Listener.subscribe((message) => {
      console.log('🎬 MP4 추천 메시지 수신:', message.data);
      
      if (message.data && message.data.trim() !== "") {
          // Mp3Recommender에서 "file_name=xxx.mp4;reply=yyy" 형식으로 전송
          const parts = message.data.split(';');
          let fileName = '';
          let reply = '';
          
          parts.forEach(part => {
              if (part.startsWith('file_name=')) {
                  fileName = part.substring('file_name='.length);
              } else if (part.startsWith('reply=')) {
                  reply = part.substring('reply='.length);
              }
          });


          console.log('🔍 파싱된 파일명:', fileName);
          console.log('🔍 파싱된 응답:', reply);

          if (fileName && fileName !== 'unknown' && !fileName.includes('unknown')) {
          
              // mp4 파일 경로 생성 (Mp3Recommender의 mp4_dir 경로 사용)
              const videoPath = `/videos/${fileName}`;
              
              console.log('🎬 비디오 표시:', videoPath);
              console.log('🗣️ Reply 텍스트:', reply);
              
              setCurrentVideo(videoPath);
              setCurrentReply(reply);
              setVideoVisible(true);
              
              // 🆕 대기 모드 종료 (Mp3Recommender 비디오가 왔으므로)
              setWaitingImage(null);
              setWaitingImageVisible(false);
              setIsWaitingImageMode(false);
              setIsWaitingAudioMode(false);
              setIsMp3WaitingMode(false);

              // searching 상태도 해제
              if (recommendStatus === 'searching') {
                  setRecommendStatus('processing');
              }
              console.log('✅ 비디오 상태 업데이트 완료');
          } else {
              console.log('🎬 유효하지 않은 파일명 또는 unknown:', fileName);
          }
      } else {
          console.log('🎬 비디오 숨김');
          setCurrentVideo(null);
          setVideoVisible(false);
          setCurrentReply('');

          // 비디오가 숨김 상태가 되면 스펙트럼 시각화 시작
          if (musicPlaying) {
              console.log('🎵 비디오 숨김 완료 - 스펙트럼 시각화 시작');
              setCanShowSpectrum(true);
              setRecommendStatus('done');
          }
      }
  });

  return () => {
      console.log('🎬 MP4 리스너 해제');
      mp4Listener.unsubscribe();
  };
}, [musicPlaying, recommendStatus]);

// 🆕 비디오 렌더링 함수
const renderVideo = () => {
  console.log('🎬 renderVideo 호출:', {
    currentVideo,
    videoVisible,
    musicPlaying
  });

  if (!currentVideo || !videoVisible) {
    console.log('🎬 비디오 렌더링 조건 불만족');
      return null;
  }



  // App.jsx - renderVideo 안
const createSafeUrl = (path) => {
  try {
    // // 파일명만 인코딩
    // const lastSlash = path.lastIndexOf('/');
    // const dir = path.substring(0, lastSlash + 1);      // '/videos/'
    // const file = path.substring(lastSlash + 1);        // '파티분위기....mp4'
    // return dir + encodeURIComponent(file);

    const lastSlash = path.lastIndexOf('/');
  const dir  = path.slice(0, lastSlash + 1);   // "/videos/"
  const file = path.slice(lastSlash + 1);      // "why so long.mp4"
  return dir + encodeURIComponent(file);      // 디렉터리 부분은 인코딩 X


  } catch (e) {
    console.error('비디오 URL 생성 오류:', e);
    return path;
  }
};



  const safeVideoUrl = createSafeUrl(currentVideo);
  console.log('🎬 안전한 비디오 URL:', safeVideoUrl);





  return (
      <div style={{
          position: 'absolute',
          top: '0',
          left: '0',
          width: '100vw',
          height: '100vh',
          zIndex: 15,
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          backgroundColor: '#000'
      }}>
          <video 
          key={safeVideoUrl} 
          ref={videoRef}
          src={safeVideoUrl}
          autoPlay
        
      
          playsInline
          controls // 🔧 임시 디버깅용 컨트롤 추가
          style={{
            width: 'auto',
            height: '100vh',
            minWidth: '100vw',
            objectFit: 'cover',
            objectPosition: 'center'
          }}
          onLoadStart={() => {
            console.log('🎬 비디오 로드 시작:', safeVideoUrl);
          }}
          onLoadedMetadata={() => {
            console.log('🎬 비디오 메타데이터 로드 완료');
          }}
          onLoadedData={() => {
            console.log('🎬 비디오 데이터 로드 완료:', safeVideoUrl);
            if (videoRef.current) {
              videoRef.current.play().then(() => {
                console.log('✅ 비디오 자동재생 성공');
              }).catch(e => {
                console.error('❌ 비디오 자동재생 실패:', e);
              });
            }
          }}

         // 🆕 핵심 수정: 비디오 종료 후 TTS 재생 시퀀스
         onEnded={() => {
          console.log('🎬 비디오 재생 완료 - TTS 대기');
          setVideoVisible(false);
          
          // TTS가 준비된 경우 즉시 재생, 아니면 대기
          if (ttsStatus === 'tts_ready') {
            console.log('🗣️ TTS 준비 완료 - 즉시 재생');
            requestTtsPlay();
          } else {
            console.log('🗣️ TTS 준비 대기 중...');
            setShowReply(true); // TTS 대기 중 표시
          }
        }}





         
          onError={(e) => {
            console.error('🎬 비디오 로드 실패:', safeVideoUrl);
            console.error('🎬 에러 상세:', e.target.error);
          }}
        />
          
        













      </div>
  );
};


// 🆕 TTS 재생 요청 함수
const requestTtsPlay = () => {
  if (ttsPlayPublisher.current) {
    const msg = new ROSLIB.Message({
      data: 'play_tts'
    });
    ttsPlayPublisher.current.publish(msg);
    console.log('🗣️ TTS 재생 요청 전송');
  }
};













// 🆕 TTS 대기 중 표시 함수
const renderTtsWaiting = () => {
  if (!showReply || !currentReply) {
    return null;
  }

  return (
    <div style={{
      position: 'absolute',
      top: '0',
      left: '0',
      width: '100vw',
      height: '100vh',
      zIndex: 20,
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      backgroundColor: 'rgba(0, 0, 0, 0.8)'
    }}>
      <div style={{
        color: '#fff',
        fontSize: '1.5rem',
        textAlign: 'center',
        padding: '20px'
      }}>
        <div>음성을 준비하는 중...</div>
        <div style={{ 
          fontSize: '1rem', 
          marginTop: '10px',
          opacity: 0.7 
        }}>
          {currentReply}
        </div>
      </div>
    </div>
  );
};














  
//글자 길이별 속도 조절
const AnimatedWord = ({ word, index, totalWords }) => {
  const animationName = `float-${index % 5}`;
  
  // 🆕 단어별 지연 시간 증가 (0.3초 → 0.5초)
  const animationDelay = `0s`;
  
  // 🆕 단어 길이에 따른 애니메이션 속도 조정
  const wordLength = word.length;
  const baseDuration = 5 + Math.random() * 2;
  const animationDuration = `${baseDuration + (wordLength * 0.1)}s`;

  return (
    <span
      style={{
        position: 'absolute',
        fontSize: '4rem',
        fontWeight: 'bold',
        color: '#FFD700',
        textShadow: '3px 3px 6px rgba(0,0,0,0.8)',
        animation: `${animationName} ${animationDuration} ${animationDelay} infinite ease-in-out`,
        zIndex: 30,
        whiteSpace: 'nowrap',
        letterSpacing: '0.05em'
      }}
    >
      {word}
    </span>
  );
};







// 2. CSS 애니메이션 정의
const createAnimationStyles = () => {
  return `
    @keyframes float-0 {
      0%, 100% { 
        transform: translate(10vw, 20vh) rotate(0deg);
      }
      25% { 
        transform: translate(80vw, 10vh) rotate(90deg);
      }
      50% { 
        transform: translate(70vw, 80vh) rotate(180deg);
      }
      75% { 
        transform: translate(20vw, 70vh) rotate(270deg);
      }
    }

    @keyframes float-1 {
      0%, 100% { 
        transform: translate(90vw, 30vh) scale(1);
      }
      33% { 
        transform: translate(10vw, 60vh) scale(1.2);
      }
      66% { 
        transform: translate(50vw, 10vh) scale(0.8);
      }
    }

    @keyframes float-2 {
      0%, 100% { 
        transform: translate(50vw, 90vh);
        opacity: 1;
      }
      25% { 
        transform: translate(10vw, 30vh);
        opacity: 0.7;
      }
      50% { 
        transform: translate(90vw, 50vh);
        opacity: 1;
      }
      75% { 
        transform: translate(30vw, 10vh);
        opacity: 0.9;
      }
    }

    @keyframes float-3 {
      0% { transform: translate(20vw, 50vh) rotate(0deg); }
      20% { transform: translate(60vw, 20vh) rotate(72deg); }
      40% { transform: translate(80vw, 70vh) rotate(144deg); }
      60% { transform: translate(40vw, 85vh) rotate(216deg); }
      80% { transform: translate(15vw, 65vh) rotate(288deg); }
      100% { transform: translate(20vw, 50vh) rotate(360deg); }
    }

    @keyframes float-4 {
      0%, 100% { 
        transform: translate(50vw, 50vh);
        filter: blur(0px);
      }
      25% { 
        transform: translate(85vw, 15vh);
        filter: blur(1px);
      }
      50% { 
        transform: translate(15vw, 85vh);
        filter: blur(0px);
      }
      75% { 
        transform: translate(75vw, 75vh);
        filter: blur(0.5px);
      }
    }
  `;
};








  // 실시간 단어 구독 useEffect 추가
  useEffect(() => {
    const realtimeWordsListener = new ROSLIB.Topic({
      ros: ros,
      name: '/realtime_words',
      messageType: 'std_msgs/String'
    });

    realtimeWordsListener.subscribe((message) => {
      try {
        const data = JSON.parse(message.data);
        if (data.type === 'word_phrase') {
          console.log('📝 실시간 단어 수신:', data.phrase);
          
          // 새로운 구문으로 교체
          setCurrentPhrase(data.phrase);
          setIsFinalPhrase(!!data.is_final);   // ★ 추가
          setIsShowingWords(true);
          
          // 선택적: 단어 히스토리 관리 (필요시)
          setRealtimeWords(prev => [...prev.slice(-4), {
            phrase: data.phrase,
            timestamp: data.timestamp,
            id: Date.now()
          }]);
        }
      } catch (e) {
        console.error('실시간 단어 JSON 파싱 오류:', e);
      }
    });

    return () => realtimeWordsListener.unsubscribe();
  }, []);


    // 단어 표시 여부 결정 함수
  const shouldShowRealtimeWords = () => {
    // trigger_detected이고, 음악이 재생 중이 아니며, 검색 상태가 아닐 때
    return triggerDetected && 
          !musicPlaying && 
          recommendStatus !== 'searching' && 
          !isWaitingAudioMode && 
          !isWaitingImageMode && 
          !isMp3WaitingMode && 
          !videoVisible && 
          isShowingWords;
  };





const renderRealtimeWords = () => {
  if (!shouldShowRealtimeWords() || !currentPhrase) {
    return null;
  }

  // 최종 문장일 때만 단어별 애니메이션
  if (isFinalPhrase) {
    // 🔥 핵심 수정: 띄어쓰기 단위로 분리
    const words = currentPhrase.split(' ').filter(word => word.trim()); // 빈 문자열 제거
    
    return (
      <>
        {/* CSS 애니메이션 스타일 추가 */}
        <style>{createAnimationStyles()}</style>
        
        {/* 각 단어별 애니메이션 */}
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          zIndex: 20,
          pointerEvents: 'none'
        }}>
          {words.map((word, index) => (
            <AnimatedWord 
              key={`${word}-${index}`}
              word={word}
              index={index}
              totalWords={words.length}
            />
          ))}
        </div>
      </>
    );
  }

  // 일반 문장은 기존 방식 유지
  return (
    <div style={{
      position: 'fixed',
      top: '50%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
      zIndex: 20,
      textAlign: 'center',
      animation: 'fadeInScale 0.3s ease-out'
    }}>
      <div style={{
        fontSize: '5rem',
        fontWeight: 'bold',
        color: '#ffffff',
        textShadow: '3px 3px 6px rgba(0,0,0,0.8)',
        maxWidth: '90vw',
        wordBreak: 'keep-all',
        whiteSpace: 'nowrap',
        letterSpacing: '0.05em'
      }}>
        {currentPhrase}
      </div>
    </div>
  );
};











  // 🆕 trigger_detected 상태 구독
  useEffect(() => {
    const triggerListener = new ROSLIB.Topic({
        ros: ros,
        name: '/trigger_status',
        messageType: 'std_msgs/String'
    });

    triggerListener.subscribe((message) => {
        const isTriggered = message.data === "triggered";
        console.log('🎯 trigger_detected 상태:', isTriggered);
        setTriggerDetected(isTriggered);
    });

    return () => triggerListener.unsubscribe();
  }, []);

  // 🆕 스펙트럼 표시 조건 함수 (기존 trigger_detected 플래그 활용)
  const shouldShowSpectrum = () => {
    if (musicPlaying) {
      return canShowSpectrum; // 음악 재생 중일 때는 기존 로직
    } else {
      // 음악이 재생 중이 아닐 때는 trigger_detected에 따라 결정
      return triggerDetected && recommendStatus !== 'searching';
    }
  };





  // 🆕 실시간 각도 구독 (주석 해제 및 수정)
  useEffect(() => {
    const directionListener = new ROSLIB.Topic({
        ros: ros,
        name: '/sound_direction_angle',
        messageType: 'std_msgs/Float32'
    });

    directionListener.subscribe((message) => {
        const angle = message.data;
        setSoundDirection(angle);
        
        // 🆕 고정 모드가 아닐 때만 실시간 화면 방향 변경
        if (!isDirectionFixed) {
            if (angle >= 180 && angle <= 360) {
                setScreenFlipped(true);
                console.log(`🔄 실시간 화면 반전: ${angle}도`);
            } else {
                setScreenFlipped(false);
                console.log(`➡️ 실시간 정상 화면: ${angle}도`);
            }
        } else {
            console.log(`📍 실시간 각도: ${angle}도 (고정 모드: ${fixedDirection}도 유지)`);
        }
    });

    return () => directionListener.unsubscribe();
  }, [isDirectionFixed, fixedDirection]);






  // 🆕 고정 각도 토픽 구독 (수정)
  useEffect(() => {
    const fixedDirectionListener = new ROSLIB.Topic({
        ros: ros,
        name: '/fixed_direction',
        messageType: 'std_msgs/Float32'
    });

    fixedDirectionListener.subscribe((message) => {
        const fixedAngle = message.data;
        console.log('🔒 고정 각도 수신:', fixedAngle);
        
        setFixedDirection(fixedAngle);
        setIsDirectionFixed(true);
        
        // 고정 각도에 따른 화면 방향 설정
        if (fixedAngle >= 180 && fixedAngle <= 360) {
            setScreenFlipped(true);
            console.log(`🔒 화면 반전 고정: ${fixedAngle}도`);
        } else {
            setScreenFlipped(false);
            console.log(`🔒 정상 화면 고정: ${fixedAngle}도`);
        }
    });

    return () => fixedDirectionListener.unsubscribe();
  }, []);






  // 🆕 음악 상태 구독 (Mp3Player waiting 지원)
  useEffect(() => {
    const statusListener = new ROSLIB.Topic({
        ros: ros,
        name: '/music_status',
        messageType: 'std_msgs/String'
    });

    statusListener.subscribe((message) => {
        console.log('음악 상태 변경:', message.data);
        
        if (message.data === 'mp3_waiting_playing') {  // 새로운 상태 처리
            console.log('🎵 Mp3Player waiting 재생 시작');
            // waiting 모드는 스펙트럼 데이터가 오면 자동으로 시작됨
        } else if (message.data === 'music_playing') {
            setMusicPlaying(true);
            setCanShowSpectrum(false);
            setIsMp3WaitingMode(false); // Mp3 waiting 모드 종료
            
        } 

    });

    return () => {
        statusListener.unsubscribe();
    };
  }, []);






// TTS 상태 변화 감지 및 처리 (수정된 버전)
useEffect(() => {
  if (ttsStatus === 'tts_ready' && !videoVisible && showReply) {
    console.log('🗣️ TTS 준비 완료 - 재생 시작');
    setShowReply(false);
    setIsTtsPlaying(true);
    requestTtsPlay();
  } else if (ttsStatus === 'tts_playing') {
    setIsTtsPlaying(true);
  } else if (ttsStatus === 'tts_done') {
    console.log('🗣️ TTS 재생 완료 - 제한적 초기화');
    
    // ✅ TTS 관련 상태만 초기화 (비디오 상태는 보존)
    setShowReply(false);
    setIsTtsPlaying(false);
    setTtsVolume(0);
    previousTtsVolumeRef.current = 0;
    
    // ✅ TTS 자막 관련만 초기화
    setTtsSubtitle(null);
    setCurrentTtsTime(0);
    setCurrentWordIndex(-1);
    
    // ❌ 이 부분들을 제거 - Mp3Recommender가 관리하도록 함
    // setCurrentVideo(null);
    // setCurrentReply('');
    // setVideoVisible(false);
    // setRecommendStatus('done');
    
    // ✅ 새 질문을 위한 최소한의 초기화만
    setCanShowSpectrum(false);
    setMusicPlaying(false);
    
    // 기타 대기 상태들 초기화
    setWaitingImage(null);
    setWaitingImageVisible(false);
    setIsWaitingImageMode(false);
    setIsWaitingAudioMode(false);
    setIsMp3WaitingMode(false);
    setIsTransitioning(false);
    
    // 실시간 단어 상태 초기화
    setRealtimeWords([]);
    setCurrentPhrase('');
    setIsShowingWords(false);
    setIsFinalPhrase(false);
    
    // 방향 관련 상태 초기화
    setIsDirectionFixed(false);
    setFixedDirection(null);
    
    console.log('✅ TTS 완료 후 제한적 상태 초기화 - 비디오 상태 보존');
  }
}, [ttsStatus, videoVisible, showReply]);









  // 🆕 이전 스펙트럼 값을 저장하는 ref
  const previousSpectrumRef = useRef([]);
  
  // 🆕 향상된 스무딩 함수
  const applyAdvancedSmoothing = (newSpectrum) => {
    const previous = previousSpectrumRef.current;
    
    if (previous.length === 0) {
      previousSpectrumRef.current = [...newSpectrum];
      return newSpectrum;
    }

    const smoothed = newSpectrum.map((current, index) => {
      const prev = previous[index] || 0;
      
      // 🆕 음악 특성에 맞는 스무딩
      if (current > prev) {
        // 비트/드럼 등 급격한 증가: 빠른 반응 (40% 새값)
        return prev * 0.4 + current * 0.6;
      } else {
        // 소리 감소: 자연스러운 감쇠 (8% 새값)
        return prev * 0.6 + current * 0.4;
      }
    });
    
    // 이전 값 업데이트
    previousSpectrumRef.current = [...smoothed];
    return smoothed;
  };



  //==========================================================

  // 🆕 마이크용 별도 스무딩 함수
  const previousMicSpectrumRef = useRef([]);

  const applyMicSmoothing = (newSpectrum) => {
    const previous = previousMicSpectrumRef.current;
    
    if (previous.length === 0) {
      previousMicSpectrumRef.current = [...newSpectrum];
      return newSpectrum;
    }

    // 마이크용 다른 스무딩 설정 (예시)
    const smoothed = newSpectrum.map((current, index) => {
      const prev = previous[index] || 0;
      
      if (current > prev) {
        return prev * 0.6 + current * 0.4; // 더 빠른 반응
      } else {
        return prev * 0.4 + current * 0.6; // 더 빠른 감쇠
      //   return prev * 0.5 + current * 0.5; // 더 빠른 반응
      // } else {
      //   return prev * 0.8 + current * 0.2; // 더 빠른 감쇠

      }
    });
    
    previousMicSpectrumRef.current = [...smoothed];
    return smoothed;
  };
  //==========================================================

  // 🆕 Mp3Player waiting 전용 스무딩 함수
  const previousMp3WaitingSpectrumRef = useRef([]);

  const applyMp3WaitingSmoothing = (newSpectrum) => {
    const previous = previousMp3WaitingSpectrumRef.current;
    
    if (previous.length === 0) {
      previousMp3WaitingSpectrumRef.current = [...newSpectrum];
      return newSpectrum;
    }

    // Mp3Player waiting 전용 스무딩 설정 (더 부드럽고 독특한 특성)
    const smoothed = newSpectrum.map((current, index) => {
      const prev = previous[index] || 0;
      
      if (current > prev) {
        // 빠른 상승 반응 (60% 새값으로 더 빠르게)
        return prev * 0.4 + current * 0.6;
      } else {
        // 느린 감쇠 (20% 새값으로 더 부드럽게)
        return prev * 0.8 + current * 0.2;
      }
    });
    
    previousMp3WaitingSpectrumRef.current = [...smoothed];
    return smoothed;
  };

  //==========================================================





  // 🆕 음악 스펙트럼 구독
  useEffect(() => {
    if (!musicPlaying) return;
    
    const musicSpectrumListener = new ROSLIB.Topic({
      ros: ros,
      name: '/audio_amplitude',
      messageType: 'std_msgs/String'
    });
    
    musicSpectrumListener.subscribe((message) => {
      try {
        const data = JSON.parse(message.data);
        if (data.spectrum) {
          const smoothedData = applyAdvancedSmoothing(data.spectrum);
          setMusicSpectrum(smoothedData);
          // setMusicSpectrum(data.spectrum);
        }
      } catch (e) {
        console.error('Music spectrum JSON parse error:', e);
      }
    });
    
    return () => {
      musicSpectrumListener.unsubscribe();
    //  previousSpectrumRef.current = [];
    };
  }, [musicPlaying]);


  // 🆕 대기 이미지 구독 (/waiting_image)
  useEffect(() => {
    const waitingImageListener = new ROSLIB.Topic({
      ros: ros,
      name: '/waiting_image',
      messageType: 'std_msgs/String'
    });
    
    waitingImageListener.subscribe((message) => {
      console.log('🖼️ 대기 이미지 메시지 수신:', message.data);
      
      if (message.data && message.data.trim() !== "") {
        const imagePath = message.data;
        console.log('🖼️ 대기 이미지 표시:', imagePath);
        setWaitingImage(imagePath);
        setWaitingImageVisible(true);
        setIsWaitingAudioMode(false); // 오디오 모드 종료
        setIsWaitingImageMode(true);  // 이미지 모드 시작
        setRecommendStatus('waiting_image'); // 대기 이미지 상태
      } else {
        console.log('🖼️ 대기 이미지 숨김');
        // 🆕 전환 상태 활성화
        setIsTransitioning(true);
        setWaitingImage(null);
        setWaitingImageVisible(false);
        setIsWaitingImageMode(false);
        setIsWaitingAudioMode(false);
        // 대기 시퀀스 완료 후 정상 상태로 복귀하지만 searching은 유지
        // Mp3Recommender 이미지가 올 때까지 대기
        // 🆕 Mp3Player waiting 모드를 미리 활성화
        setIsMp3WaitingMode(true);


        // // 🆕 Mp3Player waiting 모드를 미리 활성화하여 마이크 스펙트럼 방지
        // setIsMp3WaitingMode(true);
        // console.log('🔄 Mp3Player waiting 모드 예비 활성화 - 마이크 스펙트럼 차단');

        // 🆕 잠시 후 전환 상태 해제 (Mp3 스펙트럼이 도착할 시간 확보)
      setTimeout(() => {
        setIsTransitioning(false);
        console.log('🔄 전환 상태 해제 - Mp3Player waiting 준비 완료');
      }, 100); // 100ms 딜레이
    }

      
    });


    return () => {
      console.log('🖼️ 대기 이미지 리스너 해제');
      waitingImageListener.unsubscribe();
    };
  }, []);


    
    
  //==========================================================


    // 🆕 Mp3Player waiting 스펙트럼 구독 (/mp3_waiting_spectrum)
    useEffect(() => {
      const mp3WaitingSpectrumListener = new ROSLIB.Topic({
        ros: ros,
        name: '/mp3_waiting_spectrum',
        messageType: 'std_msgs/String'
      });
      
      mp3WaitingSpectrumListener.subscribe((message) => {
        try {
          const data = JSON.parse(message.data);
          if (data.spectrum) {
            console.log('🎵 Mp3Player waiting 스펙트럼 수신 - 독립 렌더링');
  
            // 🆕 전환 상태 해제 (Mp3 스펙트럼이 정상적으로 도착했으므로)
            setIsTransitioning(false);
            
            // 🆕 다른 모드들 완전히 비활성화
            setIsMp3WaitingMode(true);
            setIsWaitingImageMode(false);
            setIsWaitingAudioMode(false);
            setMusicPlaying(false);
  
  
            
            // 🆕 Mp3Player 전용 스무딩 적용
            // const smoothedData = applyMp3WaitingSmoothing(data.spectrum);
            const smoothedData = applyAdvancedSmoothing(data.spectrum);
            
            setMp3WaitingSpectrum(smoothedData);
          }
        } catch (e) {
          console.error('Mp3Player waiting spectrum JSON parse error:', e);
        }
      });
      
      return () => {
        mp3WaitingSpectrumListener.unsubscribe();
        // 🆕 Mp3Player 전용 스무딩 상태 초기화
        previousMp3WaitingSpectrumRef.current = [];
      };
    }, []);
  

    




 //--------------------------------------------------------------------------------------------------------- 
 // 🆕 tts 전체음량 기반 원형 스펙트럼 렌더링 
 //--------------------------------------------------------------------------------------------------------- 

//  useEffect(() => {
//   if (!isTtsPlaying) return;
  
//   const canvas = canvasRef.current;
//   if (!canvas) return;
  
//   const ctx = canvas.getContext('2d');
//   const { width, height } = canvasSize;
  
//   canvas.width = width;
//   canvas.height = height;
  
//   // TTS용 배경 (어두운 배경)
//   ctx.fillStyle = '#1a1a1a';
//   ctx.fillRect(0, 0, width, height);
  
//   // 🆕 원형 스펙트럼 설정
//   const centerX = width / 2;
//   const centerY = height / 2;
//   const baseRadius = Math.min(width, height) * 0.1;     // 기본 원 크기
//   const maxRadius = Math.min(width, height) * 0.4;      // 최대 원 크기
  
//   // 🆕 전체 음량에 비례한 원의 크기 계산
//   const volumeRadius = baseRadius + (ttsVolume * (maxRadius - baseRadius));
  
//   // 🆕 TTS용 스타일 (금색 계열, 그라데이션)
//   const gradient = ctx.createRadialGradient(
//     centerX, centerY, baseRadius,
//     centerX, centerY, volumeRadius
//   );
//   gradient.addColorStop(0, 'rgba(255, 215, 0, 0.8)');  // 금색 중심
//   gradient.addColorStop(0.7, 'rgba(255, 165, 0, 0.6)'); // 주황색 중간
//   gradient.addColorStop(1, 'rgba(255, 215, 0, 0.2)');   // 금색 외곽 (투명)

//   // 🆕 메인 원 그리기 (채워진 원)
//   ctx.beginPath();
//   ctx.arc(centerX, centerY, volumeRadius, 0, 2 * Math.PI);
//   ctx.fillStyle = gradient;
//   ctx.fill();
  
//   // 🆕 외곽선 그리기 (더 강조)
//   ctx.beginPath();
//   ctx.arc(centerX, centerY, volumeRadius, 0, 2 * Math.PI);
//   ctx.strokeStyle = `rgba(255, 215, 0, ${0.5 + ttsVolume * 0.5})`; // 음량에 따른 투명도
//   ctx.lineWidth = 3;
//   ctx.stroke();
  
//   // 🆕 중앙 점 표시 (TTS 재생 중임을 나타냄)
//   ctx.beginPath();
//   ctx.arc(centerX, centerY, 8, 0, 2 * Math.PI);
//   ctx.fillStyle = '#FFD700';
//   ctx.fill();
  
//   // 🆕 음량 표시 텍스트 (디버깅용 - 원하면 제거)
//   ctx.fillStyle = '#FFFFFF';
//   ctx.font = '16px Arial';
//   ctx.textAlign = 'center';
//   ctx.fillText(
//     `음량: ${(ttsVolume * 100).toFixed(1)}%`, 
//     centerX, 
//     centerY + volumeRadius + 30
//   );
  
// }, [ttsVolume, isTtsPlaying, canvasSize]);

    
  //==========================================================


  // 🆕 음악 스펙트럼 렌더링
useEffect(() => {
  if (!musicPlaying || musicSpectrum.length === 0) return;
  
  const canvas = canvasRef.current;
  if (!canvas) return;
  
  const ctx = canvas.getContext('2d');
  const { width, height } = canvasSize;
  
  canvas.width = width;
  canvas.height = height;
  
  // 음악용 배경
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, width, height);
  
  if (!canShowSpectrum) return;
  
  // 🆕 음악용 스펙트럼 처리 (기존 로직 유지)
  // const central = getCentralSlice(musicSpectrum, 0.6);
  const numBars = 64;

  const limitedSpectrum = musicSpectrum.slice(0,200); // 0~300 인덱스 추출
  let bars = downsampleArray(limitedSpectrum, numBars);

 // let bars = downsampleArray(musicSpectrum, numBars);
  bars = bars.map(v => Math.min(1, v * 0.0000005));
  


   // 🆕 원형 스펙트럼 설정
   const centerX = width / 2;
   const centerY = height / 2;
   const baseRadius = Math.min(width, height) * 0.2;    // 기본 원 반지름
   const maxBarLength = Math.min(width, height) * 0.15; // 최대 바 길이
 
   // 🆕 원형 스펙트럼 스타일 (검정색)
   ctx.strokeStyle = '#000';
   ctx.lineWidth = 4; // 바 두께
   ctx.lineCap = 'round';
   ctx.lineJoin = 'round';
 
   // 🆕 원형 스펙트럼 그리기
   for (let i = 0; i < numBars; i++) {
     const angle = (i / numBars) * 2 * Math.PI; // 각도 계산
     const barLength = bars[i] * maxBarLength;  // 바 길이
     const radius = baseRadius + barLength;     // 최종 반지름
     
     // 바의 시작점 (기본 원 위의 점)
     const startX = centerX + Math.cos(angle) * baseRadius;
     const startY = centerY + Math.sin(angle) * baseRadius;
     
     // 바의 끝점 (확장된 반지름의 점)
     const endX = centerX + Math.cos(angle) * radius;
     const endY = centerY + Math.sin(angle) * radius;
 
     // 바 그리기
     ctx.beginPath();
     ctx.moveTo(startX, startY);
     ctx.lineTo(endX, endY);
     ctx.stroke();


  }
}, [musicSpectrum, musicPlaying, canvasSize, canShowSpectrum]);





    
    
  //==========================================================


  // 🆕 마이크 스펙트럼 렌더링
  useEffect(() => {
    
    // if (musicPlaying || micSpectrum.length === 0 || isWaitingAudioMode) return;
    // 🆕 Mp3Player waiting 모드도 마이크 스펙트럼 비활성화 조건에 추가
    if (musicPlaying || micSpectrum.length === 0 || isWaitingAudioMode || isMp3WaitingMode  || isWaitingImageMode || isTransitioning|| videoVisible) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const { width, height } = canvasSize;
    
    canvas.width = width;
    canvas.height = height;
    
    // 마이크용 배경
    ctx.fillStyle = '#222222';
    ctx.fillRect(0, 0, width, height);
    
    if (recommendStatus === 'searching' && !videoVisible) {
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, width, height);
      return;
    }
    
    if (!triggerDetected) {
      console.log('🎵 마이크 모드 - trigger_detected=false, 배경색만 표시');
      return;
    }
    
    if (recommendStatus === 'searching') {
      console.log('🎵 마이크 모드 - 검색 중이므로 스펙트럼 숨김');
      return;
    }


    // 🆕 추가 차단 조건들 - 모든 다른 모드에서 마이크 스펙트럼 완전 차단
    if (isWaitingAudioMode) {
      console.log('🎵 마이크 모드 - UserQuestion 대기 스펙트럼 중이므로 차단');
      return;
    }

    if (isWaitingImageMode || waitingImageVisible) {
      console.log('🎵 마이크 모드 - 대기 이미지 표시 중이므로 차단');
      return;
    }

    if (isMp3WaitingMode) {
      console.log('🎵 마이크 모드 - Mp3Player 대기 스펙트럼 중이므로 차단');
      return;
    }

    if (musicPlaying || canShowSpectrum) {
      console.log('🎵 마이크 모드 - 음악 재생/스펙트럼 중이므로 차단');
      return;
    }

    if (videoVisible) {
      console.log('🎵 마이크 모드 - 이미지 표시 중이므로 차단');
      return;
    }

    if (isTransitioning) {
      console.log('🎵 마이크 모드 - 상태 전환 중이므로 차단');
      return;
    }






    
    // 🆕 마이크용 스펙트럼 처리 (나중에 다르게 커스터마이징 가능)
    // const central = getCentralSlice(micSpectrum, 0.6);
    const numBars = 20;
    const limitedSpectrum = micSpectrum.slice(80, 180); // 0~300 인덱스 추출
let bars = downsampleArray(limitedSpectrum, numBars);
  // let bars = downsampleArray(micSpectrum, numBars);


// bars = bars.map(v => v * 0.0000035);

// bars = bars.map(v => v * 0.0005);
// 🆕 바 값 정규화로 더 안정적인 높이 제어
bars = bars.map(v => {
  const normalizedValue = v * 0.00035;
  return Math.min(7, normalizedValue); // 최대값 1로 제한
});


// // 최대값 2로 제한
// bars = bars.map(v => Math.min(1, v * 0.00005));
    





const availableWidth = width * 0.9; // 화면 너비의 90% 사용
const totalGaps = (numBars - 1);
    
    const scale = Math.min(width / 2560, height / 1600);
    // const barWidth = 10 * scale;
    const barWidth = Math.max(2, availableWidth / (numBars + totalGaps * 0.5));
    // const gap = 14 * scale;
    const gap = Math.max(1, barWidth * 0.5); // 바 너비의 30%를 간격으로
  // 🆕 바 두께 별도 설정 (가로길이는 기존 barWidth 유지)
  const barThickness = barWidth * 0.8; // 바 두께를 기존의 60%로 설정

      
     const maxBarHeight = 100 * scale;

     
    // const maxBarHeight = height * 0.3;
    // const totalWidth = numBars * barWidth + (numBars - 1) * gap;
    const totalWidth = numBars * barWidth + totalGaps * gap;
    const xOffset = (width - totalWidth) / 2;
    const centerY = height / 2;





    
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = barThickness;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

  
    for (let i = 0; i < numBars; i++) {
      const x = xOffset + i * (barWidth + gap) + barWidth / 2;
      const barHeight = bars[i] * maxBarHeight;
      ctx.beginPath();
      ctx.moveTo(x, centerY - barHeight);
      ctx.lineTo(x, centerY + barHeight);
      ctx.stroke();
    }





    
  }, [micSpectrum, musicPlaying, recommendStatus, canvasSize, triggerDetected, videoVisible, isWaitingImageMode, isTransitioning]);

 




//===============================================================================================



  // 🆕 UserQuestion 대기 스펙트럼 렌더링 (개선된 부드러운 버전) - UserQuestion 노드
  useEffect(() => {
    if (!isWaitingAudioMode || waitingSpectrum.length === 0) return;

    
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const { width, height } = canvasSize;
    
    canvas.width = width;
    canvas.height = height;
    
      // 🆕 배경색 흰색으로 변경
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, width, height);
    // // 🆕 잔상 효과를 위한 반투명 배경 (완전히 지우지 않음)
    // ctx.fillStyle = 'rgba(255, 255, 255, 0.5)'; // 30% 투명도로 이전 프레임을 서서히 지움
    // ctx.fillRect(0, 0, width, height);
    
    // // 원형 스펙트럼 설정
    // const centerX = width / 2;
    // const centerY = height / 2;
    // const screenSize = Math.min(width, height);
    // const baseRadius = Math.min(width, height) * 0.2;
    // const maxBarLength = Math.min(width, height) * 0.15;
    
    // 스펙트럼 데이터 처리
    // const central = getCentralSlice(waitingSpectrum, 0.6);
    
    
    const numBars = 64;
    const limitedSpectrum = waitingSpectrum.slice(10, 200); // 0~300 인덱스 추출
    let bars = downsampleArray(limitedSpectrum, numBars);
    bars = bars.map(v => Math.min(1, v * 0.0000005));
    

    


// 🆕 원형 스펙트럼 설정
const centerX = width / 2;
const centerY = height / 2;
const baseRadius = Math.min(width, height) * 0.2;    // 기본 원 반지름
const maxBarLength = Math.min(width, height) * 0.15; // 최대 바 길이

// 🆕 원형 스펙트럼 스타일 (검정색)
ctx.strokeStyle = '#000';
ctx.lineWidth = 4; // 바 두께
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

// 🆕 원형 스펙트럼 그리기
for (let i = 0; i < numBars; i++) {
  const angle = (i / numBars) * 2 * Math.PI; // 각도 계산
  const barLength = bars[i] * maxBarLength;  // 바 길이
  const radius = baseRadius + barLength;     // 최종 반지름
  
  // 바의 시작점 (기본 원 위의 점)
  const startX = centerX + Math.cos(angle) * baseRadius;
  const startY = centerY + Math.sin(angle) * baseRadius;
  
  // 바의 끝점 (확장된 반지름의 점)
  const endX = centerX + Math.cos(angle) * radius;
  const endY = centerY + Math.sin(angle) * radius;

  // 바 그리기
  ctx.beginPath();
  ctx.moveTo(startX, startY);
  ctx.lineTo(endX, endY);
  ctx.stroke();
    
}
  }, [waitingSpectrum, isWaitingAudioMode, canvasSize]);






 //--------------------------------------------------------------------------------------------------------- 
 // 🆕 Mp3Player waiting 스펙트럼 렌더링 (현재는 UserQuestion과 동일, 나중에 커스터마이징 가능)
 //--------------------------------------------------------------------------------------------------------- 
  useEffect(() => {
    if (!isMp3WaitingMode  || mp3WaitingSpectrum.length === 0) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const { width, height } = canvasSize;
    
    canvas.width = width;
    canvas.height = height;
    
    // 🆕 배경색 흰색으로 변경
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, width, height);



    // // 🎨 Mp3Player waiting용 배경 (나중에 차별화 가능)
    // ctx.fillStyle = 'rgb(255, 255, 255)';
    // ctx.fillRect(0, 0, width, height);
    
    // // 원형 스펙트럼 설정 (UserQuestion waiting과 현재는 동일)
    // const centerX = width / 2;
    // const centerY = height / 2;
    // const baseRadius = Math.min(width, height) * 0.2;
    // const maxBarLength = Math.min(width, height) * 0.15;
    
    // // 스펙트럼 데이터 처리 (동일한 방식)
    // const central = getCentralSlice(mp3WaitingSpectrum, 0.6);
    const numBars = 64;
    const limitedSpectrum = mp3WaitingSpectrum.slice(10, 200); // 0~300 인덱스 추출
    let bars = downsampleArray(limitedSpectrum, numBars);
    bars = bars.map(v => Math.min(1, v * 0.0000005));



// 🆕 원형 스펙트럼 설정
const centerX = width / 2;
const centerY = height / 2;
const baseRadius = Math.min(width, height) * 0.2;    // 기본 원 반지름
const maxBarLength = Math.min(width, height) * 0.15; // 최대 바 길이

// 🆕 원형 스펙트럼 스타일 (검정색)
ctx.strokeStyle = '#000';
ctx.lineWidth = 4; // 바 두께
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

// 🆕 원형 스펙트럼 그리기
for (let i = 0; i < numBars; i++) {
  const angle = (i / numBars) * 2 * Math.PI; // 각도 계산
  const barLength = bars[i] * maxBarLength;  // 바 길이
  const radius = baseRadius + barLength;     // 최종 반지름
  
  // 바의 시작점 (기본 원 위의 점)
  const startX = centerX + Math.cos(angle) * baseRadius;
  const startY = centerY + Math.sin(angle) * baseRadius;
  
  // 바의 끝점 (확장된 반지름의 점)
  const endX = centerX + Math.cos(angle) * radius;
  const endY = centerY + Math.sin(angle) * radius;

  // 바 그리기
  ctx.beginPath();
  ctx.moveTo(startX, startY);
  ctx.lineTo(endX, endY);
  ctx.stroke();
    
}

    
  }, [mp3WaitingSpectrum, isMp3WaitingMode , canvasSize]);






  


  useEffect(() => {
    const updateCanvasSize = () => {
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;
  
      // 🔧 모든 모드에서 전체화면 사용 (musicPlaying 조건 제거)
      setCanvasSize({ 
        width: viewportWidth, 
        height: viewportHeight 
      });
    };
  
    updateCanvasSize();
    window.addEventListener('resize', updateCanvasSize);
    
    return () => window.removeEventListener('resize', updateCanvasSize);
  }, [isWaitingAudioMode, isMp3WaitingMode]);
  







// 🆕 renderImage 함수
const renderWaitingImage = () => {
  if (!waitingImage || !waitingImageVisible) {
    return null;
  }

  const createSafeUrl = (path) => {
    try {
      const lastSlashIndex = path.lastIndexOf('/');
      const directoryPath = path.substring(0, lastSlashIndex + 1);
      const fileName = path.substring(lastSlashIndex + 1);
      const encodedFileName = encodeURIComponent(fileName);
      return directoryPath + encodedFileName;
    } catch (e) {
      console.error("대기 이미지 URL 생성 중 오류 발생:", e);
      return path;
    }
  };

  const safeImageUrl = createSafeUrl(waitingImage);

  return (
    <div style={{
      position: 'absolute',
      top: '0',
      left: '0',
      width: '100vw',
      height: '100vh',
      zIndex: 25, // 높은 우선순위
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center'
    }}>
      <img 
        src={safeImageUrl} 
        alt="대기 중..."
        style={{
          width: 'auto',
          height: '100vh',
          minWidth: '100vw',
          objectFit: 'contain',
          objectPosition: 'center',
          borderRadius: '0px',
          boxShadow: 'none'     
        }}
        onLoad={() => console.log('🖼️ 대기 이미지 로드 성공:', safeImageUrl)}
        onError={() => console.error('🖼️ 대기 이미지 로드 실패:', safeImageUrl)}
      />
    </div>
  );
};












const getScreenTransform = () => {
  if (screenFlipped) {
    return 'scaleY(-1) scaleX(-1)'; // 상하반전
  }
  return 'scaleY(1) scaleX(1)'; // 정상
};



  







  return (
    <div style={{ 
      width: '100vw', 
      height: '100vh', 
      display: 'flex', 
      justifyContent: 'center', 
      alignItems: 'center',
      margin: 0,
      padding: 0,
      boxSizing: 'border-box',
      position: 'relative',
      // backgroundColor: (recommendStatus === 'searching' && !imageVisible) ? '#fff' : 
      //            (musicPlaying || imageVisible) ? '#000' : '#222222',
      // 🆕 대기 모드 고려한 배경색 로직
      backgroundColor: isTtsPlaying ? '#1a1a1a' :  (recommendStatus === 'searching' && !videoVisible && !isWaitingAudioMode) ? '#fff' : 
      (isWaitingAudioMode) ? '#fff' :
      (musicPlaying || videoVisible) ? '#000' : '#222222',

      // 🆕 화면 변환 적용
      transform: getScreenTransform(),
      transition: 'transform 0.5s ease-in-out' // 부드러운 전환 효과
    }}>

      {/* 🆕 방향 정보 표시 (디버깅용 - 원하면 제거) */}
      <div style={{
        position: 'absolute',
        top: '10px',
        right: '10px',
        color: screenFlipped ? '#ff6b6b' : '#4ecdc4',
        fontSize: '16px',
        fontWeight: 'bold',
        zIndex: 100,
        transform: screenFlipped ? 'scaleY(-1)' : 'scaleY(1)', // 텍스트는 정상 방향 유지
        backgroundColor: 'rgba(0,0,0,0.7)',
        padding: '5px 10px',
        borderRadius: '5px'
      }}>
        {/* {isDirectionFixed 
          ? `🔒 고정: ${Math.round(fixedDirection)}° ${screenFlipped ? '(반전)' : '(정상)'}`
          : `📍 실시간: ${Math.round(soundDirection)}° ${screenFlipped ? '(반전)' : '(정상)'}`
        } */}
      </div>

      







   {/* 캔버스 표시 조건 수정 */}
   {!videoVisible && !showReply && !waitingImageVisible && !shouldShowRealtimeWords() && !isTtsPlaying && (
      <canvas 
        ref={canvasRef}
        style={{
          width: '100vw',
          height: '100vh',
          position: 'fixed',
          top: '0',
          left: '0',
          zIndex: 10,
          border: 'none',
          outline: 'none',
          display: 'block',
          WebkitTapHighlightColor: 'transparent'
        }}
      />
    )}

    {/* 실시간 단어 표시 */}
    {renderRealtimeWords()}

    {/* 기존 이미지 표시 */}
    {videoVisible && renderVideo()}
    {/* 🆕 TTS 대기 중 표시 */}
    {renderTtsWaiting()}
    {renderWaitingImage()}

    {/* 🆕 TTS 노래방 자막 렌더링 추가 */}
    {renderTtsKaraokeSubtitle()}

    

  </div>
);
  
}

export default SpectrumVisualizer;





