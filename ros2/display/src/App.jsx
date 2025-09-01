

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

  const [isWaitingAudioMode, setIsWaitingAudioMode] = useState(false);


  // Mp3Player waiting 전용 상태 추가
  const [mp3WaitingSpectrum, setMp3WaitingSpectrum] = useState([]);
  const [isMp3WaitingMode , setIsMp3WaitingMode] = useState(false);


  // 기존 상태 변수들 다음에 추가
  const [isTransitioning, setIsTransitioning] = useState(false);






 // 🆕 TTS 관련 상태 추가
 const [ttsStatus, setTtsStatus] = useState('idle'); // idle, generating, ready, playing, done, error
 const [showReply, setShowReply] = useState(false);



// 🆕 TTS 관련 상태 추가
const [ttsVolume, setTtsVolume] = useState(0);
const [isTtsPlaying, setIsTtsPlaying] = useState(false);


// // 기존 TTS 상태들 다음에 추가
// const [ttsSubtitle, setTtsSubtitle] = useState(null); // 자막 데이터
// const [currentTtsTime, setCurrentTtsTime] = useState(0); // 현재 재생 시간
// const [currentWordIndex, setCurrentWordIndex] = useState(-1); // 현재 재생 중인 단어 인덱스

// // 기존 TTS 상태들 다음에 추가
// const [currentTtsVolume, setCurrentTtsVolume] = useState(0); // 🆕 실시간 음량 상태

// const [wordMaxVolumes, setWordMaxVolumes] = useState({}); // 🆕 각 단어별 최대 음량 저장


// 🆕 단일 단어 자막용 상태 추가
const [singleWordSubtitle, setSingleWordSubtitle] = useState(null);
const [showSingleWord, setShowSingleWord] = useState(false);
// 🆕 UserQuestion 소리크기 비례 원형 스펙트럼 관련 상태 추가
const [voiceVolume, setVoiceVolume] = useState(0);
const [isVoiceActive, setIsVoiceActive] = useState(false);



// 🆕 사용자 질문 표시용 상태 변수 추가
const [userQuestionText, setUserQuestionText] = useState('');
const [showUserQuestion, setShowUserQuestion] = useState(false);



// 🔧 Realtime 관련 상태 제거하고 질문 확인 TTS 상태로 교체
const [questionConfirmStatus, setQuestionConfirmStatus] = useState('idle'); // 'idle', 'playing', 'completed'
const [pendingVideo, setPendingVideo] = useState(null);
const [pendingReply, setPendingReply] = useState('');


// 대기 상태 통합 관리
const [pendingContent, setPendingContent] = useState(null); // { type: 'video'|'tts_only', videoPath?, reply }
const [waitingForQuestionConfirm, setWaitingForQuestionConfirm] = useState(false);




// 🆕 단일 단어 자막 구독
useEffect(() => {
  const singleWordListener = new ROSLIB.Topic({
    ros: ros,
    name: '/single_word_subtitle',
    messageType: 'std_msgs/String'
  });
  
  singleWordListener.subscribe((message) => {
    try {
      const data = JSON.parse(message.data);
      console.log('📺 단일 단어 자막 수신:', data);
      
      if (data.display_mode === 'single_word') {
        setSingleWordSubtitle(data);
        setShowSingleWord(true);
      } else if (data.display_mode === 'empty') {
        setSingleWordSubtitle(null);
        // setShowSingleWord은 유지 (다음 단어 준비)
      } else if (data.display_mode === 'finished') {
        setSingleWordSubtitle(null);
        setShowSingleWord(false);
      }
    } catch (e) {
      console.error('단일 단어 자막 JSON parse error:', e);
    }
  });
  
  return () => {
    singleWordListener.unsubscribe();
  };
}, []);










const renderSingleWordSubtitle = () => {
  if (!isTtsPlaying || !showSingleWord) {
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
      <div style={{
        maxWidth: '90vw',
        textAlign: 'center',
        padding: '40px 20px'
      }}>
        {/* 🔑 단일 단어 표시 - 고정 크기 */}
        <div style={{
          fontSize: '6rem', // 🔑 고정된 큰 크기
          fontWeight: 'bold',
          color: '#FFFFFF',
          textShadow: '3px 3px 6px rgba(0,0,0,0.8)',
          minHeight: '8rem', // 빈 공간에서도 높이 유지
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          letterSpacing: '0.05em'
        }}>
          {singleWordSubtitle ? singleWordSubtitle.word : ''}
        </div>
        
        
      </div>
    </div>
  );
};
























// 🔧 수정된 질문확인 상태 구독
useEffect(() => {
  const questionConfirmStatusListener = new ROSLIB.Topic({
    ros: ros,
    name: '/question_confirm_status',
    messageType: 'std_msgs/String'
  });
  
  questionConfirmStatusListener.subscribe((message) => {
    const newStatus = message.data;
    
    if (questionConfirmStatus !== newStatus) {
      console.log(`🎙️ 질문확인 TTS 상태 전환: ${questionConfirmStatus} → ${newStatus}`);
      setQuestionConfirmStatus(newStatus);
      
      // 🆕 질문확인 완료 시 대기 중인 컨텐츠 처리
      if (newStatus === 'completed') {
        console.log('✅ 질문확인 TTS 완료');
        
        if (waitingForQuestionConfirm && pendingContent) {
          console.log('🎬 대기 중인 컨텐츠 처리 시작');
          processPendingContent(pendingContent);
          
          // 대기 상태 초기화
          setPendingContent(null);
          setWaitingForQuestionConfirm(false);
        }
      }
    }
  });
  
  return () => {
    questionConfirmStatusListener.unsubscribe();
  };
}, [questionConfirmStatus, waitingForQuestionConfirm, pendingContent]);


 // 🆕 사용자 질문 표시용 구독 추가
 useEffect(() => {
  const userQuestionDisplayListener = new ROSLIB.Topic({
    ros: ros,
    name: '/user_question_display',
    messageType: 'std_msgs/String'
  });
  
  userQuestionDisplayListener.subscribe((message) => {
    console.log('🗨️ 사용자 질문 표시 데이터 수신:', message.data);
    
    if (message.data && message.data.trim() !== "") {
      setUserQuestionText(message.data.trim());
      setShowUserQuestion(true);
      console.log('✅ 사용자 질문 말풍선 표시:', message.data);
    } else {
      setUserQuestionText('');
      setShowUserQuestion(false);
      console.log('❌ 사용자 질문 말풍선 숨김');
    }
  });
  
  return () => {
    console.log('🗨️ 사용자 질문 표시 리스너 해제');
    userQuestionDisplayListener.unsubscribe();
  };
}, []);

// 🔧 수정된 renderUserQuestionBubble 함수
const renderUserQuestionBubble = () => {
  // 🆕 조건 완화: isWaitingAudioMode 조건 제거
  if (!showUserQuestion || !userQuestionText) {
    return null;
  }

  // 🆕 대기 상태 추가 고려
  if (isTtsPlaying || videoVisible || waitingForQuestionConfirm) {
    return null;
  }


  return (
    <div style={{
      position: 'absolute',
      top: '20%',
      left: '50%',
      transform: 'translateX(-50%)',
      zIndex: 40, // 🔧 z-index 상향 조정 (기존 35 → 40)
      maxWidth: '90vw',
      padding: '0',
      pointerEvents: 'none'
    }}>
      {/* 기존 말풍선 UI 코드 동일 */}
      <div style={{
        backgroundColor: 'rgba(255, 255, 255, 0.95)',
        color: '#333',
        padding: '30px 40px',
        borderRadius: '35px',
        fontSize: '2.2rem',
        fontWeight: '600',
        textAlign: 'center',
        boxShadow: '0 12px 48px rgba(0, 0, 0, 0.4)',
        border: '4px solid rgba(100, 200, 255, 0.8)',
        position: 'relative',
        maxWidth: '800px',
        wordWrap: 'break-word',
        lineHeight: '1.4',
        animation: 'bubbleAppear 0.3s ease-out'
      }}>
        {/* 말풍선 꼬리들과 내용은 기존과 동일 */}
        <div style={{
          position: 'absolute',
          bottom: '-20px',
          left: '50%',
          transform: 'translateX(-50%)',
          width: '0',
          height: '0',
          borderLeft: '20px solid transparent',
          borderRight: '20px solid transparent',
          borderTop: '20px solid rgba(255, 255, 255, 0.95)'
        }} />
        
        <div style={{
          position: 'absolute',
          bottom: '-24px',
          left: '50%',
          transform: 'translateX(-50%)',
          width: '0',
          height: '0',
          borderLeft: '27px solid transparent',
          borderRight: '27px solid transparent',
          borderTop: '27px solid rgba(100, 200, 255, 0.8)',
          zIndex: -1
        }} />
        
        <div style={{
          marginBottom: '10px'
        }}>
          "{userQuestionText}"
        </div>
        
        <div style={{
          fontSize: '1.4rem',
          color: '#666',
          fontWeight: '400'
        }}>
          이렇게 들었어!
        </div>
      </div>

      <style>
        {`
          @keyframes bubbleAppear {
            0% { 
              transform: translateX(-50%) translateY(-20px) scale(0.8);
              opacity: 0;
            }
            100% { 
              transform: translateX(-50%) translateY(0) scale(1);
              opacity: 1;
            }
          }
        `}
      </style>
    </div>
  );
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




// 🆕 대기 중인 비디오 재생 함수
const playPendingVideo = () => {
  if (!pendingVideo) return;
  
  console.log('🎬 대기 비디오 재생:', pendingVideo);
  setCurrentVideo(pendingVideo);
  setCurrentReply(pendingReply);
  setVideoVisible(true);
  
  // 대기 모드 종료
  setIsWaitingAudioMode(false);
  setIsMp3WaitingMode(false);
  
  if (recommendStatus === 'searching') {
    setRecommendStatus('processing');
  }
  
  // 대기 상태 초기화
  setPendingVideo(null);
  setPendingReply('');
};



// 🔧 수정된 Mp3Recommender 구독 부분
useEffect(() => {
  const mp4Listener = new ROSLIB.Topic({
    ros: ros,
    name: '/recommended_mp4',
    messageType: 'std_msgs/String'
  });

  mp4Listener.subscribe((message) => {
    console.log('🎬 MP4 메시지 수신:', message.data);
    
    if (message.data && message.data.trim() !== "") {
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

      // 🆕 질문확인 상태 확인 후 처리 방식 결정
      if (questionConfirmStatus === 'playing') {
        console.log('⏳ 질문확인 TTS 재생 중 - 컨텐츠 대기');
        
        if (fileName === 'no_video') {
          // TTS 전용 모드도 대기 처리
          setPendingContent({ 
            type: 'tts_only', 
            reply: reply 
          });
          setWaitingForQuestionConfirm(true);
        } else if (fileName && fileName !== 'unknown') {
          // 비디오 모드 대기 처리
          setPendingContent({ 
            type: 'video', 
            videoPath: `/videos/${fileName}`, 
            reply: reply 
          });
          setWaitingForQuestionConfirm(true);
        }
        
        if (recommendStatus === 'searching') {
          setRecommendStatus('processing');
        }
      } else {
        // 질문확인이 완료된 상태면 즉시 처리
        processPendingContent({ 
          type: fileName === 'no_video' ? 'tts_only' : 'video',
          videoPath: fileName !== 'no_video' ? `/videos/${fileName}` : null,
          reply: reply 
        });
      }
    }
  });

  return () => {
    mp4Listener.unsubscribe();
  };
}, [questionConfirmStatus, recommendStatus]);





// 🆕 질문확인 완료 처리 함수
const processPendingContent = (content) => {
  if (!content) return;
  
  console.log('🎬 컨텐츠 처리:', content);
  
  if (content.type === 'video') {
    // 비디오가 있는 경우: tts_ready까지 기다림
    console.log('🎬 비디오 모드 - TTS 준비 대기');
    setCurrentVideo(content.videoPath);
    setCurrentReply(content.reply);
    // 비디오는 tts_ready 상태에서 재생됨
    
  } else if (content.type === 'tts_only') {
    // 비디오가 없는 경우: 즉시 TTS 대기 상태로
    console.log('🗣️ TTS 전용 모드 - 즉시 TTS 대기');
    setVideoVisible(false);
    setCurrentVideo(null);
    setCurrentReply(content.reply);
    setShowReply(true);
  }
  
  // 공통 처리
  setIsWaitingAudioMode(false);
  setIsMp3WaitingMode(false);
  
  if (recommendStatus === 'searching') {
    setRecommendStatus('processing');
  }
};





// 🆕 즉시 비디오 재생 함수 추가
const playVideoImmediately = (videoPath, reply) => {
  console.log('🎬 즉시 비디오 재생:', videoPath);
  setCurrentVideo(videoPath);
  setCurrentReply(reply);
  setVideoVisible(true);
  
  setIsWaitingAudioMode(false);
  setIsMp3WaitingMode(false);
  
  if (recommendStatus === 'searching') {
    setRecommendStatus('processing');
  }
};





 // 🔧 비디오 대기 표시 수정
 const renderVideoPending = () => {
  if (!pendingVideo || questionConfirmStatus !== 'playing') {
    return null;
  }

  return (
    <div style={{
      position: 'absolute',
      bottom: '20px',
      right: '20px',
      backgroundColor: 'rgba(0, 0, 0, 0.85)',
      color: '#fff',
      padding: '15px 20px',
      borderRadius: '12px',
      fontSize: '0.9rem',
      zIndex: 50,
      border: '2px solid #ff6b6b'
    }}>
      <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>
        🎬 비디오 대기 중...
      </div>
      <div style={{ fontSize: '0.8rem', opacity: 0.8 }}>
        질문 확인 TTS 재생 완료 대기 중
      </div>
    </div>
  );
};





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
          setCurrentVideo(null); // ⭐ 핵심 추가
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
  if (!showReply || !currentReply|| isTtsPlaying) {
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





















// 🚀 음성 원형 스펙트럼 구독 (더 빠른 반응)
useEffect(() => {
  const voiceSpectrumListener = new ROSLIB.Topic({
    ros: ros,
    name: '/voice_spectrum',
    messageType: 'std_msgs/String'
  });

  voiceSpectrumListener.subscribe((message) => {
    try {
      const data = JSON.parse(message.data);
      if (data.type === 'voice_spectrum') {
        // 🚀 더 민감한 스케일링
        const scaledVolume = data.volume * 0.0003; // 0.00015 → 0.0003
        const clampedVolume = Math.min(1.0, scaledVolume);
        
        console.log(`🔊 원본 Peak: ${data.volume.toFixed(1)}, 스케일된값: ${clampedVolume.toFixed(3)}`);
        
        setVoiceVolume(clampedVolume);
        setIsVoiceActive(data.volume > 100); // 🚀 임계값 낮춤 (200 → 100)
      }
    } catch (e) {
      console.error('음성 스펙트럼 JSON 파싱 오류:', e);
    }
  });

  return () => voiceSpectrumListener.unsubscribe();
}, []);




// 🆕 음성 원형 스펙트럼 표시 조건 함수 (수정된 버전)
const shouldShowVoiceSpectrum = () => {
  return triggerDetected && 
         !musicPlaying && 
         recommendStatus !== 'searching' && 
         !isWaitingAudioMode && 
     
         !isMp3WaitingMode && 
         !videoVisible &&
         !isTtsPlaying &&
         !showUserQuestion && 
         isVoiceActive;
};

// 🚀 더 즉각적인 렌더링
const renderVoiceSpectrum = () => {
  if (!shouldShowVoiceSpectrum()) {
    return null;
  }

  // 🚀 더 역동적인 크기 변화
  const minRadius = 60;   // 🚀 60 → 40 (더 작은 최소값)
  const maxRadius = 600;  // 🚀 350 → 400 (더 큰 최대값)
  const radius = minRadius + (voiceVolume * (maxRadius - minRadius));
  
  // 🚀 더 강한 투명도 변화
  const minOpacity = 0.2; // 🚀 0.3 → 0.2
  const maxOpacity = 1.0; // 🚀 0.95 → 1.0 (완전 불투명)
  const opacity = minOpacity + (voiceVolume * (maxOpacity - minOpacity));

  return (
    <div style={{
      position: 'fixed',
      top: '50%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
      zIndex: 25,
      pointerEvents: 'none'
    }}>
      {/* 메인 원 */}
      <div style={{
        width: `${radius * 2}px`,
        height: `${radius * 2}px`,
        borderRadius: '50%',
        background: `radial-gradient(circle, rgba(100, 200, 255, ${opacity}) 0%, rgba(50, 150, 255, ${opacity * 0.5}) 70%, rgba(0, 100, 255, 0) 100%)`,
        border: `4px solid rgba(100, 200, 255, ${opacity})`,
        transition: 'all 0.02s linear', // 🚀 0.05s → 0.02s (더 빠른 전환)
        animation: voiceVolume > 0.05 ? 'voicePulse 0.15s infinite alternate' : 'none' // 🚀 더 빠른 애니메이션
      }} />
      
      {/* 🚀 더 역동적인 중앙 점 */}
      <div style={{
        position: 'absolute',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        width: `${8 + (voiceVolume * 20)}px`,  // 🚀 더 큰 변화폭
        height: `${8 + (voiceVolume * 20)}px`,
        borderRadius: '50%',
        backgroundColor: `rgba(100, 200, 255, ${opacity})`,
        boxShadow: `0 0 ${10 + (voiceVolume * 40)}px rgba(100, 200, 255, ${opacity})`, // 🚀 더 큰 그림자
        transition: 'all 0.01s linear' // 🚀 매우 빠른 반응
      }} />
      
      {/* 🚀 더 민감한 외곽 링 */}
      {voiceVolume > 0.05 && ( // 🚀 0.1 → 0.05 (더 민감)
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: `${radius * 3.0}px`, // 🚀 더 큰 링
          height: `${radius * 3.0}px`,
          borderRadius: '50%',
          border: `3px solid rgba(100, 200, 255, ${opacity * 0.4})`,
          animation: 'voiceRipple 0.4s infinite' // 🚀 더 빠른 리플
        }} />
      )}
      
      {/* CSS 애니메이션도 더 빠르게 */}
      <style>
        {`
          @keyframes voicePulse {
            0% { transform: scale(1); }
            100% { transform: scale(1.12); }
          }
          
          @keyframes voiceRipple {
            0% { 
              transform: translate(-50%, -50%) scale(0.6);
              opacity: 1;
            }
            100% { 
              transform: translate(-50%, -50%) scale(1.6);
              opacity: 0;
            }
          }
        `}
      </style>
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
  console.log('🔄 TTS 상태 변화 감지:', {
    ttsStatus,
    videoVisible,
    showReply,
    currentReply,
    isTtsPlaying
  });

  if (ttsStatus === 'tts_ready') {
    if (currentVideo && !videoVisible) {
      // 비디오가 있는 경우: 비디오 재생 시작
      console.log('🎬 TTS 준비 완료 - 비디오 재생 시작');
      setVideoVisible(true);
      setShowReply(false);
      setIsWaitingAudioMode(false);
      setIsMp3WaitingMode(false);
      
    } else if (!currentVideo && showReply) {
      // 비디오가 없는 경우: TTS 재생 시작
      console.log('🗣️ TTS 준비 완료 - TTS 전용 재생 시작');
      
      if (!currentReply || currentReply.trim() === '') {
        console.warn('⚠️ currentReply가 비어있음 - TTS 재생 중단');
        return;
      }

      setShowUserQuestion(false);
      setUserQuestionText('');
      setIsWaitingAudioMode(false);
      setIsMp3WaitingMode(false);
      setShowReply(false);
      setIsTtsPlaying(true);
      requestTtsPlay();
    }

  } else if (ttsStatus === 'tts_playing') {
    setIsTtsPlaying(true);
  } else if (ttsStatus === 'tts_done') {
    console.log('🗣️ TTS 재생 완료 - 초기화');
    
    // ✅ TTS 관련 상태만 초기화
    setShowReply(false);
    setIsTtsPlaying(false);
    setShowUserQuestion(false);
    setUserQuestionText('');
    setCurrentReply('');
    setVideoVisible(false);
    
    // 🆕 단일 단어 자막 상태 초기화
    setSingleWordSubtitle(null);
    setShowSingleWord(false);
    
    // ✅ 새 질문을 위한 초기화
    setCanShowSpectrum(false);
    setMusicPlaying(false);
    setIsWaitingAudioMode(false);
    setIsMp3WaitingMode(false);
    setIsTransitioning(false);
    setVoiceVolume(0);
    setIsVoiceActive(false);
    setIsDirectionFixed(false);
    setFixedDirection(null);
    
    console.log('✅ TTS 완료 후 상태 초기화');
  }
}, [ttsStatus, videoVisible, showReply, currentReply, currentVideo, isTtsPlaying]);









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
    if (musicPlaying || micSpectrum.length === 0 || isWaitingAudioMode || isMp3WaitingMode   || isTransitioning|| videoVisible) return;
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





    
  }, [micSpectrum, musicPlaying, recommendStatus, canvasSize, triggerDetected, videoVisible, isTransitioning]);

 




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
   {!videoVisible && !showReply  && !shouldShowVoiceSpectrum() && !isTtsPlaying &&!waitingForQuestionConfirm && (
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

    {/* 🆕 음성 원형 스펙트럼 표시 추가 */}
    {renderVoiceSpectrum()}

    {/* 🆕 비디오 대기 중 표시 */}
    {renderVideoPending()}



    {/* 기존 이미지 표시 */}
    {videoVisible && renderVideo()}
    {/* 🆕 TTS 대기 중 표시 */}
    {renderTtsWaiting()}

    {/* 🆕 사용자 질문 말풍선 추가 - 대기 스펙트럼과 함께 표시 */}
    {renderUserQuestionBubble()}


  {/* 🔧 기존 TTS 노래방 자막 대신 단일 단어 자막 사용 */}
  {renderSingleWordSubtitle()}
      
      

  </div>
);
  
}

export default SpectrumVisualizer;





