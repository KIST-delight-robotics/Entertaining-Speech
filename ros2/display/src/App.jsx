

//밈이미지(0602) - 스펙트럼 시작점 개선완료
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
  const [spectrum, setSpectrum] = useState([]);
  const [musicPlaying, setMusicPlaying] = useState(false);
  const [currentImage, setCurrentImage] = useState(null); // 이미지 상태 추가
  const canvasRef = useRef(null);
  const [recommendStatus, setRecommendStatus] = useState('done');
  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 400 });
  const [imageVisible, setImageVisible] = useState(false);
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



  // 🆕 수정: 의존성 제거 및 안정적인 구독
useEffect(() => {
  const gifStatusListener = new ROSLIB.Topic({
      ros: ros,
      name: '/gif_status',
      messageType: 'std_msgs/String'
  });

  gifStatusListener.subscribe((message) => {
      console.log('🎬 GIF 상태 수신:', message.data);
      
      if (message.data === 'searching') {  // 🆕 UserQuestion.py와 일치
          // 🆕 중복 실행 방지
          setRecommendStatus(prevStatus => {
              if (prevStatus !== 'searching') {
                  // 🆕 availableGifs를 직접 참조 대신 상수 사용
                  const gifs = ['1.gif', '2.gif', '3.gif', '4.gif', '5.gif', '6.gif'];
                  const randomIndex = Math.floor(Math.random() * gifs.length);
                  const selectedGif = gifs[randomIndex];
                  
                  console.log('🎬 선택된 gif:', selectedGif);
                  setCurrentGif(selectedGif);
                  setCurrentImage(null);
                  setImageVisible(false);
                  setCanShowSpectrum(false);
                  
                  return 'searching';
              }
              return prevStatus; // 이미 searching이면 변경하지 않음
          });
      } else if (message.data === 'done') {
          setRecommendStatus(prevStatus => {
              if (!musicPlaying) {
                  setCurrentGif('');
                  return 'done';
              }
              return prevStatus;
          });
      }
  });

  return () => {
      console.log('🔌 gif_status 리스너 정리');
      gifStatusListener.unsubscribe();
  };
}, []); // 🆕 의존성 배열을 비움 (한 번만 실행)








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





  // 1. 음악 상태 구독 - 타이밍 개선된 버전
  useEffect(() => {
    const statusListener = new ROSLIB.Topic({
        ros: ros,
        name: '/music_status',
        messageType: 'std_msgs/String'
    });

    statusListener.subscribe((message) => {
        console.log('음악 상태 변경:', message.data);
        
        if (message.data === 'music_playing') {
            setMusicPlaying(true);
            setCanShowSpectrum(false); // 스펙트럼 표시 초기화
            

            
        } else if (message.data === 'music_done') {
            setMusicPlaying(false);
            setRecommendStatus('done');
            setCanShowSpectrum(false);
            setImageVisible(false);
        }
    });

    return () => statusListener.unsubscribe();
  }, []);




  // 2. 상황에 따라 구독 토픽 자동 변경
  useEffect(() => {
    const topicName = musicPlaying ? '/audio_amplitude' : '/audio_visualizer';
    const spectrumListener = new ROSLIB.Topic({
      ros: ros,
      name: topicName,
      messageType: 'std_msgs/String'
    });
    spectrumListener.subscribe((message) => {
      try {
        const data = JSON.parse(message.data);
        if (data.spectrum) setSpectrum(data.spectrum);
      } catch (e) {
        console.error('JSON parse error:', e);
      }
    });
    return () => spectrumListener.unsubscribe();
  }, [musicPlaying]);



  

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || spectrum.length === 0) return;
  
    const ctx = canvas.getContext('2d');
    const { width, height } = canvasSize;
  
    canvas.width = width;
    canvas.height = height;


  
    // 전체 화면 배경색
    ctx.fillStyle = musicPlaying ? '#000' : '#222222';
    ctx.fillRect(0, 0, width, height);
  
    if (recommendStatus === 'searching'&& !imageVisible) {
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, width, height);
      return;
    }

      // 🆕 마이크 입력 시 trigger_detected=false일 때 배경색만 표시
    if (!musicPlaying && !triggerDetected) {
      console.log('🎵 마이크 모드 - trigger_detected=false, 배경색만 표시');
      // 배경색은 이미 위에서 설정했으므로 그대로 return
      return;
    }

    // 🆕 음악 재생 중 스펙트럼 표시 조건 (기존 로직 유지)
    if (musicPlaying && !canShowSpectrum) {
      console.log('🎵 음악 재생 중 - 이미지 표시 대기로 스펙트럼 숨김');
      return;
    }

    // 🆕 마이크 입력 시 스펙트럼 표시 조건
    if (!musicPlaying && recommendStatus === 'searching') {
      console.log('🎵 마이크 모드 - 검색 중이므로 스펙트럼 숨김');
      return;
    }


  
    const central = getCentralSlice(spectrum, 0.6);
    const numBars =43;
    let bars = downsampleArray(central, numBars);
    bars = bars.map(v => Math.min(1, v * 10));
  
    // 캔버스 크기에 비례한 스펙트럼 크기 (원본 비율 유지)
    const scale = Math.min(width / 1018, height / 240); // 원본 크기 기준 스케일
    
    const barWidth = 10 * scale;
    const gap = 14 * scale;
    const maxBarHeight = 120 * scale;
    
    const totalWidth = numBars * barWidth + (numBars - 1) * gap;
    
    // 화면 중앙에 배치
    const xOffset = (width - totalWidth) / 2;
    const centerY = height / 2;
  
    ctx.strokeStyle = musicPlaying ? '#ff00cc' : '#fff';
    ctx.lineWidth = barWidth;
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
  }, [spectrum, musicPlaying, recommendStatus, canvasSize, canShowSpectrum,triggerDetected]);
  
  




  useEffect(() => {
    const updateCanvasSize = () => {
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;
      
      // 오디오 스펙트럼의 원본 비율 (4.24:1)
      const spectrumAspectRatio = 1018 / 240; // 4.24
      
      // 가상 해상도 설정 (스펙트럼이 적당한 크기가 되도록)
      const virtualWidth = 1200;
      const virtualHeight = virtualWidth / spectrumAspectRatio; // 283px
      
      // viewport에 맞춰 스케일 계산 (비율 유지)
      const scaleX = viewportWidth / virtualWidth;
      const scaleY = viewportHeight / virtualHeight;
      const scale = Math.min(scaleX, scaleY) * 0.9; // 80%에서 60%로 변경 (더 많은 여백)
      
      // 실제 캔버스 크기 계산
      const canvasWidth = virtualWidth * scale;
      const canvasHeight = virtualHeight * scale;
      
      setCanvasSize({ width: canvasWidth, height: canvasHeight });
    };
  
    updateCanvasSize();
    window.addEventListener('resize', updateCanvasSize);
    
    return () => window.removeEventListener('resize', updateCanvasSize);
  }, []);



  // 4. 이미지 토픽 구독 추가
  useEffect(() => {
    const imageListener = new ROSLIB.Topic({
        ros: ros,
        name: '/current_music_image',
        messageType: 'std_msgs/String'
    });

    imageListener.subscribe((message) => {
        console.log('🖼️ 이미지 메시지 수신:', message.data);
        
        if (message.data && message.data.trim() !== "") {
            const imagePath = message.data;
           



            console.log('🖼️ 이미지 표시:', imagePath);
            setCurrentImage(imagePath);
            setImageVisible(true);
            setCurrentGif(''); // 이 줄 추가




            // searching 상태도 해제
            if (recommendStatus === 'searching') {
              setRecommendStatus('processing');
  }


        } else {
            console.log('🖼️ 이미지 숨김');
            setCurrentImage(null);
            setImageVisible(false);


            // 이미지가 숨김 상태가 되면 스펙트럼 시각화 시작
            if (musicPlaying) {
              console.log('🎵 이미지 숨김 완료 - 스펙트럼 시각화 시작');
              setCanShowSpectrum(true);
              setRecommendStatus('done');
          }





        }
    });

    return () => {
        console.log('🖼️ 이미지 리스너 해제');
        imageListener.unsubscribe();
    };
}, [musicPlaying]);

// 5. 이미지 표시 컴포넌트
const renderImage = () => {
    if (!currentImage || !imageVisible){
        return null;
    }


    // [수정] 파일명만 정확하게 인코딩하는 로직으로 변경
    const createSafeUrl = (path) => {
      try {
          // 1. 마지막 '/'를 기준으로 디렉터리 경로와 파일명을 분리합니다.
          const lastSlashIndex = path.lastIndexOf('/');
          const directoryPath = path.substring(0, lastSlashIndex + 1); // 예: "/images/"
          const fileName = path.substring(lastSlashIndex + 1);      // 예: "어.. 얘 멋있다!.jpg"

          // 2. 파일명 부분만 완벽하게 인코딩합니다.
          const encodedFileName = encodeURIComponent(fileName);

          // 3. 디렉터리 경로와 인코딩된 파일명을 다시 합쳐 완전한 URL을 만듭니다.
          return directoryPath + encodedFileName;
      } catch (e) {
          console.error("URL 생성 중 오류 발생:", e);
          return path; // 오류 발생 시 원본 경로 반환
      }
  };

  const safeImageUrl = createSafeUrl(currentImage);





    return (
      <div style={{
        position: 'absolute',
        top: '0',              // 🆕 화면 맨 위부터
        left: '0',             // 🆕 화면 맨 왼쪽부터
        width: '100vw',        // 🆕 화면 전체 너비
        height: '100vh',       // 🆕 화면 전체 높이
        zIndex: 15,
        display: 'flex',       // 🆕 중앙 정렬을 위한 flexbox
        justifyContent: 'center',
        alignItems: 'center'
    }}>
            <img 
                src={safeImageUrl} 
                alt="Music Visual"
                style={{
                  width: 'auto',              // 🆕 너비 자동 (비율 유지)
                  height: '100vh',            // 🆕 세로를 화면에 꽉 차게
                  minWidth: '100vw',           // 🆕 최소 너비로 화면 전체 커버
                  objectFit: 'contain',         // 넘치는 부분 자르기
                  objectPosition: 'center',   // 중앙 정렬
                  borderRadius: '0px',
                  boxShadow: 'none'     
                }}
                onLoad={() => console.log('🖼️ 이미지 로드 성공:', safeImageUrl)}
                onError={() => console.error('🖼️ 이미지 로드 실패:', safeImageUrl)}
            />
        </div>
    );
};


// renderImage 함수 다음에 이 함수를 새로 추가
const renderGif = () => {
  // 이미지가 표시 중일 때는 GIF 절대 표시하지 않음
  if (currentImage && imageVisible) {
      return null;
  }
  
  // searching 상태이고 GIF가 선택되었을 때만 표시
  if (recommendStatus === 'searching' && currentGif) {
      return (
          <div style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              zIndex: 5, // 이미지보다 낮은 우선순위로 변경
              backgroundColor: '#fff',
              width: `${canvasSize.width}px`,
              height: `${canvasSize.height}px`,
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center'
          }}>
              <img 
                  src={`/${currentGif}`}
                  alt="추천 중..." 
                  style={{
                      maxWidth: '500%',
                      maxHeight: '500%',
                      objectFit: 'contain'
                  }}
                  onLoad={() => {
                      console.log('🎬 GIF 로드 완료:', currentGif);
                  }}
                  onError={(e) => {
                      console.error('🎬 GIF 로드 실패:', currentGif);
                  }}
              />
          </div>
      );
  }
  
  return null;
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
      backgroundColor: (recommendStatus === 'searching' && !imageVisible) ? '#fff' : 
                 (musicPlaying || imageVisible) ? '#000' : '#222222',

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
        {isDirectionFixed 
          ? `🔒 고정: ${Math.round(fixedDirection)}° ${screenFlipped ? '(반전)' : '(정상)'}`
          : `📍 실시간: ${Math.round(soundDirection)}° ${screenFlipped ? '(반전)' : '(정상)'}`
        }
      </div>

      







    {!(musicPlaying && currentImage) && (  
      <canvas 
        ref={canvasRef}
        style={{
          width: `${canvasSize.width}px`,
          height: `${canvasSize.height}px`,
          border: 'none',
          display: 'block'
        }}
      />
    )}

      {/* 이미지 표시 - 음악 재생 중에만 */}
      {musicPlaying && renderImage()}
      
      {/* 추천 중일 때 gif 오버레이 */}
      {renderGif()}
 










    </div>
  );
  
  
  
  
  
  
}

export default SpectrumVisualizer;
