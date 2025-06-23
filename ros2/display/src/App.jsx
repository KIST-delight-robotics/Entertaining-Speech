
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
  // 🆕 분리된 스펙트럼 상태
  const [musicSpectrum, setMusicSpectrum] = useState([]);
  const [micSpectrum, setMicSpectrum] = useState([]);

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
            
        } else if (message.data === 'music_done') {
            setMusicPlaying(false);
            setRecommendStatus('done');
            setCanShowSpectrum(false);
            setImageVisible(false);
            setIsMp3WaitingMode(false); // Mp3 waiting 모드 종료
        }
    });

    return () => {
        statusListener.unsubscribe();
    };
  }, []);





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

  // 🆕 마이크 스펙트럼 구독
  useEffect(() => {
    if (musicPlaying) return;
    
    const micSpectrumListener = new ROSLIB.Topic({
      ros: ros,
      name: '/audio_visualizer',
      messageType: 'std_msgs/String'
    });
    
    micSpectrumListener.subscribe((message) => {
      try {
        const data = JSON.parse(message.data);
        if (data.spectrum) {
          // 🆕 마이크 스펙트럼 값 웹 콘솔 출력
        console.log('🎤 마이크 스펙트럼 수신:', {
          length: data.spectrum.length,
          first_10_values: data.spectrum.slice(0, 10),
          max_value: Math.max(...data.spectrum),
          min_value: Math.min(...data.spectrum),
          average: data.spectrum.reduce((a, b) => a + b, 0) / data.spectrum.length,
          timestamp: new Date().toLocaleTimeString()
        });
        

          const smoothedData = applyMicSmoothing(data.spectrum);
          setMicSpectrum(smoothedData);
          //setMicSpectrum(data.spectrum);

        }
      } catch (e) {
        console.error('Mic spectrum JSON parse error:', e);
      }
    });
    
    return () => {
      micSpectrumListener.unsubscribe();
      //previousMicSpectrumRef.current = [];
    };
  }, [musicPlaying]);


    //==========================================================

  // 🆕 대기 스펙트럼 구독 - 대기 효과음1 (/waiting_spectrum)
  useEffect(() => {
    const waitingSpectrumListener = new ROSLIB.Topic({
      ros: ros,
      name: '/waiting_spectrum',
      messageType: 'std_msgs/String'
    });
    
    waitingSpectrumListener.subscribe((message) => {
      try {
        const data = JSON.parse(message.data);
        if (data.spectrum) {
          setIsWaitingAudioMode(true);
          setIsWaitingImageMode(false);
          setRecommendStatus('waiting_audio'); // 대기 오디오 상태
          setCurrentGif(''); // 기존 GIF 제거
          const smoothedData = applyAdvancedSmoothing(data.spectrum);
          setWaitingSpectrum(smoothedData);
        }
      } catch (e) {
        console.error('Waiting spectrum JSON parse error:', e);
      }
    });
    
    return () => {
      waitingSpectrumListener.unsubscribe();
    };
  }, []);


//==========================================================


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
    if (musicPlaying || micSpectrum.length === 0 || isWaitingAudioMode || isMp3WaitingMode  || isWaitingImageMode || isTransitioning) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const { width, height } = canvasSize;
    
    canvas.width = width;
    canvas.height = height;
    
    // 마이크용 배경
    ctx.fillStyle = '#222222';
    ctx.fillRect(0, 0, width, height);
    
    if (recommendStatus === 'searching' && !imageVisible) {
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

    if (imageVisible || currentImage) {
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
    const limitedSpectrum = micSpectrum.slice(30, 180); // 0~300 인덱스 추출
let bars = downsampleArray(limitedSpectrum, numBars);
  // let bars = downsampleArray(micSpectrum, numBars);


// bars = bars.map(v => v * 0.0000035);
bars = bars.map(v => v * 0.000025);


// // 최대값 2로 제한
// bars = bars.map(v => Math.min(1, v * 0.00005));
    





const availableWidth = width * 0.9; // 화면 너비의 90% 사용
const totalGaps = (numBars - 1);
    
    const scale = Math.min(width / 1018, height / 240);
    // const barWidth = 10 * scale;
    const barWidth = Math.max(2, availableWidth / (numBars + totalGaps * 0.5));
    // const gap = 14 * scale;
    const gap = Math.max(1, barWidth * 0.5); // 바 너비의 30%를 간격으로
  // 🆕 바 두께 별도 설정 (가로길이는 기존 barWidth 유지)
  const barThickness = barWidth * 0.8; // 바 두께를 기존의 60%로 설정

      
     const maxBarHeight = 150 * scale;

     
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





    
  }, [micSpectrum, musicPlaying, recommendStatus, canvasSize, triggerDetected, imageVisible, isWaitingImageMode, isTransitioning]);

 




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


            // 🆕 대기 모드 종료 (Mp3Recommender 이미지가 왔으므로)
            setWaitingImage(null);
            setWaitingImageVisible(false);
            setIsWaitingImageMode(false);
            setIsWaitingAudioMode(false);




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
      backgroundColor: (recommendStatus === 'searching' && !imageVisible && !isWaitingAudioMode) ? '#fff' : 
      (isWaitingAudioMode) ? '#fff' :
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

      







    {/* {!(musicPlaying && currentImage) && (   */}
    {!(musicPlaying && currentImage) && !waitingImageVisible && (
      <canvas 
        ref={canvasRef}
        style={{
      // 🔧 모든 모드에서 화면 전체 사용
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



      {/* 이미지 표시 - 음악 재생 중에만 */}
      {musicPlaying && renderImage()}
      
      {/* 대기중 이미지 표시 */}
      {renderWaitingImage()}
 










    </div>
  );
  
  
  
  
  
  
}

export default SpectrumVisualizer;


