


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
        return prev * 0.6 + current * 0.4; // 더 빠른 감쇠
      //   return prev * 0.5 + current * 0.5; // 더 빠른 반응
      // } else {
      //   return prev * 0.8 + current * 0.2; // 더 빠른 감쇠

      }
    });
    
    previousMicSpectrumRef.current = [...smoothed];
    return smoothed;
  };




  // // 2. 상황에 따라 구독 토픽 자동 변경
  // useEffect(() => {
  //   const topicName = musicPlaying ? '/audio_amplitude' : '/audio_visualizer';
  //   const spectrumListener = new ROSLIB.Topic({
  //     ros: ros,
  //     name: topicName,
  //     messageType: 'std_msgs/String'
  //   });
  //   spectrumListener.subscribe((message) => {
  //     try {
  //       const data = JSON.parse(message.data);
  //       if (data.spectrum) setSpectrum(data.spectrum);
  //     } catch (e) {
  //       console.error('JSON parse error:', e);
  //     }
  //   });
  //   return () => spectrumListener.unsubscribe();
  // }, [musicPlaying]);


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
        }
      } catch (e) {
        console.error('Music spectrum JSON parse error:', e);
      }
    });
    
    return () => {
      musicSpectrumListener.unsubscribe();
      previousSpectrumRef.current = [];
    };
  }, [musicPlaying]);

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
          const smoothedData = applyMicSmoothing(data.spectrum);
          setMicSpectrum(smoothedData);
        }
      } catch (e) {
        console.error('Mic spectrum JSON parse error:', e);
      }
    });
    
    return () => {
      micSpectrumListener.unsubscribe();
      previousMicSpectrumRef.current = [];
    };
  }, [musicPlaying]);



  // 🆕 대기 스펙트럼 구독 (/waiting_spectrum)
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
        setWaitingImage(null);
        setWaitingImageVisible(false);
        setIsWaitingImageMode(false);
        setIsWaitingAudioMode(false);
        // 대기 시퀀스 완료 후 정상 상태로 복귀하지만 searching은 유지
        // Mp3Recommender 이미지가 올 때까지 대기
      }
    });

    return () => {
      console.log('🖼️ 대기 이미지 리스너 해제');
      waitingImageListener.unsubscribe();
    };
  }, []);





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
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, width, height);
  
  if (!canShowSpectrum) return;
  
  // 🆕 음악용 스펙트럼 처리 (기존 로직 유지)
  const central = getCentralSlice(musicSpectrum, 0.6);
  const numBars = 43;
  let bars = downsampleArray(central, numBars);
  bars = bars.map(v => Math.min(1, v * 10));
  
  const scale = Math.min(width / 1018, height / 240);
  const barWidth = 10 * scale;
  const gap = 14 * scale;
  const maxBarHeight = 120 * scale;
  const totalWidth = numBars * barWidth + (numBars - 1) * gap;
  const xOffset = (width - totalWidth) / 2;
  const centerY = height / 2;
  
  ctx.strokeStyle = '#ff00cc';
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
}, [musicSpectrum, musicPlaying, canvasSize, canShowSpectrum]);

  // 🆕 마이크 스펙트럼 렌더링
  useEffect(() => {
    //if (musicPlaying || micSpectrum.length === 0) return;
    if (musicPlaying || micSpectrum.length === 0 || isWaitingAudioMode) return;
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
    
    // 🆕 마이크용 스펙트럼 처리 (나중에 다르게 커스터마이징 가능)
    const central = getCentralSlice(micSpectrum, 0.6);
    const numBars = 43;
    let bars = downsampleArray(central, numBars);
    bars = bars.map(v => Math.min(1, v * 10));
    
    const scale = Math.min(width / 1018, height / 240);
    const barWidth = 10 * scale;
    const gap = 14 * scale;
    const maxBarHeight = 120 * scale;
    const totalWidth = numBars * barWidth + (numBars - 1) * gap;
    const xOffset = (width - totalWidth) / 2;
    const centerY = height / 2;
    
    ctx.strokeStyle = '#fff';
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
  }, [micSpectrum, musicPlaying, recommendStatus, canvasSize, triggerDetected, imageVisible]);


  // // 🆕 대기 스펙트럼 렌더링 (Mp3Player.py 스타일과 동일)
  // useEffect(() => {
  //   if (!isWaitingAudioMode || waitingSpectrum.length === 0) return;
    
  //   const canvas = canvasRef.current;
  //   if (!canvas) return;
    
  //   const ctx = canvas.getContext('2d');
  //   const { width, height } = canvasSize;
    
  //   canvas.width = width;
  //   canvas.height = height;
    
  //   // 대기용 배경 (음악과 동일)
  //   ctx.fillStyle = '#fff';
  //   ctx.fillRect(0, 0, width, height);
    
  //   // 🆕 대기용 스펙트럼 처리 (음악용과 동일한 로직)
  //   const central = getCentralSlice(waitingSpectrum, 0.6);
  //   const numBars = 43;
  //   let bars = downsampleArray(central, numBars);
  //   bars = bars.map(v => Math.min(1, v * 10));
    
  //   const scale = Math.min(width / 1018, height / 240);
  //   const barWidth = 10 * scale;
  //   const gap = 14 * scale;
  //   const maxBarHeight = 120 * scale;
  //   const totalWidth = numBars * barWidth + (numBars - 1) * gap;
  //   const xOffset = (width - totalWidth) / 2;
  //   const centerY = height / 2;
    
  //   // 대기 모드 전용 색상 (주황색으로 구분)
  //   ctx.strokeStyle = '#ff9500';
  //   ctx.lineWidth = barWidth;
  //   ctx.lineCap = 'round';
  //   ctx.lineJoin = 'round';
    
  //   for (let i = 0; i < numBars; i++) {
  //     const x = xOffset + i * (barWidth + gap) + barWidth / 2;
  //     const barHeight = bars[i] * maxBarHeight;
  //     ctx.beginPath();
  //     ctx.moveTo(x, centerY - barHeight);
  //     ctx.lineTo(x, centerY + barHeight);
  //     ctx.stroke();
  //   }
  // }, [waitingSpectrum, isWaitingAudioMode, canvasSize]);


  // 🆕 대기 스펙트럼 렌더링 (원형 버전)
  useEffect(() => {
    if (!isWaitingAudioMode || waitingSpectrum.length === 0) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const { width, height } = canvasSize;
    
    canvas.width = width;
    canvas.height = height;
    
    // 대기용 흰 배경
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, width, height);
    
    // 🆕 원형 스펙트럼 설정
    const centerX = width / 2;
    const centerY = height / 2;
    
    // 화면의 더 큰 비율 사용 (화면을 꽉 차게)
    const screenSize = Math.min(width, height);
    const baseRadius = Math.min(width, height) * 0.2; // 기본 반지름 (화면 크기의 20%)
    const maxBarLength = Math.min(width, height) * 0.3; // 최대 바 길이 (화면 크기의 25%)
    
    // 스펙트럼 데이터 처리
    const central = getCentralSlice(waitingSpectrum, 0.6);
    const numBars = 64; // 원형에서는 더 많은 바를 사용 (360도를 균등 분할)
    let bars = downsampleArray(central, numBars);
    bars = bars.map(v => Math.min(1, v * 8)); // 강도 조절
    
    // 원형 렌더링
    ctx.strokeStyle = '#333333'; // 흰 배경에 맞는 어두운 색상
    ctx.lineWidth = Math.max(2, Math.min(width, height) * 0.008); // 화면 크기에 비례한 선 두께
    ctx.lineCap = 'round';
    
    for (let i = 0; i < numBars; i++) {
      // 각도 계산 (360도를 균등 분할)
      const angle = (i / numBars) * 2 * Math.PI;
      
      // 바의 길이 계산
      const barLength = bars[i] * maxBarLength;
      
      // 시작점과 끝점 계산
      const startX = centerX + Math.cos(angle) * baseRadius;
      const startY = centerY + Math.sin(angle) * baseRadius;
      const endX = centerX + Math.cos(angle) * (baseRadius + barLength);
      const endY = centerY + Math.sin(angle) * (baseRadius + barLength);
      
      // 바 그리기
      ctx.beginPath();
      ctx.moveTo(startX, startY);
      ctx.lineTo(endX, endY);
      ctx.stroke();
    }
    
    // 🆕 중앙 원 그리기 (선택사항 - 시각적 완성도 향상)
    ctx.beginPath();
    ctx.arc(centerX, centerY, baseRadius * 0.1, 0, 2 * Math.PI);
    ctx.fillStyle = '#333333';
    ctx.fill();
    
  }, [waitingSpectrum, isWaitingAudioMode, canvasSize]);







  useEffect(() => {
    const updateCanvasSize = () => {
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;

      // 🆕 대기 모드일 때는 화면 전체 사용
      if (isWaitingAudioMode) {
        setCanvasSize({ 
          width: viewportWidth, 
          height: viewportHeight 
        });
        return;
      }

    
        
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
  }, [isWaitingAudioMode]);



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


// // renderImage 함수 다음에 이 함수를 새로 추가
// const renderGif = () => {
//   // 이미지가 표시 중일 때는 GIF 절대 표시하지 않음
//   if (currentImage && imageVisible) {
//       return null;
//   }
  
//   // searching 상태이고 GIF가 선택되었을 때만 표시
//   if (recommendStatus === 'searching' && currentGif) {
//       return (
//           <div style={{
//               position: 'absolute',
//               top: '50%',
//               left: '50%',
//               transform: 'translate(-50%, -50%)',
//               zIndex: 5, // 이미지보다 낮은 우선순위로 변경
//               backgroundColor: '#fff',
//               width: `${canvasSize.width}px`,
//               height: `${canvasSize.height}px`,
//               display: 'flex',
//               justifyContent: 'center',
//               alignItems: 'center'
//           }}>
//               <img 
//                   src={`/${currentGif}`}
//                   alt="추천 중..." 
//                   style={{
//                       maxWidth: '500%',
//                       maxHeight: '500%',
//                       objectFit: 'contain'
//                   }}
//                   onLoad={() => {
//                       console.log('🎬 GIF 로드 완료:', currentGif);
//                   }}
//                   onError={(e) => {
//                       console.error('🎬 GIF 로드 실패:', currentGif);
//                   }}
//               />
//           </div>
//       );
//   }
  
//   return null;
// };

// 🆕 renderGif 함수를 완전히 대체하는 대기 이미지 렌더링 함수
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
          // width: `${canvasSize.width}px`,
          // height: `${canvasSize.height}px`,
          // border: 'none',
          // display: 'block'

          // 🆕 대기 모드일 때는 화면 전체, 다른 모드일 때는 기존 크기
          width: isWaitingAudioMode ? '100vw' : `${canvasSize.width}px`,
          height: isWaitingAudioMode ? '100vh' : `${canvasSize.height}px`,
          // 🆕 대기 모드일 때는 절대 위치로 화면 전체 덮기
          position: isWaitingAudioMode ? 'fixed' : 'relative',
          top: isWaitingAudioMode ? '0' : 'auto',
          left: isWaitingAudioMode ? '0' : 'auto',
          zIndex: isWaitingAudioMode ? 10 : 'auto',
          // 🆕 테두리 완전 제거
          border: 'none',
          outline: 'none',
          display: 'block',
          // 🆕 모바일 터치 하이라이트 제거
          WebkitTapHighlightColor: 'transparent'
        }}
      />
    )}

      {/* 이미지 표시 - 음악 재생 중에만 */}
      {musicPlaying && renderImage()}
      
      {/* 추천 중일 때 gif 오버레이 */}
      {/* {renderGif()} */}
      {renderWaitingImage()}
 










    </div>
  );
  
  
  
  
  
  
}

export default SpectrumVisualizer;
