

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
  const canvasRef = useRef(null);
  const [recommendStatus, setRecommendStatus] = useState('done');
  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 400 });



    // gif 파일명들을 여기에 추가 (실제 파일명으로 수정하세요)
  const availableGifs = [
      '1.gif',
      '2.gif', 
      '3.gif',
      '4.gif',
      '5.gif',
      '6.gif'
    ];
  const [currentGif, setCurrentGif] = useState('');

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
            
            // GIF를 일정 시간 보여준 후 스펙트럼으로 전환
            setTimeout(() => {
                setRecommendStatus('done');
                console.log('음악 재생 - 스펙트럼 표시로 전환');
            }, 1500); // 1.5초 후 스펙트럼 표시
            
        } else if (message.data === 'music_done') {
            setMusicPlaying(false);
            setRecommendStatus('done');
        }
    });

    return () => statusListener.unsubscribe();
  }, []);

  // 2. mp3_recommend_status 토픽 구독 - 즉시 처리
  useEffect(() => {
    const statusListener = new ROSLIB.Topic({
        ros: ros,
        name: '/mp3_recommend_status',
        messageType: 'std_msgs/String'
    });

    statusListener.subscribe((message) => {
        console.log('추천 상태:', message.data);
        
        if (message.data === 'searching') {
            setRecommendStatus('searching');
            
            const randomIndex = Math.floor(Math.random() * availableGifs.length);
            const selectedGif = availableGifs[randomIndex];
            console.log('선택된 gif:', selectedGif);
            setCurrentGif(selectedGif);
        } else if (message.data === 'done') {
            // 즉시 done 처리하지 않고 음악 재생 여부 확인
            if (!musicPlaying) {
                setRecommendStatus('done');
            }
        } else {
            setRecommendStatus(message.data);
        }
    });

    return () => statusListener.unsubscribe();
  }, [availableGifs, musicPlaying]);



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
  
    if (recommendStatus === 'searching') {
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, width, height);
      return;
    }
  
    const central = getCentralSlice(spectrum, 0.6);
    const numBars = 43;
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
  }, [spectrum, musicPlaying, recommendStatus, canvasSize]);
  
  




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
      backgroundColor: recommendStatus === 'searching' ? '#fff' : 
                       musicPlaying ? '#000' : '#222222'
    }}>
      <canvas 
        ref={canvasRef}
        style={{
          width: `${canvasSize.width}px`,
          height: `${canvasSize.height}px`,
          border: 'none',
          display: 'block'
        }}
      />
      
      {/* 추천 중일 때 gif 오버레이 */}
      {recommendStatus === 'searching' && currentGif && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          zIndex: 10,
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
            onError={(e) => {
              console.error('이미지 로드 실패:', currentGif);
            }}
          />
        </div>
      )}
    </div>
  );
  
  
  
  
  
  
}

export default SpectrumVisualizer;
