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


  // 1. 음악 상태 구독
  useEffect(() => {
    const statusListener = new ROSLIB.Topic({
      ros: ros,
      name: '/music_status',
      messageType: 'std_msgs/String'
    });
    statusListener.subscribe((message) => {
      if (message.data === 'music_playing') setMusicPlaying(true);
      else if (message.data === 'music_done') setMusicPlaying(false);
    });
    return () => statusListener.unsubscribe();
  }, []);
  //mp3_recommend_status 토픽 구독 및 상태 관리 추가
  useEffect(() => {
    const statusListener = new ROSLIB.Topic({
      ros: ros,
      name: '/mp3_recommend_status',
      messageType: 'std_msgs/String'
    });
    statusListener.subscribe((message) => {
      setRecommendStatus(message.data);
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

  // 3. 시각화 (기존 코드와 동일)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || spectrum.length === 0) return;
    const ctx = canvas.getContext('2d');
  
    // 배경색: 음악이면 검정, 마이크면 초록
    ctx.fillStyle = musicPlaying ? '#000' : '#00c853'; // 초록: #00c853
    ctx.fillRect(0, 0, canvas.width, canvas.height);


      // 추천 중이면 텍스트만 표시
    if (recommendStatus === 'searching') {
      ctx.fillStyle = '#222';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.font = 'bold 56px sans-serif';
      ctx.fillStyle = '#ff00cc';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('음악찾는중', canvas.width / 2, canvas.height / 2);
      return;
  }
  
    const central = getCentralSlice(spectrum, 0.6);
    const numBars = 43;
    let bars = downsampleArray(central, numBars);
    bars = bars.map(v => Math.min(1, v * 10));
    const barWidth = 10;
    const gap = 14;
    const totalWidth = numBars * barWidth + (numBars - 1) * gap;
    const xOffset = (canvas.width - totalWidth) / 2;
    const centerY = canvas.height / 2;
    const maxBarHeight = canvas.height * 0.42;
  
    // 바 색: 음악이면 핑크, 마이크면 흰색
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
  }, [spectrum, musicPlaying]);

  return (
    <canvas
  ref={canvasRef}
  width={1100}
  height={340}
  style={{
    background: musicPlaying ? '#000' : '#00c853',
    display: 'block',
    margin: '40px auto',
    borderRadius: '16px'
  }}
/>
  );
}

export default SpectrumVisualizer;
