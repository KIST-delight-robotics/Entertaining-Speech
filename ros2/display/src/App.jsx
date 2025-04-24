
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
  const canvasRef = useRef(null);

  useEffect(() => {
    const spectrumListener = new ROSLIB.Topic({
      ros: ros,
      name: '/audio_amplitude',
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
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || spectrum.length === 0) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 1. 중앙 60%만 사용
    const central = getCentralSlice(spectrum, 0.6);

    // 2. 바 개수 43개
    const numBars = 43;
    let bars = downsampleArray(central, numBars);

    // 3. 바 높이 키우기 (최대 1.7배, 1로 클램프)
    bars = bars.map(v => Math.min(1, v * 10));

    // 4. 바 스타일
    const barWidth = 10;
    const gap = 14;
    const totalWidth = numBars * barWidth + (numBars - 1) * gap;
    const xOffset = (canvas.width - totalWidth) / 2;
    const centerY = canvas.height / 2;
    const maxBarHeight = canvas.height * 0.42; // 위아래로 뻗게 (전체 높이의 84%를 사용)

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
  }, [spectrum]);

  return (
    <canvas
      ref={canvasRef}
      width={1100}
      height={340}
      style={{
        background: '#000',
        display: 'block',
        margin: '40px auto',
        borderRadius: '16px'
      }}
    />
  );
}

export default SpectrumVisualizer;

