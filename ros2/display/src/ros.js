import ROSLIB from 'roslib';

const ros = new ROSLIB.Ros({
  url: 'ws://172.20.10.5:9090' // Jetson의 IP로 변경
});

ros.on('connection', () => {
  console.log('Connected to rosbridge');
});
ros.on('error', (error) => {
  console.error('Error connecting to rosbridge:', error);
});
ros.on('close', () => {
  console.log('Connection to rosbridge closed.');
});

export default ros;
