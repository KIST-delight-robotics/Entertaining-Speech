const io = require('socket.io')(3000, {
    cors: {
        origin: '*'
    }
});

io.on('connection', (socket) => {
    console.log(`Client connected: ${socket.id}`);

    // 방에 참가하여 Sender와 Receiver 연결
    socket.on('join_room', (room) => {
        socket.join(room);
        console.log(`Client ${socket.id} joined room ${room}`);
    });

    // Offer와 Answer 처리
    socket.on('offer', (data) => {
        socket.to(data.room).emit('offer', data.offer);
    });

    socket.on('answer', (data) => {
        socket.to(data.room).emit('answer', data.answer);
    });

    // ICE Candidate 교환
    socket.on('candidate', (data) => {
        socket.to(data.room).emit('candidate', data.candidate);
    });

    // 두 개의 키워드를 Receiver에게 전송
    socket.on('keywords', (data) => {
        // `to(room)`을 통해 동일한 방에 있는 Receiver에게만 전달
        socket.to(data.room).emit('keywords', {
            okt_keywords: data.okt_keywords,
            gpt_keywords: data.gpt_keywords
        });
        console.log(`Keywords sent to room ${data.room}: okt - ${data.okt_keywords}, gpt - ${data.gpt_keywords}`);
    });
});

console.log("Signaling server running on port 3000");