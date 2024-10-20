import React, { useState, useRef, useEffect } from 'react';
import { Link } from 'react-router-dom';
import './VideoChat.css';
import io from 'socket.io-client';


function VideoChat() {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [socket, setSocket] = useState(null);

    useEffect(() => {
        const newSocket = io('http://localhost:5000'); // Replace with your backend URL if different
        setSocket(newSocket);
    
        // Cleanup on unmount
        return () => newSocket.close();
      }, []);

      useEffect(() => {
        // Access the user's webcam
        navigator.mediaDevices.getUserMedia({ video: true })
          .then((stream) => {
            // Set the video source to the webcam stream
            if (videoRef.current) {
              videoRef.current.srcObject = stream;
            }
    
            // Capture frames at regular intervals and send to backend
            const interval = setInterval(() => {
              if (canvasRef.current && videoRef.current && socket) {
                const context = canvasRef.current.getContext('2d');
                context.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
                const imageData = canvasRef.current.toDataURL('image/jpeg');
                socket.emit('frame', imageData); // Send frame to backend via WebSocket
              }
            }, 200); // Adjust the interval as needed (e.g., every 200ms) edit fps here
    
            // Cleanup on unmount
            return () => clearInterval(interval);
          })
          .catch((err) => {
            console.error("Error accessing webcam: ", err);
          });
      }, [socket]);

      return (
        <div className="video-chat-container">
            <div className="video-background">
                <div className="window-controls">
                    <button className="window-control green"></button>
                    <button className="window-control yellow"></button>
                    <button className="window-control red"></button>
                </div>
                <div className="video-grid">
                    <div className="video-item">
                        <video ref={videoRef} autoPlay playsInline muted />
                        <canvas ref={canvasRef} style={{ display: 'none' }} />
                        <div className="participant-name">Live Camera Feed</div>
                    </div>
                </div>
            </div>
            <div className="controls">
                <button className="control-button mute"></button>
                <button className="control-button video"></button>
                <button className="control-button share"></button>
                <Link to="/" className="control-button end-call"></Link>
            </div>
        </div>
    );
}


export default VideoChat;