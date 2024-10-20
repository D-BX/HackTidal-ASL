import React from 'react';
import { Link } from 'react-router-dom';
import './VideoChat.css';

function VideoChat() {
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
            <video src="/api/placeholder/1280/720" autoPlay muted></video>
            <div className="participant-name">Participant 1</div>
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