import React, { useState, useEffect } from 'react';
import './App.css';
import { BrowserRouter as Router, Route, Link, Routes } from 'react-router-dom';
import VideoChat from './VideoChat'; 



function App() {
  const [inputText, setInputText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [isTranslating, setIsTranslating] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
        }
      });
    }, { threshold: 0.1 });

    document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));

    return () => observer.disconnect();
  }, []);

  const handleTranslate = async () => {
    if (!inputText.trim()) {
      setError('Please enter some text to translate.');
      return;
    }

    setIsTranslating(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:5000/translate', {  // Updated URL
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ text: inputText })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Translation failed.');
      }

      const data = await response.json();
      setTranslatedText(data.translated);
      const audioUrl = 'http://localhost:5000' + data.audio_url;
      const audio = new Audio(audioUrl);
      audio.play().catch(err => {
        console.error("Error playing audio:", err);
      });

    } catch (err) {
      setError(err.message);
    } finally {
      setIsTranslating(false);
    }
  };

  return (
    <Router>
      <Routes>
        <Route path="/" element={
          <div className="App">
          {/* ASLINATOR Section */}
          <div className="section blue">
            <h1 className="title fade-in">ASL Translationer</h1>
            <p className="subtitle fade-in">Figuring out the modernization of ASL translation.</p>
            <p className="scroll-message fade-in">Scroll down to see what we mean:</p>
            <div className="arrow-container">
              <div className="arrow"></div>
            </div>
            <img className="asl-hand" src="/aslClipartnobg.png" alt="ASL Clipart" />
          </div>

          {/* AI Section */}
          <div className="section white">
            <p className="main-message fade-in">
              We live in a world where the power of AI is unimaginable.
            </p>
            <p className="sub-message fade-in">
              How can we leave those with disabilities behind? We want to be able to translate ASL from signs to English, furthering inclusivity.
            </p>
            <p className="sub-message fade-in">
              Being able to go from ASL to formed English sentences hasn't been seen before but our aim is to make it happen.
            </p>
            <div className="arrow-container">
              <div className="arrow"></div>
            </div>
          </div>

          {/* Workflow Section */}
          <div className="section turquoise">
            <h1 className="workflow-title fade-in">Here's how it works:</h1>
            <div className="workflow-container fade-in">
                <div className="workflow">
                    <img src="/diagram.png" alt="big diagram" className='diagram' />
                </div>
                <div className="asl-demonstration">
                  <img src="/handsigns.gif" alt="ASL demonstration" className="asl-demo" />
                </div>
              </div>
              <div className="arrow-container">
                <div className="arrow"></div>
              </div>
              <img src="/talkingppl.gif" alt='people talking' className='talkingppl'></img>
              <p className="talkingtext">Text to speech in English</p>

            </div>

            {/* Why We Did It Section */}
            <div className="section why-we-did-it">
              <h1 className="section-title fade-in">Why We Did It</h1>
              <p className="why-description fade-in">
                Our hackathon project is a Live ASL Translator. The goal is to help deaf people communicate more easily in work teams with hearing individuals. Our goal is to translate sign language into speech in near real time, making it easier for people with disabilities to participate fully in the workplace. It promotes inclusivity and accessibility, so everyone can work together without barriers.
              </p>
            </div>
            <div className="arrow-container">
              <div className="arrow"></div>
            </div>
            <Link to="/video-chat" className="modern-button fade-in">
                Join Video Chat
            </Link>
          </div>
        } />
        <Route path="/video-chat" element={<VideoChat />} />
      </Routes>
    </Router>
  );
}

export default App;