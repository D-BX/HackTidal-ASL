import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [inputText, setInputText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [isTranslating, setIsTranslating] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleSubmit = () => {
    if (selectedFile) {
      console.log("File selected:", selectedFile.name);
      // Upload logic goes here
    } else {
      console.log("No file selected");
    }
  };

  const triggerFileInput = () => {
    document.getElementById('file-input').click();
  };

  const handleInputChange = (event) => {
    setInputText(event.target.value);
  };

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
    } catch (err) {
      setError(err.message);
    } finally {
      setIsTranslating(false);
    }
  };

  return (
    <div className="App">
      {/* ASLINATOR Section */}
      <div className="section blue">
        <img className="goku" src="/IMG_1932.JPG" alt="ASL Clipart" />
        <h1 className="title">ASLINATOR</h1>
        <p className="subtitle">ASL modernized to futuristic needs!</p>
        <p className="scroll-message">Scroll down to see what we mean.</p>
        <div className="arrow-container">
          <div className="arrow"></div>
        </div>
        <img className="asl-hand" src="/aslClipartnobg.png" alt="ASL Clipart" />
      </div>

      {/* AI Section */}
      <div className="section white">
        <p className="main-message">
          We live in a world where the power of AI is unimaginable.
        </p>
        <p className="sub-message">
          How can we leave those with disabilities behind? We want to be able to translate ASL from signs to English, furthering inclusivity.
        </p>
        <p className="sub-message">
          Being able to go from ASL to formed English sentences hasn't been seen before but our aim is to make it happen.
        </p>
        <div className="arrow-container">
          <div className="arrow"></div>
        </div>
      </div>

      {/* Workflow Section */}
      <div className="section turquoise">
        <h2>Here's how it works:</h2>
        <div className="workflow">
          <div>ASL Sentence</div>
          <div className="arrow-right"></div>
          <div>DINO V2</div>
          <div className="arrow-right"></div>
          <div>Direct ASL Transcription</div>
          <div className="arrow-right"></div>
          <div>LLM</div>
          <div className="arrow-right"></div>
          <div>Text to Speech/Text</div>
        </div>
        <div className="asl-demonstration">
          <img src="/handsigns.gif" alt="ASL demonstration" className="asl-demo" />
        </div>
      </div>

      {/* File Upload Section */}
      <div className="section turquoise">
        <div className="file-upload">
          <p>Drag and drop a file or browse to upload.</p>
          <input id="file-input" type="file" onChange={handleFileChange} style={{ display: 'none' }} />
          <button className="minimal-button" onClick={triggerFileInput}>Choose File</button>
          {selectedFile && <p>Selected file: {selectedFile.name}</p>}
          <button className="minimal-button" onClick={handleSubmit}>Upload</button>
        </div>
      </div>

      {/* Translation Section */}
      <div className="section lightgrey">
        <h2>Translate Text</h2>
        <textarea
          value={inputText}
          onChange={handleInputChange}
          placeholder="Enter text to translate..."
          rows="4"
          cols="50"
        />
        <br />
        <button className="minimal-button" onClick={handleTranslate} disabled={isTranslating}>
          {isTranslating ? 'Translating...' : 'Translate'}
        </button>
        {error && <p className="error-message">{error}</p>}
        {translatedText && (
          <div className="translated-section">
            <h3>Translated Text:</h3>
            <p>{translatedText}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
