import React, { useState } from 'react';
import './App.css';

  function App() {
    const [selectedFile, setSelectedFile] = useState(null);

    const handleFileChange = (event) => {
      setSelectedFile(event.target.files[0]);
    };
  
    const handleSubmit = () => {
      if (selectedFile) {
        console.log("File selected:", selectedFile.name);
        // thomas upload from here big dawg
      } else {
        console.log("No file selected");
      }
    };

    const triggerFileInput = () => {
      document.getElementById('file-input').click();
    };

    return (
      <div className="App">
        <div className="section blue">
          <h1 className="title">ASLINATOR</h1>
          <p className="subtitle">ASL modernized to futuristic needs!</p>
          <p className="scroll-message">Scroll down to see what we mean.</p>
          <div className="arrow-container">
          <div className="arrow"></div>
          &nbsp;
          </div>
          <img className="asl-hand" src="/aslClipartnobg.png" alt="ASL Clipart" />
          </div>
  
        <div className="section white">
          <p className="main-message">
            We live in a world where the power of AI is unimaginable.
          </p>
          <p className="sub-message">
            How can we leave those with disabilities behind? We want to be able to translate ASL from signs to English, furthering inclusivity.
          </p>
          <div className="arrow-container">
            <div className="arrow"></div>
          </div>
        </div>
  
        <div className="section turquoise">
          <h2>Here’s how it works:</h2>
          <div className="workflow">
            <div>DINO V2</div>
            <div>Direct ASL Transcription</div>
            <div>Text to Speech</div>
          </div>
          <div className="file-upload">
            <p>Drag and drop a file or browse to upload.</p>
            <input id="file-input" type="file" onChange={handleFileChange} style={{ display: 'none' }} />

            <button className="minimal-button" onClick={triggerFileInput}>Choose File</button>            {selectedFile && <p>Selected file: {selectedFile.name}</p>}
            <button className="minimal-button" onClick={handleSubmit}>Upload</button>
          </div>
        </div>
      </div>
    );
  }

export default App;
