from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS

from textFromText import recombin_str
from textFromText import translate
from textToSpeech import genVoice

from transformers import AutoModel

from torchvision import transforms
from flask_socketio import SocketIO, emit
import base64
from io import BytesIO
from PIL import Image
import os
import torch.nn as nn
import time
import torch
from collections import Counter

total_str = ""
secret_key = os.urandom(24)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dino = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
dino_dim = 768

active_inference = False
class CharacterPredictor:
    def __init__(self, window_size=10, threshold=0.8):
        """
        Initializes the CharacterPredictor with a specified window size and threshold.

        :param window_size: Number of recent guesses to consider for stabilization.
        :param threshold: Minimum frequency ratio for a character to be considered stable.
        """
        self.window_size = window_size
        self.threshold = threshold
        self.recent_guesses = []
        self.last_output_char = None

    def predict(self, guess):
        """
        Processes a new character guess and determines if the character has stabilized.

        :param guess: The current character prediction.
        :return: The stabilized character or None if not yet stabilized or unchanged.
        """
        # Update the list of recent guesses
        self.recent_guesses.append(guess)
        if len(self.recent_guesses) > self.window_size:
            self.recent_guesses.pop(0)

        # Count the frequency of each character in the recent guesses
        counts = Counter(self.recent_guesses)
        total = len(self.recent_guesses)
        most_common_char, freq = counts.most_common(1)[0]
        freq_ratio = freq / total

        # Check if the character has stabilized and is different from the last output
        if freq_ratio >= self.threshold and most_common_char != self.last_output_char:
            self.last_output_char = most_common_char
            return most_common_char
        elif most_common_char != self.last_output_char:
            # Character is changing but not yet stable
            return None
        else:
            # Character hasn't changed since last output
            return None


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
#        self.relu = nn.ReLU()
#        self.linear2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
#        return self.linear2(self.relu(self.linear(x)))
        return self.linear(x)

model = torch.load("../love3.pt")

app = Flask(__name__, static_folder='public', template_folder='public')
app.config['SECRET_KEY'] = secret_key

CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*")  # Initialize Socket.IO with CORS allowed

# Serve React App (index.html from the public folder)
@app.route('/')
def serve_react_app():
    return render_template('index.html')

# Serve static files like JS, CSS, and images
@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory('public', path)


@app.route('/test', methods=['GET'])
def test_route():
    return 'Flask server is running!', 200

@app.route('/speechOutputs/<path:filename>')
def serve_audio_file(filename):
    return send_from_directory('speechOutputs', filename)

def convert_text(original_text):
    global active_inference
    global recombin_str
    try:
        # Translate the text
        print(original_text)
        if not active_inference:
            translated_text = translate(recombin_str)
            active_inference = True
        else:
            translated_text = translate(original_text + original_text + original_text)
        print(f"Translated text: {translated_text}")

        # Generate speech and save as 'speech.mp3' in AUDIO_DIR
        genVoice(translated_text)  # Ensure genVoice saves to 'speech outputs/speech.mp3'
        print(f"Generated speech: speech.mp3")

        # Construct the audio URL with a timestamp to prevent caching
        timestamp = int(time.time())
        audio_url = f'/speechOutputs/speech.mp3?t={timestamp}'

        # Return the translated text and audio URL
        return jsonify({'translated': translated_text, 'audio_url': audio_url}), 200
    except Exception as e:
        print(f"Error during translation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/endRecording', methods=['POST'])
def endRecording():
    global total_str
    print(total_str)
    output = convert_text(total_str)
    total_str = ""
    return output

@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    original_text = data['text']
    return convert_text(original_text)

class CustomCrop:
    def __call__(self, img):
        # Get the original dimensions of the image
        width, height = img.size
        
        # Calculate the cropping dimensions
        new_width = width * 2 // 3  # Middle two-thirds of the width
        new_height = height * 2 // 3  # Bottom two-thirds of the height
        
        left = (width - new_width) // 2  # Left offset for horizontal center crop
        top = height // 2  # Discard top third, so we start from 1/3 of height
        right = left + new_width
        bottom = height

        # Crop the image
        return img.crop((left, top, right, bottom))

def infer_frame(frame):
    preprocess = transforms.Compose([
#        transforms.RandomHorizontalFlip(p=1.0),
        CustomCrop(),
        transforms.Resize((512, 512)),  # Resize the image to a fixed size (example)
        transforms.ToTensor(),  # Resize the image to a fixed size (example)
#        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply the transformations
    image = preprocess(frame)
    image = image.unsqueeze(0)

    with torch.no_grad():
        embeds = dino(image.to(device)).last_hidden_state[:, 0, :].reshape(1, dino_dim)

        predict = model(embeds)
        predicted_class = predict.max(dim=1).indices
        return chr(ord('A') + predicted_class[0])
        from collections import Counter

predictor = CharacterPredictor(window_size=5, threshold=0.6)


@socketio.on('frame')
def handle_frame(data):
    global total_str
    try:
        # Decode the base64 image data
        header, encoded = data.split(',', 1)
        image_data = base64.b64decode(encoded)
        image = Image.open(BytesIO(image_data))

        guess = infer_frame(image)
        result = predictor.predict(guess)

        if result:
            print(f"Found: {result}")
            if result == '[':
                pass
            else:
                total_str += result
        
        # Process the image as needed (e.g., save or run inference)
        # For example, save the image (optional)
#        image.save('received_frame.jpg')
        # Add your processing code here

        # Optionally, emit a response back to the client
        # emit('response', {'status': 'Frame received'})
    except Exception as e:
        print(f"Error processing frame: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
