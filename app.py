from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS

from textFromText import translate
from textToSpeech import genVoice

from flask_socketio import SocketIO, emit
import base64
from io import BytesIO
from PIL import Image
import os

secret_key = os.urandom(24)

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


@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    original_text = data['text']
    try:
        translated_text = translate(original_text)
        genVoice(translated_text)
        return jsonify({'translated': translated_text}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('frame')
def handle_frame(data):
    try:
        # Decode the base64 image data
        header, encoded = data.split(',', 1)
        image_data = base64.b64decode(encoded)
        image = Image.open(BytesIO(image_data))

        # Process the image as needed (e.g., save or run inference)
        # For example, save the image (optional)
        image.save('received_frame.jpg')

        # Add your processing code here
        print("Received a frame")

        # Optionally, emit a response back to the client
        # emit('response', {'status': 'Frame received'})
    except Exception as e:
        print(f"Error processing frame: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
