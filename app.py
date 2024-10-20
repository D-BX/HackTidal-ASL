from flask import Flask, request, jsonify, send_from_directory, render_template

from textFromText import translate

from flask_cors import CORS

app = Flask(__name__, static_folder='public', template_folder='public')
CORS(app, resources={r"/translate": {"origins": "http://localhost:3000"}})


# Serve React App (index.html from the public folder)
@app.route('/')
def serve_react_app():
    return render_template('index.html')

# Serve static files like JS, CSS, and images
@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory('public', path)

@app.route('/translate', methods=['POST'])
def translate_text():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        original_text = data['text']
        translated_text = translate(original_text)
        return jsonify({'translated': translated_text}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
