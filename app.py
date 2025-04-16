from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from youtube_qa import answer_question
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/answer', methods=['POST'])
def get_answer():
    try:
        data = request.json
        video_url = data.get('video_url')
        question = data.get('question')
        
        if not video_url or not question:
            return jsonify({'error': 'Missing video_url or question'}), 400
            
        answer = answer_question(video_url, question)
        return jsonify({'answer': answer})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 