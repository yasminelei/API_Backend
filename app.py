from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow React frontend to access this API

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').lower()

    if user_message == 'hi':
        reply = 'Good day, Yasmine! How can I help you?'
    else:
        reply = "I'm sorry, but I don't understand what you are trying to say."

    return jsonify({'reply': reply})

@app.route('/')
def index():
    return "Flask server is running!"

if __name__ == '__main__':
    app.run(debug=True)
