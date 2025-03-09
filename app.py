from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('https://web-production-2b055.up.railway.app/hello', methods=['POST'])
def hello():
    data = request.json
    if data and data.get('message') == 'hello':
        return jsonify({'response': 'world'})
    return jsonify({'response': 'Invalid request'}), 400

if __name__ == '__main__':
    app.run(debug=True)
