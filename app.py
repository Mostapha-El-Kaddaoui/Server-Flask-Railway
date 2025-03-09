from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/phonecall', methods=['POST'])
def PhoneCall():
    data = request.json
    if data and data.get('message') == 'hello':
        return jsonify({'response': 'world'})
    return jsonify({'response': 'Invalid request'}), 400

if __name__ == '__main__':
    app.run(debug=True)
