# !pip install flask
from flask import Flask, request, jsonify

app = Flask(__name__)
@app.route('/hello', methods=['GET'])
def hello():
    return 'Hello World'

@app.route('/param', methods=['GET'])
def param():
    param = request.args.get('name')
    return f'Hello {param}' 

@app.route('/test', methods=['GET'])
def test():
    return f'Hello test'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)