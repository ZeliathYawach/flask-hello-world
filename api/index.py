from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello, this is a demo Flask app!'

@app.route('/json')
def json_example():
    data = {'message': 'Hello, JSON response!'}
    return jsonify(data)

@app.route('/greet/<name>')
def greet(name):
    # Capitalizes the first letter of the name for display purposes.
    return f'Hello, {name.capitalize()}!'

if __name__ == '__main__':
    # Running the app in debug mode for development
    app.run(debug=True)
