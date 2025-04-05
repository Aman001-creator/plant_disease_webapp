from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello from Plant Disease Detection Model!"

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))  # ← use PORT from environment
    app.run(host='0.0.0.0', port=port, debug=True)  # ← bind to 0.0.0.0
