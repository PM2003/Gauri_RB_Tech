from flask import Flask, request, jsonify, render_template
from PIL import Image
from io import BytesIO
import requests
import base64
import os

app = Flask(__name__)

# Update this to your current Colab / ngrok URL each session
BACKEND_URL = os.environ.get('BACKEND_URL', 'http://localhost:5001/api/transform')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/preds', methods=['POST'])
def predict():
    cloth = request.files.get('cloth')
    model = request.files.get('model')
    if cloth is None or model is None:
        return jsonify({'error': 'Please upload both a cloth image and a person image.'}), 400

    try:
        resp = requests.post(BACKEND_URL, files={'cloth': cloth.stream, 'model': model.stream}, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return render_template('index.html', error=str(e))

    out_img = Image.open(BytesIO(resp.content))
    buf = BytesIO()
    out_img.save(buf, 'PNG')
    encoded = base64.b64encode(buf.getvalue()).decode()
    return render_template('index.html', op=encoded)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
