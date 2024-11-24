from flask import Flask, request, jsonify, render_template

from PIL import Image
from skimage import io
import numpy as np
import io as sio
from StoreShelfClassificator import StoreShelfClassificator

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

classificator = StoreShelfClassificator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        if 'file' in request.files:
            file = request.files['file']
        elif request.data:
            file = sio.BytesIO(request.data)
        else:
            return jsonify({'error': 'No file provided'}), 400

        image = io.imread(file)

        result = classificator.classify_image(image)

        if 'file' in request.files:
            return render_template('result.html', result=result)
        else:
            return jsonify({'prediction': result}), 200, {'Content-Type': 'application/json; charset=utf-8'}
    except Exception as e:
        app.logger.error(f"Error during classification: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
