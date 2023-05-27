import os
from flask import Flask, render_template, request
import base64

app = Flask(__name__)

from inference import get_flower_name

@app.route('/', methods=['GET', 'POST'])

def hello_world():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        try:
            file = request.files['file']
            image = file.read()
            preds = get_flower_name(image_bytes=image)
            base64_image = base64.b64encode(image).decode('utf-8')
            return render_template('results.html', mask_on = preds, imgs=base64_image)
        except:
            print('Exception')
            return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv('PORT', 5000))