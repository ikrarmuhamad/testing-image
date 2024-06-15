from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
from PIL import Image
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Path ke model
model_path = 'model.h5'
model = load_model(model_path)


# Route untuk halaman utama
@app.route('/', methods=['GET'])
def template():
    return render_template('index.html')

# Route untuk prediksi
@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'images', secure_filename(imagefile.filename))
    imagefile.save(file_path)

    img = Image.open(file_path).convert('RGB')
    img = img.resize([224,224])
    img = np.array(img)
    img = np.expand_dims(img, axis=0)

    target_class = ['chicken_curry', 'chicken_wings', 'fried_rice', 'grilled_salmon', 'hamburger',
                    'ice_cream', 'pizza', 'ramen', 'steak', 'sushi']
    pred_prob = model.predict(img)
    pred_class = str(target_class[np.argmax(pred_prob)])
    classification = f'{pred_class}, our confidence level : {np.max(pred_prob) * 100:.2f}%'
    
    return render_template('index.html', prediction=classification)

if __name__ == '__main__':
    app.run(port=3000, debug=True)