import os
from flask import Flask, request, render_template
import numpy as np
from keras.models import load_model
from keras.optimizers import Adam
from tensorflow.keras.utils import load_img, img_to_array
import cv2



app = Flask(__name__)

# Set the path to the local folder to save uploaded images
UPLOAD_FOLDER = 'static/uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

dic = {0 : 'Mild', 1 : 'Moderate', 2 : 'No_DR', 3 : 'Prolliferate_DR', 4 : 'Severe'}

@app.route("/", methods=["GET", "POST"])
def homepage():
    return render_template('homepage.html')

#### Machine Learning Code
img_size_x=224
img_size_y=224
model = load_model('model.h5')
# optimizer = Adam(learning_rate=0.001)  # Adjust learning rate as needed
# model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
def predict_label(img_path):
    img=cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    resized=cv2.resize(gray,(img_size_x,img_size_y)) 
    i = img_to_array(resized)/255.0
    i = i.reshape(1,img_size_x,img_size_y,1)
    predict_x=model.predict(i) 
    p=np.argmax(predict_x,axis=1)
    return dic[p[0]]

@app.route("/upload", methods=["GET", "POST"])
def upload():
    p = None
    img_path = None
    if request.method == "POST" and 'photo' in request.files:
        # Get the uploaded file from the form data
        file = request.files['photo']

        # Save the file to the local folder
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            img_path = file_path

            print(f"img_path: {img_path}")  # Print the value of img_path for debugging purposes

        p = predict_label(img_path)

    cp = str(p).lower() if p is not None else ""
    src = img_path.replace('\\', '/') if img_path is not None else ""

    print(f"src: {src}")  # Print the value of src for debugging purposes

    return render_template('upload.html', cp=cp, src=src)





if __name__ == "__main__":
    # Create the upload folder if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run(debug=True)
