from flask import Flask, render_template, flash, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
import cv2
from inference import predict


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ['jpg']

app = Flask(__name__)

#header for no caching
@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

#to take the image input
@app.route("/", methods = ['GET','POST'])
def home():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join('static/uploads', 'prediction_image.jpg'))
            return redirect(url_for('predict_upload'))
    return render_template('home.html')

#steps app route
@app.route("/working")
def steps():
    return render_template("working.html")

#the prediction page
@app.route('/predict')
def predict_upload():
    #read the image from static directory   
    image = cv2.imread('static/uploads/prediction_image.jpg')
    #get the predictions
    stage1,stage2,stage3,final = predict(image)

    #write the output to the static folder
    cv2.imwrite('static/result/stage1.jpg',stage1)
    cv2.imwrite('static/result/stage2.jpg',stage2)
    cv2.imwrite('static/result/stage3.jpg',stage3)
    cv2.imwrite('static/result/final.jpg',final)
    return render_template('predict.html')

@app.route('/results')
def result_file():
    return send_from_directory('static/result','result_image.jpg')

    
if __name__ == "__main__":
    app.run(debug=True)