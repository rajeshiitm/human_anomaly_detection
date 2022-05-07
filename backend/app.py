from flask import Flask, render_template, Response,  request, session, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import os
import sys
import tensorflow as tf

__author__ = 'Rajesh'
__source__ = ''

app = Flask(__name__)
UPLOAD_FOLDER = "C:/Users/91895/Downloads/human_anomaly_detection/temp"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
model = tf.keras.models.load_model('C:/Users/91895/Downloads/human_anomaly_detection/saved_model')

@app.route("/")
def index():
  return render_template("index.html")

@app.route("/about")
def about():
  return render_template("about.html")

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']

      filename = secure_filename(f.filename)
      print(filename)

      filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      print(filepath)
      f.save(filepath)
      prediction = model.predict(filepath)
      os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      return prediction
      # return render_template("uploaded.html", display_detection = filename, fname = filename)      

if __name__ == '__main__':
   app.run(port=4000, debug=True)