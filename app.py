# Common tags used for graphs in SavedModel.
from core.functions import *
import core.utils as utils
import time
from flask import Flask, render_template, Response,  request, session, redirect, url_for, send_from_directory, flash, jsonify
import numpy as np
from tensorflow.python.saved_model import tag_constants
from flask_sock import Sock
from werkzeug.utils import secure_filename
import sys
import datetime
import os
import sys
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from PIL import Image

import simpleaudio as sa    # library that's made for playing back WAV files

import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


__author__ = 'Rajesh'
__source__ = ''

app = Flask(__name__)
sock = Sock(app)

UPLOAD_FOLDER = "C:/Users/91895/Downloads/human_anomaly_detection/temp"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

saved_model_loaded = tf.saved_model.load(
    'C:/Users/91895/Downloads/human_anomaly_detection/saved_model', tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']

# # Progress Bar
# for i in range(101):
#    time.sleep(0.05)
#    self.ProcessBar.setValue(i)
# filename = QFileDialog.getOpenFileName(filter="Video (.)")[0]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@sock.route('/uploader')
def upload_file(ws):
    while (True):
        f = ws.receive()
        filename = "temp.mp4"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if os.path.isfile(filepath):
            os.remove(filepath)

        out_file = open(filepath, "wb")  # open for [w]riting as [b]inary
        out_file.write(f)
        out_file.close()

        cap = cv2.VideoCapture(filepath)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        date = datetime.datetime.now()
        out = cv2.VideoWriter(
            f'data/video/Output_{date}.avi', fourcc, 30, (int(cap.get(3)), int(cap.get(4))))
        while (True):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_data = cv2.resize(frame, (768, 768))
                image_data = image_data / 255.
                image_data = image_data[np.newaxis, ...].astype(np.float32)
                start_time = time.time()

                batch_data = tf.constant(image_data)
                pred_bbox = infer(batch_data)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]

                boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                    boxes=tf.reshape(
                        boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                    scores=tf.reshape(
                        pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                    max_output_size_per_class=50,
                    max_total_size=50,
                    iou_threshold=0.45,
                    score_threshold=0.50
                )

                # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
                original_h, original_w, _ = frame.shape
                bboxes = utils.format_boxes(
                    boxes.numpy()[0], original_h, original_w)

                pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[
                    0], valid_detections.numpy()[0]]
                print("pred_bbox :", pred_bbox)

                # read in all class names from config
                class_names = {0: 'normal', 1: 'stealing', 2: 'pickpocketing',
                               3: 'snatching', 4: 'fighting', 5: 'running'}

                # by default allow all classes in .names file
                allowed_classes = list(class_names.values())

                # count objects found
                counted_classes = count_objects(
                    pred_bbox, by_class=True, allowed_classes=allowed_classes)
                # loop through dict and print
                for key, value in counted_classes.items():
                    print("Number of {}s: {}".format(key, value))
                    # Notification.setText(f"{key}".upper())
                    if key != "normal":
                        wave_obj = sa.WaveObject.from_wave_file(
                            "alert.wav")
                        play_obj = wave_obj.play()

                image = utils.draw_bbox(frame, pred_bbox, False, counted_classes, allowed_classes=allowed_classes,
                                        read_plate=False)

                fps = 1.0 / (time.time() - start_time)
                print("FPS: %.2f" % fps)
                result = np.asarray(image)
                cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                img = Image.fromarray(result, 'RGB')
                img.save(os.path.join(app.config['UPLOAD_FOLDER'], "test.png"))
                ws.send(img)
                # displayVideo(result, 1)
                out.write(result)
                if (cv2.waitKey(25) == ord('q')):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))


@sock.route("/summary")
def summary(ws):
    while True:
        data = ws.receive()
        ws.send(int(data)+1)


if __name__ == '__main__':
    app.run(port=4000, debug=True)
