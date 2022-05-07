import sys
import datetime
from datetime import datetime
from PyQt5.QtCore import pyqtSlot, Qt, QDate
from PyQt5.uic import loadUi
from PyQt5.QtGui import QImage, QPixmap, QMovie
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog

import simpleaudio as sa    # library that's made for playing back WAV files

import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import core.utils as utils
from core.functions import *
from tensorflow.python.saved_model import tag_constants     # Common tags used for graphs in SavedModel.
import numpy as np



class Window(QDialog):
    def __init__(self):
        super(Window, self).__init__()
        loadUi("VideoSync.ui", self)

        self.WebcamButton.clicked.connect(self.STARTWebCam)
        self.VideoButton.clicked.connect(self.STARTVideo)

        self.movie = QMovie("data/images/demo.gif")
        self.label_3.setMovie(self.movie)
        self.movie.start()

    @pyqtSlot()
    def STARTWebCam(self):
        cap = cv2.VideoCapture(1)

        saved_model_loaded = tf.saved_model.load('final-anomaly-model/saved_model', tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

        # Progress Bar
        for i in range(101):
            time.sleep(0.05)
            self.ProcessBar.setValue(i)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        date = datetime.now()
        out = cv2.VideoWriter(f'data/video/Video_{date}.avi', fourcc, 10, (640, 480))

        
        cap.release()
        cv2.destroyAllWindows()


app = QApplication(sys.argv)
window = Window()
window.show()
try:
    sys.exit(app.exec_())
except:
    print("Existing!!!")
