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

        while(cap.isOpened()):
            ret, frame = cap.read()
            if(ret == True):
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
                    boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                    scores=tf.reshape(
                        pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                    max_output_size_per_class=50,
                    max_total_size=50,
                    iou_threshold=0.45,
                    score_threshold=0.50
                )

                # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
                original_h, original_w, _ = frame.shape
                bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

                pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

                # read in all class names from config
                class_names = {0: 'normal', 1: 'stealing', 2: 'pickpocketing', 3: 'snatching', 4: 'fighting', 5:'running'}

                # by default allow all classes in .names file
                allowed_classes = list(class_names.values())

                # count objects found
                counted_classes = count_objects(pred_bbox, by_class=True, allowed_classes=allowed_classes)
                # loop through dict and print
                for key, value in counted_classes.items():
                    print("Number of {}s: {}".format(key, value))
                    self.Notification.setText(f"{key}".upper())
                    if key != "normal":
                        wave_obj = sa.WaveObject.from_wave_file("alert.wav")
                        play_obj = wave_obj.play()
                image = utils.draw_bbox(frame, pred_bbox, False, counted_classes, allowed_classes=allowed_classes,
                                        read_plate=False)

                fps = 1.0 / (time.time() - start_time)
                print("FPS: %.2f" % fps)
                result = np.asarray(image)
                result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)

                self.displayVideo1(result, 1)
                out.write(result)
                if(cv2.waitKey(25) == ord('q')):
                    break
            else:
                break
        out.release()
        cap.release()
        cv2.destroyAllWindows()


app = QApplication(sys.argv)
window = Window()
window.show()
try:
    sys.exit(app.exec_())
except:
    print("Existing!!!")
