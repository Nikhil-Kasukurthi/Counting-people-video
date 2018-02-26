# Tornado Libraries
import tornado.ioloop
import tornado.web
import tornado.httpserver
import tornado.ioloop
import tornado.options


import numpy as np
import math
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json
import cv2
from markupsafe import Markup

if tf.__version__ < '1.4.0':
    raise ImportError(
        'Please upgrade your tensorflow installation to v1.4.* or later!')

# Custom files
sys.path.insert(0, 'utils')
import label_map_util
import people_class_util as class_utils
import visualization_utils as vis_util


MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# MODEL_NAME = 'faster_rcnn_nas_lowproposals_coco_2017_11_08'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used
# for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'utils/person_label_map.pbtxt'

NUM_CLASSES = 1
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def download_model():
    if not os.path.exists(MODEL_FILE):
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())
download_model()

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def annotate_video(path_to_video):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            cap = cv2.VideoCapture(path_to_video)
        # cap = cv2.VideoCapture(
            # 'The.Big.Sick.2017.720p.BluRay.H264.AAC-RARBG.mp4')
            framerate = cap.get(cv2.CAP_PROP_FPS)
            framecount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            time_ = int(math.floor(framecount // framerate))
            if time_ / 60 > 5:
                jump = 30
            elif time_ >= 60:
                jump = 15
            else:
                jump = 5
            # print(time_/60)
            frames_extract = [i * framerate for i in range(0, time_, jump)]
            # print(len(frames_extract))

            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name(
                'detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name(
                'detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name(
                'detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')

            annotations = {}
            annotations['frames'] = []
            annotations['videoShape'] = {}
            annotations['videoShape']['height'] = cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT)
            annotations['videoShape']['width'] = cap.get(
                cv2.CAP_PROP_FRAME_WIDTH)

            for frame_number in frames_extract:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if ret:
                    image_np_expanded = np.expand_dims(frame, axis=0)
                    annotations_frame = {}
                    annotations_frame['time'] = int(
                        frame_number // framerate * 100)
                    (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores,
                         detection_classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})

                    annotations_frame['annotations'], count = (
                        class_utils.get_class(np.squeeze(classes).astype(np.int32),
                                              category_index, np.squeeze(
                                                  boxes),
                                              np.squeeze(scores)))
                    annotations_frame['count'] = count
                    annotations['frames'].append(annotations_frame)

                    # cv2.imshow('frame', frame)
                    print(frame_number)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
            # print(json.dumps(annotations))
            # TODO - Proper naming for the json file to save
            with open('time.json', 'w') as file:
                json.dump(annotations, file)
            cap.release()
            # # out.release()
            cv2.destroyAllWindows()

    return annotations


class UploadHandler(tornado.web.RequestHandler):

    def get(self):
        self.render('upload.html', title='this')

    def post(self):
        file1 = self.request.files['file1'][0]
        original_fname = file1['filename']
        extension = os.path.splitext(original_fname)[1]
        output_file = open("uploads/"
                           + original_fname, 'wb')
        output_file.write(file1['body'])
        results = annotate_video("uploads/"
                                 + original_fname)
        
        self.render('vis_line.html', response = json.dumps(results))
        # self.write({'Response': True,
        #             'results': results})


app = tornado.web.Application([
    (r'/upload', UploadHandler)
], static_path='./static', debug=True)
app.listen(8001)
tornado.ioloop.IOLoop.instance().start()
