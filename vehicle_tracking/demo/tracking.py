import keras
import tensorflow as tf
from model.yolo3 import YOLOv3
from PIL import Image
from utils.tracker import Tracker, IdColor, drawID
from Appearance.model.tf_customVGG16 import customVGG16
import os
import utils.tf_util as tf_util
import cv2
import numpy as np
import time
from utils import drawing


class_list = ['truck', 'car', 'bus', 'motorbike']

IMG_SIZE = 160
FEATURE_SIZE = 256
MEM_SIZE     = 5
DATA_PATH = '/media/msis_dasol/1TB/dataset/tracking_video/video6/'

os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)

# gpu growth
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0

# load data
print('Load images to tracking')
image_list = os.listdir(DATA_PATH)
image_list.sort()
image_path = DATA_PATH + image_list[0]
image = Image.open(image_path)
height = image.height
width  = image.width

# assign unique color to each id
id_name = []
for i in range(0, 1000):
    id_name.append(i)
id_color = IdColor(id_name)

# assign session
tf.Graph().as_default()
sess = tf.Session(config=config)
keras.backend.set_session(sess)

# load tracker
tracker = Tracker(sess, MEM_SIZE, IMG_SIZE, FEATURE_SIZE, ori_height=height, ori_width=width, iou_threshold=0.3, kl_threshold=0.6)
sess.run(tf.global_variables_initializer())
log_dir = '/home/msis_dasol/master_thesis/RAN/for_paper/VGG16_skip_connection/memsize_5'
tf_util.restore_from_dir(sess, os.path.join(log_dir, 'checkpoints'))

# load detector
yolov3 = YOLOv3(sess)
total_tracking_obejct = 0

var_sizes = [np.product(list(map(int, v.shape))) * v.dtype.size for v in
                 tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
print('total parameter size :', sum(var_sizes) / (1024 ** 2), 'MB')

print('Start tracking..')
for image_name in image_list:
    # load image
    image_path = DATA_PATH + image_name
    image = Image.open(image_path)
    area = (0, 20, width, height - 300)
    image = image.crop(area)
    print(image_name)
    trk_image = image.copy()

    # check time
    detect_start_time = time.time()

    # detect object
    bboxes, scores, classes, name_list = yolov3.detect_image(image, debug=False)
    bboxes2 = []; scores2 = []; classes2 =[]
    # filter out useless detected classes
    detections = []
    for idx, class_name in enumerate(classes):
        if class_name in class_list:
            y1, x1, y2, x2 = bboxes[idx]

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 >= width:
                x2 = width -1
            if y2 >= height:
                y2 = height -1

            if (x2 - x1) * (y2 - y1) < 2500:
                continue

            copy_image = np.array(image).copy()
            object = copy_image[y1:y2, x1:x2].copy()
            object = cv2.resize(object, (IMG_SIZE, IMG_SIZE))
            object = object / 255.0

            f_x1 = int(x1) / width
            f_y1 = int(y1) / height
            f_x2 = int(x2) / width
            f_y2 = int(y2) / height

            detection = [object, [f_x1, f_y1, f_x2, f_y2], classes[idx]]
            detections.append(detection)

            # for drawing detections
            bboxes2.append([y1, x1, y2, x2])
            scores2.append(scores[idx])
            classes2.append(classes[idx])

            #cv2.imshow('object_debug', object)
            #cv2.waitKey(0)

    # debug
    for idx, detection in enumerate(detections):
        print(idx, detection[1], detection[2])

    # check time
    detect_end_time = time.time()
    print('Detecting speed : %.3f FPS' % (1 / (detect_end_time - detect_start_time)))

    # draw detected object on image
    r_image = yolov3.draw_bboxes(image, bboxes2, scores2, classes2)

    cv_image = np.array(r_image)[:, :, ::-1].copy()
    #cv2.resize(cv_image, (300, 600))
    cv2.imshow('detect result', cv_image)

    # tracking
    tracking_start_time = time.time()
    tracklets           = tracker.track(bboxes=detections)
    total_tracking_obejct += len(tracklets)
    tracking_end_time   = time.time()

    print('Tracking speed : %.3f FPS' % (1 / (tracking_end_time - tracking_start_time)))

    if(len(tracklets) >= 1):
        trk_image = drawID(trk_image, tracklets, id_color)
    cv_img_tracking = np.array(trk_image)[:, :, ::-1].copy()

    cv2.imwrite('result/video6/tracking/'+image_name, cv_img_tracking)
    cv2.imwrite('result/video6/detection/' + image_name, cv_image)
    # show detections

    # show tracklets
    cv2.imshow('tracking result', cv_img_tracking)

    cv2.waitKey(1)

print(total_tracking_obejct)



