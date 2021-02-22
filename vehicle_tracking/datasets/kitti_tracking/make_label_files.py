import cv2
import numpy as np
import glob
import xml.etree.ElementTree as ET
import time
import random
import os
import sys

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(
    basedir,
    os.path.pardir,
    os.path.pardir,
    os.path.pardir)))

from Appearance.utils import drawing

DEBUG = False

os.environ['CUDA_VISIBLE_DEVICES'] = str(1)


def main(label_type, min_size=2500):
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)

    wildcard = '/*/' if label_type == 'train' else '/*/'
    dataset_path = '/media/msis_dasol/1TB/dataset/kitti_tracking/'
    annotationPath = dataset_path + 'label/'
    imagePath = dataset_path + 'image/'


    if not os.path.exists(os.path.join('labels', label_type)):
        os.makedirs(os.path.join('labels', label_type))
    imageNameFile = open('labels/' + label_type + '/image_names.txt', 'w')

    videos = sorted(glob.glob(annotationPath + '*.txt'))

    bboxes = []
    last_frame = 0
    totalImages = len(glob.glob(imagePath + wildcard + '*.png'))

    print('totalImages', totalImages)
    classes = {
            'Person':0,
            'Tram': 1,
            'Van': 2,
            'Person_sitting': 3,
            'Truck': 4,
            'Misc': 5,
            'Cyclist': 6,
            'Car': 7,
            'Pedestrian': 8
            }

    data_set = {}

    if DEBUG:
        image_path_list =[]

    # Load dataset
    print('Load label data...')
    for vv,video in enumerate(videos):
        labels = sorted(glob.glob(video + '*.xml'))
        image_path = video.replace('label', 'image').replace('.txt', '/')
        images = sorted(glob.glob(image_path + '*.png'))
        trackColor = dict()

        video_id = video.strip().split('/')[-1].replace('.txt','')
        data_set[video_id] = {}

        if vv != 0:
            last_frame = last_frame + 1

        imageSeq = {}

        # save image names
        for idx, name in enumerate(images):
            if not DEBUG:
                imageNameFile.write(name + '\n')
            imageSeq[name] = last_frame + idx

            if DEBUG:
                image_path_list.append(name)

        last_frame = last_frame + len(images) -1

        with open(video, 'r') as f:
            for l in f.readlines():
                frame, track_id, class_name, truncated, occluded, alpha, x1, y1, x2, y2, _, _, _, _, _, _, _ = l.strip().split(' ')

                # calculate size of objects
                size = (int(float(x2)) - int(float(x1))) * (int(float(y2)) - int(float(y1)))

                #if class_name == 'DontCare':
                #    continue

                #if class_name == 'Car' or class_name == 'Van' or class_name == 'Truck' or class_name =='Cyclist' or class_name == 'Pedestrian':
                if class_name == 'Car' or class_name == 'Van' or class_name == 'Truck':
                    if size < min_size:
                        continue

                    _frame = str(frame).zfill(6)
                    _img_frame = image_path + _frame + '.png'

                    imNum = imageSeq[_img_frame]

                    if _frame not in data_set[video_id]:
                        data_set[video_id][_frame] = []

                    classInd = classes[class_name]

                    object = [int(imNum), int(track_id), int(classInd), int(float(x1)),int(float(y1)),int(float(x2)),int(float(y2)), int(occluded)]

                    data_set[video_id][_frame].append(object)

                    if imNum % 100 == 0:
                        print('Loading imNum %d of %d = %.2f%%' % (imNum, totalImages, imNum * 100.0 / totalImages))

    print('Generating Kitti tracking dataset...')
    video_id_list = list(data_set.keys())
    video_id_list.sort()

    for video_id in video_id_list:
        frame_list = list(data_set[str(video_id)].keys())
        frame_list.sort()

        for frame in frame_list:

            object_list = data_set[video_id][frame]

            if DEBUG:
                img_seq, track_id, class_id, x1, y1, x2, y2, occluded = object_list[0]
                debug_image_path = image_path_list[img_seq]

                # load image
                try:
                    debug_image = cv2.imread(debug_image_path)
                    #main_height, main_width, _ = debug_image.shape
                except IOError:
                    raise IOError("%s Check input filename." % debug_image_path)

            for object in object_list:
                img_seq, track_id, class_id, x1, y1, x2, y2, occluded = object

                bbox = [int(x1), int(y1), int(x2), int(y2), int(video_id), int(track_id), int(img_seq), int(class_id), int(occluded)]
                bboxes.append(bbox)

                if DEBUG:

                    if track_id not in trackColor:
                        trackColor[track_id] = [random.random() * 255 for _ in range(3)]

                    drawing.drawRect(debug_image, [x1, y1, x2, y2], 2, trackColor[track_id])



            if DEBUG:
                cv2.imshow('debug_image', debug_image)
                cv2.waitKey(100)


    bboxes = np.array(bboxes)

    # Reorder by video_id, then track_id, then video image number so all labels for a single track are next to each other.
    # This only matters if a single image could have multiple tracks.
    order = np.lexsort((bboxes[:,6], bboxes[:,5], bboxes[:,4]))
    bboxes = bboxes[order,:]

#   for i in bboxes:
#        print(i)
    if not DEBUG:
        np.save('labels/' + label_type + '/labels.npy', bboxes)
        np.savetxt('labels/' + label_type + '/labels.txt', bboxes)


if __name__ == '__main__':
    main('train')
    #main('val')

