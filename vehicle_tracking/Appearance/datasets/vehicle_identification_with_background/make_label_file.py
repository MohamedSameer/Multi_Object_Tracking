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


def main(label_type, time_gap=1, min_size=2000):
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)

    wildcard = '/*/' if label_type == 'train' else '/*/'
    dataset_path = '/media/msis_dasol/1TB/dataset/vehicle_identification_with_background/'

    if not os.path.exists(os.path.join('labels', label_type)):
        os.makedirs(os.path.join('labels', label_type))
    imageNameFile = open('labels/' + label_type + '/image_names.txt', 'w')
    labelFile     = open('labels/' + label_type + '/labels.txt', 'w')

    # classes
    class_list = os.listdir(dataset_path)

    last_frame = 0
    totalImages = len(glob.glob(dataset_path + wildcard + '*.png')) + len(glob.glob(dataset_path + wildcard + '*.jpg'))

    print('totalImages', totalImages)

    bboxes = []
    imageSeq = {}

    # Load dataset
    print('Load label data...')
    for vv, class_name in enumerate(class_list):

        image_path = dataset_path  + class_name + '/'
        images = sorted(glob.glob(image_path + '*.png')) + sorted(glob.glob(image_path + '*.jpg'))

        if vv != 0:
            last_frame = last_frame + 1

        # save image names
        for idx, name in enumerate(images):
            imageNameFile.write(name + '\n')
            imageSeq[name] = last_frame + idx

            bbox = [imageSeq[name], vv]
            bboxes.append(bbox)


        last_frame = last_frame + len(images) -1

    bboxes = np.array(bboxes)

    # [video_id, main_img_seq, pos_img_seq, main_track_id, neg_track_id, main_x1, main_y1, main_x2, main_y2, pos_x1, pos_y1, pos_x2, pos_y2, neg_x1, neg_y1, neg_x2, neg_y2]

    # Reorder by video_id, then track_id, then video image number so all labels for a single track are next to each other.
    # This only matters if a single image could have multiple tracks.
    # order = np.lexsort((bboxes[:,0], bboxes[:,3], bboxes[:,2]))
    # bboxes = bboxes[order,:]

    #   for i in bboxes:
    #        print(i)
    if not DEBUG:
        np.save('labels/' + label_type + '/labels.npy', bboxes)
        np.savetxt('labels/' + label_type + '/labels.txt', bboxes)

if __name__ == '__main__':
    main('train')
    #main('val')

