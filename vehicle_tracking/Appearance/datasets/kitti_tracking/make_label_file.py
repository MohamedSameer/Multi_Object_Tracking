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


def main(label_type, time_gap=1, min_size=5000):
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)

    wildcard = '/*/' if label_type == 'train' else '/*/'
    dataset_path = '/media/msis_dasol/1TB/dataset/kitti_tracking_test/'
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

                if class_name == 'Car' or class_name == 'Van' or class_name == 'Truck' or class_name == 'Cyclist' or class_name == 'Pedestrian':
                    if size < min_size:
                        continue

                    _frame = str(frame).zfill(6)
                    _img_frame = image_path + _frame + '.png'

                    imNum = imageSeq[_img_frame]

                    if _frame not in data_set[video_id]:
                        data_set[video_id][_frame] = []

                    classInd = classes[class_name]

                    object = [int(imNum), int(track_id), int(classInd), int(float(x1)),int(float(y1)),int(float(x2)),int(float(y2))]

                    data_set[video_id][_frame].append(object)

                    if imNum % 100 == 0:
                        print('imNum %d of %d = %.2f%%' % (imNum, totalImages, imNum * 100.0 / totalImages))

    print('Generating Triplet Siamese dataset...')
    video_id_list = list(data_set.keys())
    video_id_list.sort()

    for video_id in video_id_list:
        frame_list = list(data_set[str(video_id)].keys())
        frame_list.sort()

        for main_idx in range(len(frame_list)):
            main_frame = frame_list[main_idx]

            for target_idx in range(1, time_gap+1):
                target_frame = str(int(main_frame) + target_idx).zfill(6)

                if target_frame not in frame_list:
                    continue

                # get main object list in main frame
                main_object_list = data_set[video_id][main_frame]

                # get target object list in target frame
                target_object_list = data_set[video_id][target_frame]

                # write dataset
                for main_object in main_object_list:
                    main_img_seq, main_track_id, main_class_id, main_x1, main_y1, main_x2, main_y2 = main_object

                    pos_index = -1

                    # get index of positive object in target_object_list
                    for index, target_object in enumerate(target_object_list):
                        target_img_seq, target_track_id, target_class_id, target_x1, target_y1, target_x2, target_y2 = target_object

                        # check whether it has same track_id
                        if main_track_id == target_track_id and main_class_id == target_class_id:
                            pos_index = index
                            break

                    if pos_index == -1:
                        continue

                    # get positive object data
                    pos_img_seq, pos_track_id, pos_class_id, pos_x1, pos_y1, pos_x2, pos_y2 = target_object_list[pos_index]
                    # target_object_list.remove(target_object_list[pos_index])

                    if len(target_object_list) < 1:
                        continue

                    for neg_idx, neg_object in enumerate(target_object_list):

                        if neg_idx == pos_index:
                            continue

                        neg_img_seq, neg_track_id, neg_class_id, neg_x1, neg_y1, neg_x2, neg_y2 = neg_object

                        bbox = [int(video_id), int(main_img_seq), int(pos_img_seq), int(main_track_id), int(neg_track_id),
                                int(main_x1), int(main_y1), int(main_x2), int(main_y2),
                                int(pos_x1), int(pos_y1), int(pos_x2), int(pos_y2),
                                int(neg_x1), int(neg_y1), int(neg_x2), int(neg_y2)]

                        bboxes.append(bbox)

                        if DEBUG:

                            main_frame_path = image_path_list[main_img_seq]
                            pos_frame_path  = image_path_list[pos_img_seq]
                            neg_frame_path  = image_path_list[neg_img_seq]

                            print('Main : ', main_frame_path, main_track_id, main_class_id, main_x1, main_y1, main_x2,
                                  main_y2)
                            print('Pos  : ', pos_frame_path, pos_track_id, pos_class_id, pos_x1, pos_y1, pos_x2, pos_y2)
                            print('Neg  : ', neg_frame_path, neg_track_id, neg_class_id, neg_x1, neg_y1, neg_x2, neg_y2)

                            print('\n')

                            # load image
                            try:
                                main_img = cv2.imread(main_frame_path)
                                main_height, main_width, _ = main_img.shape

                            except IOError:
                                raise IOError("%s Check input filename." % main_frame_path)

                            try:
                                pos_img = cv2.imread(pos_frame_path)
                                pos_height, pos_width, _ = pos_img.shape

                            except IOError:
                                raise IOError("%s Check input filename." % pos_frame_path)

                            try:
                                neg_img = cv2.imread(neg_frame_path)
                                neg_height, neg_width, _ = neg_img.shape

                            except IOError:
                                raise IOError("%s Check input filename." % neg_frame_path)

                            if main_track_id not in trackColor:
                                trackColor[main_track_id] = [random.random() * 255 for _ in range(3)]

                            if neg_track_id not in trackColor:
                                trackColor[neg_track_id] = [random.random() * 255 for _ in range(3)]


                            drawing.drawRect(main_img, [main_x1, main_y1, main_x2, main_y2], 3, trackColor[main_track_id])
                            drawing.drawRect(pos_img, [pos_x1, pos_y1, pos_x2, pos_y2], 3, trackColor[pos_track_id])
                            drawing.drawRect(pos_img, [neg_x1, neg_y1, neg_x2, neg_y2], 3, trackColor[neg_track_id])

                            cv2.imshow('main_frame', main_img)
                            cv2.imshow('target_frame', pos_img)

                            cv2.waitKey(0)



    bboxes = np.array(bboxes)

    # [video_id, main_img_seq, pos_img_seq, main_track_id, neg_track_id, main_x1, main_y1, main_x2, main_y2, pos_x1, pos_y1, pos_x2, pos_y2, neg_x1, neg_y1, neg_x2, neg_y2]

    # Reorder by video_id, then track_id, then video image number so all labels for a single track are next to each other.
    # This only matters if a single image could have multiple tracks.
    #order = np.lexsort((bboxes[:,0], bboxes[:,3], bboxes[:,2]))
    #bboxes = bboxes[order,:]

#   for i in bboxes:
#        print(i)
    if not DEBUG:
        np.save('labels/' + label_type + '/labels.npy', bboxes)
        np.savetxt('labels/' + label_type + '/labels.txt', bboxes)



if __name__ == '__main__':
    main('train')
    #main('val')

