import cv2
import numpy as np
import glob
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


def val_provider(img_size):
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)

    wildcard = '/*/'
    dataset_path = '/media/msis_dasol/1TB/dataset/triplet_validation_neg/'

    # classes
    class_list = os.listdir(dataset_path)

    totalImages = len(glob.glob(dataset_path + wildcard + '*.png')) + len(glob.glob(dataset_path + wildcard + '*.jpg'))

    print('totalImages', totalImages)

    #pos_labelFile = open('pos_labels.txt', 'w')
    neg_labelFile = open('neg_labels.txt', 'w')

    data_set = {}

    # Load dataset
    print('Load label data...')
    for vv, class_name in enumerate(class_list):

        image_path = dataset_path  + class_name + '/'
        images = sorted(glob.glob(image_path + '*.png')) + sorted(glob.glob(image_path + '*.jpg'))

        data_set[vv] = []

        for image_name in images:
            data_set[vv].append(image_name)
            """
            image = cv2.imread(image_name)[:, :, ::-1]
            resized_image = cv2.resize(image.copy(), (img_size, img_size))
            resized_image = resized_image / 255.

            gtImages.append(resized_image)
            gtLabels.append(vv)
            """

    # offline triplet mining
    print('Generating Triplet Siamese dataset...')
    video_id_list = list(data_set.keys())
    video_id_list.sort()

    """
    # generate pos set
    for video_id in video_id_list:
        main_list = data_set[video_id]

        # select one main image
        for main_image_path in main_list:

            for pos_image_path in main_list:

                if pos_image_path is main_image_path:
                    continue

                pos_labelFile.write('{},{}\n'.format(main_image_path, pos_image_path))
    """

    for video_id in video_id_list:
        main_list = data_set[video_id]

        # select one main image
        for main_image_path in main_list:

            # select neg
            for neg_video_id in video_id_list:

                if neg_video_id is video_id:
                    continue

                neg_list = data_set[neg_video_id]

                for neg_image_path in neg_list:

                    neg_labelFile.write('{},{}\n'.format(main_image_path,neg_image_path))







if __name__ == '__main__':
    val_provider(img_size=160)
    #main('val')

