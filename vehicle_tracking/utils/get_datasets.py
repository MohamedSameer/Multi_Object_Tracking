import numpy as np
import glob
import os

def get_data_for_dataset(dataset_name, mode):
    # Implement this for each dataset.
    if dataset_name == 'kitti_tracking':
        datadir = os.path.join(
            os.path.dirname(__file__), os.path.pardir,
            'datasets',
            'kitti_tracking')
        gt = np.load(datadir + '/labels/' + mode + '/labels.npy')
        image_paths = [line.strip() for line in open(datadir + '/labels/' + mode + '/image_names.txt')]
        return {
            'gt': gt,
            'image_paths': image_paths,
        }



if __name__ == '__main__':
    print(os.path.pardir)
    get_data_for_dataset('kitti_tracking', 'train')
