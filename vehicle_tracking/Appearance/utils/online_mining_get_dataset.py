import numpy as np
import glob
import os

def get_data_for_dataset(dataset_name, mode):

    if dataset_name == 'vehicle_identification':
        datadir = os.path.join(
                  os.path.dirname(__file__), os.path.pardir,
                'datasets',
                'vehicle_identification')
        image_paths = [line.strip() for line in open(datadir + '/labels/' + mode + '/image_names.txt')]
        gt = np.load(datadir + '/labels/' + mode + '/labels.npy')

        #for line in image_paths:
        #    file_name, label = line.split(' ')

        return {
            'gt': gt,
            'image_paths': image_paths,
        }

    if dataset_name == 'vehicle_identification_with_background':
        datadir = os.path.join(
                  os.path.dirname(__file__), os.path.pardir,
                'datasets',
                'vehicle_identification_with_background')
        image_paths = [line.strip() for line in open(datadir + '/labels/' + mode + '/image_names.txt')]
        gt = np.load(datadir + '/labels/' + mode + '/labels.npy')

        #for line in image_paths:
        #    file_name, label = line.split(' ')

        return {
            'gt': gt,
            'image_paths': image_paths,
        }

if __name__ == '__main__':
    print(os.path.pardir)
    get_data_for_dataset('vehicle_identification', 'train')