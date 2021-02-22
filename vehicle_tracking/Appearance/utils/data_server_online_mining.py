import argparse
import cv2
import numpy as np
import random
import struct
import sys
import threading
import time
import Appearance.utils.online_mining_get_dataset as get_datasets

try:
    import cPickle as pickle
    import SoketServer
except ImportError:
    import pickle
    import socketserver as SocketServer

HOST = 'localhost'

class BatchCacheHandler(SocketServer.BaseRequestHandler, object):
    def handle(self):
        # self.request is the TCP socket connected to the client
        # self.server is used for communication with BatchCacheServer
        # Serves until disconnect.
        while not self.server.shut_down:
            alive = self.request.recv(1024).strip()
            print(alive)
            if len(alive) == 0:
                break
            (key, val) = self.server.get_sample(self.server.batch_cache)
            print('key', key)

            self.server.lock.acquire()
            try:
                # Send the key.
                keyPickle = pickle.dumps(key)
                messageLength = struct.pack('>I', len(keyPickle))
                self.request.sendall(messageLength)
                self.request.sendall(keyPickle)

                for image in val:
                    # Send the actual image.
                    messageLength = struct.pack('>I',len(image[0]))
                    self.request.sendall(messageLength)
                    self.request.sendall(image[0])


                    # Send the shape.
                    messageLength = struct.pack('>I', len(image[1]))
                    self.request.sendall(messageLength)
                    self.request.sendall(image[1])


            finally:
                self.server.lock.release()

class BatchCacheServer:
    def __init__(self, args):
        # Set constants
        self.max_size = args.max_size
        #self.num_unrolls = args.num_unrolls
        self.debug = args.debug
        self.vals = []
        self.keys = []

        # key -> index in vals array
        self.idxs = dict()
        self.data_hits = None

        self.data_lock = threading.Lock()

        # Start the queue monitor.
        self.keep_alive = True
        self.worker = threading.Thread(target=self.__memory_monitor, args=())
        #self.worker.daemon = True
        self.worker.start()

        # Flag for the handler to shut down
        self.shut_down = False

        self.image_paths = []
        # dataset_id, vid_id, track_id, frame_id
        self.all_keys = set()
        self.create_keys()
        #Load the first few samples.
        for i in range(min(int(self.max_size / 2), 32)):
            self.__random_load(force_append=True)

    def __del__(self):
        self.keep_alive = False
        self.worker.join()

    def __memory_monitor(self):
        while self.keep_alive:
            if self.data_hits is not None and np.sum(self.data_hits) > 0:
                self.__random_load()
            time.sleep(0.0001)

    def __random_load(self, force_append=False):
        # First find some data that hasn't been loaded already.
        try:
            # key = [dataset_ind, video_id, main_img_seq, pos_img_seq, main_track_id, neg_track_id]
            key = random.sample(self.all_keys, 1)[0]
            while key in self.idxs:
                key = random.sample(self.all_keys, 1)[0]
            val = self.lookup_func(key)

            # Next check to see if we should append or replace existing data.
            self.data_lock.acquire()
            if len(self.keys) < self.max_size or force_append: # Append
                if self.debug:
                    print('Appending new data. Num keys =', len(self.keys))
                self.vals.append(val)
                self.keys.append(key)
                if self.data_hits is None:
                    self.data_hits = np.zeros(1)
                else:
                    self.data_hits = np.append(self.data_hits, 0)
                self.idxs[key] = len(self.vals) - 1
            else: # Replace
                if np.sum(self.data_hits) == 0:
                    sys.stderr.writ/lookupe(
                            ('Something went horribly wrong. __random_load was '
                                'called and the cache is full, but none of the '
                                'elements have been hit!'))
                    sys.stderr.flush()
                    self.data_lock.release()
                    return
                total_hits = np.sum(self.data_hits)
                i = np.argmax(self.data_hits)
                if self.debug:
                    print('Replacing data. Replacing spot', i)
                del self.idxs[self.keys[i]]
                self.vals[i] = val
                self.keys[i] = key
                self.data_hits[i] = 0
                self.idxs[key] = i
            if self.debug:
                print('Total used elements:', len(self.data_hits[self.data_hits > 0]))
                print(self.data_hits[self.data_hits > 0])
            self.data_lock.release()
        except Exception as ex:
            import traceback
            trace = traceback.format_exc()
            print(trace)
            self.shut_down = True
            self.data_lock.release()
            errorFile = open('error.txt', 'a+')
            errorFile.write('exception in __random_load %s\n' % str(ex))
            errorFile.write(str(trace))

    def add_dataset(self, dataset_name):
        dataset_ind = len(self.image_paths)
        data = get_datasets.get_data_for_dataset(dataset_name, 'train')
        gt = data['gt']
        num_keys = 0

        # gt = [img_seq, class_id]

        for xx in range(gt.shape[0]):
            start_line = gt[xx,:].astype(int)

            # Add the key.
            # key = [dataset_ind, img_seq, class_id]
            self.all_keys.add((dataset_ind, start_line[0], start_line[1]))
            num_keys += 1

        if self.debug:
            print('#%s keys: %d' % (dataset_name, num_keys))

        image_paths = data['image_paths']
        # Add the array to image_paths. Note that image paths is indexed by the dataset number THEN by the image line.
        self.image_paths.append(image_paths)

    def create_keys(self):
        self.add_dataset('vehicle_identification')
        #self.add_dataset('vehicle_identification_with_background')
        time.sleep(1)

    def lookup_func(self, key):
        images = None

        try:
            images = []
            # key = [dataset_ind, video_id, main_img_seq, pos_img_seq, main_track_id, neg_track_id]
            img_seq  = key[1]
            class_id = key[2]

            if self.debug:
                imageName = self.image_paths[key[0]][img_seq]
                print('Reading image', imageName)

            path = self.image_paths[key[0]][img_seq]
            image = cv2.imread(path)[:, :, ::-1]
            shape = pickle.dumps(image.shape)

            image_string = image.tostring()
            images.append((image_string, shape))

        except Exception as ex:
            import traceback
            trace = traceback.format_exc()
            print(trace)
            errorFile = open('error.txt', 'a+')
            errorFile.write('exception in lookup_func %s\n' % str(ex))
            errorFile.write(str(trace))
        finally:
            return images


    def get_sample(self, batch_cache):
        try:
            batch_cache.data_lock.acquire()
            idx = random.randint(0, len(batch_cache.vals) - 1)
            key = batch_cache.keys[idx]
            val = batch_cache.vals[idx]
            batch_cache.data_hits[idx] += 1

        except Exception as ex:
            import traceback
            trace = traceback.format_exc()
            print(trace)
            errorFile = open('error.txt', 'a+')
            errorFile.write('exception in lookup_func %s\n' % str(ex))
            errorFile.write(str(trace))

        finally:
            batch_cache.data_lock.release()
        return (key, val)


    def serve(self, port):
        if self.debug:
            print('Server starting up')
        handler = SocketServer.TCPServer((HOST, port), BatchCacheHandler)
        handler.get_sample = self.get_sample
        handler.batch_cache = self
        handler.lock = self.data_lock
        handler.shut_down = self.shut_down
        handler.serve_forever()


def main(args):
    server = BatchCacheServer(args)
    server.serve(args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Server for network images.')
    parser.add_argument('-s', '--max_size', action='store', default=100,
            dest='max_size', type=int)
    parser.add_argument('-p', '--port', action='store', default=9985,
            dest='port', type=int)
    parser.add_argument('-d', '--debug', action='store_true', default=True,
            dest='debug')
    args = parser.parse_args()
    main(args)


