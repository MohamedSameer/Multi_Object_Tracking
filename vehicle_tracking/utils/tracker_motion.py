import time
import tensorflow as tf
import os
from model.kl_network import RAN as network
from demo.cnn_model.tf_customVGG16 import customVGG16 as cnnNetwork
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from PIL import Image, ImageDraw, ImageFont

# assign unique color for id
def IdColor(id_name):
    id_color = {}

    for id in id_name:
        color = np.random.randint(0, 255, [3])
        id_color[id] = color

    return id_color

def drawID(img, bboxes, id_color):

    font = ImageFont.truetype(font='../keras_yolo3/font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * img.size[1] + 0.5).astype('int32'))

    thickness = (img.size[0] + img.size[1]) // 400

    for i in range(len(bboxes)):


        predicted_class = bboxes[i][2]
        box             = bboxes[i][0]
        id              = bboxes[i][1]
        x1, y1, x2, y2 = box

        label = '{} {}'.format(predicted_class, id)
        draw = ImageDraw.Draw(img)
        label_size = draw.textsize(label, font)

        top = y1
        left = x1
        bottom = y2
        right = x2

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(img.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(img.size[0], np.floor(right + 0.5).astype('int32'))


        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # assign unique color for each id
        color = id_color[(id)]

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=tuple(color))
        draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=tuple(color))
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    return img

class Tracklet(object):
    count = 0

    def __init__(self, memory_size, feature_size, appearance_feature, motion, label, bbox, height, width):
        # assign id
        self.id = Tracklet.count
        Tracklet.count += 1
        self.memory_size = memory_size

        if memory_size == 1:
            cell_ratio = 1
        else:
            cell_ratio = int(memory_size / 2)

        # data for prediction
        self.appearanceMem              = np.zeros((memory_size, feature_size), dtype=np.float32)
        self.motionMem                  = np.zeros((memory_size, 4), dtype=np.float32)
        self.appearancePrevLstmState    = [np.zeros((1, 256 * cell_ratio)) for _ in range(2)]
        self.motionPrevLstmState        = [np.zeros((1, 4 * memory_size * 2)) for _ in range(2)]

        # init update
        self.appearanceMem[:] = appearance_feature
        self.motionMem[:]     = motion #xywh

        # data for association
        self.predictAppearance = None
        self.predictMotion     = None             # motion = [x1,y1,x2,y2] floating point

        self.label = label

        # tracking association
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.time_since_update = 0
        self.height = height
        self.width  = width


        # cur data
        self.cur_bbox = bbox

    def get_info(self):
        x1, y1, x2, y2 = self.cur_bbox

        x1 = int(x1 * self.width)
        x2 = int(x2 * self.width)
        y1 = int(y1 * self.height)
        y2 = int(y2 * self.height)
        bbox = [x1, y1, x2, y2]
        info = [bbox, self.id, self.label]

        return info

class Tracker(object):
    def __init__(self, sess, memory_size, img_size, feature_size, gpu_id=0 , iou_threshold=0.4, kl_threshold=0.4, ori_height = 1200, ori_width = 300):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

        if memory_size == 1:
            cell_ratio = 1
        else:
            cell_ratio = int(memory_size / 2)

        # build network for association
        self.appearanceMemPlaceholder = tf.placeholder(tf.float32, shape=(1, 1, memory_size, feature_size))
        self.motionMemPlaceholder     = tf.placeholder(tf.float32, shape=(1, 1, memory_size, 4))
        self.appearancePrevLstmState  = tuple([tf.placeholder(tf.float32, shape=(1, 256 * cell_ratio)) for _ in range(2)])
        self.motionPrevLstmState      = tuple([tf.placeholder(tf.float32, shape=(1, 4 * memory_size * 2)) for _ in range(2)])
        self.predAppearance, self.predMotion, self.appearanceState, self.motionState = \
            network(self.appearanceMemPlaceholder, self.motionMemPlaceholder, 1, 1, memory_size, self.appearancePrevLstmState, self.motionPrevLstmState)

        # build network for feature extraction
        self.imagePlaceholder = tf.placeholder(tf.float32, shape=(None, img_size, img_size, 3))
        self.appearanceFeature = cnnNetwork(inputs=self.imagePlaceholder, feature_size=feature_size)

        # trackers
        self.trackers = []

        # tracking parameters
        self.memory_size   = memory_size
        self.feature_size  = feature_size
        self.image_size    = img_size
        self.sess          = sess
        self.iou_threshold = iou_threshold
        self.kl_threshold  = kl_threshold
        self.ori_height    = ori_height
        self.ori_width     = ori_width
        self.max_age       = 300
        self.frame_count   = 0

    def KL(self, x, y):
        epsilon = 0.000001

        X = x + epsilon
        Y = y + epsilon

        divergence = np.sum(X * np.log(X / Y))

        return divergence

    def get_iou(self, a, b, epsilon=1e-5):
        # [x1, y1, x2, y2]

        # COORDINATES OF THE INTERSECTION BOX
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        # AREA OF OVERLAP - Area where the boxes intersect
        width = (x2 - x1)
        height = (y2 - y1)
        # handle case where there is NO overlap
        if (width < 0) or (height < 0):
            return 0.0
        area_overlap = width * height

        # COMBINED AREA
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        area_combined = area_a + area_b - area_overlap

        # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
        iou = area_overlap / (area_combined + epsilon)

        return iou

    def data_association(self, appearanceDetections, motionDetections, label):

        if (len(self.trackers) == 0):
            return np.empty((0, 2), dtype=int), np.arange(len(appearanceDetections)), np.empty((0, 5), dtype=int)
        """
        # calculate KL-Divergence between tracklets and detections
        #print('kl score')
        kl_matrix = np.zeros((len(appearanceDetections), len(self.trackers)), dtype=np.float32)
        for dIdx, det in enumerate(appearanceDetections):
            for tIdx, trk in enumerate(self.trackers):
                kl_score = self.KL(self.trackers[tIdx].predictAppearance, appearanceDetections[dIdx])
                #print('[d:%d][t:%d]:%.4f'%(dIdx, trk.id, kl_score))
                if kl_score <= self.kl_threshold:
                    kl_score = np.maximum(1.0 - kl_score, 0.0)
                else:
                    kl_score = 0.0
                kl_matrix[dIdx, tIdx] = kl_score
        """

        # calculate IoU score between tracklets and detections
        #print('iou')
        iou_matrix = np.zeros((len(motionDetections), len(self.trackers)), dtype=np.float32)
        for dIdx, det in enumerate(appearanceDetections):
            for tIdx, trk in enumerate(self.trackers):
                iou_score = self.get_iou(self.trackers[tIdx].predictMotion, motionDetections[dIdx])
                #print('[d:%d][t:%d]:%.4f' % (dIdx, trk.id, iou_score))
                if iou_score < self.iou_threshold:
                    iou_score = 0.0
                iou_matrix[dIdx, tIdx] = iou_score


        # calculate final score
        score_matrix = np.zeros((len(appearanceDetections), len(self.trackers)), dtype=np.float32)
        #print('score')
        for dIdx, det in enumerate(appearanceDetections):
            for tIdx, trk in enumerate(self.trackers):
                score_matrix[dIdx, tIdx] = iou_matrix[dIdx, tIdx]
                #print(trk.label, label[dIdx])
                if trk.label != label[dIdx]:
                    score_matrix[dIdx, tIdx] = 0.0
                #print('[d:%d][t:%d]:%.4f' % (dIdx, trk.id, score_matrix[dIdx, tIdx]))

        # start data association
        matched_matrix = linear_assignment(-score_matrix)
        """
        print("distance matrix")
        for d in range(len(detections)):
            for t in range(len(trackers)):
                print('[d:%d][t:%d]:%f'%(d,t,score_matrix[d,t]))
        print('[matched matrix]')
        print(matched_matrix)
        """

        unmatched_detections = []
        for d, det in enumerate(appearanceDetections):
            if (d not in matched_matrix[:, 0]):
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t, trk in enumerate(self.trackers):
            if (t not in matched_matrix[:, 1]):
                unmatched_trackers.append(t)

        matched = []
        for m in matched_matrix:
            if(score_matrix[m[0], m[1]] < 0.3):
                unmatched_trackers.append(m[1])
                unmatched_detections.append(m[0])
            else:
                matched.append(m.reshape(1, 2))

        if (len(matched) == 0):
            matched = np.empty((0, 2), dtype=int)
        else:
            matched = np.concatenate(matched, axis=0)

        #print('matched',matched)
        #print('unmatched_detections',unmatched_detections)
        #print('unmatched_trackers',unmatched_trackers)

        return matched, np.array(unmatched_detections), np.array(unmatched_trackers)

    def xyxy_to_xywh(self, bbox):
        x1, y1, x2, y2 = bbox

        x1 = x1 * self.ori_width
        x2 = x2 * self.ori_width
        y1 = y1 * self.ori_height
        y2 = y2 * self.ori_height

        cx = int((x1 + x2)/2)
        cy = int((y1 + y2)/2)
        w  = int(x2 - x1)
        h  = int(y2 - y1)

        cx = float(cx / self.ori_width)
        cy = float(cy / self.ori_height)
        w  = float(w  / self.ori_width)
        h  = float(h  / self.ori_height)

        return [cx, cy, w, h]

    def xywh_to_xyxy(self, bbox):
        cx, cy, w, h = bbox

        cx = int(cx * self.ori_width)
        cy = int(cy * self.ori_height)
        w  = int(w * self.ori_width)
        h  = int(h * self.ori_height)

        x1 = cx - (w / 2)
        x2 = cx + (w / 2)
        y1 = cy - (h / 2)
        y2 = cy + (h / 2)

        x1 = float(x1 / self.ori_width)
        x2 = float(x2 / self.ori_width)
        y1 = float(y1 / self.ori_height)
        y2 = float(y2 / self.ori_height)

        return [x1, x2, y1, y2]

    def update(self, tracker, feature, bbox, label):

        # update tracker parameters
        tracker.time_since_update = 0
        tracker.hit_streak += 1
        tracker.hits += 1
        tracker.age += 1

        # update memory
        for idx in range(self.memory_size - 2, -1, -1):
            tracker.appearanceMem[idx + 1] = tracker.appearanceMem[idx]
            tracker.motionMem[idx + 1]     = tracker.motionMem[idx]

        tracker.appearanceMem[0] = feature
        tracker.motionMem[0]     = bbox

        # update label
        tracker.label    = label
        tracker.cur_bbox = bbox

        # mem resize
        motionMem = np.expand_dims(np.expand_dims(tracker.motionMem, 0), 0)
        appearanceMem = np.expand_dims(np.expand_dims(tracker.appearanceMem, 0), 0)

        # get predictions
        predictedAppearance, predictedMotion, appearanceLstmState, motionLstmState = \
            self.sess.run([self.predAppearance, self.predMotion, self.appearanceState, self.motionState],
                          feed_dict={self.appearanceMemPlaceholder: appearanceMem, self.motionMemPlaceholder: motionMem, self.appearancePrevLstmState:tracker.appearancePrevLstmState, self.motionPrevLstmState:tracker.motionPrevLstmState})

        # update prev lstm
        tracker.appearancePrevLstmState = appearanceLstmState
        tracker.motionPrevLstmState     = motionLstmState

        # update prediction
        predictedAppearance = np.resize(predictedAppearance, (self.feature_size))
        predictedMotion = np.resize(predictedMotion, (4))

        tracker.predictAppearance = predictedAppearance
        #tracker.predictMotion     = predictedMotion
        tracker.predictMotion = bbox

    def init_update(self, tracker):
        # mem resize
        motionMem = np.expand_dims(np.expand_dims(tracker.motionMem, 0), 0)
        appearanceMem = np.expand_dims(np.expand_dims(tracker.appearanceMem, 0), 0)

        # get predictions
        predictedAppearance, predictedMotion, appearanceLstmState, motionLstmState = \
            self.sess.run([self.predAppearance, self.predMotion, self.appearanceState, self.motionState],
                          feed_dict={self.appearanceMemPlaceholder: appearanceMem,
                                     self.motionMemPlaceholder: motionMem,
                                     self.appearancePrevLstmState: tracker.appearancePrevLstmState,
                                     self.motionPrevLstmState: tracker.motionPrevLstmState})

        # update prev lstm
        tracker.appearancePrevLstmState = appearanceLstmState
        tracker.motionPrevLstmState = motionLstmState

        # update prediction
        predictedAppearance = np.resize(predictedAppearance, (self.feature_size))
        predictedMotion = np.resize(predictedMotion, (4))

        tracker.predictAppearance = predictedAppearance
        #tracker.predictMotion = predictedMotion
        tracker.predictMotion = tracker.cur_bbox

    # bboxes = [[object image, bbox, class], [object image, bbox, class], ...]
    # bbox = [x1, y1, x2, y2]
    # object image should be resized already
    def track(self, bboxes):

        self.frame_count += 1

        # extract appearance features
        appearanceDetections = np.zeros((len(bboxes), self.feature_size), dtype=np.float32)
        for t, det in enumerate(bboxes):
            feature = self.sess.run([self.appearanceFeature], feed_dict={self.imagePlaceholder: np.expand_dims(bboxes[t][0], 0)})
            feature = np.resize(feature, (self.feature_size))
            appearanceDetections[t] = feature

        # extract motion
        motionDetections = np.zeros((len(bboxes), 4), dtype=np.float32)
        for t, det in enumerate(bboxes):
            motionDetections[t] = bboxes[t][1]

        # extract label
        detectLabel = []
        for t, det in enumerate(bboxes):
            detectLabel.append(bboxes[t][2])

        # data association based on algorithm
        matched, unmatched_dets, unmatched_trks = self.data_association(appearanceDetections, motionDetections, detectLabel)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if (t not in unmatched_trks):
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                index = d[0]
                feature = appearanceDetections[index]
                bbox    = motionDetections[index]
                label   = bboxes[index][2]
                self.update(trk, feature, bbox, label)

        # update unmatched trackers
        for t, trk in enumerate(self.trackers):
            if (t in unmatched_trks):
                tracklet = self.trackers[t]
                tracklet.age += 1
                if (tracklet.time_since_update > 0):
                    self.hit_streak = 0
                tracklet.time_since_update += 1

        # create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            tracker = Tracklet(memory_size=self.memory_size, feature_size=self.feature_size, appearance_feature=appearanceDetections[i], motion=motionDetections[i], label=bboxes[i][2], bbox=motionDetections[i], height=self.ori_height, width=self.ori_width)
            self.init_update(tracker)
            self.trackers.append(tracker)
            # print('tracker id: ',tracker.id,' generated newly')

        # return tracking result
        tracklets = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            tracklet = trk.get_info()
            if ((trk.time_since_update < 1)):
                tracklets.append(tracklet)
            i -= 1

            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
                # print('tracker id: ', self.trackers[i].id, ' removed')

        return tracklets
