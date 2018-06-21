import itertools
import logging
import math
from collections import namedtuple

import cv2
import numpy as np
import tensorflow as tf
from scipy.ndimage import maximum_filter, gaussian_filter

from pose import common
from .common import CocoPairsNetwork, CocoPairs, CocoPart

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class Human:
    """
    body_parts: list of BodyPart
    """
    __slots__ = ('body_parts', 'pairs', 'uidx_list')

    def __init__(self, pairs):
        self.pairs = []
        self.uidx_list = set()
        self.body_parts = {}
        for pair in pairs:
            self.add_pair(pair)

    @staticmethod
    def _get_uidx(part_idx, idx):
        return '%d-%d' % (part_idx, idx)

    def add_pair(self, pair):
        self.pairs.append(pair)
        self.body_parts[pair.part_idx1] = BodyPart(Human._get_uidx(pair.part_idx1, pair.idx1),
                                                   pair.part_idx1,
                                                   pair.coord1[0], pair.coord1[1], pair.score)
        self.body_parts[pair.part_idx2] = BodyPart(Human._get_uidx(pair.part_idx2, pair.idx2),
                                                   pair.part_idx2,
                                                   pair.coord2[0], pair.coord2[1], pair.score)
        self.uidx_list.add(Human._get_uidx(pair.part_idx1, pair.idx1))
        self.uidx_list.add(Human._get_uidx(pair.part_idx2, pair.idx2))

    def is_connected(self, other):
        return len(self.uidx_list & other.uidx_list) > 0

    def merge(self, other):
        for pair in other.pairs:
            self.add_pair(pair)

    def part_count(self):
        return len(self.body_parts.keys())

    def get_max_score(self):
        return max([x.score for _, x in self.body_parts.items()])

    def __str__(self):
        return ' '.join([str(x) for x in self.body_parts.values()])


class BodyPart:
    """
    part_idx : part index(eg. 0 for nose)
    x, y: coordinate of body part
    score : confidence score
    """
    __slots__ = ('uidx', 'part_idx', 'x', 'y', 'score')

    def __init__(self, uidx, part_idx, x, y, score):
        self.uidx = uidx
        self.part_idx = part_idx
        self.x, self.y = x, y
        self.score = score

    def get_part_name(self):
        return CocoPart(self.part_idx)

    def __str__(self):
        return 'BodyPart:%d-(%.2f, %.2f) score=%.2f' % (self.part_idx, self.x, self.y, self.score)


class PoseEstimator:
    heatmap_supress = False
    heatmap_gaussian = False
    adaptive_threshold = False

    NMS_Threshold = 0.15
    Local_PAF_Threshold = 0.2
    PAF_Count_Threshold = 5
    Part_Count_Threshold = 4
    Part_Score_Threshold = 4.5

    PartPair = namedtuple('PartPair', [
        'score',
        'part_idx1', 'part_idx2',
        'idx1', 'idx2',
        'coord1', 'coord2',
        'score1', 'score2'
    ], verbose=False)

    def __init__(self):
        pass

    @staticmethod
    def non_max_suppression(plain, window_size=3, threshold=NMS_Threshold):
        under_threshold_indices = plain < threshold
        plain[under_threshold_indices] = 0
        return plain * (plain == maximum_filter(plain, footprint=np.ones((window_size, window_size))))

    @staticmethod
    def estimate(heat_mat, paf_mat):
        if heat_mat.shape[2] == 19:
            heat_mat = np.rollaxis(heat_mat, 2, 0)
        if paf_mat.shape[2] == 38:
            paf_mat = np.rollaxis(paf_mat, 2, 0)

        if PoseEstimator.heatmap_supress:
            heat_mat = heat_mat - heat_mat.min(axis=1).min(axis=1).reshape(19, 1, 1)
            heat_mat = heat_mat - heat_mat.min(axis=2).reshape(19, heat_mat.shape[1], 1)

        if PoseEstimator.heatmap_gaussian:
            heat_mat = gaussian_filter(heat_mat, sigma=0.5)

        if PoseEstimator.adaptive_threshold:
            _NMS_Threshold = max(np.average(heat_mat) * 4.0, PoseEstimator.NMS_Threshold)
            _NMS_Threshold = min(_NMS_Threshold, 0.3)
        else:
            _NMS_Threshold = PoseEstimator.NMS_Threshold

        # extract interesting coordinates using NMS.
        coords = []  # [[coords in plane1], [....], ...]
        for plain in heat_mat[:-1]:
            nms = PoseEstimator.non_max_suppression(plain, 5, _NMS_Threshold)
            coords.append(np.where(nms >= _NMS_Threshold))

        # score pairs
        pairs_by_conn = list()
        for (part_idx1, part_idx2), (paf_x_idx, paf_y_idx) in zip(CocoPairs, CocoPairsNetwork):
            pairs = PoseEstimator.score_pairs(
                part_idx1, part_idx2,
                coords[part_idx1], coords[part_idx2],
                paf_mat[paf_x_idx], paf_mat[paf_y_idx],
                heatmap=heat_mat,
                rescale=(1.0 / heat_mat.shape[2], 1.0 / heat_mat.shape[1])
            )

            pairs_by_conn.extend(pairs)

        # merge pairs to human
        # pairs_by_conn is sorted by CocoPairs(part importance) and Score between Parts.
        humans = [Human([pair]) for pair in pairs_by_conn]
        while True:
            merge_items = None
            for k1, k2 in itertools.combinations(humans, 2):
                if k1 == k2:
                    continue
                if k1.is_connected(k2):
                    merge_items = (k1, k2)
                    break

            if merge_items is not None:
                merge_items[0].merge(merge_items[1])
                humans.remove(merge_items[1])
            else:
                break

        # reject by subset count
        humans = [human for human in humans if human.part_count() >= PoseEstimator.PAF_Count_Threshold]

        # reject by subset max score
        humans = [human for human in humans if human.get_max_score() >= PoseEstimator.Part_Score_Threshold]

        return humans

    @staticmethod
    def score_pairs(part_idx1, part_idx2, coord_list1, coord_list2, paf_mat_x, paf_mat_y, heatmap, rescale=(1.0, 1.0)):
        connection_temp = []

        cnt = 0
        for idx1, (y1, x1) in enumerate(zip(coord_list1[0], coord_list1[1])):
            for idx2, (y2, x2) in enumerate(zip(coord_list2[0], coord_list2[1])):
                score, count = PoseEstimator.get_score(x1, y1, x2, y2, paf_mat_x, paf_mat_y)
                cnt += 1
                if count < PoseEstimator.PAF_Count_Threshold or score <= 0.0:
                    continue
                connection_temp.append(PoseEstimator.PartPair(
                    score=score,
                    part_idx1=part_idx1, part_idx2=part_idx2,
                    idx1=idx1, idx2=idx2,
                    coord1=(x1 * rescale[0], y1 * rescale[1]),
                    coord2=(x2 * rescale[0], y2 * rescale[1]),
                    score1=heatmap[part_idx1][y1][x1],
                    score2=heatmap[part_idx2][y2][x2],
                ))

        connection = []
        used_idx1, used_idx2 = set(), set()
        for candidate in sorted(connection_temp, key=lambda x: x.score, reverse=True):
            # check not connected
            if candidate.idx1 in used_idx1 or candidate.idx2 in used_idx2:
                continue
            connection.append(candidate)
            used_idx1.add(candidate.idx1)
            used_idx2.add(candidate.idx2)

        return connection

    @staticmethod
    def get_score(x1, y1, x2, y2, paf_mat_x, paf_mat_y):
        __num_inter = 10
        __num_inter_f = float(__num_inter)
        dx, dy = x2 - x1, y2 - y1
        normVec = math.sqrt(dx ** 2 + dy ** 2)

        if normVec < 1e-4:
            return 0.0, 0

        vx, vy = dx / normVec, dy / normVec

        xs = np.arange(x1, x2, dx / __num_inter_f) if x1 != x2 else np.full((__num_inter,), x1)
        ys = np.arange(y1, y2, dy / __num_inter_f) if y1 != y2 else np.full((__num_inter,), y1)
        xs = (xs + 0.5).astype(np.int8)
        ys = (ys + 0.5).astype(np.int8)

        # without vectorization
        pafXs = np.zeros(__num_inter)
        pafYs = np.zeros(__num_inter)
        for idx, (mx, my) in enumerate(zip(xs, ys)):
            pafXs[idx] = paf_mat_x[my][mx]
            pafYs[idx] = paf_mat_y[my][mx]

        # vectorization slow?
        # pafXs = pafMatX[ys, xs]
        # pafYs = pafMatY[ys, xs]

        local_scores = pafXs * vx + pafYs * vy
        thidxs = local_scores > PoseEstimator.Local_PAF_Threshold

        return sum(local_scores * thidxs), sum(thidxs)


class TfPoseEstimator:
    ENSEMBLE = 'addup'  # average, addup

    def __init__(self, graph_path, target_size=(320, 240)):
        self.target_size = target_size

        # load graph
        with tf.gfile.GFile(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        self.graph = tf.get_default_graph()
        tf.import_graph_def(graph_def, name='TfPoseEstimator')
        self.persistent_sess = tf.Session(graph=self.graph)

        # for op in self.graph.get_operations():
        #     print(op.name)

        self.tensor_image = self.graph.get_tensor_by_name('TfPoseEstimator/image:0')
        self.tensor_output = self.graph.get_tensor_by_name('TfPoseEstimator/Openpose/concat_stage7:0')

        self.heatMat = self.pafMat = None

        # warm-up
        self.persistent_sess.run(
            self.tensor_output,
            feed_dict={
                self.tensor_image: [np.ndarray(shape=(target_size[1], target_size[0], 3), dtype=np.float16)]
            }
        )

    def __del__(self):
        self.persistent_sess.close()

    @staticmethod
    def _quantize_img(npimg):
        npimg_q = npimg + 1.0
        npimg_q /= (2.0 / 2 ** 8)
        # npimg_q += 0.5
        npimg_q = npimg_q.astype(np.uint8)
        return npimg_q

    @staticmethod
    def draw_humans(npimg, humans, imgcopy=False):
        if imgcopy:
            npimg = np.copy(npimg)
        image_h, image_w = npimg.shape[:2]
        centers = {}
        for human in humans:
            # draw point
            for i in range(common.CocoPart.Background.value):
                if i not in human.body_parts.keys():
                    continue

                body_part = human.body_parts[i]
                center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
                centers[i] = center
                cv2.circle(npimg, center, 3, (255, 0, 0), thickness=3, lineType=8, shift=0)
            # draw line
            for pair_order, pair in enumerate(common.CocoPairsRender):
                if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                    continue

                cv2.line(npimg, centers[pair[0]], centers[pair[1]], (0, 255, 0), 2)
                # cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 2)

        return npimg

    @staticmethod
    def get_humans(npimg, humans, imgcopy=False):
        if imgcopy:
            npimg = np.copy(npimg)
        image_h, image_w = npimg.shape[:2]
        joints = []
        bboxes = []
        xcenter = []
        # 取出每个人的关节点
        for human in humans:
            xs = []
            ys = []
            centers = {}
            # 将所有关节点绘制到图像上
            for i in range(common.CocoPart.Background.value):
                if i not in human.body_parts.keys():
                    continue

                body_part = human.body_parts[i]
                center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))

                centers[i] = center
                xs.append(center[0])
                ys.append(center[1])
                # 绘制关节点
                cv2.circle(npimg, center, 3, common.CocoColors[i], thickness=3, lineType=8, shift=0)

            # 将属于同一人的关节点按照各个部位相连
            for pair_order, pair in enumerate(common.CocoPairsRender):
                if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                    continue
                cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)

            # 根据每个人的关节点信息生成ROI区域
            xmin = float(min(xs) / image_w)
            ymin = float(min(ys) / image_h)
            xmax = float(max(xs) / image_w)
            ymax = float(max(ys) / image_h)
            bboxes.append([xmin, ymin, xmax, ymax, 0.9999])
            joints.append(centers)
            if 1 in centers:
                xcenter.append(centers[1][0])

        return npimg, joints, bboxes, xcenter

    @staticmethod
    def get_skeleton(npimg, humans, imgcopy=False):
        if imgcopy:
            npimg = np.copy(npimg)
        sk = np.copy(npimg)
        image_h, image_w = npimg.shape[:2]
        joints = []
        bboxes = []
        xcenter = []
        for human in humans:
            xs = []
            ys = []
            centers = {}
            # draw point
            for i in range(common.CocoPart.Background.value):
                if i not in human.body_parts.keys():
                    continue

                body_part = human.body_parts[i]
                center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))

                centers[i] = center
                xs.append(center[0])
                ys.append(center[1])
                cv2.circle(sk, center, 3, (0, 0, 255), thickness=3, lineType=8, shift=0)

            # 将属于同一人的关节点按照各个部位相连
            for pair_order, pair in enumerate(common.CocoPairsRender):
                if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                    continue
                cv2.line(sk, centers[pair[0]], centers[pair[1]], (0, 255, 0), 2)
            xmin = float(min(xs) / image_w)
            ymin = float(min(ys) / image_h)
            xmax = float(max(xs) / image_w)
            ymax = float(max(ys) / image_h)
            bboxes.append([xmin, ymin, xmax, ymax, 0.9999])
            joints.append(centers)
            if 1 in centers:
                xcenter.append(centers[1][0])

        return npimg, joints, bboxes, xcenter, sk

    def _get_scaled_img(self, npimg, scale):
        get_base_scale = lambda s, w, h: max(self.target_size[0] / float(w), self.target_size[1] / float(h)) * s
        img_h, img_w = npimg.shape[:2]

        if scale is None:
            if npimg.shape[:2] != (self.target_size[1], self.target_size[0]):
                # resize
                npimg = cv2.resize(npimg, self.target_size)
            return [npimg], [(0.0, 0.0, 1.0, 1.0)]
        elif isinstance(scale, float):
            # scaling with center crop
            base_scale = get_base_scale(scale, img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale)
            ratio_x = (1. - self.target_size[0] / float(npimg.shape[1])) / 2.0
            ratio_y = (1. - self.target_size[1] / float(npimg.shape[0])) / 2.0
            roi = self._crop_roi(npimg, ratio_x, ratio_y)
            return [roi], [(ratio_x, ratio_y, 1. - ratio_x * 2, 1. - ratio_y * 2)]
        elif isinstance(scale, tuple) and len(scale) == 2:
            # scaling with sliding window : (scale, step)
            base_scale = get_base_scale(scale[0], img_w, img_h)
            base_scale_w = self.target_size[0] / (img_w * base_scale)
            base_scale_h = self.target_size[1] / (img_h * base_scale)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale)
            window_step = scale[1]
            rois = []
            infos = []
            for ratio_x, ratio_y in itertools.product(np.arange(0., 1.01 - base_scale_w, window_step),
                                                      np.arange(0., 1.01 - base_scale_h, window_step)):
                roi = self._crop_roi(npimg, ratio_x, ratio_y)
                rois.append(roi)
                infos.append((ratio_x, ratio_y, base_scale_w, base_scale_h))
            return rois, infos
        elif isinstance(scale, tuple) and len(scale) == 3:
            # scaling with ROI : (want_x, want_y, scale_ratio)
            base_scale = get_base_scale(scale[2], img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale)
            ratio_w = self.target_size[0] / float(npimg.shape[1])
            ratio_h = self.target_size[1] / float(npimg.shape[0])

            want_x, want_y = scale[:2]
            ratio_x = want_x - ratio_w / 2.
            ratio_y = want_y - ratio_h / 2.
            ratio_x = max(ratio_x, 0.0)
            ratio_y = max(ratio_y, 0.0)
            if ratio_x + ratio_w > 1.0:
                ratio_x = 1. - ratio_w
            if ratio_y + ratio_h > 1.0:
                ratio_y = 1. - ratio_h

            roi = self._crop_roi(npimg, ratio_x, ratio_y)
            return [roi], [(ratio_x, ratio_y, ratio_w, ratio_h)]

    def _crop_roi(self, npimg, ratio_x, ratio_y):
        target_w, target_h = self.target_size
        h, w = npimg.shape[:2]
        x = max(int(w * ratio_x - .5), 0)
        y = max(int(h * ratio_y - .5), 0)
        cropped = npimg[y:y + target_h, x:x + target_w]

        cropped_h, cropped_w = cropped.shape[:2]
        if cropped_w < target_w or cropped_h < target_h:
            npblank = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)

            copy_x, copy_y = (target_w - cropped_w) // 2, (target_h - cropped_h) // 2
            npblank[copy_y:copy_y + cropped_h, copy_x:copy_x + cropped_w] = cropped
        else:
            return cropped

    def inference(self, npimg, scales=None):
        if npimg is None:
            raise Exception('The image is not valid. Please check your image exists.')

        if not isinstance(scales, list):
            scales = [None]

        if self.tensor_image.dtype == tf.quint8:
            # quantize input image
            npimg = TfPoseEstimator._quantize_img(npimg)
            # pass

        rois = []
        infos = []
        for scale in scales:
            roi, info = self._get_scaled_img(npimg, scale)
            rois.extend(roi)
            infos.extend(info)

        logger.debug('inference+')
        output = self.persistent_sess.run(self.tensor_output, feed_dict={self.tensor_image: rois})

        heatMats = output[:, :, :, :19]
        pafMats = output[:, :, :, 19:]
        logger.debug('inference-')

        output_h, output_w = output.shape[1:3]
        max_ratio_w = max_ratio_h = 10000.0
        for info in infos:
            max_ratio_w = min(max_ratio_w, info[2])
            max_ratio_h = min(max_ratio_h, info[3])
        mat_w, mat_h = int(output_w / max_ratio_w), int(output_h / max_ratio_h)

        resized_heatMat = np.zeros((mat_h, mat_w, 19), dtype=np.float32)
        resized_pafMat = np.zeros((mat_h, mat_w, 38), dtype=np.float32)
        resized_cntMat = np.zeros((mat_h, mat_w, 1), dtype=np.float32)
        resized_cntMat += 1e-12

        for heatMat, pafMat, info in zip(heatMats, pafMats, infos):
            w, h = int(info[2] * mat_w), int(info[3] * mat_h)
            heatMat = cv2.resize(heatMat, (w, h))
            pafMat = cv2.resize(pafMat, (w, h))
            x, y = int(info[0] * mat_w), int(info[1] * mat_h)

            if TfPoseEstimator.ENSEMBLE == 'average':
                # average
                resized_heatMat[max(0, y):y + h, max(0, x):x + w, :] += heatMat[max(0, -y):, max(0, -x):, :]
                resized_pafMat[max(0, y):y + h, max(0, x):x + w, :] += pafMat[max(0, -y):, max(0, -x):, :]
                resized_cntMat[max(0, y):y + h, max(0, x):x + w, :] += 1
            else:
                # add up
                resized_heatMat[max(0, y):y + h, max(0, x):x + w, :] = np.maximum(
                    resized_heatMat[max(0, y):y + h, max(0, x):x + w, :], heatMat[max(0, -y):, max(0, -x):, :])
                resized_pafMat[max(0, y):y + h, max(0, x):x + w, :] += pafMat[max(0, -y):, max(0, -x):, :]
                resized_cntMat[max(0, y):y + h, max(0, x):x + w, :] += 1

        if TfPoseEstimator.ENSEMBLE == 'average':
            self.heatMat = resized_heatMat / resized_cntMat
            self.pafMat = resized_pafMat / resized_cntMat
        else:
            self.heatMat = resized_heatMat
            self.pafMat = resized_pafMat / (np.log(resized_cntMat) + 1)
        # self.heatMat = tf.cast(self.heatMat, tf.float16)
        # self.pafMat = tf.cast(self.pafMat, tf.float16)

        humans = PoseEstimator.estimate(self.heatMat, self.pafMat)
        return humans
