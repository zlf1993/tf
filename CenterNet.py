from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys
import os
from mobilenet_v3_block import BottleNeck, h_swish
from yolov3_layer_utils import upsample_layer, yolo_conv2d, yolo_block
from utils.utils import resize_image

sys.path.append("../")

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(),array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("", ""))
        text += "  {}".format(array.dtype)
    print(text)


class EpochRecord(tf.keras.callbacks.Callback):
    def __init__(self, name):
        super(EpochRecord, self).__init__()
        self.name = name

    def on_epoch_end(self, epoch, logs={}):
        if not os.path.exists(self.name+"/epoch.txt"):
            file = open(self.name+"/epoch.txt", 'w')
            file.write("0")
            file.close()
        file = open(self.name+"/epoch.txt", 'r')
        epoch = int(str(file.readline()))
        file.close()
        epoch += 1
        epoch = str(epoch)
        file = open(self.name + "/epoch.txt", 'w')
        file.write(epoch)
        file.close()


class CentLoss(tf.keras.layers.Layer):
    def __init__(self, batch_size, num_class, decay, stride, **kwargs):
        super(CentLoss, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.num_class = num_class
        self.decay = decay
        self.stride = stride


    def call(self, inputs, **kwargs):
        keypoints, preg, fpn, ground_truth, masks, scalar_y, scalar_x = inputs
        losses = self._centernet_loss(ground_truth, masks, scalar_y, scalar_x, keypoints, preg, fpn)

        self.add_loss(losses[3], inputs=True)
        self.add_metric(losses[0], aggregation="mean", name="iou")
        self.add_metric(losses[1], aggregation="mean", name="heatmap")
        self.add_metric(losses[2], aggregation="mean", name="gravity")
        return losses[3]

    def _centernet_loss(self, ground_truth, masks, scalar_y, scalar_x, keypoints, preg, fpn):
        pshape = [tf.shape(keypoints)[1], tf.shape(keypoints)[2]]
        h = tf.range(0., tf.cast(pshape[0], tf.float32), dtype=tf.float32)
        w = tf.range(0., tf.cast(pshape[1], tf.float32), dtype=tf.float32)

        # shape of coordinate equals [h_y_num, w_x_mun]
        [meshgrid_x, meshgrid_y] = tf.meshgrid(w, h)
        total_loss = []
        for i in range(self.batch_size):
            gt_i = ground_truth[i, ...]
            masks_i = masks[i, ...]
            scalar_y_i = scalar_y[i, ...]
            scalar_x_i = scalar_x[i, ...]
            slice_index = tf.argmin(gt_i, axis=0)[0]
            gt_i = tf.gather(gt_i, tf.range(0, slice_index, dtype=tf.int64))  # [max_num, 7]
            masks_i = masks_i[..., 0:slice_index]
            scalar_y_i = scalar_y_i[..., 0:slice_index]
            scalar_x_i = scalar_x_i[..., 0:slice_index]
            loss = self._compute_one_image_loss(keypoints[i, ...], preg[i, ...], fpn[i, ...], gt_i, masks_i, scalar_y_i,
                                                scalar_x_i, meshgrid_y, meshgrid_x, pshape)
            total_loss.append(loss)
        mean_loss = tf.reduce_mean(total_loss, axis=0)
        return mean_loss

    def _compute_one_image_loss(self, gravity_pred, dist_pred, heatmap_pred, ground_truth, masks, scalar_y, scalar_x,
                                grid_y, grid_x, pshape):
        gravity_y = (ground_truth[..., 0] + 0.5) / self.stride  # [gt_valid,1]
        gravity_x = (ground_truth[..., 1] + 0.5) / self.stride
        gbbox_y1 = (ground_truth[..., 2] + 0.5) / self.stride  # not a rect-box shape, is mask shape
        gbbox_x1 = (ground_truth[..., 3] + 0.5) / self.stride
        gbbox_y2 = (ground_truth[..., 4] + 0.5) / self.stride
        gbbox_x2 = (ground_truth[..., 5] + 0.5) / self.stride
        class_id = tf.cast(ground_truth[..., 6], dtype=tf.int32)  # not a rect-box shape, is mask shape

        # for gravity center
        gravi_yx = ground_truth[..., 0:2] / self.stride
        gravi_yx_round = tf.floor(gravi_yx)
        gravi_yx_round_int = tf.cast(gravi_yx_round, tf.int64)

        gravity_y = tf.reshape(gravity_y, [1, 1, -1])
        gravity_x = tf.reshape(gravity_x, [1, 1, -1])

        # for mask part
        gravity_y_tile = tf.reshape(gravity_y, [1, 1, -1])
        gravity_x_tile = tf.reshape(gravity_x, [1, 1, -1])

        gbbox_y1 = tf.reshape(gbbox_y1, [1, 1, -1])
        gbbox_x1 = tf.reshape(gbbox_x1, [1, 1, -1])
        gbbox_y2 = tf.reshape(gbbox_y2, [1, 1, -1])
        gbbox_x2 = tf.reshape(gbbox_x2, [1, 1, -1])
        num_g = tf.shape(gbbox_y1)[-1]

        grid_y = tf.expand_dims(grid_y, -1) + 0.5
        grid_x = tf.expand_dims(grid_x, -1) + 0.5
        grid_y = tf.tile(grid_y, [1, 1, num_g])  # (y, x, num_g)
        grid_x = tf.tile(grid_x, [1, 1, num_g])
        dist_l = grid_x - gbbox_x1  # (y, x, num_g)
        dist_r = gbbox_x2 - grid_x
        dist_t = grid_y - gbbox_y1
        dist_b = gbbox_y2 - grid_y
        grid_y_mask = tf.cast(dist_t > 0., tf.float32) * tf.cast(dist_b > 0., tf.float32)
        grid_x_mask = tf.cast(dist_l > 0., tf.float32) * tf.cast(dist_r > 0., tf.float32)

        heatmask = grid_y_mask * grid_x_mask * tf.cast(masks, tf.float32)  # not a rect-box shape, is mask shape (y, x, num_g)
        dist_l *= heatmask  # not a rect-box shape, is mask shape shape (y, x, num_g)
        dist_r *= heatmask
        dist_t *= heatmask
        dist_b *= heatmask

        loc = tf.reduce_max(heatmask, axis=-1)  # (y, x) objects mask
        dist_area = (dist_l + dist_r) * (dist_t + dist_b)  # not a rect-box shape, is mask shape shape (y, x, num_g)
        dist_area_ = dist_area + (1. - heatmask) * 1e8
        dist_area_min = tf.reduce_min(dist_area_, axis=-1, keepdims=True)  # small things on the top, background is 1e8
        # not overlap things mask (y, x, num_g)
        dist_mask = tf.cast(tf.equal(dist_area, dist_area_min), tf.float32) * tf.expand_dims(loc, axis=-1)

        gravity_y_tile *= dist_mask  # not a rect-box shape, gravity center mask (y, x, num_g)
        gravity_x_tile *= dist_mask
        gbbox_y1 *= dist_mask  # (y, x, num_g)
        gbbox_x1 *= dist_mask
        gbbox_y2 *= dist_mask
        gbbox_x2 *= dist_mask

        dist_l *= dist_mask  # valid dist l, r, t, b
        dist_r *= dist_mask
        dist_t *= dist_mask
        dist_b *= dist_mask
        dist_l = tf.reduce_max(dist_l, axis=-1)  # not overlap 1 (y, x)
        dist_r = tf.reduce_max(dist_r, axis=-1)
        dist_t = tf.reduce_max(dist_t, axis=-1)
        dist_b = tf.reduce_max(dist_b, axis=-1)
        dist_pred_l = dist_pred[..., 0]  # (y, x)
        dist_pred_r = dist_pred[..., 1]
        dist_pred_t = dist_pred[..., 2]
        dist_pred_b = dist_pred[..., 3]

        inter_width = tf.minimum(dist_l, dist_pred_l) + tf.minimum(dist_r, dist_pred_r)
        inter_height = tf.minimum(dist_t, dist_pred_t) + tf.minimum(dist_b, dist_pred_b)
        inter_area = inter_width * inter_height
        union_area = (dist_l + dist_r) * (dist_t + dist_b) + (dist_pred_l + dist_pred_r) * (
                dist_pred_t + dist_pred_b) - inter_area
        iou = inter_area / (union_area + 1e-12)

        # for normal distribution
        reduction = tf.exp(-(((grid_y - 0.5 - gravity_y//1)/tf.math.sqrt(scalar_y)) ** 2 +
                             ((grid_x - 0.5 - gravity_x//1)/tf.math.sqrt(scalar_x)) ** 2) / (2 * 1 ** 2))
        iou_reduction = tf.reduce_max(reduction, axis=-1)  # [y, x, num_g] --> [y, x]
        iou_loss = tf.reduce_sum(-tf.math.log(iou + 1e-12) * loc * (iou_reduction*4 + 1.0))
        # TODO: try to use circle gaussion distribution
        # sigma = self._gaussian_radius(gbbox_y2 - gbbox_y1, gbbox_x2 - gbbox_x1, 0.7)
        # sigma = tf.reshape(sigma, [1, 1, -1])
        # reduction = tf.exp(-((grid_y - 0.5 - gravity_y // 1) ** 2 +
        #                      (grid_x - 0.5 - gravity_x // 1) ** 2) / (2 * sigma ** 2))

        zero_like = tf.expand_dims(tf.zeros(pshape, dtype=tf.float32), axis=-1)
        gt_keypoints = []
        heatmap_gt = []
        reduction_gt = []
        for i in range(self.num_class):
            # [num_g, 1]
            exist_i = tf.equal(class_id - 1, i)  # pass BG CLASS_ID: 0
            # [num_g_of_i, y, x]
            reduce_i = tf.boolean_mask(reduction, exist_i, axis=2)
            # [y, x, 1] heat_map for class i , if null class i, product zero_like_map
            reduce_i = tf.cond(
                tf.equal(tf.shape(reduce_i)[-1], 0),
                lambda: zero_like,
                lambda: tf.expand_dims(tf.reduce_max(reduce_i, axis=2), axis=-1)
            )
            reduction_gt.append(reduce_i)

            heatmask_i = tf.boolean_mask(dist_mask, exist_i, axis=2)
            heatmap_i = tf.cond(
                tf.equal(tf.shape(heatmask_i)[-1], 0),
                lambda: zero_like,
                lambda: tf.reduce_max(heatmask_i, axis=2, keepdims=True)
            )
            heatmap_gt.append(heatmap_i)

            # according to  class_i index extract gbbox_yx, [num_g_i , 2]
            gbbox_yx_i = tf.boolean_mask(gravi_yx_round_int, exist_i)
            # [y, x, 1]
            gt_keypoints_i = tf.cond(
                tf.equal(tf.shape(gbbox_yx_i)[0], 0),
                lambda: zero_like,
                lambda: tf.expand_dims(tf.sparse.to_dense(
                    tf.sparse.SparseTensor(gbbox_yx_i, tf.ones_like(gbbox_yx_i[..., 0], tf.float32),
                                           dense_shape=pshape), validate_indices=False), axis=-1)
            )
            gt_keypoints.append(gt_keypoints_i)
        reduction_gt = tf.concat(reduction_gt, axis=-1)
        heatmap_gt = tf.concat(heatmap_gt, axis=-1)
        gt_keypoints = tf.concat(gt_keypoints, axis=-1)

        keypoints_pos_loss = - tf.pow(1. - tf.sigmoid(gravity_pred), 2.) * \
            tf.math.log_sigmoid(gravity_pred) * gt_keypoints

        keypoints_neg_loss = -tf.pow(1. - reduction_gt, 4) * tf.pow(tf.sigmoid(gravity_pred), 2.) * \
            (-gravity_pred + tf.math.log_sigmoid(gravity_pred)) * (1. - gt_keypoints)

        gravity_loss = tf.reduce_sum(keypoints_pos_loss) + tf.reduce_sum(keypoints_neg_loss)

        heatmap_pos_loss = - 10 * tf.pow(1. - tf.sigmoid(heatmap_pred), 2.) * \
            tf.math.log_sigmoid(heatmap_pred) * heatmap_gt

        heatmap_neg_loss = - 10 * tf.pow(tf.sigmoid(heatmap_pred), 2.) * \
            (-heatmap_pred + tf.math.log_sigmoid(heatmap_pred)) * (1. - heatmap_gt)

        heatmap_loss = tf.reduce_sum(heatmap_neg_loss) / (tf.cast((pshape[0] * pshape[1] * self.num_class), tf.float32) - tf.reduce_sum(heatmap_gt))
        if tf.reduce_sum(heatmap_gt) != 0:
            heatmap_loss += tf.reduce_sum(heatmap_pos_loss) / tf.reduce_sum(heatmap_gt)
            iou_loss = iou_loss / tf.reduce_sum(loc)
            gravity_loss = gravity_loss / tf.cast(num_g, tf.float32)
            total_loss = iou_loss + gravity_loss + heatmap_loss
            return iou_loss, heatmap_loss, gravity_loss, total_loss
        else:
            return .0, heatmap_loss, .0, heatmap_loss

    def _gaussian_radius(self, height, width, min_overlap=0.7):
        a1 = 1.
        b1 = (height + width)
        c1 = width * height * (1. - min_overlap) / (1. + min_overlap)
        sq1 = tf.sqrt(b1 ** 2. - 4. * a1 * c1)
        r1 = (b1 + sq1) / 2.
        a2 = 4.
        b2 = 2. * (height + width)
        c2 = (1. - min_overlap) * width * height
        sq2 = tf.sqrt(b2 ** 2. - 4. * a2 * c2)
        r2 = (b2 + sq2) / 2.
        a3 = 4. * min_overlap
        b3 = -2. * min_overlap * (height + width)
        c3 = (min_overlap - 1.) * width * height
        sq3 = tf.sqrt(b3 ** 2. - 4. * a3 * c3)
        r3 = (b3 + sq3) / 2.

        return tf.reduce_min([r1, r2, r3])


class CenterNet:
    def __init__(self, config, name):
        self.name = name
        self.config = config
        assert config.MODEL in ['train', 'infer']
        self.mode = config.MODEL
        self.data_shape = config.IMAGE_SHAPE
        self.data_format = config.DATA_FORMAT
        self.image_size = config.IMAGE_MAX_DIM
        self.stride = config.STRIDE
        self.num_classes = config.NUM_CLASSES
        self.loss_decay = config.LOSS_DECAY
        self.l2_decay = config.L2_DECAY
        self.batch_size = config.BATCH_SIZE if config.MODEL == 'train' else 1
        self.max_gt_instances = config.MAX_GT_INSTANCES
        self.gt_channel = config.GT_CHANNEL
        self.seg_threshold = config.SEG_THRESHOLD
        #
        self.top_k_results_output = config.DETECTION_MAX_INSTANCES
        self.nms_threshold = config.DETECTION_NMS_THRESHOLD
        self.train_bn = config.TRAIN_BN
        self.box_threshold = config.BOX_THRESHOLD
        #
        self.score_threshold = config.SCORE_THRESHOLD
        self.is_training = True if config.MODEL == 'train' else False

        if not os.path.exists(name):
            os.mkdir(name)
        self.checkpoint_path = name

        if not os.path.exists(name + "/log"):
            os.mkdir(name + "/log")
        self.log_dir = name + "/log"

        if not os.path.exists(name+"/epoch.txt"):
            file = open(name+"/epoch.txt", 'w')
            file.write("0")
            file.close()

        file = open(name + "/epoch.txt", 'r')
        self.pro_epoch = int(str(file.readline()))
        file.close()
        self._define_inputs()
        self._build_backbone()
        self._build_graph()
        if self.pro_epoch != 0:
            self.load_weight(self.pro_epoch)

    def _define_inputs(self):
        # model inputs: [images, ground_truth, mask_ground_truth]
        shape = self.data_shape
        self.images = tf.keras.Input(shape=shape, dtype=tf.float32)

        if self.mode == 'train':
            gt_shape = [self.max_gt_instances, self.gt_channel]
            self.ground_truth = tf.keras.Input(shape=gt_shape, dtype=tf.float32)
            mask_shape = [self.image_size/int(self.stride), self.image_size/int(self.stride), self.max_gt_instances]
            self.mask_ground_truth = tf.keras.Input(shape=mask_shape, dtype=tf.float32)
            self.scalar_y = tf.keras.Input(shape=mask_shape, dtype=tf.float32)
            self.scalar_x = tf.keras.Input(shape=mask_shape, dtype=tf.float32)

    def _build_backbone(self):
        self.conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=2, padding="same")  # 160
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.bneck1_1 = BottleNeck(in_size=16, exp_size=16, out_size=16, s=1, is_se_existing=False, NL="RE", k=3)  # 112
        self.bneck1_2 = BottleNeck(in_size=16, exp_size=64, out_size=24, s=2, is_se_existing=False, NL="RE", k=3)  # 80

        self.bneck2_1 = BottleNeck(in_size=24, exp_size=72, out_size=24, s=1, is_se_existing=False, NL="RE", k=3)  # 80 s_4
        self.bneck2_2 = BottleNeck(in_size=24, exp_size=72, out_size=40, s=2, is_se_existing=True, NL="RE", k=5)  # 40

        self.bneck3_1 = BottleNeck(in_size=40, exp_size=120, out_size=40, s=1, is_se_existing=True, NL="RE", k=5)  # 28
        self.bneck3_2 = BottleNeck(in_size=40, exp_size=120, out_size=40, s=1, is_se_existing=True, NL="RE", k=5)  # 28 s_8
        self.bneck3_3 = BottleNeck(in_size=40, exp_size=240, out_size=80, s=2, is_se_existing=False, NL="HS", k=3)  # 20

        self.bneck4_1 = BottleNeck(in_size=80, exp_size=200, out_size=80, s=1, is_se_existing=False, NL="HS", k=3)  # 14
        self.bneck4_2 = BottleNeck(in_size=80, exp_size=184, out_size=80, s=1, is_se_existing=False, NL="HS", k=3)  # 14
        self.bneck4_3 = BottleNeck(in_size=80, exp_size=184, out_size=80, s=1, is_se_existing=False, NL="HS", k=3)  # 14
        self.bneck4_4 = BottleNeck(in_size=80, exp_size=480, out_size=112, s=1, is_se_existing=True, NL="HS", k=3)  # 14
        self.bneck4_5 = BottleNeck(in_size=112, exp_size=672, out_size=112, s=1, is_se_existing=True, NL="HS", k=3)  # 14 s_16
        self.bneck4_6 = BottleNeck(in_size=112, exp_size=672, out_size=160, s=2, is_se_existing=True, NL="HS", k=5)  # 10

        self.bneck5_1 = BottleNeck(in_size=160, exp_size=960, out_size=160, s=1, is_se_existing=True, NL="HS", k=5)  # 10
        self.bneck5_2 = BottleNeck(in_size=160, exp_size=960, out_size=160, s=1, is_se_existing=True, NL="HS", k=5)  # 10 s_32

    def _fusion_feature(self):
        inter1 = yolo_conv2d(self.s_32, 512, 3, 1)  # /32   14
        inter1 = yolo_conv2d(inter1, 512, 3, 1)  # /32
        inter1 = self._dconv_bn_activation(inter1, 256, 4, 2)  # /16    28
        concat1 = tf.concat([inter1, self.s_16], axis=3)  # 256+112=368  /16
        inter2 = yolo_conv2d(concat1, 256, 3, 1)  # /16
        inter2 = yolo_conv2d(inter2, 256, 3, 1)  # /16
        inter2 = self._dconv_bn_activation(inter2, 256, 4, 2)  # /8     56
        concat2 = tf.concat([inter2, self.s_8], axis=3)  # 256+40=296  /8
        inter3 = yolo_conv2d(concat2, 256, 3, 1)  # /8
        inter3 = yolo_conv2d(inter3, 256, 3, 1)  # /8

        inter4 = yolo_conv2d(self.s_4, 24, 3, 2)  # /4  56
        concat3 = tf.concat([inter3, inter4], axis=3)  # 256+24=280  /4
        inter5 = yolo_conv2d(concat3, 256, 3, 1)  # /8
        feature = yolo_conv2d(inter5, 256, 3, 1)  # /4

        return feature

    def _detect_head(self, bottom):
        conv1 = self._conv_bn_activation(bottom, 256, 3, 1)
        conv1 = self._conv_bn_activation(conv1, 256, 3, 1)
        conv1 = self._conv_bn_activation(conv1, 256, 3, 1)
        conv1 = self._conv_bn_activation(conv1, 256, 3, 1)
        conv1 = self._conv_bn_activation(conv1, 256, 3, 1)
        keypoints = self._conv_activation(conv1, self.num_classes, 3, 1, activation=None)

        conv2 = self._conv_bn_activation(bottom, 256, 3, 1)
        conv2 = self._conv_bn_activation(conv2, 256, 3, 1)
        conv2 = self._conv_bn_activation(conv2, 256, 3, 1)
        conv2 = self._conv_bn_activation(conv2, 256, 3, 1)
        conv2 = self._conv_bn_activation(conv2, 256, 3, 1)
        preg = self._conv_activation(conv2, 4, 3, 1, activation=tf.exp)

        conv3 = self._conv_bn_activation(bottom, 256, 3, 1)
        conv3 = self._conv_bn_activation(conv3, 256, 3, 1)
        conv3 = self._conv_bn_activation(conv3, 256, 3, 1)
        conv3 = self._conv_bn_activation(conv3, 256, 3, 1)
        conv3 = self._conv_bn_activation(conv3, 256, 3, 1)
        fpn = self._conv_activation(conv3, self.num_classes, 3, 1, activation=None)

        return keypoints, preg, fpn

    def _build_graph(self):
        x = self.conv1(self.images)
        x = self.bn1(x)
        x = h_swish(x)  # 208...112
        x = self.bneck1_1(x)  # 112
        x = self.bneck1_2(x)  # 56
        self.s_4 = self.bneck2_1(x)  # 56
        x = self.bneck2_2(self.s_4)  # 28
        x = self.bneck3_1(x)  # 28
        self.s_8 = self.bneck3_2(x)  # 28
        x = self.bneck3_3(self.s_8)  # 14
        x = self.bneck4_1(x)  # 14
        x = self.bneck4_2(x)  # 14
        x = self.bneck4_3(x)  # 14
        x = self.bneck4_4(x)  # 14
        self.s_16 = self.bneck4_5(x)  # 14
        x = self.bneck4_6(self.s_16)  # 7
        x = self.bneck5_1(x)  # 7
        self.s_32 = self.bneck5_2(x)  # 7

        feature_map = self._fusion_feature()
        keypoints, preg, fpn = self._detect_head(feature_map)

        if self.mode == 'train':
            center_loss = CentLoss(self.batch_size, self.num_classes, self.loss_decay, self.stride)\
                ([keypoints, preg, fpn, self.ground_truth, self.mask_ground_truth, self.scalar_y, self.scalar_x])
            inputs = [self.images, self.ground_truth, self.mask_ground_truth, self.scalar_y, self.scalar_x]
            outputs = [keypoints, preg, fpn, center_loss]
        else:
            pshape = [self.image_size/self.stride, self.image_size/self.stride]
            h = tf.range(0., tf.cast(pshape[0], tf.float32), dtype=tf.float32)
            w = tf.range(0., tf.cast(pshape[1], tf.float32), dtype=tf.float32)
            # shape of coordinate equals [h_y_num, w_x_mun]
            [meshgrid_x, meshgrid_y] = tf.meshgrid(w, h)
            meshgrid_y = tf.expand_dims(meshgrid_y, axis=-1)  # [Y, X, -1]
            meshgrid_x = tf.expand_dims(meshgrid_x, axis=-1)
            # [y, x, 2]
            center = tf.concat([meshgrid_y, meshgrid_x], axis=-1)

            # [batch_size, y, x, class_num] activate feature maps
            # [y, x, class_num]
            keypoints = tf.sigmoid(keypoints)
            fpn = tf.sigmoid(fpn)
            preg = preg

            for i in range(self.batch_size):
                # # [1, y, x, class_num]
                pic_keypoints = tf.expand_dims(keypoints[i], axis=0)

                # [1, y, x, 4]
                pic_preg = tf.expand_dims(preg[i], axis=0)

                pic_seg = tf.expand_dims(fpn[i], axis=0)
                # # [y, x, 1]
                # TODO: tensorlite not support squeeze
                # category = tf.expand_dims(tf.squeeze(tf.argmax(pic_keypoints, axis=-1, output_type=tf.int32)), axis=-1)
                category = tf.expand_dims(tf.argmax(pic_keypoints, axis=-1, output_type=tf.int32)[0], axis=-1)

                # [y, x, 1 + 2(y, x) + 1(index_of_class)=4]
                meshgrid_xyz = tf.concat([tf.zeros_like(category), tf.cast(center, tf.int32), category], axis=-1)

                # [y, x, 1]
                pic_keypoints = tf.gather_nd(pic_keypoints, meshgrid_xyz)
                # TODO: no necessary to squeeze
                # pic_keypoints = tf.squeeze(pic_keypoints)
                # [1, y, x, 1(top_value)]
                pic_keypoints = tf.expand_dims(pic_keypoints, axis=0)
                pic_keypoints = tf.expand_dims(pic_keypoints, axis=-1)

                # 3*3 to be peak value
                keypoints_peak = self._max_pooling(pic_keypoints, 3, 1)
                # mask for each peak_point in each 3*3 area, [1, y, x, 1] (0,1)
                keypoints_mask = tf.cast(tf.equal(pic_keypoints, keypoints_peak), tf.float32)
                # [1, y, x, 1] (true, false)
                pic_keypoints = pic_keypoints * keypoints_mask
                # [y*x]
                scores = tf.reshape(pic_keypoints, [-1])
                # [y*x]
                class_id = tf.reshape(category, [-1])
                # [(y* x), 2]
                grid_yx = tf.reshape(center, [-1, 2])
                # [(y*x), 4]
                bbox_lrtb = tf.reshape(pic_preg, [-1, 4])

                # TODO: manually order and select
                # score_mask = scores > self.score_threshold
                # scores = tf.boolean_mask(scores, score_mask)
                # class_id = tf.boolean_mask(class_id, score_mask)
                # grid_yx = tf.boolean_mask(grid_yx, score_mask) + 0.5
                # bbox_lrtb = tf.boolean_mask(bbox_lrtb, score_mask)

                # TODO: ATTENTION, order are lrtb in prediction, but tlbr in ground_truth
                # [num, 4(y1, x1, y2, x2)]
                bbox = tf.concat([grid_yx - bbox_lrtb[..., -2::-2], grid_yx + bbox_lrtb[..., -1::-2]], axis=-1)

                select_indices = tf.image.non_max_suppression(bbox, scores, self.top_k_results_output,
                                                              self.nms_threshold, score_threshold=self.score_threshold)
                # [num_select, ?]
                select_scores = tf.gather(scores, select_indices)
                select_center = tf.gather(grid_yx, select_indices)
                select_class_id = tf.gather(class_id, select_indices)
                select_bbox = tf.gather(bbox, select_indices)
                select_lrtb = tf.gather(bbox_lrtb, select_indices)
                class_seg = tf.cast(pic_seg > self.seg_threshold, tf.float32)

                # TODO: Could be mute
                # final_masks = tf.zeros([pshape[0], pshape[1], tf.shape(select_indices)[0]], tf.float32)
                # for i in range(self.num_classes):
                #     exist_i = tf.equal(select_class_id, i)  # [0,1,...]
                #     exist_int = tf.cast(exist_i, tf.float32)
                #     index = tf.where(condition=exist_int>0)
                #     num_i = tf.reduce_sum(exist_int)
                #     masks = self.seg_instance(index, select_bbox, exist_i, class_seg[0, ..., i], num_i, pic_preg,
                #                       meshgrid_y, meshgrid_x, pshape, tf.shape(select_indices)[0])
                #     final_masks = final_masks + masks
                # end of tensor masks
                select_scores = tf.expand_dims(select_scores, axis=0)

                select_center = tf.expand_dims(select_center, axis=0)
                print("============", tf.shape(select_center))
                select_class_id = tf.expand_dims(select_class_id, axis=0)
                select_bbox = tf.expand_dims(select_bbox, axis=0)
                select_lrtb = tf.expand_dims(select_lrtb, axis=0)
                # select_masks = tf.expand_dims(final_masks, axis=0)
                # TODO: concatenate the batch
            # for post_processing outputs
            outputs = [select_center, select_scores, select_bbox, select_class_id, select_lrtb, class_seg, pic_preg]





            # # [y, x]
            # category = tf.argmax(keypoints[0], axis=2, output_type=tf.int32)
            # # [1, y, x, 1]
            # max_key = tf.reduce_max(keypoints, axis=2, keepdims=True)
            # # 3*3 to be peak value
            # peak_key = self._max_pooling(max_key, 3, 1)
            # # mask for each peak_point in each 3*3 area, [y, x] (0,1)
            # mask_key = tf.cast(tf.equal(max_key, peak_key), tf.float32)
            # # [1, y, x, 1]
            # mask_key = max_key * mask_key
            # # [y*x]
            # scores = tf.reshape(mask_key, [-1])
            # # [y*x]
            # class_id = tf.reshape(category, [-1])
            # # [(y* x), 2]
            # grid_yx = tf.reshape(center, [-1, 2])
            # # [(y*x), 4]
            # bbox_lrtb = tf.reshape(preg, [-1, 4])
            # # [y*x, 4(y1, x1, y2, x2)]
            # bbox = tf.concat([grid_yx - bbox_lrtb[..., -2::-2], grid_yx + bbox_lrtb[..., -1::-2]], axis=-1)
            # select_indices = tf.image.non_max_suppression(bbox, scores, self.top_k_results_output,
            #                                               self.nms_threshold, score_threshold=self.score_threshold)
            # # [num_select, ?]
            # select_scores = tf.gather(scores, select_indices)
            # select_center = tf.gather(grid_yx, select_indices)
            # select_class_id = tf.gather(class_id, select_indices)
            # select_bbox = tf.gather(bbox, select_indices)
            # # select_lrtb = tf.gather(bbox_lrtb, select_indices)
            # class_seg = tf.cast(tf.greater(fpn, self.seg_threshold), tf.float32)
            #
            # select_scores = tf.expand_dims(select_scores, 0)
            # select_bbox = tf.expand_dims(select_bbox, 0)
            # select_class_id = tf.expand_dims(select_class_id, 0)
            # select_center = tf.expand_dims(select_center, 0)
            # print(tf.shape(select_scores))
            # print(tf.shape(select_bbox))
            # print(tf.shape(select_class_id))
            # print(tf.shape(select_center))
            # print(tf.shape(class_seg))

            outputs = [select_center, select_scores, select_bbox, select_class_id, class_seg, preg]
            inputs = [self.images]
        self.CenterNetModel = tf.keras.Model(inputs=inputs, outputs=outputs)

    def compile(self):
        """Gets the model ready for training. Adds losses including regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Add L2 Regularization
        reg_losses = self.l2_decay * tf.add_n([tf.nn.l2_loss(var) for var in self.CenterNetModel.trainable_weights])
        self.CenterNetModel.add_loss(lambda: tf.reduce_sum(reg_losses))

        # Optimizer object
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        # optimizer = tf.keras.optimizers.SGD(learning_rate=0.00001)

        self.CenterNetModel.compile(optimizer=optimizer)

    def train_epochs(self, dataset, valset, config, epochs=50):
        self.compile()
        # iter_data = dataset.generator(config.BATCH_SIZE, config.STEPS_PER_EPOCH)
        # val_generator = valset.generator(config.BATCH_SIZE, config.VALIDATION_STEPS)

        epochRec = EpochRecord(self.name)
        callbacks = [
            epochRec,
            tf.keras.callbacks.ProgbarLogger(),
            # tf.keras.callbacks.ReduceLROnPlateau(moniter='val_loss', factor=0.1, patience=2, mode='min', min_lr=1e-7),
            tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=0, write_graph=True,
                                           write_images=False),
            tf.keras.callbacks.ModelCheckpoint(self.checkpoint_path + "/weights.{epoch:03d}-{loss:.2f}.hdf5", verbose=0, save_weights_only=True)
        ]
        step = int(config.PIC_NUM / config.BATCH_SIZE)
        print("=====ready for model.fit_generator======")
        self.CenterNetModel.fit_generator(
            dataset,
            initial_epoch=self.pro_epoch,
            epochs=epochs,
            max_queue_size=4,
            workers=1,
            steps_per_epoch=step,
            use_multiprocessing=False,
            # validation_data=val_generator,
            # validation_steps=self.config.VALIDATION_STEPS,
            # validation_freq=1,
            callbacks=callbacks
        )

    def test_one_image(self, images):
        self.is_training = False
        image, window, scale, padding, crop = resize_image(
            images,
            min_dim=self.image_size,
            min_scale=0,
            max_dim=self.image_size,
            mode="square")
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        mean = np.reshape(mean, [1, 1, 3])
        std = np.reshape(std, [1, 1, 3])
        image = (image / 255. - mean) / std
        image = tf.convert_to_tensor(np.expand_dims(image, axis=0))

        # converter = tf.lite.TFLiteConverter.from_keras_model(self.CenterNetModel)
        #
        # # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        # # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
        # #                                        tf.lite.OpsSet.SELECT_TF_OPS]
        #
        # # converter.representative_dataset = representative_dataset_gen
        # tf_lite_model = converter.convert()
        # open("converted_model.tflite", "wb").write(tf_lite_model)

        pred = self.CenterNetModel.predict(
            image,
            batch_size=1,
            verbose=0,
            steps=None,
            callbacks=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False
        )
        return pred

    def load_weight(self, epoch):
        # latest = tf.train.latest_checkpoint(self.checkpoint_path)
        epoch = str(epoch).zfill(3)
        latest = ""
        for filename in os.listdir(self.checkpoint_path):
            root, ext = os.path.splitext(filename)
            if root.startswith('weights.' + epoch) and ext == '.hdf5':
                latest = filename
                break
        self.CenterNetModel.load_weights("./" + self.checkpoint_path + "/" + latest, by_name=True)
        print('load weight', latest, 'successfully')

    def load_pretrained_weight(self, path):
        self.pretrained_saver.restore(self.sess, path)
        print('load pretrained weight', path, 'successfully')

    def _bn(self, bottom):
        bn = tf.keras.layers.BatchNormalization()(bottom)
        return bn

    def _conv_bn_activation(self, bottom, filters, kernel_size, strides, activation=h_swish):
        conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            data_format=self.data_format
        )(bottom)
        bn = self._bn(conv)
        if activation is not None:
            return activation(bn)
        else:
            return bn

    def _conv_activation(self, bottom, filters, kernel_size, strides, activation=h_swish):
        conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            data_format=self.data_format
        )(bottom)
        if activation is not None:
            return activation(conv)
        else:
            return conv

    def _dconv_bn_activation(self, bottom, filters, kernel_size, strides, activation=h_swish):
        conv = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            data_format=self.data_format
        )(bottom)
        bn = self._bn(conv)
        if activation is not None:
            bn = activation(bn)
        return bn

    def _max_pooling(self, bottom, pool_size, strides, name=None):
        a = tf.keras.layers.MaxPool2D(
            pool_size=pool_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
            name=name
        )(bottom)
        return a


def representative_dataset_gen(CocoDataset, Config):
    for i in range(50000):
        image = CocoDataset.load_image(i)
        image, window, scale, padding, crop = resize_image(
            image,
            min_dim=Config.IMAGE_MIN_DIM,
            min_scale=Config.IMAGE_MIN_SCALE,
            max_dim=Config.IMAGE_MAX_DIM,
            mode=Config.IMAGE_RESIZE_MODE)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        mean = np.reshape(mean, [1, 1, 3])
        std = np.reshape(std, [1, 1, 3])
        img = (image / 255. - mean) / std

        yield [img]