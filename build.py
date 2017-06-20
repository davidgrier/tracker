#!/usr/bin/env python

# pared down to just what's needed for Lorenz-Mie tracking

import tensorflow as tf
import tensorflow.contrib.slim as slim
import inception_v1 as inception
from rezoom import rezoom

def googlenet_model(x, H, reuse):
    with slim.arg_scope(inception.inception_v1_arg_scope()):
        _, T = inception.inception_v1(x, is_training=True,
                                      num_classes=1001,
                                      spatial_squeeze=False,
                                      reuse=reuse)
    coarse_feat = T[H['slim_top_lname']][:, :, :, :H['later_feat_channels']]
    assert coarse_feat.op.outputs[0].get_shape()[3] == H['later_feat_channels']
    attention_lname = H.get('slim_attention_lname', 'Mixed_3b')
    early_feat = T[attention_lname]
    return coarse_feat, early_feat

def build_overfeat_inner(H, lstm_input):
    '''
    build simple overfeat decoder
    '''
    outputs = []
    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    with tf.variable_scope('Overfeat', initializer=initializer):
        w = tf.get_variable('ip', shape=[H['later_feat_channels'], H['lstm_size']])
        outputs.append(tf.matmul(lstm_input, w))
    return outputs


def build_forward(H, x, reuse):
    '''
    Construct the forward model
    '''

    grid_size = H['grid_width'] * H['grid_height']
    outer_size = grid_size * H['batch_size']
    input_mean = 117.
    x -= input_mean
    cnn, early_feat = googlenet_model(x, H, reuse)
    early_feat_channels = H['early_feat_channels']
    early_feat = early_feat[:, :, :, :early_feat_channels]

    if H['avg_pool_size'] > 1:
        pool_size = H['avg_pool_size']
        cnn1 = cnn[:, :, :, :700]
        cnn2 = cnn[:, :, :, 700:]
        cnn2 = tf.nn.avg_pool(cnn2, ksize=[1, pool_size, pool_size, 1],
                              strides=[1, 1, 1, 1], padding='SAME')
        cnn = tf.concat([cnn1, cnn2], 3)

    cnn = tf.reshape(cnn, [outer_size, H['later_feat_channels']])
    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    with tf.variable_scope('decoder', reuse=reuse, initializer=initializer):
        scale_down = 0.01
        lstm_input = tf.reshape(cnn * scale_down, (outer_size, H['later_feat_channels']))
        lstm_outputs = build_overfeat_inner(H, lstm_input)

        pred_boxes = []
        pred_logits = []
        output = lstm_outputs[0]
        box_weights = tf.get_variable('box_ip0', shape=(H['lstm_size'], 4))
        conf_weights = tf.get_variable('conf_ip0', shape=(H['lstm_size'], 2))
        pred_boxes_step = tf.matmul(output, box_weights) * 50
        pred_boxes.append(pred_boxes_step)
        pred_logits.append(tf.matmul(output, conf_weights))

        pred_boxes = tf.concat(pred_boxes, 1)
        pred_logits = tf.concat(pred_logits, 1)
        pred_confidences = tf.nn.softmax(pred_logits)

        if H['use_rezoom']:
            pred_confs_deltas = []
            pred_boxes_deltas = []
            w_offsets = H['rezoom_w_coords']
            h_offsets = H['rezoom_h_coords']
            num_offsets = len(w_offsets) * len(h_offsets)
            rezoom_features = rezoom(H, pred_boxes, early_feat, early_feat_channels, w_offsets, h_offsets)
            delta_features = tf.concat([lstm_outputs[0], rezoom_features[:, 0, :] / 1000.], 1)
            dim = 128
            shape = [H['lstm_size'] + early_feat_channels*num_offsets, dim]
            delta_weights = tf.get_variable('delta_ip10', shape = shape)
            ip1 = tf.nn.relu(tf.matmul(delta_features, delta_weights))
            delta_confs_weights = tf.get_variable('delta_ip20', shape=[dim, 2])
            if H['reregress']:
                delta_boxes_weights = tf.get_variable('delta_ip_boxes0', shape=[dim, 4])
                pred_boxes_deltas.append(tf.matmul(ip1, delta_boxes_weights) * 5)
                scale = H.get('rezoom_conf_scale', 50)
                pred_confs_deltas.append(tf.matmul(ip1, delta_confs_weights) * scale)
            pred_confs_deltas = tf.concat(pred_confs_deltas, 1)
            if H['reregress']:
                pred_boxes_deltas = tf.concat(pred_boxes_deltas, 1)
            return pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas

    return pred_boxes, pred_logits, pred_confidences
