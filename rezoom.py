import tensorflow as tf

def to_idx(vec, w_shape):
    '''
    vec = (idn, idh, idw)
    w_shape = [n, h, w, c]
    '''
    return vec[:, 2] + w_shape[2] * (vec[:, 1] + w_shape[1] * vec[:, 0])

def interp(w, i, channel_dim):
    '''
    Input:
        w: A 4D block tensor of shape (n, h, w, c)
        i: A list of 3-tuples [(x_1, y_1, z_1), (x_2, y_2, z_2), ...],
            each having type (int, float, float)

        The 4D block represents a batch of 3D image feature volumes with c channels.
        The input i is a list of points  to index into w via interpolation. Direct
        indexing is not possible due to y_1 and z_1 being float values.
    Output:
        A list of the values: [
            w[x_1, y_1, z_1, :]
            w[x_2, y_2, z_2, :]
            ...
            w[x_k, y_k, z_k, :]
        ]
        of the same length == len(i)
    '''
    w_as_vector = tf.reshape(w, [-1, channel_dim]) # gather expects w to be 1-d
    upper_l = tf.to_int32(tf.concat([i[:, 0:1], tf.floor(i[:, 1:2]), tf.floor(i[:, 2:3])], 1))
    upper_r = tf.to_int32(tf.concat([i[:, 0:1], tf.floor(i[:, 1:2]), tf.ceil(i[:, 2:3])], 1))
    lower_l = tf.to_int32(tf.concat([i[:, 0:1], tf.ceil(i[:, 1:2]), tf.floor(i[:, 2:3])], 1))
    lower_r = tf.to_int32(tf.concat([i[:, 0:1], tf.ceil(i[:, 1:2]), tf.ceil(i[:, 2:3])], 1))

    upper_l_idx = to_idx(upper_l, tf.shape(w))
    upper_r_idx = to_idx(upper_r, tf.shape(w))
    lower_l_idx = to_idx(lower_l, tf.shape(w))
    lower_r_idx = to_idx(lower_r, tf.shape(w))

    upper_l_value = tf.gather(w_as_vector, upper_l_idx)
    upper_r_value = tf.gather(w_as_vector, upper_r_idx)
    lower_l_value = tf.gather(w_as_vector, lower_l_idx)
    lower_r_value = tf.gather(w_as_vector, lower_r_idx)

    alpha_lr = tf.expand_dims(i[:, 2] - tf.floor(i[:, 2]), 1)
    alpha_ud = tf.expand_dims(i[:, 1] - tf.floor(i[:, 1]), 1)

    upper_value = (1 - alpha_lr) * upper_l_value + (alpha_lr) * upper_r_value
    lower_value = (1 - alpha_lr) * lower_l_value + (alpha_lr) * lower_r_value
    value = (1 - alpha_ud) * upper_value + (alpha_ud) * lower_value
    return value

def bilinear_select(H, pred_boxes, early_feat, early_feat_channels, w_offset, h_offset):
    '''
    Function used for rezooming high level feature maps. Uses bilinear interpolation
    to select all channels at index (x, y) for a high level feature map, where x and y are floats.
    '''
    grid_size = H['grid_width'] * H['grid_height']
    outer_size = grid_size * H['batch_size']

    fine_stride = 8. # pixels per 60x80 grid cell in 480x640 image
    coarse_stride = H['region_size'] # pixels per 15x20 grid cell in 480x640 image
    batch_ids = []
    x_offsets = []
    y_offsets = []
    for n in range(H['batch_size']):
        for i in range(H['grid_height']):
            for j in range(H['grid_width']):
                batch_ids.append([n])
                x_offsets.append([coarse_stride / 2. + coarse_stride * j])
                y_offsets.append([coarse_stride / 2. + coarse_stride * i])

    batch_ids = tf.constant(batch_ids)
    x_offsets = tf.constant(x_offsets)
    y_offsets = tf.constant(y_offsets)

    scale_factor = coarse_stride / fine_stride # scale difference between 15x20 and 60x80 features

    pred_x_center = (pred_boxes[:, 0:1] + w_offset * pred_boxes[:, 2:3] + x_offsets) / fine_stride
    pred_x_center_clip = tf.clip_by_value(pred_x_center,
                                     0,
                                     scale_factor * H['grid_width'] - 1)
    pred_y_center = (pred_boxes[:, 1:2] + h_offset * pred_boxes[:, 3:4] + y_offsets) / fine_stride
    pred_y_center_clip = tf.clip_by_value(pred_y_center,
                                          0,
                                          scale_factor * H['grid_height'] - 1)

    interp_indices = tf.concat([tf.to_float(batch_ids), pred_y_center_clip, pred_x_center_clip], 1)
    return interp_indices

def rezoom(H, pred_boxes, early_feat, early_feat_channels, w_offsets, h_offsets):
    '''
    Rezoom into a feature map at multiple interpolation points in a grid.

    If the predicted object center is at X, len(w_offsets) == 3, and len(h_offsets) == 5,
    the rezoom grid will look as follows:

    [o o o]
    [o o o]
    [o X o]
    [o o o]
    [o o o]

    Where each letter indexes into the feature map with bilinear interpolation
    '''

    grid_size = H['grid_width'] * H['grid_height']
    indices = []
    for w_offset in w_offsets:
        for h_offset in h_offsets:
            indices.append(bilinear_select(H,
                                           pred_boxes,
                                           early_feat,
                                           early_feat_channels,
                                           w_offset, h_offset))

    interp_indices = tf.concat(indices, 0)
    rezoom_features = interp(early_feat, interp_indices, early_feat_channels)
    rezoom_features_r = tf.reshape(rezoom_features,
                                   [len(w_offsets) * len(h_offsets),
                                    grid_size,
                                    1,
                                    early_feat_channels])
    rezoom_features_t = tf.transpose(rezoom_features_r, [1, 2, 0, 3])
    return tf.reshape(rezoom_features_t,
                      [grid_size,
                       1,
                       len(w_offsets) * len(h_offsets) * early_feat_channels])
