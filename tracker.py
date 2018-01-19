try:
    import tensorflow as tf
except ImportError:
    raise ImportError('tracker requires tensorflow to be installed and active')
import json
import numpy as np
from build import build_forward
import cv2
import os

class tracker(object):

    def __init__(self, model='lorenzmie'):
        package_directory = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(package_directory, 'models')
        self.config = os.path.join(model_dir, model+'.json')
        self.checkpoint = os.path.join(model_dir, model+'.ckpt')

        with open(self.config, 'r') as f:
            H = json.load(f)
        shape = [H['image_height'], H['image_width'], 3]
        self.grid_height = H['grid_height']
        self.grid_width = H['grid_width']
        self.region_size = H['region_size']
        
        tf.reset_default_graph()
        self.x_in = tf.placeholder(tf.float32, name='x_in', shape=shape)
        (boxes,
         logits,
         confidences,
         conf_deltas,
         box_deltas) = build_forward(H, tf.expand_dims(self.x_in, 0), reuse=False)
        self.confidences = tf.nn.softmax(conf_deltas)
        self.boxes = boxes + box_deltas
        
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, self.checkpoint)
        self.sess = sess

    def predict(self, img, min_confidence=0.99):
        feed = {self.x_in: img}
        boxes, confidences = self.sess.run([self.boxes, self.confidences], feed_dict=feed)
        boxes = np.reshape(boxes, (self.grid_height, self.grid_width, 4))
        confidences = np.reshape(confidences, (self.grid_height, self.grid_width, 2))
        goodx, goody = (confidences[:,:,1] > min_confidence).nonzero()
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                bbox = boxes[y, x, :]
                boxes[y, x, 0] = int(bbox[0]) + self.region_size/2 + self.region_size * x
                boxes[y, x, 1] = int(bbox[1]) + self.region_size/2 + self.region_size * y
        goodx, goody = (confidences[:,:,1] > min_confidence).nonzero()
        boxes = boxes[goodx,goody,:]
        rects, _ = cv2.groupRectangles(boxes.tolist(), 1)
        return rects

def main():
    import glob
    from scipy.misc import imread
    
    a = tracker()
    fn = glob.glob('../data/lorenzmie/validation/*.png')
    im = imread(fn[15], mode='RGB')
    pred, conf = a.predict(im)
    
