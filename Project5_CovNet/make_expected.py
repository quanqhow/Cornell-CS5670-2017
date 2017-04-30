"""
Saves expected values for the unit tests
"""

import load_alexnet
import student_solution as student
import numpy as np

#### LOAD ALEXNET ####
# Fix the PYTHONPATH before importing caffe
import os, sys
caffe_root = os.path.expanduser('~/caffe/')
sys.path.insert(0, caffe_root + 'python')
import caffe
net, transformer = load_alexnet.load_alexnet(caffe_root, gpu=False)

def _save(fname, arr):
    assert arr.dtype == np.float32
    np.savez_compressed(fname, arr)

# Prepare image
data = transformer.preprocess(
    'data', caffe.io.load_image('dataset/test-dog/HOND1.jpg')
).astype(np.float32)
_save('expected/data.npz', data)


def expected_compute_dscore_dimage():
    net.blobs['data'].data[0, ...] = data.copy()
    net.forward()
    grad = student.compute_dscore_dimage(net, data, 254)
    _save('expected/compute_dscore_dimage-grad.npz', grad)


def expected_normalized_sgd_with_momentum_update():
    grad = np.random.randn(*data.shape).astype(np.float32)
    velocity = np.random.randn(*data.shape).astype(np.float32)
    _save('expected/normalized_sgd_with_momentum_update-grad.npz', grad)
    _save('expected/normalized_sgd_with_momentum_update-velocity.npz', velocity)

    momentum = 0.99
    learning_rate = 100

    new_data, new_velocity = student.normalized_sgd_with_momentum_update(
        data.copy(), grad, velocity, momentum, learning_rate)

    _save('expected/normalized_sgd_with_momentum_update-new_data.npz', new_data)
    _save('expected/normalized_sgd_with_momentum_update-new_velocity.npz', new_velocity)


def expected_fooling_image_gradient():
    net.blobs['data'].data[0, ...] = data.copy()
    net.forward()
    orig_data = data + np.random.randn(*data.shape).astype(np.float32)
    _save('expected/fooling_image_gradient-orig_data.npz', orig_data)

    target_class = 113
    regularization = 1e-3

    grad = student.fooling_image_gradient(
        net, orig_data, data.copy(), target_class, regularization)

    _save('expected/fooling_image_gradient-grad.npz', grad)


def expected_class_visualization_gradient():
    cur_data = data + np.random.randn(*data.shape).astype(np.float32)
    _save('expected/class_visualization_gradient-cur_data.npz', cur_data)

    target_class = 234
    regularization = 1e-3

    net.blobs['data'].data[0, ...] = cur_data
    net.forward()

    grad = student.class_visualization_gradient(
        net, cur_data, target_class, regularization)

    _save('expected/class_visualization_gradient-grad.npz', grad)


def expected_feature_inversion_gradient():
    blob_name = 'conv3'
    regularization = 2e-3

    cur_data = np.random.randn(*data.shape).astype(np.float32)
    _save('expected/feature_inversion_gradient-cur_data.npz', cur_data)

    net.blobs['data'].data[0, ...] = cur_data
    net.forward(end=blob_name)
    target_feat = np.copy(net.blobs[blob_name].data[0, ...])
    _save('expected/feature_inversion_gradient-target_feat.npz', target_feat)

    grad = student.feature_inversion_gradient(
        net, cur_data, blob_name, target_feat, regularization)

    _save('expected/feature_inversion_gradient-grad.npz', grad)


print "Saving expected results..."
expected_normalized_sgd_with_momentum_update()
expected_compute_dscore_dimage()
expected_fooling_image_gradient()
expected_class_visualization_gradient()
expected_feature_inversion_gradient()
print "Done"
