import numpy as np


def canvas(img_size, kernel_size):
    img_width = img_size[1]
    img_height = img_size[0]
    kernel_width = kernel_size[1]
    kernel_height = kernel_size[0]
    output_width = img_width - (kernel_width - 1)
    output_height = img_height - (kernel_height - 1)
    return np.zeros((output_height, output_width))


def handle_edge(img, output_size, type='extend'):
    if img.shape == output_size:
        return img
    assert type == 'extend'
    img_height = img.shape[0]
    img_width = img.shape[1]
    n_extension_rows = output_size[0] - img_height
    n_extension_columns = output_size[1] - img_width
    assert n_extension_rows % 2 == 0
    assert n_extension_columns % 2 == 0

    top = np.repeat(img[:1, :], n_extension_rows / 2, axis=0)
    img = np.vstack((top, img))
    bottom = np.repeat(img[-1:, :], n_extension_rows / 2, axis=0)
    img = np.vstack((img, bottom))
    left = np.repeat(img[:, :1], n_extension_columns / 2, axis=1)
    img = np.c_[left, img]
    right = np.repeat(img[:, -1:], n_extension_columns / 2, axis=1)
    img = np.c_[img, right]
    return img

