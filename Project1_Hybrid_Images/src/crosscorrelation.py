import numpy as np


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
