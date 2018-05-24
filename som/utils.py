import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def argmax_2d(tensor):
    # input format: HxW
    assert rank(tensor) == 2

    # flatten the Tensor along the height and width axes
    flat_tensor = tf.reshape(tensor, [-1])

    # argmax of the flat tensor
    argmax = tf.cast(tf.argmax(flat_tensor), tf.int32)

    # convert indexes into 2D coordinates
    argmax_x = argmax // tf.shape(tensor)[1]
    argmax_y = argmax % tf.shape(tensor)[1]

    # stack and return 2D coordinates
    return tf.stack((argmax_x, argmax_y))


def rank(tensor):
    # return the rank of a Tensor
    return len(tensor.get_shape())


def mesh_2d(num_x, num_y):
    x = np.linspace(0, num_x-1, num_x, dtype=np.float32)
    y = np.linspace(0, num_y-1, num_y, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="ij")
    mesh = np.stack([X, Y], axis=-1)
    return mesh


def matrix_indices_2d(num_x, num_y):
    mesh = mesh_2d(num_x, num_y)
    return mesh.reshape([num_x * num_y, 2])


def view_bmus(bestmatches, labels, cmap=None, annotate=False, legend=False, default_marker_size=20):
    if cmap is None:
        cmap = plt.cm.get_cmap('RdBu')
    if labels is None:
        colors = None
        sizes = None
        bmus_to_show = bestmatches
    else:
        labels_set = set(labels)
        colors_list = [cmap(float(i) / len(labels_set)) for i in range(len(labels_set))]

        label_to_color = dict(zip(labels_set, colors_list))

        colors = []
        sizes = []
        counter = Counter([str(b)+str(l) for (b, l) in zip(bestmatches, labels)])
        bmus_to_show = []
        labels_to_show = []

        for i in range(bestmatches.shape[0]):
            bmu = bestmatches[i]
            l = labels[i]
            if str(bmu)+str(l) in counter:
                colors.append(label_to_color[l])
                sizes.append(default_marker_size * counter[str(bestmatches[i]) + str(l)] ** 2)
                bmus_to_show.append(bmu)
                labels_to_show.append(l)
                del counter[str(bmu)+str(l)]

        #
        # for i, l in enumerate(labels):
        #     colors.append(label_to_color[l])
        #     sizes.append(default_marker_size * counter[str(bestmatches[i])+str(l)]**2)
        bmus_to_show = np.array(bmus_to_show)
    plt.figure(figsize=(14, 14))
    plt.grid()
    plt.scatter(bmus_to_show[:, 0], bmus_to_show[:, 1], c=colors, s=sizes, alpha=0.5)

    if labels_to_show is not None:
        for label, col, row in zip(labels_to_show,
                                   bmus_to_show[:, 0], bmus_to_show[:, 1]):
            if label is not None and annotate:
                plt.annotate(label, xy=(col, row), xytext=(10, -5),
                             textcoords='offset points', ha='left',
                             va='bottom',
                             bbox=dict(boxstyle='round,pad=0.3',
                                       fc='white', alpha=0.8))
    if legend:
        plt.legend(scatter_points=1)


    plt.show()
    return plt


def test_view_bmus():
    bmus = [[1, 1],
            [1, 1],
            [3, 7],
            [3, 4],
            [5, 9],
            [9, 1],
            [1, 1],
            [5, 9]]

    labels = [1, 1, 2, 3, 4, 5, 1, 4]

    print(str(bmus[0])+"label")
    view_bmus(np.array(bmus), labels)

# test_view_bmus()
