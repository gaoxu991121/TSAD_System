import numpy as np


def convertToSlidngWindow(data,window_size):

    windows = []

    for i, g in enumerate(data):
        if i >= window_size:
            w = data[i - window_size + 1:i + 1]
        else:

            w = np.concatenate([np.tile(data[0], window_size - i).reshape(window_size - i, -1), data[1:i + 1]])

        windows.append(w)
    return np.stack(windows)



