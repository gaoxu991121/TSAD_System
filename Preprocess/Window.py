import numpy as np


def convertToWindow(data, window_size):
    """
    stride为1，前window_size -1 个时间点的时间窗口，通过复制前面元素构成
    """
    windows = []

    for i, g in enumerate(data):
        if i >= window_size:
            w = data[i - window_size + 1:i + 1]
        else:

            w = np.concatenate([np.tile(data[0], window_size - i).reshape(window_size - i, -1), data[1:i + 1]])

        windows.append(w)
    return np.stack(windows)


def convertToSlidingWindow(data, window_size, stride=1, start_discont=np.array([])):
    """
    :param start_discont: the start points of each sub-part in case the data is just multiple parts joined together
    :param data: dim 0 is time, dim 1 is channels
    :param window_size: size of window used to create subsequences from the data
    :param stride: number of time points the window will move between two subsequences
    :return:
    """
    excluded_starts = []
    [excluded_starts.extend(range((start - window_size + 1), start)) for start in start_discont if start > window_size]
    seq_starts = np.delete(np.arange(0, data.shape[0] - window_size + 1, stride), excluded_starts)
    data = data
    x_seqs = np.array([data[i:i + window_size] for i in seq_starts])
    return x_seqs
