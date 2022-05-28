import numpy as np


def load_data(path):
    """Loads data from path."""
    return np.load(path + 'X_all_train.npy'), np.load(path + 'X_all_test.npy'), \
           np.load(path + 'X_select_train.npy'), np.load(path + 'X_select_test.npy'), \
           np.load(path + 'T_train.npy'), np.load(path + 'T_test.npy'), \
           np.load(path + 'E_train.npy'), np.load(path + 'E_test.npy')


# 找到最接近0.5的那个值
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
