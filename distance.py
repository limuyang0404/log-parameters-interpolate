import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import segyio
import pandas as pd
def segy_load(segy_file):
    '''
    :param segy_file: segy file's path
    :return: seismic amplitude cube's mean value, seismic amplitude cube's deviation, seismic amplitude cube
    This function use segyio to read segy file
    '''
    cube = segyio.tools.cube(segy_file)
    print('The segy file have been loaded!')
    print(r"cube's size:", type(cube), cube.shape)
    cube_amp_mean = np.mean(cube)
    cube_amp_deviation = np.var(cube) ** 0.5
    cube = (cube - cube_amp_mean) / cube_amp_deviation
    return cube_amp_mean, cube_amp_deviation, cube

def well_log_file(file_path, para_name):
    delete_list = 0
    df = pd.read_csv(file_path, header=0)
    mean = df[para_name].mean()
    print('mean:', mean, type(mean))
    if mean is np.nan:
        delete_list = 1
    print('mean:', mean, file_path)
    output = np.zeros(shape=(664,), dtype='float32') + mean
    for i in range(len(df)):
        if 1560<=df['TWT'][i]<=2886:
            a = int((df['TWT'][i]-1560)/2)
            # print('a:', a)
            output[a] = df[para_name][i]
    return output, delete_list
def weight_cal(distance):
    in_distance = 1/(distance + 0.00000001)
    weight_output = in_distance * 1
    whole_weight = 0
    for i in range(len(weight_output)):
        weight_output[i] = weight_output[i]
        whole_weight += in_distance[i]
    weight_output = weight_output / whole_weight
    return weight_output
def predict(weight, para_value):
    output = np.zeros(shape=(664,), dtype='float32')
    # print('output:', output)
    for i in range(len(weight)):
        # print('output:', output)
        # print('weight:', weight[i], para_value[i], i)
        output += weight[i] * para_value[i]
    return output
def well_para_list(well_name, para_name, well_I, well_X):
    output = []
    output_I = []
    output_X = []
    for i in range(len(well_name)):
        file_path = '/home/limuyang/yigoushujv/SW/lowpass_log/' + well_name[i] + '_' + para_name + '.csv'
        print('file_path:', file_path)
        new_output, new_delete = well_log_file(file_path, para_name)
        if new_delete == 0:
            output.append(new_output)
            output_I.append(well_I[i])
            output_X.append(well_X[i])

    return output, output_I, output_X
def distance_cal(distance_I, distance_X):
    output = distance_I * 1
    for i in range(len(distance_I)):
        output[i] = (distance_I[i] ** 2 + distance_X[i] ** 2) ** 0.5
    return output
def distance_inte(data_cube, well_I, well_X, well_para_value_list):
    well_I = np.array(well_I)
    well_X = np.array(well_X)
    predict_cube = np.zeros(shape=data_cube.shape, dtype='float32')
    for i in range(398):
        for j in range(300):
            # distance = ((well_I - i - 3553) ** 2 + (well_X - j - 1801) ** 2) ** 0.5

            distance = distance_cal(well_I - i - 3553, well_X - j - 1801)
            # print('distance:', distance, i, j)
            weight = weight_cal(distance)
            # print('weight:', weight, i, j)
            predict_cube[i, j, :] = predict(weight, well_para_value_list)
            # print('predict:', predict_cube[i, j, :], i, j)
    predict_cube.tofile(r"predict_SW20211223.dat", format='%f')
    print(predict_cube[104, :, :])
    return
from scipy.spatial import cKDTree

class tree(object):
    """
    Compute the score of query points based on the scores of their k-nearest neighbours,
    weighted by the inverse of their distances.

    @reference:
    https://en.wikipedia.org/wiki/Inverse_distance_weighting

    Arguments:
    ----------
        X: (N, d) ndarray
            Coordinates of N sample points in a d-dimensional space.
        z: (N,) ndarray
            Corresponding scores.
        leafsize: int (default 10)
            Leafsize of KD-tree data structure;
            should be less than 20.

    Returns:
    --------
        tree instance: object

    Example:
    --------

    # 'train'
    idw_tree = tree(X1, z1)

    # 'test'
    spacing = np.linspace(-5., 5., 100)
    X2 = np.meshgrid(spacing, spacing)
    X2 = np.reshape(X2, (2, -1)).T
    z2 = idw_tree(X2)

    See also:
    ---------
    demo()

    """
    def __init__(self, X=None, z=None, leafsize=10):
        if not X is None:
            self.tree = cKDTree(X, leafsize=leafsize )
        if not z is None:
            self.z = np.array(z)

    def fit(self, X=None, z=None, leafsize=10):
        """
        Instantiate KDtree for fast query of k-nearest neighbour distances.

        Arguments:
        ----------
            X: (N, d) ndarray
                Coordinates of N sample points in a d-dimensional space.
            z: (N,) ndarray
                Corresponding scores.
            leafsize: int (default 10)
                Leafsize of KD-tree data structure;
                should be less than 20.

        Returns:
        --------
            idw_tree instance: object

        Notes:
        -------
        Wrapper around __init__().

        """
        return self.__init__(X, z, leafsize)

    def __call__(self, X, k=19, eps=1e-6, p=2, regularize_by=1e-9):
        """
        Compute the score of query points based on the scores of their k-nearest neighbours,
        weighted by the inverse of their distances.

        Arguments:
        ----------
            X: (N, d) ndarray
                Coordinates of N query points in a d-dimensional space.

            k: int (default 6)
                Number of nearest neighbours to use.

            p: int or inf
                Which Minkowski p-norm to use.
                1 is the sum-of-absolute-values "Manhattan" distance
                2 is the usual Euclidean distance
                infinity is the maximum-coordinate-difference distance

            eps: float (default 1e-6)
                Return approximate nearest neighbors; the k-th returned value
                is guaranteed to be no further than (1+eps) times the
                distance to the real k-th nearest neighbor.

            regularise_by: float (default 1e-9)
                Regularise distances to prevent division by zero
                for sample points with the same location as query points.

        Returns:
        --------
            z: (N,) ndarray
                Corresponding scores.
        """
        self.distances, self.idx = self.tree.query(X, k, eps=eps, p=p)
        self.distances += regularize_by
        weights = self.z[self.idx.ravel()].reshape(self.idx.shape)
        mw = np.sum(weights/self.distances, axis=1) / np.sum(1./self.distances, axis=1)
        return mw

    def transform(self, X, k=19, p=2, eps=1e-6, regularize_by=1e-9):
        """
        Compute the score of query points based on the scores of their k-nearest neighbours,
        weighted by the inverse of their distances.

        Arguments:
        ----------
            X: (N, d) ndarray
                Coordinates of N query points in a d-dimensional space.

            k: int (default 6)
                Number of nearest neighbours to use.

            p: int or inf
                Which Minkowski p-norm to use.
                1 is the sum-of-absolute-values "Manhattan" distance
                2 is the usual Euclidean distance
                infinity is the maximum-coordinate-difference distance

            eps: float (default 1e-6)
                Return approximate nearest neighbors; the k-th returned value
                is guaranteed to be no further than (1+eps) times the
                distance to the real k-th nearest neighbor.

            regularise_by: float (default 1e-9)
                Regularise distances to prevent division by zero
                for sample points with the same location as query points.

        Returns:
        --------
            z: (N,) ndarray
                Corresponding scores.

        Notes:
        ------

        Wrapper around __call__().
        """
        return self.__call__(X, k, eps, p, regularize_by)
well_name = ['DK1', 'N80', 'N81', 'N83', 'N84', 'N85', 'N91', 'N102', 'W11', 'W39', 'W53', 'W54', 'W55', 'W57', 'W69', 'W70', 'W72', 'W78', 'W79']
well_position_I = [3657, 3858, 3922, 3645, 3596, 3666, 3889, 3573, 3911, 3615, 3638, 3742, 3661, 3572, 3590, 3800, 3941, 3606, 3713]
well_position_X = [1810, 1912, 1988, 1997, 2020, 2040, 1903, 2056, 1865, 1812, 1836, 1809, 1854, 1815, 1987, 1850, 1803, 1913, 1915]

# _, _, cube = segy_load(r"/nfs/opendtect-data/Niuzhuang/Export/seismic_east.sgy")
# cube = cube[:, :, 780:1444]
well_parameter_list, well_position_I, well_position_X = well_para_list(well_name, para_name='AC', well_I=well_position_I, well_X=well_position_X)
# # print('well_parameter_list[0]:', well_parameter_list[0])
# print('%'*50)
print('len well_parameter_list:', len(well_parameter_list))
# # print('well_parameter_list:', well_parameter_list)
# distance_inte(cube, well_position_I, well_position_X, well_parameter_list)
X = np.zeros(shape=(len(well_position_I), 2))
for i in range(len(well_position_I)):
    X[i, 0] = well_position_I[i] - 3553
    X[i, 1] = well_position_X[i] - 1801
X2 = np.zeros(shape=(398, 300, 2))
for i in range(398):
    for j in range(300):
        X2[i, j, 0] = i
        X2[i, j, 1] = j
X2 = X2.reshape(398*300, 2)
def idw_predict(X, X2, well_parameter):
    predict_result = np.zeros(shape=(398, 300, 664), dtype='float32')
    for i in range(664):
        Z = np.zeros(shape=(X.shape[0],))
        for j in range(Z.shape[0]):
            Z[j] = well_parameter[j][i]
        idw_tree = tree(X, Z)
        output = idw_tree(X2)
        output = output.reshape(398, 300)
        predict_result[:, :, i] = output
    predict_result.tofile(r"/home/limuyang/yigoushujv/AC/predict_AC20211223.dat", format='%f')
    return
idw_predict(X, X2, well_parameter_list)



