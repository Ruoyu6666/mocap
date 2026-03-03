import os
import tqdm
import shutil
import numpy as np
import scipy.io
import pandas as pd
import pickle
import h5py
import logging


def compute_svd(points):
    """
    :args 
        points: (n_keypoints, 3D) numpy array. Should be hip, coord and back to approximate the plane of the mouse back.
    :returns: 
        barycenter: (3,) numpy array — mean position of valid points
        transition matrix: (3, 3), both can be in transform_points 
    """
    points = points[~np.any(np.isnan(points), axis=1)]
    if len(points) == 0:
        return np.nan, np.nan

    barycenter = np.mean(points, axis=0)
    _, _, Vt = np.linalg.svd(points - barycenter)
    return barycenter, Vt.T
    











def open_and_extract_data(f, file_type, dlc_likelihood_threshold):
    """
    :args f: (str) path to data file

    Supported formats:
       - .mat from QUALISYS software
       - .csv format with 3 columns per keypoints with names {kp}_x, {kp}_y, {kp}_z

    If keypoints not found in file, then they will be named '0', '1', '2', ...

    :return data: numpy array of shape (timesteps, n_keypoints, 2D or 3D)
    :return keypoints: list of keypoint names as strings
    """
    if file_type == 'mat_dannce':
        mat = scipy.io.loadmat(f)
        # for Rat7M dataset
        data = np.moveaxis(np.array(list(mat['mocap'][0][0])), 1, 0)
        keypoints = list(mat['mocap'][0][0].dtype.fields.keys())

    elif file_type == 'mat_qualisys':
        # for in house mouse data, QUALISYS software
        mat = scipy.io.loadmat(f)
        exp_name = [m for m in mat.keys() if m[:2] != '__'][0]  ## TOCHANGE
        data = np.moveaxis(mat[exp_name][0, 0]['Trajectories'][0, 0]['Labeled']['Data'][0, 0],
                           2, 0)
        keypoints = [label[0].replace('coordinate', 'coord') for label in
                     mat[exp_name][0, 0]['Trajectories'][0, 0]['Labeled']['Labels'][0, 0][0]]

        # very important
        # make sure the keypoints are always in the same order even if not saved so in the original files
        new_order = np.argsort(keypoints)
        keypoints = [keypoints[n] for n in new_order]
        data = data[:, new_order, :]

    elif file_type == 'simple_csv':
        ## for fish data from Liam
        df = pd.read_csv(f)  # columns time, keypoint_x, kp_y, kp_z
        # sort the keypoints with np.unique
        keypoints = list(np.unique([c.rstrip('_xyz') for c in df.columns if c.endswith('_x') or c.endswith('_y') or c.endswith('_z')]))
        columns = []
        for k in keypoints:
            columns.extend([k + '_x', k + '_y', k + '_z'])
        # get the columns corresponding to sorted keypoints so the data can be stacked
        data = df.loc[:, columns].values.reshape((df.shape[0], len(keypoints), -1))

    elif file_type == 'npy':
        ## for human MoCap files
        data = np.array(np.load(f))
        logging.info(f'[WARNING][CREATE_DATASET][OPEN_AND_EXTRACT_DATA function][NPY INPUT FILES] keypoints cannot be loaded from input files. '
                     f'Expected behavior: the columns correspond to the keypoints and are in fixed order')
        # WARNING - here no information about keypoint, so we expect that the columns match for every file
        keypoints = [f'{i:02d}' for i in range(data.shape[1])]

    elif file_type == 'df3d_pkl':
        ## for DeepFly data
        with open(f, 'rb') as openedf:
            pkl_content = pickle.load(openedf)
        data = pkl_content['points3d']
        logging.info(f'[WARNING][CREATE_DATASET][OPEN_AND_EXTRACT_DATA function][PKL INPUT FILES] keypoints cannot be loaded from input files. '
                     f'Expected behavior: the columns correspond to the keypoints and are in fixed order')
        keypoints = [f'{i:02d}' for i in range(data.shape[1])]
        """ from DeepFly3D paper
        38 landmarks per animal: (i) five on each limb – the thorax-coxa, coxa-femur, femur-tibia, and tibia-tarsus 
        joints as well as the pretarsus, (ii) six on the abdomen - three on each side, and (iii) one on each antenna
         - for measuring head rotations.
         see image on github too
        """

    elif file_type == 'dlc_csv':
        ## for csv from DeepLabCut
        df = pd.read_csv(f, header=[1, 2])
        keypoints = [bp for bp in df.columns.levels[0] if bp != 'bodyparts']
        keypoints.sort()
        coordinates = [c for c in df.columns.levels[1] if c != 'likelihood' and c != 'coords']
        likelihood_columns = []
        for k in keypoints:
            likelihood_columns.append((k, 'likelihood'))

        columns = []
        for k in keypoints:
            for c in coordinates:
                df.loc[df.loc[:, (k, 'likelihood')] <= dlc_likelihood_threshold, (k, c)] = np.nan
                columns.append((k, c))
        data = df.loc[:, columns].values.reshape((df.shape[0], len(keypoints), -1))

    elif file_type == 'dlc_h5':
        content = h5py.File(f)
        extracted_content = np.vstack([c[1] for c in content['df_with_missing']['table'][:]])
        mask_columns_likelihood = np.all((extracted_content <= 1) * (extracted_content >= 0), axis=0)
        likelihood_columns = extracted_content[:, mask_columns_likelihood] <= dlc_likelihood_threshold
        coordinates_columns = extracted_content[:, ~mask_columns_likelihood]
        n_dim = coordinates_columns.shape[1] / likelihood_columns.shape[1]
        assert int(n_dim) == n_dim
        n_dim = int(n_dim)
        for i_dim in range(n_dim):
            coordinates_columns[:, i_dim::n_dim][likelihood_columns] = np.nan
        data = coordinates_columns.reshape((coordinates_columns.shape[0], -1, n_dim))
        keypoints = [f'{i:02d}' for i in range(data.shape[1])]

    elif file_type == 'sleap_h5':
        ## compatibility with SLEAP analysis h5 files
        with h5py.File(f, 'r') as openedf:
            if 'tracks_3D_smooth' in openedf.keys():
                data = openedf['tracks_3D_smooth'][:]
                # shape: ['numFrames' 'numFish' 'numBodyPoints' 'XYZ']
                data = np.moveaxis(data, 1, 3)
                # reshape in (numFrames, numBodyPoints, xyz, numFish
                keypoints = [str(i) for i in range(data.shape[1])]
            else:
                data = openedf['tracks'][:].T
                keypoints = [n.decode() for n in openedf["node_names"][:]]

        if data.shape[3] > 1:
            # multi-animal scenario
            new_keypoints = []
            # data.shape[3] should be number of animals
            for animal_id in range(data.shape[3]):
                new_keypoints.extend([f'animal{animal_id}_{k}' for k in keypoints])
            keypoints = new_keypoints
            # move number of animals from 3 to 1 to flatten the 2nd dimension in num animals x num keypoints
            data = np.moveaxis(data, 3, 1).reshape(data.shape[0], -1, data.shape[2])
        else:
            # one animal, remove the last axis
            data = data[..., 0]

        # very important
        # make sure the keypoints are always in the same order even if not saved so in the original files
        new_order = np.argsort(keypoints)
        keypoints = [keypoints[n] for n in new_order]
        data = data[:, new_order]

    else:
        raise ValueError(f'File format not understood {f}, should be one of the following: mat_dannce, '
                         f'mat_qualisys,simple_csv, dlc_csv, npy, df3d_pkl, sleap_h5')

    # we replace the spaces by underscore because when dealing with set of keypoints we separate them by spaces (fake and orginal holes)
    keypoints = [k.replace(' ', '_') for k in keypoints]
    
    return data, keypoints