import os
import tqdm
import shutil
import numpy as np
import scipy.io
import pandas as pd
import pickle
import h5py
import logging

from .transform import ViewInvariant, Normalize, NormalizeCube


label_map = {
    "CP1A":{
        "A": {"drug":"V", "concentration":0},
        "B": {"drug":"CP", "concentration":0.03},
        "C": {"drug":"CP", "concentration":0.01},
        "D": {"drug":"CP", "concentration":0.3},
    },
    "CP1B":{
        "A": {"drug":"V", "concentration":0},
        "B": {"drug":"CP", "concentration":0.03},
        "C": {"drug":"CP", "concentration":0.01},
        "D": {"drug":"CP", "concentration":0.3},
    }, 
    "INH1":{
        "A": {"drug":"V", "concentration":0},
        "B": {"drug":"PF", "concentration":30},
        "C": {"drug":"MJ", "concentration":2.5},
    },
    "INH2":{
        "A": {"drug":"V", "concentration":0},
        "B": {"drug":"AM", "concentration":3},
        "C": {"drug":"PF", "concentration":10},
        "D": {"drug":"MJ", "concentration":1.25},
        "E": {"drug":"AMPF", "concentration":[3, 10]},
        "F": {"drug":"AMMJ", "concentration":[3, 1.25]},
    }, 
    "MOS1aD":{
        "1": {"drug":"V", "concentration":0},
        "2": {"drug":"V", "concentration":0},
        "B": {"drug":"CP", "concentration":0.3},
        "C": {"drug":"PF", "concentration":30},
        "D": {"drug":"MJ", "concentration":2.5},
        "E": {"drug":"H", "concentration":20},
    }
}

class PreprocessPipeline:

    NUM_KEYPOINTS = 10
    KPTS_DIMENSIONS = 3

    def __init__(self,
                path_to_data,
                task            = "CLB",
                left_idx        = 3,       # default left hip
                right_idx       = 8,       # default right hip
                sampling_rate   = 5,
                num_frames      = 300,
                sliding_window  = 150,
                index_frame     = 149,      # int(length_input_seq / 2)
                normalizer      = 'normal', # 'cube' or 'normal'
        ):
        self.task = task
        self.path_to_data  = path_to_data
        self.left_idx      = left_idx
        self.right_idx     = right_idx

        self._sampling_rate = sampling_rate
        self.max_keypoints_len = num_frames
        self.sliding_window = sliding_window
        self.index_frame   = index_frame

        self.vi   = ViewInvariant(index_frame = index_frame, left_idx = left_idx, right_idx = right_idx,)
        self.norm = Normalize() if normalizer == 'normal' else NormalizeCube()

        self.load()
        self.sample()

    def __str__(self):
        return (f'SkeletonPreprocessingPipeline('
                f'subseq_len={self.max_keypoints_len}, '
                f'sliding_window={self.sliding_window}, ')

    
    # Step 0: load raw data
    def load(self):
        filepath = self.path_to_data
        if not filepath.exists():
            raise FileNotFoundError(f"[Pipeline] File not found: {filepath}")

        with open(filepath, 'rb') as file:  # Read pickle file 
            self.raw_data = pickle.load(file)
        

    # Step 1: Sample 1/5 frames & Prepare subsequences Ids 
    def sample(self):
        seq_keypoints = []
        keypoints_ids = []
        sub_seq_length = self.max_keypoints_len
        self.labels = []
        
        for seq_ix, (seq_name, sequence) in enumerate(self.raw_data.items()):
            self.labels.append(sequence["label"])

            vec_seq = sequence["data"]
            vec_seq = vec_seq.reshape(-1, self._sampling_rate, self.NUM_KEYPOINTS, self.KPTS_DIMENSIONS).transpose(1, 0, 2, 3)[0]   # [T/5, 10, 3]

            # Pads the beginning and end of the sequence with duplicate frames
            pad_vec = np.pad(vec_seq, ((sub_seq_length// 2, sub_seq_length - sub_seq_length // 2), (0, 0), (0, 0)), mode="edge", )
            seq_keypoints.append(pad_vec)
            keypoints_ids.extend([(seq_ix, i) for i in np.arange(0, len(pad_vec) - sub_seq_length + 1, self.sliding_window)])
            
        #seq_keypoints = np.array(seq_keypoints, dtype=np.float32)
        self.seq_keypoints = seq_keypoints #numpy array: [num_sequences, T/5 + pad_vect, 10, 3]
        self.keypoints_ids = keypoints_ids

        del self.raw_data
    

    # Step 2: Apply ViewInvariant → Normalize to a single subsequence.  
    def transform(self, seq):
        """
        Args:       seq: (subseq_len, J, 3)
        Returns:    processed: (subseq_len, J, 3)
                    metadata:  dict — VI_angle, VI_barycenter, min_sample, max_sample
        """
        kwargs = {}
        seq, _, kwargs = self.vi(seq,   x_supp=(), **kwargs)
        seq, _, kwargs = self.norm(seq, x_supp=(), **kwargs)
        return seq, kwargs

    def untransform(self, seq, metadata):
        """Invert the full pipeline: Normalize⁻¹ → ViewInvariant⁻¹.
        Args:       seq:      (subseq_len, J, 3)
                    metadata: dict returned by transform()
        Returns:    (subseq_len, J, 3) in original coordinate space
        """
        seq = self.norm.untransform(seq, **metadata)
        seq = self.vi.untransform(seq,   **metadata)
        return seq
    
    #  Run
    def run(self):
        data = []
        self.drug_label = []
        self.concentration_label = []
        dir_name, file_name = os.path.split(self.path_to_data)# Split directory and filename
        name, ext = os.path.splitext(file_name)               # Split filename and extension: CP1A, pkl
        label_map_data = label_map[name[:-4]]

        for subseq_ix in self.keypoints_ids:
            subsequence = self.seq_keypoints[subseq_ix[0]][subseq_ix[1] : subseq_ix[1] + self.max_keypoints_len]
            feats, _ = self.transform(subsequence)
            data.append(feats)

            label = label_map_data[self.labels[subseq_ix[0]]]
            self.drug_label.append(label["drug"])
            self.concentration_label.append(label["concentration"])
        
        data = np.array(data, dtype=np.float32)
        # Write data to local
        processed_data = {
            "data":data,
            "drug": self.drug_label,
            "concentration": self.concentration_label
        }
        new_file_name = f"{name}_processed{ext}"              # Create new filename
        output_path = os.path.join(dir_name, new_file_name)   # Build new path

        with open(output_path, 'wb') as file:
            pickle.dump(processed_data, file)











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