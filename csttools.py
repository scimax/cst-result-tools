import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import glob
from io import StringIO
import json
from pandas.errors import EmptyDataError
import h5py
from scipy.constants import m_e, c, e

import logging


# def read_CST_ASCII_format(path, scanParamName):
#     '''
#     path: str
#         path to the txt file exported from CST
#     scanParamName: str
#         parameter name for matching the column names and stitching the tables together 
#     '''
#     df_nPart_time_by_phi = None
#     strBuf1 = StringIO("")
#     with open(path, "r") as f:
#         for line in f:
#             if line != "\n":
#                 # 1. append to buffer
#                 strBuf1.write(line)
#             else:
#                 # 1. Reset buffer position
#                 strBuf1.seek(0)
#                 # 2. Create dataframe from buffer
#                 df_nPart_temp=pd.read_csv(strBuf1,
#                         sep=r" {8,}", comment="--", engine="python")
#                 # 3. Clear Buffer
#                 strBuf1 = StringIO("")
#                 # 4. Rename dataframe 
#                 old_colName = df_nPart_temp.columns[1]
#                 new_colName = re.search(r"{}\=\d+".format(scanParamName), old_colName)[0]
#                 df_nPart_temp.rename(columns={old_colName : new_colName}, inplace=True )
#                 # 5. append dataframe if it's not the first one
#                 if df_nPart_time_by_phi is None:
#                     df_nPart_time_by_phi = df_nPart_temp.copy()
#                 else:
#                     df_nPart_time_by_phi[new_colName] = df_nPart_temp[new_colName]
#     return df_nPart_time_by_phi     

# df_nPart_time_by_phi = None
def read_CST_ASCII_format(path, return_separately=True):
    '''
    path: str
        path to the txt file exported from CST
    return_separately: boolean, optional
        If True every table contained in the ASCII file is returned as a dataframe. If
        False the dataframes are merged using pd.merge(how="left"). Make sure manually 
        that the merge makes sense.
        Default is True.
    '''
    strBuf1 = StringIO("")
    dfs_in_file = []
    with open(path, "r") as f:
        for line in f:
            if line != "#\n":
                # 1. append to buffer
                if line.startswith('#"'):
                    strBuf1.write(line[1:])
                elif not line.startswith("#-"):
                    strBuf1.write(line)  
            else:
                # 1. Reset buffer position
                strBuf1.seek(0)
                # 2. Create dataframe from buffer
                try:
                    df=pd.read_csv(strBuf1, sep="\s+")
                    dfs_in_file.append(df.copy())
                # 3. Clear Buffer
                except EmptyDataError as e:
                    print("empty dataset. continue parsing.")
                finally:
                    strBuf1 = StringIO("")
        # Parse remaining lines at the end
        # 1. Reset buffer position
        strBuf1.seek(0)
        # 2. Create dataframe from buffer
        try:
            df=pd.read_csv(strBuf1, sep="\s+")
            dfs_in_file.append(df.copy())
        # 3. Clear Buffer
        except EmptyDataError as e:
            print("empty dataset. continue parsing.")
    if return_separately:
        return dfs_in_file
    else:
        df_all_in_one = dfs_in_file[0]
        for df_temp in dfs_in_file[1:]:
            df_all_in_one = pd.merge(df_all_in_one, df_temp, how="left")
        return df_all_in_one


# https://gist.github.com/scimax/fd368299408bf99c359ea5bbf693865a

def read_parametric_CST_ASCII_format(path):
    '''
    path: str
        path to the txt file exported from CST
    '''
    params_and_dfs = []
    param_dict = None
    strBuf1 = StringIO("")
    with open(path, "r") as f:
        for line in f:
            if not line.startswith("#Parameters"):
                # 1. append to buffer
                strBuf1.write(line)
            else:
                # 1. Reset buffer position
                strBuf1.seek(0)
                # 2. Create dataframe from buffer
                if param_dict is not None:
                    df = pd.read_csv(strBuf1, skiprows=[1], delimiter="\t")
                    params_and_dfs.append((param_dict, df.copy()))
                param_str = line.replace(";", ",")
                param_str = param_str[param_str.find("{") : ].replace('=', '"=').replace(', ',', "').replace("=", ":")
                param_str = param_str[:1] + '"' + param_str[1:]
                param_dict = json.loads(param_str)
                # 3. Clear Buffer
                strBuf1 = StringIO("")
        if strBuf1.tell() > 0:
            strBuf1.seek(0)
            df = pd.read_csv(strBuf1, skiprows=[1], delimiter="\t")
            params_and_dfs.append((param_dict, df.copy()))
    return params_and_dfs

# def read_phase_space_frames(path, filenamepattern, columns_per_frame=['z /mm', 'py (norm.)'], t_range=None):
#     '''
#     path: str
#         main path where the exported files of the phase space monitor are located. This is passed to `glog.glob( )` 
#         as `root_dir` parameter to retrieve the matching files.
#     filenamepattern: str
#         file name or pattern passed to `glob.glob()` to find all phase space monitor files.
#     '''

def convert_phase_space_monitor_data_to_hdf5(path, filenamepattern, output_filename, 
                                             columns_per_frame=['z /mm', 'v_y / m/s'], 
                                             t_range=np.linspace(0,0.2, 100, endpoint=False)):
    '''
    Read the ASCII csv files for the PIC phase space monitor, which were exported via the 'Export Elot Data...' functionality.

    path: str
        main path where the exported files of the phase space monitor are located. This is passed to `glog.glob( )` 
        as `root_dir` parameter to retrieve the matching files. The output hdf5 file will be located at the same path.
    filenamepattern: str
        file name or pattern passed to `glob.glob()` to find all phase space monitor files.
    '''
    frames_and_params = []
    # gamma = 293
    for filename in sorted(glob.glob(filenamepattern, root_dir = path)):
        logging.debug(f'reading file {filename} ...')
        temp = read_parametric_CST_ASCII_format(os.path.join(path, filename))
        logging.debug(f'number of frames in read file: {len(temp)}')
        frames_and_params.extend(temp)
    n_frames = len(frames_and_params)
    logging.debug(f'{n_frames} frames in total')
    with h5py.File(os.path.join(path, output_filename), mode='a') as f:
        dset = f.create_dataset('phasespace', (n_frames, *frames_and_params[0][1].shape),
                                dtype=frames_and_params[0][1].dtypes[0])
        for i, frame in enumerate(frames_and_params):
            logging.debug(f'current frame number: {i}')
            n_particles = frame[1].shape[0]
            if n_particles == dset.shape[1]:
                dset[i, :, :] = frame[1].values
            else:
                dset[i, :n_particles, :] = frame[1].values
                dset[i, n_particles:, :] = np.nan

        # f['data'] = frames_and_params[0][1].values
        f['col_labels'] = columns_per_frame
        f['col_labels'].make_scale('phase space coordinates')
        dset.dims[2].attach_scale(f['col_labels'])

        f['frame_numbers'] = np.arange(n_frames)
        f['frame_time_ns'] = t_range
        f['frame_time_ns'].make_scale('time')
        dset.dims[0].attach_scale(f['frame_time_ns'])

        for key, val in frames_and_params[0][0].items():
            dset.attrs[key] = val

def convert_exported_phase_space_monitor_data_to_hdf5(main_path, filenamepattern, output_filename, 
                                                      columns_per_frame=['z /mm', 'v_y / m/s'], 
                                                      t_range=np.linspace(0,0.2, 100, endpoint=False)):
    '''
    Read the ASCII csv files for the PIC phase space monitor, which were exported via Post-processing steps 
    'ASCII export' and 'copy export folder (parametric)'. 

    path: str
        main path where the exported files of the phase space monitor are located. This is passed to `glog.glob( )` 
        as `root_dir` parameter to retrieve the matching files. The output hdf5 file will be located at the same path.
    filenamepattern: str
        file name or pattern passed to `glob.glob()` to find all phase space monitor files.
    '''
    files_to_be_parsed = glob.glob(filenamepattern, root_dir = main_path)
    n_frames = len(files_to_be_parsed)
    n_particles = 0
    frames = []
    for filename in sorted(files_to_be_parsed):
        logging.debug(f'reading file {filename} ...')
        frame_arr = np.loadtxt(os.path.join(main_path, filename), delimiter='\t')
        if n_particles < frame_arr.shape[0]:
            n_particles = frame_arr.shape[0]
            logging.debug(f'particle number updated to {n_particles}')
        # temp = read_parametric_CST_ASCII_format(os.path.join(path, filename))
        frames.append(frame_arr.copy())
    logging.debug(f'{n_frames} frames in total')

    with h5py.File(os.path.join(main_path, output_filename), mode='a') as f:
        dset = f.create_dataset('phasespace', (n_frames, n_particles, frames[1].shape[1] ),
                                dtype=frames[1].dtype)
        for i, frame in enumerate(frames):
            logging.debug(f'current frame number: {i}')
            n_particles_i = frame.shape[0]
            if n_particles_i == dset.shape[1]:
                dset[i, :, :] = frame
            else:
                dset[i, :n_particles_i, :] = frame
                dset[i, n_particles_i:, :] = np.nan

        # f['data'] = frames_and_params[0][1].values
        f['col_labels'] = columns_per_frame
        f['col_labels'].make_scale('phase space coordinates')
        dset.dims[2].attach_scale(f['col_labels'])

        f['frame_numbers'] = np.arange(n_frames)
        f['frame_time_ns'] = t_range
        f['frame_time_ns'].make_scale('time')
        dset.dims[0].attach_scale(f['frame_time_ns'])

        # for key, val in frames_and_params[0][0].items():
        #     dset.attrs[key] = val


    
    


# Old version
def read_CST_tr_phase_space_mon(path, columns_per_frame=["x", "px"]):
    '''
    path: str
        path to the txt file exported from CST
    startingFrame: int
        number of the frame with which the exported file starts
    '''
    filesInDir = np.array(os.listdir(path))
    dirFiles= filesInDir[  np.argsort(  [ float(os.path.splitext(y)[0].split(" ")[1]) for y in filesInDir ] ) ]
    frameDfs = []
    for filename in dirFiles:
        df_temp = pd.read_csv(
            os.path.join(path, filename)
        )
        index = pd.MultiIndex.from_product([[os.path.splitext(filename)[0]], columns_per_frame])
        df_temp.columns = index
        frameDfs.append(df_temp.copy())
#     pd.concat(frameDfs, axis=1)))
    return pd.concat(frameDfs, axis=1)

def convert_astra_distribution_to_cst_pit(path_to_astra_distribution_file, output_filepath=None, set_global_z = None,
                                          x_offset = 0, y_offset = 0):
    """
    Convert the ASTRA distribution file to a .pit file compatible with the CST PIC solver.
    The format of the .pit file is given in the CST Help under the section of the 'Particle Import Interface', 
    also shown in the example

    Parameters
    ----------
    path_to_astra_distribution_file: str
        path to the ASTRA distribution file which is converted.
    output_filepath: str, optional
        path to which the .pit file will be exported. If `None`, which is the default, the same filename as the
        astra file will be used, and it will be stored in the same directory.
    set_global_z: float,
        The ASTRA distribution uses a relative `z` position with respect to the absolute `z` position of a reference 
        particle. The CST distribution uses an absolute z position for all particles. When converting to CST format, 
        a different reference `z` position can be given as `set_global_z`. In case of the *default*, `set_global_z=None`,
        the z postion of the ASTRA reference particle is used. Given in *meter*.
    x_offset, y_offset: float
        Additional offset in `x` and `y` between ASTRA and CST distribution. By default, they are both 0. Given in *meter*.

    Examples
    --------
    The .pit format, to which this function converts to, has the following structure::

        % Use always SI units.
        % The momentum (mom) is equivalent to beta * gamma.
        % The data need not to be chronological ordered.
        %
        % Columns: pos_x  pos_y  pos_z  mom_x  mom_y  mom_z  mass  charge  charge(macro)  time

        1.0e-3   4.0e-3  -1.0e-3   1.0   2.0   1.0   9.11e-31  -1.6e-19   -2.6e-15   0e-6
        2.0e-3   4.0e-3   1.0e-3   1.0   2.0   1.0   9.11e-31  -1.6e-19   -3.9e-15   1e-6
        3.0e-3   2.0e-3   1.0e-3   1.0   2.0   2.0   9.11e-31  -1.6e-19   -3.9e-15   2e-6
        4.0e-3   4.0e-3   5.0e-3   1.0   2.0   1.0   9.11e-31  -1.6e-19   -2.6e-15   3e-6

    For a CST import file in which the bunch shall be translated to the origin of the `z`-axis, independent of the 
    ASTRA reference particle, the conversion is done by:
    
    >>> convert_astra_distribution_to_cst_pit('Ares_twac_wp1p5_full_to_sh.gen.1694.003', 'Ares_twac_wp1p5_full_to_sh.pit',
    ...          set_global_z=0)

    """
    # the import is done here due to the fact that all other functions could not be used if this package were not available
    import sys
    sys.path.append('/Users/maxkellermeier/Nextcloud/dev/packages')
    from astratools.distribution import ASTRADist

    astra_dist = ASTRADist(path_to_astra_distribution_file, compact_names=False)
    if set_global_z is None:
        set_global_z = astra_dist.reference['z /m']

    if output_filepath is None:
        output_filepath = path_to_astra_distribution_file + '.pit'

    t_offset = astra_dist.distr['t /ns'].min()
    if np.all(astra_dist.distr['species'] == 1 ):
        # all particles are electrons
        df = pd.DataFrame({
            'x /m' : astra_dist.distr['x /m'] + x_offset,
            'y /m' : astra_dist.distr['y /m'] + y_offset,
            'z /m' : astra_dist.distr['z /m'] + set_global_z,
            'p_x / (norm.)' : astra_dist.distr['px /eV/c']/(m_e * c**2 / e),
            'p_y / (norm.)' : astra_dist.distr['py /eV/c']/(m_e * c**2 / e),
            'p_z / (norm.)' : (astra_dist.reference['pz /eV/c'] + astra_dist.distr['pz /eV/c']) /(m_e * c**2 / e),
            'm /kg' : np.broadcast_to(m_e, astra_dist.distr.shape[0]),
            'q /C' : np.broadcast_to(e, astra_dist.distr.shape[0]),
            'Q /C' : astra_dist.distr['q /nC'] * 1e-9,
            't /s' : (astra_dist.distr['t /ns'] - t_offset) * 1e-9  #time values in CST must be positive
        })
        df.to_csv(output_filepath, sep=' ', header=False, index=False)
        return df
    else:
        raise NotImplemented

    
