#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warnings
import re
import fnmatch
import psutil

import pandas as pd
import numpy as np
import ants
from aicsimageio import AICSImage
from glob import glob
from tqdm import tqdm
from readlif.reader import LifFile

import mbflim.shared as sha


def has_handle(fpath):
    for proc in psutil.process_iter():
        try:
            for item in proc.open_files():
                if fpath == item.path:
                    return True
        except Exception:
            pass

    return False


def lif_to_ants(lif_data, img_ix=0):


    lif_stack = lif_data.get_image(img_ix)
    scale = np.array(lif_stack.scale)
    dims = \
        [lif_stack.dims.x, lif_stack.dims.y, lif_stack.dims.z, lif_stack.channels];
    # real_dims = np.array(dims[:3]) / scale[:3]

    # init
    stack = np.empty(dims)

    # fill array
    for c in range(0, dims[3]):
        for i in range(0, lif_stack.dims_n[3]):
            img = lif_stack.get_plane(
                display_dims = (1, 2), c = c, requested_dims = {3: i}
                )
            img = np.transpose(np.array(img))
            stack[:, :, i, c] = img

    # reorganize channels
    stack = stack[:, :, :, [1, 0]]

    img = ants.from_numpy(stack, has_components=True)
    img.set_spacing(tuple(1.0 / scale[:3]))

    return img


def import_dataset(dataset_paths, output_dir, filter_out=['']):

    os.makedirs(output_dir, exist_ok=True)

    elem_path_V = []
    elem_name_V = []
    elem_type_V = []
    elem_hemisphere_V = []
    elem_fly_V = []
    elem_subject_V = []

    lif_filepaths = []
    lif_file_nb = 0

    for dataset_path in dataset_paths:

        filepath_pattern = dataset_path +'*.lif'
        lif_filepaths += glob(filepath_pattern)

    lif_file_nb = len(lif_filepaths)
    file_timer = sha.Timer(lif_file_nb, 'File')

    for i_f, filepath in enumerate(lif_filepaths):

        lif_reader = LifFile(filepath)
        img_nb = len(lif_reader.image_list)
        img_timer = sha.Timer(img_nb, 'Image')

        for i_i in range(img_nb):

            lif_img = lif_reader.get_image(i_i)
            name = lif_img.name

            if not any(fnmatch.fnmatch(name, pattern) for pattern in filter_out):

                name = name.replace('P', 'ApBp')
                name = name.replace('-', '_')

                elem_name_V.append(name)
                local_import_path = name + '.nii.gz'
                elem_path_V.append(local_import_path)
                print(name)

                info_regex = re.compile(r'(\w+)_Fly(\d+a?)_(\w)')
                match = info_regex.search(name)

                ty = match.group(1)
                fly = match.group(2)
                subject = ty + '_Fly' + fly

                elem_subject_V.append(subject)
                elem_type_V.append(ty)
                elem_fly_V.append(fly)
                elem_hemisphere_V.append(match.group(3))

                ants_img = lif_to_ants(lif_reader, i_i)
                full_import_path = output_dir + local_import_path
                ants.image_write(ants_img, full_import_path)

            print(img_timer.remains(i_i + 1))

        print(file_timer.remains(i_f + 1))

    set_info = pd.DataFrame(
        {
         "name": elem_name_V,
         "path": elem_path_V,
         "subject": elem_subject_V,
         "type": elem_type_V,
         "fly": elem_fly_V,
         "hemisphere": elem_hemisphere_V,
         })

    set_info.to_csv(output_dir + 'files.csv')

    return set_info


def import_irf(irf_fpath, bin_nb, time_step):

    # read IRF data
    irf_df = pd.read_csv(irf_fpath, skiprows=1, delimiter='\t')
    file_irf_times = irf_df.iloc[:, 2] * 1e-9
    file_irf_values = irf_df.iloc[:, 3]

    # pad IRF
    irf_values = np.zeros(bin_nb)
    irf_times = time_step * np.arange(bin_nb)

    for t, v in zip(file_irf_times, file_irf_values):
        if not np.isnan(t):
            irf_values[np.argmin(np.abs(irf_times - t))] = v

    irf_data = {
        'irf': irf_values,
        'times': irf_times,
        }

    return irf_data


# Spacing is required because I cannot find information about the voxel size in
# the z dimension in the bin files (flim_img.physical_pixel_sizes.Z is None)
def import_bin_files(bin_fpaths, spacing=None):

    flim_slice_nb = len(bin_fpaths)

    # read all FLIM z-slices
    for i_z, bin_fpath in enumerate(bin_fpaths):

        # tqdm.write(bin_fpath)

        flim_img = AICSImage(bin_fpath)

        # suppress annoying warning
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=FutureWarning)
            data = np.squeeze(flim_img.data) # squeeze channel dimension

        data = np.transpose(data, (2, 1, 0)) # reorder dimensions

        if i_z == 0:

            metadata = flim_img.metadata

            try:
                time_annotations = dict(
                    metadata.structured_annotations[0].value[0][0].items()
                    )
            except TypeError:
                v = metadata.structured_annotations[0].value
                time_annotations = v.any_elements[0].children[0].attributes

            time_step = float(time_annotations['Step']) * 1e-12
            time_end = float(time_annotations['End']) * 1e-12

            flim_array = np.zeros(
                data.shape[0:2] + (flim_slice_nb,) + (data.shape[2],),
                dtype=np.uint32
                )

        flim_array[:, :, i_z, :] = data

    # For calibration files, no spacing in z, so information in bin files is
    # enough
    if spacing is None:
        spacing = (
            1 / flim_img.physical_pixel_sizes.X,
            1 / flim_img.physical_pixel_sizes.Y,
            1.0,
            )

    # create FLIM ANTS object
    flim_ants_img = ants.from_numpy(
        flim_array,
        spacing=spacing + (time_step,),
        has_components=False,
        )

    return flim_ants_img, time_step, time_end#, metadata


def check_zstack_coherence(ants_img, bin_fpaths, name):

    sp_shape = ants_img.shape

    coherence_bool = (len(bin_fpaths) == sp_shape[2])

    sp_shape = ants_img.shape

    if not coherence_bool:
        tqdm.write(
            'X Incoherent z-stack for ' + name
            + ': FLIM = ' + str(len(bin_fpaths))
            + ' / SP = ' + str(sp_shape[2])
            )

    return coherence_bool


def make_zstack_coherent(ants_img, bin_fpaths):

    sp_shape = ants_img.shape

    if len(bin_fpaths) > sp_shape[2]:
        bin_fpaths = bin_fpaths[:sp_shape[2]]
        msg = 'Dropping excess FLIM slices'
    else:
        msg = 'No implemented solution'

    tqdm.write('> ' + msg)

    return ants_img, bin_fpaths
