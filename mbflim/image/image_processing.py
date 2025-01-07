#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 17:05:00 2022

@author: proussel
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
import tempfile
import re
import fnmatch
import io
from itertools import repeat
import multiprocessing
from abc import ABC

import itk
from readlif.reader import LifFile
import ants
from ants.internal import get_lib_fn
import skimage
from skimage.util import view_as_windows
import scipy
from scipy import ndimage as ndi
from scipy.stats import median_abs_deviation as med_abs_dev
from scipy.stats import norm
import nipype.interfaces.ants as nipype_ants
import nibabel as nib
from tqdm import tqdm


import mbflim.files_io as fio


#%%

# basic image processing function
def read_process_save(in_path, out_path, process_func, *args, recompute=True, **kwargs):

    # print('\n' + in_path)
    if not recompute and os.path.exists(out_path):
        return

    img = read_img(in_path)
    img = process_func(img, *args, **kwargs)
    ants.image_write(img, out_path)

    return

# this function returns ANTS images or numpy arrays depending on what is provided as input
def normalize_map(avg_map_imgs, neuron_nbs, kc_density_threshold='li'):

    avg_stacks = []

    if isinstance(avg_map_imgs[0], ants.ANTsImage) or isinstance(avg_map_imgs[0], str):
        for img in avg_map_imgs:

                img = read_img(img)
                stack = img.numpy()
                avg_stacks.append(stack)

        ref_img = img

    # if numpy array (TODO: test with elif)
    else:
        avg_stacks = avg_map_imgs
        ref_img = None

    kc_density_stack = np.zeros_like(avg_stacks[0])
    type_density_stacks = []

    # avg_stacks is transformed into density_stacks
    # ! not copied for performance !
    for i_t in range(3):
        stack = avg_stacks[i_t]

        # compute density
        stack = stack / np.sum(stack) * neuron_nbs[i_t]
        type_density_stacks.append(stack)

        kc_density_stack += stack

    mask = remove_small_objects_stack(kc_density_stack, threshold=kc_density_threshold)
    kc_density_stack[np.logical_not(mask)] = 0.0

    type_local_proba_stacks = []
    # type_global_proba_stacks = []

    for i_t in range(3):
        stack = type_density_stacks[i_t].copy()

        # compute local proba
         # apply mask
        stack[mask == 0] = 0.0
         # compute probability within mask
        stack[mask] = stack[mask] / kc_density_stack[mask]

        type_local_proba_stacks.append(stack) # copy for global proba !

        # compute global proba
        # stack = stack / np.sum(stack) * neuron_nbs[i_t]
        # type_global_proba_stacks.append(stack)

    if ref_img is not None:
        # create KC density image
        kc_density_map = ref_img.new_image_like(kc_density_stack)

        type_density_maps = []
        type_local_proba_maps = []
        # type_global_proba_maps = []

        for i_t in range(3):
            type_density_maps.append(ref_img.new_image_like(type_density_stacks[i_t]))
            type_local_proba_maps.append(ref_img.new_image_like(type_local_proba_stacks[i_t]))
            # type_global_proba_maps.append(ref_img.new_image_like(type_global_proba_stacks[i_t]))

    else:
        kc_density_map = kc_density_stack
        type_density_maps = type_density_stacks
        type_local_proba_maps = type_local_proba_stacks
        # type_global_proba_maps = type_global_proba_stacks

    return kc_density_map, type_density_maps, type_local_proba_maps #, type_global_proba_maps



def split_write(in_fpath, tmp_fpath, component):

    if component is not None and get_channel_nb(in_fpath) > 1:
        img = read_img(in_fpath)
        imgc = ants.split_channels(img)[component]
        ants.image_write(imgc, tmp_fpath)
    else:
        shutil.copy(in_fpath, tmp_fpath)



def read_img(path_or_img):

    if isinstance(path_or_img, str):
        img = ants.image_read(path_or_img)
    elif isinstance(path_or_img, ants.ANTsImage):
        img = path_or_img
    else:
        raise TypeError("Argument 'path_or_img' should be of type str or "
                        + "ants.ANTsImage, not " + str(type(path_or_img)) + ".")
    return img


def write_img(img, path):

    ants.image_write(img, path)



def select_img(folder_path, index_or_name):

    set_info = pd.read_csv(folder_path + 'files.csv', index_col=0)

    if isinstance(index_or_name, str):
        for i, name in enumerate(set_info['name']):
            b = fnmatch.fnmatch(set_info['name'][i], index_or_name)
            if b: break

        if b:
            index = i
        else:
            raise Warning("No element matching '" + str(index_or_name) + "'.")

    else:
        index = index_or_name

    path = folder_path + set_info.loc[index, 'path']

    return path



def get_info(dpath):

    set_info = pd.read_csv(dpath + 'files.csv', index_col=0)

    return set_info



def get_full_info(dpath):

    set_info = get_info(dpath)

    fpaths = [dpath + p for p in set_info['path']]

    header_info = []

    for fpath in fpaths:
        header_info.append(pd.DataFrame({
            'shape': [get_dimensions(fpath)],
            'channel_nb': [get_channel_nb(fpath)],
            'spacing': [get_spacing(fpath)],
            }))

    header_info = pd.concat(header_info, ignore_index=True)

    full_info = set_info.join(header_info)

    return full_info



def format_img(in_path, out_path, extension='.nii'):

    # read .lif file
    lif_data = LifFile(in_path)

    # format
    ants_image = fio.lif_to_ants(lif_data)

    # store
    ants.image_write(ants_image, out_path)



def mirror_img(img, axis=0):

    img = read_img(img)

    stack = img.numpy()
    stack = np.flip(stack, axis=axis)

    img = img.new_image_like(stack)

    return img



def remove_small_objects_img(
        img,
        min_width=None,
        threshold=0.0,
        component=0,
        mask_channels=True,
        erosion_dilation=0,
        smoothing=None,
        automask=False,
        fill_holes=True
        ):

    img = read_img(img)
    spacing = np.array(img.spacing)

    if img.has_components:
        channels = ants.split_channels(img)
        ref_img = channels[component]
    else:
        channels = [img]
        ref_img = img

    stack = ref_img.numpy()

    if automask:
        threshold_mask = stack > 0.0
    else:
        threshold_mask = None

    mask = remove_small_objects_stack(
        stack,
        spacing=spacing,
        min_width=min_width,
        threshold=threshold,
        erosion_dilation=erosion_dilation,
        smoothing=smoothing,
        threshold_mask=threshold_mask,
        fill_holes=fill_holes,
        )

    # apply to all channels
    if mask_channels:
        channels = [ref_img.new_image_like(mask * c.numpy())
                    for c in channels]
    # apply to selected component only
    else:
        channels[component] = ref_img.new_image_like(mask * stack)

    return merge_channels(channels)



def remove_small_objects_stack(
        stack,
        spacing,
        min_width=None,
        threshold=0.0,
        erosion_dilation=0,
        smoothing=None,
        threshold_mask=None,
        fill_holes=True,
        automask=False,
        ):

    if automask:
        threshold_mask = stack > 0

    if threshold_mask is None:
        threshold_mask = np.ones_like(stack, dtype='bool')

    if smoothing is not None:
        smoothing = np.array(smoothing).astype('float32') / spacing
        stack = stack.copy()
        stack = skimage.filters.gaussian(stack, sigma=smoothing, mode='mirror', truncate=4.0)

    if threshold == 'otsu':
        threshold = skimage.filters.threshold_otsu(stack[threshold_mask].ravel())
    elif threshold == 'multiotsu':
        threshold = skimage.filters.threshold_multiotsu(stack[threshold_mask].ravel(), classes=3)[1]
    elif threshold == 'isodata':
        threshold = skimage.filters.threshold_isodata(stack[threshold_mask].ravel())
    elif threshold == 'li':
        sel_stack = stack[threshold_mask]
        tolerance = (sel_stack.max() - sel_stack.min()) / 1000
        threshold = skimage.filters.threshold_li(
            image=sel_stack,
            tolerance=tolerance,
            initial_guess=skimage.filters.threshold_otsu
            )
    elif threshold == 'it':
        threshold = skimage.filters.threshold_otsu(stack[threshold_mask].ravel())
        threshold = skimage.filters.threshold_otsu(stack[stack > threshold].ravel())
    elif np.isscalar(threshold):
        pass
    else:
        raise ValueError('Unknown thresolding method.')

    mask = (stack > threshold)

    if erosion_dilation < 0:
        mask = ndi.binary_erosion(mask, iterations=-erosion_dilation)
    elif erosion_dilation > 0:
        mask = ndi.binary_dilation(mask, iterations=erosion_dilation)

    # import tools.visualization as vis
    # vis.plot_maxproj(ants.from_numpy(mask.astype(float), spacing=tuple(spacing)), norm='minmax')

    # same procedure as in skimage.morphology.remove_small_objects
    #  array defining how pixels are considered connected
    footprint = ndi.generate_binary_structure(mask.ndim, connectivity=1)
    #  prepare output for labeling
    ccs = np.zeros_like(mask, dtype=np.int32)
    #  labeling
    ndi.label(mask, footprint, output=ccs)
    #  measure objects size
    component_sizes = np.bincount(ccs.ravel())

    vox_volume = np.prod(spacing)

    if min_width is None:
        # select biggest object (exclude label 0 that is background)
        min_vol = np.max(component_sizes[1:]) * vox_volume
    else:
        min_vol = min_width**3

    too_small = component_sizes < np.floor((min_vol / vox_volume))
    too_small_mask = too_small[ccs]
    mask[too_small_mask] = 0


    # fill holes in mask
    if fill_holes:
        # max hole size equal to size of  biggest object
        # (exclude label 0 that is background)
        max_hole_size = np.max(component_sizes[1:]) * vox_volume

        mask = skimage.morphology.remove_small_holes(
            mask,
            area_threshold=max_hole_size,
            )

    return mask



def smartcrop_img(
    img,
    margins,
    in_pixel=False,
    reference='center_of_mass',
    channel=None,
    extend=True,
    ):

    img = read_img(img)
    stack = img.numpy()

    if img.has_components and channel is not None:
        ref_img = ants.split_channels(img)[channel]
    else:
        ref_img = img

    img_shape = np.array(stack.shape)
    img_spacing = np.array(img.spacing)[:3]
    margins_px = np.round(margins / np.transpose(np.vstack((img_spacing, img_spacing)))) if not in_pixel else margins
    margins_px = margins_px.astype('int')

    if isinstance(reference, np.ndarray) or reference in ['center', 'center_of_mass', 'corner']:

        crop_lims = np.zeros((3, 2), dtype='int')

        if isinstance(reference, np.ndarray):
            c_px = reference
            margin_dirs = [-1, 1]
        elif reference == 'center':
            c_px = np.ceil(img_shape / 2).astype(int)
            margin_dirs = [-1, 1]
        elif reference == 'center_of_mass':
            c_px = np.round(ants.get_center_of_mass(ref_img) / img_spacing).astype(int)
            margin_dirs = [-1, 1]
        elif reference == 'corner':
            c_px = np.zeros(3, dtype=int)
            margin_dirs = [1, 1]
        else:
            raise ValueError('Unknown reference.')

        for i in range(3):
            for j in range(2):
                crop_lims[i, j] = c_px[i] + margin_dirs[j] * margins_px[i, j]

        for i in range(3):
            lim = int(img_shape[i] - crop_lims[i, 1])
            if lim > 0:
                stack = np.take(stack, range(crop_lims[i, 1]), axis=i)
            elif lim < 0 and extend:
                block_shape = np.array(stack.shape)
                block_shape[i] = -lim
                stack = np.concatenate([stack, np.zeros(block_shape)], axis=i)

        for i in range(3):
            lim = int(crop_lims[i, 0])
            if lim > 0:
                stack = np.take(stack, range(lim, stack.shape[i]), axis=i)
            elif lim < 0 and extend:
                block_shape = np.array(stack.shape)
                block_shape[i] = -lim
                stack = np.concatenate([np.zeros(block_shape), stack], axis=i)

    elif reference == 'border':
        crop_lims = np.zeros((3, 2), dtype='int')

        for i in range(3):
            crop_lims[i, 0] = margins_px[i, 0]
            crop_lims[i, 1] = img_shape[i] - margins_px[i, 1]

        for i in range(3):
            stack = np.take(stack, range(int(crop_lims[i, 0]), int(crop_lims[i, 1])), axis=i)

    img_crop = ants.from_numpy(
        stack,
        has_components=img.has_components,
        spacing=img.spacing,
        )

    return img_crop


def resample_img(img, spacing, use_voxels, interp_type, smoothing_sigma=None):

    img = read_img(img)

    # decompose the image to create one image per component
    if img.has_components:
        img_comps = ants.split_channels(img)
    else:
        img_comps = [img]

    resampled_img_comps = []

    # apply resampling to each component separately
    for comp in img_comps:
        if smoothing_sigma == 'auto':
            factor = np.array(img.spacing) / np.array(spacing)
            smoothing_sigma = np.array([
                f * 1.5 * s if f < 1 else 0.0 for (f, s) in zip(factor, img.spacing)
                ])

        if any(smoothing_sigma):
            comp = ants.utils.smooth_image(
                comp,
                smoothing_sigma,
                sigma_in_physical_coordinates=False,
                FWHM=False,
                max_kernel_width=4*smoothing_sigma
                )

        resampled_img_comps.append(
            ants.resample_image(comp, spacing, use_voxels, interp_type))

    # fuse the resampled components back together
    return merge_channels(resampled_img_comps)

# from tqdm import tqdm
# import time
def match_histogram_img(img, ref_or_filt, component=None, num_bins=20, num_points=10):

    # load and split image
    img = read_img(img)
    if component is not None:
        img_chans = split_channels(img)
        img = img_chans[component]

    # convert to ITK image
    img_ar = img.numpy()
    zero_lgcs = img_ar == 0.0

    img_vals = img_ar[~zero_lgcs]
    img_itk = itk.GetImageFromArray(img_vals)


    img_type = type(img_itk)

    if isinstance(ref_or_filt, ants.ANTsImage) or isinstance(ref_or_filt, str):
        # load and split reference
        ref_img = read_img(ref_or_filt)
        if component is not None:
            ref_img = split_channels(ref_img)[component]

        # convert to ITK image
        ref_img_ar = ref_img.numpy()
        ref_img_itk = itk.GetImageFromArray(ref_img_ar)

        # prepare filter
        filt = itk.HistogramMatchingImageFilter[img_type, img_type].New()
        filt.SetReferenceImage(ref_img_itk)
        filt.ThresholdAtMeanIntensityOn()
        filt.SetNumberOfHistogramLevels(num_bins)
        filt.SetNumberOfMatchPoints(num_points)


    elif isinstance(
            ref_or_filt,
            itk.itkHistogramMatchingImageFilterPython.
            itkHistogramMatchingImageFilterIF3IF3):

        filt = ref_or_filt

    # set input and update filter
    filt.SetInput(img_itk)
    filt.Update()

    # apply filter and reorganize components
    img_itk = filt.GetOutput()

    img_vals = itk.array_from_image(img_itk)
    img_vals = np.swapaxes(img_vals, 0, 2)
    img_ar[~zero_lgcs] = img_vals

    # create output image
    out_img = img.new_image_like(img_ar)

    if component is not None:
        img_chans[component] = out_img
        out_img = merge_channels(img_chans)

    return out_img





def align_img_nipype(img_path, ref_path, params):

    params['fixed_image'] = ref_path
    params['moving_image'] = img_path

    reg = nipype_ants.Registration()

    reg.terminal_output = 'allatonce'

    for key in params.keys():
        setattr(reg.inputs, key, params[key])

    # print(reg.cmdline)
    # sys.exit()

    reg_res = reg.run()

    return reg_res



def transform_img(img, transforms, ref_img=None, interpolator='linear'):

    if ref_img is None:
        ref_img = img

    img = read_img(img)
    ref_img = read_img(img)

    if ref_img.has_components:
        ref_img = split_channels(ref_img)[0]

    if img.has_components:
        img_comps = split_channels(img)
    else:
        img_comps = [img]

    tf_img_comps = []

    for img_comp in img_comps:
        tf_img_comps.append(ants.apply_transforms(
            ref_img,
            img_comp,
            transforms,
            interpolator=interpolator,
            ))

    if len(tf_img_comps) > 1:
        tf_img = merge_channels(tf_img_comps)
    else:
        tf_img = tf_img_comps[0]

    return tf_img



def compute_similarity_metrics(img1, img2, parameters, component=0, mask_threshold=None):

    if not isinstance(parameters, list):
        parameters = [parameters]

    # create temporary folder
    func_tmp_dir_obj = tempfile.TemporaryDirectory(prefix=os.environ["TMPDIR"])
    func_tmp_dir = func_tmp_dir_obj.name + '/'

    img_paths = []
    mask_imgs = []

    need_files = False
    for param_set in parameters:
        if param_set['metric'] in ['CC', 'Mattes']:
            need_files = True
            break

    imgs = [img1, img2]
    for i, img in enumerate(imgs):
        # create single-channel image if necessary
        if component is not None and get_channel_nb(img) > 1:
            img = read_img(img)
            img = ants.split_channels(img)[component]
            imgs[i] = img

        # write image if necessary
        if need_files:
            if isinstance(img, ants.ANTsImage):
                img_path = tempfile.NamedTemporaryFile(
                    prefix=func_tmp_dir,
                    suffix='.nii',
                    delete=False).name
                ants.image_write(img, img_path)
            else:
                img_path = img

            img_paths.append(img_path)

        if mask_threshold is not None:
            img = read_img(img)
            mask_imgs.append(ants.get_mask(img, low_thresh=mask_threshold, high_thresh=None, cleanup=0))

    # take union of the two masks and write it
    if mask_threshold is not None:
        mask = np.logical_or(mask_imgs[0].numpy(), mask_imgs[1].numpy())
        mask_img = mask_imgs[0].new_image_like(mask.astype(float))

        if need_files:
            mask_img_path = func_tmp_dir + 'mask.nii'
            ants.image_write(mask_img, mask_img_path)
    else:
        mask_img = None

    metric_values = []

    for param_set in parameters:

        if param_set['metric'] in ['CC', 'Mattes', 'GC']:

            sim = nipype_ants.MeasureImageSimilarity()

            # avoid console output ('none' not working)
            sim.terminal_output = 'allatonce'
            sim.inputs.dimension = 3
            sim.inputs.metric = param_set['metric']
            sim.inputs.fixed_image = img_paths[0]
            sim.inputs.moving_image = img_paths[1]
            sim.inputs.metric_weight = 1.0 # not used according to nipype doc
            sim.inputs.radius_or_number_of_bins = \
                param_set['radius_or_number_of_bins']
            sim.inputs.sampling_strategy = \
                param_set['sampling_strategy']
            sim.inputs.sampling_percentage = \
                param_set['sampling_percentage']

            if mask_threshold is not None:
                sim.inputs.fixed_image_mask = mask_img_path
                sim.inputs.moving_image_mask = mask_img_path

            res = sim.run()
            sim_val = res.outputs.get()['similarity']

        elif param_set['metric'] in ['MeanSquares', 'GC']:
            for i, img in enumerate(imgs):
                imgs[i] = read_img(img)

            sim_val = ants.image_similarity(
                imgs[0], imgs[1],
                metric_type=param_set['metric'],
                fixed_mask=mask_img,
                moving_mask=mask_img,
                sampling_strategy=param_set['sampling_strategy'],
                sampling_percentage=param_set['sampling_percentage'])

        elif param_set['metric'] in ['densityRMSE']:
            for i, img in enumerate(imgs):
                img = read_img(img)
                img = img.numpy()
                img = img / np.linalg.norm(img.ravel(), ord=1)
                imgs[i] = img

            sim_val = float(np.linalg.norm((imgs[0] - imgs[1]).ravel(), ord=2))

        else:
            raise ValueError("Unknown metric '" + param_set['metric'] + "'.")

        metric_values.append(sim_val)

    # remove temporary folder
    func_tmp_dir_obj.cleanup()

    return metric_values



def get_channel_nb(img_or_path):

    if isinstance(img_or_path, str):
        return ants.image_header_info(img_or_path)['nComponents']

    elif isinstance(img_or_path, ants.ANTsImage):
        return img_or_path.components

    else:
        raise TypeError("'img_or_path' must be of type 'str' or 'ANTsImage'. "
                        "Type " + str(type(img_or_path)) + " was provided.")

        return



def get_dimensions(img_or_path):

    if isinstance(img_or_path, str):
        dimensions = ants.image_header_info(img_or_path)['dimensions']
        dimensions = tuple([int(d) for d in dimensions])

        return dimensions

    elif isinstance(img_or_path, ants.ANTsImage):
        return img_or_path.shape

    else:
        raise TypeError("'img_or_path' must be of type 'str' or 'ANTsImage'. "
                        "Type " + str(type(img_or_path)) + " was provided.")

        return



def get_spacing(img_or_path):

    if isinstance(img_or_path, str):
        # use lower level function to avoid the rounding in 'image_header_info'
        libfn = get_lib_fn("antsImageHeaderInfo")
        retval = libfn(img_or_path)
        return retval['spacing']

    elif isinstance(img_or_path, ants.ANTsImage):
        return img_or_path.spacing

    else:
        raise TypeError("'img_or_path' must be of type 'str' or 'ANTsImage'. "
                        "Type " + str(type(img_or_path)) + " was provided.")

        return



def parse_ants_log(log_filepath):

    type_rx = re.compile(r'Running (\w+) registration')
    new_rx = re.compile(r'DIAGNOSTIC|XXDIAGNOSTIC')
    end_rx = re.compile(r'Elapsed')

    registrations = []

    with open(log_filepath) as file_object:

        line = file_object.readline()

        while line:

            # look for registration type
            match = re.search(type_rx, line)

            if match:

                    registration = {'type': match.group(1),
                                    'stages': []}
                    level = 1

                    # skip lines and store column names
                    for i in range(2):
                        line = file_object.readline()
                    col_names = line.split(',')
                    col_names = [name.strip('\n') for name in col_names]
                    line = file_object.readline()

                    match = []
                    table_text = []
                    registration_end = False

                    while not registration_end:

                        stage = {'level': level,
                                  'data': None}
                        table_text = ''
                        stage_end = False

                        while not stage_end:

                            # look for new stage or registration end
                            new_match = re.match(new_rx, line)
                            end_match = re.search(end_rx, line)

                            if not (new_match or end_match):

                                # print(line)

                                table_text = table_text + line
                                # line_nb += 1
                                line = file_object.readline()

                            else:

                                stage['data'] = pd.read_csv(
                                    io.StringIO(table_text),
                                    sep=",", names=col_names,
                                    index_col=False,
                                    usecols = [i for i in range(6)])
                                registration['stages'].append(stage)
                                level += 1
                                stage_end = True
                                # print('Line nb:', line_nb)

                                # if registration end
                                if end_match:
                                    registrations.append(registration)
                                    registration_end = True

                        line = file_object.readline()

            line = file_object.readline()

    return registrations


# def evaluate_similarity_set(input_path, metrics, component=0, comparison_nb=None):

#     set_info = pd.read_csv(input_path + 'files.csv', index_col=0)
#     img_nb = len(set_info.index)

#     # create temporary folder
#     func_tmp_dir_obj = tempfile.TemporaryDirectory(prefix=os.environ["TMPDIR"])
#     func_tmp_dir = func_tmp_dir_obj.name + '/'

#     # create single-channel images if necessary (otherwise all channels in the
#     #  file are used to compute the similarity)
#     img_paths = input_path + set_info['path'].values
#     imgc_paths = []

#     for path in img_paths:
#         if component is not None:
#             if get_channel_nb(path) > 1:
#                 img = read_img(path)
#                 imgc_path = tempfile.NamedTemporaryFile(
#                     prefix=func_tmp_dir,
#                     suffix='.nii',
#                     delete=False).name
#                 img = ants.split_channels(img)[component]
#                 ants.image_write(img, imgc_path)
#                 imgc_paths.append(imgc_path)

#         else:
#             imgc_paths.append(path)

#     if not isinstance(metrics, list):
#         metrics = [metrics]
#     metric_nb = len(metrics)

#     values = np.empty((img_nb, img_nb, metric_nb), 'float')


#     # compute all pairwise comparisons
#     if comparison_nb is None:
#         values[:] = 0
#         timer = Timer(img_nb * (img_nb - 1) / 2, 'Similarity')
#         k = 1

#         for i in range(img_nb):
#             for j in range(i):
#                 values[i, j, :] = compute_similarity_metrics(imgc_paths[i],
#                                                               imgc_paths[j],
#                                                               metrics)
#                 print(timer.remains(k))
#                 k += 1

#         for i_m in range(metric_nb):
#             values[:, :, i_m] += np.transpose(values[:, :, i_m])
#             for i in range(img_nb):
#                 values[i, i, i_m] = np.NaN

#     # compute only 'comparison_nb' comparisons per image
#     else:
#         values[:] = np.NaN
#         comparison_nb = min(img_nb - 1, comparison_nb)
#         timer = Timer(img_nb * comparison_nb, 'Similarity')

#         comp_count = np.zeros((img_nb, 1))


#         for i in range(img_nb):
#             rand_js = np.random.permutation(img_nb)
#             j = 0

#             while comp_count[i] < comparison_nb:
#                 rand_j = rand_js[j]
#                 j += 1

#                 if (np.isnan(values[i, rand_j, 0])
#                     and i != rand_j):

#                     if not np.isnan(values[rand_j, i, 0]):
#                         values[i, rand_j, :] = values[rand_j, i, 0]
#                     else:
#                         values[i, rand_j, :] = \
#                             compute_similarity_metrics(imgc_paths[i],
#                                                        imgc_paths[rand_j],
#                                                        metrics)
#                     comp_count[i] += 1


#                     print(timer.remains(np.sum(comp_count)))

#     func_tmp_dir_obj.cleanup()

#     return values



def avg_imgs_nipype(input_paths, output_path):

    # create output image
    #  this prevents some crash happening when using a file generated with
    #  tempfile.NamedTemporaryFile
    # img0 = read_img(input_paths[0])
    # stack = img0.numpy()
    # stack = np.zeros_like(stack)
    # img0 = img0.new_image_like(stack)
    # ants.image_write(img0, output_path)
    # shutil.copyfile(input_paths[0], output_path)

    avg = nipype_ants.AverageImages()
    avg.terminal_output = 'allatonce'
    avg.inputs.dimension = 3
    avg.inputs.output_average_image = output_path
    avg.inputs.normalize = False
    avg.inputs.images = input_paths
    avg.run()



def compute_mcc(confusion_mat):

    c = np.sum(np.diag(confusion_mat))
    s = confusion_mat.sum()
    p = confusion_mat.sum(axis=0)
    t = confusion_mat.sum(axis=1)
    MCC = (
        (c * s - np.dot(t, p))
        / np.sqrt(s**2 - np.dot(p, p))
        / np.sqrt(s**2 - np.dot(t, t))
        )

    return MCC


def define_outliers(outlier_dpath, mad_threshold):

    if isinstance(outlier_dpath, str):
        info_path = outlier_dpath + 'files.csv'
        set_info = pd.read_csv(info_path, index_col=0)
    else:
        set_info = outlier_dpath

    # json_path = outlier_dpath + 'outlier_data.json'

    # with open(json_path, 'r') as json_file:
    #     json_data = json.load(json_file)


    outlierness_vals = set_info['outlierness'].to_numpy()

    # tf
    # outlierness_vals = 1 + outlierness_vals
    # sig = np.std(outlierness_vals)
    # mean = np.mean(outlierness_vals)
    # from scipy import special
    # outlierness_vals = np.maximum(0, special.erf((outlierness_vals - mean) / sig / np.sqrt(2)))

    # img_dists_L = json_data['img_dists_L']
    # for i, img_dists in enumerate(img_dists_L):
    #     outlierness_vals[i] = np.mean(img_dists)


    names = set_info['name'].to_numpy()

    med = np.median(outlierness_vals)
    mad = med_abs_dev(outlierness_vals)
    # estimate outlier-free standard deviation from MAD
    std = 1 / norm.ppf(0.75) * mad
    threshold = med + mad_threshold * std
    # threshold = mad_threshold

    lower_bools = outlierness_vals >= threshold
    lower_names = names[lower_bools]

    return lower_names, threshold, lower_bools, outlierness_vals



def mask_img(img, mask):

    if isinstance(mask, ants.ANTsImage):
        mask = mask.numpy()

    mask = mask.astype('bool')

    stack = img.numpy()

    stack[np.logical_not(mask), :] = 0.0

    return img.new_image_like(stack)



def split_channels(ants_img):

    ants_img = read_img(ants_img)

    if get_channel_nb(ants_img) > 1:
        return ants.split_channels(ants_img)

    else:
        return [ants_img]



def merge_channels(ants_imgs):

    channels = []

    for img in ants_imgs:
        img = read_img(img)
        channels += split_channels(img)

    if len(ants_imgs) > 1:
        merged_img = ants.merge_channels(channels)

    else:
        merged_img = ants_imgs[0]

    return merged_img



def avg_imgs(
        img_paths,
        component=None,
        mode='mean',
        split_nb=1,
        process_nb=1,
        nz_mask=False,
        waitbar=True,
        ):

    img_nb = len(img_paths)

    if mode == 'mean':
        avg_func = np.nanmean
    elif mode == 'median':
        avg_func = np.nanmedian
    elif mode == 'std':
        avg_func = np.nanstd

    dims = get_dimensions(img_paths[0])
    spacing = get_spacing(img_paths[0])
    channel_nb = get_channel_nb(img_paths[0])
    # shape = read_img(img_paths[0]).numpy().shape
    # if len(shape) <= 3:
    #     channel_nb = 1
    # else:
    #     channel_nb = shape[-1]

    # print(img_paths[0], read_img(img_paths[0]).numpy().shape)
    # return

    # channels = np.arange(channel_nb)



    if component is not None:
        if component >= channel_nb:
            raise ValueError(
                'Channel index (' + str(component) \
                + ') should be inferior to channel number (' \
                + str(channel_nb) + ').')

        # channels = [component]
        channel_nb = 1

    # print(channel_nb)
    # sys.exit()

    img_proxies = []

    for path in img_paths:
        img_proxies.append(nib.load(path))

    # splits all images along the first dimensions
    split_nb = min(split_nb, dims[0])

    split_x_ixs = np.array_split(np.arange(dims[0]), split_nb)

    yzc_dims = dims[1:3] + (channel_nb,)

    input_tuples = zip(
        repeat(img_proxies),
        repeat(yzc_dims),
        split_x_ixs,
        repeat(component),
        repeat(nz_mask),
        repeat(avg_func),
        )

    actual_process_nb = min(split_nb, process_nb)

    if actual_process_nb > 1:
        with multiprocessing.get_context('fork').Pool(actual_process_nb) \
            as pool:
            slices = list(tqdm(
                pool.istarmap(_avg_slices, input_tuples),
                total=split_nb,
                desc='Average ' + str(img_nb) + ' images',
                file=sys.stdout,
                smoothing=0,
                disable=not waitbar,
                ))
    else:
        slices = []
        for input_tuple in input_tuples:
            slices.append(_avg_slices(*input_tuple))

    # concatenate and remove singleton channel dimension
    avg_data = np.concatenate(slices, axis=0)

    if channel_nb == 1:
        avg_data = np.squeeze(avg_data, axis=3)

    has_components = channel_nb > 1

    avg_img = ants.from_numpy(
        avg_data,
        spacing=spacing,
        has_components=has_components
        )

    return avg_img


# helper function for avg_imgs
def _avg_slices(img_proxies, yzc_dims, x_ixs, component, nz_mask, avg_func):

    img_nb = len(img_proxies)
    buffer_dims = (img_nb, len(x_ixs),) + yzc_dims
    buffer = np.zeros(buffer_dims)

    dim_nb = len(img_proxies[0].shape)

    for i_p in range(img_nb):

        if component is None:
            if dim_nb == 3:
                buffer[i_p, :, :, :, :] = np.expand_dims(
                    img_proxies[i_p].dataobj[x_ixs[0]:(x_ixs[-1] + 1), :, :],
                    axis=-1,
                    )
            else:
                buffer[i_p, :, :, :, :] = img_proxies[i_p].dataobj[x_ixs[0]:(x_ixs[-1] + 1), :, :, 0, :]
        else:

            buffer[i_p, :, :, :, :] = np.expand_dims(
                img_proxies[i_p].dataobj[x_ixs[0]:(x_ixs[-1] + 1), :, :, 0, component],
                axis=-1,
                )

    if nz_mask:
        buffer[buffer == 0.0] = np.nan

    m = avg_func(buffer, axis=0)
    m = np.nan_to_num(m, nan=0.0)

    return m



def volume_discretize(array, px_nb):

    vals = array.ravel()
    sort_ixs = np.argsort(vals)[::-1]
    sel_ixs = sort_ixs[0:px_nb]

    out_vals = np.zeros_like(vals)
    out_vals[sel_ixs] = 1.0

    # s = out_vals.sum()

    # out_vals /= s
    out_array = out_vals.reshape(array.shape)

    return out_array


#  return an image in which each pixel is the local sum of the pixels of the
# original image
def convolve_img(
        img,
        radius=None,
        stride=1,
        filt='sum',
        padding='constant',
        kernel=None,
        keep_mask=False,
        ):

    # detect type of input and load array
    if isinstance(img, ants.ANTsImage):
        ants_img_bool = True
        array = img.numpy()
    else:
        ants_img_bool = False
        array = img.copy()

    # get number of channels
    if array.ndim > 3:
        chan_nb = array.shape[3]
    else:
        chan_nb = 1

    # if single channel, create singleton dimension to standardize processing
    if chan_nb == 1:
        array = np.expand_dims(array, axis=-1)

    # optionally save mask
    if keep_mask:
        mask = np.any(array > 0.0, axis=-1)

    if kernel is None:
        # create ball kernel
        kernel, _, px_radii = anisotropic_ball(radius, img.spacing)
    else:
        px_radii = ((np.array(kernel.shape) - 1) / 2).astype(int)

    # set kernel weights
    if filt == 'mean':
        kernel = kernel / kernel.sum()
    elif filt == 'sum':
        pass

    # add singleton channel dimension to kernel
    kernel = np.expand_dims(kernel, axis=-1)

    # pad array along space dimensions
    pad_width = tuple([(px_radii[i], px_radii[i]) for i in range(3)]) + ((0, 0),)
    array = np.pad(array, pad_width=pad_width, mode=padding)

    # prepare strides
    if not isinstance(stride, tuple):
        stride = (stride,) * 3
    dim_strides = stride + (1,)

    import time
    t0 = time.time()

    # create view of array to compute strided convolution
    view = view_as_windows(array, kernel.shape, step=dim_strides)

    t1 = time.time()
    print('\nview', t1 - t0)




    # optionally apply mask
    if keep_mask:

        t0 = time.time()

        view = view[mask]

        t1 = time.time()
        print('\nmask', t1 - t0, view.shape)

        t0 = time.time()

        vx_vals = np.einsum(
            'ncijkl,ijkl->nc',
            view,
            kernel,
            )

        t1 = time.time()
        print('compute', t1 - t0, vx_vals.shape)

        t0 = time.time()

        # unpad
        array = array[tuple(slice(r, -r if r > 0 else None) for r in px_radii)]

        t1 = time.time()
        print('unpad', t1 - t0, array.shape)

        # array = array[
        #     px_radii[0]:-px_radii[0],
        #     px_radii[1]:-px_radii[1],
        #     px_radii[2]:-px_radii[2],
        # ]

        t0 = time.time()

        array[mask] = vx_vals

        t1 = time.time()
        print('fill', t1 - t0, array[mask].shape)

    else:


        array = np.einsum(
            'xyzcijks,ijks->xyzc',
            view,
            kernel,
            )

    # # optionally apply mask
    # if keep_mask:
    #     array[~mask, :] = 0.0

    # remove potential singleton channel dimension
    if chan_nb == 1:
        array = np.squeeze(array, axis=-1)

    # produce output according to input type
    if ants_img_bool:

        new_spacing = tuple(
            [w * s for (w, s) in zip(dim_strides, img.spacing)]
            )

        img = ants.from_numpy(
            array,
            spacing=new_spacing,
            has_components=img.has_components,
            )
    else:
        img = array

    return img



#  return an image that has been decimated with a given stride
def decimate_img(
        img,
        stride=1,
        ):

    # detect type of input and load array
    if isinstance(img, ants.ANTsImage):
        ants_img_bool = True
        array = img.numpy()
    else:
        ants_img_bool = False
        array = img.copy()

    # if no channel, create singleton dimension to standardize processing
    if array.ndim > 3:
        expand_chan = False
    else:
        expand_chan = True
        array = np.expand_dims(array, axis=-1)

    # prepare strides
    if not isinstance(stride, tuple):
        stride = (stride,) * 3
    dim_strides = stride + (1,)

    # tqdm.write(str(array.shape))

    # use view of array to do the decimation
    array = view_as_windows(array, 1, step=dim_strides)
    array = np.squeeze(array, (4, 5, 6, 7))

    # tqdm.write(str(array.shape))

    # remove potential singleton channel dimension
    if expand_chan:
        array = np.squeeze(array, axis=-1)

    # produce output according to input type
    if ants_img_bool:

        new_spacing = tuple(
            [w * s for (w, s) in zip(dim_strides, img.spacing)]
            )

        img = ants.from_numpy(
            array,
            spacing=new_spacing,
            has_components=img.has_components,
            )
    else:
        img = array

    return img


# produce adaptive binning of image
def adaptive_bin_img(img, min_count, stride=1, padding='constant', max_radius=1):

    MAX_ITER = 25

    # detect type of input and load array
    if isinstance(img, ants.ANTsImage):
        ants_img_bool = True
        array = img.numpy()
    else:
        ants_img_bool = False
        array = img.copy()

    # get number of channels
    if array.ndim > 3:
        chan_nb = array.shape[3]
    else:
        chan_nb = 1

    # if single channel, create singleton dimension to standardize processing
    if chan_nb == 1:
        array = np.expand_dims(array, axis=-1)

    # prepare strides
    if not isinstance(stride, tuple):
        stride = (stride,) * 3
    dim_strides = stride + (1,)

    # n = (output_array.sum(axis=3) > 0).sum()

    # create biggest kernel
    spacing = np.array(img.spacing)[:3].copy()
    v, s, _ = anisotropic_ball(max_radius, spacing)

    unq_dists = np.unique(np.round(s[v], 3))[1:]  # remove distance 0
    # diff = np.diff(unq_dists)
    # print(unq_dists, diff)
    # diff = np.append(diff, diff[-1])
    # unq_dists += diff / 2
    unq_dists += np.finfo(np.float32).eps
    if len(unq_dists > MAX_ITER):
        unq_dists = unq_dists[:MAX_ITER]

    ## init
    if stride != 1:
        # create single element kernel
        kernel = skimage.morphology.ball(radius=0).astype('float')

        # kernel, _, px_radii = anisotropic_ball(unq_dists[-1], spacing)

        # add singleton channel dimension to kernel
        kernel = np.expand_dims(kernel, -1)
        # kernel = np.stack((kernel,) * chan_nb, axis=-1)

        # pad array along space dimensions
        # pad_width = tuple([(px_radii[i], px_radii[i]) for i in range(3)]) + ((0, 0),)
        # padded_array = np.pad(array, pad_width=pad_width, mode=padding)

        # create view of array to compute strided convolution
        view = view_as_windows(array, kernel.shape, step=dim_strides)

        # apply kernel and sum along kernel space dimensions
        # (singleton channel dimension of windows is removed at the same time)
        kernel_axes = tuple(np.arange(4, view.ndim, dtype=int))
        output_array = np.squeeze(view, axis=kernel_axes).copy()
    else:
        output_array = array.copy()

    # tqdm.write(str(array.shape))
    # tqdm.write(str(view.shape))

    # return

    # find pixels to bin
    bin_bools = np.logical_and(
        output_array.sum(axis=3) > 0, # select only non-zero pixels
        output_array.sum(axis=3) < min_count # select pixels with low count
        )

    # it = 0

    for dist in unq_dists:

        # print(dist)

        if bin_bools.sum() == 0: # and el < 180:
            break
        # print('remaining', bin_bools.sum(), '/', n, 'after r = ', radius)

        # radius += 1

        # # create ball kernel
        # kernel = skimage.morphology.ball(radius=radius).astype('float')

        kernel, _, px_radii = anisotropic_ball(dist, spacing)

        # add singleton channel dimension to kernel
        kernel = np.expand_dims(kernel, -1)

        # pad array along space dimensions
        pad_width = tuple([(px_radii[i], px_radii[i]) for i in range(3)]) + ((0, 0),)
        padded_array = np.pad(array, pad_width=pad_width, mode=padding)

        # create view of array to compute strided convolution
        view = view_as_windows(padded_array, kernel.shape, step=dim_strides)

        nz_ixs = np.nonzero(bin_bools)
        # tqdm.write(str(bin_bools.shape))

        # apply kernel on all selected windows
        for x, y, z in zip(*nz_ixs):

            output_array[x, y, z] = np.sum(
                view[x, y, z] * kernel,
                axis=(1, 2, 3, 4)
                )

        bin_bools = np.logical_and(
            bin_bools,
            output_array.sum(axis=3) < min_count
            )

        # it += 1


    # remove singleton channel dimension
    if chan_nb == 1:
        output_array = np.squeeze(output_array, axis=-1)

    # produce output according to input type
    if ants_img_bool:
        output_img = ants.from_numpy(
            output_array,
            spacing=tuple([w * s for (w, s) in zip(dim_strides, img.spacing)]),
            has_components=img.has_components,
            )
    else:
        output_img = output_array

    # tqdm.write('final radius: ' + str(radius))

    return output_img



def anisotropic_ball(radius, spacing):

    spacing = np.round(np.array(spacing)[:3].copy(), 4)

    px_radii = np.floor(radius / spacing).astype('int')
    px_nbs = 2 * px_radii + 1

    X, Y, Z = np.mgrid[-px_radii[0]:px_radii[0]:(px_nbs[0] * 1j),
                       -px_radii[1]:px_radii[1]:(px_nbs[1] * 1j),
                       -px_radii[2]:px_radii[2]:(px_nbs[2] * 1j)]
    s = np.sqrt(
        (X * spacing[0]) ** 2 + (Y * spacing[1]) ** 2 + (Z * spacing[2]) ** 2
        )

    v = s <= radius

    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure(figsize=(10, 10), dpi = 150)
    # ax = fig.add_subplot(3, 3, 1, projection=Axes3D.name)
    # ax.voxels(v)

    # unq_dists = np.unique(s[v])[1:] # remove distance 0
    # print(unq_dists)
    # print('unique dists: ', len(unq_dists))

    return v, s, px_radii



def compute_tf_mat(scaling=None, rotation=None):

    tf_mat = np.identity(2)

    # scaling
    if scaling is not None:
        tf_mat = tf_mat * scaling

    # rotation
    if rotation is not None:
        rot_mat = np.array([
            [np.cos(rotation), -np.sin(rotation)],
            [np.sin(rotation), np.cos(rotation)]
            ])
        tf_mat = np.matmul(rot_mat, tf_mat)

    return tf_mat


def gaussian_kernel(kernlen=3, std=1, dim=3):

    gkern1d = scipy.signal.gaussian(kernlen, std=std).reshape(kernlen, 1)

    if dim == 1:
        return gkern1d

    gkern2d = gkern1d * gkern1d.T

    if dim == 2:
        return gkern2d

    if dim == 3:
        gkern3d = gkern2d.T[:, :, None] * gkern1d.T[:, None]
        return gkern3d

    else:
        raise ValueError('Only supports up to 3 dimensions.')




class AbstractImgProcess(ABC):

    INDEP_CHAN_PROCESS = False

    def __init__(self, img, channel=None):

        self.channel = channel

        # detect type of input and load array
        if isinstance(img, ants.ANTsImage):

            self.ants_img_bool = True
            self.array = img.numpy()

            # store input characteristics
            self.in_spacing = img.spacing
            self.in_has_components = img.has_components

        else:
            self.ants_img_bool = False
            self.array = img.copy()

        if channel is not None:
            # select channel to be processed
            self.chan_array = self.array[:, :, :, channel]
        else:
            self.chan_array = self.array

        # standardize process by adding singleton channel dimension
        if self.chan_array.ndim < 4:
            self.chan_array = np.expand_dims(self.chan_array, -1)
            self.expanded = True
        else:
            self.expanded = False

        self.process_chan_nb = self.chan_array.shape[-1]


    def run(self, *args, **kwargs):

        if self.INDEP_CHAN_PROCESS and self.process_chan_nb > 1:
            # process channels independently
            output_array = [
                self.process(self.chan_array[:, :, :, [i_c]], *args, **kwargs)
                for i_c in range(self.process_chan_nb)
                ]
            output_array = np.concatenate(output_array, axis=3)

        else:
            # process channels together
            output_array = self.process(self.chan_array, *args, **kwargs)

        # define output characteristics
        self.define_output()

        if self.expanded:
            output_array = np.squeeze(output_array, axis=-1)

        if self.channel is not None:
            self.array[:, :, :, self.channel] = output_array
        else:
            self.array = output_array

        # produce output according to input type
        if self.ants_img_bool:
            output_img = ants.from_numpy(
                self.array,
                spacing=self.out_spacing,
                has_components=self.out_has_components,
                )
        else:
            output_img = self.array

        return output_img


    def process(self, array, *args, **kwargs):
        pass


    # overload if necessary
    def define_output(self):
        self.out_spacing = self.in_spacing
        self.out_has_components = self.in_has_components



class ThresholdImg(AbstractImgProcess):

    def process(self, array, mode='li', keep='upper', exclude_zeros=False):#, channel=None):

        # if channel is not None:
        #     chan_array = array[:, :, :, channel]
        # else:
        #     chan_array = array

        if exclude_zeros:
            vals = array[array > 0]
        else:
            vals = array.copy().ravel()

        if np.isscalar(mode):
            threshold = mode



        elif mode == 'li':
            tolerance = (vals.max() - vals.min()) / 1000

            threshold = skimage.filters.threshold_li(
                image=vals,
                tolerance=tolerance,
                initial_guess=skimage.filters.threshold_otsu
                )
        else:
            raise ValueError('Unknown mode: ', mode)


        if keep == 'upper':
            rm_bools = array < threshold
        elif keep == 'lower':
            rm_bools = array > threshold

        output_array = array.copy()
        output_array[rm_bools] = 0

        return output_array



class SelectChannelImg(AbstractImgProcess):

    def process(self, array, chan_ixs):

        self.out_chan_nb = len(chan_ixs) if isinstance(chan_ixs, list) else 1
        output_array = array[:, :, :, chan_ixs]

        return output_array

    def define_output(self):

        AbstractImgProcess.define_output(self)

        if self.out_chan_nb > 1:
            self.out_has_components = True
        else:
            self.out_has_components = False



class PadImg32(AbstractImgProcess):

    def process(self, array):

        sh = np.array(array.shape)[:-1]
        new_sh = (np.ceil(sh / 32) * 32).astype('int')
        total_pads = new_sh - sh
        pads = [(p//2, p-p//2) for p in total_pads] + [(0, 0)]
        array = np.pad(array, pads)

        return array


class ResampleImg(AbstractImgProcess):

    def process(self, array, factors=None, spacing=None, anti_aliasing=True, anti_aliasing_sigma='custom', order=1):

        if factors is not None and spacing is not None:
            raise ValueError("'factors' and 'spacing' are mutually exclusive. Please provide only one of them.")

        if factors is not None:
            scale = factors
            self.target_spacing = 1.0 / scale * np.array(self.in_spacing[:3])

        if spacing is not None:
            scale = np.array(self.in_spacing[:3]) / np.array(spacing)
            self.target_spacing = spacing

        if anti_aliasing and anti_aliasing_sigma == 'custom':
            # stronger anti-aliasing than when using scikit-image
            anti_aliasing_sigma = 1.0 / (2.0 * scale)
            anti_aliasing_sigma = np.where(
                scale < 1.0,
                anti_aliasing_sigma,
                0.0
                )

        if array.ndim > 3:
            channel_axis = 3
            if anti_aliasing_sigma is not None:
                anti_aliasing_sigma = np.append(anti_aliasing_sigma, 0.0)
        else:
            channel_axis = None

        array = skimage.transform.rescale(
            array,
            scale,
            order=order,
            mode='reflect',
            anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma,
            channel_axis=channel_axis
            )

        return array


    def define_output(self):

        AbstractImgProcess.define_output(self)

        self.out_spacing = np.array(self.in_spacing)
        self.out_spacing[:3] = self.target_spacing
        self.out_spacing = tuple(self.out_spacing)

        # print('out', self.out_spacing)


class NormImg(AbstractImgProcess):

    def process(self, array):

        array /= 255.0

        return array


def binarize_array(array):

    mask = (array > np.finfo(array.dtype).eps).astype('bool')

    return mask



class MaskImg(AbstractImgProcess):

    def process(self, array, mask, mask_channel=None, value=0.0):

        if isinstance(mask, str):
            mask = read_img(mask)

        if isinstance(mask, ants.ANTsImage):
            mask = mask.numpy()

        if mask_channel is not None:
            mask = mask[:, :, :, mask_channel]

        if mask.dtype != 'bool':
            mask = binarize_array(mask)

        array[~mask, :] = value

        return array


class NormalizeImg(AbstractImgProcess):

    def process(self, array, func=np.mean, factor=1.0):

        array = array / func(array) * factor

        return array



class NormPercentImg(AbstractImgProcess):

    def process(self, array, percents, target_pc_vals, exclude_zeros=False):

        if exclude_zeros:
            lgcs = array > 0
        else:
            lgcs = np.ones_like(array, dtype='bool')

        array_vals = array[lgcs]

        pc_vals = np.percentile(array_vals, percents)

        amp = pc_vals[1] - pc_vals[0]
        new_amp = target_pc_vals[1] - target_pc_vals[0]
        array_vals = (array_vals - pc_vals[0]) / amp * new_amp + target_pc_vals[0]

        array[lgcs] = array_vals

        return array



class MatchHistImg(AbstractImgProcess):

    def process(
            self,
            array,
            ref_img=None,
            ref_hist_comps=None,
            bin_nb=None,
            pt_nb=100,
            threshold=None,
            ):

        if ref_img is None and ref_hist_comps is None:
            raise ValueError("Requires 'ref_img' or 'ref_hist_comps'.")
        elif ref_img is not None and ref_hist_comps is not None:
            raise ValueError("Only one of 'ref_img' and 'ref_hist_comps' can be provided.")

        if ref_img is not None:
            ref_array = ref_img.numpy()

            if threshold is not None:
                if np.isscalar(threshold):
                    ref_array = ref_array[ref_array > threshold]
                else:
                    ref_array = ref_array[ref_array > threshold(ref_array)]

            # compute reference histogram
            ref_hist, ref_edges = np.histogram(ref_array.ravel(), bins=bin_nb)
        else:
            ref_hist, ref_edges = ref_hist_comps
            bin_nb = len(ref_edges) - 1

        # convert to integer frequencies
        int_ref_hist = np.round(ref_hist / ref_hist.max() * 1e9)

        # create ITK histogram
        itk_hist = itk.Histogram.New(MeasurementVectorSize=1)
        size = itk.Size[1]()
        size[0] = int(bin_nb)
        lower_lound = (float(ref_edges[0]),)
        upper_bound = (float(ref_edges[-1]),)
        # print(size, lowerBound, upperBound)
        # print('ZZZ')
        itk_hist.Initialize(size, lower_lound, upper_bound)
        # print('XXX')

        ixs = np.arange(bin_nb)
        for ix, f in zip(ixs, int_ref_hist):
            itk_hist.SetFrequency(int(ix), int(f))

        # zero_lgcs = np.zeros(array.shape, dtype=bool)

        thres_val = self.get_threshold_value(array, threshold)
        thres_lgcs = array > thres_val

        vals = array[thres_lgcs]

        # convert input array to ITK image
        vals = np.reshape(vals, (-1, 1))
        img_itk = itk.GetImageFromArray(vals)
        img_type = type(img_itk)

        # prepare filter
        filt = itk.HistogramMatchingImageFilter[img_type, img_type].New()
        filt.SetGenerateReferenceHistogramFromImage(False)
        filt.SetReferenceHistogram(itk_hist)
        filt.SetThresholdAtMeanIntensity(False)
        filt.SetNumberOfHistogramLevels(bin_nb)
        filt.SetNumberOfMatchPoints(pt_nb)

        # set input and update filter
        filt.SetInput(img_itk)
        filt.Update()

        # apply filter and reorganize components
        img_itk = filt.GetOutput()

        vals = itk.array_from_image(img_itk).ravel()
        # print('vals.shape', vals.shape)


        array[thres_lgcs] = vals

        # print('array.shape', array.shape)


        return array

    @staticmethod
    def get_threshold_value(array, threshold):

        if threshold is None:
            thres_val = -np.Inf
        elif threshold == 'mean':
            thres_val = np.mean(array)
        elif np.isscalar(threshold):
            thres_val = threshold
        else:
            raise ValueError('Unknown value of threshold argument.')

        return thres_val











# def ants_to_mip(ants_img):

#     array = ants_img.numpy()
#     mip_img = np_to_mip(array, ants_img.spacing)

#     return mip_img


# def np_to_mip(array, spacing):

#     array = np.moveaxis(array, [0, 1, 2], [2, 1, 0])
#     mip_img = mipImage(array, spacing[::-1])

#     return mip_img

# def mip_to_ants(mip_img):

#     array, spacing = mip_to_np(mip_img)
#     ants_img = ants.from_numpy(
#         array,
#         spacing=spacing,
#         )

#     return ants_img

# def mip_to_np(mip_img):

#     array = np.moveaxis(mip_img, [0, 1, 2], [2, 1, 0])
#     spacing=mip_img.spacing[::-1]

#     return array, spacing


# class MIPDeconvImg(AbstractImgProcess):

#     INDEP_CHAN_PROCESS = True

#     def process(self, array, rescale=False):

#         if rescale:
#             max_val = array.max()

#         mip_img = np_to_mip(array[:, :, :, 0], spacing=self.in_spacing)

#         xy_res, z_res = compute_SFSC_resolution(mip_img)

#         fwhm = [
#             z_res,
#             xy_res
#             ]

#         psf_generator = psfgen.PsfFromFwhm(fwhm)
#         psf = psf_generator.volume()

#         deconvolved = wiener.wiener_deconvolution(
#             mip_img,
#             psf,
#             snr=200,
#             add_pad=100
#             )

#         array, _ = mip_to_np(deconvolved)

#         array = np.expand_dims(array, -1)

#         if rescale:
#             array *= max_val / array.max()

#         return array


# def compute_SFSC_resolution(img):

#     if isinstance(img, ants.ANTsImage):
#         mip_img = ants_to_mip(img)
#     elif isinstance(img, mipImage):
#         mip_img = img
#     else:
#         raise TypeError('Unsupported image type.')

#     z_correction = mip_img.spacing[0] / mip_img.spacing[1]

#     # print('mip_img.shape', mip_img.shape)

#     # prepare cube image
#     cube_img = imops.zoom_to_isotropic_spacing(mip_img, order=0)

#     # Find the index of the largest dimension
#     largest_dim = np.argmax(cube_img.shape)

#     # Check if the largest dimension is odd
#     if cube_img.shape[largest_dim] % 2 != 0:

#         # Remove the last index
#         array = np.take(
#             cube_img,
#             range(cube_img.shape[largest_dim] - 1),
#             axis=largest_dim
#             )
#         cube_img = mipImage(array, cube_img.spacing)

#     # print('cube_img.shape 1', cube_img.shape)

#     cube_img = imops.zero_pad_to_cube(cube_img)

#     # print('cube_img.shape 2', cube_img.shape)

#     args_list = [
#         None,
#         '--bin-delta=10',
#         '--resolution-threshold-criterion=snr',
#         '--resolution-snr-value=0.5',
#         '--angle-delta=15',
#         '--enable-hollow-iterator',
#         '--extract-angle-delta=.1',
#         '--resolution-point-sigma=0.01',
#         '--frc-curve-fit-type=spline'
#         ]
#     args = options.get_frc_script_options(args_list)
#     result = fsc.calculate_one_image_sectioned_fsc(
#         cube_img,
#         args,
#         z_correction=z_correction
#         )

#     xy_res = result[0].resolution["resolution"]
#     z_res = result[90].resolution["resolution"]

#     return xy_res, z_res



class SharpenImg(AbstractImgProcess):

    def process(self, array, blending_weight):

        # short alias for clarity
        w = blending_weight

        # make ANTsImage object to be able to use the ANTs method
        array_img = ants.from_numpy(array)

        # sharpen
        sharp_array = ants.utils.iMath(array_img, "Sharpen").numpy()

        # blend original and sharpened image
        array = w * array + (1 - w) * sharp_array

        return array



class SumChanImg(AbstractImgProcess):

    def process(self, array):

        array = array.sum(axis=3)

        return array


    def define_output(self):

        self.out_spacing = self.in_spacing[:3]
        self.out_has_components = False



class EraseAreaImg(AbstractImgProcess):

    def process(self, array, axis, coord, direction):

        # transform in voxel coordinates
        ix_coord = np.floor(coord / self.in_spacing[axis]).astype('int')
        ix_coord = min(ix_coord, array.shape[axis])
        if axis > 0:
            ix_coord = array.shape[axis] - ix_coord
        else:
            direction = -direction

        # create a slice object for all dimensions
        slices = [slice(None)] * array.ndim

        if direction == -1:
            slices[axis] = slice(ix_coord, None)  # from coord to the end
        elif direction == 1:
            slices[axis] = slice(None, ix_coord)  # from the beginning to coord
        else:
            raise ValueError("Direction must be 1 or -1.")

        # set the specified values to zero
        array[tuple(slices)] = 0

        return array














