#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segment, mask and bin 'MB-STn' files

"""

import os

import matplotlib.pyplot as plt
import numpy as np

import mbflim.utils as ut
import mbflim.visualization as vis
import mbflim.image.image_processing as pro
import mbflim.image.image_set_processing as setpro
import mbflim.parameters as par

#%% Parameters

# Parallelism
PROCESS_NB = 16
MAX_THREAD_NB = 32

#%% Fixed parameters

imaging = 'FLIM-marker'
study = 'MB-STn'

# Binning
bin_name = 'fixed_kernel'

# Load configuration file
config = ut.load_config()

#%% Set global variables

# Parallelism
os.environ["MBFLIM_NUM_CPUS"] = str(PROCESS_NB)
os.environ["MBFLIM_NUM_THREADS"] = str(MAX_THREAD_NB)


#%% Prepare paths

study_dpath = os.path.join(
    config['paths']['source_dpath'],
    imaging,
    study + '/',
)
marker_in_dpath = os.path.join(study_dpath, 'marker/')
flim_in_dpath = os.path.join(study_dpath, 'FLIM/')

analysis_dpath = os.path.join(
    config['paths']['output_dpath'],
    imaging,
    study,
)
marker_out_dpath = os.path.join(analysis_dpath, 'marker/')
flim_out_dpath = os.path.join(analysis_dpath, 'FLIM/')

# Marker processing steps
steps = [
    'crop',
    'clean1',
    'clean2',
]
marker_dpaths = {}
for i_s, step in enumerate(steps):
    marker_dpaths.update({step: os.path.join(marker_out_dpath, str(i_s) + '_' + step + '/')})

# FLIM processing steps
steps = [
    'crop',
    'mask',
    'bin',
]
flim_dpaths = {}
for i_s, step in enumerate(steps):
    flim_dpaths.update({step: os.path.join(flim_out_dpath, str(i_s) + '_' + step + '/')})

#%% Crop marker images

margins = [[60, 60], [40, 40], [100, 100]]
reference = 'center_of_mass'
in_pixel = False
extend = False

setpro.SetSimpleProcess(
    marker_in_dpath,
    marker_dpaths['crop'],
    process_func=pro.smartcrop_img,
    prefix='crop_',
    margins=margins,
    in_pixel=in_pixel,
    reference=reference,
    extend=extend,
)

vis.SetPlotProj(
    marker_dpaths['crop'],
    marker_dpaths['crop'] + 'chan0_figs/',
    component=0,
    norm='minmax',
)

#%% Crop FLIM images

if reference == 'center_of_mass':
    setpro.SetRefCrop(
        [flim_in_dpath, marker_in_dpath],
        flim_dpaths['crop'],
        margins=margins,
        in_pixel=in_pixel,
        extend=extend,
    )
else:
    setpro.SetSimpleProcess(
        flim_in_dpath,
        flim_dpaths['crop'],
        process_func=pro.smartcrop_img,
        prefix='crop_',
        margins=margins,
        in_pixel=in_pixel,
        reference=reference,
        extend=extend
    )

vis.SetPlotProj(
    flim_dpaths['crop'],  flim_dpaths['crop'] + 'sum_figs/',
    component=0,
    norm='minmax',
)

#%% Segment marker images - step 1

min_width = 1 # in µm or None to keep only largest element
smoothing = 0.5 # in µm
clean_threshold = 'li'

setpro.SetSimpleProcess(
    marker_dpaths['crop'],
    marker_dpaths['clean1'],
    process_func=pro.remove_small_objects_img,
    prefix='clean_',
    min_width=min_width,
    threshold=clean_threshold,
    smoothing=smoothing,
    fill_holes=False,
    automask=False,
)

vis.SetPlotProj(
    marker_dpaths['clean1'],
    marker_dpaths['clean1'] + 'chan0_figs/',
    component=0,
    norm='minmax'
)

#%% Segment marker images - step 1

min_width = 0.5 # in µm
smoothing = None # in µm
clean_threshold = 'li'

setpro.SetSimpleProcess(
    marker_dpaths['clean1'],
    marker_dpaths['clean2'],
    process_func=pro.remove_small_objects_img,
    prefix='clean_',
    min_width=min_width,
    threshold=clean_threshold,
    smoothing=smoothing,
    fill_holes=False,
    automask=True,
)

vis.SetPlotProj(
    marker_dpaths['clean2'],
    marker_dpaths['clean2'] + 'chan0_figs/',
    component=0,
    norm='minmax'
)

#%% Mask FLIM

mask_dpath = flim_dpaths['mask'] + 'none' + '/'

setpro.SetMask(
    [flim_dpaths['crop'], marker_dpaths['clean2']],
    mask_dpath,
)

vis.SetPlotProj(
    mask_dpath,
    mask_dpath + 'sum_figs/',
    norm='minmax',
    mode='max',
)

#%% Mask FLIM images based on photon count

threshold = 'IQR'
factor = 3
thresh_name = threshold + str(factor)

mask_dpath = flim_dpaths['mask'] + 'none' + '/'
sumchan_dpath = flim_dpaths['mask'] + 'none' + '/' + 'sumchan/'

# sum
setpro.SetProcess(
    mask_dpath,
    sumchan_dpath,
    pro.SumChanImg,
)

setpro.SetNormPercentiles(
    sumchan_dpath,
    sumchan_dpath + 'norm_pc/',
    exclude_zeros=True,
)

# plot histograms
hist_summary = vis.SetSummarizeHistograms(
    sumchan_dpath + 'norm_pc/',
    sumchan_dpath + 'norm_pc/' + thresh_name + '/',
    exclude_zeros=True,
    bin_nb=100,
    sharex=True,
    yscale='linear',
    vrange=(0.0, 1000),
    include_sup=True,
    threshold=threshold,
    factor=factor,
)


# plot images
vis.SetPlotProj(
    sumchan_dpath, sumchan_dpath + 'figs/',
    norm='minmax',
    mode='max',
)

# plot histograms
hist_summary = vis.SetSummarizeHistograms(
    sumchan_dpath,
    sumchan_dpath + thresh_name + '/',
    exclude_zeros=True,
    bin_nb=100,
    sharex=True,
    yscale='linear',
    vrange=(0.0, 1000),
    include_sup=True,
    threshold=threshold,
    factor=factor,
    )

thresholds = hist_summary.thresholds

norms = [(0.0, t[1]) for t in thresholds]

# plot images
vis.SetPlotProj(
    sumchan_dpath, sumchan_dpath + thresh_name + '/' + 'figs/',
    norm=norms,
    mode='max',
    colors=plt.cm.gray.with_extremes(over='r'),
)

thresholds = np.array(hist_summary.thresholds)
# low_thresholds = thresholds[:, 0]
high_thresholds = thresholds[:, 1]

# threshold high counts
setpro.SetProcess(
    sumchan_dpath,
    sumchan_dpath + thresh_name + '/' + 'th_up/',
    pro.ThresholdImg,
    keep='lower',
    distrib_kwargs={'mode': high_thresholds},
)

restrict_dpath = flim_dpaths['mask'] + thresh_name + '/'

setpro.SetMask(
    [mask_dpath, sumchan_dpath + thresh_name + '/' + 'th_up/'],
    restrict_dpath,
    # process_nb=8,
)

vis.SetPlotProj(
    restrict_dpath, restrict_dpath + 'sum_figs/',
    norm='minmax',
    mode='mean',
)

#%% Bin FLIM images

RESTRICT_NAME = 'IQR3'

setpro.SetSimpleProcess(
    flim_dpaths['mask'] + RESTRICT_NAME + '/',
    flim_dpaths['bin'] + bin_name + '/' + RESTRICT_NAME + '/',
    process_func=pro.convolve_img,
    keep_mask=True,
    process_nb=8,
    kernel=par.BIN_KERNEL,
)

vis.SetPlotProj(
    flim_dpaths['bin'] + bin_name + '/' + RESTRICT_NAME + '/',
    flim_dpaths['bin'] + bin_name + '/' + RESTRICT_NAME + '/' + 'sum_figs/',
    norm='minmax',
    mode='max',
)

# sum
setpro.SetProcess(
    flim_dpaths['bin'] + bin_name + '/' + RESTRICT_NAME + '/' ,
    flim_dpaths['bin'] + bin_name + '/' + RESTRICT_NAME + '/sumchan/' ,
    pro.SumChanImg,
)

hist_summary = vis.SetSummarizeHistograms(
    flim_dpaths['bin'] + bin_name + '/' + RESTRICT_NAME + '/sumchan/' ,
    flim_dpaths['bin'] + bin_name + '/' + RESTRICT_NAME + '/sumchan/',
    exclude_zeros=True,
    bin_nb=100,
    sharex=True,
    yscale='linear',
    vrange=(0.0, 10000),
    include_sup=True,
    threshold='fixed',
    factor=500,
)

#%% decimate 3

stride = (3, 3, 1)

setpro.SetSimpleProcess(
    flim_dpaths['bin'] + bin_name + '/' + RESTRICT_NAME + '/',
    flim_dpaths['decimate'] + bin_name + '/' + RESTRICT_NAME + '/',
    process_func=pro.decimate_img,
    stride=stride,
)

vis.SetPlotProj(
    flim_dpaths['decimate'] + bin_name + '/' + RESTRICT_NAME + '/',
    flim_dpaths['decimate'] + bin_name + '/' + RESTRICT_NAME + '/' + 'sum_figs/',
    norm='minmax',
    mode='max',
)
