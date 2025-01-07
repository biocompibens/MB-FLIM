#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segment, mask and bin 'MB-KCc' files

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
study = 'MB-KCc'

# Binning
bin_name = 'fixed_kernel'

# Cleaning
smoothing = 2 # Gaussian kernel STD in pixel

# Template
SEL_HEMISPHERES = 'both'

# Registration metrics
reg_metrics = par.get_registration_metrics()
metrics = [reg_metrics['mi_metric'], reg_metrics['cc_metric']]

# Registration parameters
iterations = [250, 250, 250, 0]
registration_parameters = par.get_registration_parameters(
    metrics=metrics,
    affine_iterations=iterations,
    syn_iterations=iterations,
    shrink_factors=[8, 4, 2, 1],
    smoothing_sigmas=[4, 2, 1, 0],
    max_thread_nb=MAX_THREAD_NB,
)
rig_tf_params = registration_parameters['rig_tf_params']
rig_aff_syn_tf_params = registration_parameters['rig_aff_syn_tf_params']

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
    'clean',
]
marker_dpaths = {}
for i_s, step in enumerate(steps):
    marker_dpaths.update({step: os.path.join(marker_out_dpath, str(i_s) + '_' + step + '/')})

# FLIM processing steps
steps = [
    'mask',
    'bin',
]
flim_dpaths = {}
for i_s, step in enumerate(steps):
    flim_dpaths.update({step: os.path.join(flim_out_dpath, str(i_s) + '_' + step + '/')})

# Template path
template_dpath = os.path.join(
    config['paths']['output_dpath'],
    'markers',
    'MB-KCc-STn',
    'template',
    SEL_HEMISPHERES,
    '2_template_center'  + '/',
)

# Mask paths
somata_mask_dpath = os.path.join(
    config['paths']['output_dpath'],
    imaging,
    'MB-KCc-KCn',
    'masks',
    SEL_HEMISPHERES,
    '4_somata_mask'  + '/',
)
calyx_mask_dpath = os.path.join(
    config['paths']['output_dpath'],
    imaging,
    'MB-KCc-KCn',
    'masks',
    SEL_HEMISPHERES,
    '6_calyx_mask'  + '/',
)


#%% Clean marker images

min_width = None # in µm or None to keep only largest element
clean_threshold = 'li'

setpro.SetSimpleProcess(
    marker_in_dpath,
    marker_dpaths['clean'],
    process_func=pro.remove_small_objects_img,
    prefix='clean_',
    min_width=min_width,
    threshold=clean_threshold,
    component=0,
    smoothing=smoothing,
)

vis.SetPlotProj(
    marker_dpaths['clean'],
    marker_dpaths['clean'] + 'chan0_figs/',
    component=0,
    legend=['KC'],
    norm='minmax',
)


#%% Mask FLIM with cleaned marker images

restrict_name = 'none'
restrict_dpath = marker_dpaths['clean']
mask_dpath = flim_dpaths['mask'] + restrict_name + '/'

setpro.SetMask(
    [flim_in_dpath, restrict_dpath],
    mask_dpath,
)

vis.SetPlotProj(
    mask_dpath, mask_dpath + 'sum_figs/',
    norm='minmax',
    mode='midslice',
)

#%% Resample template and mask to resolution of each image

restrict_name = 'reg_mask'

# template
setpro.SetMatchResolution(
    [marker_dpaths['clean'], template_dpath],
    marker_dpaths['clean'] + 'matched_res_tpl/',
)

vis.SetPlotProj(
    marker_dpaths['clean'] + 'matched_res_tpl/',
    marker_dpaths['clean'] + 'matched_res_tpl/' + 'chan0_figs/',
    norm='minmax',
    mode='max',
)

# mask
setpro.SetMatchResolution(
    [marker_dpaths['clean'], somata_mask_dpath],
    marker_dpaths['clean'] + 'matched_res_mask/',
    anti_aliasing=False,
    order=0,
)

vis.SetPlotProj(
    marker_dpaths['clean'] + 'matched_res_mask/',
    marker_dpaths['clean'] + 'matched_res_mask/' + 'chan0_figs/',
    norm='minmax',
    mode='max',
)

# calyx mask
setpro.SetMatchResolution(
    [marker_dpaths['clean'], calyx_mask_dpath],
    marker_dpaths['clean'] + 'matched_res_calyx_mask/',
    anti_aliasing=False,
    order=0,
    )

vis.SetPlotProj(
    marker_dpaths['clean'] + 'matched_res_calyx_mask/',
    marker_dpaths['clean'] + 'matched_res_calyx_mask/' + 'chan0_figs/',
    norm='minmax',
    mode='max',
)

#%% Crop

# crop parameters
tight_margins = par.get_tight_margins()
wide_margins = par.get_wide_margins()
reference = 'center_of_mass'
in_pixel = False
extend = True

# step 1

setpro.SetSimpleProcess(
    marker_dpaths['clean'],
    marker_dpaths['clean'] + 'crop1/',
    process_func=pro.smartcrop_img,
    margins=tight_margins,
    in_pixel=in_pixel,
    reference=reference,
    extend=extend,
)

# step 2

setpro.SetSimpleProcess(
    marker_dpaths['clean'] + 'crop1/',
    marker_dpaths['clean'] + 'crop2/',
    process_func=pro.smartcrop_img,
    margins=wide_margins,
    in_pixel=in_pixel,
    reference=reference,
    extend=extend,
    )

vis.SetPlotProj(
    marker_dpaths['clean'] + 'crop2/',
    marker_dpaths['clean'] + 'crop2/' + 'chan0_figs/',
    component=0,
    norm='minmax',
)

# crop FLIM images

# step 1
in_dpath = flim_dpaths['mask'] + 'none' + '/'
out_dpath = flim_dpaths['mask'] + 'none' + '/crop1/'

if reference == 'center_of_mass':
    setpro.SetRefCrop(
        [in_dpath, marker_dpaths['clean']],
        out_dpath,
        margins=tight_margins,
        in_pixel=in_pixel,
        extend=extend,
        )
else:
    setpro.SetSimpleProcess(
        in_dpath,
        out_dpath,
        process_func=pro.smartcrop_img,
        prefix='crop_',
        margins=tight_margins,
        in_pixel=in_pixel,
        reference=reference,
        extend=extend
        )

# step 2
in_dpath = flim_dpaths['mask'] + 'none' + '/crop1/'
out_dpath = flim_dpaths['mask'] + 'none' + '/crop2/'

if reference == 'center_of_mass':
    setpro.SetRefCrop(
        [in_dpath, marker_dpaths['clean'] + 'crop1/'],
        out_dpath,
        margins=wide_margins,
        in_pixel=in_pixel,
        extend=extend,
)
else:
    setpro.SetSimpleProcess(
        in_dpath,
        out_dpath,
        process_func=pro.smartcrop_img,
        prefix='crop_',
        margins=wide_margins,
        in_pixel=in_pixel,
        reference=reference,
        extend=extend
        )

# crop template

# step 1
setpro.SetSimpleProcess(
    marker_dpaths['clean'] + 'matched_res_tpl/',
    marker_dpaths['clean'] + 'crop1_tpl/',
    process_func=pro.smartcrop_img,
    margins=tight_margins,
    in_pixel=in_pixel,
    reference=reference,
    extend=extend,
)

# step 2
setpro.SetSimpleProcess(
    marker_dpaths['clean'] + 'crop1_tpl/',
    marker_dpaths['clean'] + 'crop2_tpl/',
    process_func=pro.smartcrop_img,
    margins=wide_margins,
    in_pixel=in_pixel,
    reference=reference,
    extend=extend,
)

vis.SetPlotProj(
    marker_dpaths['clean'] + 'crop2_tpl/',
    marker_dpaths['clean'] + 'crop2_tpl/' + 'chan0_figs/',
    norm='minmax',
    mode='max',
)

# crop mask

# step 1
in_dpath = marker_dpaths['clean'] + 'matched_res_mask/'
out_dpath = marker_dpaths['clean'] + 'crop1_mask/'

if reference == 'center_of_mass':
    setpro.SetRefCrop(
        [in_dpath, marker_dpaths['clean'] + 'matched_res_tpl/'],
        out_dpath,
        margins=tight_margins,
        in_pixel=in_pixel,
        extend=extend,
)
else:
    setpro.SetSimpleProcess(
        in_dpath,
        out_dpath,
        process_func=pro.smartcrop_img,
        prefix='crop_',
        margins=tight_margins,
        in_pixel=in_pixel,
        reference=reference,
        extend=extend
)

# step 2
in_dpath = marker_dpaths['clean'] + 'crop1_mask/'
out_dpath = marker_dpaths['clean'] + 'crop2_mask/'

if reference == 'center_of_mass':
    setpro.SetRefCrop(
        [in_dpath, marker_dpaths['clean'] + 'crop1_tpl/'],
        out_dpath,
        margins=wide_margins,
        in_pixel=in_pixel,
        extend=extend,
)
else:
    setpro.SetSimpleProcess(
        in_dpath,
        out_dpath,
        process_func=pro.smartcrop_img,
        prefix='crop_',
        margins=wide_margins,
        in_pixel=in_pixel,
        reference=reference,
        extend=extend
)

# crop calyx mask

# step 1
in_dpath = marker_dpaths['clean'] + 'matched_res_calyx_mask/'
out_dpath = marker_dpaths['clean'] + 'crop1_calyx_mask/'

if reference == 'center_of_mass':
    setpro.SetRefCrop(
        [in_dpath, marker_dpaths['clean'] + 'matched_res_tpl/'],
        out_dpath,
        margins=tight_margins,
        in_pixel=in_pixel,
        extend=extend,
)
else:
    setpro.SetSimpleProcess(
        in_dpath,
        out_dpath,
        process_func=pro.smartcrop_img,
        prefix='crop_',
        margins=tight_margins,
        in_pixel=in_pixel,
        reference=reference,
        extend=extend
)

# step 2
in_dpath = marker_dpaths['clean'] + 'crop1_calyx_mask/'
out_dpath = marker_dpaths['clean'] + 'crop2_calyx_mask/'

if reference == 'center_of_mass':
    setpro.SetRefCrop(
        [in_dpath, marker_dpaths['clean'] + 'crop1_tpl/'],
        out_dpath,
        margins=wide_margins,
        in_pixel=in_pixel,
        extend=extend,
)
else:
    setpro.SetSimpleProcess(
        in_dpath,
        out_dpath,
        process_func=pro.smartcrop_img,
        prefix='crop_',
        margins=wide_margins,
        in_pixel=in_pixel,
        reference=reference,
        extend=extend
)

#%% Register

setpro.SetRegister(
    [
        marker_dpaths['clean'] + 'crop2_tpl/',
        marker_dpaths['clean'] + 'crop2/',
    ],
    marker_dpaths['clean'] + 'reg/',
    params=rig_aff_syn_tf_params,
    component=0,
)

vis.SetPlotProj(
    marker_dpaths['clean'] + 'reg/',
    marker_dpaths['clean'] + 'reg/' + 'chan0_figs/',
    norm='minmax',
    mode='max',
)

#%% Apply registration

setpro.SetApplyRegistration(
    [
        marker_dpaths['clean'] + 'crop2_mask/',
        marker_dpaths['clean'] + 'reg/',
    ],
    marker_dpaths['clean'] + 'reg_mask/',
    interpolator="nearestNeighbor",
)

vis.SetPlotProj(
    marker_dpaths['clean'] + 'reg_mask/',
    marker_dpaths['clean'] + 'reg_mask/' + 'chan0_figs/',
    norm='minmax',
    mode='max',
)

setpro.SetApplyRegistration(
    [
        marker_dpaths['clean'] + 'crop2_calyx_mask/',
        marker_dpaths['clean'] + 'reg/',
    ],
    marker_dpaths['clean'] + 'reg_calyx_mask/',
    interpolator="nearestNeighbor",
)

vis.SetPlotProj(
    marker_dpaths['clean'] + 'reg_calyx_mask/',
    marker_dpaths['clean'] + 'reg_calyx_mask/' + 'chan0_figs/',
    norm='minmax',
    mode='max',
)

#%% Mask

# mask
setpro.SetMask(
    [marker_dpaths['clean'] + 'crop2/', marker_dpaths['clean'] + 'reg_mask/'],
    marker_dpaths['clean'] + 'masked/',
)

vis.SetPlotProj(
    marker_dpaths['clean'] + 'masked/',
    marker_dpaths['clean'] + 'masked/' + 'chan0_figs/',
    norm='minmax',
    mode='max',
)

mask_dpath = flim_dpaths['mask'] + 'reg_mask' + '/'

setpro.SetMask(
    [flim_dpaths['mask'] + 'none' + '/crop2/', marker_dpaths['clean'] + 'reg_mask/'],
    mask_dpath,
    process_nb=4,
)

vis.SetPlotProj(
    mask_dpath, mask_dpath + 'sum_figs/',
    norm='minmax',
    mode='max',
)

# calyx mask
setpro.SetMask(
    [marker_dpaths['clean'] + 'crop2/' , marker_dpaths['clean'] + 'reg_calyx_mask/'],
    marker_dpaths['clean'] + 'calyx_masked/',
)

vis.SetPlotProj(
    marker_dpaths['clean'] + 'calyx_masked/',
    marker_dpaths['clean'] + 'calyx_masked/' + 'chan0_figs/',
    norm='minmax',
    mode='max',
)

mask_dpath = flim_dpaths['mask'] + 'calyx_reg_mask' + '/'

setpro.SetMask(
    [flim_dpaths['mask'] + 'none' + '/crop2/', marker_dpaths['clean'] + 'reg_calyx_mask/'],
    mask_dpath,
    process_nb=4,
)

vis.SetPlotProj(
    mask_dpath,
    mask_dpath + 'sum_figs/',
    norm='minmax',
    mode='max',
)

#%% Mask FLIM images based on photon count

masked_region = 'somata'

match masked_region:
   case 'posterior' :
       restrict_name = 'reg_mask'
       prefix = ''

   case 'somata':
       restrict_name = 'tpl_reg_mask'
       prefix = 'tpl_'

   case 'calyx':
       restrict_name = 'calyx_reg_mask'
       prefix = 'calyx_'

mask_dpath = flim_dpaths['mask'] + restrict_name + '/'
sumchan_dpath = mask_dpath + 'sumchan/'

threshold = 'IQR'
factor = 3
thresh_name = prefix + threshold + str(factor)

sumchan_dpath = mask_dpath + 'sumchan/'

# sum
setpro.SetProcess(
    mask_dpath,
    sumchan_dpath,
    pro.SumChanImg,
)

# plot images
vis.SetPlotProj(
    sumchan_dpath, sumchan_dpath + 'figs/',
    norm='minmax',
    mode='max',
)

# compute normalized count images (for visualization)
setpro.SetNormPercentiles(
    sumchan_dpath,
    sumchan_dpath + 'norm_pc/',
    exclude_zeros=True,
    target_vals=[9, 10],
)

# plot normalized count histograms (for visualization)
hist_summary = vis.SetSummarizeHistograms(
    sumchan_dpath + 'norm_pc/',
    sumchan_dpath + 'norm_pc/' + thresh_name + '/',
    exclude_zeros=True,
    bin_nb=100,
    sharex=True,
    yscale='linear',
    vrange=(7, 14),
    include_sup=True,
    threshold=threshold,
    factor=factor,
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
    process_nb=4,
)

vis.SetPlotProj(
    restrict_dpath, restrict_dpath + 'sum_figs/',
    norm='minmax',
    mode='mean',
)


#%% bin FLIM images

match masked_region:
   case 'posterior' :
       restrict_name = 'IQR3'

   case 'somata':
       restrict_name = 'tpl_IQR3'

   case 'calyx':
       restrict_name = 'calyx_IQR3'

mask_dpath = flim_dpaths['mask'] + restrict_name + '/'
bin_dpath = flim_dpaths['bin'] + bin_name + '/' + restrict_name + '/'

setpro.SetSimpleProcess(
    mask_dpath,
    bin_dpath,
    process_func=pro.convolve_img,
    kernel=par.BIN_KERNEL,
    keep_mask=True,
    process_nb=4,
)

vis.SetPlotProj(
    bin_dpath,
    bin_dpath + 'sum_figs/',
    norm='minmax',
    mode='midslice',
)

#%% Decimate

stride = (3, 3, 1)

dec_dpath = flim_dpaths['decimate'] + bin_name + '/' + restrict_name + '/'

setpro.SetSimpleProcess(
    bin_dpath,
    dec_dpath,
    process_func=pro.decimate_img,
    stride=stride,
)

vis.SetPlotProj(
    dec_dpath, dec_dpath + 'sum_figs/',
    norm='minmax',
    mode='midslice',
)
