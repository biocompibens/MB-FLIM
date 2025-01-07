#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segment, mask and bin 'brain-KCc' files

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
study = 'brain-KCc'

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
    'split',
    'posterior',
    'whole',
    'peduncle',
    'vertical',
    'medial',
    'merge_vis',
    'join_vis',
]
marker_dpaths = {}
for i_s, step in enumerate(steps):
    marker_dpaths.update({step: os.path.join(marker_out_dpath, str(i_s) + '_' + step + '/')})

# FLIM processing steps
steps = [
    'mask',
    'bin',
    'decimate',
]
flim_dpaths = {}
for i_s, step in enumerate(steps):
    flim_dpaths.update({step: os.path.join(flim_out_dpath, str(i_s) + '_' + step + '/')})


#%% Split hemispheres

split_coords = [182, 190, 200, None, 175]

setpro.SetSplitHems(
    marker_in_dpath,
    marker_dpaths['split'],
    split_coords=split_coords,
    process_nb=5
)

vis.SetPlotProj(
    marker_dpaths['split'], marker_dpaths['split'] + 'chan0_figs/',
    component=0,
)

#%% Segment posterior region

erase_dpath = marker_dpaths['posterior'] + 'erase/'

distrib_kwargs = {
    'coord': [100, 105, 80, 65, 80, 80, 75, 80, 95, 100]
    }

setpro.SetProcess(
    marker_dpaths['split'],
    erase_dpath,
    pro.EraseAreaImg,
    axis=1,
    direction=1,
    distrib_kwargs=distrib_kwargs,
    )

vis.SetPlotProj(
    erase_dpath,
    erase_dpath + 'chan0_figs/',
    component=0,
    legend=['KC'],
    norm='minmax',
)

clean_dpath = marker_dpaths['posterior'] + 'clean/'

setpro.SetSimpleProcess(
    erase_dpath,
    clean_dpath,
    process_func=pro.remove_small_objects_img,
    min_width=10,
    threshold='it',
    component=0,
    smoothing=4,
)

vis.SetPlotProj(
    clean_dpath,
    clean_dpath + 'chan0_figs/',
    component=0,
    norm='minmax',
)

#%% Segment whole MB

erase_dpath = marker_dpaths['whole'] + 'erase/'

distrib_kwargs = {
    'coord': [210, 210, 220, 220, 220, 220, 220, 220, 215, 215]
    }

setpro.SetProcess(
    marker_dpaths['split'],
    erase_dpath,
    pro.EraseAreaImg,
    axis=2,
    direction=1,
    distrib_kwargs=distrib_kwargs,
    )

vis.SetPlotProj(
    erase_dpath,
    erase_dpath + 'chan0_figs/',
    component=0,
    norm='minmax',
)

clean_dpath = marker_dpaths['split'] + 'clean/'

setpro.SetSimpleProcess(
    erase_dpath,
    clean_dpath,
    process_func=pro.remove_small_objects_img,
    min_width=50,
    threshold='li',
    component=0,
    smoothing=2,
)

vis.SetPlotProj(
    clean_dpath,
    clean_dpath + 'chan0_figs/',
    component=0,
    grid=True,
)

#%% Segment peduncle

clean_dpath = marker_dpaths['whole'] + 'clean/'
erase1_dpath = marker_dpaths['peduncle'] + 'erase1/'

distrib_kwargs = {
    'coord': [138, 140, 143, 138, 148, 138, 140, 140, 138, 133]
    }

setpro.SetProcess(
    clean_dpath,
    erase1_dpath,
    pro.EraseAreaImg,
    axis=2,
    direction=1,
    distrib_kwargs=distrib_kwargs,
    )

vis.SetPlotProj(
    erase1_dpath,
    erase1_dpath + 'chan0_figs/',
    component=0,
    norm='minmax',
    grid=True
)

erase2_dpath = marker_dpaths['peduncle'] + 'erase2/'

distrib_kwargs = {
    'coord': [110, 70, 125, 70, 135, 73, 118, 75, 100, 75],
    'direction': [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
    }

setpro.SetProcess(
    erase1_dpath,
    erase2_dpath,
    pro.EraseAreaImg,
    axis=0,
    distrib_kwargs=distrib_kwargs,
    )

vis.SetPlotProj(
    erase2_dpath,
    erase2_dpath + 'chan0_figs/',
    component=0,
    norm='minmax',
    grid=True,
)

erase3_dpath = marker_dpaths['peduncle'] + 'erase3/'

distrib_kwargs = {
    'coord': [165, 157, 140, 140, 145, 152, 140, 135, 150, 150]
    }

setpro.SetProcess(
    erase2_dpath,
    erase3_dpath,
    pro.EraseAreaImg,
    axis=1,
    direction=1,
    distrib_kwargs=distrib_kwargs,
    )

vis.SetPlotProj(
    erase3_dpath,
    erase3_dpath + 'chan0_figs/',
    component=0,
    norm='minmax',
    grid=True,
)

erase4_dpath = marker_dpaths['peduncle'] + 'erase4/'

distrib_kwargs = {
    'coord': [80, 80, 60, 60, 60, 70, 60, 60, 70, 70]
    }

setpro.SetProcess(
    erase3_dpath,
    erase4_dpath,
    pro.EraseAreaImg,
    axis=1,
    direction=-1,
    distrib_kwargs=distrib_kwargs,
    )

vis.SetPlotProj(
    erase4_dpath,
    erase4_dpath + 'chan0_figs/',
    component=0,
    norm='minmax',
)

clean_dpath = marker_dpaths['peduncle'] + 'clean/'

setpro.SetSimpleProcess(
    erase4_dpath,
    clean_dpath,
    process_func=pro.remove_small_objects_img,
    min_width=10,
    threshold='otsu',
    component=0,
    smoothing=2,
)

vis.SetPlotProj(
    clean_dpath,
    clean_dpath + 'chan0_figs/',
    component=0,
    norm='minmax',
)

#%% Segment vertical lobe

clean_dpath = marker_dpaths['whole'] + 'clean/'
erase1_dpath = marker_dpaths['vertical'] + 'erase1/'

distrib_kwargs = {
    'coord': [130, 130, 120, 110, 110, 120, 110, 110, 130, 130],
    }

setpro.SetProcess(
    clean_dpath,
    erase1_dpath,
    pro.EraseAreaImg,
    axis=1,
    direction=-1,
    distrib_kwargs=distrib_kwargs,
    )

vis.SetPlotProj(
    erase1_dpath,
    erase1_dpath + 'chan0_figs/',
    component=0,
    norm='minmax',
)

erase2_dpath = marker_dpaths['vertical'] + 'erase2/'

distrib_kwargs = {
    'coord': [185, 188, 170, 165, 173, 175, 160, 160, 180, 185],
    }

setpro.SetProcess(
    erase1_dpath,
    erase2_dpath,
    pro.EraseAreaImg,
    axis=1,
    direction=1,
    distrib_kwargs=distrib_kwargs,
    )

vis.SetPlotProj(
    erase2_dpath,
    erase2_dpath + 'chan0_figs/',
    component=0,
    norm='minmax',
)

erase3_dpath = marker_dpaths['vertical'] + 'erase3/'

distrib_kwargs = {
    'coord': [152, 148, 148, 148, 158, 150, 143, 140, 130, 132]
    }

setpro.SetProcess(
    erase2_dpath,
    erase3_dpath,
    pro.EraseAreaImg,
    axis=2,
    direction=-1,
    distrib_kwargs=distrib_kwargs,
    )

vis.SetPlotProj(
    erase3_dpath,
    erase3_dpath + 'chan0_figs/',
    component=0,
    norm='minmax',
)

clean_dpath = marker_dpaths['vertical'] + 'clean/'

setpro.SetSimpleProcess(
    erase3_dpath,
    clean_dpath,
    process_func=pro.remove_small_objects_img,
    min_width=None,
    threshold='otsu',
    component=0,
    smoothing=2,
)

vis.SetPlotProj(
    clean_dpath,
    clean_dpath + 'chan0_figs/',
    component=0,
    norm='minmax',
)

#%% Segment medial lobe

clean_dpath = marker_dpaths['whole'] + 'clean/'
erase1_dpath = marker_dpaths['medial'] + 'erase1/'

distrib_kwargs = {
    'coord': [162, 162, 138, 138, 145, 147, 135, 134, 150, 150]
}

setpro.SetProcess(
    clean_dpath,
    erase1_dpath,
    pro.EraseAreaImg,
    axis=1,
    direction=-1,
    distrib_kwargs=distrib_kwargs,
)

vis.SetPlotProj(
    erase1_dpath,
    erase1_dpath + 'chan0_figs/',
    component=0,
    norm='minmax',
)

erase2_dpath = marker_dpaths['medial'] + 'erase2/'

distrib_kwargs = {
    'coord': [148, 148, 146, 146, 158, 148, 143, 140, 128, 132]
    }

setpro.SetProcess(
    erase1_dpath,
    erase2_dpath,
    pro.EraseAreaImg,
    axis=2,
    direction=1,
    distrib_kwargs=distrib_kwargs,
)

vis.SetPlotProj(
    erase2_dpath,
    erase2_dpath + 'chan0_figs/',
    component=0,
    norm='minmax',
)

erase3_dpath = marker_dpaths['medial'] + 'erase3/'

distrib_kwargs = {
    'coord': [110, 80, 120, 70, 130, 80, 112, 80, 95, 90],
    'direction': [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
    }

setpro.SetProcess(
    erase2_dpath,
    erase3_dpath,
    pro.EraseAreaImg,
    axis=0,
    distrib_kwargs=distrib_kwargs,
    )

vis.SetPlotProj(
    erase3_dpath,
    erase3_dpath + 'chan0_figs/',
    component=0,
    norm='minmax',
)

clean_dpath = marker_dpaths['medial'] + 'clean/'

setpro.SetSimpleProcess(
    erase3_dpath,
    clean_dpath,
    process_func=pro.remove_small_objects_img,
    min_width=10,
    threshold='otsu',
    component=0,
    smoothing=2,

)

vis.SetPlotProj(
    clean_dpath,
    clean_dpath + 'chan0_figs/',
    component=0,
    norm='minmax',
)

#%% Merge for visualization

whole_dpath = marker_dpaths['whole'] + 'clean/'
posterior_dpath = marker_dpaths['posterior'] + 'clean/'
peduncle_dpath = marker_dpaths['peduncle'] + 'clean/'
vertical_dpath = marker_dpaths['vertical'] + 'clean/'
medial_dpath = marker_dpaths['medial'] + 'clean/'

setpro.SetMerge(
    [whole_dpath, posterior_dpath, peduncle_dpath, vertical_dpath, medial_dpath],
    marker_dpaths['merge_vis'],
    )

vis.SetPlotProj(
    marker_dpaths['merge_vis'],
    marker_dpaths['merge_vis'] + 'chans_figs/',
    norm='minmax',
    colors=par.SEG_COLORS,
    legend=['whole', 'somata+calyx', 'peduncle', 'vert. lobe', 'medial lobes'],
)

#%% Join hemispheres for visualization

setpro.SetJoinHems(
    marker_dpaths['merge_vis'],
    marker_dpaths['join_vis'],
    )

vis.SetPlotProj(
    marker_dpaths['join_vis'],
    marker_dpaths['join_vis'] + 'chans_figs/',
    norm='minmax',
    colors=par.SEG_COLORS,
    legend=['whole', 'somata+calyx', 'peduncle', 'vert. lobe', 'medial lobes'],
)

#%% Join hemispheres for each segmentation

for dpath in [whole_dpath, posterior_dpath, peduncle_dpath, vertical_dpath, medial_dpath]:

    out_dpath = dpath + 'join/'

    setpro.SetJoinHems(
        dpath,
        out_dpath,
        )

    vis.SetPlotProj(
        out_dpath,
        out_dpath + 'chan0_figs/',
        norm='minmax',
    )

#%% Mask FLIM

mask_dpaths = [
    posterior_dpath,
    peduncle_dpath,
    vertical_dpath,
    medial_dpath,
    ]

mask_dpaths = [p + 'join/' for p in mask_dpaths]

part_names = ['posterior', 'peduncle', 'vertical', 'medial']
part_nb = len(part_names)

for i_p in range(part_nb):

    setpro.SetMask(
        [flim_in_dpath, mask_dpaths[i_p]],
        flim_dpaths['mask'] + part_names[i_p] + '/',
    )

    vis.SetPlotProj(
        flim_dpaths['mask'] + part_names[i_p] + '/',
        flim_dpaths['mask'] + part_names[i_p] + '/' + 'sum_figs/',
        norm='minmax',
        mode='mean',
    )

    # compute IQR3 thresholding

    sum_dpath = flim_dpaths['mask'] + part_names[i_p] + '/' + 'sum/'
    threshold = 'IQR'
    factor = 3
    thresh_name = threshold + str(factor)

    # sum
    setpro.SetProcess(
        flim_dpaths['mask'] + part_names[i_p] + '/',
        sum_dpath,
        pro.SumChanImg,
        )

    # plot images
    vis.SetPlotProj(
        sum_dpath, sum_dpath + 'figs/',
        norm='minmax',
        mode='max',
        )

    # plot histograms
    hist_summary = vis.SetSummarizeHistograms(
        sum_dpath,
        sum_dpath + thresh_name + '/',
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
        sum_dpath, sum_dpath + thresh_name + '/' + 'figs/',
        norm=norms,
        mode='max',
        colors=plt.cm.gray.with_extremes(over='r'),
        )

    thresholds = np.array(hist_summary.thresholds)
    high_thresholds = thresholds[:, 1]

    # threshold high counts
    setpro.SetProcess(
        sum_dpath,
        sum_dpath + thresh_name + '/' + 'th_up/',
        pro.ThresholdImg,
        keep='lower',
        distrib_kwargs={'mode': high_thresholds},
        )

    restrict_dpath = flim_dpaths['mask'] + part_names[i_p] + '/' + thresh_name + '/'

    setpro.SetMask(
        [
            flim_dpaths['mask'] + part_names[i_p] + '/',
            sum_dpath + thresh_name + '/' + 'th_up/'
        ],
        restrict_dpath,
        process_nb=4,
        )

    vis.SetPlotProj(
        restrict_dpath, restrict_dpath + 'sum_figs/',
        norm='minmax',
        mode='mean',
        )

#%% Bin FLIM images

bin_process_nb = 3

for i_p in range(part_nb):

    RESTRICT_NAME = part_names[i_p]

    setpro.SetSimpleProcess(
        flim_dpaths['mask'] + RESTRICT_NAME + '/' + thresh_name + '/',
        flim_dpaths['bin'] + RESTRICT_NAME + '/' + thresh_name + '/' + bin_name + '/',
        process_func=pro.convolve_img,
        kernel=par.BIN_KERNEL,
        keep_mask=True,
        process_nb=bin_process_nb,
        )

    vis.SetPlotProj(
        flim_dpaths['bin'] + RESTRICT_NAME + '/' + thresh_name + '/' + bin_name + '/',
        flim_dpaths['bin'] + RESTRICT_NAME + '/' + thresh_name + '/' + bin_name + '/' + 'sum_figs/',
        norm='minmax',
        mode='mean',
        )

#%% Sum binned images

part_names = ['somas', 'peduncle', 'vertical lobe', 'medial lobes']
part_dnames = [s.replace(' ', '_') for s in part_names]

bin_dpaths = [
    flim_dpaths['bin'] + dname + '/' + thresh_name + '/' + bin_name + '/'
    for dname in part_dnames
    ]

setpro.SetSum(
    bin_dpaths,
    flim_dpaths['bin'] + 'bin_sum/' + thresh_name + '/' + bin_name + '/',
    process_nb=3,
)

vis.SetPlotProj(
    flim_dpaths['bin'] + 'bin_sum/' + thresh_name + '/' + bin_name + '/',
    flim_dpaths['bin'] + 'bin_sum/' + thresh_name + '/' + bin_name + '/' + 'sum_figs/',
    norm='minmax',
    mode='mean',
    )

#%% decimate 3

RESTRICT_NAME = 'bin_sum'

stride = (3, 3, 1)

setpro.SetSimpleProcess(
    flim_dpaths['bin'] + RESTRICT_NAME + '/' + thresh_name + '/' + bin_name + '/',
    flim_dpaths['decimate'] + RESTRICT_NAME + '/' + thresh_name + '/' + bin_name + '/',
    process_func=pro.decimate_img,
    stride=stride,
    )

vis.SetPlotProj(
    flim_dpaths['decimate'] + RESTRICT_NAME + '/' + thresh_name + '/' + bin_name + '/',
    flim_dpaths['decimate'] + RESTRICT_NAME + '/' + thresh_name + '/' + bin_name + '/' + 'sum_figs/',
    norm='minmax',
    mode='midslice',
    )
