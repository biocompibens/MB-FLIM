#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process markers_MB-KCc-KCn files

"""

import os

import mbflim.utils as ut
import mbflim.image.image_processing as pro
import mbflim.image.image_set_processing as setpro
import mbflim.visualization as vis

#%% Parameters

study = 'MB-KCc-STn' # 'MB-KCc-KCn' or 'MB-KCc-STn'

# Parallelism
PROCESS_NB = 16
MAX_THREAD_NB = 32

#%% Fixed parameters

imaging = 'markers'
analysis = 'preprocess'

# Marker names for visualizations
match study:
    case 'MB-KCc-KCn':
        markers = ['KCc', 'KCn']
    case 'MB-KCc-STn':
        markers = ['KCc', '${subtype}n']

# Load configuration file
config = ut.load_config()

# Uniform spacing in µm
spacing = config['params']['spacing']

# Crops (lower and upper margins from center of mass in µm for each axis)
tight_crop = config['params']['tight_crop']
crop = config['params']['crop']

# Intensity harmonization
BIN_NB = config['params']['bin_nb']
PT_NB = config['params']['pt_nb']

# Cleaning
MIN_WIDTH = None  # in µm or None to keep only largest element
CLEAN_THRESHOLD = "li"

#%% Prepare paths

input_dpath = os.path.join(config['paths']['source_dpath'], imaging, study + '/')
study_dpath = os.path.join(config['paths']['output_dpath'], imaging, study, analysis + '/')

# Processing steps
steps = [
    'resample',
    'clean',
    'crop1',
    'crop2',
    'harmonize_ch0',
    'mirror',
]

dpaths = {}
for i_s, step in enumerate(steps):
    dpaths.update({step: os.path.join(study_dpath, str(i_s) + '_' + step + '/')})


#%% Set global variables

# Parallelism
os.environ["MBFLIM_NUM_CPUS"] = str(PROCESS_NB)
os.environ["MBFLIM_NUM_THREADS"] = str(MAX_THREAD_NB)

# Temporary files
os.environ["TMPDIR"] = str(config['paths']['tmp_dpath'])

#%% Unify resolution

# Resample
setpro.SetProcess(
    input_dpath,
    dpaths['resample'],
    pro.ResampleImg,
    spacing=spacing,
)

# Visualize
vis.SetPlotProj(
    dpaths['resample'],
    dpaths['resample'] + "chans_figs/",
    legend=markers,
    norm="minmax",
)

#%% Clean

setpro.SetCleanBackground(
    dpaths['resample'],
    dpaths['clean'],
    min_width=MIN_WIDTH,
    threshold=CLEAN_THRESHOLD,
    component=0,
    mask_channels=True,
    smoothing=1,
)

vis.SetPlotProj(
    dpaths['clean'],
    dpaths['clean'] + "chans_figs/",
    legend=markers,
    norm="minmax",
)

#%% Crop and center

# Tight crop
setpro.SetCrop(
    dpaths['clean'],
    dpaths['crop1'],
    channel=0,
    margins=tight_crop,
)

# Visualize
vis.SetPlotProj(
    dpaths['crop1'],
    dpaths['crop1'] + "chans_figs/",
    legend=markers,
    norm="minmax",
)

# Crop
setpro.SetCrop(
    dpaths['crop1'],
    dpaths['crop2'],
    channel=0,
    margins=crop,
)

# Visualize
vis.SetPlotProj(
    dpaths['crop2'],
    dpaths['crop2'] + "chans_figs/",
    legend=markers,
    norm="minmax",
)

#%% Harmonize intensity histograms

# Harmonize
setpro.SetHarmonizeIntensity(
    dpaths['crop2'],
    dpaths['harmonize_ch0'],
    num_bins=BIN_NB,
    num_points=PT_NB,
    channel=0,
    plot_hist=True,
    threshold="mean",
)

# Visualize
vis.SetPlotProj(
    dpaths['harmonize_ch0'],
    dpaths['harmonize_ch0'] + "chans_figs/",
    legend=markers,
    norm="minmax",
)

# %% Mirror right MBs

# Mirror
setpro.SetMirror(
    dpaths['harmonize_ch0'],
    dpaths['mirror'],
    hemisphere="R",
    keep_original=True,
)

# Visualize
vis.SetPlotProj(
    dpaths['mirror'],
    dpaths['mirror'] + "chans_figs/",
    component=0,
    legend=["KC"],
    norm="minmax",
)
