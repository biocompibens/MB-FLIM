#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate template from 'MB-KCc-STn' files

"""

import os

import numpy as np
import glob

import mbflim.utils as ut
import mbflim.parameters as par
import mbflim.image.image_set_processing as setpro
import mbflim.visualization as vis

#%% Parameters

# Hemisphere
SEL_HEMISPHERES = 'both' # 'left', 'right' or 'both'

# Parallelism
PROCESS_NB = 16
MAX_THREAD_NB = 32

#%% Fixed parameters

imaging = 'markers'
study = 'MB-KCc-STn'
analysis = 'map_subtypes'

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
    thread_nb=MAX_THREAD_NB,
)
rig_tf_params = registration_parameters['rig_tf_params']
rig_aff_syn_tf_params = registration_parameters['rig_aff_syn_tf_params']

# Hemisphere filtering
if SEL_HEMISPHERES == "left":
    hem_name_filter = ["*_L*"]
elif SEL_HEMISPHERES == "right":
    hem_name_filter = ["*_R*"]
elif SEL_HEMISPHERES == "both":
    hem_name_filter = ["*_L*", "*_mirR*"]

# Warp field display
WARP_DS = 20

# Cleaning
MIN_WIDTH = None  # in µm or None to keep only largest element
CLEAN_THRESHOLD = "li"

# Colors
col = par.get_colors()

#%% Prepare paths

preprocess_dpath = os.path.join(
    config['paths']['output_dpath'],
    imaging,
    study,
    'preprocess',
    '5_mirror/',
)
analysis_dpath = os.path.join(
    config['paths']['output_dpath'],
    imaging,
    study,
    analysis,
    SEL_HEMISPHERES + '/',
)

# Processing steps
steps = [
    'harmonize',
    'register',
    'mask',
    'normalize',
    'average',
]

dpaths = {}
for i_s, step in enumerate(steps):
    dpaths.update({step: os.path.join(analysis_dpath, str(i_s) + '_' + step + '/')})

# Template path
template_dpath = os.path.join(
    config['paths']['output_dpath'],
    imaging,
    'MB-KCc-STn',
    'template',
    SEL_HEMISPHERES,
    '2_template_center'  + '/',
)

# Mask path
mask_dpath = os.path.join(
    config['paths']['output_dpath'],
    imaging,
    'MB-KCc-KCn',
    'masks',
    SEL_HEMISPHERES,
    '4_somata_mask'  + '/',
)

#%% Set global variables

# Parallelism
os.environ["MBFLIM_NUM_CPUS"] = str(PROCESS_NB)
os.environ["MBFLIM_NUM_THREADS"] = str(MAX_THREAD_NB)

# Temporary files
os.environ["TMPDIR"] = str(config['paths']['tmp_dpath'])

#%% Harmonize intensity histograms

setpro.SetHarmonizeIntensity(
    [preprocess_dpath, template_dpath],
    dpaths["harmonize"],
    num_bins=BIN_NB,
    num_points=PT_NB,
    name_filter=[hem_name_filter, "*"],
    channel=0,
    plot_hist=True,
    threshold="mean",
)

vis.SetPlotProj(
    dpaths["harmonize"],
    dpaths["harmonize"] + "chan0_figs/",
    component=0,
    legend=["KCc"],
    norm="minmax",
)

#%% Register images to template

setpro.SetRegister(
    [dpaths["harmonize"], template_dpath],
    dpaths["register"],
    name_filter=[hem_name_filter, "*"],
    params=rig_aff_syn_tf_params,
    component=0,
    keep_inv_tfs=True,
)

vis.SetPlotProj(
    [dpaths["register"], template_dpath],
    dpaths["register"] + "ref_figs/",
    warp_field=True,
    component=0,
    colors=[col["green"], col["gray"]],
    warp_downsampling=WARP_DS,
    legend=["image", "template"],
    norm="minmax",
)

vis.plot_convergence_set(dpaths["register"], rig_aff_syn_tf_params, save=True)

#%% Mask images

setpro.SetMask(
    dpaths["register"],
    dpaths["mask"],
    mask_dpath=mask_dpath,
    mask_ratio=True,
)

vis.SetPlotProj(
    dpaths["mask"],
    dpaths["mask"] + "chans_figs/",
    legend=['KC', '${Subtype}n'],
    norm="minmax",
)

#%% Harmonize intensity histograms of channel 1

setpro.SetHarmonizeIntensity(
    dpaths["mask"],
    dpaths["harmonize_ch1"],
    num_bins=BIN_NB,
    num_points=PT_NB,
    channel=1,
    plot_hist=True,
    by_type=True,
    threshold="mean",
)

vis.SetPlotProj(
    dpaths["harmonize_ch1"],
    dpaths["harmonize_ch1"] + "chans_figs/",
    legend=['KC', '${Subtype}n'],
    norm="minmax",
)

#%% Normalize channel 1 to neuron density

setpro.SetNormalize(
    dpaths["harmonize_ch1"],
    dpaths["normalize"],
    mode=np.sum,
    value="unit",
    component=1,
)

vis.SetPlotProj(
    dpaths["normalize"],
    dpaths["normalize"] + "chan1_figs/",
    component=1,
    legend=['${Subtype}n'],
    norm="minmax",
)

#%% Average images

setpro.SetTypeAverage(
    dpaths["normalize"],
    dpaths["average"],
    component=1,
    mask=glob.glob(mask_dpath, '*.nii.gz')[0],
    primes_corr=None,
)

vis.SetPlotProj(
    dpaths["average"],
    dpaths["average"] + "chans_figs/",
    name_filter="*maps*",
    mode='midlsice',
    legend=["AB", "ApBp", "G"],
    norm='minmax',
)

