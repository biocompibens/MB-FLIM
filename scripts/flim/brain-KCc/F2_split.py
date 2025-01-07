#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split results according to segmentations

"""

import os

import numpy as np

import mbflim.utils as ut
import mbflim.visualization as vis
import mbflim.image.image_set_processing as setpro
import mbflim.parameters as par
import tools.flim.flim_set_processing as fl_setpro
import tools.flim.flim_visualization as fl_vis

#%% Parameters

# Parallelism
PROCESS_NB = 16
MAX_THREAD_NB = 32

#%% Fixed parameters

imaging = 'FLIM-marker'
study = 'brain-KCc'

# Binning
bin_name = 'fixed_kernel'

# Mask
RESTRICT_NAME = 'bin_sum/IQR3'

# Fitting
FIT_INIT_NB = 5
MIN_PHOTON_COUNT = 500
BOUND_INDICES = [2, -1]
model_class = 'ICPPModel'
method = 'MLE'
solver = 'Nelder-Mead'
exp_nb = 2

# Load configuration file
config = ut.load_config()

# Colors
col = par.get_colors()

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
irf_in_dpath = os.path.join(study_dpath, 'IRF/')

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
    'fit',
    'split',
]
flim_dpaths = {}
for i_s, step in enumerate(steps):
    flim_dpaths.update({step: os.path.join(flim_out_dpath, str(i_s) + '_' + step + '/')})

#%% Split fitting results according to segmentations

FLIM_DNAME = 'fit2_filter'
flim_dpath = flim_dpaths['fit'] + RESTRICT_NAME + '/' + bin_name + '/' \
    + FLIM_DNAME + '/'

# part masks
posterior_dpath = marker_dpaths['posterior'] + 'clean/'
peduncle_dpath = marker_dpaths['peduncle'] + 'peduncle_clean/'
vertical_dpath = marker_dpaths['vertical'] + 'vertical_clean/'
medial_dpath = marker_dpaths['medial'] + 'medial_clean/'

mask_dpaths = [
    posterior_dpath,
    peduncle_dpath,
    vertical_dpath,
    medial_dpath,
    ]

part_names = ['posterior', 'peduncle', 'vertical', 'medial']
part_nb = len(part_names)

norm_dict = {
    'alpha': (0.6, 0.9),
    }

for i_p in range(part_nb):

    out_dpath = flim_dpaths['split'] \
        + RESTRICT_NAME + '/' + bin_name + '/' + FLIM_DNAME + '/' + part_names[i_p] + '/'

    setpro.SetMask(
        [flim_dpath, mask_dpaths[i_p] + 'join_hems/'],
        out_dpath,
        )

    fl_vis.SetAnalyzeConditions(
        out_dpath,
        out_dpath + 'condition_analysis/',
        avg_mode='mean',
        asym_norm_distrib='Johnson-SU',
        fit_info_dpath=flim_dpath,
        )

    fl_vis.SetPlotFitParameters(
        out_dpath,
        out_dpath + '/param_figs/',
        norm_dict=norm_dict,
        mode='mean',
        params=['alpha'],
        fit_info_dpath=flim_dpath,
        )
