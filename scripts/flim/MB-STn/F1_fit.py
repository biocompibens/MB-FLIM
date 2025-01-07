#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit FLIM data

"""

import os

import mbflim.utils as ut
import mbflim.parameters as par
import tools.flim.flim_set_processing as fl_setpro
import tools.flim.flim_visualization as fl_vis

#%% Parameters

# Parallelism
PROCESS_NB = 16
MAX_THREAD_NB = 32

#%% Fixed parameters

imaging = 'FLIM-marker'
study = 'MB-STn'

# Binning
bin_name = 'fixed_kernel'

# Mask
RESTRICT_NAME = 'IQR3'

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

# Cleaning
MIN_WIDTH = None  # in µm or None to keep only largest element
CLEAN_THRESHOLD = "li"

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


#%% First fit

exp_param_names = []
other_param_names = ['tau_2', 'alpha', 'irf_shift', 'bg_noise']
prev_param_names = []

fixed_parameters = {'tau_1': 0.4e-9}

in_dpath = flim_dpaths['decimate'] + RESTRICT_NAME + '/' + bin_name + '/'
prev_dpath = []
out_dpath = flim_dpaths['fit'] + RESTRICT_NAME + '/' + bin_name + '/' + 'fit1/'

fl_setpro.SetFitDecay(
    in_dpath,
    out_dpath,
    irf_dpath=irf_in_dpath,
    calibration_dpath=None,
    prev_fit_dpath=None,
    bound_indices=BOUND_INDICES,
    exp_nb=exp_nb,
    inits=FIT_INIT_NB,
    solver=solver,
    method=method,
    count_threshold=MIN_PHOTON_COUNT,
    model_class=model_class,
    exp_param_names=exp_param_names,
    other_param_names=other_param_names,
    prev_param_names=prev_param_names,
    fixed_parameters=fixed_parameters,
    irf_merge='subject',
    )

fl_vis.SetAnalyzeConditions(
    out_dpath,
    out_dpath + 'condition_analysis/',
    avg_mode='mean',
    )

fl_vis.SetPlotTypicalFits(
    [in_dpath, out_dpath, out_dpath + 'fit_resp/'],
    out_dpath + 'typical_fits/',
    )

norm_dict = {
    'alpha': (0.25, 1.0),
    }

fl_vis.SetPlotFitParameters(
    out_dpath,
    out_dpath + '/param_figs/',
    norm_dict=norm_dict,
    mode='mean',
    params=['alpha']
    )

#%% Filter first fit

in_dpath = flim_dpaths['fit'] + RESTRICT_NAME + '/' + bin_name + '/' + 'fit1/'
out_dpath = flim_dpaths['fit'] + RESTRICT_NAME + '/' + bin_name + '/' + 'fit1_filter/'

param_lims = {
    'alpha': (0.25, 1.0),
    'cost': (0.0, 200),
}

fl_setpro.SetFilterParameters(
    in_dpath,
    out_dpath,
    param_lims=param_lims,
    )

fl_vis.SetAnalyzeConditions(
    out_dpath,
    out_dpath + 'condition_analysis/',
    avg_mode='mean',
    asym_norm_distrib='Skew normal',
    )

norm_dict = {
    'alpha': (0.6, 0.85),
    }

fl_vis.SetPlotFitParameters(
    out_dpath,
    out_dpath + '/param_figs/',
    norm_dict=norm_dict,
    mode='mean',
    params=['alpha'],
    )

#%% Second fit

exp_param_names = []
other_param_names = ['tau_2', 'alpha', 'bg_noise']
prev_param_names = ['irf_shift']

fixed_parameters = {'tau_1': 0.4e-9}

in_dpath = flim_dpaths['bin'] + RESTRICT_NAME + '/' + bin_name + '/'
prev_dpath = flim_dpaths['fit'] + RESTRICT_NAME + '/' + bin_name + '/' + 'fit1_filter/'
out_dpath = flim_dpaths['fit'] + RESTRICT_NAME + '/' + bin_name + '/' + 'fit2/'

fl_setpro.SetFitDecay(
    in_dpath,
    out_dpath,
    irf_dpath=irf_in_dpath,
    calibration_dpath=None,
    prev_fit_dpath=prev_dpath,
    bound_indices=BOUND_INDICES,
    exp_nb=exp_nb,
    inits=FIT_INIT_NB,
    solver=solver,
    method=method,
    count_threshold=MIN_PHOTON_COUNT,
    model_class=model_class,
    exp_param_names=exp_param_names,
    other_param_names=other_param_names,
    prev_param_names=prev_param_names,
    fixed_parameters=fixed_parameters,
    irf_merge='subject',
    )

fl_vis.SetAnalyzeConditions(
    out_dpath,
    out_dpath + 'condition_analysis/',
    avg_mode='mean',
    remove_outliers=False,
    )

fl_vis.SetPlotTypicalFits(
    [in_dpath, out_dpath, out_dpath + 'fit_resp/'],
    out_dpath + 'typical_fits/',
    )

norm_dict = {
    'tau_2': (2e-9, 8e-9),
    'irf_shift': (-3e-10, 3e-10),
    'bg_noise': (0.0, 3e-2),
    'rel_amp_1': (0.0, 1.0),
    'rel_amp_2': (0.0, 1.0),
    'mean_tau': (0.0, 3e-9),
    'cost': 'minmax',
    'count': 'minmax',
    'alpha': (0.60, 0.85),
    }

fl_vis.SetPlotFitParameters(
    out_dpath,
    out_dpath + '/param_figs/',
    norm_dict=norm_dict,
    mode='midslice',
    )

#%% Filter second fit

in_dpath = flim_dpaths['fit'] + RESTRICT_NAME + '/' + bin_name + '/' + 'fit2/'
out_dpath = flim_dpaths['fit'] + RESTRICT_NAME + '/' + bin_name + '/' + 'fit2_filter/'

param_lims = {
    'alpha': (0.25, 1.0),
    'cost': (0.0, 200),
}

fl_setpro.SetFilterParameters(
    in_dpath,
    out_dpath,
    param_lims=param_lims,
    )

fl_vis.SetAnalyzeConditions(
    out_dpath,
    out_dpath + 'condition_analysis/',
    avg_mode='mean',
    asym_norm_distrib='Johnson-SU',
    )

norm_dict = {
    'alpha': (0.6, 0.9),
    }

fl_vis.SetPlotFitParameters(
    out_dpath,
    out_dpath + '/param_figs/',
    norm_dict=norm_dict,
    mode='midslice',
    params=['alpha'],
    )