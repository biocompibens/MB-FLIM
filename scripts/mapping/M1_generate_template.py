#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate template from 'MB-KCc-STn' files

"""

import os
import glob

import mbflim.utils as ut
import mbflim.parameters as par
import mbflim.image.image_set_processing as setpro
import mbflim.visualization as vis

#%% Parameters

# Hemisphere
SEL_HEMISPHERES = 'both' # 'left', 'right' or 'both'

# Parallelism
PROCESS_NB = 20
MAX_THREAD_NB = 40

#%% Fixed parameters

imaging = 'markers'
study = 'MB-KCc-STn'
analysis = 'template'

# Template
TPL_IT_NB = 8
TPL_AVG_MODE = "mean"
TPL_UPDATE_NB = 1
GRAD_STEP = 0.2
BLENDING_WEIGHT = 0.75

# Cleaning
MIN_WIDTH = None  # in µm or None to keep only largest element
CLEAN_THRESHOLD = "li"

# Load configuration file
config = ut.load_config()

# Uniform spacing in µm
spacing = config['params']['spacing']

# Crops (lower and upper margins from center of mass in µm for each axis)
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

# Template registration parameters
tpl_reg_params = rig_aff_syn_tf_params.copy()
tpl_reg_params["use_histogram_matching"] = True

# Hemisphere filtering
if SEL_HEMISPHERES == "left":
    hem_name_filter = ["*_L*"]
elif SEL_HEMISPHERES == "right":
    hem_name_filter = ["*_R*"]
elif SEL_HEMISPHERES == "both":
    hem_name_filter = ["*_L*", "*_mirR*"]

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
    'template',
    'template_clean',
    'template_center',
    'template_mirror',
    'template_align',
    'template_symmetrize',
]

dpaths = {}
for i_s, step in enumerate(steps):
    dpaths.update({step: os.path.join(analysis_dpath, str(i_s) + '_' + step + '/')})

# Prepare left hemisphere path
left_dpath = os.path.join(
    config['paths']['output_dpath'],
    imaging,
    study,
    analysis,
    'left' + '/',
)
left_dpaths = {}
for i_s, step in enumerate(steps):
    left_dpaths.update({step: os.path.join(left_dpath, str(i_s) + '_' + step + '/')})


#%% Set global variables

# Parallelism
os.environ["MBFLIM_NUM_CPUS"] = str(PROCESS_NB)
os.environ["MBFLIM_NUM_THREADS"] = str(MAX_THREAD_NB)

# Temporary files
os.environ["TMPDIR"] = str(config['paths']['tmp_dpath'])

# %% create template (affine+SyN)

tpl_reg_params = rig_aff_syn_tf_params.copy()
tpl_reg_params["use_histogram_matching"] = True

setpro.SetBuildTemplate(
    preprocess_dpath,
    dpaths['template'],
    name_filter=hem_name_filter,
    params=rig_aff_syn_tf_params,
    iterations=TPL_IT_NB,
    gradient_step=GRAD_STEP,
    blending_weight=BLENDING_WEIGHT,
    initial_template=None,
    update_nb=TPL_UPDATE_NB,
    component=0,
    avg_mode=TPL_AVG_MODE,
)

vis.SetPlotProj(
    dpaths["template"],
    dpaths["template"] + "figs/",
    norm="minmax",
)

vis.plot_template_convergence(
    dpaths["template"] + "tpl_convergence_data.json",
    fig_fpath=dpaths["template"] + "figs/tpl_convergence.png",
)

#%% Clean template

setpro.SetCleanBackground(
    dpaths["template"],
    dpaths["template_clean"],
    min_width=MIN_WIDTH,
    threshold=CLEAN_THRESHOLD,
    component=0,
    mask_channels=True,
    smoothing=1,
    name_filter="*best*",
)

vis.SetPlotProj(
    dpaths["template_clean"],
    dpaths["template_clean"] + "figs/",
    norm="minmax",
)

#%% Center template

setpro.SetCrop(dpaths["template_clean"], dpaths["template_center"], margins=crop)

vis.SetPlotProj(
    dpaths["template_center"],
    dpaths["template_center"] + "figs/",
    norm="minmax",
)

#%% Symmetrize right template

if SEL_HEMISPHERES == "right":

    left_tpl_dpath = left_dpaths["template_center"]
    left_tpl_fpath = glob.glob(left_tpl_dpath + "*best*")

    if len(left_tpl_fpath) == 0:
        raise ValueError("Left template was not found.")

    # should be a single file
    left_tpl_fpath = left_tpl_fpath[0]

    # Mirror
    setpro.SetMirror(
        dpaths["template_center"],
        dpaths["template_mirror"],
        prefix="mir_",
        hemisphere="T",
    )

    vis.SetPlotProj(
        dpaths["template_mirror"],
        dpaths["template_mirror"] + "figs/",
        norm="minmax",
    )

    # Align
    setpro.SetRegister(
        [dpaths["template_mirror"], left_tpl_dpath],
        dpaths["template_align"],
        name_filter=["*", "*best*"],
        params=rig_tf_params,
        component=0,
    )

    vis.SetPlotProj(
        dpaths["template_align"],
        dpaths["template_align"] + "figs/",
        norm="minmax",
    )

    # Mirror back
    setpro.SetMirror(
        dpaths["template_align"],
        dpaths["template_symmetrize"],
        hemisphere="mirT",
    )

    vis.SetPlotProj(
        dpaths["template_symmetrize"],
        dpaths["template_symmetrize"] + "figs/",
        norm="minmax",
    )

