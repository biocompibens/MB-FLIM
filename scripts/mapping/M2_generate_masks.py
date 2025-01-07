#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate template from 'MB-KCc-STn' files

"""

import os

import pandas as pd
import numpy as np

import mbflim.utils as ut
import mbflim.parameters as par
import mbflim.image.image_processing as pro
import mbflim.image.image_set_processing as setpro
import mbflim.visualization as vis

#%% Parameters

# Hemisphere
SEL_HEMISPHERES = 'both' # 'left', 'right' or 'both'

# Parallelism
PROCESS_NB = 16
MAX_THREAD_NB = 32

# Cleaning
MIN_WIDTH = None  # in µm or None to keep only largest element
CLEAN_THRESHOLD = "li"

#%% Fixed parameters

imaging = 'markers'
study = 'MB-KCc-KCn'
analysis = 'masks'

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
    max_thread_nb=MAX_THREAD_NB,
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

# Warp field display
WARP_DS = 20

# Colors
col = par.get_colors()

#%% Prepare paths

preprocess_dpath = os.path.join(
    config['paths']['output_dpath'],
    imaging,
    study,
    analysis,
    'mirror/',
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
    'harmonize_ch0',
    'register',
    'harmonize_ch1',
    'average',
    'somata_mask',
    'global_mask',
    'calyx_mask',
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

#%% Set global variables

# Parallelism
os.environ["MBFLIM_NUM_CPUS"] = str(PROCESS_NB)
os.environ["MBFLIM_NUM_THREADS"] = str(MAX_THREAD_NB)

# Temporary files
os.environ["TMPDIR"] = str(config['paths']['tmp_dpath'])

#%% Harmonize intensity histograms of channel 0

setpro.SetHarmonizeIntensity(
    [preprocess_dpath, template_dpath],
    dpaths["harmonize_ch0"],
    num_bins=BIN_NB,
    num_points=PT_NB,
    name_filter=[hem_name_filter, "*"],
    channel=0,
    plot_hist=True,
    threshold="mean",
)

vis.SetPlotProj(
    dpaths["harmonize_ch0"],
    dpaths["harmonize_ch0"] + "chan0_figs/",
    component=0,
    legend=["KCc"],
    norm="minmax",
)


# %% warp mask images to template
ut.print_time("Register AllMB images to template")

setpro.SetRegister(
    [dpaths["harmonize_ch0"], template_dpath],
    dpaths["register"],
    params=rig_aff_syn_tf_params,
    component=0,
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

vis.SetPlotProj(
    dpaths["register"],
    dpaths["register"] + "chans_figs/",
    warp_field=True,
    warp_downsampling=WARP_DS,
    legend=["KCc", 'KCn'],
    norm="minmax",
)

vis.plot_convergence_set(dpaths["register"], rig_aff_syn_tf_params, save=True)

#%% Harmonize intensity histograms of channel 1

setpro.SetHarmonizeIntensity(
    dpaths["register"],
    dpaths["harmonize_ch1"],
    num_bins=BIN_NB,
    num_points=PT_NB,
    channel=1,
    plot_hist=True,
    threshold="mean",
)

vis.SetPlotProj(
    dpaths["harmonize_ch1"],
    dpaths["harmonize_ch1"] + "chans_figs/",
    legend=["KC", "KCn"],
    norm="minmax",
)

#%% Average images
ut.print_time("Average warped AllMB images")

setpro.SetAverage(
    dpaths["harmonize_ch1"],
    dpaths["average"],
    channel=1,
)

vis.SetPlotProj(
    dpaths["average"],
    dpaths["average"] + "figs/",
    legend=["${subtype}"],
    norm='minmax',
    component=0,
)

#%% Compute somata mask
ut.print_time("Compute mask based on AllMB average")

setpro.SetComputeMask(
    dpaths["average"],
    dpaths["somata_mask"],
    smoothing_sigma=3.0,
)

vis.SetPlotProj(
    [dpaths["average"], dpaths["somata_mask"]],
    dpaths["somata_mask"] + "figs/",
    component=0,
    norm="minmax",
    colors=[col["blue"], 0.3 * col["red"]],
)

#%% Compute global mask

setpro.SetComputeMask(
    template_dpath,
    dpaths["global_mask"],
    smoothing_sigma=5.0, #3.0,
    method='multiotsu',
    min_width=None,
)

vis.SetPlotProj(
    [template_dpath, dpaths["global_mask"]],
    dpaths["global_mask"] + "figs/",
    component=0,
    norm="minmax",
    colors=[col["blue"], 0.3 * col["red"]],
)

#%% Compute calyx mask

s_mask = pro.read_img(pro.select_img(dpaths["somata_mask"], 0))
sc_mask = pro.read_img(pro.select_img(dpaths["global_mask"], 0))

s_array = s_mask.numpy()
sc_array = sc_mask.numpy()

c_mask = (sc_mask.numpy() > 0).astype(float) - (s_array > 0).astype(float)

c_mask = pro.remove_small_objects_stack(
    c_mask,
    min_width=None,
    spacing=sc_mask.spacing,
    smoothing=1.0,
    threshold='li',
)
c_mask = sc_mask.new_image_like(c_mask.astype(np.float32))

c_mask = (c_mask.numpy() > 0).astype(np.float32)

c_mask = sc_mask.new_image_like(c_mask)

name = 'calyx_mask'
fpath = 'calyx_mask.nii.gz'


os.makedirs(dpaths["calyx_mask"], exist_ok=True)
pro.write_img(c_mask, dpaths["calyx_mask"] + fpath)

info = pd.DataFrame({
    'name': ['calyx_mask'],
    'path': [fpath],
    'type': ['mask'],
    'hemisphere': [SEL_HEMISPHERES],
})

info.to_csv(dpaths["calyx_mask"] + 'files.csv')

vis.SetPlotProj(
    dpaths["calyx_mask"],
    dpaths["calyx_mask"] + 'figs/',
    norm='minmax',
)
