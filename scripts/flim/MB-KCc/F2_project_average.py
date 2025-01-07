#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project on template and average

"""

import os

import mbflim.utils as ut
import mbflim.visualization as vis
import mbflim.image.image_processing as pro
import mbflim.image.image_set_processing as setpro
import mbflim.flim.flim_visualization as fl_vis
import mbflim.parameters as par

#%% Parameters

# Parallelism
PROCESS_NB = 16
MAX_THREAD_NB = 32

# Mask
# Somata region: 'IQR3'
# Posterior region: 'tpl_IQR3'
# Calyx region: 'calyx_IQR3'
RESTRICT_NAME = 'tpl_IQR3'

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

# Warp field display
WARP_DS = 20

# Load configuration file
config = ut.load_config()

# Uniform spacing in µm
spacing = config['params']['spacing']

# Crop parameters
tight_margins = par.get_tight_margins()
wide_margins = par.get_wide_margins()
reference = 'center_of_mass'
in_pixel = False
extend = True

# Intensity harmonization
BIN_NB = config['params']['bin_nb']
PT_NB = config['params']['pt_nb']

# Hemisphere filtering
if SEL_HEMISPHERES == "left":
    hem_name_filter = ["*_L*"]
elif SEL_HEMISPHERES == "right":
    hem_name_filter = ["*_R*"]
elif SEL_HEMISPHERES == "both":
    hem_name_filter = ["*_L*", "*_mirR*"]

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
    'scale',
    'crop1',
    'crop2',
    'mirror'
]
marker_dpaths = {}
for i_s, step in enumerate(steps):
    marker_dpaths.update({step: os.path.join(marker_out_dpath, str(i_s) + '_' + step + '/')})
marker_in_dpath = marker_dpaths['clean'] + 'crop2/'

# FLIM processing steps
steps = [
    'mask',
    'bin',
    'fit',
    'scale',
    'crop1',
    'crop2',
    'mirror'
]
flim_dpaths = {}
for i_s, step in enumerate(steps):
    flim_dpaths.update({step: os.path.join(flim_out_dpath, str(i_s) + '_' + step + '/')})
flim_in_dpath = flim_dpaths['fit'] + 'fit2_filter/'

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
global_mask_dpath = os.path.join(
    config['paths']['output_dpath'],
    imaging,
    'MB-KCc-KCn',
    'masks',
    SEL_HEMISPHERES,
    '5_global_mask'  + '/',
)
calyx_mask_dpath = os.path.join(
    config['paths']['output_dpath'],
    imaging,
    'MB-KCc-KCn',
    'masks',
    SEL_HEMISPHERES,
    '6_calyx_mask'  + '/',
)

#%% Unify marker resolution
ut.print_time('Unify KC scale')

setpro.SetProcess(
    marker_in_dpath,
    marker_dpaths['scale'] + '/' + RESTRICT_NAME + '/',
    pro.ResampleImg,
    spacing=spacing,
    )

vis.SetPlotProj(
    marker_dpaths['scale'] + RESTRICT_NAME + '/',
    marker_dpaths['scale'] + RESTRICT_NAME + '/' + 'chan0_figs/',
    component=0,
    legend=['KCc'],
    norm='minmax',
)

#%% Unify FLIM resolution

setpro.SetProcess(
    flim_in_dpath,
    flim_dpaths['scale'] + RESTRICT_NAME + '/' + bin_name + '/',
    pro.ResampleImg,
    spacing=spacing,
    order=0, # nearest neighbor
    anti_aliasing=False,
)

#%% Crop marker images

# step 1

setpro.SetSimpleProcess(
    marker_dpaths['scale'] + RESTRICT_NAME + '/',
    marker_dpaths['crop1'] + RESTRICT_NAME + '/',
    process_func=pro.smartcrop_img,
    margins=tight_margins,
    in_pixel=in_pixel,
    reference=reference,
    extend=extend,
)

vis.SetPlotProj(
    marker_dpaths['crop1'] + RESTRICT_NAME + '/',
    marker_dpaths['crop1'] + RESTRICT_NAME + '/' + 'chan0_figs/',
    component=0,
    norm='minmax',
)

# step 2

setpro.SetSimpleProcess(
    marker_dpaths['crop1'] + RESTRICT_NAME + '/',
    marker_dpaths['crop2'] + RESTRICT_NAME + '/',
    process_func=pro.smartcrop_img,
    margins=wide_margins,
    in_pixel=in_pixel,
    reference=reference,
    extend=extend,
)

vis.SetPlotProj(
    marker_dpaths['crop2'] + RESTRICT_NAME + '/',
    marker_dpaths['crop2'] + RESTRICT_NAME + '/' + 'chan0_figs/',
    component=0,
    norm='minmax',
)

#%% crop FLIM images

# step 1
in_dpath = flim_dpaths['scale'] + RESTRICT_NAME + '/' + bin_name + '/'
out_dpath = flim_dpaths['crop1'] + RESTRICT_NAME + '/' + bin_name + '/'

if reference == 'center_of_mass':
    setpro.SetRefCrop(
        [in_dpath, marker_dpaths['scale'] + RESTRICT_NAME + '/'],
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
in_dpath = flim_dpaths['crop1'] + RESTRICT_NAME + '/' + bin_name + '/'
out_dpath = flim_dpaths['crop2'] + RESTRICT_NAME + '/' + bin_name + '/'

if reference == 'center_of_mass':
    setpro.SetRefCrop(
        [in_dpath, marker_dpaths['crop1'] + RESTRICT_NAME + '/'],
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

fl_vis.SetAnalyzeConditions(
    out_dpath,
    out_dpath + 'condition_analysis/',
    avg_mode='median',
    min_count=2000,
    fit_info_dpath=flim_in_dpath,
)

norm_dict = {
    'alpha': (0.3, 0.9),
}

fl_vis.SetPlotFitParameters(
    out_dpath,
    out_dpath + '/param_figs/',
    norm_dict=norm_dict,
    mode='mean',
    fit_info_dpath=flim_in_dpath,
    params=['alpha'],
)

#%% Mirror marker images

setpro.SetMirror(
    marker_dpaths['crop2'] + RESTRICT_NAME + '/',
    marker_dpaths['mirror'] + RESTRICT_NAME + '/',
    hemisphere='R',
    keep_original=True
)

vis.SetPlotProj(
    marker_dpaths['mirror'] + RESTRICT_NAME + '/',
    marker_dpaths['mirror'] + RESTRICT_NAME + '/' + 'chan0_figs/',
)

#%% Mirror FLIM images

setpro.SetMirror(
    flim_dpaths['crop2'] + RESTRICT_NAME + '/' + bin_name + '/',
    flim_dpaths['mirror'] + RESTRICT_NAME + '/' + bin_name + '/',
    hemisphere='R',
    keep_original=True
)

#%% Harmonize intensity histograms of marker images to template
ut.print_time("Harmonize to template")

setpro.SetHarmonizeIntensity(
    [marker_dpaths['mirror'] + RESTRICT_NAME + "/", template_dpath],
    marker_dpaths['harmonize'] + RESTRICT_NAME + "/" + SEL_HEMISPHERES + "/",
    num_bins=BIN_NB,
    num_points=PT_NB,
    name_filter=[hem_name_filter, "*"],
    plot_hist=True,
    threshold="mean",
)

vis.SetPlotProj(
    marker_dpaths['harmonize'] + RESTRICT_NAME + "/" + SEL_HEMISPHERES + "/",
    marker_dpaths['harmonize'] + RESTRICT_NAME + "/" + SEL_HEMISPHERES + "/" + "chan0_figs/",
    component=0,
    legend=["KCc"],
)

#%% Register marker images to template

setpro.SetRegister(
    [marker_dpaths['harmonize'] + RESTRICT_NAME + "/" + SEL_HEMISPHERES + "/", template_dpath],
    marker_dpaths['register'] + RESTRICT_NAME + "/" + SEL_HEMISPHERES + "/",
    name_filter=[hem_name_filter, "*"],
    prefix="tplwarp_",
    params=rig_aff_syn_tf_params,
    component=0,
)

vis.SetPlotProj(
    [marker_dpaths['register'] + RESTRICT_NAME + "/" + SEL_HEMISPHERES + "/", template_dpath],
    marker_dpaths['register'] + RESTRICT_NAME + "/" + SEL_HEMISPHERES + "/" + "ref_figs/",
    name_filter=["*", "*"],
    component=0,
    colors=[col["green"], col["gray"]],
    warp_field=True,
    warp_downsampling=WARP_DS,
    legend=["image", "template"],
)

vis.SetPlotProj(
    marker_dpaths['register'] + RESTRICT_NAME + "/" + SEL_HEMISPHERES + "/",
    marker_dpaths['register'] + RESTRICT_NAME + "/" + SEL_HEMISPHERES + "/" + "chan0_figs/",
    component=0,
    warp_field=True,
    warp_downsampling=WARP_DS,
)

vis.plot_convergence_set(
    marker_dpaths['register'] + RESTRICT_NAME + "/" + SEL_HEMISPHERES + "/",
    rig_aff_syn_tf_params,
    save=True,
)

#%% Apply registration to FLIM images

setpro.SetApplyRegistration(
    [
        flim_dpaths['mirror'] + RESTRICT_NAME + '/' + bin_name + '/',
        marker_dpaths['register'] + RESTRICT_NAME + "/" + SEL_HEMISPHERES + "/",
    ],
    flim_dpaths['register'] + RESTRICT_NAME + "/" + bin_name + '/' + SEL_HEMISPHERES + "/",
    name_filter=[hem_name_filter, hem_name_filter],
    interpolator="nearestNeighbor",
)

norm_dict = {
    "alpha": (0.60, 0.85),
}

fl_vis.SetPlotFitParameters(
    flim_dpaths['register'] + RESTRICT_NAME + "/" + bin_name + '/' + SEL_HEMISPHERES + "/",
    flim_dpaths['register'] + RESTRICT_NAME + "/" + bin_name + '/' + SEL_HEMISPHERES + "/" + "param_figs/",
    norm_dict=norm_dict,
    mode="mean",
    fit_info_dpath=flim_in_dpath,
    params=['alpha'],
)

#%% Average images

setpro.SetAverage(
    flim_dpaths['register'] + RESTRICT_NAME + "/" + bin_name + '/' + SEL_HEMISPHERES + "/",
    flim_dpaths['average'] + RESTRICT_NAME + "/" + bin_name + '/' + SEL_HEMISPHERES + "/",
    channel=None,
    nz_mask=True,
)

norm_dict = {
    "alpha": (0.60, 0.85),
}

fl_vis.SetPlotFitParameters(
    flim_dpaths['average'] + RESTRICT_NAME + "/" + bin_name + '/' + SEL_HEMISPHERES + "/",
    flim_dpaths['average'] + RESTRICT_NAME + "/" + bin_name + '/' + SEL_HEMISPHERES + "/" + "param_figs/",
    norm_dict=norm_dict,
    mode="mean",
    fit_info_dpath=flim_in_dpath,
)

#%% Mask average

ut.print_time("Apply mask")

setpro.SetMask(
   flim_dpaths['average'] + RESTRICT_NAME + "/" + bin_name + '/' + SEL_HEMISPHERES + "/",
   flim_dpaths['average'] + RESTRICT_NAME + "/" + bin_name + '/' + SEL_HEMISPHERES + "/" + 'tpl_mask/',
   mask_dpath=global_mask_dpath,
)

norm_dict = {
    "alpha": (0.65, 0.85),
}

fl_vis.SetPlotFitParameters(
    flim_dpaths['average'] + RESTRICT_NAME + "/" + bin_name + '/' + SEL_HEMISPHERES + "/" + 'tpl_mask/',
    flim_dpaths['average'] + RESTRICT_NAME + "/" + bin_name + '/' + SEL_HEMISPHERES + "/" + 'tpl_mask/' + "param_figs/",
    norm_dict=norm_dict,
    mode="mean",
    fit_info_dpath=flim_in_dpath,
)
