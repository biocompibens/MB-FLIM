#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
from skimage.morphology import disk


def get_registration_parameters(
        metrics,
        affine_iterations,
        syn_iterations,
        shrink_factors,
        smoothing_sigmas,
        thread_nb,
        syn_tf_params=(0.2, 3.0, 1.0,),
        ):

    # Rigid transform
    rig_tf_params = {
        "fixed_image": 'name.nii',
        "moving_image": 'name.nii',
        "output_transform_prefix": "rigid_",
        "transforms": ['Rigid'],
        "transform_parameters": [(0.2,)],
        "number_of_iterations": [affine_iterations],
        "shrink_factors":       [shrink_factors],
        "smoothing_sigmas":     [smoothing_sigmas],
        "dimension": 3,
        "write_composite_transform": False,
        "collapse_output_transforms": False,
        "initialize_transforms_per_stage": False,
        "metric": [metrics[0]['metric']],
        "metric_weight": [1], # Default (value currently ignored by ANTs)
        "num_threads": thread_nb,
        "radius_or_number_of_bins": [metrics[0]['radius_or_number_of_bins']],
        "sampling_strategy": [metrics[0]['sampling_strategy']],
        "sampling_percentage": [metrics[0]['sampling_percentage']],
        "convergence_threshold": [1.e-6],
        "convergence_window_size": [10],
        "sigma_units": ['vox'],
        "use_histogram_matching": [False],
        "verbose": True,
        "float": True,
        "initial_moving_transform_com": True,
        }

    # Affine transform
    aff_tf_params = {
        "fixed_image": 'name.nii',
        "moving_image": 'name.nii',
        "output_transform_prefix": "affine_",
        "transforms": ['Affine'],
        "transform_parameters": [(0.2,)], # Gradient step
        "number_of_iterations": [affine_iterations],
        "shrink_factors":       [shrink_factors],
        "smoothing_sigmas":     [smoothing_sigmas],
        "dimension": 3,
        "write_composite_transform": False,
        "collapse_output_transforms": False,
        "initialize_transforms_per_stage": False,
        "metric": [metrics[0]['metric']],
        "metric_weight": [1], # Default (value currently ignored by ANTs)
        "num_threads": thread_nb,
        "radius_or_number_of_bins": [metrics[0]['radius_or_number_of_bins']],
        "sampling_strategy": [metrics[0]['sampling_strategy']],
        "sampling_percentage": [metrics[0]['sampling_percentage']],
        "convergence_threshold": [1.e-6],
        "convergence_window_size": [10],
        "sigma_units": ['vox'],
        "use_histogram_matching": [False],
        "verbose": True,
        "float": True,
        "initial_moving_transform_com": False,
        }

    # SyN transform
    # 'transform_parameters' for SyN
    # Param 0: Gradient step for the deformation update
    # Param 1: Variance of Gaussian kernel to smooth deformation update (fluid model)
    # Param 2: Variance of Gaussian kernel to smooth total deformation (elastic model)
    syn_tf_parameters = {
        "fixed_image": 'name.nii',
        "moving_image": 'name.nii',
        "output_transform_prefix": "syn_",
        "output_warped_image": 'test_output.nii',
        "transforms": ['SyN'],
        "transform_parameters": [(0.2, 3.0, 1.0)],
        "number_of_iterations": [syn_iterations],
        "shrink_factors":       [shrink_factors],
        "smoothing_sigmas":     [smoothing_sigmas],
        "dimension": 3,
        "write_composite_transform": False,
        "collapse_output_transforms": False,
        "initialize_transforms_per_stage": False,
        "metric": [metrics[1]['metric']],
        "metric_weight": [1], # Default (value ignored currently by ANTs)
        "num_threads": thread_nb,
        "radius_or_number_of_bins": [metrics[1]['radius_or_number_of_bins']],
        "sampling_strategy": [metrics[1]['sampling_strategy']],
        "sampling_percentage": [metrics[1]['sampling_percentage']],
        "convergence_threshold": [1.e-6],
        "convergence_window_size": [10],
        "sigma_units": ['vox'],
        "use_histogram_matching": [False],
        "verbose": True,
        "float": True,
        "initial_moving_transform_com": False,
        }

    # Rigid + affine + SyN transform
    rig_aff_syn_tf_params = {
        "fixed_image": 'name.nii',
        "moving_image": 'name.nii',
        "output_transform_prefix": "rig_aff_syn_",
        "transforms": ['Rigid', 'Affine', 'SyN'],
        "transform_parameters": (
            rig_tf_params['transform_parameters']
            + aff_tf_params['transform_parameters']
            + syn_tf_parameters['transform_parameters']
            ),
        "number_of_iterations": (
            rig_tf_params['number_of_iterations']
            + aff_tf_params['number_of_iterations']
            + syn_tf_parameters['number_of_iterations']
            ),
        "dimension": 3,
        "write_composite_transform": False,
        "collapse_output_transforms": True,
        "initialize_transforms_per_stage": False,
        "metric": (
            rig_tf_params['metric']
            + aff_tf_params['metric']
            + syn_tf_parameters['metric']
            ),
        "metric_weight": [1.0]*3, # Default (value ignored currently by ANTs)
        "num_threads": thread_nb,
        "radius_or_number_of_bins": (
            rig_tf_params['radius_or_number_of_bins']
            + aff_tf_params['radius_or_number_of_bins']
            + syn_tf_parameters['radius_or_number_of_bins']
            ),
        "sampling_strategy": (
            rig_tf_params['sampling_strategy']
            + aff_tf_params['sampling_strategy']
            + syn_tf_parameters['sampling_strategy']
            ),
        "sampling_percentage": (
            rig_tf_params['sampling_percentage']
            + aff_tf_params['sampling_percentage']
            + syn_tf_parameters['sampling_percentage']
            ),
        "convergence_threshold": (
            rig_tf_params['convergence_threshold']
            + aff_tf_params['convergence_threshold']
            + syn_tf_parameters['convergence_threshold']
            ),
        "convergence_window_size": (
            rig_tf_params['convergence_window_size']
            + aff_tf_params['convergence_window_size']
            + syn_tf_parameters['convergence_window_size']),
        "smoothing_sigmas": (
            rig_tf_params['smoothing_sigmas']
            + aff_tf_params['smoothing_sigmas']
            + syn_tf_parameters['smoothing_sigmas']
            ),
        "sigma_units": (
            rig_tf_params['sigma_units']
            + aff_tf_params['sigma_units']
            + syn_tf_parameters['sigma_units']
            ),
        "shrink_factors": (
            rig_tf_params['shrink_factors']
            + aff_tf_params['shrink_factors']
            + syn_tf_parameters['shrink_factors']
            ),
        "use_histogram_matching": [False]*3,
        "verbose": True,
        "float": True,
        "initial_moving_transform_com": True,
        }

    parameters = {
        'rig_tf_params': rig_tf_params,
        'aff_tf_params': aff_tf_params,
        'syn_tf_parameters': syn_tf_parameters,
        'rig_aff_syn_tf_params': rig_aff_syn_tf_params,
    }

    return parameters


def get_registration_metrics(cc_radius=3):

    mi_metric = {
        "metric": "Mattes",
        "radius_or_number_of_bins": 32,
        "sampling_strategy": "None",
        "sampling_percentage": 1.0,
        "dimension": 3,
    }

    cc_metric = {
        "metric": "CC",
        "radius_or_number_of_bins": cc_radius,
        "sampling_strategy": "None",
        "sampling_percentage": 1.0,
        "dimension": 3,
    }

    gc_metric = {
        "metric": "GC",
        "radius_or_number_of_bins": 0,
        "sampling_strategy": "None",
        "sampling_percentage": 1.0,
        "dimension": 3,
    }

    mse_metric = {
        "metric": "MeanSquares",
        "radius_or_number_of_bins": 0,
        "sampling_strategy": "None",
        "sampling_percentage": 1.0,
        "dimension": 3,
    }

    drmse_metric = {"metric": "densityRMSE"}

    metrics = {
        'mi_metric': mi_metric,
        'cc_metric': cc_metric,
        'gc_metric': gc_metric,
        'mse_metric': mse_metric,
        'drmse_metric': drmse_metric,
    }

    return metrics


def get_colors():

    colors = {
        'white': np.ones((3,)),
        'red': np.array([1.0, 0.0, 0.0]),
        'green': np.array([0.0, 1.0, 0.0]),
        'blue': np.array([0.0, 0.0, 1.0]),
        'purple': np.array([0.8, 0.0, 1.0]),
        'gray': 0.5 * np.ones((3,)),
        'black': np.zeros((3,))
        }

    return colors


def get_type_list():
    return ['AB', 'ApBp', 'G']


def get_neuron_names():
    return {
        'AB': r'$\mathrm{\alpha/\beta}$',
        'ApBp': r"$\mathrm{\alpha'/\beta'}$",
        'G': r'$\mathrm{\gamma}$',
        'AB+ApBp': r"$\alpha\beta+\alpha'\beta'$",
        'ApBp-constant': r"uniform $\alpha'\beta'$",
        'C0': 'Unpaired',
        'C1': 'Paired',
        }


def get_neuron_nbs():
    return {'AB': 1002, 'ApBp': 370, 'G': 671} #{'AB': 990, 'ApBp': 350, 'G': 675}


def get_neuron_ratios():
    neuron_nbs = get_neuron_nbs()
    types = list(neuron_nbs.keys())
    nbs = list(neuron_nbs.values())

    ratios = [float(nb) / float(sum(nbs)) for nb in nbs]

    zip_it = zip(types, ratios)

    return dict(zip_it)


def get_sp_colors():

    colors = get_colors()

    color_dict = {
        'AB': colors['red'],
        'ApBp': colors['green'],
        'G': colors['blue'],
        'KC': colors['black'],
        }

    return color_dict


def rgb_tab_color(tab_color):

    return matplotlib.colors.to_rgb(mcolors.TABLEAU_COLORS[tab_color])


def get_soft_sp_colors():

    colors = get_colors()

    color_dict = {
        'AB': np.array([250, 77, 86]) / 255, #rgb_tab_color('tab:red'),
        'ApBp': np.array([36, 161, 72]) / 255, #rgb_tab_color('tab:green'),
        'ApBp-constant': rgb_tab_color('tab:green'),
        'G': np.array([69, 137, 255])/255, #rgb_tab_color('tab:blue'),
        'KC': colors['black'],
        'somas': np.array([165, 110, 255]) / 255,
        'calyx': np.array([0, 157, 154]) / 255,
    }

    return color_dict


SEG_COLORS = [
    mcolors.to_rgb(mcolors.CSS4_COLORS['lightgrey']),
    np.array((255, 196, 0)) / 255, #np.array((0, 114, 195)) / 255,
    np.array((25, 128, 56)) / 255,
    np.array((61, 219, 217)) / 255, #np.array((0, 125, 121)) / 255,
    np.array((15, 98, 254)) / 255,

    ]

SEG_PART_COLORS = SEG_COLORS[1:]


kernels_2d = {
    '0': np.ones((1, 1, 1)),
    '1': np.expand_dims(disk(radius=1), -1),
    '1.5': np.ones((3, 3, 1)),
    '2': np.expand_dims(disk(radius=2), -1),
    }


BIN_KERNEL = np.array(
    [[[False],
     [ True],
     [ True],
     [ True],
     [False]],
    [[ True],
     [ True],
     [ True],
     [ True],
     [ True]],
    [[ True],
     [ True],
     [ True],
     [ True],
     [ True]],
    [[ True],
     [ True],
     [ True],
     [ True],
     [ True]],
    [[False],
     [ True],
     [ True],
     [ True],
     [False]]]
)


