#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 17:08:36 2022

@author: proussel
"""

import os
import re
from itertools import repeat
import itertools
from pathlib import Path
import warnings
import json
import glob

import numpy as np
import pandas as pd
from skimage import measure
from scipy.interpolate import interp1d
import scipy
import skimage
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from statannotations.Annotator import Annotator
import ants

import mbflim.shared as sha
import mbflim.image.image_processing as pro
import mbflim.parameters as par


class SavePlot(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, save_fpath, *args, **kwargs):

        output = self.func(*args, **kwargs)

        if save_fpath:
            save_dpath = Path(save_fpath).parent
            os.makedirs(save_dpath, exist_ok=True)
            plt.savefig(save_fpath, bbox_inches="tight")
            plt.close()

        return output


# Simple set visualization class
class SetSimpleVisualization(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = True
    MP = True
    EXT = ".png"

    def process_set(self, vis_func, *args, **kwargs):

        repeat_args = [repeat(arg) for arg in args]

        save_plot_func = SavePlot(vis_func)

        args_iter = zip(self.out_fpaths, self.in_fpaths, *repeat_args)

        self.process_tasks(
            args_iter=args_iter,
            function=save_plot_func,
            task_nb=self.in_img_nb,
            description="Process",
            kwargs_dict=kwargs,
        )


# Set plotting class
class SetPlotProj(sha.AbstractSetProcess):

    NEED_TMP = False
    PROCESS_NAME = "plot-proj"
    EXT = ".png"
    MP = True

    def process_set(
        self,
        mode=None,
        component=None,
        similarity=None,
        warp_field=False,
        warp_downsampling=1,
        colors=None,
        threshold=None,
        center_of_mass=False,
        legend=None,
        norm='minmax',
        white_zeros=False,
        **kwargs,
    ):

        if self.input_nb > 1:
            if self.in_img_nb[1] == 1:
                imgs_L = [
                    [self.in_fpaths[0][i], self.in_fpaths[1][0]] for i in range(self.in_img_nb[0])
                ]
                imgs_info_L = [
                    [self.set_info[0].iloc[i], self.set_info[1].iloc[0]]
                    for i in range(self.in_img_nb[0])
                ]
            elif self.in_img_nb[1] == self.in_img_nb[0]:
                imgs_L = [
                    [self.in_fpaths[0][i], self.in_fpaths[1][i]] for i in range(self.in_img_nb[0])
                ]
                imgs_info_L = [
                    [self.set_info[0].iloc[i], self.set_info[1].iloc[i]]
                    for i in range(self.in_img_nb[0])
                ]
            else:
                raise ValueError(
                    "The second input must contain either a single image or "
                    + "the same number of images as the first input."
                )
        else:
            imgs_L = self.in_fpaths[0]
            imgs_info_L = [self.set_info[0].iloc[i] for i in range(self.in_img_nb[0])]

        # legend
        if legend:
            imgs_legend = []
            for imgs_info in imgs_info_L:
                if not isinstance(imgs_info, list):
                    imgs_info = [imgs_info]
                img_legend = []
                for i, img_info in enumerate(imgs_info):
                    for leg in legend:
                        if leg[0] == "$":

                            import re

                            # Regular expression to match ${var} and trailing characters
                            match = re.match(r"\$\{(\w+)\}(.)", leg)

                            if match:
                                col = match.group(1) # Extracts variable
                                trail = match.group(2) or '' # Extracts trailing characters
                            else:
                                print("Legend pattern not matched.")
                                col = ''

                            if col in img_info:
                                img_legend.append(img_info[col] + trail)
                            else:
                                img_legend.append("")
                        else:
                            img_legend.append(leg)

                imgs_legend.append(img_legend)
        else:
            imgs_legend = [None for i in range(self.in_img_nb[0])]

        if warp_field:
            tf_prefixes = self.set_info[0]["transform_prefix"].tolist()
            tf_paths = []

            for tf_prefix in tf_prefixes:
                # avoid case when tf_prefix is empty in the case of the
                # reference image that has not been warped
                if tf_prefix and isinstance(tf_prefix, str):
                    tf_path = glob.glob(self.in_dpath[0] + tf_prefix + "*Warp.nii*")[
                        0
                    ]  # + '*[0-9]Warp*')[0]
                    tf_paths.append(tf_path)
                else:
                    tf_paths.append(None)
        else:
            tf_paths = [None for i in range(len(self.in_fpaths[0]))]

        if isinstance(norm[0], tuple) and len(norm) == self.in_img_nb[0]:
            norm_arg = norm
        else:
            norm_arg = repeat(norm)

        plot_save_func = SavePlot(plot_maxproj)

        args_iter = zip(
            self.out_fpaths,
            imgs_L,
            self.in_names[0],
            repeat(mode),
            repeat(component),
            repeat(similarity),
            tf_paths,
            repeat(warp_downsampling),
            repeat(center_of_mass),
            repeat(colors),
            repeat(threshold),
            imgs_legend,
            norm_arg,
            repeat(white_zeros),
        )

        self.process_tasks(
            args_iter,
            plot_save_func,
            task_nb=self.in_img_nb[0],
            description="Plot max. proj.",
            kwargs_dict=kwargs,
        )


class SetPlotHist(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = True
    MP = True
    EXT = ".png"

    def process_set(self, tpl_mask_path, component=None, complementary=False):

        mask_fpath = pro.select_img(tpl_mask_path, "mask_*")
        mask = pro.read_img(mask_fpath).numpy().astype("bool")

        img_types = self.set_info["type"].tolist()

        input_tuples = zip(
            self.in_fpaths,
            self.out_fpaths,
            img_types,
            repeat(mask),
            repeat(component),
            repeat(complementary),
        )

        self.process_tasks(
            input_tuples,
            self.plot_hist_save,
            self.in_img_nb,
            description="Plot histograms",
            waitbar=True,
        )

    @staticmethod
    def plot_hist_save(in_fpath, out_fpath, img_type, mask, component, complementary):

        bin_nb = 20

        # prepare
        img = pro.read_img(in_fpath)

        # select channel
        img_array = img.numpy()
        if component is not None:
            img_array = img_array[:, :, :, component]
        img_array = img_array[mask]

        # prepare figure
        fig, axs = plt.subplots(
            nrows=2,
            ncols=1,
            dpi=150,
            figsize=(5, 7),
            # sharex=True
        )

        # compute distribution
        vals, edges = np.histogram(img_array, bins=bin_nb, range=(0, 1))
        vals = vals / vals.sum()
        x = (edges[0:-1] + edges[1:]) / 2
        w = edges[1] - edges[0]

        # plot
        plt.sca(axs[0])
        plt.bar(x, vals, width=w, alpha=1, align="center")

        # axis
        plt.xlim((0, 1))
        plt.xticks(np.arange(0, 1, 0.1))
        plt.xlabel("Probability of observing subpopulation")
        plt.ylabel("Proportion of pixels")
        plt.title("Distribution")

        # compute (complementary) cumulative distribution
        if complementary:
            vals = np.flip(vals)
            vals = np.cumsum(vals)
            vals = np.flip(vals)
            title = "Complementary cumulative distribution"
        else:
            vals = np.cumsum(vals)
            title = "Cumulative distribution"

        # plot
        plt.sca(axs[1])
        plt.bar(x, vals, width=w, alpha=1, align="center")

        # axis
        plt.xlim((0, 1))
        plt.xlabel("Probability of observing subpopulation")
        plt.ylabel("Proportion of pixels")
        plt.title(title)

        # save
        plt.tight_layout()
        plt.savefig(out_fpath)



class SetSummarizeHistograms(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = True
    MP = True
    EXT = ".png"
    SAVE_INFO = False

    def process_set(
            self,
            channel=None,
            bin_nb=50,
            exclude_zeros=False,
            sharex=False,
            yscale='linear',
            vrange=(None, None),
            include_sup=False,
            threshold=None,
            factor=1.0,
        ):

        args_iter = zip(
            self.in_fpaths,
        )

        kwargs_dict = {
            'channel': channel,
            'exclude_zeros': exclude_zeros,
            'bin_nb': bin_nb,
            'vrange': vrange,
            'include_sup': include_sup,
            'threshold': threshold,
            'factor': factor,
            }

        results = self.process_tasks(
            args_iter,
            self.compute_hist,
            task_nb=self.in_img_nb,
            description="Compute histograms",
            waitbar=True,
            kwargs_dict=kwargs_dict,
        )

        # summarize histograms in single figure
        CM = 1 / 2.54
        col_nb = 3
        row_nb = np.ceil(self.in_img_nb / col_nb).astype('int')
        fig, axs = plt.subplots(
            row_nb,
            col_nb,
            dpi=300,
            figsize=(4 * CM * col_nb, row_nb * 2.5 * CM),
            sharex=sharex,
        )

        axs = axs.ravel()

        hists_array = np.empty((self.in_img_nb, bin_nb))

        for i, res in enumerate(results):
            hist, bin_edges, thresholds = res

            hists_array[i, :] = hist

            ax = axs[i]
            plt.sca(ax)
            ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges[:2]), align='edge')

            ax.tick_params(labelsize=7, pad=3)
            plt.title(self.in_names[i], fontsize=7, pad=1)
            plt.yscale(yscale)
            plt.axvline(thresholds[-1], color='k', linestyle=':', linewidth=1)

            plt.xlim(vrange)

            if i >= self.in_img_nb - col_nb:
                plt.xlabel('Photon count', fontsize=7)
            if np.mod(i, col_nb) == 0:
                plt.ylabel('Voxel count', fontsize=7)

        for i in range(self.in_img_nb, (col_nb * row_nb)):
            ax = axs[i]
            plt.sca(ax)
            plt.axis('off')

        plt.subplots_adjust(hspace=0.4, wspace=0.45)


        plt.savefig(self.out_dpath + 'histograms.png', bbox_inches='tight')

        self.thresholds = [res[2] for res in results]

    def compute_hist(
            self,
            in_fpath,
            channel,
            bin_nb,
            exclude_zeros,
            vrange,
            include_sup,
            threshold,
            factor,
        ):

        img = pro.read_img(in_fpath)
        array = img.numpy()

        if channel is not None:
            array = np.take(array, channel, axis=3)

        if exclude_zeros:
            array = array[array > 0]

        if threshold is None:
            thresholds = []
        elif threshold == 'multiotsu':
            thresholds = skimage.filters.threshold_multiotsu(array)
        elif threshold == 'li':
            thresholds = [skimage.filters.threshold_li(array)]
        elif threshold == 'otsu':
            thresholds = [skimage.filters.threshold_otsu(array)]
        elif threshold == 'IQR':
            q1, q3 = np.percentile(array, [25, 75])
            iqr = q3 - q1
            thresholds = [q1 - factor * iqr, q3 + factor * iqr]
        elif threshold == 'MAD':
            med = np.median(array)
            mad = 1.4826 * np.median(np.abs(array - med))
            thresholds = [med - factor * mad, med + factor * mad]
        elif threshold == 'median':
            med = np.median(array)
            thresholds = [med / factor, med * factor]
        elif threshold == 'fixed':
            thresholds = [factor]
        else:
            raise ValueError('Unknown threshold: ' + threshold)

        if vrange[0] is None:
            vrange[0] = array.min()
        if vrange[1] is None:
            vrange[1] = array.max()

        bin_edges = np.linspace(vrange[0], vrange[1], bin_nb + 1)
        max_edge = bin_edges[-1]

        if include_sup:
            bin_edges[-1] = np.inf

        hist, _ = np.histogram(array.ravel(), bins=bin_edges)
        bin_edges[-1] = max_edge

        return hist, bin_edges, thresholds


class SetPlotSlices(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = False
    MP = True
    EXT = "/"

    def process_set(self, channel=0, sup_kwargs=None, **kwargs):

        if self.input_nb > 1:
            sup_in_fpaths = self.in_fpaths[1]
        else:
            sup_in_fpaths = repeat(None)


        input_tuples = zip(
            self.in_fpaths[0],
            self.out_fpaths,
            sup_in_fpaths,
        )

        kwargs_dict = {
            'channel': channel,
            'sup_kwargs': sup_kwargs,
        }

        kwargs_dict.update(kwargs)

        self.process_tasks(
            input_tuples,
            self._plot_slices,
            self.in_img_nb[0],
            description="Plot slices",
            waitbar=True,
            kwargs_dict=kwargs_dict,
        )

    def _plot_slices(self, in_fpath, out_dpath, sup_in_fpath, channel, sup_kwargs, cb_label=None, **kwargs):

        os.makedirs(out_dpath, exist_ok=True)

        # prepare
        img = pro.read_img(in_fpath)
        array = img.numpy()

        if sup_in_fpath is not None:
            sup_img = pro.read_img(sup_in_fpath)
            sup_array = sup_img.numpy()
            if sup_array.ndim < 4:
                sup_array = np.expand_dims(sup_array, -1)

        if array.ndim < 4:
            array = np.expand_dims(array, -1)

        if channel is not None:
            array = np.take(array, channel, axis=-1)

        z_nb = array.shape[2]
        xy_size = np.array(img.spacing[:2]) * np.array(array.shape[:2])
        CM = 1/2.54

        for i_z in np.arange(0, 65, 10):

            sl = np.take(array, i_z, axis=2)
            sl = np.swapaxes(sl, 0, 1)

            fig, ax = plt.subplots(1, 1, dpi=200, figsize=(15*CM, 10*CM))

            show_proj(
                ax,
                sl,
                xy_size,
                **kwargs,
                )

            CB_SHIFT = 0.05
            CB_W = 0.05

            # create inset axis to hold colobar
            ax.inset_axes([1 + CB_SHIFT, 0, CB_W, 1])

            if sup_in_fpath is not None:


                for i_c in range(sup_array.shape[-1]):
                    chan_array = np.take(sup_array, i_c, axis=-1)
                    sup_sl = np.take(chan_array, i_z, axis=2)
                    sup_sl = np.swapaxes(sup_sl, 0, 1)

                    plot_contour(sup_sl, img.spacing, color=sup_kwargs['colors'][i_c], linewidth=2)

            plt.tick_params(size=8)
            plt.xlabel('x (µm)', fontsize=8)
            plt.ylabel('y (µm)', fontsize=8)
            z_um = np.round((z_nb - i_z) * img.spacing[2]).astype('int')
            plt.title('z = ' + str(z_um) + ' µm', fontsize=8)

            plt.savefig(out_dpath + 'z' + str(i_z) + '.png', bbox_inches='tight')
            plt.close()



def plot_contour(array, spacing, color, ax=None, min_len=None, ds_factor=5, **kwargs):

    if ax is None:
        ax = plt.gca()

    contour_array = np.flipud(array.copy())

    contours = measure.find_contours(contour_array > 0, 0.5)

    lines = []

    for contour in contours:

        if (min_len is not None) and (len(contour) < min_len):
            continue

        print(len(contour))

        # add extra point
        contour = np.concatenate((contour, contour[[1], :]))

        n = contour.shape[0]
        distance = np.cumsum(np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1)))
        distance = np.insert(distance, 0, 0) / distance[-2]

        interp_domain = np.linspace(0, 1, max(n // ds_factor, 10))

        method = 'cubic'

        interpolator =  interp1d(distance, contour, kind=method, axis=0)
        interpolated_points = interpolator(interp_domain)

        interpolated_points = interpolated_points[:-1, ::-1] + 0.5
        interpolated_points = interpolated_points * np.reshape(spacing[:2], (1, -1)) + 0.5

        # Create a Path object
        path = mpl.path.Path(interpolated_points, closed=True)

        # Create a PathPatch from the Path
        patch = mpl.patches.PathPatch(path, facecolor='none', edgecolor=color, **kwargs)

        ax.add_patch(patch)
        lines.append(patch)

    return lines


def plot_joint_hist(img1, img2, component):

    img1 = ants.split_channels(pro.read_img(img1))[component]
    img2 = ants.split_channels(pro.read_img(img2))[component]

    vals1 = img1.numpy().ravel()
    vals2 = img2.numpy().ravel()

    plt.hist2d(
        vals1,
        vals2,
        bins=200,
        cmin=255.0 / 100.0,
        cmax=255.0,
    )

    plt.xlabel("Original")
    plt.ylabel("Reconstruction")


def plot_maxproj(
    imgs,
    title="",
    mode="max",
    component=None,
    similarity=None,
    tf_path=None,
    warp_downsampling=1,
    center_of_mass=None,
    colors=None,
    threshold=None,
    legend=None,
    norm=None,
    white_zeros=False,
    midpoint=None,
    figsize=None,
    style=None,
    colorbar_label=None,
    scalebar_color='white',
    scalebar_loc='left',
    **kwargs,
):

    if not isinstance(imgs, list):
        imgs = [imgs]

    img_nb = len(imgs)

    for i in range(img_nb):
        # load image if only a path is provided
        imgs[i] = pro.read_img(imgs[i])

        if imgs[i].has_components:

            # optionally select single component
            if (component is not None) and imgs[i].has_components:
                imgs[i] = pro.split_channels(imgs[i])[component]

            # or decompose image
            else:
                img = imgs[i]
                imgs.pop(i)
                for i_c in range(img.components):
                    imgs.insert(i + i_c, pro.split_channels(img)[i_c])

        # for FLIM images
        elif len(imgs[i].shape) == 4:
            sum_array = imgs[i].sum(axis=3)
            spacing = imgs[i].spacing[0:3]
            imgs[i] = ants.from_numpy(sum_array, spacing=spacing)

    img_nb = len(imgs)

    # prepare colors
    if not colors:
        if img_nb > 1 and img_nb < 4:
            # default is red, green, blue
            colors = list(np.eye(3))
        elif img_nb == 4:
            colors = [np.ones((3,))] + list(np.eye(3))
        else:
            # default is white
            colors = [np.ones((3,))]

    # prepare legend
    if legend is not None:
        leg_patches = []
        for i_leg, leg in enumerate(legend):
            if colors and len(colors) > i_leg:
                leg_patches.append(mpatches.Patch(color=colors[i_leg], label=leg))

    for i in range(img_nb):

        if i == 0:
            stack_shape = imgs[i].shape[:3]

            real_size = np.array(stack_shape) * np.array(imgs[i].spacing)[:3]
            stacks = np.zeros(stack_shape + (img_nb,))

            if center_of_mass:
                c_o_m = ants.get_center_of_mass(imgs[i])

        stack = imgs[i].numpy()
        stacks[:, :, :, i] = stack

    if mode is None:
        mode = "max"

    if midpoint is None:
        nan_val = 0.0
    else:
        nan_val = midpoint

    # compute projections
    if mode == "max":
        # max intensity
        max_projs = [stacks.max(d) for d in range(3)]
        method_name = "Maximum intensity projections"

    elif mode == "mean":
        stacks_nan = stacks.copy()
        stacks_nan[stacks == 0.0] = np.nan

        with warnings.catch_warnings():
            # ignore 'Mean of empty slice warning'
            warnings.filterwarnings("ignore", message="Mean of empty slice")

            # average
            max_projs = [np.nanmean(stacks_nan, axis=d) for d in range(3)]

        # max_projs = [np.mean(stacks, axis=d, where=stacks>0) for d in range(3)]
        for i in range(3):
            max_projs[i][np.isnan(max_projs[i])] = nan_val
        method_name = "Mean intensity projections"

    elif mode == "midslice":
        # middle slice
        max_projs = []
        slice_ixs = []
        slice_positions = []
        for i in range(3):
            mid_ix = np.array([stack_shape[i] // 2])
            slice_ixs.append(mid_ix)
            slice_positions.append((mid_ix + 0.5) / stack_shape[i])
            stack_slice = np.take(stacks, mid_ix, axis=i)
            stack_slice = np.squeeze(stack_slice, axis=i)
            max_projs.append(stack_slice)
        slice_positions = np.array(slice_positions)
        method_name = "Median slices"

    elif mode == "optslice":
        max_projs = []
        slice_ixs = []
        slice_positions = []
        nz_stacks = stacks > 0
        for i in range(3):
            plan_ixs = tuple(set(np.arange(4)) - set([i]))
            # print(plan_ixs, type(plan_ixs[0]))
            stack_sums = np.sum(nz_stacks, axis=plan_ixs)
            opt_ix = np.array([np.argmax(stack_sums)])
            slice_ixs.append(opt_ix)
            slice_positions.append((opt_ix + 0.5) / stack_shape[i])

            stack_slice = np.take(stacks, opt_ix, axis=i)
            stack_slice = np.squeeze(stack_slice, axis=i)
            max_projs.append(stack_slice)
        slice_positions = np.array(slice_positions)
        method_name = "Optimal slices"

    elif isinstance(mode, tuple):
        max_projs = [np.take(stacks, mode[i], axis=i) for i in range(3)]

        method_name = 'Slices ' + str(mode)

    else:
        raise ValueError("Unknown mode '" + str(mode) + "'.")

    gs_kw = {
        "width_ratios": [real_size[0], real_size[2]],
        "height_ratios": [real_size[2], real_size[1]],
    }

    # optional: compute similarity metrics
    if img_nb > 1 and similarity:
        display_metrics = True
        thread_nb = int(int(os.environ["MBMAP_NUM_THREADS"]) / int(os.environ["MBMAP_NUM_CPUS"]))
        metric_values = pro.compute_similarity_metrics(
            imgs[0], imgs[1], similarity, thread_nb=thread_nb
        )
    else:
        display_metrics = False

    if figsize is None:
        CM = 1/2.54
        width = 20*CM
        height = width / (real_size[0] + real_size[2]) * (real_size[1] + real_size[2])
        height = round(height)
        figsize = (width, height)

    # create subplot structure
    fig, axs = plt.subplots(
        2, 2, gridspec_kw=gs_kw, figsize=figsize, sharex="col", sharey="row", dpi=500
    )
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.15, hspace=0.15)

    # plot max intensity projections
    ax_coords_L = [(0, 0), (1, 0), (1, 1)]
    proj_plan_L = [[0, 2], [0, 1], [2, 1]]
    proj_dim_L = [1, 2, 0]

    # unique norm if single channel
    chan_nb = stacks.shape[3]
    if chan_nb == 1 and norm == "minmax":
        norm_min = min([p.min() for p in max_projs])
        norm_max = min([p.max() for p in max_projs])
        norm = (norm_min, norm_max)

    for i_p in range(3):
        ax_coords = ax_coords_L[i_p]
        proj_plan = proj_plan_L[i_p]
        dim_x = proj_plan[0]
        dim_y = proj_plan[1]
        proj_dim = proj_dim_L[i_p]

        ax = axs[ax_coords]

        max_proj = max_projs[proj_dim]

        if i_p == 0:
            ax.set_ylabel("z (µm)")

        elif i_p == 1:
            ax.set_xlabel("x (µm)")
            ax.set_ylabel("y (µm)")

        elif i_p == 2:
            ax.set_xlabel("z (µm)")
            max_proj = np.swapaxes(max_proj, 0, 1)
            max_proj = np.flip(max_proj, axis=0)

        if mode == "midslice" or mode == "optslice":
            ax.plot(
                slice_positions[dim_x] * [1.0, 1.0],
                [0, 1],
                transform=ax.transAxes,
                color="red",
                linestyle="dashed",
                linewidth=1.0,
            )
            ax.plot(
                [0, 1],
                slice_positions[dim_y] * [1.0, 1.0],
                transform=ax.transAxes,
                color="red",
                linestyle="dashed",
                linewidth=1.0,
            )

        # prepare for imshow's behavior
        max_proj = np.swapaxes(max_proj, 0, 1)

        proj_size = real_size[proj_plan]
        plot_handle = show_proj(
            ax,
            max_proj,
            proj_size,
            colors=colors,
            threshold=threshold,
            norm=norm,
            white_zeros=white_zeros,
            **kwargs,
        )

        ax.minorticks_on()
        ax.tick_params(labelsize=8)

        if style == "article":

            # remove ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # remove labels
            ax.set_ylabel("")
            ax.set_xlabel("")


            if i_p == 1:

                SB_X_MARGIN = 10
                SB_Y_MARGIN = 10
                SB_LENGTH = 25

                if scalebar_loc == 'left':
                    sb_xs = [SB_X_MARGIN, SB_X_MARGIN + SB_LENGTH]
                elif scalebar_loc == 'right':
                    sb_xs = [
                        proj_size[0] - SB_X_MARGIN - SB_LENGTH,
                        proj_size[0] - SB_X_MARGIN
                    ]

                sb_ys = [SB_Y_MARGIN, SB_Y_MARGIN]

                ax.plot(sb_xs, sb_ys, color=scalebar_color)

        # colorbar
        if i_p == 0 and isinstance(norm, tuple):

            if isinstance(colors, str):
                cmap = plot_handle.cmap
            else:
                cmap = mpl.cm.gray


            col_norm = mpl.colors.Normalize(vmin=norm[0], vmax=norm[1])

            # create inset axis to hold colobar
            CB_CORRECTION = 0.02
            cax = axs[0, 1].inset_axes([0, CB_CORRECTION / 2, 0.05, 1.0 - CB_CORRECTION])

            # plot colorbar
            cbar = plt.colorbar(
                mpl.cm.ScalarMappable(norm=col_norm, cmap=cmap),
                ax=axs[0, 0],
                cax=cax
            )
            cbar.set_label(label=colorbar_label, size=9)
            cbar.ax.tick_params(labelsize=8)

        if center_of_mass:
            ax.scatter(c_o_m[dim_x], c_o_m[dim_y], s=100, c="g", marker="+")

    # make unused subplot invisible
    axs[0, 1].axis("off")

    if legend is not None: #style == "article" and legend is not None
        axs[0, 1].legend(handles=leg_patches, fontsize=8, loc="center", handlelength=1.0)

    if display_metrics:
        if not isinstance(similarity, list):
            similarity = [similarity]

        text = "Similarity metrics\n\n"

        for i, metric in enumerate(similarity):
            text += metric["metric"] + ": " + str(-metric_values[i]) + "\n"

        plt.text(
            0.5,
            0.5,
            text,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=16,
            transform=axs[0, 1].transAxes,
        )

    if title is not None:
        fig.suptitle(method_name + "\n" + title, fontsize=9, fontweight="bold")

    if tf_path is not None:
        if mode == "max":
            tf_mode = "mean"
        else:
            tf_mode = slice_ixs

        fig = plot_transformation(
            tf_path,
            fig=fig,
            downsampling_factor=warp_downsampling,
            mode=tf_mode,
        )

    plt.tight_layout()

    return plot_handle


import matplotlib.colors as mpl_colors
import matplotlib.cm as mpl_cm


def show_proj(
    ax,
    max_proj,
    proj_size,
    colors=None,
    threshold=None,
    mode="add",
    norm=None,
    norm_type=None,
    white_zeros=False,
    grid=False,
    **kwargs,
):

    if max_proj.ndim > 2:
        chan_nb = max_proj.shape[2]
    else:
        max_proj = np.expand_dims(max_proj, 2)
        chan_nb = 1

    if mode == "add":
        rgb_mat = np.zeros(max_proj.shape[:2] + (3,))
    elif mode == "sub":
        rgb_mat = np.ones(max_proj.shape[:2] + (3,))

    # print(norm)

    # normalize
    if norm is None:
        max_proj = max_proj / 255.0
    elif norm == "minmax":

        axis = axis = (0, 1)
        axis = None

        max_proj = max_proj - np.amin(max_proj, axis=axis, keepdims=True)
        # print(np.amax(max_proj, axis=(0, 1, 2), keepdims=True))

        # print(np.amax(max_proj, axis=axis, keepdims=True))

        # return zero when division by zero
        max_proj = np.divide(
            max_proj,
            np.amax(max_proj, axis=axis, keepdims=True),
            out=np.zeros_like(max_proj),
            where=np.amax(max_proj, axis=axis) != 0,
        )


    elif isinstance(norm, tuple):

        if not (isinstance(colors, str) or isinstance(colors, (mpl_colors.Colormap, mpl_cm.ScalarMappable))):

            max_proj = max_proj - norm[0]
            max_proj = max_proj / norm[1]

    else:
        raise ValueError("Unknown norm '" + str(norm) + "'.")

    # function transforming intensity to remove values below threshold
    # threshold = 0.1
    # print(threshold)
    if threshold is not None:
        i_max = 1.0
        c = i_max / (1.0 - threshold)
        i_fun = lambda i: (c * (i - threshold) if i > threshold else 0.0)
    else:
        i_fun = lambda i: i
    i_vecfun = np.vectorize(i_fun)

    if colors is None:
        if chan_nb > 1 and chan_nb < 4:
            # default is red, green, blue
            colors = list(np.eye(3))
        elif chan_nb == 4:
            colors = [np.ones((3,))] + list(np.eye(3))
        else:
            # default is white
            colors = [np.ones((3,))]

    if isinstance(colors, str) or isinstance(colors, (mpl_colors.Colormap, mpl_cm.ScalarMappable)):
        if chan_nb == 1:
            img_mat = max_proj[:, :, 0]

            if white_zeros:
                img_mat[img_mat <= 0] = np.nan

            if isinstance(colors, str):
                cmap = mpl.colormaps[colors]
            else:
                cmap = colors

            if white_zeros:
                cmap.set_bad(mpl.colors.CSS4_COLORS['lightgrey'])

            plot_handle = ax.imshow(
                img_mat,
                cmap=cmap,
                extent=[0, proj_size[0], 0, proj_size[1]],
                vmin=norm[0],
                vmax=norm[1],
                norm=norm_type,
                **kwargs,
            )
            ax.set(facecolor="white")

        else:
            raise ValueError("Cannot have colormap with multiple channels.")
    else:
        for i_c in range(chan_nb):
            if not isinstance(colors, list):
                colors = [colors]
            color = colors[i_c]

            if mode == "add":
                op = 1.0
            elif mode == "sub":
                color = np.ones((3,)) - color
                op = -1.0

            intensity = i_vecfun(max_proj[:, :, i_c])

            for i_rgb in range(3):
                rgb_mat[:, :, i_rgb] = rgb_mat[:, :, i_rgb] + op * intensity * color[i_rgb]

        rgb_mat = np.clip(rgb_mat, 0.0, 1.0)

        plot_handle = ax.imshow(
            rgb_mat, extent=[0, proj_size[0], 0, proj_size[1]], vmin=0.0, vmax=1.0, **kwargs,
        )

    if grid:
        ax.grid(
            visible=True,
            color=[0.98039216, 0.63137255, 0.60784314],
            which='both',
            linestyle='--',
            linewidth=0.2,
        )

    return plot_handle


def compare_histograms(
    img, ref, match, num_bins, component=0, save_fpath=None, hist_range=(0.0, 255.0)
):

    img = pro.read_img(img)
    ref = pro.read_img(ref)
    match = pro.read_img(match)

    imgs = [ref, img, match]

    for i, img in enumerate(imgs):
        imgs[i] = pro.read_img(img)

        if pro.get_channel_nb(img) > 1:
            imgs[i] = pro.split_channels(img)[component]

    fig, axs = plt.subplots(2, 1, sharex="col", figsize=(10, 10))

    for i_sp in range(2):

        # plot reference data
        plt.sca(axs[i_sp])
        pixels = imgs[0].numpy().ravel()
        m = pixels.mean()

        vals, bins, _ = plt.hist(
            pixels, num_bins, range=hist_range, facecolor="red", alpha=0.5, label="reference"
        )

        # set limit
        if i_sp == 0:
            ymax = vals[bins[:-1] > m].max()

        plt.ylim(top=ymax)
        plt.axvline(x=m, color="red", linestyle="--")

        # plot img/match data
        pixels = imgs[i_sp + 1].numpy().ravel()
        m = pixels.mean()

        plt.hist(pixels, num_bins, range=hist_range, facecolor="blue", alpha=0.5, label="image")
        plt.axvline(x=m, color="blue", linestyle="--")

        plt.xlabel("Pixel count")
        if i_sp == 1:
            plt.ylabel("Intensity")

    for ax in axs:
        ax.legend()

    if save_fpath:
        save_dpath = Path(save_fpath).parent
        os.makedirs(save_dpath, exist_ok=True)
        plt.savefig(save_fpath, bbox_inches="tight")
        plt.close(fig)

    return


def plot_similarity_metrics(
    metrics, values=None, input_path=None, component=0, save=True, comparison_nb=None
):

    if values is None:
        if input_path is not None:
            values = pro.evaluate_similarity_set(
                input_path, metrics, component=component, comparison_nb=comparison_nb
            )

        else:
            raise ValueError("Either 'values' or 'input_path' should be set.")

    if input_path is not None:
        set_info = pd.read_csv(input_path + "files.csv", index_col=0)
        img_names = set_info["name"].values

    values = -values

    img_nb = values.shape[0]

    if values.ndim > 2:
        metric_nb = values.shape[2]
    else:
        metric_nb = 1

    if metric_nb == 2:
        two_metrics = True
        row_nb = 2
    else:
        two_metrics = False
        row_nb = 1

    if not isinstance(metrics, list):
        metrics = [metrics]

    mean_vals_M = np.empty((metric_nb, img_nb))

    fig, axs = plt.subplots(row_nb, metric_nb, figsize=(15, 15))

    hue = np.arange(img_nb)

    for i_m in range(metric_nb):

        metric = metrics[i_m]

        metric_values = values[:, :, i_m]

        mean_vals_M[i_m, :] = np.nanmean(metric_values, axis=1)

        plt.subplot(row_nb, metric_nb, i_m + 1)
        sns.violinplot(x=None, y=mean_vals_M[i_m, :], color="0.8")
        sns.stripplot(x=None, y=mean_vals_M[i_m, :], jitter=True, zorder=1, hue=hue)
        plt.ylabel(metric["metric"])
        plt.title("Mean = " + str(np.nanmean(mean_vals_M[i_m, :])))

    if two_metrics:
        plt.subplot(row_nb, 1, 2)
        plt.scatter(x=mean_vals_M[0, :], y=mean_vals_M[1, :])
        plt.xlabel(metrics[0]["metric"])
        plt.ylabel(metrics[1]["metric"])

        if "img_names" in locals():
            for i, txt in enumerate(img_names):
                plt.annotate(
                    txt,
                    (mean_vals_M[0, i], mean_vals_M[1, i]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                )
    title = Path(input_path).stem
    fig.suptitle("Average pairwise similarity\n" + title, fontsize=14, fontweight="bold")

    if save:
        if isinstance(save, str):
            output_filepath = save

        else:
            if input_path is not None:
                # save figure
                output_path = input_path + "similarity/"
                os.makedirs(output_path, exist_ok=True)
                output_filepath = output_path + "similarity.png"
                info = {
                    "Pairwise similarity": metric_values.tolist(),
                    "Names": img_names.tolist(),
                    "Metrics": metrics,
                }

                with open(output_path + "similarity_info.json", "w") as f:
                    json.dump(info, f)

            else:
                warnings.warn(
                    (
                        "Figure was not saved. If 'input_path' is not"
                        + "set, a filepath must be provided in 'save'."
                    ),
                    UserWarning,
                )
                return fig

        plt.savefig(output_filepath, bbox_inches="tight")
        plt.close(fig)

    return


def plot_convergence_set(input_path, tf_parameters, save=True):

    set_info = pd.read_csv(input_path + "files.csv", index_col=0)
    log_paths = set_info["transform_log"]
    log_paths.fillna("empty", inplace=True)
    log_paths = log_paths.values
    names = set_info["name"].values

    img_nb = len(set_info)

    tfs = tf_parameters["transforms"]
    tf_nb = len(tfs)

    metrics = tf_parameters["metric"]

    first = True

    final_metric_values = np.zeros((img_nb, tf_nb))
    final_metric_values[:] = np.nan

    for i in range(img_nb):

        log_path = log_paths[i]

        if log_path != "empty":

            reg_data = pro.parse_ants_log(input_path + log_path)

            # get number of stages with at least 1 iteration
            stage_nb_L = []
            valid_stages_L = []
            shrink_factors_L = []
            for i_tf in range(tf_nb):

                it_nbs = tf_parameters["number_of_iterations"][i_tf]

                valid_stages = [n > 0 for n in it_nbs]
                valid_stages_L.append(valid_stages)

                stage_nb_L.append(sum(valid_stages))

                shrink_factors = tf_parameters["shrink_factors"][i_tf]
                shrink_factors = [
                    f for (f, valid_bool) in zip(shrink_factors, valid_stages) if valid_bool
                ]
                shrink_factors_L.append(shrink_factors)

            max_stage_nb = max(stage_nb_L)

            for i_tf in range(tf_nb):

                it_nbs = tf_parameters["number_of_iterations"][i_tf]
                valid_stages = [n > 0 for n in it_nbs]

                shrink_factors = shrink_factors_L[i_tf]
                valid_stages = valid_stages_L[i_tf]
                stage_nb = stage_nb_L[i_tf]

                for i_s in range(stage_nb):
                    if valid_stages[i_s]:

                        data = reg_data[i_tf]["stages"][i_s]["data"]

                        for i_v, str_val in enumerate(data["convergenceValue"]):
                            data["convergenceValue"].iloc[i_v] = float(str_val)

                        # do not plot first point for stages > 0
                        if i_s > 0:
                            data.drop(data.index[0], inplace=True)

                        if first:
                            fig, axs = plt.subplots(
                                tf_nb,
                                max_stage_nb + 1,
                                figsize=(20, 15),
                                dpi=150,
                            )
                            first = False

                        if tf_nb == 1:
                            ax = axs[i_s]
                        else:
                            ax = axs[i_tf, i_s]

                        plt.sca(ax)
                        plt.plot(data["Iteration"].values, -data["metricValue"].values)
                        ax.set_title("Resolution 1/" + str(shrink_factors[i_s]))
                        ax.set_ylabel(metrics[i_tf])
                        ax.set_xlabel("Iteration")

                        final_metric_values[i, i_tf] = data["metricValue"].values[-1]

    lgd = fig.legend(names, loc="right", ncol=2)

    for ax in axs[:, -1]:
        ax.axis("off")

    if save:
        if tf_nb > 1:
            # store final values of first transform
            set_info["tf1_metric_value"] = final_metric_values[:, 0]
            set_info.to_csv(input_path + "files.csv")
            print(set_info)

        if isinstance(save, str):
            output_filepath = save

        else:
            output_path = input_path + "similarity/"
            os.makedirs(output_path, exist_ok=True)
            output_filepath = output_path + "convergence.png"

        plt.savefig(output_filepath, bbox_inches="tight", bbox_extra_artists=(lgd,))

    return


def plot_transformation(
    transform, fig=None, img1=None, img2=None, component=0, downsampling_factor=10, mode="mean"
):

    arrows_scale = 1 / 5
    arrow_sign = -1

    if isinstance(transform, str):
        warp_field = pro.read_img(transform)
    else:
        warp_field = transform

    arrows_scale = 1.0
    units = "x"

    if fig is None and img1 is not None:
        if img2 is None:
            fig = plot_maxproj(img1, ref_img=img2, component=component)
        else:
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    fig_ax_list = fig.axes
    col = (1.0, 1.0, 1.0, 0.5)

    tf_stack = warp_field.numpy()
    tf_shape = tf_stack.shape

    for i in range(3):
        tf_stack[:, :, :, i] = tf_stack[:, :, :, i] * warp_field.spacing[i]

    # for warp field produced by VoxelMorph
    if tf_shape[0] == 3:
        tf_stack = np.swapaxes(tf_stack, 0, -1)
        tf_shape = tf_stack.shape

    real_dims = np.array(tf_shape)[:3] * np.array(warp_field.spacing)[:3]

    x = np.linspace(0, real_dims[0], tf_shape[0], dtype="float")
    y = np.linspace(0, real_dims[1], tf_shape[1], dtype="float")
    z = np.linspace(0, real_dims[2], tf_shape[2], dtype="float")
    coords = [x, y, z]

    d = downsampling_factor

    transpose_UV_bools = [True, False, False]
    proj_planes = [(2, 1), (0, 2), (0, 1)]
    proj_axes = [0, 1, 2]
    fig_ax_ixs = [3, 0, 2]

    for t_bool, proj_p, proj_ax, fig_ax_ix in zip(
        transpose_UV_bools,
        proj_planes,
        proj_axes,
        fig_ax_ixs,
    ):

        fig_ax = fig_ax_list[fig_ax_ix]

        if mode == "mean":
            U = np.mean(tf_stack[:, :, :, proj_p[0]], axis=proj_ax).astype("float")
            V = np.mean(tf_stack[:, :, :, proj_p[1]], axis=proj_ax).astype("float")
        else:
            U = np.take(
                tf_stack[:, :, :, proj_p[0]],
                mode[proj_ax],
                axis=proj_ax,
            ).astype("float")
            V = np.take(
                tf_stack[:, :, :, proj_p[1]],
                mode[proj_ax],
                axis=proj_ax,
            ).astype("float")

            U = U.squeeze()
            V = V.squeeze()

        U *= arrow_sign
        V *= arrow_sign

        if t_bool:
            U = np.transpose(U)
            V = np.transpose(V)

        C1, C2 = np.meshgrid(coords[proj_p[0]], coords[proj_p[1]])
        C1 = np.transpose(C1)
        C2 = np.transpose(C2)

        # decimate
        U = U[::d, ::d]
        V = V[::d, ::d]
        C1 = C1[::d, ::d]
        C2 = C2[::d, ::d]

        fig_ax.quiver(
            C1,
            C2,
            U,
            V,
            scale=arrows_scale,
            color=col,
            angles="xy",
            units=units,
            scale_units=units,
        )

    return fig


def plot_template_similarity_evaluation(json_path, save=False, output_fpath=None):

    with open(json_path) as json_file:
        data = json.load(json_file)

    measure_nb = len(data["measures"])
    type_nb = len(data["types"])
    metric_nb = len(data["metrics"])

    values = np.array(data["values"], dtype=object)

    fig, axs = plt.subplots(
        metric_nb, type_nb, squeeze=False, figsize=(5 * type_nb, 5 * metric_nb)
    )

    for i_met in range(metric_nb):
        for i_t in range(type_nb):

            ax = axs[i_met, i_t]

            vals = values[:, i_t, i_met]

            x = []
            y = []

            for i_meas in range(measure_nb):

                meas_vals = vals[i_meas]
                meas_vals = [-v for v in meas_vals]
                n = len(meas_vals)

                y += meas_vals
                x += [i_meas] * n

            plt.sca(ax)
            plt.ylabel(data["metrics"][i_met])

            sns.violinplot(x=x, y=y, color="0.8")
            sns.stripplot(x=x, y=y, jitter=True, zorder=1)
            plt.xticks(ticks=list(range(measure_nb)), labels=data["measures"])
            plt.title("Type " + data["types"][i_t])

    if output_fpath is None:
        output_fpath = str(Path(json_path).parent) + "/" + str(Path(json_path).stem) + ".png"

    if save:
        plt.savefig(output_fpath)


def plot_template_convergence(json_fpath, fig_fpath=None):

    with open(json_fpath) as json_file:
        data = json.load(json_file)

    var_names = list(data.keys())

    var_name = var_names[0]
    vals = data[var_name]
    it_nb = len(vals)
    its = list(range(it_nb))

    fig, ax1 = plt.subplots(figsize=(5, 4), dpi=200)

    color = "tab:red"
    ax1.plot(its, vals, color=color, marker=".", markersize=10)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Mean displacement of update field")
    ax1.tick_params(axis="y")
    plt.xticks(ticks=its)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    if fig_fpath is None:
        plt.show()
    else:
        os.makedirs(Path(fig_fpath).parent, exist_ok=True)
        plt.savefig(fig_fpath)


def plot_evaluation(json_path, eval_names):

    with open(json_path, "r") as json_file:
        data = json.load(json_file)

    eval_nb = len(eval_names)

    types = data[eval_names[0]]["types"]
    type_nb = len(types)

    metrics = data[eval_names[0]]["metrics"]
    metric_nb = len(metrics)

    fig, axs = plt.subplots(metric_nb, type_nb, sharex="col", figsize=(5 * type_nb, 5 * metric_nb))

    for i_t, typ in enumerate(types):

        for i_m, metric in enumerate(metrics):

            metric_name = metric["metric"]
            sim_name = "Similarity (" + metric_name + ")"

            x = []
            y = []

            for i_e, eval_name in enumerate(eval_names):

                vals = [img_data[i_m] for img_data in data[eval_name]["values"][i_t]]

                # vals = plot_data['values'][i_t]
                vals = [-v for v in vals]
                y += vals
                n = len(vals)
                x += (i_e * np.ones((n,))).tolist()

            ax = axs[i_m, i_t]
            plt.sca(ax)
            sns.violinplot(x=x, y=y, color="0.8")
            sns.stripplot(x=x, y=y, jitter=True, zorder=1)
            plt.title(typ)
            plt.xticks(ticks=list(range(eval_nb)), labels=eval_names, rotation=-45)

            if i_t == 0:
                plt.ylabel(sim_name)
            else:
                if metric_name not in ["MeanSquares"]:
                    ax.get_shared_y_axes().join(axs[i_m, 0], ax)
                    ax.set_yticklabels([])


def plot_evaluation_summary(json_path):

    eval_names = [
        "raw_thres_data",
        "sh_raw_thres_data",
        "lincomb_thres_data",
        "sh_lincomb_thres_data",
    ]
    disp_names = ["Original", "Shuffled\nlabels", "Original", "Shuffled\nlabels"]

    with open(json_path, "r") as json_file:
        data = json.load(json_file)

    types = data[eval_names[0]]["types"]
    types.sort()
    type_nb = len(types)

    metrics = data[eval_names[0]]["metrics"]
    metrics = [metric for metric in metrics if metric["metric"] == "MeanSquares"]
    metric_nb = len(metrics)

    fig, axs = plt.subplots(
        metric_nb,
        type_nb,
        sharex="col",
        figsize=(5 * type_nb, 5 * metric_nb),
        squeeze=False,
        dpi=200,
    )

    std_palette = sns.color_palette(n_colors=2)
    pastel_palette = sns.color_palette("pastel", n_colors=2)
    palette = [val for pair in zip(std_palette, pastel_palette) for val in pair]

    for i_t, typ in enumerate(types):

        for i_m, metric in enumerate(metrics):

            metric_name = metric["metric"]

            if metric_name == "MeanSquares":
                sim_name = "Mean squared error"
            else:
                sim_name = "Similarity (" + metric_name + ")"

            x = []
            y = []

            for i_e, eval_name in enumerate(eval_names):

                vals = [img_data[i_m] for img_data in data[eval_name]["values"][i_t]]

                y += vals
                n = len(vals)
                x += (i_e * np.ones((n,))).tolist()

                if i_e == 0:
                    df = pd.DataFrame({eval_name: vals})
                else:
                    df[eval_name] = vals

            ax = axs[i_m, i_t]
            plt.sca(ax)
            sns.violinplot(data=df, color="0.8")
            sns.stripplot(data=df, jitter=True, zorder=1, palette=palette, label="tmp")
            plt.title(typ)
            plt.xticks(rotation=0)
            ax.set_xticklabels(disp_names)

            if i_t == 0:
                plt.ylabel(sim_name)
                handles, labels = ax.get_legend_handles_labels()
                handles = handles[0:3:2]
                labels = ["Projection", "Reconstruction"]
                plt.legend(handles, labels)
            else:
                if metric_name not in ["MeanSquares"]:
                    ax.get_shared_y_axes().join(axs[i_m, 0], ax)
                    ax.set_yticklabels([])

            ax.plot(
                [0.5, 0.5],
                [0, 1],
                transform=ax.transAxes,
                color="k",
                linestyle="dashed",
                linewidth=1.0,
            )

            annot = Annotator(ax, data=df, pairs=[(eval_names[0], eval_names[1])])
            annot.configure(test="Mann-Whitney-ls", text_format="star", verbose=2, loc="inside")
            annot.apply_test()
            ax, test_results = annot.annotate(line_offset_to_group=0.0)

            annot = Annotator(ax, data=df, pairs=[(eval_names[0], eval_names[2])])
            annot.configure(test="Mann-Whitney-gt", text_format="star", verbose=2, loc="inside")
            annot.apply_test()
            ax, test_results = annot.annotate(line_offset_to_group=0.0)

            annot = Annotator(ax, data=df, pairs=[(eval_names[2], eval_names[3])])
            annot.configure(test="Mann-Whitney-ls", text_format="star", verbose=2, loc="inside")
            annot.apply_test()
            ax, test_results = annot.annotate(line_offset_to_group=0.15)

    plt.suptitle("Distances between warped average map and single subject images")

    return handles


def plot_evaluation_summary2(json_path):

    eval_names = ["raw_thres_data", "sh_raw_thres_data"]
    disp_names = ["Original", "Shuffled\nlabels"]

    with open(json_path, "r") as json_file:
        data = json.load(json_file)

    types = data[eval_names[0]]["types"]
    types.sort()
    type_nb = len(types)

    metrics = data[eval_names[0]]["metrics"]
    metrics = [metric for metric in metrics if metric["metric"] == "MeanSquares"]
    metric_nb = len(metrics)

    fig, axs = plt.subplots(
        metric_nb,
        type_nb,
        sharex="col",
        figsize=(5 * type_nb, 5 * metric_nb),
        squeeze=False,
        dpi=200,
    )

    std_palette = sns.color_palette(n_colors=2)
    pastel_palette = sns.color_palette("pastel", n_colors=2)
    palette = [val for pair in zip(std_palette, pastel_palette) for val in pair]

    for i_t, typ in enumerate(types):

        for i_m, metric in enumerate(metrics):

            metric_name = metric["metric"]

            if metric_name == "MeanSquares":
                sim_name = "Mean squared error"
            else:
                sim_name = "Similarity (" + metric_name + ")"

            x = []
            y = []

            for i_e, eval_name in enumerate(eval_names):

                vals = [img_data[i_m] for img_data in data[eval_name]["values"][i_t]]

                y += vals
                n = len(vals)
                x += (i_e * np.ones((n,))).tolist()

                disp_name = disp_names[i_e]
                if i_e == 0:
                    df = pd.DataFrame({eval_name: vals})
                else:
                    df[eval_name] = vals

            ax = axs[i_m, i_t]
            plt.sca(ax)
            sns.violinplot(data=df, color="0.8")
            h = sns.stripplot(data=df, jitter=True, zorder=1, palette=palette, label="tmp")
            plt.title(typ)
            plt.xticks(rotation=0)
            ax.set_xticklabels(disp_names)

            if i_t == 0:
                plt.ylabel(sim_name)
            else:
                if metric_name not in ["MeanSquares"]:
                    ax.get_shared_y_axes().join(axs[i_m, 0], ax)
                    ax.set_yticklabels([])

            annot = Annotator(ax, data=df, pairs=[(eval_names[0], eval_names[1])])
            annot.configure(test="Mann-Whitney-ls", text_format="star", verbose=2, loc="inside")
            annot.apply_test()
            ax, test_results = annot.annotate(line_offset_to_group=0.3)

    plt.suptitle("Distances between warped average map and single subject images")

    return


def plot_map_evaluation_summary(json_path):

    eval_names = ["mapsim_thres_data", "sh_lab_mapsim_thres_data", "sh_vals_mapsim_thres_data"]
    disp_names = ["Original", "Shuffled\nlabels", "Shuffled\nvalues"]

    with open(json_path, "r") as json_file:
        data = json.load(json_file)

    types = data[eval_names[0]]["types"]
    types.sort()
    type_nb = len(types)

    metrics = data[eval_names[0]]["metrics"]
    metrics = [metric for metric in metrics if metric["metric"] == "MeanSquares"]
    metric_nb = len(metrics)

    fig, axs = plt.subplots(
        metric_nb,
        type_nb,
        sharex="col",
        figsize=(5 * type_nb, 5 * metric_nb),
        squeeze=False,
        dpi=200,
    )

    std_palette = sns.color_palette(n_colors=3)
    pastel_palette = sns.color_palette("pastel", n_colors=3)
    palette = [std_palette[2], pastel_palette[2], pastel_palette[2]]

    for i_t, typ in enumerate(types):

        for i_m, metric in enumerate(metrics):

            metric_name = metric["metric"]

            if metric_name == "MeanSquares":
                sim_name = "Mean squared error"
            else:
                sim_name = "Similarity (" + metric_name + ")"

            x = []
            y = []

            for i_e, eval_name in enumerate(eval_names):

                vals = [img_data[i_m] for img_data in data[eval_name]["values"][i_t]]

                y += vals
                n = len(vals)
                x += (i_e * np.ones((n,))).tolist()

                disp_name = disp_names[i_e]
                if i_e == 0:
                    df = pd.DataFrame({disp_name: vals})
                else:
                    df[disp_name] = vals

            ax = axs[i_m, i_t]
            plt.sca(ax)
            sns.violinplot(data=df, color="0.8")
            sns.stripplot(data=df, jitter=True, zorder=1, palette=palette)
            plt.title(typ)
            plt.xticks(rotation=0)

            if i_t == 0:
                plt.ylabel(sim_name)
            else:
                if metric_name not in ["MeanSquares"]:
                    ax.get_shared_y_axes().join(axs[i_m, 0], ax)
                    ax.set_yticklabels([])

            annot = Annotator(
                ax, data=df, pairs=[(disp_names[0], disp_names[1]), (disp_names[0], disp_names[2])]
            )
            annot.configure(test="Mann-Whitney-ls", text_format="star", verbose=2, loc="inside")
            annot.apply_test()
            ax, test_results = annot.annotate(line_offset_to_group=0.25)

    plt.suptitle("Distances between average maps created from separate subsets of subject")

    return


def plot_map_confusion(json_path, out_dpath):

    os.makedirs(out_dpath, exist_ok=True)

    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)

    # unpack data
    confusion_mats = np.asarray(json_data["confusion_mats"])
    proba_thresholds = json_data["proba_thresholds"]
    train_sizes = json_data["train_sizes"]
    test_size = json_data["test_size"]
    fold_nb = json_data["fold_nb"]
    class_labels = json_data["class_labels"]

    titles = [
        "Signal confusion matrix",
        "Input-normalized confusion matrix",
        "Output-normalized confusion matrix",
    ]
    colorbar_labels = [
        "Neuron equivalent amount of signal",
        "Proportion of input signal",
        "Proportion of output signal",
    ]

    threshold_nb = len(proba_thresholds)
    mat_nb = 3
    class_nb = len(class_labels)
    types = class_labels[0:3]
    type_nb = len(types)
    subset_nb = len(train_sizes)

    # fold-averaged confusion matrix for full set
    lg_size_mean_mat = confusion_mats[:, :, :, :, -1].mean(axis=3)

    # preallocation
    diag_vals = np.empty((class_nb, threshold_nb, mat_nb))
    tprs = np.empty((class_nb, threshold_nb))
    fprs = np.empty((class_nb, threshold_nb))

    # plot full set confusion matrices for each threshold and compute values required
    # for ROC curves
    for i_th, thres in enumerate(proba_thresholds):

        fig, axs = plt.subplots(1, mat_nb, figsize=(20, 7), sharey="row")

        raw_mat = lg_size_mean_mat[:, :, i_th]

        norm_vec = raw_mat.sum(axis=1, keepdims=True)
        norm_vec[norm_vec == 0.0] = 1.0
        input_norm_mat = raw_mat / norm_vec

        norm_vec = raw_mat.sum(axis=0, keepdims=True)
        norm_vec[norm_vec == 0.0] = 1.0
        output_norm_mat = raw_mat / norm_vec

        mats = [raw_mat, input_norm_mat, output_norm_mat]

        for i_ty in range(type_nb):
            tprs[i_ty, i_th] = raw_mat[i_ty, i_ty] / raw_mat[i_ty, :].sum()
            fprs[i_ty, i_th] = (raw_mat[:, i_ty].sum() - raw_mat[i_ty, i_ty]) / (
                raw_mat.sum() - raw_mat[i_ty, :].sum()
            )

        for i_m in range(mat_nb):

            ax = axs[i_m]
            plt.sca(ax)

            mat = mats[i_m]

            title = titles[i_m]
            cb_label = colorbar_labels[i_m]
            lg_size_MCC = pro.compute_mcc(mat)

            plot_h = plt.matshow(mat, fignum=0, cmap="Greys", vmin=0)

            # display values in the cells
            for (i, j), z in np.ndenumerate(mat):
                ax.text(j, i, "{:0.1f}".format(z), ha="center", va="center", backgroundcolor="w")

            plt.xlabel("Output signal")

            if i_m == 0:
                plt.ylabel("Input signal")

            plt.xticks(ticks=range(len(class_labels)), labels=class_labels)
            plt.yticks(ticks=range(len(class_labels)), labels=class_labels)

            ax.xaxis.set_label_position("top")

            plt.title(
                title
                + "\n("
                + str(fold_nb)
                + " folds, "
                + "MCC = "
                + str(round(lg_size_MCC, 3))
                + ", "
                + "test set size = 3 x "
                + str(test_size)
                + ", "
                + "train size = 3 x "
                + str(train_sizes[-1])
                + ")"
            )

            plt.colorbar(plot_h, label=cb_label, fraction=0.046, pad=0.04)

            # store values
            diag_vals[:, i_th, i_m] = np.diag(mat)

        plt.suptitle("Probability threshold: " + str(round(thres, 2)))
        plt.savefig(out_dpath + "confusion_thres" + str(i_th) + ".png")

    # compute AUC and MCC for all subsets
    # preallocation
    subset_tprs = np.empty((type_nb, threshold_nb, subset_nb))
    subset_fprs = np.empty((type_nb, threshold_nb, subset_nb))
    AUCs = np.empty((type_nb, subset_nb))
    th05_MCCs = []
    th05_ix = np.argmin(np.abs(np.array(proba_thresholds) - 0.5))
    subset_sizes = []

    for i_s, s in enumerate(train_sizes):

        subset_sizes.append(train_sizes[i_s] * 3)

        mean_mats = confusion_mats[:, :, :, :, i_s].mean(axis=3)

        th05_mean_mats = mean_mats[:, :, th05_ix]
        th05_MCCs.append(pro.compute_mcc(th05_mean_mats))

        for i_th, th in enumerate(proba_thresholds):
            mat = mean_mats[:, :, i_th]

            # compute TPR and FPR
            for i_ty in range(type_nb):
                subset_tprs[i_ty, i_th, i_s] = mat[i_ty, i_ty] / mat[i_ty, :].sum()
                subset_fprs[i_ty, i_th, i_s] = (mat[:, i_ty].sum() - mat[i_ty, i_ty]) / (
                    mat.sum() - mat[i_ty, :].sum()
                )

        for i_ty in range(type_nb):
            x = np.squeeze(subset_fprs[i_ty, :, i_s])
            y = np.squeeze(subset_tprs[i_ty, :, i_s])

            AUCs[i_ty, i_s] = np.trapz(y=y, x=-x)

    # first summary figure, first line
    types = class_labels[0:3]
    neuron_nbs = par.get_neuron_nbs()
    total_neuron_nb = sum(neuron_nbs.values())

    diag_vals[np.isnan(diag_vals)] = 0.0

    type_nb = len(types)
    fig, axs = plt.subplots(2, type_nb, figsize=(18, 10), sharey="row", sharex="row", dpi=200)

    for i_t, ty in enumerate(types):
        ax = axs[0, i_t]

        base_prop = neuron_nbs[ty] / total_neuron_nb

        plt.sca(ax)

        plt.plot(proba_thresholds, np.squeeze(diag_vals[i_t, :, 1]), "o-")
        plt.plot(proba_thresholds, np.squeeze(diag_vals[i_t, :, 2]), "o-")
        plt.plot(
            [proba_thresholds[0], proba_thresholds[-1]],
            [base_prop, base_prop],
            color="k",
            linestyle="dashed",
            linewidth=1.0,
        )

        plt.xlabel("Probability threshold")
        if i_t == 0:
            plt.ylabel("Proportion (%)")

        plt.legend(
            [
                "Captured proportion of input",
                "Contribution to output",
                "True subpopulation proportion in input",
            ]
        )
        plt.title(ty)

    plt.suptitle(
        "Balance between sensitivity (captured proportion) and precision (output contribution) obtained by applying a threshold on the probability maps"
    )

    # first summary figure, second line
    for i_t, ty in enumerate(types):
        ax = axs[1, i_t]

        plt.sca(ax)
        x = fprs[i_t, :]
        y = tprs[i_t, :]

        plt.fill_between(x, y, color="tab:blue", alpha=0.5)
        plt.plot(x, y, "o-")
        area = np.trapz(y, x=-x)
        plt.xlabel("False positive proportion")
        if i_t == 0:
            plt.ylabel("True positive proportion")

        # plt.legend(['Captured proportion of input', 'Contribution to output', 'True subpopulation proportion in input'])
        plt.title(ty + " (area = " + str(round(area, 2)) + ")")

    fig.tight_layout()
    plt.savefig(out_dpath + "thres_evolution.png")

    # second summary figure
    fig, axs = plt.subplots(1, type_nb + 1, figsize=(18, 5), dpi=200)

    # first_axis = None
    for i_t, ty in enumerate(types):

        ax = axs[i_t]
        plt.sca(ax)

        # first curve
        color = "tab:red"
        ax.plot(subset_sizes, AUCs[i_t, :], "o-", color=color)
        plt.ylim((0.0, 1.0))

        ax.set_xlabel("Number of images")
        if i_t == 0:
            ax.set_ylabel("AUC", color=color)
        ax.tick_params(axis="y", labelcolor=color)

        plt.title(ty)

    ax = axs[type_nb]
    plt.sca(ax)

    color = "tab:blue"
    ax.plot(subset_sizes, th05_MCCs, "o-", color=color)
    plt.ylim((0.0, 1.0))

    ax.set_xlabel("Number of images")
    ax.set_ylabel("MCC (threshold 0)", color=color)
    ax.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    plt.savefig(out_dpath + "set_size_evolution.png")


def plot_opt_map_confusion(json_path, out_dpath):

    os.makedirs(out_dpath, exist_ok=True)

    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)

    # unpack data
    confusion_mats = np.asarray(json_data["confusion_mats"])
    train_sizes = json_data["train_sizes"]
    test_size = json_data["test_size"]
    fold_nb = json_data["fold_nb"]
    class_labels = json_data["class_labels"]
    undetermination = json_data["undetermination"]

    titles = [
        "Signal confusion matrix",
        "Input-normalized confusion matrix",
        "Output-normalized confusion matrix",
    ]
    colorbar_labels = [
        "Neuron equivalent amount of signal",
        "Proportion of input signal",
        "Proportion of output signal",
    ]

    mat_nb = 3
    types = class_labels[0:3]
    type_nb = len(types)
    neuron_nbs = par.get_neuron_nbs()

    # TEMP
    class_labels = class_labels[0:3]
    confusion_mats = confusion_mats[0:3, 0:3, :, :]

    # fold-averaged confusion matrix for full set
    lg_size_mean_mat = confusion_mats[:, :, :, -1].mean(axis=2)

    fig, axs = plt.subplots(2, mat_nb, figsize=(20, 7), sharey="row")

    raw_mat = lg_size_mean_mat

    if undetermination:
        conf_mat = np.zeros((type_nb, type_nb + 1))
        conf_mat[0:type_nb, 0:type_nb] = raw_mat

        for i_t, ty in enumerate(types):
            conf_mat[i_t, type_nb] = neuron_nbs[ty] - conf_mat[i_t, :].sum()

        raw_mat = conf_mat
        xtick_labels = types + ["None"]
    else:
        xtick_labels = types

    norm_vec = raw_mat.sum(axis=1, keepdims=True)
    norm_vec[norm_vec == 0.0] = 1.0
    input_norm_mat = raw_mat / norm_vec

    norm_vec = raw_mat.sum(axis=0, keepdims=True)
    norm_vec[norm_vec == 0.0] = 1.0
    output_norm_mat = raw_mat / norm_vec

    mats = [raw_mat, input_norm_mat, output_norm_mat]

    # compute AUC and MCC for all subsets
    MCC_means = []
    MCC_stds = []
    subset_sizes = []
    for i_s, s in enumerate(train_sizes):

        subset_sizes.append(train_sizes[i_s] * 3)

        fold_MCCs = []
        for i_f in range(fold_nb):
            mat = confusion_mats[:, :, i_f, i_s]
            fold_MCCs.append(pro.compute_mcc(mat))

        MCC_means.append(np.mean(fold_MCCs))
        MCC_stds.append(np.std(fold_MCCs))

    for i_m in range(mat_nb):

        ax = axs[0, i_m]
        plt.sca(ax)

        mat = mats[i_m]

        title = titles[i_m]
        cb_label = colorbar_labels[i_m]
        ref_MCC_mean = MCC_means[-1]
        ref_MCC_std = MCC_stds[-1]

        if i_m == 0:
            out_MCC = ref_MCC_mean.copy()

        plot_h = plt.matshow(mat, fignum=0, cmap="Greys", vmin=0)

        # display values in the cells
        for (i, j), z in np.ndenumerate(mat):
            ax.text(j, i, "{:0.2f}".format(z), ha="center", va="center", backgroundcolor="w")

        plt.xlabel("Output signal")

        if i_m == 0:
            plt.ylabel("Input signal")

        plt.xticks(ticks=range(len(xtick_labels)), labels=xtick_labels)
        plt.yticks(ticks=range(type_nb), labels=types)

        ax.xaxis.set_label_position("top")

        plt.title(
            title
            + "\n"
            + "test set size = 3 x "
            + str(test_size)
            + ", "
            + "train size = 3 x "
            + str(train_sizes[-1])
            + ", "
            + str(fold_nb)
            + " folds"
            + "\n"
            + "MCC = "
            + str(round(ref_MCC_mean, 3))
            + " ± "
            + str(round(ref_MCC_std, 3))
        )

        plt.colorbar(plot_h, label=cb_label, fraction=0.046, pad=0.04)

    ax = axs[1, 0]
    plt.sca(ax)
    # ax.plot(subset_sizes, MCC_means, 'o-')
    plt.errorbar(subset_sizes, MCC_means, fmt="-", yerr=MCC_stds)
    plt.ylim((0.0, 1.0))
    plt.xlabel("Train set size")
    plt.ylabel("MCC")

    plt.savefig(out_dpath + "opt_map_confusion.png")

    return out_MCC


def plot_map_classification(json_path, out_dpath):

    os.makedirs(out_dpath, exist_ok=True)

    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)

    res_nb = len(json_data)
    res_names = list(json_data.keys())

    colorbar_labels = [
        "Proportion",
        "Proportion",
    ]

    fig, axs = plt.subplots(2, res_nb, figsize=(res_nb * 7, 2 * 4), sharex="row")

    for i_m in range(res_nb):

        name = res_names[i_m]
        cb_label = colorbar_labels[i_m]

        # load data
        data = json_data[name]
        mats = np.asarray(data["confusion_mats"])
        train_sizes = np.asarray(data["train_sizes"])
        test_size = data["test_size"]
        labels = data["types"]
        fold_nb = data["fold_nb"]

        size_nb = len(train_sizes)

        # extract largest set data
        lg_size_mat = mats[:, :, :, -1].mean(axis=2)
        lg_size_MCC = pro.compute_mcc(lg_size_mat)
        norm_lg_size_mat = lg_size_mat / lg_size_mat.sum(axis=1, keepdims=True)

        fold_subset_MCCs = np.empty((fold_nb, size_nb))
        subset_sizes = [s * 3 for s in train_sizes]

        for i_s, size in enumerate(subset_sizes):
            for i_f in range(fold_nb):
                subset_mat = mats[:, :, i_f, i_s]
                fold_subset_MCCs[i_f, i_s] = pro.compute_mcc(subset_mat)

        fold_subset_MCCs = np.nan_to_num(fold_subset_MCCs, nan=0.0)

        mean_MCCs = np.mean(fold_subset_MCCs, axis=0)
        std_MCCs = np.std(fold_subset_MCCs, axis=0)

        ax = axs[0, i_m]
        plt.sca(ax)

        plot_h = plt.matshow(norm_lg_size_mat, fignum=0, cmap="Greys", vmin=0, vmax=1)

        # display values in the cells
        for (i, j), z in np.ndenumerate(norm_lg_size_mat):
            ax.text(j, i, "{:0.2f}".format(z), ha="center", va="center", backgroundcolor="w")

        plt.xlabel("Prediction")

        if i_m == 0:
            plt.ylabel("Truth")

        plt.xticks(ticks=range(len(labels)), labels=labels)
        plt.yticks(ticks=range(len(labels)), labels=labels)

        ax.xaxis.set_label_position("top")

        plt.title(
            name
            + "\n("
            + str(fold_nb)
            + " folds, "
            + "MCC = "
            + str(round(lg_size_MCC, 3))
            + ", "
            + "test set size = 3 x "
            + str(test_size)
            + ", "
            + "train size = 3 x "
            + str(train_sizes[-1])
            + ")"
        )
        plt.colorbar(plot_h, label=cb_label, fraction=0.046, pad=0.04)

        ax = axs[1, i_m]
        plt.sca(ax)

        plt.plot(subset_sizes, mean_MCCs, "o-", color="tab:blue")
        print(type(mean_MCCs[0]))
        plt.errorbar(subset_sizes, mean_MCCs, std_MCCs, color="tab:blue")
        plt.xlabel("Train set size")
        plt.ylabel("MCC")
        plt.ylim((0.0, 1.1))

    plt.savefig(out_dpath + "classification_results.png")


def get_set_summary(dpath):

    info = pd.read_csv(dpath + "files.csv", index_col=0)

    types = sorted(list(np.unique(info["type"].values)))
    hems = ["L", "R"]
    hem_cols = ["Left", "Right"]

    print(types)
    print(hems)

    if not ("subject" in info):
        info["subject"] = np.array(info["type"]) + "_" + np.array(info["fly"]).astype(str)

    summary = pd.DataFrame(
        0, index=types + ["Total"], columns=["Subjects", "Left", "Right", "Both"]
    )

    for ty in types:

        summary.loc[ty]["Subjects"] = len(np.unique(info[(info["type"] == ty)]["subject"]))

        for hem, hem_col in zip(hems, hem_cols):

            summary.loc[ty][hem_col] = len(
                info[(info["type"] == ty) & (info["hemisphere"] == hem)]
            )

        summary.loc[ty]["Both"] = sum(summary.loc[ty][hem_cols])

    summary.loc["Total"] = summary.sum()

    print(info)
    print(summary)


def plot_alpha(df, avg_mode, param_name=None, ax=None, paired=False):

    plot_param(
        df,
        avg_mode,
        param_name=r"$f_{free}$",
        ax=ax,
        ylims_1=(0.55, 1.0), #0.85
        ylims_2=(0.55, 1.0),
        avg_category="Subject",
        paired=paired,
    )


def plot_param(
    df, avg_mode, param_name=None, ax=None, ylims_1=None, ylims_2=None, avg_category=None,
    paired=False,
):

    df = df.copy()
    df = df.dropna()

    if param_name is None:
        param_name = "parameter"

    types = list(sorted(list(set(df["type"].values))))
    type_nb = len(types)

    type_names = par.get_neuron_names()
    tick_labels = types.copy()

    for i_t, ty in enumerate(tick_labels):
        if ty in type_names:
            tick_labels[i_t] = type_names[ty]

    type_nb = len(types)

    if ax is None:
        _, ax = plt.subplots(1, 1, dpi=300, figsize=(type_nb * 1 + 1, 4))
    else:
        plt.sca(ax)

    if type_nb == 2:
        palette = ["tab:blue", "tab:orange"]
    else:
        palette = par.get_soft_sp_colors()

    print(palette)

    sns.boxplot(
        df,
        x="type",
        y="average",
        palette=palette,
        order=types,
    )

    type_avg_list = []
    ms = []

    if avg_mode == "mean":
        avg_fun = np.mean
    elif avg_mode == "median":
        avg_fun = np.median

    for t in types:
        avg_vals = df[df["type"] == t]["average"].values
        type_avg_list.append(avg_vals.ravel())
        ms.append(avg_fun(df[df["type"] == t]["average"].values))
        print("mean", t, np.mean(df[df["type"] == t]["average"].values))
        print("std", t, np.std(df[df["type"] == t]["average"].values))

    if type_nb > 1:

        combinations = itertools.combinations(types, 2)
        pairs = [tuple(pair) for pair in combinations]

        if avg_mode == "median":
            # > 2 samples: Kruskal-Wallis test
            _, p = scipy.stats.kruskal(*type_avg_list)
            # 2 samples: Mann-Whitney U-test:
            test_2s = "Mann-Whitney"
        elif avg_mode == "mean":
            # > 2 samples: ANOVA
            _, p = scipy.stats.f_oneway(*type_avg_list)
            # 2 samples: Mann-Whitney:
            if paired:
                test_2s = "t-test_paired"
            else:
                test_2s = "t-test_ind"

        annot = Annotator(
            ax,
            data=df,
            pairs=pairs,
            x="type",
            y="average",
            order=types,
        )
        annot.configure(
            test=test_2s,
            text_format="simple",
            verbose=1,
            loc="inside",
            # comparisons_correction="Holm-Bonferroni",
            correction_format="replace",
            show_test_name=False,
        )
        annot.apply_test()

        # display tests
        ax, test_results = annot.annotate(line_offset_to_group=0.01, line_offset=0.0)

    if ylims_2 is not None:
        plt.ylim(ylims_2)

    plt.xlabel("Type")
    if avg_category is None:
        plt.ylabel(avg_mode.capitalize() + " " + param_name)
    else:
        plt.ylabel(avg_category.capitalize() + " " + param_name)

    ax.set_xticks(np.arange(type_nb), tick_labels)

    m_strs = ["{:.3f}".format(m) for m in ms]
    text = (
        avg_mode.capitalize()
        + " = ["
        + "".join([s + ", " for s in m_strs[:-1]])
        + m_strs[-1]
        + "]"
    )

    plt.title(text)

    plt.tight_layout()


def plot_resolution(dpath):

    # load
    info_fpath = dpath + "files.csv"
    info = pd.read_csv(info_fpath)

    # extract the columns containing the word "resolution"
    resolution_cols = [col for col in info.columns if "resolution" in col]

    # extract the channel numbers using regex
    channels = [int(re.findall(r"Channel (\d+)", col)[0]) for col in resolution_cols]
    chan_nb = max(channels) + 1

    # extract the resolution names using regex (assuming consistent resolutions across all channels)
    resolution_names = [
        re.findall(r"(\w+) resolution", col)[0] for col in resolution_cols[0:chan_nb]
    ]

    fig, axs = plt.subplots(
        1,
        chan_nb + 1,
        dpi=300,
        figsize=((chan_nb + 1) * 3, 3),
    )

    outlier_names = []

    # extract data
    for i_c in range(chan_nb):

        plt.sca(axs[i_c])
        if i_c > 0:
            axs[0].sharex(axs[i_c])
            axs[0].sharey(axs[i_c])

        col_names = []
        thresholds = []

        for i_r, res in enumerate(resolution_names):
            col_name = "Channel " + str(i_c) + " " + res + " resolution"

            col_names.append(col_name)
            vals = info[col_name].values
            threshold = vals.mean() + 3 * vals.std()

            thresholds.append(threshold)

        xvals = info[col_names[0]].values
        yvals = info[col_names[1]].values

        outlier_lgcs = np.logical_or(xvals > thresholds[0], yvals > thresholds[1])
        outlier_names += info.loc[outlier_lgcs, "name"].tolist() + [" "]

        plt.scatter(xvals[~outlier_lgcs], yvals[~outlier_lgcs], alpha=0.5)
        plt.scatter(xvals[outlier_lgcs], yvals[outlier_lgcs], alpha=0.5)
        plt.axvline(thresholds[0], color="gray", linestyle=":")
        plt.axhline(thresholds[1], color="gray", linestyle=":")

        plt.xlabel(col_names[0])
        plt.ylabel(col_names[1])

    ax = axs[-1]
    plt.sca(ax)

    # Set the x and y coordinates for each string
    x_pos = 0.1
    y_pos = 0.9
    y_step = 0.1

    # Iterate over the strings and display them in a column
    for string in outlier_names:
        ax.text(x_pos, y_pos, string, fontsize=10, ha="left", va="top")
        y_pos -= y_step

    # Set limits and labels
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    plt.tight_layout()

    print(outlier_names)

