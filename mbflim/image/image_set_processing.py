#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 17:05:00 2022

@author: proussel
"""

import shutil
import sys
import os
import glob
from datetime import datetime
import json
import tempfile
import multiprocessing
from itertools import repeat

import nipype.interfaces.ants as nipype_ants
import ants
from ants.registration.create_jacobian_determinant_image import create_jacobian_determinant_image
import itk
from sklearn.decomposition import NMF
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np

import mbflim.shared as sha
import mbflim.visualization as vis
import mbflim.parameters as par
import mbflim.image.image_processing as pro
import mbflim.utils as ut

# %% simple SetProcessing


class SetSimpleProcess(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = True
    MP = True

    def process_set(self, process_func, *args, **kwargs):

        repeat_args = [repeat(arg) for arg in args]

        input_tuples = zip(
            self.in_fpaths,
            self.out_fpaths,
            repeat(process_func),
            *repeat_args
        )

        self.process_tasks(
            args_iter=input_tuples,
            function=pro.read_process_save,
            task_nb=self.in_img_nb,
            description='Process',
            kwargs_dict=kwargs,
            # process_nb=1,
        )


class SetProcess(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = True
    MP = True

    def process_set(self, process_class, *args, channel=None, distrib_kwargs=None, **kwargs):

        repeat_args = [repeat(arg) for arg in args]

        if distrib_kwargs is not None:
            distrib_kwargs_list = []
            for i in range(self.in_img_nb):
                distrib_kwargs_i = {}
                for key in distrib_kwargs.keys():
                    distrib_kwargs_i.update({key: distrib_kwargs[key][i]})
                distrib_kwargs_list.append(distrib_kwargs_i)
        else:
            distrib_kwargs_list = repeat({})

        input_tuples = zip(
            self.in_fpaths,
            self.out_fpaths,
            repeat(process_class),
            repeat(channel),
            distrib_kwargs_list,
            *repeat_args,
        )

        self.process_tasks(
            args_iter=input_tuples,
            function=self.read_run_save,
            task_nb=self.in_img_nb,
            description='Process',
            kwargs_dict=kwargs,
            # method='spawn',
        )

    def read_run_save(
            self,
            in_fpath,
            out_fpath,
            process_class,
            channel,
            distrib_kwargs,
            *args,
            recompute=True,
            **kwargs
    ):

        if not recompute and os.path.exists(out_fpath):
            return

        img = ants.image_read(in_fpath)
        img = process_class(img, channel).run(*args, **kwargs, **distrib_kwargs)
        ants.image_write(img, out_fpath)

        return


# %% Image set processing classes

class SetMirror(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = True
    MP = True

    def process_set(self, hemisphere=None, keep_original=False):

        print(self.set_info)

        # define elements to mirror
        if hemisphere:
            mirror_ixs = self.set_info[
                self.set_info['hemisphere'] == hemisphere].index.tolist()
        else:
            mirror_ixs = self.set_info.index.tolist()

        if keep_original:
            mirror_lgcs = [False] * self.in_img_nb + [True] * len(mirror_ixs)
            mir_info = self.set_info.iloc[mirror_ixs].reset_index(drop=True)

            self.out_info = pd.concat(
                [self.out_info, mir_info]).reset_index(drop=True)
            self.in_fpaths = self.in_fpaths + \
                [self.in_fpaths[ix] for ix in mirror_ixs]
            self.in_img_nb = len(self.in_fpaths)
        else:
            mirror_lgcs = [False] * self.in_img_nb
            for ix in mirror_ixs:
                mirror_lgcs[ix] = True

        for ix, row in self.out_info.iterrows():
            if mirror_lgcs[ix]:
                # change hemisphere name
                self.out_info.at[ix, 'hemisphere'] = 'mir' + row['hemisphere']
                # change hemisphere in name
                self.out_info.at[ix, 'name'] = self.out_info.iloc[ix]['name'].replace(
                    '_' + hemisphere, '_' + self.out_info.iloc[ix]['hemisphere'])
                # update file path
                self.out_info.at[ix, 'path'] = self.prefix + \
                    self.out_info.iloc[ix]['name'] + self.EXT

        self.out_fpaths = (self.out_dpath + self.out_info['path']).tolist()

        # mirror defined elements and copy the rest
        input_tuples = zip(mirror_lgcs, self.in_fpaths, self.out_fpaths)

        self.process_tasks(
            args_iter=input_tuples,
            function=self.mirror_or_copy,
            task_nb=self.in_img_nb,
            description='Mirror',
        )

        with multiprocessing.Pool(self.process_nb) as pool:
            list(tqdm(pool.istarmap(self.mirror_or_copy, input_tuples),
                      total=self.in_img_nb,
                      desc='Mirror',
                      file=sys.stdout,
                      smoothing=0))

    @staticmethod
    def mirror_or_copy(mirror_bool, in_path, out_path):

        if mirror_bool:
            pro.read_process_save(in_path, out_path, pro.mirror_img)
        else:
            shutil.copy(in_path, out_path)


class SetScale(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = True
    MP = True

    def process_set(self, spacing='min', interp_type=1, use_voxels=False, smoothing_sigma=None):

        if spacing in ['min', 'max']:
            spacings = np.array([])
            for path in self.in_fpaths:
                img_spacing = np.array(pro.get_spacing(path))
                spacings = np.vstack([spacings, img_spacing]) \
                    if spacings.size else img_spacing

            if spacing == 'min':
                spacing = np.amin(spacings, 0)
            else:
                spacing = np.amin(spacings, 0)

        input_tuples = zip(self.in_fpaths, self.out_fpaths,
                           repeat(pro.resample_img), repeat(spacing),
                           repeat(use_voxels), repeat(interp_type),
                           repeat(smoothing_sigma))

        self.process_tasks(
            args_iter=input_tuples,
            function=pro.read_process_save,
            task_nb=self.in_img_nb,
            description='Scale',
        )


class SetCleanBackground(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = True
    MP = True

    def process_set(
            self,
            min_width,
            threshold,
            component=0,
            mask_channels=False,
            smoothing=None,
    ):

        input_tuples = zip(
            self.in_fpaths,
            self.out_fpaths,
        )

        kwargs_dict = {
            'process_func': pro.remove_small_objects_img,
            'min_width': min_width,
            'threshold': threshold,
            'component': component,
            'mask_channels': mask_channels,
            'smoothing': smoothing,
        }

        self.process_tasks(
            args_iter=input_tuples,
            function=pro.read_process_save,
            task_nb=self.in_img_nb,
            description='Clean',
            kwargs_dict=kwargs_dict,
        )


class SetCrop(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = True
    MP = True

    def process_set(self, margins, **kwargs):

        input_tuples = zip(
            self.in_fpaths,
            self.out_fpaths,
            repeat(pro.smartcrop_img),
            repeat(margins),
        )

        self.process_tasks(
            args_iter=input_tuples,
            kwargs_dict=kwargs,
            function=pro.read_process_save,
            task_nb=self.in_img_nb,
            description='Crop'
        )


class SetHarmonizeIntensity(sha.AbstractSetProcess):

    NEED_TMP = True
    MP = True

    def process_set(
            self,
            num_bins=20,
            num_points=10,
            channel=None,
            plot_hist=False,
            by_type=False,
            ref_filters=None,
            prescale=False,
            threshold=None,
    ):

        if self.input_nb == 1:
            existing_ref = False

        else:

            if self.in_img_nb[1] != 1:
                raise ValueError(
                    "The second input directory should contain a single "
                    + "image (after name filtering).")

            ref_fpath = self.in_fpaths[1][0]
            existing_ref = True

        # define groups based on type or make up a single group
        if by_type:
            if existing_ref:
                raise ValueError(
                    "Harmonization cannot be done by type with an existing " +
                    "reference.")

            types = self.set_info[0]['type'].tolist()
            unq_types = list(set(types))

            ixs_groups = []

            for t in unq_types:
                type_ixs = self.set_info[0][
                    self.set_info[0]['type'] == t].index.tolist()
                ixs_groups.append(type_ixs)
        else:
            ixs_groups = [self.set_info[0].index.tolist()]
            unq_types = ['notype']

        # process by group
        for i_g, group_ixs in enumerate(ixs_groups):

            group_in_fpaths = [self.in_fpaths[0][ix] for ix in group_ixs]
            group_out_fpaths = [self.out_fpaths[ix] for ix in group_ixs]

            group_img_nb = len(group_in_fpaths)

            input_tuples = zip(
                group_in_fpaths,
            )

            kwargs_dict = {
                'channel': channel,
                'threshold': threshold,
            }

            outputs = self.process_tasks(
                args_iter=input_tuples,
                function=self.load,
                task_nb=group_img_nb,
                description='Load',
                kwargs_dict=kwargs_dict,
            )

            vals_list = [o[0] for o in outputs]
            max_list = [o[1] for o in outputs]
            mean_list = [o[2] for o in outputs]

            in_range_max = max(max_list)

            in_edges = np.histogram_bin_edges(
                None,
                bins=num_bins,
                range=(0.0, in_range_max),
            )

            input_tuples = zip(
                vals_list,
            )

            kwargs_dict = {
                'edges': in_edges,
            }

            hist_L = self.process_tasks(
                args_iter=input_tuples,
                function=self.compute_hist,
                task_nb=group_img_nb,
                description='Compute histograms',
                kwargs_dict=kwargs_dict,
            )

            # stack all normalized histograms
            group_hists_array = np.stack(hist_L, axis=0)

            if not existing_ref:

                sc_vals_list = []
                sc_max_list = []

                if prescale:
                    for i, m in enumerate(mean_list):
                        sc_vals_list[i] = vals_list[i] / m
                        sc_max_list[i] = max_list[i] / m
                    sc_max_range = max(sc_max_list)

                    sc_in_edges = np.histogram_bin_edges(
                        None,
                        bins=num_bins,
                        range=(0.0, sc_max_range),
                    )

                    input_tuples = zip(
                        sc_vals_list,
                    )

                    kwargs_dict = {
                        'edges': sc_in_edges,
                    }

                    sc_hist_L = self.process_tasks(
                        args_iter=input_tuples,
                        function=self.compute_hist,
                        task_nb=group_img_nb,
                        description='Compute histograms',
                        kwargs_dict=kwargs_dict,
                    )

                    # stack all normalized histograms
                    sc_group_hists_array = np.stack(sc_hist_L, axis=0)

                    ref_hist = np.mean(sc_group_hists_array, axis=0)
                    ref_edges = sc_in_edges / sc_max_range * in_range_max

                else:

                    # average all histograms
                    ref_hist = np.mean(group_hists_array, axis=0)
                    ref_edges = in_edges

            else:

                vals, ref_range_max, _ = self.load(
                    ref_fpath,
                    channel=channel,
                    threshold=threshold,
                )
                ref_edges = np.histogram_bin_edges(
                    None,
                    bins=num_bins,
                    range=(0.0, ref_range_max)
                )
                ref_hist = self.compute_hist(vals, ref_edges)
                print('ref_range_max', ref_range_max)

            args_iter = zip(
                group_in_fpaths,
                group_out_fpaths,
            )

            kwargs_dict = {
                'channel': channel,
                'ref_hist': ref_hist,
                'ref_edges': ref_edges,
                'num_points': num_points,
                'threshold': threshold,
            }

            matched_hists = self.process_tasks(
                args_iter=args_iter,
                function=self.harmonize_img,
                kwargs_dict=kwargs_dict,
                task_nb=group_img_nb,
                description='Match histograms',
                # process_nb=1,
            )

            if plot_hist:
                hist_dir = self.out_dpath + 'hist_figs/'
                os.makedirs(hist_dir, exist_ok=True)

                names = [self.set_info[0]['name'][ix] for ix in group_ixs]

                fig_paths = [hist_dir + self.prefix + n + '.png'
                             for n in names]

                input_tuples = zip(
                    group_hists_array,
                    repeat(ref_hist),
                    matched_hists,
                    repeat(ref_edges),
                    repeat(in_edges),
                    fig_paths,
                )

                self.process_tasks(
                    args_iter=input_tuples,
                    function=self.compare_histograms,
                    task_nb=group_img_nb,
                    description='Plot histograms')

    @staticmethod
    def load(path, channel, threshold):

        img = pro.read_img(path)

        if pro.get_channel_nb(path) > 1:
            img = ants.split_channels(img)[channel]

        vals = img.numpy().ravel()

        thres_val = pro.MatchHistImg.get_threshold_value(vals, threshold)
        thres_lgcs = vals > thres_val

        # if threshold is not None:

        #     if np.isscalar(threshold):
        #         thres_lgcs = (vals > threshold)
        #     elif threshold == 'mean':
        #         thres_lgcs = (vals > np.mean(vals))
        #     else:
        #         raise ValueError('Incorrect value of threshold argument.')

        vals = vals[thres_lgcs]

        return vals, vals.max(), vals.mean()  # np.percentile(vals, 99)

    @staticmethod
    def compute_hist(vals, edges):

        hist, _ = np.histogram(vals, bins=edges)

        # normalize
        hist = hist / hist.sum()

        return hist

    def harmonize_img(
            self,
            in_fpath,
            out_fpath,
            channel,
            ref_hist,
            ref_edges,
            num_points,
            threshold,
    ):

        img = pro.read_img(in_fpath)

        out_img = pro.MatchHistImg(img, channel=channel).run(
            ref_hist_comps=(ref_hist, ref_edges),
            pt_nb=num_points,
            threshold=threshold,
        )

        ants.image_write(out_img, out_fpath)

        out_vals, _, _ = self.load(out_img, channel, threshold)
        matched_hist = self.compute_hist(out_vals, ref_edges)

        return matched_hist

    def compare_histograms(
            self,
            input_hist,
            ref_hist,
            match_hist,
            ref_edges,
            in_edges,
            save_fpath=None,
    ):

        hist_L = [input_hist, match_hist]
        edges_L = [in_edges, ref_edges]

        labels = ['original', 'matched']

        fig, axs = plt.subplots(2, 1, sharex="col", figsize=(10, 10))

        for i_sp in range(2):

            # plot reference data
            plt.sca(axs[i_sp])
            plt.stairs(ref_hist, ref_edges, color='red', label='reference')

            hist = hist_L[i_sp]
            edges = edges_L[i_sp]
            plt.stairs(hist, edges, color='blue', label=labels[i_sp])

            plt.ylabel('Frequency')
            if i_sp == 1:
                plt.xlabel('Intensity')

        for ax in axs:
            ax.legend()

        if save_fpath:
            from pathlib import Path
            save_dpath = Path(save_fpath).parent
            os.makedirs(save_dpath, exist_ok=True)
            plt.savefig(save_fpath, bbox_inches="tight")
            plt.close(fig)

        return



class SetRegister(sha.AbstractSetProcess):

    NEED_TMP = True
    MP = True

    @sha.profile
    def process_set(
            self,
            params,
            component=0,
            switch_images=False,
            keep_inv_tfs=False,
    ):

        if len(self.in_fpaths[1]) == 1:
            # prepare reference
            ref_fpath = self.in_fpaths[1][0]

            # check if reference file is included in input directory
            ref_bool_L = [p == ref_fpath for p in self.in_fpaths[0]]

            # loading image and write single channel file if necessary
            ref_img, tmp_ref_c_fpath = self.load_split_write(ref_fpath, component)

            # create local copy of complete reference image
            tmp_ref_fpath = tempfile.NamedTemporaryFile(
                prefix=self.tmp_dpath,
                suffix='_ref.nii',
                delete=False).name

            ants.image_write(ref_img, tmp_ref_fpath)

            target_fpaths = repeat(tmp_ref_fpath)
            target_c_fpaths = repeat(tmp_ref_c_fpath)
            target_names = [self.in_names[1][0]] * self.in_img_nb[0]

        else:
            ref_bool_L = repeat(False)

            merge_info = pd.merge(
                self.set_info[0],
                self.set_info[1],
                on='name',
            )

            if len(merge_info) != len(self.set_info[0]):
                raise ValueError('Inputs not matching.')

            print(merge_info)

            target_fpaths = (self.in_dpath[1] + merge_info['path_y']).tolist()
            target_c_fpaths = repeat(None)
            target_names = (merge_info['name']).tolist()

        #  naming transform files
        if switch_images:
            tf_names = [
                target_names[i] + '_to_' + self.in_names[0][i] for i in range(self.in_img_nb[0])
            ]
        else:
            tf_names = [
                self.in_names[0][i] + '_to_' + target_names[i] for i in range(self.in_img_nb[0])
            ]

        args_iter = zip(
            self.in_fpaths[0],
            target_c_fpaths,
            target_fpaths,
            ref_bool_L,
            repeat(component),
            repeat(params),
            repeat(switch_images),
            self.out_fpaths,
            tf_names,
        )

        kwargs_dict = {
            'keep_inv_tfs': keep_inv_tfs,
        }

        results = self.process_tasks(
            args_iter,
            self.split_register_transform,
            task_nb=self.in_img_nb[0],
            description='Register',
            kwargs_dict=kwargs_dict,
        )

        tf_prefixes = [res[0] for res in results]
        log_fpaths = [res[1] for res in results]
        metric_values = [res[2] for res in results]

        # update output info
        self.out_info['metric_value'] = metric_values
        self.out_info['transform_log'] = log_fpaths
        self.out_info['transform_prefix'] = tf_prefixes

        if keep_inv_tfs:
            itf_loc_prefixes = [res[3] for res in results]
            self.out_info['inverse_transform_prefix'] = itf_loc_prefixes

        sha.print_stats()

    def load_split_write(self, fpath, component):

        img = pro.read_img(fpath)

        if component is not None and pro.get_channel_nb(fpath) > 1:

            tmp_c_fpath = tempfile.NamedTemporaryFile(
                prefix=self.tmp_dpath,
                suffix='.nii',
                delete=False).name

            img_c = pro.split_channels(img)[component]

            ants.image_write(img_c, tmp_c_fpath)

        else:

            tmp_c_fpath = fpath

        return img, tmp_c_fpath

    def split_register_transform(
            self,
            fpath,
            tmp_ref_c_fpath,
            tmp_ref_fpath,
            ref_bool,
            component,
            params,
            switch_images,
            out_fpath,
            tf_name,
            keep_inv_tfs,
    ):

        if ref_bool:
            shutil.copyfile(fpath, out_fpath)
            return '', '', None

        # create temporary folder
        func_tmp_dir_obj = tempfile.TemporaryDirectory(prefix=self.tmp_dpath)
        func_tmp_dir = func_tmp_dir_obj.name + '/'

        # func_tmp_dir = self.tmp_dpath + pathlib.Path(fpath).stem + '/'
        if tmp_ref_c_fpath is None:
            _, tmp_ref_c_fpath = self.load_split_write(tmp_ref_fpath, component)

        # loading image and write single channel file if necessary
        img, tmp_c_fpath = self.load_split_write(fpath, component)

        # prepare output transform file directory
        dname = 'transforms'
        tf_dpath = self.out_dpath + dname + '/'
        os.makedirs(tf_dpath, exist_ok=True)
        tf_loc_prefix = 'transforms/' + tf_name

        # prepare output inverse transform file directory
        if keep_inv_tfs:
            dname = 'inv_transforms/'
            itf_dpath = self.out_dpath + dname + '/'
            os.makedirs(itf_dpath, exist_ok=True)
            itf_loc_prefix = dname + tf_name
        else:
            itf_loc_prefix = None

        # temporary transform files
        tf_tmp_dpath = func_tmp_dir + 'transforms/'
        os.makedirs(tf_tmp_dpath, exist_ok=False)
        tmp_tf_prefix = tf_tmp_dpath + tf_name

        # output log file
        out_log_loc_fpath = 'transforms/' + tf_name + '.log'

        f_params = params.copy()

        # temp prefix
        f_params['output_transform_prefix'] = tmp_tf_prefix + '_'

        # registration
        if switch_images:
            reg_res = pro.align_img_nipype(
                tmp_ref_c_fpath, tmp_c_fpath, f_params)
        else:
            reg_res = pro.align_img_nipype(
                tmp_c_fpath, tmp_ref_c_fpath, f_params)

        # paths to transform files are stored (the order of the transforms is
        # reversed to comply with 'apply_transforms')
        tf_fpaths = reg_res.outputs.forward_transforms[::-1]
        metric_value = reg_res.outputs.metric_value

        # write log file
        with open(self.out_dpath + out_log_loc_fpath, 'w+') as f:
            f.write(reg_res.runtime.stdout)

        # transformation of all channels
        if switch_images:
            tf_img = pro.transform_img(tmp_ref_fpath, tf_fpaths, img)
        else:
            tf_img = pro.transform_img(img, tf_fpaths, tmp_ref_fpath)

        # write registered image
        ants.image_write(tf_img, out_fpath)

        # copy transform files
        for tf_fpath in tf_fpaths:
            shutil.copy(tf_fpath, tf_dpath)

        # copy inverse warp file
        if keep_inv_tfs:
            for tf_fpath in reg_res.outputs.reverse_transforms:
                shutil.copy(tf_fpath, itf_dpath)

        func_tmp_dir_obj.cleanup()

        return tf_loc_prefix, out_log_loc_fpath, metric_value, itf_loc_prefix



class SetApplyRegistration(sha.AbstractSetProcess):

    NEED_TMP = True
    MP = True

    def process_set(self, interpolator='linear', channel=None):

        tf_prefixes = self.set_info[1]['transform_prefix']

        args_iter = zip(
            self.in_fpaths[0],
            tf_prefixes,
            self.out_fpaths,
        )

        kwargs_dict = {
            'interpolator': interpolator,
            'channel': channel,
        }

        self.process_tasks(
            args_iter,
            self.read_transform_write,
            task_nb=self.in_img_nb[0],
            description='Apply registration',
            kwargs_dict=kwargs_dict,
        )

    def read_transform_write(
            self,
            in_fpath,
            tf_prefix,
            out_fpath,
            interpolator,
            channel
    ):

        img = pro.read_img(in_fpath)

        if channel is not None:
            img = pro.split_channels(img)[channel]

        tf_fpaths = glob.glob(self.in_dpath[1] + tf_prefix + '_[0-1]*')
        tf_fpaths = sorted(tf_fpaths)[::-1]

        tf_img = pro.transform_img(
            img,
            tf_fpaths,
            ref_img=None,
            interpolator=interpolator,
        )

        ants.image_write(tf_img, out_fpath)



class SetBuildTemplate(sha.AbstractSetProcess):

    NEED_TMP = True
    MP = True
    SINGLE_INPUT_ONLY = True

    def process_set(
            self,
            params,
            iterations=3,
            gradient_step=0.2,
            blending_weight=0.75,
            initial_template=None,
            update_nb=1,
            component=0,
            avg_mode='mean'
    ):

        # check available memory in temp directory
        ut.print_memory_check(self.tmp_dpath)

        # copy files or create temporary file with selected channel only
        tmp_img_fpaths = [self.tmp_dpath + p for p in self.in_loc_fpaths]

        input_tuples = zip(self.in_fpaths, tmp_img_fpaths, repeat(component))

        # print('self.in_fpaths', self.in_fpaths)

        results = self.process_tasks(
            input_tuples,
            pro.split_write,
            task_nb=self.in_img_nb,
            description='Prepare',
            kwargs_dict={},
        )

        # check available memory in temp directory
        ut.print_memory_check(self.tmp_dpath)

        # prepare paths to store initial, intermediate and final templates
        output_name = 'template'
        if self.prefix:
            output_name = self.prefix + '_template'
        template_paths = []
        template_loc_paths = []
        template_names = []
        for i in range(iterations):

            name = output_name + '_' + str(i) + 'it'

            if i == iterations:
                name += '_final'

            template_paths.append(self.tmp_dpath + name + '.nii')
            template_loc_paths.append(name + '.nii')
            template_names.append(name)

        # create initial template
        if initial_template is None:
            # pro.avg_imgs_nipype(tmp_img_fpaths, template_paths[0])
            initial_template = pro.avg_imgs(
                tmp_img_fpaths,
                mode=avg_mode,
            )
        else:
            initial_template = pro.read_img(initial_template)

        ants.image_write(initial_template, template_paths[0])

        # prepare temporary storage of warped images
        tf_prefixes = [self.tmp_dpath + 'w1_' + n + '_'
                       for n in self.in_names]
        warped_fpaths = [self.tmp_dpath + 'warped_' + n + '.nii'
                         for n in self.in_names]

        # prepare temporary storage of intermediate objects
        xavg_path = self.tmp_dpath + 'xavg.nii'  # average image
        wavg_path = self.tmp_dpath + 'wavg.nii'  # average field
        wup_path = self.tmp_dpath + 'wup.nii'  # update field
        afavg_path = self.tmp_dpath + 'afavg.mat'  # update field

        wup_mds = []
        iter_timer = sha.Timer(iterations, 'Iteration')

        # check available memory in temp directory
        ut.print_memory_check(self.tmp_dpath)

        for i in range(iterations):
            print('Start iteration ' + str(i), datetime.now())

            warp_tf_paths = []
            affine_tf_paths = []

            input_tuples = zip(
                tmp_img_fpaths,
                repeat(template_paths[i]),
                repeat(params),
                tf_prefixes,
                warped_fpaths)

            results = self.process_tasks(
                input_tuples,
                self.register_to_template,
                task_nb=self.in_img_nb,
                description='Register',
                kwargs_dict={},
                # process_nb=1,
            )

            # check available memory in temp directory
            ut.print_memory_check(self.tmp_dpath)

            warp_tf_paths = [res[0] for res in results]
            affine_tf_paths = [res[1] for res in results]

            # average all warped images in new xavg
            xavg = pro.avg_imgs(
                warped_fpaths,
                mode=avg_mode
            )
            ants.image_write(xavg, xavg_path)

            # average all warp fields in wavg
            pro.avg_imgs_nipype(warp_tf_paths, wavg_path)

            # compute update warp field wup using average warp field
            wscl = -1.0 * gradient_step
            mult = nipype_ants.MultiplyImages()
            mult.terminal_output = 'allatonce'
            mult.inputs.dimension = 3
            mult.inputs.first_input = wavg_path
            mult.inputs.second_input = wscl
            mult.inputs.output_product_image = wup_path
            mult.inputs.num_threads = params['num_threads']
            mult.run()

            # compute mean amplitude of the update warp field
            wup_itk = itk.imread(wavg_path, itk.Vector[itk.F, 3])
            wup_mat = itk.array_from_image(wup_itk)
            wup_md = np.linalg.norm(wup_mat, ord=2, axis=3).mean()
            print('Average field mean displacement: '
                  + str(wup_md))
            wup_mds.append(wup_md)

            if i == iterations - 1:
                break

            # average all affine transforms
            avgtf = nipype_ants.AverageAffineTransform()
            avgtf.terminal_output = 'allatonce'
            avgtf.inputs.dimension = 3
            avgtf.inputs.output_affine_transform = afavg_path
            avgtf.inputs.transforms = affine_tf_paths
            avgtf.inputs.num_threads = params['num_threads']
            avgtf.run()

            # apply inverse average affine transform to update field
            at = nipype_ants.ApplyTransforms()
            at.inputs.dimension = 3
            at.inputs.input_image = wup_path
            at.inputs.reference_image = wup_path
            at.inputs.output_image = wup_path
            at.inputs.input_image_type = 1  # vector type
            at.inputs.transforms = [afavg_path]
            at.inputs.invert_transform_flags = [True]
            at.inputs.num_threads = params['num_threads']
            at.run()

            # update template
            at = nipype_ants.ApplyTransforms()
            at.terminal_output = 'allatonce'
            at.inputs.input_image = xavg_path
            at.inputs.reference_image = xavg_path
            at.inputs.transforms = [wup_path] * update_nb + [afavg_path]
            at.inputs.invert_transform_flags = [False] * update_nb + [True]
            at.inputs.output_image = template_paths[i + 1]
            at.inputs.num_threads = params['num_threads']
            at.run()

            # sharpen current template
            if blending_weight is not None:
                template_img = pro.read_img(template_paths[i + 1])
                template_img = (
                    template_img * blending_weight
                    + ants.iMath(template_img, "Sharpen")
                    * (1.0 - blending_weight)
                )
                ants.image_write(template_img, template_paths[i + 1])

            print(iter_timer.remains(i + 1))

            # check available memory in temp directory
            ut.print_memory_check(self.tmp_dpath)

        # add best template info
        min_ix = np.argmin(wup_mds)
        best_name = template_names[min_ix] + '_best'
        template_names.append(best_name)
        best_loc_path = best_name + '.nii'
        template_loc_paths.append(best_loc_path)

        # copy best template
        shutil.copy(template_paths[min_ix], self.out_dpath + best_loc_path)

        # copy templates to output folder
        for p in template_paths:
            shutil.copy(p, self.out_dpath)

        self.out_info = pd.DataFrame(
            {
                "name": template_names,
                "path": template_loc_paths,
                "hemisphere": ['T'] * len(template_names)
            })

        # numpy types are not compatible with json: conversion to float
        wup_mds = [float(v) for v in wup_mds]
        tpl_convergence_data = {'Average field mean displacement': wup_mds}

        with open(self.out_dpath + 'tpl_convergence_data.json', 'w') as f:
            json.dump(tpl_convergence_data, f, indent=4)

    @staticmethod
    def register_to_template(
            tmp_img_fpath,
            template_path,
            params,
            tf_prefix,
            warped_fpath
    ):

        params['output_transform_prefix'] = tf_prefix
        params['output_warped_image'] = warped_fpath

        reg_res = pro.align_img_nipype(
            tmp_img_fpath, template_path, params)

        reg_outputs = reg_res.outputs

        # SyN transformation
        warp_tf_path = reg_outputs.forward_transforms[1]
        # affine transformation
        aff_tf_path = reg_outputs.forward_transforms[0]

        # remove inverse warp file (unused) to save space
        inv_path = glob.glob(tf_prefix + '*InverseWarp.nii.gz')[0]

        os.remove(inv_path)

        return warp_tf_path, aff_tf_path



class SetAverage(sha.AbstractSetProcess):

    NEED_TMP = False
    MP = True
    SINGLE_INPUT_ONLY = True

    def process_set(
            self,
            mode='mean',
            channel=None,
            nz_mask=False,
    ):

        out_fname = 'avg'
        out_file = out_fname + '.nii'
        out_fpath = self.out_dpath + out_file

        avg_img = pro.avg_imgs(
            self.in_fpaths,
            component=channel,
            mode=mode,
            split_nb=100,  # self.process_nb,
            process_nb=self.process_nb,
            nz_mask=nz_mask,
        )
        ants.image_write(avg_img, out_fpath)

        self.out_info = pd.DataFrame(
            {
                "name": [out_fname],
                "path": [out_file],
                "type": ['avg'],
                "hemisphere": ['avg'],
            })



class SetTypeAverage(sha.AbstractSetProcess):

    NEED_TMP = True
    MP = False
    SINGLE_INPUT_ONLY = True

    def process_set(
            self,
            # template_dpath=None,
            # template_name=None,
            component=None,
            mask=None,
            type_weighted_avg=True,
            mode='mean',
            nz_mask=False,
            combined_img=True,
            primes_corr=None,
            smoothing_sigma=None,
    ):

        # 1] Average images
        types = sorted(list((set(self.set_info['type']))))

        type_info_L = []
        for t in types:
            type_info_L.append(self.set_info[self.set_info['type'] == t])

        output_names = []
        output_filenames = []
        type_L = []
        hem_L = []

        hem = self.set_info['hemisphere'][0]

        avg_imgs = []

        for (type_info, t) in zip(type_info_L, types):

            in_paths = (self.in_dpath + type_info.path).tolist()
            output_name = t + '_avg'
            out_filename = output_name + '.nii'

            output_names.append(output_name)
            output_filenames.append(out_filename)

            # average
            avg_img = pro.avg_imgs(
                in_paths,
                component=component,
                mode=mode,
                nz_mask=nz_mask
            )

            # smooth
            if smoothing_sigma is not None:
                avg_img = ants.utils.smooth_image(
                    avg_img,
                    sigma=smoothing_sigma,
                    sigma_in_physical_coordinates=False,
                    FWHM=False,
                    max_kernel_width=4*smoothing_sigma,
                    )

            ants.image_write(avg_img, self.out_dpath + out_filename)
            avg_imgs.append(avg_img)
            type_L.append(t)
            hem_L.append(hem)

        if primes_corr == 'AB':
            avg_imgs[types.index('ApBp')] = avg_imgs[types.index('AB')]
        elif primes_corr is None:
            pass
        else:
            raise ValueError("Unknown 'prime_corr' value :" + str(primes_corr))

        if type_weighted_avg:
            weighted_avg_imgs = []

            # 2] Create average KC file
            neuron_ratios = par.get_neuron_ratios()
            neuron_ratios = np.array([neuron_ratios[ty] for ty in types])

            if mask is not None:
                mask = pro.read_img(mask)
                mask = mask.numpy().astype('bool')

            kc_avg = None
            for i, _ in enumerate(types):

                stack = avg_imgs[i].numpy()

                # apply mask
                if mask is not None:
                    stack[np.logical_not(mask)] = 0.0

                stack /= stack.sum()

                weighted_avg_img = neuron_ratios[i] * stack
                weighted_avg_imgs += [weighted_avg_img]

                if kc_avg is None:
                    kc_avg = weighted_avg_img.copy()
                else:
                    kc_avg += weighted_avg_img

            kc_avg_img = avg_imgs[0].new_image_like(kc_avg)

            output_name = 'KC_avg'
            out_filename = output_name + '.nii'

            output_names.append(output_name)
            output_filenames.append(out_filename)
            type_L.append('KC')
            hem_L.append(hem)

            ants.image_write(kc_avg_img, self.out_dpath + out_filename)

            # print((kc_avg > 0.0).mean())
            # sys.exit()

            # compute proportions
            kc_mask = kc_avg > 0.0
            mask_n = (kc_mask > 0.0).sum()

            densities = np.stack(weighted_avg_imgs, axis=-1)
            proportions = densities.copy()
            # proportions[kc_mask] = np.divide(
            #     densities[kc_mask],
            #     np.expand_dims(kc_avg[kc_mask], -1),
            #     )

            # print('densities[kc_mask].shape', densities[kc_mask].shape)
            # print('np.sum(densities[kc_mask], -1, keepdims=True).shape', np.sum(densities[kc_mask], -1, keepdims=True).shape)
            proportions[kc_mask] = np.divide(
                densities[kc_mask],
                np.sum(densities[kc_mask], -1, keepdims=True),
            )

            # # refine proportions
            # for i_r in range(5):
            #     # colum-wise normalization
            #     proportions[kc_mask] /= proportions[kc_mask].sum(axis=0, keepdims=True)
            #     proportions[kc_mask] *= np.expand_dims(neuron_ratios, axis=0)
            #     # row-wise normalization
            #     proportions[kc_mask] /= proportions[kc_mask].sum(axis=1, keepdims=True)

            col = par.get_colors()
            colors = [col['red'], col['green'], col['blue']]

            fig, axs = plt.subplots(3, 1, dpi=150, sharex=True, sharey=True, figsize=(4, 4))
            for i in range(3):

                a = proportions[:, :, :, i]
                # plt.hist(a[kc_mask].ravel(), range=(0.0, 1.0), bins=20, color=colors[i])

                # compute distribution
                vals, edges = np.histogram(a[kc_mask], bins=20, range=(0, 1))
                vals = vals / vals.sum()
                x = (edges[0:-1] + edges[1:]) / 2
                w = edges[1] - edges[0]

                # plot
                plt.sca(axs[i])
                plt.bar(x, vals, width=w, alpha=1, align='center', color=colors[i])

                plt.axvline(neuron_ratios[i], 0, 1, linestyle="--", color='k')

                plt.ylabel('% voxels')

            plt.xlabel('% subpopulation')

            # prepare entropy computation
            tmp = np.multiply(
                proportions[kc_mask, :],
                np.log2(proportions[kc_mask, :] / np.expand_dims(neuron_ratios, 0))
            )
            tmp[proportions[kc_mask] == 0.0] = 0.0
            # voxel_entropies = -tmp.sum(axis=-1)

            # compute subpop entropies
            subpop_entropies = tmp.sum(axis=0) / mask_n

            # compute combined entropy
            combined_entropy = tmp.sum() / mask_n

            # prepare entropy list
            entropies = np.append(subpop_entropies, 0.0)

            # # compute combined entropy
            # kc_mask = (kc_avg > 0.0)
            # densities = np.stack(weighted_avg_imgs, axis=-1)
            # proportions = densities.copy()
            # proportions[kc_mask] = np.divide(
            #     densities[kc_mask],
            #     np.expand_dims(kc_avg[kc_mask], axis=-1)
            #     )
            # tmp = proportions.copy()
            # tmp[kc_mask] = np.multiply(
            #     proportions[kc_mask],
            #     np.log2(proportions[kc_mask])
            #     )
            # tmp[proportions == 0.0] = 0.0
            # entropies = -tmp.sum(axis=-1)
            # entropy = entropies[kc_mask].sum()

            # store entropy
            norm_props = proportions.copy()
            # print(norm_props.min(axis=(0, 1, 2)))
            # norm_props = norm_props - norm_props.min(axis=(0, 1, 2))
            # print(norm_props.max(axis=(0, 1, 2)))
            # norm_props = np.divide(
            #     norm_props,
            #     norm_props.max(axis=(0, 1, 2))
            #     )
            # norm_props[kc_mask] = np.divide(
            #     norm_props[kc_mask],
            #     norm_props[kc_mask].sum(axis=-1, keepdims=True)
            #     )

            norm_props_img = ants.from_numpy(
                norm_props,
                spacing=avg_imgs[0].spacing,
                has_components=True,
            )

            # save proportion images
            output_name = 'proportions'
            out_filename = output_name + '.nii'
            ants.image_write(norm_props_img, self.out_dpath + out_filename)
            output_names.append(output_name)
            output_filenames.append(out_filename)
            type_L.append('All')
            hem_L.append(hem)
            entropies = np.append(entropies, combined_entropy)

            # create proportion image
            vis.plot_maxproj(
                norm_props_img,
                norm='minmax',
                mode='mean',
                legend=types,
                # white_zeros=True,
            )
            # remove colorbar
            # cb_h = plt.colorbar()
            # cb_h.remove()

            plt.suptitle('Proportions: mean voxel entropy = ' + str(combined_entropy))
            plt.savefig(self.out_dpath + 'proportions.png')

        # 3] Create map file
        # if template_dpath and template_name:
            # tpl_info = pd.read_csv(template_dpath + 'files.csv')
            # bool_V = [fnmatch.fnmatch(tpl_info['name'][i], template_name)
            #           for i in range(len(tpl_info))]
            # tpl_fpath = template_dpath + tpl_info['path'][bool_V].tolist()[0]
            # tpl_fpath = tpl_info['path'][bool_V].tolist()[0]

            # tpl_img = pro.read_img(tpl_fpath)

            # map_img = pro.merge_channels([tpl_img] + avg_imgs)
        if combined_img:
            weighted_img = ants.from_numpy(
                densities,
                spacing=avg_imgs[0].spacing,
                has_components=True,
            )
            map_img = weighted_img  # !!!
            # map_img = pro.merge_channels(avg_imgs)

            output_name = 'combined_'
            for t in types:
                output_name += t + '_'
            output_name += 'maps'
            out_filename = output_name + '.nii'
            output_names.append(output_name)
            output_filenames.append(out_filename)
            type_L.append('All')
            hem_L.append(hem)
            entropies = np.append(entropies, combined_entropy)

            ants.image_write(map_img, self.out_dpath + out_filename)

        self.out_info = pd.DataFrame({
            "name": output_names,
            "path": output_filenames,
            "type": type_L,
            "hemisphere": hem_L,
        })

        if type_weighted_avg and combined_img:
            self.out_info['entropy'] = entropies.tolist()



class SetComputeSimilarity(sha.AbstractSetProcess):

    SAVE_INFO = False
    NEED_TMP = True
    MP = True

    def process_set(
            self,
            metrics,
            component=1,
            pairing='all',
            subset=None,
            tf_params=None,
            seed=None,
            sim_threshold=None,
            by_type=False,
            add_column=None,
    ):

        # if single input, duplicate it
        if self.input_nb == 1:
            self.set_info = self.set_info * 2
            self.in_img_nb = self.in_img_nb * 2
            self.in_fpaths = self.in_fpaths * 2

            if pairing is None:
                pairing = 'combine'

        if by_type:
            # extract types
            types = list(set(self.set_info[0]['type'].tolist()))
            types = sorted(types)
        else:
            types = [None]

        type_vals_list = []

        if pairing == 'random_ref':
            ref_names = []

        for t in types:

            # extract type-specific info
            type_fpaths_L = []
            type_info_L = []
            img_nbs = []
            for i in range(2):
                if (t is None) or (pairing == 'single_ref' and i == 1):
                    img_nb = self.in_img_nb[i]
                    type_fpaths = self.in_fpaths[i]
                    type_info = self.set_info[i]

                else:
                    bool_V = (self.set_info[i]['type'] == t).tolist()

                    img_nb = sum(bool_V)
                    type_fpaths = [p for (p, b) in zip(
                        self.in_fpaths[i], bool_V) if b]
                    type_info = self.set_info[i][bool_V]

                type_info_L.append(type_info)
                img_nbs.append(img_nb)
                type_fpaths_L.append(type_fpaths)

            if pairing == 'match':
                if img_nbs[0] != img_nbs[1]:
                    raise ValueError(
                        "The number of images should be the same in both groups for pairing mode 'match'.")
                pair_ixs = np.tile(np.arange(img_nbs[0]), (2, 1))

            elif pairing == 'all':
                ixs = [np.arange(img_nbs[i]) for i in range(2)]
                pair_ixs = np.array(np.meshgrid(*ixs)).reshape((2, -1))

            elif pairing == 'combine':
                if img_nbs[0] != img_nbs[1]:
                    raise ValueError(
                        "The number of images should be the same in both groups for pairing mode 'combine'.")
                pair_ixs = np.stack(np.triu_indices(
                    img_nbs[0], k=1), axis=-1).T

            elif pairing == 'random_ref':
                if self.input_nb != 1:
                    raise ValueError(
                        "There should be only one input directory for pairing mode 'random_ref'.")

                # select random indices
                #  selection is done so that with a given seed, analyses with
                #  prefixes 'randref(i)_' generate different indices
                rng = np.random.default_rng(seed)
                rand_ixs = rng.choice(img_nbs[0], img_nbs[0])
                prefix_rand_nb = sum([ord(c) for c in self.prefix])
                rand_ix = rand_ixs[np.mod(prefix_rand_nb, img_nbs[0])]

                other_ixs = np.arange(img_nbs[0])
                other_ixs = np.delete(other_ixs, rand_ix)

                pair_ixs = np.stack(
                    [other_ixs, rand_ix * np.ones(len(other_ixs))])
                ref_name = type_info_L[1]['name'].iloc[rand_ix]
                ref_names.append(ref_name)

            elif pairing == 'single_ref':
                pair_ixs = np.concatenate(
                    (np.arange(img_nbs[0]).reshape(1, -1),
                     np.zeros((1, img_nbs[0]))),
                    axis=0)

            pair_nb = pair_ixs.shape[1]
            if subset:
                if subset < 1:
                    subset = int(subset * pair_nb)
                subset_ixs = np.random.choice(pair_nb, subset, replace=False)
                pair_ixs = pair_ixs[:, subset_ixs]
                pair_nb = subset

            paths = []
            for i in range(2):
                paths.append([type_fpaths_L[i][int(ix)] for ix in pair_ixs[i]])

            args_iter = zip(
                paths[0],
                paths[1],
                repeat(pairing),
                repeat(metrics),
                repeat(component),
                repeat(tf_params),
                repeat(self.tmp_dpath),
                repeat(sim_threshold)
            )

            type_sim_vals = self.process_tasks(
                args_iter=args_iter,
                function=self.compute_similarity,
                task_nb=pair_nb,
                description='Compute similarity')

            type_vals_list.append(type_sim_vals)

        # add data to json file
        res_data = {'values': type_vals_list,
                    'types': types,
                    'metrics': metrics}

        if pairing == 'random_ref':
            res_data['ref_names'] = ref_names

        res_data_entry = {(self.prefix + 'data'): res_data}
        json_path = self.out_dpath + 'similarity_data.json'

        if os.path.isfile(json_path) and os.stat(json_path).st_size != 0:
            with open(json_path, 'r') as json_file:
                file_data = json.load(json_file)
        else:
            file_data = {}

        file_data.update(res_data_entry)

        with open(json_path, 'w') as json_file:
            json.dump(file_data, json_file, indent=4)

        if add_column is not None:
            if pairing == 'single_ref':
                data = type_vals_list[0]
                data = [d[0] for d in data]
                info_fpath = self.in_dpath[0] + 'files.csv'
                set_info = pd.read_csv(info_fpath, index_col=0)
                set_info[add_column] = data
                set_info.to_csv(info_fpath)

            else:
                raise ValueError(
                    "Column cannot only be added in '"
                    + str(pairing) + "' mode."
                )

    @staticmethod
    def compute_similarity(path_1, path_2, pairing, metrics, component, tf_params, tmp_dpath, mask_threshold):

        if pairing == 'random_ref':
            # register image path_1 on image path_2
            tf_params['output_transform_prefix'] = tmp_dpath
            warped_fpath_1 = tmp_dpath + 'wp_img1.nii'
            tf_params['output_warped_image'] = warped_fpath_1
            pro.align_img_nipype(path_1, path_2, tf_params)

            # apply masking based on img2 on second channel of img1
            img2_chans = pro.split_channels(pro.read_img(path_2))
            mask = img2_chans[0].numpy() > 0

            wp_img1_chan1 = pro.split_channels(pro.read_img(warped_fpath_1))[1]
            wp_img1_chan1_stack = wp_img1_chan1.numpy()
            wp_img1_chan1_stack[not mask] = 0

            wp_img1_chan1 = wp_img1_chan1.new_image_like(wp_img1_chan1_stack)

            # prepare single-channel images
            path_1 = wp_img1_chan1
            path_2 = img2_chans[1]
            component = 0

        sim_vals = pro.compute_similarity_metrics(
            path_1, path_2, metrics, component=component, mask_threshold=mask_threshold)

        return sim_vals



class SetNormalize(sha.AbstractSetProcess):

    MP = True
    SINGLE_INPUT_ONLY = True

    def process_set(self, mode, value, component=None):

        if value == 'type_neuron_nb':
            type_neuron_nbs = par.get_neuron_nbs()

            values = []
            for ty in self.set_info['type'].tolist():
                values.append(type_neuron_nbs[ty])

        elif value == 'unit':
            values = [1.0] * self.in_img_nb

        else:
            raise ValueError('Value ' + str(value) + ' unsupported.')

        args_iter = zip(self.in_fpaths,
                        self.out_fpaths,
                        repeat(mode),
                        values,
                        repeat(component))

        self.process_tasks(
            args_iter,
            self.read_normalize_write,
            self.in_img_nb,
            description='Normalize'
        )

        # with multiprocessing.Pool(self.process_nb) as pool:
        #     list(tqdm(pool.istarmap(self.read_normalize_write, input_tuples),
        #               total=self.in_img_nb, desc='Normalize', file=sys.stdout))

    @staticmethod
    def read_normalize_write(in_fpath, out_fpath, mode, value, component):

        img = pro.read_img(in_fpath)

        img = pro.NormalizeImg(img, channel=component).run(
            func=mode,
            factor=value,
        )

        ants.image_write(img, out_fpath)

        return



class SetMask(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = False
    MP = True

    def process_set(
            self,
            mask_dpath=None,
            channel=None,
            mask_ratio=False,
            **kwargs
    ):

        if self.input_nb == 2:

            merge_info = pd.merge(
                self.set_info[0],
                self.set_info[1],
                on='name',
                suffixes=(None, '_mask'),
                )

            if len(merge_info) != len(self.set_info[0]):
                raise Exception('Missing mask.')

            self.in_fpaths[1] = (self.in_dpath[1] + merge_info['path_mask']).tolist()

            args_iter = zip(
                self.in_fpaths[0],
                self.out_fpaths,
                self.in_fpaths[1],
            )

        else:
            mask_fpath = pro.select_img(mask_dpath, 0)
            mask = pro.read_img(mask_fpath).numpy()
            mask = pro.binarize_array(mask)

            args_iter = zip(
                self.in_fpaths[0],
                self.out_fpaths,
                repeat(mask),
            )

        kwargs_dict = kwargs
        kwargs_dict.update({
            'channel': channel,
            'mask_ratio': mask_ratio,
        })

        print(kwargs_dict)

        img_nb = self.in_img_nb[0]

        ratios = self.process_tasks(
            args_iter=args_iter,
            function=self.read_mask_write,
            task_nb=img_nb,
            description='Mask',
            waitbar=True,
            kwargs_dict=kwargs_dict,
        )

        if mask_ratio:
            self.out_info['mask_ratio'] = ratios

    @staticmethod
    def read_mask_write(in_fpath, out_fpath, mask, channel, mask_ratio, **kwargs):

        # if not isinstance(mask, np.ndarray):
        #     mask = pro.read_img(mask)

        # if channel is not None:
        #     mask = mask[:, :, :, channel]

        img = pro.read_img(in_fpath)

        # if mask.shape != img.shape[:3]:
        #     print(in_fpath, img.shape, mask.shape)

        mask_img = pro.MaskImg(img).run(mask=mask, **kwargs)
        ants.image_write(mask_img, out_fpath)

        # specific to computation of masked channel 1 signal ratio
        if mask_ratio:
            ratio = pro.split_channels(mask_img)[1].sum() \
                / pro.split_channels(img)[1].sum()
        else:
            ratio = 0.0

        return ratio


class SetDiscretize(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = True
    MP = True

    def process_set(self, tpl_mask_path, component):

        mask_fpath = pro.select_img(tpl_mask_path, 'mask_*')
        mask = pro.read_img(mask_fpath).numpy()
        px_nb = np.count_nonzero(mask)

        neuron_ratios = par.get_neuron_ratios()
        types = list(neuron_ratios.keys())
        type_px_nbs = neuron_ratios.copy()
        for ty in types:
            type_px_nbs[ty] = round(neuron_ratios[ty] * px_nb)

        img_types = self.set_info['type'].tolist()

        input_tuples = zip(
            self.in_fpaths,
            self.out_fpaths,
            img_types,
            repeat(type_px_nbs),
            repeat(component)
        )

        self.process_tasks(
            input_tuples,
            self.read_discretize_write,
            self.in_img_nb,
            description='Discretize',
            waitbar=True
        )

    @staticmethod
    def read_discretize_write(in_fpath, out_fpath, img_type, type_px_nbs,
                              component):

        # prepare
        img = pro.read_img(in_fpath)
        px_nb = type_px_nbs[img_type]
        # print(px_nb)

        # select channel
        img_array = img.numpy()
        comp_array = img_array[:, :, :, component]

        # discretize
        comp_array = pro.volume_discretize(comp_array, px_nb)

        # create output image
        img_array[:, :, :, component] = comp_array
        img = img.new_image_like(img_array)

        # write
        ants.image_write(img, out_fpath)


class SetNMF(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = True
    MP = True
    SAVE_INFO = False

    def process_set(self, tpl_mask_path, channel=0, comp_nb=2, cluster_nb=2):

        # prepare output folder
        out_dir = os.path.join(self.out_dpath, str(comp_nb) + ' components/')
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        types = (list(set(self.set_info['type'])))
        types.sort()
        type_nb = len(types)

        # load template mask
        mask_fpath = pro.select_img(tpl_mask_path, '*mask*')
        mask_img = pro.read_img(mask_fpath)
        mask_vals = mask_img.numpy()
        mask = mask_vals.astype('bool')
        px_nb = (mask > 0).sum()

        nmf_model = NMF(
            comp_nb,
            alpha_W=0.1,
            alpha_H=0.5,
            beta_loss='frobenius',
            solver='cd',
            tol=1e-4,
            init='nndsvd',
            max_iter=int(1e5),
        )  # , regularization='components')

        for t_i, ty in tqdm(
                enumerate(types),
                total=type_nb,
                desc='NMF decomposition',
                file=sys.stdout):

            type_info = self.set_info[self.set_info['type'] == ty]
            type_img_nb = len(type_info)
            type_names = type_info['name'].tolist()

            # visible variables mat
            vv_mat = np.zeros((type_img_nb, px_nb))

            for i in range(type_img_nb):

                img_path = os.path.join(
                    self.in_dpath,
                    type_info.iloc[i]['path']
                )

                img_vals = pro.read_img(img_path).numpy()[:, :, :, channel]
                vv_mat[i, :] = img_vals[mask]

            w_mat = nmf_model.fit_transform(vv_mat)
            hv_mat = nmf_model.components_
            err = nmf_model.reconstruction_err_

            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=cluster_nb, random_state=0).fit(w_mat)
            cluster_labels = kmeans.labels_

            if comp_nb <= 5:
                for i_c in range(comp_nb):

                    comp_vals = np.zeros_like(mask_vals)
                    comp_vals[mask] = hv_mat[i_c, :]
                    comp_img = mask_img.new_image_like(comp_vals)

                    _ = vis.plot_maxproj(
                        comp_img,
                        title='Type ' + ty + ': component ' + str(i_c),
                        norm='minmax',
                    )

                    plt.savefig(out_dir + ty + '_comp' + str(i_c) + '.png')

            _ = plt.figure(dpi=150)
            norm_w_mat = w_mat
            plt.imshow(norm_w_mat)
            plt.colorbar(ax=plt.gca())
            plt.title('Type ' + ty + ' weights')

            plt.savefig(out_dir + ty + '_weights_heatmap.png')

            if comp_nb == 2:
                _ = plt.figure(dpi=150)

                plt.scatter(w_mat[:, 0], w_mat[:, 1],
                            c=cluster_labels, marker='x')

                for i, n in enumerate(type_names):
                    plt.annotate(
                        n, (w_mat[i, 0], w_mat[i, 1]), fontsize='xx-small')
                plt.title('Type ' + ty)
                plt.xlabel('Component 0')
                plt.ylabel('Component 1')

                plt.savefig(out_dir + ty + '_weights_scatter.png')

            elif comp_nb == 3:
                fig = plt.figure(dpi=150, figsize=(10, 10))
                # axs = plt.subplots(nrows=2, ncols=2)

                comp_pairs = [[0, 1], [0, 2], [1, 2]]

                for i_p, pair in enumerate(comp_pairs):

                    # plt.sca(axs[0])

                    fig.add_subplot(2, 2, i_p + 1)
                    plt.scatter(w_mat[:, pair[0]], w_mat[:,
                                pair[1]], c=cluster_labels, marker='x')

                    for i, n in enumerate(type_names):
                        plt.annotate(
                            n, (w_mat[i, pair[0]], w_mat[i, pair[1]]), fontsize='xx-small')

                    plt.xlabel('Component ' + str(pair[0]))
                    plt.ylabel('Component ' + str(pair[1]))

                ax = fig.add_subplot(2, 2, 4, projection='3d')
                # ax = fig.add_subplot(projection='3d')
                ax.scatter(w_mat[:, 0], w_mat[:, 1],
                           w_mat[:, 2], c=cluster_labels)
                ax.set_xlabel('Component 0')
                ax.set_ylabel('Component 1')
                ax.set_zlabel('Component 2')

            plt.suptitle('Type ' + ty + ' (err = ' +
                         '{:.2e}'.format(err) + ')')
            plt.savefig(out_dir + ty + '_weights_scatter.png')

            from sklearn.manifold import TSNE

            if comp_nb > 2:

                # use TSNE to visualize
                X_embedded = TSNE(n_components=2, learning_rate=200,
                                  perplexity=10, init='random').fit_transform(w_mat)

                _ = plt.figure(dpi=150)
                plt.scatter(X_embedded[:, 0], X_embedded[:,
                            1], c=cluster_labels, marker='x')

                for i, n in enumerate(type_names):
                    plt.annotate(
                        n, (X_embedded[i, 0], X_embedded[i, 1]), fontsize='xx-small')

                plt.title('Type ' + ty + ': TSNE')
                plt.xlabel('TSNE component 1')
                plt.ylabel('TSNE component 2')

                plt.savefig(out_dir + ty + '_tsne_weights_scatter.png')

            # show cluster average images
            for i_c in range(cluster_nb):

                cluster_paths = (
                    self.in_dpath + type_info.iloc[cluster_labels == i_c]['path']).tolist()
                cluster_avg = pro.avg_imgs(cluster_paths, component=1)

                vis.plot_maxproj(
                    cluster_avg,
                    title='Type ' + ty + ': cluster ' + str(i_c) + ' average',
                    norm='minmax'
                )

                plt.savefig(out_dir + ty + '_cluster_' +
                            str(i_c) + '_average.png')



class SetMerge(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = False
    MP = True

    def process_set(
            self,
    ):

        args_iter = zip(
            self.out_fpaths,
            *self.in_fpaths,
        )

        self.process_tasks(
            args_iter,
            self.merge_save,
            task_nb=self.in_img_nb[0],
            description='Merge images',
            kwargs_dict={}
        )

    @staticmethod
    def merge_save(out_fpath, *in_fpaths):

        merged_img = pro.merge_channels(in_fpaths)
        ants.image_write(merged_img, out_fpath)



class SetSum(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = False
    MP = True

    def process_set(
            self,
    ):

        args_iter = zip(
            self.out_fpaths,
            *self.in_fpaths,
        )

        self.process_tasks(
            args_iter,
            self.sum_save,
            task_nb=self.in_img_nb[0],
            description='Merge images',
            kwargs_dict={}
        )

    @staticmethod
    def sum_save(out_fpath, *in_fpaths):

        sum_array = None

        for fpath in in_fpaths:
            img = pro.read_img(fpath)
            array = img.numpy()

            if sum_array is None:
                sum_array = array.copy()
            else:
                sum_array += array

        img = ants.from_numpy(sum_array, spacing=img.spacing)
        ants.image_write(img, out_fpath)



class SetComputeMask(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = True
    MP = True

    def process_set(self, smoothing_sigma=2.0, method='multiotsu', **kwargs):

        tpl_fpath = self.in_fpaths[0]
        ref_img = pro.read_img(tpl_fpath)

        self.out_info.loc[0, 'name'] = 'mask_' + self.out_info.loc[0, 'name']
        self.out_fpaths[0] = self.out_dpath + \
            self.out_info.loc[0, 'name'] + self.EXT

        # compute mask and save it as image
        mask, mask_img, mask_ratio = self.compute_mask_from_allmb(
            ref_img,
            smoothing_sigma,
            method=method,
            **kwargs,
        )
        self.out_info['mask_ratio'] = [mask_ratio]

        ants.image_write(mask_img, self.out_fpaths[0])

    @staticmethod
    def compute_mask_from_allmb(avg_img, smoothing_sigma, method, **kwargs):

        # # designed to work best with isotropic spacing
        # spacing = np.array(avg_img.spacing).mean()

        stack = avg_img.numpy()

        # # test
        # mask0 = pro.remove_small_objects_stack(
        #     stack,
        #     avg_img.spacing,
        #     smoothing=2,
        #     threshold='otsu',
        #     threshold_mask=(stack>0.0),
        #     )

        # # compute mask in two steps
        # mask0 = pro.remove_small_objects_stack(
        #     stack,
        #     avg_img.spacing,
        #     smoothing=4,
        #     threshold='li',
        #     threshold_mask=(stack>0.0),
        #     )

        print(np.mean(stack > 0.0))

        # compute mask in two steps
        mask0 = pro.remove_small_objects_stack(
            stack,
            avg_img.spacing,
            smoothing=smoothing_sigma,
            threshold=method,
            **kwargs,
            # threshold_mask=mask0,
            # threshold_mask=(stack>0.0),
        )

        # erosion = int(np.round(0.5 / spacing))
        # from scipy import ndimage as ndi
        # mask0 = ndi.binary_erosion(mask0, iterations=erosion)

        # compute mask ratio
        mask_ratio = stack[mask0].sum() / stack.sum()
        print('Mask ratio:', mask_ratio)

        # convert mask in image
        mask_img = avg_img.new_image_like(mask0.astype('float') * 255.0)

        vis.plot_maxproj(mask_img)

        return mask0, mask_img, mask_ratio



class SetCorrectHist(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = True
    MP = True
    SAVE_INFO = False
    EXT = '.png'

    def process_set(self, channel, bin_nb=200):

        self.neuron_nb_dict = par.get_neuron_nbs()

        # read data and range

        args_iter = zip(
            self.in_fpaths,
            self.set_info['type'].tolist()
        )

        kwargs_dict = {
            'channel': channel,
        }

        data = self.process_tasks(
            args_iter,
            self.read_img_data,
            kwargs_dict=kwargs_dict,
            task_nb=self.in_img_nb,
            description='Read',
        )

        arrays = [d[0] for d in data]
        mins = [d[1] for d in data]
        maxs = [d[2] for d in data]
        self.val_range = [min(mins), max(maxs)]

        # compute histograms

        args_iter = zip(
            arrays,
        )

        kwargs_dict = {
            'bin_nb': bin_nb,
        }

        hists = self.process_tasks(
            args_iter,
            self.compute_hist,
            kwargs_dict=kwargs_dict,
            task_nb=self.in_img_nb,
            description='Compute histograms',
        )

        edges = np.histogram_bin_edges(
            arrays[0],
            bins=bin_nb,
            range=self.val_range
        )

        # plot histograms

        args_iter = zip(
            hists,
            self.out_fpaths,
        )

        kwargs_dict = {
            'edges': edges,
        }

        self.process_tasks(
            args_iter,
            self.plot_save_hist,
            kwargs_dict=kwargs_dict,
            task_nb=self.in_img_nb,
            description='Plot histograms',
        )

        return

    def read_img_data(self, fpath, ty, channel):

        img = pro.read_img(fpath)
        array = img.numpy()[:, :, :, channel]
        array = array[array > 0]
        array *= self.neuron_nb_dict[ty]

        return array, array.min(), array.max()

    def compute_hist(self, array, bin_nb):

        hist, _ = np.histogram(
            array,
            bins=bin_nb,
            range=self.val_range
        )

        return hist

    def plot_save_hist(self, hist, out_fpath, edges):

        fig, _ = plt.subplots(1, 1, dpi=300)

        plt.stairs(hist, edges)

        plt.savefig(out_fpath)

        plt.close()

        return


class SetSFSCRes(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = True
    MP = True
    SAVE_INFO = True
    # EXT = '.png'

    def process_set(self):

        chan_nb = pro.get_channel_nb(self.in_fpaths[0])

        args_iter = zip(
            self.in_fpaths,
        )

        resolutions = self.process_tasks(
            args_iter,
            self.read_compute,
            task_nb=self.in_img_nb,
            description='Compute resolution',
        )

        resolutions = np.array(resolutions)

        for i_c in range(chan_nb):
            for i_d, direction in enumerate(['xy', 'z']):
                chan_name = 'Channel ' + str(i_c) + ' ' + direction + ' resolution'
                chan_res = resolutions[:, i_c, i_d].squeeze()
                self.out_info[chan_name] = chan_res
        return

    def read_compute(self, fpath):

        img = pro.read_img(fpath)
        chan_imgs = pro.split_channels(img)

        resolutions = []
        for chan_img in chan_imgs:
            xy_res, z_res = pro.compute_SFSC_resolution(chan_img)
            chan_res = (xy_res, z_res)
            resolutions.append(chan_res)

        return resolutions


class SetComputeJacobian(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = True
    MP = True
    SAVE_INFO = True

    def process_set(self, ref_fpath):

        warp_fpaths = (self.in_dpath + self.set_info['transform_prefix'] + '*Warp*').tolist()
        warp_fpaths = [glob.glob(p)[0] for p in warp_fpaths]

        args_iter = zip(
            warp_fpaths,
            self.out_fpaths,
        )

        kwargs_dict = {
            'ref_fpath': ref_fpath,
        }

        self.process_tasks(
            args_iter,
            self.compute_save,
            kwargs_dict=kwargs_dict,
            task_nb=self.in_img_nb,
            description='Compute jacobian',
            # process_nb=1,
        )

        return

    def compute_save(self, warp_fpath, out_fpath, ref_fpath):

        ref_img = pro.read_img(ref_fpath)

        # provide deformation as path instead of ANTsImage to avoid the creation of a temporary file in /tmp
        jac_img = create_jacobian_determinant_image(ref_img, warp_fpath)
        pro.write_img(jac_img, out_fpath)

        return



class SetComputeVolumeChange(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = True
    MP = True
    SAVE_INFO = True
    # EXT = '.png'

    def process_set(self):

        args_iter = zip(
            self.in_fpaths,
            self.out_fpaths,
        )

        vol_changes = self.process_tasks(
            args_iter,
            self.compute_save,
            kwargs_dict={},
            task_nb=self.in_img_nb,
            description='Compute jacobian',
            # process_nb=1,
        )

        self.out_info['volume_change'] = vol_changes

        return

    def compute_save(self, in_fpath, out_fpath):

        jac_img = pro.read_img(in_fpath)
        vol_change = (jac_img - 1).abs()
        vol_change_img = jac_img.new_image_like(vol_change)
        pro.write_img(vol_change_img, out_fpath)

        return vol_change.mean()



class SetRefCrop(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = False
    MP = True
    SAVE_INFO = True

    def process_set(
            self,
            margins,
            in_pixel,
            extend,
            channel=None,
            ref_channel=None,
            ):

        args_iter = zip(
            self.in_fpaths[1],
        )

        kwargs_dict = {
            'ref_channel': ref_channel,
        }

        coms = self.process_tasks(
            args_iter,
            self.compute_com,
            kwargs_dict=kwargs_dict,
            task_nb=self.in_img_nb[1],
            description='Compute c.o.m.',
        )

        args_iter = zip(
            self.in_fpaths[0],
            self.out_fpaths,
            repeat(pro.smartcrop_img),
            repeat(margins),
            repeat(in_pixel),
            coms,
        )

        kwargs_dict = {
            'channel': channel,
            'extend': extend,
        }

        self.process_tasks(
            args_iter,
            pro.read_process_save,
            kwargs_dict=kwargs_dict,
            task_nb=self.in_img_nb[0],
            description='Crop images',
            # process_nb=1,
        )

        return

    def compute_com(self, ref_fpath, ref_channel):

        img = pro.read_img(ref_fpath)

        if img.has_components and ref_channel is not None:
            img = ants.split_channels(img)[ref_channel]

        spacing = np.array(img.spacing)[:3]

        com = np.round(ants.get_center_of_mass(img) / spacing).astype(int)

        return com



class SetMatchResolution(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = False
    MP = True
    SAVE_INFO = True

    def process_set(
            self,
            **kwargs,
            ):

        target_fpaths = self.in_fpaths[0]
        source_fpath = self.in_fpaths[1][0]

        args_iter = zip(
            target_fpaths,
            self.out_fpaths,
        )

        kwargs_dict = {
            'source_fpath': source_fpath,
        }
        kwargs_dict.update(**kwargs)

        self.process_tasks(
            args_iter,
            self.match_resolution,
            kwargs_dict=kwargs_dict,
            task_nb=self.in_img_nb[0],
            description='Match resolution',
        )

        return

    def match_resolution(self, target_fpath, out_fpath, source_fpath, **kwargs):

        source_img = pro.read_img(source_fpath)
        target_spacing = pro.get_spacing(target_fpath)

        matched_img = pro.ResampleImg(source_img).run(spacing=target_spacing, **kwargs)

        pro.write_img(matched_img, out_fpath)

        return



class SetNormPercentiles(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = True
    MP = True
    SAVE_INFO = True

    def process_set(
            self,
            percents=[25,75],
            channel=None,
            exclude_zeros=False,
            target_vals=None,
            ):

        # get percentile values

        if target_vals is None:
            args_iter = zip(
                self.in_fpaths,
            )

            kwargs_dict = {
                'percents': percents,
                'channel': channel,
                'exclude_zeros': exclude_zeros,
            }

            pc_vals_l = self.process_tasks(
                args_iter,
                self.get_percentiles,
                kwargs_dict=kwargs_dict,
                task_nb=self.in_img_nb,
                description='Get percentiles',
            )

            target_vals = np.array(pc_vals_l).mean(axis=0)

        # normalize images
        SetProcess(
            self.in_dpath,
            self.out_dpath,
            pro.NormPercentImg,
            channel=channel,
            percents=percents,
            target_pc_vals=target_vals,
            exclude_zeros=exclude_zeros,
            )


    def get_percentiles(self, in_fpath, percents, channel, exclude_zeros):

        img = pro.read_img(in_fpath)

        if channel is not None:
            img = pro.split_channels(img)[channel]

        array = img.numpy()

        if exclude_zeros:
            array = array[array > 0]
        else:
            array = array.ravel()

        pc_vals = np.percentile(array, percents)

        return pc_vals



class SetSplitHems(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = True
    MP = True
    SAVE_INFO = True
    SUFFIXES = ['_L', '_R']
    HEM_NAMES = ['left', 'right']

    def process_set(
            self,
            split_coords=None,
            ):

        if split_coords is None:
            split_coords = repeat(None)

        args_iter = zip(range(self.in_img_nb), split_coords)

        out_info = self.process_tasks(
            args_iter,
            self._split_hems,
            task_nb=self.in_img_nb,
            description='Split hemispheres',
        )

        self.out_info = pd.concat(out_info, ignore_index=True)

        return


    def _split_hems(self, img_ix, split_coord):

        in_fpath = self.in_fpaths[img_ix]
        in_info = self.set_info.iloc[[img_ix]]

        # prepare files info
        out_info = []
        for i_h in range(2):
            hem_info = in_info.copy()
            hem_info['name'] = hem_info['name'] + self.SUFFIXES[i_h]
            hem_info['hemisphere'] = self.HEM_NAMES[i_h]
            hem_info['path'] = hem_info['name'] + self.EXT
            out_info.append(hem_info)
        out_info = pd.concat(out_info, ignore_index=True)

        # load
        img = pro.read_img(in_fpath)
        array = img.numpy()

        # compute split index
        if split_coord is None:
            split_coord = array.shape[0] * img.spacing[0] / 2
        index_coord = np.round(split_coord / img.spacing[0]).astype('int')

        # split
        arrays = [
            array[:index_coord],
            array[index_coord:]
            ]

        # save
        for i_h in range(2):
            out_fpath = self.out_dpath + out_info.loc[i_h, 'path']
            out_img = ants.from_numpy(
                arrays[i_h],
                spacing=img.spacing,
                has_components=img.has_components,
            )
            pro.write_img(out_img, out_fpath)

        return out_info



class SetJoinHems(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = True
    MP = True
    SAVE_INFO = True
    HEM_NAMES = ['left', 'right']
    SUFFIX_LEN = 2

    def process_set(
            self,
            ):

        subject_list = list(set(self.set_info['subject'].tolist()))
        subject_nb = len(subject_list)

        args_iter = zip(subject_list)

        out_info = self.process_tasks(
            args_iter,
            self._join_hems,
            task_nb=subject_nb,
            description='Join hemispheres',
        )

        self.out_info = pd.concat(out_info, ignore_index=True)

        return


    def _join_hems(self, subject):

        subject_info = self.out_info[self.out_info['subject'] == subject]
        hem_infos = [subject_info[subject_info['hemisphere'] == h] for h in self.HEM_NAMES]

        in_fpaths = [self.in_dpath + hem_info['path'].iloc[0] for hem_info in hem_infos]

        # prepare output info
        out_info = hem_infos[0]
        out_info['name'] = out_info['name'].iloc[0][:-self.SUFFIX_LEN]
        out_info['path'] = out_info['name'].iloc[0] + self.EXT
        out_info['hemisphere'] = 'both'

        out_fpath = self.out_dpath + out_info['path'].iloc[0]

        # load and concatenate
        imgs = [pro.read_img(p) for p in in_fpaths]
        arrays = [img.numpy() for img in imgs]
        spacing = imgs[0].spacing
        has_components = imgs[0].has_components
        del imgs
        array = np.concatenate(arrays, axis=0)

        # create output image and save it
        del arrays
        img = ants.from_numpy(array, spacing=spacing, has_components=has_components)
        pro.write_img(img, out_fpath)

        return out_info












