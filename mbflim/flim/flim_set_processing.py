#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from itertools import repeat
import os
import pathlib
import shutil

import ants
import json
import pandas as pd
import numpy as np

import mbflim.shared as sha
import mbflim.utils as ut
import mbflim.image.image_processing as pro
import mbflim.flim.flim_processing as fl
import mbflim.files_io as fio


class SetFitDecay(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = True
    MP = True

    def process_set(
            self,
            calibration_dpath=None,
            irf_dpath=None,
            prev_fit_dpath=None,
            exp_nb=2,
            inits=10,
            solver=None,
            method='LS',
            model_class='DFCMResponseModel',
            exp_param_names=['amps', 'taus'],
            other_param_names=['irf_shift', 'bg_noise'],
            prev_param_names=[],
            fixed_parameters={},
            count_threshold=0,
            bound_indices=(0, None),
            # fixed_shift=None,
            avg_all=False,
            start_index=0,
            irf_merge='name',
    ):

        # process calibration info
        if calibration_dpath is not None:

            # read set info
            cal_info = pd.read_csv(calibration_dpath + 'files.csv', index_col=0)

            # merge to get corresponding calibration for each image
            merge_info = pd.merge(
                self.set_info,
                cal_info,
                on='experiment',
                suffixes=(None, '_cal'),
            )

            # check
            if len(merge_info) != len(self.set_info):
                raise ValueError('Missing calibration.')

            # get calibration filepaths
            cal_fpaths = (calibration_dpath + merge_info['path_cal']).tolist()

        else:
            cal_fpaths = repeat(None)

        # process IRF info
        if irf_dpath is not None:

            # read set info
            irf_info = pd.read_csv(irf_dpath + 'files.csv', index_col=0)

            merge_info = pd.merge(self.set_info, irf_info, on=irf_merge, suffixes=(None, '_irf'))

            # check
            if len(merge_info) < len(self.set_info):
                raise ValueError('Missing IRF.')

            # get IRF filepaths
            irf_fpaths = (irf_dpath + merge_info['path_irf']).tolist()

        else:
            irf_fpaths = repeat(None)

        # process previous fit info
        if prev_fit_dpath is not None:

            # read info
            prev_fit_info = pd.read_csv(prev_fit_dpath + 'files.csv', index_col=0)

            # merge to match images
            prev_fit_info = pd.merge(self.set_info['name'], prev_fit_info, on='name')

            # set fixed parameter values for each image
            fixed_parameters_L = []
            for i in range(self.in_img_nb):
                img_fixed_params = fixed_parameters.copy()

                for param_name in prev_param_names:
                    img_fixed_params.update({
                        param_name: prev_fit_info.loc[i, param_name],
                        })

                fixed_parameters_L.append(img_fixed_params)
        else:
            fixed_parameters_L = repeat(fixed_parameters)

        # prepare outout directory and filepaths for fitted responses
        out_resp_dpath = os.path.join(
            self.out_dpath,
            'fit_resp/'
            )
        os.makedirs(out_resp_dpath, exist_ok=True)
        out_resp_fpaths = [out_resp_dpath + p for p in self.out_loc_fpaths]

        kwargs_dict = {
            'exp_nb': exp_nb,
            'inits': inits,
            'solver': solver,
            'method': method,
            'model_class': model_class,
            'exp_param_names': exp_param_names,
            'other_param_names': other_param_names,
            'count_threshold': count_threshold,
            'avg_all': avg_all,
            'bound_indices': bound_indices,
            }

        args_list = list(zip(
                self.in_fpaths,
                cal_fpaths,
                irf_fpaths,
                self.out_fpaths,
                out_resp_fpaths,
                fixed_parameters_L,
                ))
        args_list = args_list[start_index:] + args_list[:start_index]

        if not avg_all:
            self.process_nb = 1

        results = self.process_tasks(
            args_list,
            self.read_fit_save,
            task_nb=self.in_img_nb,
            description='Fit images',
            kwargs_dict=kwargs_dict,
            )

        results = results[-start_index:] + results[:-start_index]

        # read variable names
        for res in results:
            if res[0] is not None:
                fit_var_names = res[0]
                break

        if fit_var_names is None:
            return

        fit_var_nb = len(fit_var_names)

        # unpack median values
        med_vals = np.zeros((self.in_img_nb, fit_var_nb))
        med_lgcs = np.zeros((self.in_img_nb, 1), dtype='bool')
        for i_r, res in enumerate(results):
            med_lgcs[i_r] = res[1] is not None
            if med_lgcs[i_r]:
                med_vals[i_r, :] = res[1]

        med_df = pd.DataFrame(
            data=med_vals,
            columns=fit_var_names,
            index=self.out_info.index
            )
        new_info = self.out_info.join(med_df)

        # reload files info
        info_fpath = self.out_dpath + 'files.csv'
        lock_fpath = self.out_dpath + 'files_csv.lock'
        lock = ut.acquireLock(lock_fpath)
        if os.path.exists(info_fpath):
            self.out_info = pd.read_csv(info_fpath, index_col=0)
            self.out_info.loc[med_lgcs] = new_info.loc[med_lgcs]
        else:
            self.out_info = self.out_info.join(med_df)

        # save output info for fitted responses
        self.out_info.to_csv(out_resp_dpath + 'files.csv')

        ut.releaseLock(lock)

        # save model info
        out_json_fpath = self.out_dpath + 'fit_info.json'
        fit_info = {
            'fit_var_names': fit_var_names
            }
        with open(out_json_fpath, 'w') as f:
            json.dump(fit_info, f, indent=4, cls=sha.NumpyEncoder)

    def read_fit_save(
            self,
            in_fpath,
            cal_fpath,
            irf_fpath,
            out_fpath,
            out_resp_fpath,
            fixed_parameters,
            bound_indices,
            # irf_shift=None,
            exp_nb=2,
            inits=10,
            solver=None,
            method='LS',
            model_class=None,
            exp_param_names=[],
            other_param_names=[],
            count_threshold=0,
            avg_all=False,
            ):

        if os.path.isfile(out_fpath):
            out_dpath = str(pathlib.Path(out_fpath).parent) + '/'
            # name = pathlib.Path(out_fpath).stem

            fit_info_fpath = out_dpath + 'fit_info.json'

            if os.path.isfile(fit_info_fpath):
                with open(fit_info_fpath, 'r') as json_file:
                    fit_info = json.load(json_file)
                fit_var_names = fit_info['fit_var_names']
            else:
                fit_var_names = None

            if os.stat(out_fpath).st_size > 0 and not fio.has_handle(str(out_fpath)):
                array = pro.read_img(out_fpath).numpy()
                valid_bools = np.any(array > 0, axis=-1)
                med_fit_vals = np.median(array[valid_bools, :], axis=0)
            else:
                med_fit_vals = None

            return fit_var_names, med_fit_vals


        # create empty output file if it does not exist
        # keeps active handle to avoid modifications/reads from other processes
        f_h = open(out_fpath, 'a');

        img = pro.read_img(in_fpath)
        name = pathlib.Path(in_fpath).stem
        print('Fitted image:', name)

        valid_bools = img.sum(axis=3) > count_threshold

        if cal_fpath is not None:
            with open(cal_fpath, 'r') as json_file:
                cal_data = json.load(json_file)

            cal = np.array(cal_data['decay_sum'])
            cal /= cal.sum()

            ref_tau = cal_data['ref_tau']

            # update model parameters
            fixed_parameters.update({
                'ref': cal,
                'ref_tau': ref_tau,
            })

        if irf_fpath is not None:
            with open(irf_fpath, 'r') as json_file:
                irf_data = json.load(json_file)

            irf = np.array(irf_data['irf'])

            # normalize IRF
            irf /= irf.sum()

            # update model parameters
            fixed_parameters.update({
                'irf': irf,
            })

        bin_nb = img.shape[3]
        step = img.spacing[3]
        fit_times = np.arange(bin_nb) * step

        if bound_indices[1] is None:
            bound_indices[1] = bin_nb - 1

        # select valid pixels and time bins
        samples = img.numpy()[valid_bools, :]  # ixs[0]:ixs[1]]
        irf = irf[:bin_nb]

        # compute average decay
        avg_decay = samples.mean(axis=0)

        if avg_all:
            img_shape = (3, 3, 3)
            samples = samples.sum(axis=0).reshape(1, -1)
            valid_bools = np.ones((3, 3, 3), dtype=bool)
        else:
            img_shape = img.shape[:3]

        # update model parameters
        fixed_parameters.update({
            'exp_nb': exp_nb,
            'times': fit_times,
            'avg_decay': avg_decay,
        })

        # create model
        class_ = getattr(fl, model_class)
        model = class_(exp_param_names, other_param_names, fixed_parameters)

        if avg_all:
            parallel = False
        else:
            parallel = True

        # fit sampled
        fit_params, fit_costs, fit_resps = fl.fit_samples(
            samples,
            model,
            init_params=inits,
            solver=solver,
            method=method,
            parallel=parallel,
            waitbar=True,
            valid_ixs=bound_indices,
        )

        # param_names = model.param_names
        param_var_nb = model.param_var_nb
        split_ixs = model.split_ixs

        add_var_nb = 0

        # add space for apparent lifetime
        if (
            'taus' in model.exp_param_names
            or 'tau_2' in model.other_param_names
            ):
            fixed_taus = False
            add_var_nb += 1
        else:
            fixed_taus = True

        # add space for cost, count
        add_var_nb += 2

        # add space for relative amplitudes
        if exp_nb > 1:
            add_var_nb += exp_nb

        out_fit_array = np.zeros(
            img_shape + (param_var_nb + add_var_nb,)
        )

        med_fit_vals = np.zeros(param_var_nb + add_var_nb)
        added_var_names = []

        # store all fitted parameters
        out_fit_array[valid_bools, :param_var_nb] = fit_params
        med_fit_vals[:param_var_nb] = np.median(fit_params, axis=0)

        # compute and store relative amplitudes
        param_arrays = model.create_params(
            *np.split(fit_params, split_ixs, axis=1)
        )

        # index of next parameter to write
        p_ix = param_var_nb

        sample_nb = fit_params.shape[0]

        if exp_nb > 1:

            # store relative amplitudes
            if 'alpha' in model.other_param_names:
                alpha_vals = param_arrays.alpha.reshape(-1, 1)
                rel_amps = np.concatenate(
                    (alpha_vals, 1 - alpha_vals),
                    axis=-1,
                    )
            else:
                rel_amps = (
                    param_arrays.amps /
                    param_arrays.amps.sum(axis=1, keepdims=True)
                )
            out_fit_array[valid_bools, p_ix:(p_ix + exp_nb)] = (
                rel_amps
            )
            med_fit_vals[p_ix:(p_ix + exp_nb)] = np.median(
                rel_amps,
                axis=0,
            )
            for i_e in range(exp_nb):
                added_var_names += ['rel_amp_' + str(i_e + 1)]
            p_ix += exp_nb

            # store apparent lifetime
            if not fixed_taus:
                if 'taus' in model.exp_param_names:
                    tau_vals = param_arrays.taus
                elif 'tau_1' in model.other_param_names and 'tau_2' in model.other_param_names:
                    tau_vals = np.concatenate((
                        param_arrays.tau_1,
                        param_arrays.tau_2
                        ), axis=1)
                elif 'tau_2' in model.other_param_names:
                    tau_vals = np.concatenate((
                        fixed_parameters['tau_1'] * np.ones((sample_nb, 1)),
                        param_arrays.tau_2
                        ), axis=1)

                mean_tau = np.multiply(
                    rel_amps,
                    tau_vals,
                ).sum(axis=1)
                out_fit_array[valid_bools, p_ix] = mean_tau
                med_fit_vals[p_ix] = np.median(mean_tau, axis=0)
                added_var_names += ['mean_tau']
                p_ix += 1

        # store cost
        fit_costs = np.squeeze(fit_costs)
        out_fit_array[valid_bools, p_ix] = fit_costs
        med_fit_vals[p_ix] = np.median(fit_costs)
        added_var_names += ['cost']
        p_ix += 1

        # store photon count
        counts = np.sum(samples, axis=1)
        out_fit_array[valid_bools, p_ix] = counts
        med_fit_vals[p_ix] = np.median(counts)
        added_var_names += ['count']
        p_ix += 1

        # create an image storing parameters and save it
        out_fit_img = ants.from_numpy(
            out_fit_array,
            spacing=img.spacing[:3],
            has_components=True,
        )
        ants.image_write(out_fit_img, out_fpath)
        del out_fit_array
        del out_fit_img

        # create array containing responses
        out_resp_array = np.zeros(img_shape + (fit_resps.shape[1],))
        out_resp_array[valid_bools, :] = fit_resps

        # create an image storing responses and save it
        out_resp_img = ants.from_numpy(
            out_resp_array,
            spacing=img.spacing,
            has_components=False,
        )
        ants.image_write(out_resp_img, out_resp_fpath)

        # names of the variables in the fitting results
        fit_var_names = model.param_var_names + added_var_names

        # release file handle
        f_h.close()

        return fit_var_names, med_fit_vals


class SetFilterParameters(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = True
    SAVE_INFO = True
    MP = True

    def process_set(
            self,
            param_lims,
            save_masks=False,
            ):

        # load model_info JSON file
        json_fpath = self.in_dpath + 'fit_info.json'
        with open(json_fpath, 'r') as json_file:
            model_info = json.load(json_file)
        self.param_names = model_info['fit_var_names']

        self.param_lims = param_lims
        self.filter_params = list(param_lims.keys())

        # optionally prepare mask filepaths
        if save_masks:
            mask_dpath = self.out_dpath + 'masks/'
            os.makedirs(mask_dpath, exist_ok=True)
            mask_fpaths = [mask_dpath + p for p in self.out_loc_fpaths]
        else:
            mask_fpaths = repeat(None)

        args_iter = zip(
            self.in_fpaths,
            self.out_fpaths,
            mask_fpaths,
        )

        median_vals = self.process_tasks(
            args_iter=args_iter,
            function=self.read_filter_save,
            task_nb=self.in_img_nb,
            description='Filter parameters',
        )

        median_vals = np.array(median_vals)

        for i_p, p in enumerate(self.param_names):
            self.out_info[p] = median_vals[:, i_p]

        # optionally save mask files info
        if save_masks:
            self.out_info.to_csv(mask_dpath + 'files.csv')

        # copy model_info JSON file
        shutil.copy(json_fpath, self.out_dpath)


    def read_filter_save(self, in_fpath, out_fpath, mask_fpath):

        # read image
        img = pro.read_img(in_fpath)
        array = img.numpy()

        median_vals = []
        valid_lgcs = np.ones(array.shape[:3], dtype='bool')

        for p in self.filter_params:

            lims = self.param_lims[p]
            p_ix = self.param_names.index(p)

            # filter
            lgcs = (array[:, :, :, p_ix] >= lims[0]) & (array[:, :, :, p_ix] <= lims[1])
            valid_lgcs = valid_lgcs & lgcs

        array[~valid_lgcs, :] = 0.0

        # compute median values
        for i_p, _ in enumerate(self.param_names):
            lgcs = array[:, :, :, i_p] > 0.0
            median_vals.append(np.median(array[lgcs, i_p]))

        # create and save image
        img = img.new_image_like(array)
        pro.write_img(img, out_fpath)

        # optionally create and save mask
        if mask_fpath is not None:
            mask_img = ants.from_numpy(
                valid_lgcs.astype('float'),
                spacing=img.spacing,
                has_components=False,
            )
            pro.write_img(mask_img, mask_fpath)

        return median_vals




