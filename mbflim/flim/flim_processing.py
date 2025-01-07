#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 14:46:09 2022

@author: proussel
"""

from collections import namedtuple
from abc import ABC, abstractmethod
import os
import sys
import warnings
from itertools import repeat
import multiprocessing.pool as mpp
from tqdm import tqdm
import time

import numpy as np
import scipy

import tools.shared as sha


class AstractResponseModel(ABC):

    def __init__(
            self,
            exp_param_names=['amps', 'taus'],
            other_param_names=['irf_shift', 'bg_noise'],
            constants={},
    ):

        # store parameter names
        self.exp_param_names = exp_param_names
        self.other_param_names = other_param_names

        # store constants
        self.constants = constants
        self.times = constants['times']
        self.exp_nb = constants['exp_nb']

        # process time constants
        self.bin_nb = len(self.times)
        self.timestep = np.diff(self.times[:2])

        # compute numbers of parameters
        self.exp_param_nb = len(self.exp_param_names)
        self.other_param_nb = len(self.other_param_names)
        exp_var_nbs = self.exp_nb * np.ones(self.exp_param_nb, dtype=int)
        other_var_nbs = np.ones(self.other_param_nb, dtype=int)
        self.param_nbs = np.concatenate((exp_var_nbs, other_var_nbs), axis=0)
        self.param_var_nb = np.sum(self.param_nbs)

        # process parameter names
        self.param_names = self.exp_param_names + self.other_param_names
        self.param_var_names = []
        for i_n in range(self.exp_param_nb):
            for i_e in range(self.exp_nb):
                self.param_var_names += [
                    self.exp_param_names[i_n][:-1]
                    + '_' + str(i_e + 1)
                ]
        self.param_var_names += self.other_param_names

        # compute split indices
        if self.exp_param_nb > 0:
            exp_ixs = np.arange(
                self.exp_nb,
                self.exp_nb * self.exp_param_nb,  # (self.exp_param_nb + 1),
                self.exp_nb
            )
            if self.other_param_nb > 0:
                exp_ixs = np.append(exp_ixs, self.exp_nb * self.exp_param_nb)
        else:
            exp_ixs = np.zeros((0,), dtype='int64')

        if self.other_param_nb > 0:
            other_ixs = (
                self.exp_nb * self.exp_param_nb
                + np.arange(1, self.other_param_nb)
                )
            self.split_ixs = np.concatenate((exp_ixs, other_ixs))
        else:
            self.split_ixs = exp_ixs

        if self.exp_param_nb > 0 and self.exp_param_nb > 0:
            self.split_ixs = np.concatenate((exp_ixs, other_ixs))

        self.exp_nb * self.exp_param_nb

        # pre-compute shifted IRF
        if 'irf_shift' in self.constants:
            self.shifted_irf = self.shift_signal(
                self.constants['irf'],
                self.constants['irf_shift']
                )

        # set default values (scales, intializations, etc.)
        # default values can be changed in adapt_model()
        self.define_default_values()

        # adapt default values, pre-computed values to each child class
        self.adapt_model()

        # preparation, common to all models
        self.prepare_optimization()

    def define_default_values(self):

        # define default values (can be changed in prepare_model())
        self.scales_dict = {
            'amps': 1e-1,
            'taus': 1e-10,
            'tau_1': 1e-10,
            'tau_2': 1e-10,
            'alpha': 1e-2,
            'irf_shift': np.diff(self.times[:2]) / 10.0,
            'bg_noise': 1e-3,
            'sl_noise': 1e-3,
        }
        self.abs_bools_dict = {
            'amps': True,
            'taus': True,
            'tau_1': True,
            'tau_2': True,
            'alpha': True,
            'irf_shift': False,
            'bg_noise': True,
            'sl_noise': True,
        }
        self.init_mins_dict = {
            'amps': 1e-3,
            'taus': 2e-11,
            'tau_1': 1e-10,
            'tau_2': 1e-9,
            'alpha': 0.01,
            'irf_shift': -1e-10,
            'bg_noise': 0.0,
            'sl_noise': 0.0,
        }
        self.init_maxs_dict = {
            'amps': 2.5,
            'taus': 8e-9,
            'tau_1': 1e-9,
            'tau_2': 8e-9,
            'alpha': 0.99,
            'irf_shift': 2e-10,
            'bg_noise': 5e-3,
            'sl_noise': 1.0,
        }
        self.bounds_dict = {
            'amps': (0, np.inf),
            'taus': (1e-11, 1e-8),
            'tau_1': (1e-10, 1e-9),
            'tau_2': (0.0, np.inf),
            'alpha': (0.0, 1.0),
            'irf_shift': (-self.times[-1] / 2, self.times[-1] / 2),
            'bg_noise': (0.0, np.inf),
            'sl_noise': (0, np.inf),
        }

    @abstractmethod
    def adapt_model(self):
        pass

    def prepare_optimization(self):

        # define scales etc.
        self.param_scales = self.vectorize_params(self.scales_dict)
        self.param_abs_bools = self.vectorize_params(self.abs_bools_dict)
        self.param_init_mins = self.vectorize_params(self.init_mins_dict)
        self.param_init_maxs = self.vectorize_params(self.init_maxs_dict)

        self.bounds = []
        for name, nb in zip(self.param_names, self.param_nbs):
            self.bounds += [self.bounds_dict[name]] * nb

        self.lower_bounds = []
        for name, nb in zip(self.param_names, self.param_nbs):
            self.lower_bounds += [self.bounds_dict[name][0]] * nb

        self.upper_bounds = []
        for name, nb in zip(self.param_names, self.param_nbs):
            self.upper_bounds += [self.bounds_dict[name][1]] * nb

        self.scaled_bounds = []
        for name, nb in zip(self.param_names, self.param_nbs):
            scale = self.scales_dict[name]
            bounds = self.bounds_dict[name]
            scaled_bounds = []
            for bound in bounds:
                if bound is None:
                    scaled_bound = None
                else:
                    scaled_bound = bound / scale
                scaled_bounds += [scaled_bound]
            scaled_bounds = tuple(scaled_bounds)

            self.scaled_bounds += [scaled_bounds] * nb

    def create_params(self, *args, **kwargs):
        Params = namedtuple(
            "Params",
            self.param_names
        )

        if len(args) == 0:
            return Params(**kwargs)
        else:
            return Params(*args)

    def vectorize_params(self, params):

        if isinstance(params, dict):
            params_v = np.zeros(
                self.param_var_nb,
                dtype=type(list(params.values())[0])
            )
            ix = 0

            for i_p, name in enumerate(self.param_names):
                nb = self.param_nbs[i_p]
                params_v[ix:(ix + nb)] = params[name]
                ix += nb

        else:
            params_v = np.zeros(self.param_var_nb, dtype=params[0].dtype)
            ix = 0

            for i_p, name in enumerate(self.param_names):
                nb = self.param_nbs[i_p]
                params_v[ix:(ix + nb)] = getattr(params, name)
                ix += nb

        return params_v

    def unvectorize_params(self, params_v):

        params_nt = self.create_params(*np.split(params_v, self.split_ixs))

        return params_nt

    def order_params(self, params_nt):

        ixs = np.argsort(params_nt.taus)

        params_nt = params_nt._replace(
            taus=params_nt.taus[ixs],
        )

        if 'alpha' in self.other_param_names:
            # if there is an inversion (case of 2 exponentials)
            if ixs[0] == 1:
                params_nt = params_nt._replace(
                    alpha=1-params_nt.alpha,
                )
        else:
            params_nt = params_nt._replace(
                amps=params_nt.amps[ixs],
            )

        return params_nt

    def generate_init_params(self):

        mins = self.unvectorize_params(self.param_init_mins)
        maxs = self.unvectorize_params(self.param_init_maxs)
        rng = np.random.default_rng(time.time_ns())

        init_params_v = np.empty(self.param_var_nb)
        ix = 0

        for i_p, n in enumerate(self.param_names):
            nb = self.param_nbs[i_p]

            p_min = getattr(mins, n)
            p_max = getattr(maxs, n)

            param_init_vals = rng.uniform(size=nb) * (p_max - p_min) + p_min
            init_params_v[ix:(ix + nb)] = param_init_vals

            ix += nb

        init_params_nt = self.unvectorize_params(init_params_v)

        return init_params_nt

    def response(self, params_v):

        params_nt = self.unvectorize_params(params_v)

        # shift IRF
        if 'irf_shift' in self.other_param_names:
            shifted_irf = self.shift_signal(self.irf, params_nt.irf_shift)
        elif 'irf_shift' in self.constants:
            shifted_irf= self.shifted_irf
        else:
            shifted_irf = self.irf

        if 'alpha' in self.other_param_names:
            alpha = params_nt.alpha
            amps = np.array([alpha, 1 - alpha])
        elif self.exp_nb == 1:
            amps = np.array([1.0])
        else:
            amps = params_nt.amps

        if 'taus' in self.exp_param_names:
            taus = params_nt.taus
        elif 'tau_1' in self.other_param_names and 'tau_2' in self.other_param_names:
            taus = np.array([params_nt.tau_1[0], params_nt.tau_2[0]])
        elif 'tau_1' in self.other_param_names:
            taus = params_nt.tau_1
        elif 'tau_1' in self.constants and 'tau_2' in self.other_param_names:
            taus = np.array([self.constants['tau_1'], params_nt.tau_2[0]])
        elif 'tau_1' in self.constants and 'tau_2' in self.constants:
            taus = np.array([self.constants['tau_1'], self.constants['tau_2']])
        elif 'taus' in self.constants:
            taus = np.array(self.constants['taus'])
        else:
            raise ValueError('Missing lifetime specification.')

        response = self.compute_decay(
            amps=amps,
            taus=taus,
            shifted_irf=shifted_irf
        )

        # add noises
        if 'bg_noise' in self.other_param_names:
            bg_noise = params_nt.bg_noise
        elif'bg_noise' in self.constants:
            bg_noise = self.constants['bg_noise']
        else:
            bg_noise = 0.0
        response += bg_noise

        return response

    @abstractmethod
    def compute_decay(self, amps, taus, shifted_irf):
        pass

    def shift_signal(self, signal, shift):
        if shift != 0.0:
            f = scipy.interpolate.interp1d(
                self.times,
                signal,
                fill_value='extrapolate',
                kind='cubic', #'linear',
                assume_sorted=True,
            )
            shifted_signal = f(self.times + shift)
            shifted_signal[shifted_signal < 0] = 0.0
        else:
            shifted_signal = signal
        return shifted_signal


# naive impulse response model
class ICModel(AstractResponseModel):

    def adapt_model(self):

        self.irf = self.constants['irf']

        self.scales_dict['amps'] = 1e-4
        self.scales_dict['taus'] = 1e-11
        self.scales_dict['bg_noise'] = 1e-5

        self.init_mins_dict['amps'] = 1e-4
        self.init_mins_dict['taus'] = 1e-11
        self.init_mins_dict['bg_noise'] = 0.0

        self.init_maxs_dict['amps'] = 5e-2
        self.init_maxs_dict['taus'] = 5e-9
        self.init_maxs_dict['bg_noise'] = 5e-3

    def compute_decay(self, amps, taus, shifted_irf):

        # compute raw decay
        decay = np.sum(
            amps.reshape(-1, 1) * np.exp(-self.times / taus.reshape(-1, 1)),
            axis=0
        )

        # convolve decay and IRF
        decay = np.convolve(shifted_irf, decay)[:self.bin_nb]

        return decay


# naive impulse response model with previous pulse
class ICPPModel(ICModel):

    def adapt_model(self):
        super().adapt_model()
        self.ext_times = np.arange(2 * self.bin_nb) * self.timestep

    def compute_decay(self, amps, taus, shifted_irf):

        # compute raw decay
        decay = np.sum(
            amps.reshape(-1, 1) * np.exp(
                -self.ext_times /
                taus.reshape(-1, 1)),
            axis=0
        )

        # convolve decay and IRF
        decay = np.convolve(shifted_irf, decay)
        decay = decay[:self.bin_nb] + decay[self.bin_nb:(2 * self.bin_nb)]

        decay /= decay.sum()

        return decay


class SampleFit(ABC):

    def __init__(self, model):
        self.model = model

    def fit(self, measure, init_params=1, solver=None, valid_ixs=[None, None]):

        # prepare boolean vector for valid bins
        bin_nb = len(measure)
        self.valid_bools = np.zeros(bin_nb, dtype=bool)
        self.valid_bools[valid_ixs[0]:valid_ixs[-1]] = True
        self.valid_bools = np.logical_and(self.valid_bools, measure > 0)

        # prepare valid measure elements
        self.valid_measure = measure[self.valid_bools]

        # store norm
        self.measure_norm = self.valid_measure.sum()

        # if namedtuple
        if sha.isnamedtupleinstance(init_params):
            init_params_list = [init_params]
        # if list
        elif isinstance(init_params, list):
            init_params_list = init_params
        # if int
        elif isinstance(init_params, int):
            init_params_list = [
                self.model.generate_init_params()
                for i in range(init_params)
            ]
        else:
            raise(ValueError(
                "Unsupported type "
                + str(type(init_params))
                + " for 'init_params'."
            ))

        fit_params_list, fit_costs = self.optimize(init_params_list, solver)

        best_ix = np.argmin(fit_costs)
        fit_cost = fit_costs[best_ix]

        fit_params = fit_params_list[best_ix]
        if 'taus' in self.model.exp_param_names and self.model.exp_nb > 1:
            fit_params = self.model.order_params(fit_params)
        fit_params = self.model.vectorize_params(fit_params)

        fit_resp = self.norm_response(fit_params)

        return fit_params, fit_cost, fit_resp

    def fit_samples(self, measures, **kwargs):

        # initialize
        measure_nb = len(measures)
        fit_params_array = np.empty((measure_nb, self.model.param_var_nb,))
        fit_cost_array = np.empty((measure_nb, 1))
        fit_resp_array = np.empty((measure_nb, self.model.bin_nb,))

        for i_m, measure in enumerate(measures):

            fit_params, fit_cost, fit_resp = self.fit(measure, **kwargs)

            fit_params_array[i_m, :] = fit_params
            fit_cost_array[i_m] = fit_cost
            fit_resp_array[i_m, :] = fit_resp

        return fit_params_array, fit_cost_array, fit_resp_array

    def norm_response(self, fit_params):
        resp = self.model.response(fit_params)

        if resp.sum() == 0:
            return resp
        else:
            return resp / resp[self.valid_bools].sum() * self.measure_norm

    @abstractmethod
    def loss(self):
        pass


class SampleFitLS(SampleFit):

    def optimize(self, init_params_list, solver):

        # as in Warren et al. 2013
        avg_decay = self.model.constants['avg_decay']
        weights = avg_decay.copy()
        weights[weights == 0.0] = 1.0
        valid_weights = weights[self.valid_bools]

        fit_params_list = []
        init_nb = len(init_params_list)
        fit_costs = np.zeros(init_nb)

        for i_i, init_params_nt in enumerate(init_params_list):

            res = scipy.optimize.least_squares(
                fun=self.loss,
                x0=self.model.vectorize_params(init_params_nt),
                kwargs={
                    'valid_weights': valid_weights,
                },
                x_scale=self.model.param_scales,
                verbose=0,
                # max_nfev=2,
                method=solver,
            )

            fit_params_list.append(self.model.unvectorize_params(res.x))

            # normalize cost
            fit_costs[i_i] = res.cost / self.measure_norm

        return fit_params_list, fit_costs

    def loss(self, params_v, valid_weights):

        # force defined parameters to be positive
        bools = self.model.param_abs_bools
        params_v[bools] = np.abs(params_v[bools])

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in reduce")
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in multiply")
            warnings.filterwarnings(
                "ignore", message="overflow encountered in reduce")
            warnings.filterwarnings(
                "ignore", message="overflow encountered in multiply")
            warnings.filterwarnings(
                "ignore", message="overflow encountered in exp")

            response = self.norm_response(params_v)

        # print('loss', self.valid_bools)
        # print('r', len(response))
        valid_response = response[self.valid_bools]
        # print('vr', len(valid_response))

        diff = valid_response - self.valid_measure
        weighted_diff = np.divide(diff, valid_weights)

        return weighted_diff


class SampleFitMLE(SampleFit):

    def optimize(self, init_params_list, solver):

        init_nb = len(init_params_list)
        fit_params_list = []
        fit_costs = np.zeros(init_nb)

        for i_i, init_params_nt in enumerate(init_params_list):

            # scale
            init_params_v = self.model.vectorize_params(init_params_nt)
            scaled_init_params_v = init_params_v / self.model.param_scales

            res = scipy.optimize.minimize(
                fun=self.loss,
                x0=scaled_init_params_v,
                args=(),
                bounds=self.model.scaled_bounds,
                method=solver,
            )

            # unscale
            scaled_fit_params_v = res.x
            fit_params_v = scaled_fit_params_v * self.model.param_scales

            fit_params_list.append(self.model.unvectorize_params(fit_params_v))
            fit_costs[i_i] = self.loss(scaled_fit_params_v)

        return fit_params_list, fit_costs

    def loss(self, scaled_params_v):

        # unscale
        params_v = scaled_params_v * self.model.param_scales

        with warnings.catch_warnings(record=True) as _:
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in reduce")
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in multiply")
            warnings.filterwarnings(
                "ignore", message="overflow encountered in reduce")
            warnings.filterwarnings(
                "ignore", message="overflow encountered in multiply")
            warnings.filterwarnings(
                "ignore", message="overflow encountered in exp")
            warnings.filterwarnings(
                "ignore", message="divide by zero encountered in divide")
            warnings.filterwarnings(
                "ignore", message="overflow encountered in divide")

            response = self.norm_response(params_v)
            valid_response = response[self.valid_bools]

            L = 2 * np.sum(np.multiply(
                    self.valid_measure,
                    np.log(self.valid_measure / (valid_response + np.finfo(float).eps))
                ))

        return L


def fit_samples(
        samples,
        model,
        init_params,
        solver,
        method,
        valid_ixs,
        parallel=True,
        waitbar=True,
):

    max_thread_nb = int(os.environ["MF_NUM_THREADS"])

    sample_nb = len(samples)

    block_nb = max(max_thread_nb, np.round(sample_nb / 1000).astype(int))

    args_iter = zip(
        np.array_split(samples, block_nb),
    )

    kwargs_iter = {
        # 'model': model,
        'init_params': init_params,
        'solver': solver,
        'valid_ixs': valid_ixs,
    }

    if method == 'LS':
        fit_obj = SampleFitLS(model)
    elif method == 'MLE':
        fit_obj = SampleFitMLE(model)

    if parallel and sample_nb > 1:

        with mpp.Pool(max_thread_nb) as pool:
            results = list(tqdm(
                sha.istarmap_with_kwargs(
                    pool,
                    fit_obj.fit_samples,
                    args_iter,
                    repeat(kwargs_iter),
                ),
                total=block_nb,
                desc='Fit blocks',
                file=sys.stdout,
                smoothing=0,
                disable=not waitbar,
                position=1,
                mininterval=2,
            ))

    else:
        results = []

        for input_tuple in list(tqdm(args_iter,
                                     total=block_nb,
                                     desc='Fit blocks',
                                     file=sys.stdout,
                                     smoothing=0,
                                     disable=True,
                                     position=1,
                                     )):
            results.append(fit_obj.fit(*input_tuple, **kwargs_iter))

    fit_params = np.vstack([res[0] for res in results])
    fit_costs = np.vstack([res[1] for res in results])
    fit_resps = np.vstack([res[2] for res in results])

    return fit_params, fit_costs, fit_resps
