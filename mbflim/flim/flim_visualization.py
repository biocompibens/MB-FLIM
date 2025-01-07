#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
from itertools import repeat

import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
from statannotations.Annotator import Annotator
import robustats

import tools.shared as sha
import tools.image.image_processing as pro
import tools.parameters as par
import tools.visualization as vis


class SetAnalyzeConditions(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = False
    SAVE_INFO = False
    MP = True

    def process_set(
            self,
            channel_ixs=None,
            min_count=0,
            avg_mode='median',
            fit_info_dpath=None,
            remove_outliers=False,
            asym_norm_distrib='Skew normal',
            ):

        bin_nb = 100

        self.asym_norm_distrib = asym_norm_distrib

        # extract type info
        if 'type' not in self.set_info[0]:
            self.set_info[0]['type'] = 'notype'
            self.set_info[0]['fly'] = np.arange(len(self.set_info[0]))
        type_set = sorted(list(set(self.set_info[0]['type'])))
        type_nb = len(type_set)
        img_types = np.array(self.set_info[0]['type'])
        types_img_nbs = [(img_types == ty).sum() for ty in type_set]

        if type_nb == 3:
            color_dict = par.get_soft_sp_colors()
            type_colors = [color_dict[ty] for ty in type_set]
        elif type_nb == 2:
            type_colors = ['tab:blue', 'tab:orange']
            color_dict = dict.fromkeys(type_set)
            for i_t, ty in enumerate(type_set):
                color_dict[ty] = type_colors[i_t]
        elif type_nb == 1:
            type_colors = ['k']
            color_dict = {'KC': type_colors[0]}

        # extract channel info
        fpath = pro.select_img(self.in_dpath[0], 0)
        channel_nb = pro.get_channel_nb(fpath)
        self.channels = np.arange(channel_nb)

        # select channels
        if channel_ixs is not None:
            self.channels = self.channels[channel_ixs]
        self.param_nb = len(self.channels)

        # load model_info JSON file
        if fit_info_dpath is None:
            fit_info_dpath = self.in_dpath[0]
        with open(fit_info_dpath + 'fit_info.json', 'r') as json_file:
            model_info = json.load(json_file)

        self.param_names = model_info['fit_var_names']

        alpha_ix = self.param_names.index('alpha')

        # prepare tested parameters
        untested_params = ['amp_1', 'amp_2', 'rel_amp_2']
        if 'alpha' in self.param_names:
            untested_params += ['rel_amp_1']
        test_params = [p for p in self.param_names if p not in untested_params]
        test_param_nb = len(test_params)

        # prepare thresholds for all parameters
        perc_thresh_dict = {
            'tau_1': [1, 99],
            'amp_1': [1, 99],
            'amp_2': [1, 99],
            'count': [0, 95],
            'bg_noise': [0, 99],
            }

        self.perc_threshs = np.zeros((self.param_nb, 2))
        param_thresh_strs = []

        for i_p, p_name in enumerate(self.param_names):
            if p_name in perc_thresh_dict.keys():
                self.perc_threshs[i_p, :] = perc_thresh_dict[p_name]
            else:
                self.perc_threshs[i_p, :] = [0, 100]

            param_thresh_strs.append(
                str(int(self.perc_threshs[i_p, 0]))
                + '-' + str(int(self.perc_threshs[i_p, 1])) + '%'
                )

        # prepare subject info
        if 'subject' not in self.set_info[0]:
            if 'experiment' in self.set_info[0]:
                self.set_info[0]['subject'] = self.set_info[0]['experiment']
            else:
                self.set_info[0]['subject'] = self.set_info[0]['type']

        img_subjects = self.set_info[0]['subject'].to_numpy()
        subject_set = set(img_subjects)
        subject_nb = len(subject_set)

        # prepare hemisphere info
        if 'hemisphere' in self.set_info[0]:
            hem_set, hem_ixs = sha.map_info(self.set_info[0]['hemisphere'])

        # optionally load weights
        if self.input_nb > 1:
            merged_info = self.set_info[0].merge(
                self.set_info[1],
                on='name',
                how='left'
                )
            weight_fpaths = list(self.in_dpath[1] + merged_info['path_y'])
        else:
            weight_fpaths = repeat(None)

        # load images data
        args_iter = zip(
            self.in_fpaths[0],
            weight_fpaths,
            )

        kwargs_dict = {
            'avg_mode': avg_mode,
            'min_count': min_count,
            'remove_outliers': remove_outliers,
            }

        results = self.process_tasks(
            args_iter,
            self.load_img,
            task_nb=self.in_img_nb[0],
            description='Load',
            kwargs_dict=kwargs_dict,
            )

        # unpack results
        imgs_vals = [res[0] for res in results]
        imgs_min_vals = [res[1] for res in results]
        imgs_max_vals = [res[2] for res in results]
        imgs_avg_vals = np.stack([res[3] for res in results], axis=0)
        imgs_cv_vals = np.stack([res[4] for res in results], axis=0)

        # compute overall extrema
        param_mins = np.amin(np.vstack(imgs_min_vals), axis=0)
        param_maxs = np.amax(np.vstack(imgs_max_vals), axis=0)

        # fit alpha values
        self.skew_vals = np.empty((self.in_img_nb[0],))
        self.mean_vals = np.empty((self.in_img_nb[0],))
        self.var_vals = np.empty((self.in_img_nb[0],))

        if self.asym_norm_distrib == 'Johnson-SU':
            self.js_params = np.empty((self.in_img_nb[0], 4))

        for i_i in range(self.in_img_nb[0]):
            data = imgs_vals[i_i][:, alpha_ix]

            if self.asym_norm_distrib == 'Skew normal':
                self.skew_vals[i_i], self.mean_vals[i_i], self.var_vals[i_i] = \
                    scipy.stats.skewnorm.fit(data)
            elif self.asym_norm_distrib == 'Johnson-SU':
                a, b, loc, scale = scipy.stats.johnsonsu.fit(data)
                self.mean_vals[i_i], self.var_vals[i_i], self.skew_vals[i_i] = \
                    scipy.stats.johnsonsu.stats(a, b, loc, scale, 'mvs')
                self.js_params[i_i, :] = [a, b, loc, scale]
            else:
                raise ValueError('Unknown asymmetric distribution: ' + str(self.asym_norm_distrib))

        # compute and print global stats
        means_of_avg = []
        cvs_of_avg = []
        means_of_cv = []

        for i_p, name in enumerate(self.param_names):

            # tau1_ix = self.param_names.index('tau_1')
            mean_of_avg = imgs_avg_vals[:, i_p].mean()
            cv_of_avg = scipy.stats.variation(imgs_avg_vals[:, i_p], ddof=1)
            mean_of_cv = np.abs(imgs_cv_vals[:, i_p]).mean()

            means_of_avg.append(mean_of_avg)
            cvs_of_avg.append(cv_of_avg)
            means_of_cv.append(mean_of_cv)

            print('-- ', name)
            print('    Avg of ' + avg_mode, mean_of_avg)
            print('    CV of ' + avg_mode, cv_of_avg)
            print('    Avg of CV', mean_of_cv)

        param_stats_df = pd.DataFrame({
            'parameter': self.param_names,
            'img_avg_mean': means_of_avg,
            'img_avg_cv': cvs_of_avg,
            'img_cv_mean': means_of_cv,
            })

        param_stats_df.to_csv(self.out_dpath + 'stats.csv')

        # compute image histograms
        args_iter = zip(
            imgs_vals,
            )
        kwargs_dict = {
            'hist_min_vals': param_mins,
            'hist_max_vals': param_maxs,
            'bin_nb': bin_nb,
            }
        img_hists = self.process_tasks(
            args_iter,
            self.compute_img_hists,
            task_nb=self.in_img_nb[0],
            description='Compute statistics',
            kwargs_dict=kwargs_dict,
            )
        img_hists = np.stack(img_hists, axis=0)

        # get histogram bins for each parameter
        param_bin_centers = np.zeros((self.param_nb, bin_nb))
        param_bin_widths = np.zeros((self.param_nb,))
        for i_p in range(self.param_nb):
            bin_edges = np.histogram_bin_edges(
                None,
                bins=bin_nb,
                range=(param_mins[i_p],
                       param_maxs[i_p])
                )
            param_bin_centers[i_p, :] = (bin_edges[:-1] + bin_edges[1:]) / 2
            param_bin_widths[i_p] = np.diff(bin_edges[:2])

        # average per subject
        subject_hists = np.zeros((subject_nb,) + img_hists.shape[1:])
        subjects_avgs = np.zeros((subject_nb, self.param_nb))
        subjects_gen_gaussians = np.zeros((subject_nb, 3))
        if self.asym_norm_distrib == 'Johnson-SU':
           subjects_js_params = np.zeros((subject_nb, self.js_params.shape[1]))
        subject_types = []
        for i_s, subject in enumerate(subject_set):
            subj_bools = (img_subjects == subject)
            subject_hists[i_s, :, :] = img_hists[subj_bools, :, :].mean(axis=0)
            subjects_avgs[i_s, :] = imgs_avg_vals[subj_bools, :].mean(axis=0)
            subjects_gen_gaussians[i_s, :] = (
                self.mean_vals[subj_bools].mean(),
                self.var_vals[subj_bools].mean(),
                self.skew_vals[subj_bools].mean(),
                )
            if self.asym_norm_distrib == 'Johnson-SU':
                subjects_js_params[i_s, :] = self.js_params[subj_bools, :].mean(axis=0)

            subject_img_types = img_types[subj_bools]
            subject_types.append(subject_img_types[0])
        subject_types = np.array(subject_types)

        # average per type
        type_hists = np.zeros((type_nb,) + img_hists.shape[1:])
        type_gen_gaussians = np.zeros((type_nb, 3))
        if self.asym_norm_distrib == 'Johnson-SU':
           type_js_params = np.zeros((type_nb, subjects_js_params.shape[1]))
        for i_t, ty in enumerate(type_set):
            ty_bools = (subject_types == ty)
            type_hists[i_t, :, :] = subject_hists[ty_bools, :, :].mean(axis=0)
            type_gen_gaussians[i_t, :] = subjects_gen_gaussians[ty_bools, :].mean(axis=0)
            if self.asym_norm_distrib == 'Johnson-SU':
                type_js_params[i_t, :] = subjects_js_params[ty_bools, :].mean(axis=0)

        # create string detailing the number of subject for each type
        type_subject_nb = [(subject_types == ty).sum() for ty in type_set]
        type_subject_nb_str = ''
        for i_t in range(type_nb):
            type_subject_nb_str += (
                type_set[i_t] + ': ' + str(type_subject_nb[i_t])
                )
            if i_t < type_nb - 1:
                type_subject_nb_str += ', '

        ### SKEW NORMAL PARAMETERS

        x = np.linspace(param_mins[alpha_ix], param_maxs[alpha_ix], 100)
        fig, ax = plt.subplots(1, 1, dpi=300, figsize=(2, 2))

        for i_t in range(type_nb):

            if self.asym_norm_distrib == 'Skew normal':
                dist_curve = scipy.stats.skewnorm.pdf(
                    x,
                    type_gen_gaussians[i_t, 2],
                    type_gen_gaussians[i_t, 0],
                    type_gen_gaussians[i_t, 1]
                    )
            elif self.asym_norm_distrib == 'Johnson-SU':
                dist_curve = scipy.stats.johnsonsu.pdf(
                    x,
                    *type_js_params[i_t, :],
                    )

            dist_curve /= dist_curve.sum()

            ax.plot(
                x,
                dist_curve,
                color=type_colors[i_t],
                )
            print(
                type_set[i_t] + ':\n',
                'm', type_gen_gaussians[i_t, 0], '\n',
                'v', type_gen_gaussians[i_t, 1], '\n',
                's', type_gen_gaussians[i_t, 2], '\n',
                )

        if type_nb == 2:
            _, axs = plt.subplots(1, 3, dpi=300, figsize=(6, 3))
            for i, (p_name, p_vals) in enumerate(zip(
                    ['$f_{free}$ mean', '$f_{free}$ variance', '$f_{free}$ skewness'],
                    list(subjects_gen_gaussians.T)
                    )):
                print(p_vals)
                df = pd.DataFrame({'type': subject_types, 'average': p_vals})
                vis.plot_param(df, avg_mode='mean', param_name=p_name, ax=axs[i], avg_category='subject')
                plt.tight_layout()

            plt.suptitle(self.asym_norm_distrib)
            plt.tight_layout()

        ### HEMISPHERE COMPARISON

        if 'hemisphere' in self.set_info[0] and len(hem_set) > 1:
            row_nb = np.ceil(test_param_nb/3).astype('int')
            hem_fig, hem_axs = plt.subplots(
                ncols=3,
                nrows=row_nb,
                figsize=(7, row_nb * 2.5),
                dpi=150,
                layout='constrained',
                )
            hem_axs = hem_axs.ravel()

            for i_p in range(test_param_nb):
                param_name = test_params[i_p]

                param_avg_vals = imgs_avg_vals [:, self.param_names.index(param_name)]

                ax = hem_axs[i_p]
                plt.sca(ax)

                subj_diffs = []
                hems_avg_vals = []

                for i_s, subject in enumerate(subject_set):
                    subj_bools = (img_subjects == subject)

                    if subj_bools.sum() > 1:
                        subj_avg_vals = param_avg_vals[subj_bools]

                        hems_avg_vals += subj_avg_vals.tolist()

                        subj_diff = np.abs(np.diff(subj_avg_vals)[0])

                        subj_diffs.append(subj_diff)

                n = len(hems_avg_vals)
                hems_avg_vals = np.array(hems_avg_vals)
                img_diffs = hems_avg_vals.reshape(-1, 1) - hems_avg_vals.reshape(1, -1)
                img_diffs = np.abs(img_diffs[np.triu_indices(n, k=1)])

                diff_types = ['Hem. diff.'] * len(subj_diffs) + ['Img. diff.'] * len(img_diffs)
                diffs = subj_diffs + img_diffs.tolist()

                df = pd.DataFrame({
                    'diff_type': diff_types,
                    'diff': diffs,
                    })

                if not df.empty:
                    sns.boxplot(data=df, x='diff_type', y='diff')

                plt.title(param_name)

            plt.suptitle(type_subject_nb_str)
            plt.savefig(
                self.out_dpath + 'fig0_hem_comparison.png'
                )

        ### TYPE MEAN HISTOGRAMS FIGURE

        # save data
        hist_data = {
            'param_bin_centers': param_bin_centers,
            'type_hists': type_hists,
            'param_names': self.param_names,
            'type_set': type_set
            }
        with open(self.out_dpath + 'hist_data.json', 'w') as f:
            json.dump(hist_data, f, indent=4, cls=sha.NumpyEncoder)


        row_nb = np.ceil(self.param_nb/2).astype('int')
        mean_fig, mean_axs = plt.subplots(
            ncols=2,
            nrows=row_nb,
            figsize=(15, row_nb * 2),
            dpi=150,
            layout='constrained',
            )
        mean_axs = mean_axs.ravel()

        # display mean histograms
        for i_p in range(self.param_nb):
            for i_t in range(type_nb):
                plt.sca(mean_axs[i_p])
                plt.step(
                    param_bin_centers[i_p, :],
                    type_hists[i_t, i_p, :],
                    # width=width,
                    # alpha=0.4,
                    color=type_colors[i_t],
                    where='mid',
                    )

            # finalize
            plt.sca(mean_axs[i_p])
            plt.legend(type_set)
            title_str = self.param_names[i_p]
            if remove_outliers:
                title_str += ' (' + param_thresh_strs[i_p] + ')'
            plt.title(title_str)

        plt.suptitle(type_subject_nb_str)
        plt.savefig(
            self.out_dpath + 'fig0_mean_distributions.png'
            )

        ### SUBJECT AVERAGE VALUES FIGURE

        # prepare figure for subject average values
        row_nb = np.ceil(test_param_nb/2).astype('int')
        avg_fig, avg_fig_axs = plt.subplots(
            ncols=2,
            nrows=np.ceil(test_param_nb/2).astype('int'),
            figsize=(2 * (1 * type_nb + 1), row_nb * 2),
            dpi=150,
            layout='constrained',
            )
        avg_fig_axs = avg_fig_axs.ravel()

        # define pairs to test
        combinations = itertools.combinations(type_set, 2)
        pairs = [tuple(pair) for pair in combinations]

        for i_p in range(self.param_nb):
            if self.param_names[i_p] in test_params:

                i_t = test_params.index(self.param_names[i_p])

                # prepare distributions of averages
                df = pd.DataFrame({'type': subject_types, 'average': subjects_avgs[:, i_p]})

                # display distributions of averages
                plt.sca(avg_fig_axs[i_t])
                sns.boxplot(data=df, x='type', y='average', palette=type_colors[:type_nb], order=type_set)

                # display title and axes labels
                plt.xlabel('Type')
                plt.ylabel(avg_mode.capitalize())

                if type_nb > 1:

                    type_medians_list = [
                        subjects_avgs[subject_types == ty, i_p].ravel()
                        for ty in type_set
                        ]

                    if avg_mode == 'median':
                        # > 2 samples: Kruskal-Wallis test
                        _, p = scipy.stats.kruskal(*type_medians_list)
                        # 2 samples: Mann-Whitney U-test:
                        test_2s = 'Mann-Whitney'
                    elif avg_mode == 'mean':
                        # > 2 samples: ANOVA
                        _, p = scipy.stats.f_oneway(*type_medians_list)
                        # 2 samples: Mann-Whitney:
                        test_2s = 't-test_ind'

                    print(self.param_names[i_p], p)

                    if p < 0.05:
                        h = True
                    else:
                        h = False

                    # if necessary, post-hoc tests are computed and displayed
                    # (integrating with Bonferroni-Holm correction)
                    if h:
                        # test
                        annot = Annotator(
                            avg_fig_axs[i_t],
                            data=df,
                            pairs=pairs,
                            x='type',
                            y='average',
                            order=type_set,
                            )
                        annot.configure(
                            test=test_2s,
                            text_format='star',
                            verbose=1,
                            loc='inside',
                            # comparisons_correction='Holm-Bonferroni',
                            correction_format='replace'
                            )
                        annot.apply_test()

                        # display tests
                        ax, test_results = annot.annotate(line_offset_to_group=0.2)

                        # increase y-axis top limit
                        min_y, max_y = plt.ylim()
                        ax.set_ylim(min_y, max_y * 1.03)

                        plt.title(test_params[i_t] + ' (p = ' + '{:.2e}'.format(p) + ')')
                else:
                    plt.title(test_params[i_t])


                if self.param_names[i_p] == 'alpha':

                    df.to_csv(self.out_dpath + 'alpha_vals.csv')

                    vis.plot_alpha(df, avg_mode)

                    plt.savefig(
                        self.out_dpath + 'fig0_alpha_avgs.png'
                        )
                    plt.savefig(
                        self.out_dpath + 'fig0_alpha_avgs.svg'
                        )

        # finalize figure
        plt.suptitle(type_subject_nb_str)
        plt.savefig(
            self.out_dpath + 'fig0_subject_avgs_test.png'
            )

        ### PAIRWISE PARAMETER CORRELATIONS FIGURE

        # save data
        avg_data = {
            'subjects_avgs': subjects_avgs,
            'param_names': self.param_names,
            'type_set': type_set,
            'subject_types': subject_types,
            }
        with open(self.out_dpath + 'avg_data.json', 'w') as f:
            json.dump(avg_data, f, indent=4, cls=sha.NumpyEncoder)

        _ = plt.figure(dpi=150, figsize=(8, 8))
        corr_params = set(test_params) - set(['cost'])
        data_dict = {
            name: subjects_avgs[:, i_p]
            for i_p, name in enumerate(self.param_names)
            if name in corr_params
            }

        data_dict.update({'Type': subject_types})
        df = pd.DataFrame(data_dict)

        sns.pairplot(
            df,
            hue='Type',
            palette=type_colors[:type_nb],
            hue_order=type_set,
            )

        # save figure
        plt.savefig(
            self.out_dpath + 'fig0_pairplot.png'
            )

        ### IMAGES PARAMETER HISTOGRAMS FIGURES

        imgs_hists_L = [
            img_hists[:, i_p, :].squeeze() for i_p in range(self.param_nb)
            ]
        imgs_avg_vals_L = [
            imgs_avg_vals[:, i_p] for i_p in range(self.param_nb)
            ]

        # save data
        param_hist_data = {
            'img_hists': img_hists,
            'imgs_avg_vals': imgs_avg_vals,
            'img_types': img_types,
            'param_mins': param_mins,
            'param_maxs': param_maxs,
            'param_names': self.param_names,
            }
        with open(self.out_dpath + 'param_hist_data.json', 'w') as f:
            json.dump(param_hist_data, f, indent=4, cls=sha.NumpyEncoder)

        # create parameter figures
        args_iter = zip(
            imgs_hists_L,
            imgs_avg_vals_L,
            param_mins,
            param_maxs,
            self.param_names,
            param_thresh_strs,
            )
        kwargs_dict = {
            'types_img_nbs': types_img_nbs,
            'type_nb': type_nb,
            'type_set': type_set,
            'img_types': img_types,
            'bin_nb': bin_nb,
            'type_colors': type_colors,
            }
        self.process_tasks(
            args_iter,
            self.create_imgs_param_hists_fig,
            task_nb=self.param_nb,
            description='Create parameter figures',
            kwargs_dict=kwargs_dict,
            )

        return

    def create_imgs_param_hists_fig(
            self,
            imgs_hists,
            imgs_avg_vals,
            param_min,
            param_max,
            param_name,
            param_thresh_str,
            type_nb,
            types_img_nbs,
            type_set,
            img_types,
            bin_nb,
            type_colors,
            ):

        # compute number of rows for single-parameter figures
        param_row_nb = max(types_img_nbs)

        # prepare parameter figure
        param_fig, param_axs = plt.subplots(
            ncols=type_nb,
            nrows=param_row_nb,
            figsize=(type_nb * 3, param_row_nb * 1 + 1),
            dpi=150,
            sharex='all',
            sharey='all',
            squeeze=False,
            )

        # loop on types
        for i_t, ty in enumerate(type_set):

            type_img_nb = types_img_nbs[i_t]
            type_img_ixs = np.nonzero(img_types == ty)[0]

            # loop on images of given type
            for i_i in range(type_img_nb):

                img_ix = type_img_ixs[i_i]

                hist_vals = imgs_hists[img_ix, :].squeeze()
                avg_val = imgs_avg_vals[img_ix]

                # compute plot parameters
                bin_edges = np.histogram_bin_edges(
                    None,
                    bins=bin_nb,
                    range=(param_min,param_max)
                    )
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                width = np.diff(bin_edges[:2])

                # plot histograms
                plt.sca(param_axs[i_i, i_t])
                plt.bar(
                    bin_centers,
                    hist_vals,
                    width=width,
                    color=type_colors[i_t],
                    )

                # plot fitted distribution
                if param_name == 'alpha':
                    if self.asym_norm_distrib == 'Johnson-SU':
                        img_params = self.js_params[img_ix, :]
                        fit_dist = scipy.stats.johnsonsu.pdf(bin_centers.astype('float'), *img_params)

                    elif self.asym_norm_distrib == 'Skew normal':
                        fit_dist = scipy.stats.skewnorm.pdf(
                            bin_centers.astype('float'),
                            self.skew_vals[img_ix],
                            self.mean_vals[img_ix],
                            self.var_vals[img_ix],
                            )

                    else:
                        print('Undefined asymmetrical distribution.')
                        break

                    fit_dist *= hist_vals.sum() / fit_dist.sum()
                    plt.plot(bin_centers, fit_dist, linestyle=':', color='gray')

                # plot median
                plt.axvline(x=avg_val, linestyle=':', color='k')

                # add title
                plt.title(self.set_info[0]['name'].iloc[img_ix])

        # finalize single-parameter figure
        plt.figure(param_fig)
        plt.suptitle(param_name + ' (' + param_thresh_str + ')')
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(
            self.out_dpath + 'fig1_overview_' + param_name + '.png'
            )
        plt.close(param_fig)

        return

    def load_img(
            self,
            img_path,
            weight_fpath,
            avg_mode,
            min_count,
            remove_outliers
            ):

        # find index of count variable
        count_var_ix = self.param_names.index('count')

        # load image data
        img = pro.read_img(img_path)
        array = img.numpy()

        if array.ndim < 4:
            array = np.expand_dims(array, axis=2)

        data = array[:, :, :, self.channels]

        # remove voxels where photon count is too low
        # (it includes voxels outside of mask)
        valid_vox_bools = data[:, :, :, count_var_ix] > min_count
        data = data[valid_vox_bools, :]

        # load weight
        if weight_fpath is not None:
            weight_array = pro.read_img(weight_fpath).numpy()
            weight_array = weight_array[valid_vox_bools]
            weight_array /= weight_array.sum()
        else:
            weight_array = None

        thresh_lows = np.zeros((1, self.param_nb))
        thresh_highs = np.zeros((1, self.param_nb))

        for i_p, p_name in enumerate(self.param_names):

            thresh_lows[0, i_p] = np.percentile(
                data[:, i_p],
                self.perc_threshs[i_p, 0]
                )
            thresh_highs[0, i_p] = np.percentile(
                data[:, i_p],
                self.perc_threshs[i_p, 1]
                )

        # apply thresholds before average
        if remove_outliers:
            lgcs = np.logical_and(np.all(data >= thresh_lows, axis=1), np.all(data <= thresh_highs, axis=1))

            data = data[lgcs, :]

            if weight_array is not None:
                weight_array = weight_array[lgcs]

        # compute average values
        if avg_mode == 'median':
            if weight_array is None:
                avg = np.median(data, axis=0)
            else:
                avg = np.array([
                    robustats.weighted_median(data[:, i], weight_array)
                    for i in range(self.param_nb)
                     ])
        elif avg_mode == 'mean':
            avg = np.average(data, axis=0, weights=weight_array)

        # compute CV
        cv = scipy.stats.variation(data, ddof=1, axis=0)

        # apply thresholds after average
        if not remove_outliers:
            lgcs = np.logical_and(np.all(data >= thresh_lows, axis=1), np.all(data <= thresh_highs, axis=1))
            data = data[lgcs, :]

            if weight_array is not None:
                weight_array = weight_array[lgcs]

        # compute extrema
        min_vals = np.amin(data, axis=0)
        max_vals = np.amax(data, axis=0)

        return data, min_vals, max_vals, avg, cv, weight_array

    @staticmethod
    def compute_img_hists(img_vals, hist_min_vals, hist_max_vals, bin_nb):

        param_nb = img_vals.shape[1]
        img_hists = np.zeros((param_nb, bin_nb))

        for i_p in range(param_nb):

            hist_vals, _ = np.histogram(
                img_vals[:, i_p],
                bins=bin_nb,
                range=(hist_min_vals[i_p], hist_max_vals[i_p])
                )

            # normalize to density
            hist_vals = hist_vals / hist_vals.sum()

            # store histogram
            img_hists[i_p, :] = hist_vals

        return img_hists


class SetPlotTypicalFits(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = False
    SAVE_INFO = True
    MP = True
    EXT = '.png'

    # takes three directories as input:
    # - raw FLIM images
    # - fitted parameters (to get cost)
    # - responses images
    def process_set(self, avg_all=False):

        # load model_info JSON file
        with open(self.in_dpath[1] + 'fit_info.json', 'r') as json_file:
            model_info = json.load(json_file)

        param_names = model_info['fit_var_names']

        # merge to get corresponding raw image for each parameter image
        raw_set_info = self.set_info[0]
        param_set_info = self.set_info[1]
        merge_info = pd.merge(param_set_info, raw_set_info, on='name')

        raw_fpaths = (self.in_dpath[0] + merge_info['path_y']).tolist()

        # other filepaths should correspond
        param_fpaths = self.in_fpaths[1]
        resp_fpaths = self.in_fpaths[2]

        input_tuples = zip(
            raw_fpaths,
            param_fpaths,
            resp_fpaths,
            self.out_fpaths,
            )

        kwargs_dict = {
            'param_names': param_names,
            'avg_all': avg_all,
            }

        self.process_tasks(
                input_tuples,
                plot_typical_fits,
                self.in_img_nb[1],
                description='Plot typical fits',
                waitbar=True,
                kwargs_dict=kwargs_dict,
                )


class SetPlotFitParameters(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = True
    MP = True
    EXT = '.png'
    SAVE_INFO = False

    def process_set(self, norm_dict, mode='maxproj', fit_info_dpath=None, params=None, colors='inferno'):

        # load model_info JSON file
        if fit_info_dpath is None:
            fit_info_dpath = self.in_dpath
        with open(fit_info_dpath + 'fit_info.json', 'r') as json_file:
            model_info = json.load(json_file)

        param_names = model_info['fit_var_names']

        if params is None:
            disp_params = param_names
        else:
            disp_params = params

        for i_p, p_name in enumerate(param_names):

            if p_name not in disp_params:
                print('xxx')
                continue

            if p_name in norm_dict:
                norm = norm_dict[p_name]
            else:
                norm='minmax'

            vis.SetPlotProj(
                self.in_dpath,
                self.out_dpath + p_name + '_figs/',
                component=i_p,
                norm=norm,
                colors=colors,
                white_zeros=True,
                mode=mode,
                )


class SetPlotParametersDensity(sha.AbstractSetProcess):

    SINGLE_INPUT_ONLY = True
    MP = True
    EXT = '.png'

    def process_set(self, norm_dict, names):

        # load model_info JSON file
        with open(self.in_dpath + 'fit_info.json', 'r') as json_file:
            model_info = json.load(json_file)

        param_names = model_info['fit_var_names']

        ix1 = param_names.index(names[0])
        ix2 = param_names.index(names[1])

        hist_range = [norm_dict[n] for n in names]

        args_iter = zip(
            self.in_fpaths,
            self.out_fpaths,
            )

        kwargs_dict = {
            'ixs': [ix1, ix2],
            'names': names,
            'hist_range': hist_range,
            }

        self.process_tasks(
                args_iter,
                self.read_plot_save,
                self.in_img_nb,
                description='Plot density',
                waitbar=True,
                kwargs_dict=kwargs_dict,
                )

    def read_plot_save(self, in_fpath, out_fpath, ixs, names, hist_range):

        # load image data
        img = pro.read_img(in_fpath)
        data = img.numpy()

        # remove voxels where all features are equal to zero
        # (i.e voxels outside of mask)
        valid_vox_bools = np.logical_not(np.all(data == 0, axis=3))
        data = data[valid_vox_bools, :]

        # select parameters
        data = data[:, ixs]

        plot_density(data, names, hist_range)

        # save
        plt.tight_layout()
        plt.savefig(out_fpath)


def plot_irf(fpath):

    with open(fpath, 'r') as json_file:
        file_data = json.load(json_file)

    t = np.array(file_data['times'])
    irf = np.array(file_data['irf'])

    _, ax = plt.subplots(nrows=1, ncols=1, sharex='col', dpi=150, figsize=(5, 3))

    plt.sca(ax)

    plt.plot(t, irf)
    plt.xlabel('Time (s)')
    plt.ylabel('IRF')


def plot_typical_fits(raw_img, fit_img, resp_img, out_fpath, param_names, avg_all=False):

    # load raw FLIM image
    raw_img = pro.read_img(raw_img)
    raw_array = raw_img.numpy()

    if avg_all:
        decay_sum = raw_array.sum(axis=(0, 1, 2))
        raw_array = np.empty((3, 3, 3, raw_img.shape[3]))
        raw_array[:, :, :, :] = decay_sum

    # load parameters image
    fit_img = pro.read_img(fit_img)
    fit_array = fit_img.numpy()

    if fit_array.ndim < 4:
        fit_array = np.expand_dims(fit_array, axis=2)

    # define valid voxels
    valid_bools = np.sum(fit_array, axis=3) > 0

    # load raw samples and fitted responses
    raw_samples = raw_array[valid_bools, :]
    resp_samples = pro.read_img(resp_img).numpy()[valid_bools, :]

    # cost is assumed to be in the last variable
    cost_ix = param_names.index('cost')
    sample_costs = fit_array[valid_bools, cost_ix]

    N = 5
    fit_types = ['worst', 'average', 'best']
    sort_ixs = np.argsort(sample_costs)
    times = raw_img.spacing[3] * np.arange(raw_array.shape[3])

    fig = plt.figure(figsize=(15, 15), layout='constrained')
    import matplotlib.gridspec as gridspec
    outer = gridspec.GridSpec(5, 3, wspace=0.2, hspace=0.2, figure=fig)

    for i_f, fit_type in enumerate(fit_types):

        if fit_type == 'worst':
            ixs = sort_ixs[-N:]

        elif fit_type == 'average':
            mid_cost_ix = np.round(len(sample_costs) / 2)
            start_ix = int(mid_cost_ix - np.round(N / 2))
            ixs = sort_ixs[start_ix:(start_ix + N)]

        elif fit_type == 'best':
            ixs = sort_ixs[:N]

        sel_raw_samples = raw_samples[ixs, :]
        sel_resp_samples = resp_samples[ixs, :]
        resp_len = sel_resp_samples.shape[1]

        for i_s in range(N):

            out_ix = i_f + 3 * i_s
            inner = gridspec.GridSpecFromSubplotSpec(3, 1,
                            subplot_spec=outer[out_ix],
                            # wspace=0.1, hspace=0.1,
                            )

            upper_ax = fig.add_subplot(inner[:2])
            sel_raw_sample = sel_raw_samples[i_s, :]
            plt.plot(times, sel_raw_sample)
            plt.plot(
                times[:resp_len],
                sel_resp_samples[i_s, :]
                )
            upper_ax.tick_params(labelbottom=False)

            if i_f == 0:
                plt.ylabel('Count')


            _ = fig.add_subplot(inner[2])
            res = sel_raw_sample[:resp_len] - sel_resp_samples[i_s, :]
            plt.axhline(y=0, color=0.3*np.ones(3), linestyle=':')
            plt.plot(times[:resp_len],  res, 'k')

            if i_s == N - 1:
                plt.xlabel('Time (s)')

    plt.suptitle('Worst, average and best fits')
    plt.savefig(out_fpath)
    plt.close(fig)

    return


def plot_density(data, names=None, ranges=None):

    from scipy.stats import kde

    hist_ranges = np.zeros((2, 2))

    for i_r, r in enumerate(ranges):
        if r is None or r == 'minmax':
            hist_ranges[i_r, :] = [np.amin(data[:, i_r]), np.amax(data[:, i_r])]
        else:
            hist_ranges[i_r, :] = r

    data = data[np.all(data > hist_ranges[:, 0].reshape(1, -1), axis=1), :]
    data = data[np.all(data < hist_ranges[:, 1].reshape(1, -1), axis=1), :]

    nbins = 20
    x, y = data.T

    res = scipy.stats.spearmanr(x, y)

    # Create a figure with 6 plot areas
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(15, 5), dpi=150, sharex='row', sharey='col')
    axes = axes.ravel()
    ix = 0

    # Everything starts with a Scatterplot
    ax = axes[ix]
    ax.set_title("r = " + "{:.2f}".format(res.statistic) + " (p = " + "{:.2e}".format(res.pvalue) + ")")
    ax.plot(x, y, 'ko')
    ax.set_xlim(hist_ranges[0])
    ax.set_ylim(hist_ranges[1])
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ix += 1

    # 2D Histogram
    ax = axes[ix]
    ax.hist2d(x, y, bins=nbins, range=hist_ranges, cmap='inferno')
    ax.set_xlabel(names[0])
    ix += 1

    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    k = kde.gaussian_kde(np.vstack((x, y)))
    xi, yi = np.mgrid[hist_ranges[0][0]:hist_ranges[0][1]:nbins*1j, hist_ranges[1][0]:hist_ranges[1][1]:nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # 2D Density with shading and contour
    ax = axes[ix]
    ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap='inferno')
    ax.contour(xi, yi, zi.reshape(xi.shape))
    ax.set_xlabel(names[0])
    ix += 1