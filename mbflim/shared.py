#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from abc import ABC, abstractmethod
import json
import inspect
import tempfile
import fnmatch
import multiprocessing.pool as mpp
import datetime
import sys
from itertools import repeat
from glob import glob
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm


class Timer(object):

    def __init__(self, total, text=''):
        self.start = datetime.datetime.now()
        self.total = float(total)

        if text:
            text = text + ' '
        self.text = text

    def remains(self, done):
        now  = datetime.datetime.now()
        done = float(done)

        left = (self.total - done) * (now - self.start) / done
        d = left.days
        s = left.seconds

        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)

        time_str = '{:02} day(s) {:02}:{:02}:{:02}'.format(int(d), int(h), int(m), int(s))

        msg_str = (self.text + "[" + str(int(done)) + "/"
                   + str(int(self.total)) + "] "
                   + time_str + " remaining")

        return msg_str


# Modify global variables involved in multi-threading
def set_thread_nb(thread_nb):

    thread_nb = int(thread_nb)

    os.environ["OMP_NUM_THREADS"] = str(thread_nb)
    os.environ["OPENBLAS_NUM_THREADS"] = str(thread_nb)
    os.environ["MKL_NUM_THREADS"] = str(thread_nb)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(thread_nb)
    os.environ["NUMEXPR_NUM_THREADS"] = str(thread_nb)
    os.environ["BLIS_NUM_THREADS"] = str(thread_nb)
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(thread_nb)


# Patch allowing tdqm to be used with starmap
# obtained from:
# https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm
def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap


# Wrapper for istarmap
def istarmap_with_kwargs(pool, fn, args_iter, kwargs_iter=repeat({})):

    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)

    return pool.istarmap(apply_args_and_kwargs, args_for_starmap)


def apply_args_and_kwargs(fn, args, kwargs=None):

    if kwargs is None:
        return fn(*args)
    else:
        return fn(*args, **kwargs)


# JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def safe_serialize(obj):
  default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
  return json.dumps(obj, default=default, indent=4)


# Decorator
def store_arguments(func):
    def inner(*args, **kwargs):

        pos_arg_names = inspect.getfullargspec(func).args

        arg_dict = {}
        for arg_name, arg_val in zip(pos_arg_names, args):
            arg_dict.update({arg_name: arg_val})

        arg_dict.update(kwargs)

        output_path = arg_dict['output_dirpath']
        os.makedirs(output_path, exist_ok=True)

        date_time = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")

        # check for existing argument files
        arg_fpaths = glob(output_path + 'args_' + '*.json')

        # if more than 2 exist, order by date and keep only last 2 ones
        if len(arg_fpaths) > 2:
            arg_fpaths = sorted(arg_fpaths)
            for p in arg_fpaths[:-2]:
                try:
                    os.remove(p)
                except Exception:
                    warnings.warn('Could not remove old argument file.')

        # write argument file
        with open(output_path + 'args_' + date_time + '.json', 'w') as f:
            f.write(safe_serialize(arg_dict))

        func(*args, **kwargs)

    return inner


def isnamedtupleinstance(x):
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple: return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple): return False
    return all(type(n)==str for n in f)


def distribute(oranges, plates):
    base, extra = divmod(oranges, plates)
    return [base + (i < extra) for i in range(plates)]


# Transform list of information of any type into array of integers or other
# values defined in 'map_values'
def map_info(info_values, map_values=None):

    info_set = sorted(set(info_values))

    if map_values is None:
        map_values = np.arange(len(info_set))

    info_map = {info: map_values[i] for i, info in enumerate(info_set)}
    mapped_info = [info_map[info_val] for info_val in info_values]
    mapped_info = np.array(mapped_info)

    return info_set, mapped_info


class AbstractSetProcess(ABC):
    """
    Abstract class defining the skeleton of algorithms used to process a set of
    images.
    """

    """
    set_info

    in_dpath
    in_fpaths
    in_loc_fpaths

    out_dpath
    out_fpaths
    out_loc_fpaths
    extension
    prefix

    tmp_dir_obj
    tmp_dpath
    """

    NEED_TMP = False
    EXT = '.nii.gz'
    SAVE_INFO = True
    MP = False
    SINGLE_INPUT_ONLY = False

    @store_arguments
    def __init__(
            self,
            input_dirpath,
            output_dirpath,
            *args,
            name_filter=None,
            name_filterout=None,
            process_nb=None,
            max_thread_nb=None,
            extension=None,
            prefix='',
            **kwargs
            ):

        # store
        self.in_dpath = input_dirpath
        self.out_dpath = output_dirpath
        self.extension = extension
        self.prefix = prefix
        self.name_filter = name_filter
        self.name_filterout = name_filterout
        self.args = args
        self.kwargs = kwargs

        if extension is None:
            self.extension = self.EXT
        else:
            self.extension = extension

        if process_nb is None:
            if self.MP:
                self.process_nb = int(os.environ["MBFLIM_NUM_CPUS"])
            else:
                self.process_nb = 1
        else:
            self.process_nb = process_nb

        if max_thread_nb is None:
            self.max_thread_nb = int(os.environ["MBFLIM_NUM_THREADS"])
        else:
            self.max_thread_nb = max_thread_nb

        # set current thread number
        self.set_thread_nb()

        # prepare paths and info dataframes
        self.prepare_files_info()

        # create output directory
        os.makedirs(self.out_dpath, exist_ok=True)

        # create temporary directory
        if self.NEED_TMP:
            self.create_tmp_dir()

        self.process_set(*args, **kwargs)

        # save output info
        if self.SAVE_INFO:
            self.out_info.to_csv(self.out_dpath + 'files.csv')

        if self.NEED_TMP:
            self.tmp_dir_obj.cleanup()


    def prepare_files_info(self):

        if not isinstance(self.in_dpath, list):
            self.in_dpath = [self.in_dpath]

        self.input_nb = len(self.in_dpath)

        if self.SINGLE_INPUT_ONLY and self.input_nb > 1:
            raise ValueError(
                'This method only accepts a single input directory.')

        self.set_info = []
        self.in_names = []
        self.in_loc_fpaths = []
        self.in_fpaths = []
        self.in_img_nb = []

        # prepare name filter
        if self.name_filter is not None:
            if self.input_nb == 1:
                self.name_filter = [self.name_filter]
            elif len(self.name_filter) != self.input_nb:
                raise ValueError("Number of name filters does not correspond "
                                 + "to number of inputs.")
        # prepare name filter out
        if self.name_filterout is not None:
            if self.input_nb == 1:
                self.name_filterout = [self.name_filterout]
            elif len(self.name_filterout) != self.input_nb:
                raise ValueError("Number of name filters does not correspond "
                                 + "to number of inputs.")

        for i, in_p in enumerate(self.in_dpath):

            # load set info DataFrame
            set_info = pd.read_csv(in_p + 'files.csv',
                                   index_col=0)

            if self.name_filter is not None:
                name_filters = self.name_filter[i]

                bool_V = [self.multiple_fnmatch(set_info['name'][i], name_filters)
                          for i in range(len(set_info))]
                set_info = set_info[bool_V]

                set_info = set_info.reset_index(drop=True)

            if self.name_filterout is not None:
                name_filterouts = self.name_filterout[i]

                bool_V = [self.multiple_fnmatch(set_info['name'][i], name_filterouts)
                          for i in range(len(set_info))]
                set_info = set_info[np.logical_not(bool_V)]

                set_info = set_info.reset_index(drop=True)

            self.set_info.append(set_info)
            self.in_img_nb.append(len(set_info))

            # extract info
            self.in_names.append(self.set_info[-1]['name'].tolist())
            self.in_loc_fpaths.append(self.set_info[-1]['path'].tolist())

            # prepare input filepaths
            self.in_fpaths.append([in_p + p for p in self.in_loc_fpaths[-1]])

        # prepare output filepaths
        self.out_loc_fpaths = [self.prefix + n + self.extension
                              for n in self.set_info[0]['name']]
        self.out_fpaths = [self.out_dpath + p for p in self.out_loc_fpaths]

        # prepare default output info
        self.out_info = self.set_info[0].copy()
        self.out_info['path'] = self.out_loc_fpaths

        # no list if single input path
        if self.SINGLE_INPUT_ONLY:
            self.in_dpath = self.in_dpath[0]
            self.set_info = self.set_info[0]
            self.in_img_nb = self.in_img_nb[0]
            self.in_names = self.in_names[0]
            self.in_loc_fpaths = self.in_loc_fpaths[0]
            self.in_fpaths = self.in_fpaths[0]


    def set_thread_nb(self, thread_nb=None, process_nb=None):

        if process_nb is None:
            process_nb = self.process_nb

        if thread_nb is None:
            self.thread_nb = int(self.max_thread_nb / process_nb)
        else:
            self.thread_nb = thread_nb

        if self.thread_nb == 0:
            raise Exception("The number of threads cannot be 0.")

        if self.thread_nb * process_nb > self.max_thread_nb:
            raise Warning("The total number of threads could exceed the "
                          + "maximum allowed number of threads.")

        os.environ["OMP_NUM_THREADS"] = str(self.thread_nb)
        os.environ["OPENBLAS_NUM_THREADS"] = str(self.thread_nb)
        os.environ["MKL_NUM_THREADS"] = str(self.thread_nb)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(self.thread_nb)
        os.environ["NUMEXPR_NUM_THREADS"] = str(self.thread_nb)
        os.environ["BLIS_NUM_THREADS"] = str(self.thread_nb)
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(self.thread_nb)


    @abstractmethod
    def process_set(self, *args, **kwargs):
        pass


    def create_tmp_dir(self):

        # create temporary folder object
        os.makedirs(os.environ["TMPDIR"], exist_ok=True)
        self.tmp_dir_obj = tempfile.TemporaryDirectory(
            dir=os.environ["TMPDIR"]
            )

        # prepare temporary folder path
        self.tmp_dpath = self.tmp_dir_obj.name + '/'
        os.makedirs(self.tmp_dpath, exist_ok=True)


    @staticmethod
    def multiple_fnmatch(string, patterns):

        if not isinstance(patterns, list):
            patterns = [patterns]

        for pattern in patterns:
            if fnmatch.fnmatch(string, pattern):
                return True
        return False


    def process_tasks(
            self,
            args_iter,
            function,
            task_nb,
            description,
            kwargs_dict={},
            waitbar=True,
            process_nb=None,
            method='fork',
            ):

        disable = not(waitbar)

        if process_nb is None:
            process_nb = min(task_nb, self.process_nb)

        # adapt number of threads to number of processes
        self.set_thread_nb(process_nb=process_nb)

        if process_nb > 1:

            with mpp.get_context(method).Pool(process_nb) as pool:
                results = list(tqdm(
                    istarmap_with_kwargs(
                        pool,
                        function,
                        args_iter,
                        repeat(kwargs_dict)
                        ),
                    total=task_nb,
                    desc=description,
                    file=sys.stdout,
                    smoothing=0,
                    disable=disable))

        else:
            results = []

            for input_tuple in tqdm(
                    args_iter,
                    total=task_nb,
                    desc=description,
                    file=sys.stdout,
                    disable=disable):

                result = function(*input_tuple, **kwargs_dict)
                results.append(result)

        # return to default
        self.set_thread_nb()

        return results







