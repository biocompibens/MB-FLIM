#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 13:19:13 2022

@author: proussel
"""

import os
import shutil
from glob import glob
import datetime
import socket
import fcntl
import json
from importlib.resources import files
from pathlib import Path

import numpy as np


# list subdirectories of given directory
def list_subdirs(dpath):

    paths = glob(dpath + '*')

    subdir_paths = []
    for p in paths:
        if os.path.isdir(p):
            subdir_paths.append(p + '/')

    return subdir_paths


def empty_dir(dpath):
    for filename in os.listdir(dpath):
        file_path = os.path.join(dpath, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


# time display function
def print_time(text=""):
    time_str = (
        datetime.datetime.now().strftime("\n%y/%m/%d %H:%M:%S")
        + " ------------- "
        )
    print(time_str, text)


def print_memory_check(dpath):
    statvfs = os.statvfs(dpath)

    # bytes to Gb
    f = 1e9

    print('Memory check:')
    # Size of filesystem
    print(' Size = ' + str(statvfs.f_frsize * statvfs.f_blocks / f) + 'Gb')
    # Actual number of free bytes
    print(' Free = ' + str(statvfs.f_frsize * statvfs.f_bfree / f) + 'Gb')
    # Number of free bytes that ordinary users are allowed to use
    # (excl. reserved space)
    print(' Available = ' + str(statvfs.f_frsize * statvfs.f_bavail / f) + 'Gb')


def acquireLock(lock_fpath):
    ''' acquire exclusive lock file access '''
    locked_file_descriptor = open(lock_fpath, 'w+')
    fcntl.lockf(locked_file_descriptor, fcntl.LOCK_EX)
    return locked_file_descriptor


def releaseLock(locked_file_descriptor):
    ''' release exclusive lock file access '''
    locked_file_descriptor.close()


def format_pvalues(p_values):

    text_pvalues = []

    for p_val in p_values:
        if p_val < 1e-3:
            txt = r'$\ast\ast\!\ast$'
        elif p_val < 1e-2:
            txt = r'$\ast\ast$'
        elif p_val < 5e-2:
            txt = r'$\ast$'
        else:
            txt = 'p = ' + str(np.round(p_val, 2))

        text_pvalues.append(txt)

    return text_pvalues


# Load the configuration file from the package
def load_config():

    config_path = Path(str(files("mbflim").joinpath(''))).joinpath("config.json")

    with open(config_path, "r") as f:

        return json.load(f)