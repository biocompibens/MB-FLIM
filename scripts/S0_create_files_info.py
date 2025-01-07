#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create "files.csv" files in the directories downloaded from BioImage Archive

"""

import os
import glob

import pandas as pd

import mbflim.utils as ut

config = ut.load_config()
source_dpath = config['paths']['source_dpath']

lists_dpath = os.path.join(source_dpath, 'file_lists')
list_fpaths = glob.glob(os.path.join(lists_dpath, '*.tsv'))

flim_dnames = ['marker', 'FLIM', 'IRF']

for fpath in list_fpaths:

    # Extract the filename
    fname = os.path.basename(fpath)

    # Remove the extension
    base_name = os.path.splitext(fname)[0]

    # Split the base name into parts
    imaging, study, _ = base_name.split('_')

    # Load the list
    list_df = pd.read_csv(fpath, delimiter='\t')

    # Change all column names to lowercase
    list_df.columns = list_df.columns.str.lower()

    # Rename the column "files" to "path"
    list_df = list_df.rename(columns={"files": "path"})

    # Replace the paths with just the filename and extension
    list_df["path"] = list_df["path"].apply(os.path.basename)

    if imaging == 'FLIM-marker':

        # Create a list to store the file info DataFrames
        info_dfs = [
            list_df[list_df["type"] == name].drop(columns=["type"])
            for name in flim_dnames
        ]

        # Prepare output paths
        out_fpaths = [
            os.path.join(source_dpath, imaging, study, dname, 'files.csv') for dname in flim_dnames
        ]

    elif imaging == 'markers':

        # Create a list to store the file info DataFrames
        info_dfs = [list_df.drop(columns=["type"])]

        # Prepare output paths
        out_fpaths = [os.path.join(source_dpath, imaging, study, 'files.csv')]

    else:
        raise ValueError('Unknown imaging type.')

    for info_df, out_fpath in zip(info_dfs, out_fpaths):

        print(out_fpath)
        print(info_df)

        info_df.to_csv(out_fpath)







