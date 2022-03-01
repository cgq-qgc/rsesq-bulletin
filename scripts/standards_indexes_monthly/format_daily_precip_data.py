# -*- coding: utf-8 -*-
"""
A script to prepare precipitation data for SPI and SPLI calculations.
"""
from datetime import datetime
import os.path as osp
import pandas as pd
import numpy as np

from selected_stations import selected_stations

outdir = osp.dirname(__file__)
dirname = osp.join(osp.dirname(outdir), 'climate_data_rsesq')
filename = "PREC_for_rsesq_1960-2021.csv"

precip = pd.read_csv(
    osp.join(dirname, filename),
    header=[3, 4, 5],
    index_col=0,
    parse_dates=[0])
precip.index.name = 'datetime'
precip.columns = precip.columns.droplevel(level=[1, 2])

connect_table = pd.read_csv(
    osp.join(dirname, "connect_table_rsesq.csv"),
    index_col=0)
connect_table = connect_table.set_index('loc_id')

grid_indexes = connect_table.loc[selected_stations].grid_idx.astype(str).values
site_precip = precip.loc[:, grid_indexes]
site_precip.columns = selected_stations

# Some precip values are way to large on 23-12-2020 and need to be
# corrected.
date = datetime(2020, 12, 23)
mask = (site_precip.loc[date] > 100)
if np.sum(mask.values) == 0:
    raise ValueError("It appears the error was corrected. "
                     "This might not be necessary anymore")
site_precip.loc[date, mask] = site_precip.loc[date, mask] / 1000

site_precip.to_csv(osp.join(outdir, "precip_daily_2022-02-28.csv"))
