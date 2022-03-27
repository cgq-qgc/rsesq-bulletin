# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 15:33:13 2021
@author: User
"""

import netCDF4
import numpy as np

for year in range(1980, 2022):
    # filepath = 'C:/Users/jean-/Downloads/PREC_2020.nc'
    filepath = f'D:/Data/GrilleInfoClimat2021/PREC_{year}.nc'
    netcdf_dset = netCDF4.Dataset(filepath, 'r+')

    time = np.array(netcdf_dset['time'])
    prec = np.array(netcdf_dset['PREC'])

    print(year, '-----------------------')
    for i in range(365):
        maxval = np.max(prec[i, :, :])
        if maxval > 100:
            print(i, '{:0.2f}'.format(maxval))
    # netcdf_dset.close()
    break
