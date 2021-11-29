# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 15:33:13 2021
@author: User
"""

import netCDF4
import numpy as np

filepath = 'D:/Data/GrilleInfoClimat2021/PREC_1985.nc'
netcdf_dset = netCDF4.Dataset(filepath, 'r+')

print(netcdf_dset)

x = np.array(netcdf_dset['x'])
y = np.array(netcdf_dset['y'])

latitude = np.array(netcdf_dset['lat'])
longitude = np.array(netcdf_dset['lon'])
time = np.array(netcdf_dset['time'])
prec = np.array(netcdf_dset['PREC'])

netcdf_dset.title

netcdf_dset.close()
