# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 15:33:13 2021
@author: User
"""

import netCDF4
import numpy as np

filepath = 'D:/Data/InfoClimatNew/NetCDF/PREC_1984.nc'
netcdf_dset = netCDF4.Dataset(filepath, 'r+')

print(netcdf_dset)

latitude = np.array(netcdf_dset['lat'])
longitude = np.array(netcdf_dset['lon'])
time = np.array(netcdf_dset['time'])
prec = np.array(netcdf_dset['PREC'])

netcdf_dset.close()
