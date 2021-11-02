# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 10:47:48 2021
@author: User
"""

# Note: There is currently no binary wheel for Fiona and Rasterio that are
# available on Pypi. So if using a pip installed version of Python, you need
# to install, in the right order, the following packages using wheels from
# Christopher Gohlke’s website.

# (1) https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal
# (2) https://www.lfd.uci.edu/~gohlke/pythonlibs/#fiona
# (3) https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely
# (4) https://www.lfd.uci.edu/~gohlke/pythonlibs/#rasterio

# You will also need a numpy version >= 1.20.0 for contextily to work properly.
# Numpy can be installed directly from PyPi. There is no need to use a wheel
# from Christopher Gohlke’s website.

import os.path as osp
import fiona
import geopandas as gpd
from shapely.geometry import Point

lat_ddeg = 46.40819
lon_ddeg = -70.3709

# https://www.donneesquebec.ca/recherche/dataset/decoupages-administratifs
dirname = "C:/Users/User/rsesq-bulletin"
gdbfile = "SDA_ 2018-05-25 .gdb.zip"

list_layers = fiona.listlayers(osp.join(dirname, gdbfile))

munic_s = gpd.read_file(
    osp.join(dirname, gdbfile), driver='FileGDB', layer='munic_s')


def get_region_mrc_municipality_at(lat_ddeg, lon_ddeg, munic_geometry):
    loc_point = Point(lon_ddeg, lat_ddeg)

    contains = munic_s[munic_s['geometry'].contains(loc_point)]
    region = contains.iloc[0]['MUS_NM_REG']
    mrc = contains.iloc[0]['MUS_NM_MRC']
    municipality = contains.iloc[0]['MUS_NM_MUN']

    return region, mrc, municipality


# %%
from sardes.database.accessors import DatabaseAccessorSardesLite
database = "D:/Desktop/rsesq_prod_28-06-2021.db"
accessor = DatabaseAccessorSardesLite(database)
accessor.connect()

obs_wells = accessor.get_observation_wells_data()
for index, obswell_data in obs_wells.iterrows():
    region, mrc, municipality = get_region_mrc_municipality_at(
        obswell_data['latitude'],
        obswell_data['longitude'],
        munic_s['geometry'])

    if obswell_data['municipality'] != municipality:
        print('| {} | {} | {} | {} | {} |'.format(
            obswell_data['obs_well_id'],
            obswell_data['municipality'],
            municipality,
            obswell_data['latitude'],
            obswell_data['longitude']
            ))

accessor.close_connection()
