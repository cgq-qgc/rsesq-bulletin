# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 16:27:49 2021
@author: User
"""
import sys
import datetime

sys.path.append('C:/Users/User/rsesq-bulletin')
sys.path.append('C:/Users/User/sardes')

from infoclimat_reader import InfoClimatGridReader
from sardes.database.accessors import DatabaseAccessorSardesLite

# %%
# Extraire les informations des stations du RSESQ de la base de données.

database = "D:/Desktop/rsesq_prod_28-06-2021.db"
accessor = DatabaseAccessorSardesLite(database)
accessor.connect()
obs_wells = accessor.get_observation_wells_data()
accessor.close_connection()

# %%
# Extraire les données météos des grilles interpolée d'Info-climat.

infoclim_reader = InfoClimatGridReader("D:/Data/GrilleInfoClimat2021")

loc_id = obs_wells['obs_well_id'].values.tolist()
lat_dd = obs_wells['latitude'].astype(float).values.round(8).tolist()
lon_dd = obs_wells['longitude'].astype(float).values.round(8).tolist()

connect_table = infoclim_reader.create_connect_table(
    lat_dd, lon_dd, loc_id)
connect_table.save_to_csv('connect_table_rsesq.csv')

for varname in InfoClimatGridReader.VARNAMES:
    first_year = 1960
    last_year = 2021
    climate_data = infoclim_reader.get_climate_data(
        varname, connect_table, first_year=first_year, last_year=last_year)
    climate_data.save_to_csv(
        f'{varname}_for_rsesq_{first_year}-{last_year}.csv')
