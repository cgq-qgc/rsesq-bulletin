# -*- coding: utf-8 -*-
"""
A script to fetch and format readings data from the RSESQ database and export
the results in a csv file compatible with GWHAT format.
"""
from datetime import datetime
import sys
import csv
import os
import os.path as osp
import numpy as np
import pandas as pd

from sardes.api.timeseries import DataType
from sardes.database.accessors import DatabaseAccessorSardesLite
from sardes.utils.data_operations import format_reading_data

dirname = osp.dirname(__file__)
root = osp.dirname(dirname)
sys.path.append(root)

from selected_stations import selected_stations

# =============================================================================
# ---- Format water level data
# =============================================================================
database = "D:/Desktop/rsesq_prod_2022-02-22.db"
accessor = DatabaseAccessorSardesLite(database)
accessor.connect()

stations_data = accessor.get('observation_wells_data')
repere_data = accessor.get('repere_data')

outdir = osp.join(dirname, 'Niveaux')
os.makedirs(outdir, exist_ok=True)
for i, station_name in enumerate(selected_stations):
    print("{:02d} Preparing data for station {}..."
          .format(i + 1, station_name))

    station_data = stations_data[
        stations_data['obs_well_id'] == station_name
        ].iloc[0]
    station_repere = (
        repere_data
        [repere_data['sampling_feature_uuid'] == station_data.name]
        .sort_values(by=['end_date'], ascending=[True]))
    last_repere = station_repere.iloc[-1]
    ground_alt = last_repere['top_casing_alt'] - last_repere['casing_length']

    readings_data = accessor.get_timeseries_for_obs_well(station_data.name)
    formatted_reading_data = format_reading_data(readings_data, station_repere)

    station_wl = formatted_reading_data[
        ['datetime', DataType.WaterLevel]
        ].copy()
    station_wl.columns = ['Time', 'WL']
    station_wl.loc[:, 'WL'] = ground_alt - station_wl['WL'].values

    # Save the data to the CSV file.
    outfile = osp.join(outdir, f'{station_name}_wldata_gwhat.csv')
    station_wl.to_csv(outfile, encoding='utf8', index=False)

    # Add the metadata to the header of the CSV file.
    header = [
        ['Well', station_data['common_name']],
        ['Well ID', station_data['obs_well_id']],
        ['Province', 'Quebec'],
        ['Municipality', station_data['municipality']],
        ['Latitude', station_data['latitude']],
        ['Longitude', station_data['longitude']],
        ['Elevation', ground_alt],
        []
        ]
    with open(outfile, 'r') as csvfile:
        data = list(csv.reader(csvfile))
    with open(outfile, 'w', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
        writer.writerows(header + data)

accessor.close_connection()

# =============================================================================
# ---- Format weather data
# =============================================================================
outdir = osp.join(dirname, 'Météo')
os.makedirs(outdir, exist_ok=True)

connect_table = pd.read_csv(
    osp.join(root, 'climate_data_rsesq', "connect_table_rsesq.csv"),
    index_col=0,
    dtype={'grid_idx': 'str'})
connect_table = connect_table.set_index('loc_id')

grid_indexes = connect_table.loc[selected_stations].grid_idx.values
grid_indexes = np.unique(grid_indexes)

data_all = {}
for var in ['PREC', 'TMIN', 'TMOY', 'TMAX']:
    data = pd.read_csv(
        osp.join(root, 'climate_data_rsesq', f"{var}_for_rsesq_1960-2021.csv"),
        header=[3, 4, 5],
        index_col=0,
        parse_dates=[0])
    data.index.name = 'datetime'
    data.columns = data.columns.droplevel(level=[1, 2])
    data = data.loc[:, grid_indexes]

    if var == 'PREC':
        # Some precip values are way to large on 23-12-2020 and need to be
        # corrected.
        date = datetime(2020, 12, 23)
        mask = data.loc[date] > 100
        if np.sum(mask.values) == 0:
            raise ValueError("It appears the error was corrected. "
                             "This might not be necessary anymore")
        data.loc[date, mask] = data.loc[date, mask] / 1000

    data_all[var] = data

for index in grid_indexes:
    df = pd.DataFrame(data_all['PREC'].loc[:, index])
    df.columns = ['Total Precip']
    df['Max Temp'] = data_all['TMAX'].loc[:, index]
    df['Min Temp'] = data_all['TMIN'].loc[:, index]
    df['Mean Temp'] = data_all['TMOY'].loc[:, index]
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day

    df = df[['Year', 'Month', 'Day', 'Total Precip',
             'Min Temp', 'Mean Temp', 'Max Temp']].copy()

    # Save the data to the CSV file.
    outfile = osp.join(outdir, f'{index}_wxdata_gwhat.csv')
    df.to_csv(outfile, encoding='utf8', index=False)

    # Add the metadata to the header of the CSV file.
    index_info = connect_table[connect_table.grid_idx == index].iloc[0]
    header = [['Station Name', 'Grille Info-climat'],
              ['Station ID', index],
              ['Latitude', index_info['grid_lat_dd']],
              ['Longitude', index_info['grid_lon_dd']],
              []
              ]
    with open(outfile, 'r') as csvfile:
        data = list(csv.reader(csvfile))
    with open(outfile, 'w', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
        writer.writerows(header + data)
