# -*- coding: utf-8 -*-
"""
A script to fetch and format readings data from the RSESQ database and export
the results in a csv file.
"""
import os.path as osp
import pandas as pd

from sardes.api.timeseries import DataType
from sardes.database.accessors import DatabaseAccessorSardesLite
from sardes.utils.data_operations import format_reading_data

from selected_stations import selected_stations

dirname = osp.dirname(__file__)


database = "D:/Desktop/rsesq_prod_2022-02-22.db"
accessor = DatabaseAccessorSardesLite(database)
accessor.connect()

stations_data = accessor.get('observation_wells_data')
repere_data = accessor.get('repere_data')

daily_wl_all = pd.DataFrame()
yearly_mean_wl = pd.DataFrame()
for i, station_name in enumerate(selected_stations):
    print("{:02d} Procession readings data for station {}..."
          .format(i + 1, station_name))

    station_data = stations_data[
        stations_data['obs_well_id'] == station_name
        ].iloc[0]
    station_repere = (
        repere_data
        [repere_data['sampling_feature_uuid'] == station_data.name]
        .sort_values(by=['end_date'], ascending=[True]))

    readings_data = accessor.get_timeseries_for_obs_well(station_data.name)
    formatted_reading_data = format_reading_data(readings_data, station_repere)

    station_wl = (
        formatted_reading_data
        .set_index('datetime')
        [[DataType.WaterLevel]]
        )
    station_wl.columns = [station_name]

    if i == 0:
        daily_wl_all = station_wl.copy()
    else:
        daily_wl_all = daily_wl_all.merge(
            station_wl, left_index=True, right_index=True, how='outer')
accessor.close_connection()

# Save the daily water levels data in a CSV file.
daily_wl_all.to_csv(osp.join(dirname, "waterlevels_daily_2022-02-28.csv"))
