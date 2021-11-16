# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 14:29:22 2021
@author: User
"""

# ---- Standard Library imports
from __future__ import annotations
import functools
import time
import os
import os.path as osp
from pathlib import Path
import datetime

# ---- Third Party imports
import numpy as np
import pandas as pd
import netCDF4


def timethis(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print('runtime: {} -> {:0.5} sec'.format(func.__name__, end-start))
        return result
    return wrapper


def calc_dist_from_coord(lat1, lon1, lat2, lon2):
    """
    Compute the  horizontal distance in km between a location given in
    decimal degrees and a set of locations also given in decimal degrees.

    https://en.wikipedia.org/wiki/Haversine_formula
    https://www.nhc.noaa.gov/gccalc.shtml
    """
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)

    r = 6373  # Earth radius in km.

    # Note that the units used for r determine the return value units.

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # Note that arctan2(sqrt(a), sqrt(1-a)) is the same as arcsin(sqrt(a)) in
    # this case.

    return r * c


class ConnectTable(pd.DataFrame):
    # https://pandas.pydata.org/pandas-docs/stable/development/extending.html#extending-subclassing-pandas

    @property
    def _constructor(self):
        return ConnectTable

    def save_to_csv(self, filepath):
        """
        Parameters
        ----------
        filepath : str | Path
            The path of the csv file where to save this connect table.
        """
        filepath = osp.abspath(filepath)
        if not osp.exists(osp.dirname(filepath)):
            os.makedirs(osp.dirname(filepath))

        self.to_csv(filepath, sep=',', line_terminator='\n', encoding='utf-8')


class ClimateDataFrame(pd.DataFrame):
    # https://pandas.pydata.org/pandas-docs/stable/development/extending.html#extending-subclassing-pandas
    _metadata = ["varname", "source"]

    @property
    def _constructor(self):
        return ClimateDataFrame

    def save_to_csv(self, filepath: str | Path):
        """
        Parameters
        ----------
        filepath : str | Path
            The path of the csv file where to save this climate dataframe.
        """
        filepath = osp.abspath(filepath)
        if not osp.exists(osp.dirname(filepath)):
            os.makedirs(osp.dirname(filepath))

        print("Saving '{}' data to '{}'...".format(self.varname, filepath))

        now_str = datetime.datetime.now().strftime('%Y-%m-%d')
        index_strlist = self.columns.get_level_values(0).astype(str).tolist()
        lat_strlist = self.columns.get_level_values(1).astype(str).tolist()
        lon_strlist = self.columns.get_level_values(2).astype(str).tolist()

        # Define the content of the file header.
        fheader = [
            [self.varname],
            [''],
            ['Created on:', now_str],
            ['Source:', self.source],
            [''],
            ['Grid index:'] + index_strlist,
            ['Latitude (dd):'] + lat_strlist,
            ['Longitude (dd):'] + lon_strlist,
            [''],
            ['']]
        fheader = '\n'.join([','.join(row) for row in fheader])

        # Define the content of the file data.
        fdata = self.to_csv(
            header=None, sep=',', line_terminator='\n', encoding='utf-8')

        # Save the content of the file buffer to disk.
        with open(filepath, mode='w', encoding='utf-8') as f:
            f.write(fheader + fdata)


class InfoClimatGridReader(object):
    """
    A class to read and format daily climate data from the interpolated grid
    produced by Info-climat at MDDELCC.

    Parameters
    ----------
    dirpath_netcdf : str | Path
        The path of the directory where the grid netcdf files are stored.
    """
    VARNAMES = ['PREC', 'NEIGE', 'NESOL', 'PLUIE', 'PREC',
                'TMAX', 'TMIN', 'TMOY']

    def __init__(self, dirpath_netcdf: str | Path):
        super().__init__()
        self.dirpath_netcdf = dirpath_netcdf
        self.grid_latdd = np.array([])
        self.grid_londd = np.array([])
        self._setup_grid()

    @timethis
    def _setup_grid(self):
        """
        Fetch the latitude and longitude coordinates of the cells of the grid.
        """
        # We assume that the grid is the same for all netcdf files, so we
        # simply load the grid from the first file of the grid contained
        # in 'dirpath_netcdf'.
        for file in os.listdir(self.dirpath_netcdf):
            if file.endswith('.nc'):
                break
        else:
            return

        ncfilepath = osp.join(self.dirpath_netcdf, file)
        with netCDF4.Dataset(ncfilepath, 'r+') as ncdset:
            self.grid_latdd = np.array(ncdset['lat']).flatten()
            self.grid_londd = np.array(ncdset['lon']).flatten()

    def _get_idx_from_latlon(self, latitudes: list[float],
                             longitudes: list[float]) -> list[int]:
        """
        Return the logical indexes of the cells of the flatened grid
        containing each pair of latitude and longitude coordinates.

        Parameters
        ----------
        latitudes : array-like, Iterable, or scalar value
            Contains the latitude values, in decimal degrees, of the
            coordinates for which we want to find the logical indexes of
            the corresponding grid cells.
        longitudes : array-like, Iterable, or scalar value
            Contains the longitude values, in decimal degrees, of the
            coordinates for which we want to find the logical indexes of
            the corresponding grid cells.

        Returns
        -------
        list[int]
            A list containing the flattened grid indexes of the cells
            containint the location at which climate data is to be extracted.
        """
        return [
            np.argmin(calc_dist_from_coord(
                lat, lon, self.grid_latdd, self.grid_londd)) for
            lat, lon in zip(latitudes, longitudes)
            ]

    def create_connect_table(self, lat_dd: list[float], lon_dd: list[float],
                             loc_id: list = None) -> ConnectTable:
        """
        Create a connection table that contains the relation between a set
        of location coordinates and the cell of the grids.

        Parameters
        ----------
        latitudes : list[float]
            Contains the latitude coordinates, in decimal degrees, of the
            locations for which you want to extract climate data from the grid.
        longitudes : list[float]
            Contains the longitude coordinates, in decimal degrees, of the
            locations for which you want to extract climate data from the grid
        loc_id : list
            An optional list of custom ID to identify each location for which
            you want to extract climate data from the grid. If not 'loc_id'
            are specified, a list of incremental integer values will be
            generated and used by default.

        Returns
        -------
        ConnectTable
            A pandas dataframe containing the following columns:

            * loc_id: The identifiers of the locations for which climate data
              is to be extracted from the grid.
            * loc_lat_dd: The latitude coordinates of the locations for which
              climate data is to be extracted from the grid.
            * loc_lon_dd : The latitude coordinates of the locations for which
              climate data is to be extracted from the grid.
            * grid_idx: The cell indexes of the flattened grid containing
              the locations for which climate data is to be extracted.
            * grid_lat_dd: The latitude coordinates of the cells containing
              the location for which climate data is to be extracted.
            * grid_lon_dd: The longitude coordinates of the cells containing
              the locations for which climate data is to be extracted.
            * dist_km: The distance in km between the locations for which
              climate data is to be extracted and the location of the
              corresponding cells of the grid.
        """
        connect_table = ConnectTable([])

        if loc_id is None:
            connect_table['loc_id'] = list(range(len(connect_table)))
        else:
            connect_table['loc_id'] = loc_id
        connect_table['loc_lat_dd'] = lat_dd
        connect_table['loc_lon_dd'] = lon_dd

        indexes = self._get_idx_from_latlon(lat_dd, lon_dd)

        connect_table['grid_idx'] = indexes
        connect_table['grid_lat_dd'] = self.grid_latdd[indexes]
        connect_table['grid_lon_dd'] = self.grid_londd[indexes]
        connect_table['dist_km'] = calc_dist_from_coord(
            connect_table['loc_lat_dd'].values,
            connect_table['loc_lon_dd'].values,
            connect_table['grid_lat_dd'].values,
            connect_table['grid_lon_dd'].values)

        return connect_table

    @timethis
    def get_climate_data(self, varname: str, connect_table: pd.DataFrame,
                         first_year: int, last_year: int
                         ) -> ClimateDataFrame:
        """
        Extract from the grid the daily climatic data corresponding to
        'varname' for a list of years and latitude and longitude coordinates.

        Parameters
        ----------
        varname : {'PREC', 'NEIGE', 'NESOL', 'PLUIE', 'PREC',
                   'TMAX', 'TMIN', 'TMOY'}
            The climatic variable for which values are to be extracted from
            the grid.
        connect_table: ConnectTable
            The connection table containing the information on the
            locations for which climate data is to be extracted and their
            relation with the nodes of the grid.
        first_year: int
            The first year of the period for which climate data is to be
            extracted from the grid.
        last_year: int
            The last year of the period for which climate data is to be
            extracted from the grid.

        Returns
        -------
        climate_data: ClimateDataFrame
            A pandas dataframe containing the daily climatic data that was
            extracted from the grid.
        """
        if varname not in self.VARNAMES:
            raise ValueError("Valid values for 'varnames are: {}.".format(
                ', '.join(self.VARNAMES)))
        years = np.arange(first_year, last_year + 1)

        grid_extract_info = (
            connect_table.copy()
            .drop_duplicates(subset='grid_idx')
            .sort_values('grid_idx'))

        grid_lat_dd = grid_extract_info['grid_lat_dd'].values
        grid_lon_dd = grid_extract_info['grid_lon_dd'].values
        grid_idx = grid_extract_info['grid_idx'].values.tolist()

        data_stack = []
        index_stack = pd.Index([], dtype='datetime64[ns]')
        source = ''
        for year in years:
            print('Fetching daily {} data for year {}...'.format(
                varname, year))

            ncfilename = '{}_{}.nc'.format(varname, year)
            ncfilepath = osp.join(self.dirpath_netcdf, ncfilename)
            if not osp.exists(ncfilepath):
                print("'{}' does not exist: skipping.".format(ncfilename))
                continue

            with netCDF4.Dataset(ncfilepath, 'r+') as ncdset:
                array = np.array(ncdset[varname])
                source = ncdset.title

            data_stack.append(
                array.reshape(array.shape[0], -1)[:, grid_idx])
            index_stack = index_stack.append(
                pd.date_range(start=datetime.datetime(year, 1, 1),
                              end=datetime.datetime(year, 12, 31)))

        climate_data = ClimateDataFrame(
            data=np.vstack(data_stack),
            index=index_stack,
            columns=pd.MultiIndex.from_tuples(
                zip(grid_idx, grid_lat_dd, grid_lon_dd),
                names=['Grid index', 'Latitude(dd)', 'Longitude (dd)'])
            )
        climate_data = climate_data.sort_index()
        climate_data.varname = varname
        climate_data.source = source

        # Apply a mask to set missing values to NaN.
        # It looks like missing values were attributed a value of
        # -1.7976931348623157e+308 in the netcdf files.
        climate_data = climate_data[climate_data > -999].copy()

        return climate_data


if __name__ == '__main__':
    infoclim_reader = InfoClimatGridReader("D:/Data/GrilleInfoClimat2021")

    loc_id = ['loc1', 'loc2', 'loc3']
    lat_dd = [45.42571, 49.1564, 45.43753]
    lon_dd = [-73.0764, -68.24755, -73.0813]

    connect_table = infoclim_reader.create_connect_table(
        lat_dd, lon_dd, loc_id)
    climate_data = infoclim_reader.get_climate_data(
        'TMAX', connect_table, first_year=2019, last_year=2021)

    print(climate_data)
    print()
    print(connect_table)

    connect_table.save_to_csv('connect_table.csv')
    climate_data.save_to_csv('tmax_2019-2021.csv')
