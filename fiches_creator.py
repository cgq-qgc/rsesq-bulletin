# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 14:47:16 2021
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


# https://www.donneesquebec.ca/recherche/dataset/decoupages-administratifs
import os
import os.path as osp
import fiona
import geopandas as gpd
from shapely.geometry import Point
import subprocess
from sardes.api.timeseries import DataType
from sardes.database.accessors import DatabaseAccessorSardesLite
from sardes.tools.hydrographs import HydrographCanvas
from sardes.tools.hydrostats import SatisticalHydrographCanvas
from sardes.utils.data_operations import format_reading_data

import pandas as pd
import numpy as np
import itertools


def intervals_extract(iterable):
    """
    Given a list of sequential numbers, convert the given list into
    a list of intervals.

    Code adapted from:
    https://www.geeksforgeeks.org/python-make-a-list-of-intervals-with-sequential-numbers/

    Code taken from the Sardes projet.
    https://github.com/cgq-qgc/sardes/blob/master/sardes/utils/data_operations.py
    """
    sequence = sorted(set(iterable))
    for key, group in itertools.groupby(enumerate(sequence),
                                        lambda v: v[1] - v[0]):
        group = list(group)
        yield [group[0][1], group[-1][1]]


class RSESQFichesCreator(object):
    def __init__(self):
        super().__init__()
        self.workdir = "C:/Users/User/rsesq-bulletin"

        self.munic_s = gpd.read_file(
            osp.join(self.workdir, "SDA_ 2018-05-25.gdb.zip"),
            driver='FileGDB',
            layer='munic_s')

        self.dbaccessor = DatabaseAccessorSardesLite(
            osp.join(self.workdir, "rsesq_prod_28-06-2021.db"))
        self.stations = self.dbaccessor.get_observation_wells_data()
        self.repere_data = self.dbaccessor.get_repere_data()

    def get_region_mrc_mun_at(self, lat_ddeg, lon_ddeg):
        loc_point = Point(lon_ddeg, lat_ddeg)
        contains = self.munic_s[
            self.munic_s['geometry'].contains(loc_point)
            ].iloc[0]
        region = contains['MUS_NM_REG']
        mrc = contains['MUS_NM_MRC']
        municipality = contains['MUS_NM_MUN']
        code = contains["MUS_CO_GEO"]

        return region, mrc, municipality, code

    def create_schema_for(self, station_uuid):
        print("Creating the well construction schema...")
        data, name = self.dbaccessor.get_attachment(
            station_uuid, attachment_type=1)

        if data is not None:
            dirname = osp.join(self.workdir, 'fiches', 'schema_puits_rsesq')
            if not osp.exists(dirname):
                os.makedirs(dirname)

            filename = osp.join(dirname, name).replace('\\', '/')
            if not osp.exists(filename):
                with open(filename, 'wb') as f:
                    f.write(data)

            return filename
        else:
            return ''

    def create_hydrostat_for(self, station_name, formatted_data):
        print("Creating the statistitical hydrograph...")
        dirname = osp.join(self.workdir, 'fiches', 'hydrogrammes_statistiques')
        if not osp.exists(dirname):
            os.makedirs(dirname)

        wlevels = (
            formatted_data[[DataType.WaterLevel, 'datetime']]
            .set_index('datetime', drop=True))
        last_month = wlevels.index[-1].month
        last_year = wlevels.index[-1].year
        if last_month != 12:
            last_year -= 1

        hydrostat = SatisticalHydrographCanvas()
        hydrostat.set_data(wlevels, year=last_year, month=last_month)

        filename = osp.join(
            dirname, 'hydrostat_{}.pdf'.format(station_name)
            ).replace('\\', '/')
        try:
            hydrostat.figure.savefig(filename, dpi=300)
        except PermissionError as e:
            print(e)
        return filename

    def create_hydrograph_for(self, formatted_data, station_data,
                              ground_altitude, is_alt_geodesic):
        print("Creating the well hydrograph...")
        dirname = osp.join(self.workdir, 'fiches', 'hydrogrammes')
        if not osp.exists(dirname):
            os.makedirs(dirname)

        hydrograph = HydrographCanvas(
            formatted_data,
            station_data,
            ground_altitude,
            is_alt_geodesic,
            fontname='Arial')

        filename = osp.join(
            dirname, 'hydrogramme_{}.pdf'.format(station_data['obs_well_id'])
            ).replace('\\', '/')
        try:
            hydrograph.figure.savefig(filename, dpi=300)
        except PermissionError as e:
            print(e)
        return filename

    def create_tex_rsesq_data(self, station_name):
        station_data = self.stations[
            self.stations['obs_well_id'] == station_name].iloc[0]
        station_uuid = station_data.name

        lat_ddeg = station_data['latitude']
        lon_ddeg = station_data['longitude']
        region, mrc, municipality, code = self.get_region_mrc_mun_at(
            lat_ddeg, lon_ddeg)

        station_repere_data = (
            self.repere_data
            [self.repere_data['sampling_feature_uuid'] == station_uuid]
            .copy())
        if not station_repere_data.empty:
            station_repere_data = (
                station_repere_data
                .sort_values(by=['end_date'], ascending=[True]))
        else:
            station_repere_data = pd.Series([], dtype=object)
        last_repere_data = station_repere_data.iloc[-1]
        ground_altitude = (
            last_repere_data['top_casing_alt'] -
            last_repere_data['casing_length'])
        is_alt_geodesic = last_repere_data['is_alt_geodesic']
        geosysprec = 'géodésique' if is_alt_geodesic else 'approximative'

        confinement = station_data['confinement']
        is_influenced = station_data['is_influenced'].lower()
        if confinement == 'nd':
            if is_influenced == 'nd':
                typenappe = 'nd'
            elif is_influenced == 'oui':
                typenappe = "influencée"
            elif is_influenced is False:
                typenappe = "non influencée"
        else:
            typenappe = confinement.lower()
            if is_influenced == 'oui':
                typenappe += " et influencée"
            if is_influenced == 'non':
                typenappe += " et non influencée"

        aquifer_type = station_data['aquifer_type'].lower()
        type_aquifer_crepine = 'roc' if aquifer_type == 'roc' else 'granulaire'

        readings = self.dbaccessor.get_timeseries_for_obs_well(station_uuid)
        is_station_active = station_data['is_station_active']
        if readings.empty:
            operationperiod = "aucune"
        else:
            operationperiod = readings['datetime'].min().strftime('%Y-%m-%d')
            if is_station_active:
                operationperiod += " à aujourd'hui"
            else:
                operationperiod += ' au {}'.format(
                    readings['datetime'].max().strftime('%Y-%m-%d'))

        if readings.empty:
            gapsdata = 'non applicable'
        else:
            avail_years = readings['datetime'].dt.year.drop_duplicates()
            missing_years = pd.Series(np.arange(
                avail_years.min(), avail_years.max() + 1))
            missing_years = missing_years[~missing_years.isin(avail_years)]
            if missing_years.empty:
                gapsdata = 'aucune'
            else:
                gapsdata = ', '.join([
                    '{}, {}'.format(*interval) if
                    np.diff(interval)[0] == 1 else
                    '{} à {}'.format(*interval) for
                    interval in intervals_extract(missing_years.values)
                    ])

        # =====================================================================
        # Station info
        # =====================================================================
        content = [
            r"% " + r"=" * 77,
            r"% Station Info",
            r"% " + r"=" * 77,
            r"\newcommand{\stationid}" + "{{{}}}".format(station_name),
            r"\newcommand{\munname}" + "{{{}}}".format(municipality),
            r"\newcommand{\muncode}" + "{{{}}}".format(code),
            r"\newcommand{\mrc}" + "{{{}}}".format(mrc),
            r"\newcommand{\region}" + "{{{}}}".format(region),
            r"\newcommand{\typenappe}" + "{{{}}}".format(typenappe),
            r"\newcommand{\typeaquicrepine}" + "{{{}}}".format(type_aquifer_crepine),
            r"\newcommand{\typeaquinappe}{granulaire}",
            r"\newcommand{\operationperiod}" + "{{{}}}".format(operationperiod),
            r"\newcommand{\gapsdata}" + "{{{}}}".format(gapsdata),
            r"\newcommand{\notes}{}",
            r"\newcommand{\latitude}" + "{{{:0.6f}}}".format(lat_ddeg),
            r"\newcommand{\longitude}" + "{{{:0.6f}}}".format(lon_ddeg),
            r"\newcommand{\altitude}" + "{{{:0.1f} m NMM}}".format(ground_altitude),
            r"\newcommand{\geosysprec}" + "{{{}}}".format(geosysprec),
            ""
            ]

        pathlocal = "./cartes_puits_rsesq/Cartes_Puits_RSESQ_{}".format(
            station_name)

        # =====================================================================
        # Filepaths
        # =====================================================================
        pathschem = self.create_schema_for(station_uuid)

        if readings.empty:
            pathhgraph = ''
            pathhstat = ''
        else:
            formatted_data = format_reading_data(
                readings, station_repere_data)
            pathhgraph = self.create_hydrograph_for(
                formatted_data, station_data, ground_altitude, is_alt_geodesic)

            if is_station_active:
                pathhstat = self.create_hydrostat_for(
                    station_name, formatted_data)
            else:
                pathhgraph = ''

        pathcontext = osp.join(
            self.workdir, 'fiches', 'matrices_contexte',
            'Matrices_Puits_RSESQ_{}.pdf'.format(station_name)
            ).replace('\\', '/')

        content += [
            r"% " + r"=" * 77,
            r"% Filepaths",
            r"% " + r"=" * 77,
            r"\newcommand{\pathlocal}" + "{{{}}}".format(pathlocal),
            r"\newcommand{\pathphoto}{./photos_puits_rsesq/photo_puit_P19_scaled}",
            r"\newcommand{\pathschem}" + "{{{}}}".format(pathschem),
            r"\newcommand{\pathhgraph}" + "{{{}}}".format(pathhgraph),
            r"\newcommand{\pathhstat}" + "{{{}}}".format(pathhstat),
            r"\newcommand{\pathcontext}" + "{{{}}}".format(pathcontext),
            r"\newcommand{\pathbrf}{./brf_puits_rsesq/brf_puits_03040001}",
            ""
            ]

        # =====================================================================
        # Pages
        # =====================================================================
        content += [
            r"% " + r"=" * 77,
            r"% Pages",
            r"% " + r"=" * 77,
            r"\newcommand{\inputpageone}{\input{fiches-rsesq-page1}}"]
        content.append(
            r"\newcommand{\inputpagetwo}{\input{fiches-rsesq-page2}}" if
            osp.exists(pathschem) else
            r"\newcommand{\inputpagetwo}{}")
        content.append(
            r"\newcommand{\inputpagethree}{\input{fiches-rsesq-page3}}" if
            osp.exists(pathhgraph) else
            r"\newcommand{\inputpagethree}{}")
        content.append(
            r"\newcommand{\inputpagefour}{\input{fiches-rsesq-page4}}" if
            osp.exists(pathhgraph) else
            r"\newcommand{\inputpagefour}{}")
        content.append(
            r"\newcommand{\inputpagefive}{\input{fiches-rsesq-page5}}" if
            osp.exists(pathcontext) else
            r"\newcommand{\inputpagefive}{}")
        content.append(
            r"\newcommand{\inputpagesix}{}")
        content.append("")

        filename = osp.join(self.workdir, 'fiches', 'fiches-rsesq-station.tex')
        with open(filename, 'w', encoding='utf8') as textfile:
            textfile.write('\n'.join(content))

    def create_datasheet_for(self, station_id):
        print("Creating data sheet for station {}...".format(station_id))

        self.create_tex_rsesq_data(station_id)
        subprocess.run(
            'xelatex.exe -synctex=1 -interaction=nonstopmode "fiches-rsesq.tex"',
            cwd=osp.join(self.workdir, 'fiches')
            )
        subprocess.run(
            'xelatex.exe -synctex=1 -interaction=nonstopmode "fiches-rsesq.tex"',
            cwd=osp.join(self.workdir, 'fiches')
            )

        print("Data sheet created successfully for station {}."
              .format(station_id))


if __name__ == '__main__':
    datasheet_creator = RSESQDataSheetCreator()
    readings = datasheet_creator.create_datasheet_for('02507001')
