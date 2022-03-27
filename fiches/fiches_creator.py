# -*- coding: utf-8 -*-
"""
Outil permettant de générer.
"""
# Note importante:

# Il n'y a présentement pas de "wheel" pour Fiona et Rasterio sur
# PyPi. Si vous utiliser pip pour installer vos modules Python, il faudra
# installer, dans le bon ordre, les modules suivant en utilisant les
# "wheels" qui sont distribués sur le site de Christopher Gohlke.

# (1) https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal
# (2) https://www.lfd.uci.edu/~gohlke/pythonlibs/#fiona
# (3) https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely
# (4) https://www.lfd.uci.edu/~gohlke/pythonlibs/#rasterio

# Vous aurez également besoin d'une version de numpy plus grande ou
# égale à 1.20.0 pour que le module "contextily" marche correctement.
# Numpy peut être installé directement de PyPi. Il n'est as nécessaire
# d'utiliser une "wheel" de Christopher Gohlke.

# https://www.donneesquebec.ca/recherche/dataset/decoupages-administratifs

import os
import os.path as osp
import geopandas as gpd
from shapely.geometry import Point
import subprocess
from sardes.api.timeseries import DataType
from sardes.database.accessors import DatabaseAccessorSardesLite
from sardes.tools.hydrographs import HydrographCanvas
from sardes.tools.hydrostats import (
    SatisticalHydrographCanvas, compute_monthly_percentiles)
from sardes.utils.data_operations import format_reading_data
import shutil

import pandas as pd
import numpy as np
import itertools

# Month abbreviation in French :
# http://bdl.oqlf.gouv.qc.ca/bdl/gabarit_bdl.asp?id=3619
MONTHSABB = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun',
             'Jui', 'Aoû', 'Sep', 'Oct', 'Nov', 'Dec']


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


class StationDataSheet(object):
    def __init__(self, station_data, readings_data, repere_data,
                 munic_s, strati_data):
        """
        Parameters
        ----------
        station_data : TYPE
            DESCRIPTION.
        readings_data : TYPE
            DESCRIPTION.
        repere_data : TYPE
            DESCRIPTION.
        region : str
            Administrative region where the station is located.
        mrc : str
            Name of the MRC where the station is located.
        municipality : str
            Name of the municipality where the station is located.
        code : str
            Code of the municipality where the station is located.
        """
        super().__init__()
        self.station_data = station_data
        self.station_name = station_data['obs_well_id']
        self.station_uuid = station_data.name
        self.confinement = station_data['confinement'].lower()
        self.is_influenced = station_data['is_influenced'].lower()
        self.aquifer_type = station_data['aquifer_type'].lower()
        self.is_station_active = station_data['is_station_active']
        self.lat_ddeg = station_data['latitude']
        self.lon_ddeg = station_data['longitude']

        self.strati_data = strati_data

        self.repere_data = repere_data
        self.last_repere_data = repere_data.iloc[-1]
        self.ground_altitude = (
            self.last_repere_data['top_casing_alt'] -
            self.last_repere_data['casing_length'])
        self.is_alt_geodesic = self.last_repere_data['is_alt_geodesic']

        self.munic_s = munic_s
        self.region = munic_s['MUS_NM_REG']
        self.mrc = munic_s['MUS_NM_MRC']
        self.municipality = munic_s['MUS_NM_MUN']
        self.code = munic_s["MUS_CO_GEO"]

        self.readings_data = readings_data
        if not readings_data.empty:
            self.formatted_data = format_reading_data(
                readings_data, repere_data)
        else:
            self.formatted_data = pd.DataFrame([], dtype=object)

        self.wlevels = (
            self.formatted_data[[DataType.WaterLevel, 'datetime']]
            .set_index('datetime', drop=True)
            )

        self.type_aquifer_crepine = (
            'roc' if self.aquifer_type == 'roc' else 'granulaire')
        self.type_aquifer_nappe = self._calcul_type_aquifer_nappe()

    def _calcul_type_aquifer_nappe(self):
        top_masl = self.ground_altitude - self.strati_data['Depth']
        bottom_masl = self.ground_altitude - self.strati_data['Bottom']

        max_wl = max(min(
            self.wlevels.max()[0], top_masl.max()), bottom_masl.min())
        min_wl = max(min(
            self.wlevels.min()[0], top_masl.max()), bottom_masl.min())

        is_artesien = self.wlevels.max()[0] > top_masl.max()

        within_max_wl = self.strati_data[
            (max_wl <= top_masl) & (max_wl >= bottom_masl)
            ].iloc[0].squeeze()
        within_min_wl = self.strati_data[
            (min_wl <= top_masl) & (min_wl >= bottom_masl)
            ].iloc[0].squeeze()

        roc_graphic_classes = [
            'LIMESTONE', 'SANDSTONE', 'SHALE', 'BEDROCK', 'BASALT']
        aquitype_min_wl = (
            'roc' if within_min_wl['Graphic'] in
            roc_graphic_classes else 'granulaire')
        aquitype_max_wl = (
            'roc' if within_max_wl['Graphic'] in
            roc_graphic_classes else 'granulaire')

        aquitypes_wl = list(set([aquitype_min_wl, aquitype_max_wl]))
        if is_artesien:
            aquitypes_wl.insert(0, 'hors-sol')
        if len(aquitypes_wl) == 3:
            return "{}, {} et {}".format(*aquitypes_wl)
        elif len(aquitypes_wl) == 2:
            return "{} et {}".format(*aquitypes_wl)
        else:
            return "{}".format(*aquitypes_wl)

    def create_percentiles_table(self):
        percentiles, nyear = compute_monthly_percentiles(
            self.wlevels,
            q=[0, 10, 25, 50, 75, 90, 100],
            pool='min_max_median')

        percentiles.index = MONTHSABB
        percentiles = percentiles.apply(
            lambda series: series.apply(lambda value: f"{value:0.2f}"))
        percentiles = percentiles.reset_index()
        percentiles['nyear'] = nyear.astype(str)

        return [' & '.join(row.to_list()) + '\\\\' for
                index, row in percentiles.iterrows()]

    def create_statistical_hydrograph(self, filename):
        wlevels = (
            self.formatted_data[[DataType.WaterLevel, 'datetime']]
            .set_index('datetime', drop=True))
        last_month = self.wlevels.index[-1].month
        last_year = self.wlevels.index[-1].year
        if last_month != 12:
            last_year -= 1

        hydrostat = SatisticalHydrographCanvas(figsize=(7, 4.5))
        hydrostat.set_data(wlevels, year=last_year, month=last_month)

        try:
            hydrostat.figure.savefig(filename, dpi=300)
        except PermissionError as e:
            print(e)

    def create_hydrograph(self, filename):
        hydrograph = HydrographCanvas(
            self.formatted_data,
            self.station_data,
            self.ground_altitude,
            self.is_alt_geodesic,
            fontname='Arial')

        try:
            hydrograph.figure.savefig(filename, dpi=300)
        except PermissionError as e:
            print(e)


class DataSheetCreator(object):

    def __init__(self, workdir):
        super().__init__()
        self.workdir = workdir

        self._dirphoto = osp.join(
            self.workdir, 'fiches', "img_photos_puits")
        if not osp.exists(self._dirphoto):
            os.makedirs(self._dirphoto)
        self._dircontext = osp.join(
            self.workdir, 'fiches', "img_matrices_contexte")
        if not osp.exists(self._dircontext):
            os.makedirs(self._dircontext)
        self._dirhstat = osp.join(
            self.workdir, 'fiches', "img_hydrogrammes_statistiques")
        if not osp.exists(self._dirhstat):
            os.makedirs(self._dirhstat)
        self._dirhgraph = osp.join(
            self.workdir, 'fiches', "img_hydrogrammes")
        if not osp.exists(self._dirhgraph):
            os.makedirs(self._dirhgraph)
        self._dirschema = osp.join(
            self.workdir, 'fiches', "img_schema_puits")
        if not osp.exists(self._dirschema):
            os.makedirs(self._dirschema)
        self._dirbrf = osp.join(
            self.workdir, 'fiches', "img_fonction_reponse_baro")
        if not osp.exists(self._dirbrf):
            os.makedirs(self._dirbrf)
        self._dirlocal = osp.join(
            self.workdir, 'fiches', "img_cartes_localisation_puits")
        if not osp.exists(self._dirlocal):
            os.makedirs(self._dirlocal)

        self.munic_s = gpd.read_file(
            osp.join(self.workdir, "SDA_ 2018-05-25.gdb.zip"),
            driver='FileGDB',
            layer='munic_s')

        self.dbaccessor = DatabaseAccessorSardesLite(
            osp.join(self.workdir, "rsesq_prod_28-06-2021.db"))
        self.stations = self.dbaccessor.get_observation_wells_data()
        self.repere_data = self.dbaccessor.get_repere_data()

    def datasheet(self, station_name):
        station_data = self.stations[
            self.stations['obs_well_id'] == station_name
            ].iloc[0]

        repere_data = (
            self.repere_data
            [self.repere_data['sampling_feature_uuid'] == station_data.name]
            .copy())
        if not repere_data.empty:
            repere_data = (
                repere_data
                .sort_values(by=['end_date'], ascending=[True]))

        datasheet = StationDataSheet(
            station_data,
            self.get_readings_data_for_station(station_name),
            repere_data,
            self.get_munic_s_for_station(station_name),
            self.get_strati_for_station(station_name)
            )
        return datasheet

    def get_station_uuid_from_name(self, station_name):
        return (
            self.stations
            [self.stations['obs_well_id'] == station_name]
            .iloc[0].name)

    def get_munic_s_for_station(self, station_name):
        station_data = self.stations[
            self.stations['obs_well_id'] == station_name
            ].iloc[0]
        lat_ddeg = station_data['latitude']
        lon_ddeg = station_data['longitude']
        return self.munic_s[
            self.munic_s['geometry'].contains(Point(lon_ddeg, lat_ddeg))
            ].iloc[0]

    def get_readings_data_for_station(self, station_name):
        station_uuid = self.stations[
            self.stations['obs_well_id'] == station_name
            ].iloc[0].name
        return self.dbaccessor.get_timeseries_for_obs_well(station_uuid)

    def get_strati_for_station(self, station_name):
        strati_data = pd.read_excel(
            osp.join(self.workdir, "Logs_complet_06-2021.xlsx"),
            sheet_name='STRATIGRAPHIE')
        station_strati = (
            strati_data[strati_data['PointID'] == station_name].copy())
        return station_strati

    def create_schema_for_station(self, station_name):
        print("Creating the well construction schema...")
        station_uuid = self.get_station_uuid_from_name(station_name)
        data, name = self.dbaccessor.get_attachment(
            station_uuid, attachment_type=1)

        if data is not None:
            filename = osp.join(self._dirschema, name).replace('\\', '/')
            if not osp.exists(filename):
                with open(filename, 'wb') as f:
                    f.write(data)

            return filename
        else:
            return ''

    def create_texfile_for_station(self, station_name):
        datasheet = self.datasheet(station_name)

        geosysprec = (
            'géodésique' if datasheet.is_alt_geodesic else 'approximative')

        confinement = datasheet.confinement
        is_influenced = datasheet.is_influenced
        if confinement == 'nd':
            if is_influenced == 'nd':
                typenappe = 'nd'
            elif is_influenced == 'oui':
                typenappe = "influencée"
            elif is_influenced is False:
                typenappe = "non influencée"
        else:
            typenappe = confinement
            if is_influenced == 'oui':
                typenappe += " et influencée"
            if is_influenced == 'non':
                typenappe += " et non influencée"

        if datasheet.readings_data.empty:
            operationperiod = "aucune"
            gapsdata = 'non applicable'
        else:
            operationperiod = (
                datasheet.readings_data['datetime']
                .min().strftime('%Y-%m-%d'))
            if datasheet.is_station_active:
                operationperiod += " à aujourd'hui"
            else:
                operationperiod += ' au {}'.format(
                    datasheet.readings_data['datetime']
                    .max().strftime('%Y-%m-%d'))

            avail_years = (
                datasheet.readings_data['datetime'].dt.year.drop_duplicates())
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
            r"\newcommand{\stationid}" + "{{{}}}".format(
                datasheet.station_name),
            r"\newcommand{\munname}" + "{{{}}}".format(
                datasheet.municipality),
            r"\newcommand{\muncode}" + "{{{}}}".format(
                datasheet.code),
            r"\newcommand{\mrc}" + "{{{}}}".format(
                datasheet.mrc),
            r"\newcommand{\region}" + "{{{}}}".format(
                datasheet.region),
            r"\newcommand{\typenappe}" + "{{{}}}".format(
                typenappe),
            r"\newcommand{\typeaquicrepine}" + "{{{}}}".format(
                datasheet.type_aquifer_crepine),
            r"\newcommand{\typeaquinappe}" + "{{{}}}".format(
                datasheet.type_aquifer_nappe),
            r"\newcommand{\operationperiod}" + "{{{}}}".format(
                operationperiod),
            r"\newcommand{\gapsdata}" + "{{{}}}".format(
                gapsdata),
            r"\newcommand{\notes}" + "{{{}}}".format(
                ''),
            r"\newcommand{\latitude}" + "{{{:0.6f}}}".format(
                datasheet.lat_ddeg),
            r"\newcommand{\longitude}" + "{{{:0.6f}}}".format(
                datasheet.lon_ddeg),
            r"\newcommand{\altitude}" + "{{{:0.1f} m NMM}}".format(
                datasheet.ground_altitude),
            r"\newcommand{\geosysprec}" + "{{{}}}".format(
                geosysprec),
            ""
            ]

        # =====================================================================
        # Filepaths
        # =====================================================================
        pathlocal = osp.join(
            self._dirlocal,
            'Cartes_Puits_RSESQ_{}.pdf'.format(station_name)
            ).replace('\\', '/')
        if not osp.exists(pathlocal):
            pathlocal = osp.join(
                self.workdir, 'fiches', "contexte_puits_non_disponible.pdf"
                ).replace('\\', '/')

        pathhgraph = ''
        if not datasheet.readings_data.empty:
            print("Creating the well hydrograph...")
            pathhgraph = osp.join(
                self._dirhgraph,
                'hydrogramme_{}.pdf'.format(station_name)
                ).replace('\\', '/')
            datasheet.create_hydrograph(pathhgraph)

        pathhstat = ''
        if not datasheet.readings_data.empty and datasheet.is_station_active:
            print("Creating the statistitical hydrograph...")
            pathhstat = osp.join(
                self._dirhstat,
                'hydrostat_{}.pdf'.format(station_name)
                ).replace('\\', '/')
            datasheet.create_statistical_hydrograph(pathhstat)

        pathcontext = osp.join(
            self._dircontext,
            'Matrices_Puits_RSESQ_{}.pdf'.format(station_name)
            ).replace('\\', '/')

        pathphoto = osp.join(
            self._dirphoto,
            "photo_puit_{}.jpg".format(station_name)
            ).replace('\\', '/')
        if not osp.exists(pathphoto):
            pathphoto = osp.join(
                self.workdir, 'fiches', "photo_non_disponible.pdf"
                ).replace('\\', '/')

        pathbrf = osp.join(
            self._dirbrf,
            "fonction_reponse_baro_{}.pdf".format(station_name)
            ).replace('\\', '/')

        pathschem = self.create_schema_for_station(station_name)

        content += [
            r"% " + r"=" * 77,
            r"% Filepaths",
            r"% " + r"=" * 77,
            r"\newcommand{\pathlocal}" + "{{{}}}".format(pathlocal),
            r"\newcommand{\pathphoto}" + "{{{}}}".format(pathphoto),
            r"\newcommand{\pathschem}" + "{{{}}}".format(pathschem),
            r"\newcommand{\pathhgraph}" + "{{{}}}".format(pathhgraph),
            r"\newcommand{\pathhstat}" + "{{{}}}".format(pathhstat),
            r"\newcommand{\pathcontext}" + "{{{}}}".format(pathcontext),
            r"\newcommand{\pathbrf}" + "{{{}}}".format(pathbrf),
            ""
            ]

        # =====================================================================
        # Percentiles
        # =====================================================================
        content += [
            r"% " + r"=" * 77,
            r"% Percentiles",
            r"% " + r"=" * 77,
            r"\newcommand{\percentiles}{%"]
        content += datasheet.create_percentiles_table()
        content += [
            r"}",
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
            osp.exists(pathhstat) else
            r"\newcommand{\inputpagefour}{}")
        content.append(
            r"\newcommand{\inputpagefive}{\input{fiches-rsesq-page5}}" if
            osp.exists(pathcontext) else
            r"\newcommand{\inputpagefive}{}")
        content.append(
            r"\newcommand{\inputpagesix}{\input{fiches-rsesq-page6}}" if
            osp.exists(pathbrf) else
            r"\newcommand{\inputpagesix}{}")
        content.append("")

        filename = osp.join(self.workdir, 'fiches', 'fiches-rsesq-station.tex')
        with open(filename, 'w', encoding='utf8') as textfile:
            textfile.write('\n'.join(content))

    def build_datasheet_for_station(self, station_name):
        print("Creating fiche for station {}...".format(station_name))

        self.create_texfile_for_station(station_name)

        # We need to run this two times in order for the layout to be
        # properly generated.
        subprocess.run(
            'xelatex.exe -synctex=1 -interaction=nonstopmode "fiches-rsesq.tex"',
            cwd=osp.join(self.workdir, 'fiches')
            )
        subprocess.run(
            'xelatex.exe -synctex=1 -interaction=nonstopmode "fiches-rsesq.tex"',
            cwd=osp.join(self.workdir, 'fiches')
            )

        src = osp.join(
            self.workdir, 'fiches', "fiches-rsesq.pdf")
        dst = osp.join(
            self.workdir, 'fiches', 'pdf_fiches_stations',
            "fiche_{}.pdf".format(station_name))
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))
        shutil.copyfile(src, dst)

        print("Fiche created successfully for station {}."
              .format(station_name))


if __name__ == '__main__':
    dscreator = DataSheetCreator(workdir=osp.dirname(__file__))

    station_names = []
    for index, data in dscreator.stations.iterrows():
        station_name = data['obs_well_id']
        context_filepath = osp.join(
            dscreator._dirlocal,
            'Cartes_Puits_RSESQ_{}.pdf'.format(station_name))
        if not osp.exists(context_filepath):
            continue
        station_names.append(station_name)
        dscreator.build_datasheet_for_station(station_name)
