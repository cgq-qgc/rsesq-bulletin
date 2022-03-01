# -*- coding: utf-8 -*-
"""
A script to calculate SPI and SPLI at selected piezometric stations
of the RSESQ.

Contrairement à la version 1, ce script utilise une loi normale directement
pour le calcul des SPI et SPLI, au lieu d'utiliser un loi Gamma pour les
précipitations et d'un estimateur à noyau pour les niveaux d'eau.
"""
import os
import os.path as osp
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm

os.chdir(osp.dirname(__file__))
from selected_stations import selected_stations
from plot_results import (
    plot_spli_overview, plot_spli_vs_classes,
    plot_pdf_niveau, plot_pdf_precip, plot_cross_corr)
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.rcParams['axes.unicode_minus'] = False
plt.close('all')

PRECIP_DAILY_ALL = pd.read_csv(
    'precip_daily_2022-02-28.csv',
    index_col=[0], parse_dates=[0])

WLVL_DAILY_ALL = pd.read_csv(
    'waterlevels_daily_2022-02-28.csv',
    index_col=[0], parse_dates=[0])


def calc_std_indexes(staname, precip_win: int = 12, wlvl_win: int = 3):
    # =========================================================================
    # Préparation des séries mensuelles
    # =========================================================================
    print("Calculating standard indexes for station {}...".format(staname))

    # Construction d'une série de précipitations mensuelles.
    sta_precip_daily = (
        PRECIP_DAILY_ALL[staname].to_frame(name='precip').dropna())

    precip = sta_precip_daily.copy()
    precip['year'] = precip.index.year
    precip['month'] = precip.index.month
    precip = precip.groupby(['year', 'month']).sum()
    precip = precip.rolling(precip_win).sum()
    precip = precip.unstack(level=1)
    precip = precip[precip.index >= 1981].copy()

    # Construction d'une série de niveaux d'eau moyens mensuels.
    sta_wl_daily = (
        WLVL_DAILY_ALL[staname].to_frame(name='water_level').dropna())

    wlvl = sta_wl_daily.copy()
    wlvl['year'] = wlvl.index.year
    wlvl['month'] = wlvl.index.month
    wlvl = wlvl.groupby(['year', 'month']).mean()
    wlvl = wlvl.rolling(3).mean()
    wlvl = wlvl.unstack(level=1)

    # Nous calculons les SPI et SPLI seulement pour les données récentes.
    precip = precip[precip.index >= 1981].copy()
    wlvl = wlvl[wlvl.index >= 1981].copy()

    # =========================================================================
    # Calcul des SPI et des SPLI mensuels
    # =========================================================================
    std_indexes = pd.DataFrame(
        data=[],
        index=precip.index,
        columns=pd.MultiIndex.from_tuples(
            [('SPI_ref', i) for i in range(1, 13)] +
            [('SPI', i) for i in range(1, 13)] +
            [('SPLI', i) for i in range(1, 13)] +
            [('SPLI_corr', i) for i in range(1, 13)],
            names=['std index', 'month'])
        )
    std_indexes.attrs['precip_win'] = precip_win
    std_indexes.attrs['wlvl_win'] = wlvl_win
    std_indexes.attrs['staname'] = staname

    precip_norm = []
    precip_pdf = []
    wlvl_norm = []
    wlvl_pdf = []
    for m in range(1, 13):
        print("Processing month {}".format(m))

        wlvl_m = wlvl[('water_level', m)].dropna()
        precip_m = precip[('precip', m)].dropna()

        # Calcul des valeurs de SPI en utilisant les données de la période
        # de référence 1981 à 2010 (inclusivement).
        x = precip_m.loc[1981:2010].values
        loc, scale = norm.fit(x)
        precip_norm.append((loc, scale))
        precip_pdf.append((x, norm.pdf(x, loc, scale)))

        spi_ref = (precip_m.values - loc) / scale
        std_indexes.loc[precip_m.index, ('SPI_ref', m)] = spi_ref

        # Calcul des valeurs de SPI en utilisant les données pour les années
        # correspondantes à celles pour lesquelles des données de niveaux d'eau
        # sont disponible.
        x = precip_m.loc[wlvl_m.index].values
        loc, scale = norm.fit(x)

        spi = (precip_m.values - loc) / scale
        std_indexes.loc[precip_m.index, ('SPI', m)] = spi

        # Calcul des paramètres d'un modèle de régression linéaire qui
        # permettra de corriger les SPLI.
        p = np.polyfit(spi_ref, spi, deg=1)

        # Calcul des SPLI.
        # https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
        # https://stackabuse.com/kernel-density-estimation-in-python-using-scikit-learn/
        x = wlvl_m.values

        loc, scale = norm.fit(x)
        wlvl_norm.append((loc, scale))
        wlvl_pdf.append((x, norm.pdf(x, loc, scale)))

        spli = (x - loc) / scale
        std_indexes.loc[wlvl_m.index, ('SPLI', m)] = spli

        # Correction des valeurs de SPLI pour la période de référence.
        spli_corr = p[0] * spli + p[1]
        std_indexes.loc[wlvl_m.index, ('SPLI_corr', m)] = spli_corr

    # =========================================================================
    # Sauvegarde des calculs
    # =========================================================================
    print("Formatting results for station {}...".format(staname))

    std_indexes = std_indexes.stack(level=1, dropna=False)
    std_indexes['day'] = 1
    std_indexes = std_indexes.reset_index()
    std_indexes.index = pd.to_datetime(
        std_indexes[['year', 'month', 'day']])
    std_indexes = std_indexes[['SPI_ref', 'SPI', 'SPLI', 'SPLI_corr']]
    std_indexes.index.name = 'Date/Time'

    # =========================================================================
    # Plot results
    # =========================================================================
    print("Plotting results for station {}...".format(staname))
    fig = plot_spli_overview(
        staname, sta_wl_daily, sta_precip_daily, std_indexes)

    fig2 = plot_pdf_niveau(wlvl_norm, wlvl_pdf, wlvl_win, staname)
    fig3 = plot_pdf_precip(precip_norm, precip_pdf, precip_win, staname)
    fig4 = plot_cross_corr(std_indexes)
    fig5 = plot_spli_vs_classes(std_indexes)

    figures = (fig, fig2, fig3, fig4, fig5)

    return std_indexes, figures


# %%
plt.ioff()

WLVL_WIN = 12
PRECIP_WIN = 12

figures_stack = []
std_indexes_stack = []
for staname in selected_stations:
    std_indexes, figures = calc_std_indexes(
        staname=staname,
        precip_win=PRECIP_WIN,
        wlvl_win=WLVL_WIN)
    figures_stack.append(figures)
    std_indexes_stack.append(std_indexes)
    plt.close('all')

DIRNAME = osp.join(osp.dirname(__file__), 'results_std_indexes')
FILENAMES = [
    f'(spi{PRECIP_WIN}_spli{WLVL_WIN}) spi_vs_spli.pdf',
    f'(spi{PRECIP_WIN}_spli{WLVL_WIN}) pdf_niveau.pdf',
    f'(spi{PRECIP_WIN}_spli{WLVL_WIN}) pdf_precip.pdf',
    f'(spi{PRECIP_WIN}_spli{WLVL_WIN}) correlation_croisee.pdf',
    f'(spi{PRECIP_WIN}_spli{WLVL_WIN}) spli_vs_classes.pdf']
for i, filename in enumerate(FILENAMES):
    filepath = osp.join(DIRNAME, filename)
    with PdfPages(filepath) as pdf:
        for figures in figures_stack:
            pdf.savefig(figures[i])
