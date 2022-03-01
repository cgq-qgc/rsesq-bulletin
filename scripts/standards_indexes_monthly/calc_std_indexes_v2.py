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
import matplotlib.dates as mdates
from scipy.stats import norm

os.chdir(osp.dirname(__file__))
from selected_stations import selected_stations
from plot_results import (
    plot_spli_overview, plot_spli_vs_classes, plot_pdf_niveau, plot_pdf_precip)
matplotlib.rcParams['axes.unicode_minus'] = False
plt.close('all')

WORKDIR = osp.dirname(__file__)
PRECIP_DAILY_ALL = pd.read_csv(
    'precip_daily_2022-02-28.csv',
    index_col=[0], parse_dates=[0])

WLVL_DAILY_ALL = pd.read_csv(
    'waterlevels_daily_2022-02-28.csv',
    index_col=[0], parse_dates=[0])


def calc_std_indexes(staname, dirname,
                     precip_win: int = 12, wlvl_win: int = 3):
    print("Calculating standard indexes for station {}...".format(staname))
    outdir = osp.join(dirname, staname)
    if not osp.exists(outdir):
        os.makedirs(outdir)

    # =========================================================================
    # Préparation des séries mensuelles
    # =========================================================================

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
    print("Saving results for station {}...".format(staname))

    std_indexes = std_indexes.stack(level=1, dropna=False)
    std_indexes['day'] = 1
    std_indexes = std_indexes.reset_index()
    std_indexes.index = pd.to_datetime(
        std_indexes[['year', 'month', 'day']])
    std_indexes = std_indexes[['SPI_ref', 'SPI', 'SPLI', 'SPLI_corr']]
    std_indexes.index.name = 'Date/Time'
    std_indexes.to_csv(osp.join(outdir, 'std_indexes_resultats.csv'))

    # =========================================================================
    # Plot results
    # =========================================================================
    print("Plotting results for station {}...".format(staname))
    fig = plot_spli_overview(
        staname, sta_wl_daily, sta_precip_daily, std_indexes)
    fig.savefig(osp.join(outdir, 'spi_vs_spli.pdf'))

    fig2 = plot_pdf_niveau(wlvl_norm, wlvl_pdf, wlvl_win, staname)
    fig2.savefig(osp.join(outdir, f'pdf_niveau_{wlvl_win}_mois.pdf'))

    fig3 = plot_pdf_precip(precip_norm, precip_pdf, precip_win, staname)
    fig3.savefig(osp.join(outdir, f'pdf_precip_{precip_win}_mois.pdf'))

    fig4 = plot_cross_corr(std_indexes, staname)
    fig4.savefig(osp.join(outdir, 'correlation_croisee.pdf'))

    fig5 = plot_spli_vs_classes(std_indexes, staname)
    fig5.savefig(osp.join(outdir, 'spli_vs_classes.pdf'))

    figures = (fig, fig2, fig3, fig4, fig5)
    return std_indexes, figures





def plot_cross_corr(std_indexes, staname):
    x = std_indexes['SPLI_corr'].values.astype(float)
    y = std_indexes['SPI_ref'].values.astype(float)
    shifts = np.arange(-24, 25)
    corrcoeffs = []
    for shift in shifts:
        if shift < 0:
            ys = np.hstack([y[-shift:], [np.nan] * -shift])
        elif shift > 0:
            ys = np.hstack([[np.nan] * shift, y[:-shift]])
        mask = (~np.isnan(x)) & (~np.isnan(ys))
        corrcoeffs.append(np.corrcoef(x[mask], ys[mask])[0, 1])

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(shifts, corrcoeffs, marker='.', zorder=100)
    ax.axvline(shifts[np.argmax(corrcoeffs)], color='red', zorder=10)

    ax.set_ylabel('Corrélation', labelpad=15, fontsize=14)
    ax.set_xlabel('Décalage SPLI p/r SPI (mois)', labelpad=10, fontsize=14)
    ax.set_xticks(shifts[::4])
    ax.set_xticks(shifts, minor=True)
    ax.axis(xmin=-24, xmax=24)

    fig.suptitle(f"Station {staname}", fontsize=16)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.subplots_adjust(top=0.85)

    return fig






std_indexes, figures = calc_std_indexes(
    staname='02360001',
    dirname=osp.join(dirname, 'results_std_indexes')
    )
# calc_std_indexes(
#     staname='05080001',
#     dirname=osp.join(dirname, 'results_std_indexes')
#     )

# # Setup legend.
# ax.legend(
#     bbox_to_anchor=[0, 1], loc='lower left', ncol=4,
#     handletextpad=0.5, numpoints=1, fontsize=10, frameon=False,
#     borderpad=0, labelspacing=0, borderaxespad=0.1)
