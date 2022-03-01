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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm

os.chdir(osp.dirname(__file__))
from selected_stations import selected_stations
from plot_results import plot_spli_overview
plt.close('all')

MONTH_NAMES = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet',
               'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']

dirname = osp.dirname(__file__)
PRECIP_DAILY_ALL = pd.read_csv(
    'precip_daily_2022-02-28.csv',
    index_col=[0], parse_dates=[0])

WLVL_DAILY_ALL = pd.read_csv(
    'waterlevels_daily_2022-02-28.csv',
    index_col=[0], parse_dates=[0])


def calc_std_indexes(staname, dirname,
                     precip_win: int = 12,
                     wlvl_win: int = 3):

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

        # Calcul des paramètres d'un modèle de régression linéaire qui permettra
        # de corriger les SPLI.
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
    # fig = plot_spi_vs_spli(std_indexes, staname, precip_win, wlvl_win)
    # fig.savefig(osp.join(outdir, 'spi_vs_spli.pdf'))

    fig = plot_spli_overview(
        staname, sta_wl_daily, sta_precip_daily, std_indexes)
    fig.savefig(osp.join(outdir, 'spi_vs_spli.pdf'))

    fig2 = plot_pdf_niveau(wlvl_norm, wlvl_pdf, wlvl_win)
    fig2.savefig(osp.join(outdir, f'pdf_niveau_{wlvl_win}_mois.pdf'))

    fig3 = plot_pdf_precip(precip_norm, precip_pdf, precip_win)
    fig3.savefig(osp.join(outdir, f'pdf_precip_{precip_win}_mois.pdf'))

    fig4 = plot_cross_corr(std_indexes)
    fig4.savefig(osp.join(outdir, 'correlation_croisee.pdf'))

    fig5 = plot_spli_vs_classes(std_indexes, staname)
    fig5.savefig(osp.join(outdir, 'spli_vs_bandes_couleurs.pdf'))

    figures = (fig, fig2, fig3, fig4, fig5)
    return std_indexes, figures


def plot_spi_vs_spli(std_indexes: pd.DataFrame, staname: str,
                     precip_win: int, wlvl_win: int):
    gridcolor = '0.66'
    spi_color = 'orange'
    spli_color = '#0080FF'

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(std_indexes['SPI_ref'],
            label='SPI_{} (1981-2010)'.format(precip_win),
            color=spi_color)
    ax.plot(std_indexes['SPLI_corr'],
            label='SPLI_{} corrigés'.format(wlvl_win),
            color=spli_color)

    y_min = min(std_indexes['SPI_ref'].min(), std_indexes['SPLI_corr'].min())
    y_max = max(std_indexes['SPI_ref'].max(), std_indexes['SPLI_corr'].max())
    y_min -= (y_max - y_min) * 0.1
    y_max += (y_max - y_min) * 0.1

    # Setup xaxis.
    year_min = 1995
    year_max = 2022
    delta_year = year_max - year_min

    if delta_year <= 15:
        base = 1
    elif delta_year <= 30:
        base = 2
    else:
        base = 5

    xmin = datetime(year_min, 1, 1)
    xmax = datetime(year_max, 1, 1)
    ax.axis(xmin=xmin, xmax=xmax, ymin=y_min, ymax=y_max)

    ax.xaxis.set_major_locator(mdates.YearLocator(
        base=base, month=1, day=1))
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter('%Y-%m-%d'))

    # Setup minor x-ticks.
    if base > 1:
        ax.xaxis.set_minor_locator(mdates.YearLocator(
            base=1, month=1, day=1))
        ax.tick_params(axis='x', which='minor', bottom=True)

    # Setup grid.
    ax.grid(visible=True, which='major', axis='y', color=gridcolor,
            linestyle='-', linewidth=0.5)
    # if base > 1:
    #     ax.grid(visible=True, which='minor', axis='x', color=gridcolor,
    #             linestyle='-', linewidth=0.5)

    fig.autofmt_xdate()
    fig.suptitle("Station {}".format(staname), fontsize=16)

    # Setup legend.
    ax.legend(
        bbox_to_anchor=[0, 1], loc='lower left', ncol=4,
        handletextpad=0.5, numpoints=1, fontsize=10, frameon=False,
        borderpad=0, labelspacing=0, borderaxespad=0.1)

    return fig


def plot_pdf_niveau(wlvl_norm: list, wlvl_pdf: list, wlvl_win: int,
                    staname: str):
    fig, axes = plt.subplots(4, 3, figsize=(11, 8.5))

    for i, ax in enumerate(axes.flatten()):
        dx = 0.01
        xp = np.arange(0, 1000 + dx/2, dx)
        loc, scale = wlvl_norm[i]
        yp = norm.pdf(xp, loc, scale)
        ax.plot(xp, yp, '-')

        x = wlvl_pdf[i][0]
        y = wlvl_pdf[i][1]
        ax.plot(x, y, '.')

        n, bins, patches = ax.hist(x, density=True, color='0.8')

        ax.set_title(MONTH_NAMES[i])
        if i % 3 == 0:
            ax.set_ylabel('Densité')
        if i > 8:
            ax.set_xlabel('Niveau (m)')

        axis_xmin = np.floor(np.min(x)) - 0.5
        axis_xmax = np.ceil(np.max(x)) + 0.5
        ax.axis(xmin=axis_xmin, xmax=axis_xmax)

    suptitle = f"PDF Niveaux moyens ({wlvl_win} mois) - Station {staname}"
    fig.suptitle(suptitle, fontsize=16)
    fig.align_ylabels()
    fig.subplots_adjust(
        top=0.9, bottom=0.1, hspace=0.5, left=0.1, right=0.975)
    return fig


def plot_pdf_precip(precip_norm: list, precip_pdf: list, precip_win: int,
                    staname: str):
    fig, axes = plt.subplots(4, 3, figsize=(11, 8.5))

    for i, ax in enumerate(axes.flatten()):
        dx = 0.01
        loc, scale = precip_norm[i]
        xp = np.arange(0, 2000 + dx/2, dx)
        yp = norm.pdf(xp, loc, scale)
        ax.plot(xp, yp, '-')

        x = precip_pdf[i][0]
        y = precip_pdf[i][1]
        ax.plot(x, y, '.')

        n, bins, patches = ax.hist(x, density=True, color='0.8')

        ax.set_title(MONTH_NAMES[i])
        if i % 3 == 0:
            ax.set_ylabel('Densité')
        if i > 8:
            ax.set_xlabel('Précipitation (mm)')

        axis_xmin = np.floor(np.min(x)) - 50
        axis_xmax = np.ceil(np.max(x)) + 50
        ax.axis(xmin=axis_xmin, xmax=axis_xmax)

    suptitle = f"PDF Précipitations ({precip_win} mois) - Station {staname}"
    fig.suptitle(suptitle, fontsize=16)
    fig.align_ylabels()
    fig.subplots_adjust(
        top=0.9, bottom=0.1, hspace=0.5, left=0.1, right=0.975)

    return fig


def plot_cross_corr(std_indexes):
    x = std_indexes['SPLI_corr'].values.astype(float)
    y = std_indexes['SPI_ref'].values.astype(float)
    shifts = np.arange(-12, 13)
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
    ax.set_xticks(shifts[::2])
    ax.set_xticks(shifts, minor=True)
    ax.axis(xmin=-12, xmax=12)
    fig.tight_layout()
    return fig


def plot_spli_vs_classes(std_indexes, staname):
    gridcolor = '0.66'
    spi_color = 'orange'
    spli_color = '#0080FF'

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {
        'tres_bas': "#db442c",
        'bas': '#f18e00',
        'mod_bas': '#ffdd57',
        'proche_moy': '#6dc55a',
        'mod_haut': '#32a9dd',
        'haut': '#1b75bb',
        'tres_haut': '#286273'
        }

    marker = 'o'
    marker_size = 8

    ax.axhspan(-5, -1.28, color="#db442c")
    ax.axhspan(-1.28, -0.84, color="#f18e00")
    ax.axhspan(-0.84, -0.25, color="#ffdd57")
    ax.axhspan(-0.25, 0.25, color="#6dc55a")
    ax.axhspan(0.25, 0.84, color="#32a9dd")
    ax.axhspan(0.84, 1.28, color="#1b75bb")
    ax.axhspan(1.28, 5, color="#286273")

    ax.plot(std_indexes['SPLI_corr'].dropna(), color='black')

    y_min = std_indexes['SPLI_corr'].dropna().min()
    y_max = std_indexes['SPLI_corr'].dropna().max()
    y_min -= (y_max - y_min) * 0.05
    y_max += (y_max - y_min) * 0.05

    # Setup xaxis.
    year_min = std_indexes['SPLI_corr'].dropna().index.min().year
    year_max = std_indexes['SPLI_corr'].dropna().index.max().year + 1
    delta_year = year_max - year_min

    if delta_year <= 15:
        base = 1
    elif delta_year <= 30:
        base = 2
    else:
        base = 5

    xmin = datetime(year_min, 1, 1)
    xmax = datetime(year_max, 1, 1)
    ax.axis(xmin=xmin, xmax=xmax, ymin=y_min, ymax=y_max)

    ax.xaxis.set_major_locator(mdates.YearLocator(
        base=base, month=1, day=1))
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter('%Y-%m-%d'))

    # setup y-ticks.
    ax.set_yticks([-1.28, -0.84, -0.25, 0.25, 0.84, 1.28])
    ax.set_ylabel('SPLI_3mois corrigés', fontsize=14, labelpad=15)

    # Setup minor x-ticks.
    if base > 1:
        ax.xaxis.set_minor_locator(mdates.YearLocator(
            base=1, month=1, day=1))
        ax.tick_params(axis='x', which='minor', bottom=True)

    fig.autofmt_xdate()
    fig.suptitle("Station {}".format(staname), fontsize=16)
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
