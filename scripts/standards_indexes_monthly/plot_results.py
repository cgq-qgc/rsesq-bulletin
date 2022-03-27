# -*- coding: utf-8 -*-
"""
A script to calculate SPI and SPLI at selected piezometric stations
of the RSESQ.
"""
from datetime import datetime
import os
import os.path as osp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from scipy.stats import norm

COLORS = {
    'blue dark': '#0080FF',
    'blue light': '#CCCCFF',
    'Precip': 'orange'}
GRIDCOLOR = '0.66'
MONTH_NAMES = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet',
               'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']


def plot_spli_overview(staname, wlvl_daily, precip_daily, std_indexes):
    # Nous calculons les SPI et SPLI seulement pour les données récentes.
    # wlvl_mly = wlvl_mly[wlvl_mly.index >= 1981].copy()
    # wlvl_mly.unstack(level=0)

    fig, axs = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)

    # We make sure there is at least one data entry for each month. This avoid
    # having long straigth lines between two data point that are on each side
    # of a big data gap.
    year_range = pd.Index(np.arange(
        wlvl_daily.index.year.min(), wlvl_daily.index.year.max() + 1))
    for year in year_range:
        for month in range(1, 13):
            mask = ((wlvl_daily.index.year == year) &
                    (wlvl_daily.index.month == month))
            if wlvl_daily[mask].isnull().all().values[0]:
                wlvl_daily.loc[datetime(year, month, 1)] = np.nan
    wlvl_daily = wlvl_daily.sort_index()

    axs[0].plot(wlvl_daily, color=COLORS['blue light'], zorder=1, lw=1,
                label="Niveaux journaliers")

    # Plot orphans daily water levels if any.
    orphan_idx = []
    values = wlvl_daily.values
    for i in range(len(values)):
        if i == 0:
            above_isnull = True
        else:
            above_isnull = pd.isnull(values[i - 1])
        if i == len(values) - 1:
            below_isnull = True
        else:
            below_isnull = pd.isnull(values[i + 1])
        if above_isnull and below_isnull:
            orphan_idx.append(i)
    axs[0].plot(wlvl_daily.iloc[orphan_idx], color=COLORS['blue light'],
                marker='.', ls='None', ms=2,
                zorder=1)
    axs[0].set_ylabel("Niveau d'eau (m NMM)")

    # Plot water monthly means.
    wlvl_mly = wlvl_daily.copy()
    wlvl_mly['year'] = wlvl_mly.index.year
    wlvl_mly['month'] = wlvl_mly.index.month
    wlvl_mly['day'] = 15
    wlvl_mly = wlvl_mly.groupby(['year', 'month']).mean()
    wlvl_mly = wlvl_mly.reset_index()
    wlvl_mly.index = pd.to_datetime(
        wlvl_mly[['year', 'month', 'day']])
    wlvl_mly = wlvl_mly.drop(['year', 'month', 'day'], axis=1)

    axs[0].plot(wlvl_mly.index,
                wlvl_mly.values,
                marker='None', color=COLORS['blue dark'], ls='-',
                zorder=100,
                label="Niveaux mensuels moyens")

    # Plot water level yearly mean
    wl_mean = wlvl_daily.copy()
    wl_mean['year'] = wl_mean.index.year
    wl_mean = wl_mean.groupby(['year']).mean()
    wl_mean = wl_mean.reset_index()
    wl_mean['month'] = 6
    wl_mean['day'] = 15
    wl_mean.index = pd.to_datetime(wl_mean[['year', 'month', 'day']])

    axs[0].plot(wl_mean['water_level'],
                marker='None', ms=3, color='red',
                ls='--', lw=1, zorder=120,
                label='Niveaux annuels moyens')
    axs[0].plot(wl_mean['water_level'], marker='o', ms=3,
                color='red', ls='None', zorder=120)

    # Plot mean water level.
    waterlvl_mean = np.nanmean(wlvl_daily.values)
    axs[0].axhline(waterlvl_mean, ls='-', color='black', lw=1,
                   zorder=10, label='Niveau moyen')

    # Plot yearly total precipitation.
    precip = precip_daily.copy()
    precip['year'] = precip.index.year
    precip = precip.groupby(['year']).sum()
    precip = precip.reset_index()
    precip['month'] = 6
    precip['day'] = 15
    precip.index = pd.to_datetime(precip[['year', 'month', 'day']])

    axs[1].plot(precip.index, precip['precip'].values,
                marker='o', color=COLORS['blue dark'], ls='--', zorder=100,
                label='Précipitations annuelles', ms=5)
    axs[1].set_ylabel("Précipitations (mm)")

    # Plot total precip yearly normal.
    mask = (precip.index.year >= 1981) & (precip.index.year <= 2010)
    precip_yearly_normal = precip.loc[mask, 'precip'].mean()
    axs[1].axhline(precip_yearly_normal, ls='-', color='black',
                   lw=1, zorder=1,
                   label='Précipitations annuelles normales (1981-2010)')

    # Plot SPI and SPLI results.
    spi = std_indexes['SPI_ref']
    spli_corr = std_indexes['SPLI_corr']

    precip_win = std_indexes.attrs['precip_win']
    wlvl_win = std_indexes.attrs['wlvl_win']

    axs[2].plot(spi.index, spi.values,
                marker='None', color=COLORS['Precip'], zorder=5,
                label=f'SPI_{precip_win}mois (1981-2010)')
    axs[2].plot(spli_corr.index, spli_corr.values, zorder=10,
                marker='None', ls='--', color=COLORS['blue dark'],
                label=f'SPLI_{wlvl_win}mois corrigés')

    axs[2].set_ylabel("Écart normalisé")
    axs[2].grid(visible=True, which='major', axis='y',
                linestyle='-', linewidth=0.5, color=GRIDCOLOR)

    # Setup xaxis.
    mask = pd.notnull(std_indexes['SPLI_corr'].values)
    year_min = min(std_indexes.index[mask].min().year, 2010)
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
    axs[0].axis(xmin=xmin, xmax=xmax)

    axs[0].xaxis.set_major_locator(mdates.YearLocator(
        base=base, month=1, day=1))
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Setup minot x-ticks.
    if base > 1:
        axs[0].xaxis.set_minor_locator(mdates.YearLocator(
            base=1, month=1, day=1))
        axs[0].tick_params(axis='x', which='minor', bottom=True)

    fig.autofmt_xdate()
    fig.suptitle("Station {}".format(staname), fontsize=16)
    fig.align_ylabels()
    fig.subplots_adjust(
        top=0.875, bottom=0.075, hspace=0.25, left=0.1, right=0.975)

    # Setup grid.
    for ax in axs:
        ax.grid(visible=True, which='major', axis='x', color=GRIDCOLOR,
                linestyle='-', linewidth=0.5)
        if base > 1:
            ax.grid(visible=True, which='minor', axis='x', color=GRIDCOLOR,
                    linestyle='-', linewidth=0.5)

    # Setup legend.
    for ax in axs:
        ax.legend(
            bbox_to_anchor=[0, 1], loc='lower left', ncol=4,
            handletextpad=0.5, numpoints=1, fontsize=10, frameon=False,
            borderpad=0, labelspacing=0.3, borderaxespad=0.1)

    return fig


def plot_spli_vs_classes(std_indexes):
    staname = std_indexes.attrs['staname']
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

    ax.axhspan(-10, -1.28, color=colors['tres_bas'])
    ax.axhspan(-1.28, -0.84, color=colors['bas'])
    ax.axhspan(-0.84, -0.25, color=colors['mod_bas'])
    ax.axhspan(-0.25, 0.25, color=colors['proche_moy'])
    ax.axhspan(0.25, 0.84, color=colors['mod_haut'])
    ax.axhspan(0.84, 1.28, color=colors['haut'])
    ax.axhspan(1.28, 10, color=colors['tres_haut'])

    ax.plot(std_indexes['SPLI_corr'], color='black')

    y_min = min(std_indexes['SPLI_corr'].dropna().min(), 2010)
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

    xmin = datetime(min(year_min, 2010), 1, 1)
    xmax = datetime(year_max, 1, 1)
    ax.axis(xmin=xmin, xmax=xmax, ymin=y_min, ymax=y_max)

    ax.xaxis.set_major_locator(mdates.YearLocator(base=base, month=1, day=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # setup y-ticks.
    wlvl_win = std_indexes.attrs['wlvl_win']
    ax.set_yticks([-1.28, -0.84, -0.25, 0.25, 0.84, 1.28])
    ax.set_ylabel(f'SPLI_{wlvl_win}mois corrigés', fontsize=14, labelpad=15)

    # Setup grid.
    ax.grid(visible=True, which='both', axis='x', color='black',
            linestyle='--', linewidth=0.5)

    # Setup minor x-ticks.
    if base > 1:
        ax.xaxis.set_minor_locator(mdates.YearLocator(
            base=1, month=1, day=1))
        ax.tick_params(axis='x', which='minor', bottom=True)

    fig.autofmt_xdate()
    fig.suptitle("Station {}".format(staname), fontsize=16)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.subplots_adjust(top=0.9)

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
    staname = std_indexes.attrs['staname']
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

    # Plot a vertical line to identify the positive shift where the
    # correlation coefficient is maximum.
    _corrcoeffs = np.array(corrcoeffs)
    _corrcoeffs[shifts < 0] = -999
    ax.axvline(shifts[np.argmax(_corrcoeffs)], color='red', zorder=10)

    ax.set_ylabel('Corrélation', labelpad=15, fontsize=14)
    ax.set_xlabel('Décalage SPLI p/r SPI (mois)', labelpad=10, fontsize=14)
    ax.set_xticks(shifts[::4])
    ax.set_xticks(shifts, minor=True)
    ax.axis(xmin=-24, xmax=24)

    fig.suptitle(f"Station {staname}", fontsize=16)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.subplots_adjust(top=0.85)

    return fig
