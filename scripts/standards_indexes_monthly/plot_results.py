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
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl

from selected_stations import selected_stations

# mpl.rcParams['figure.max_open_warning'] = len(selected_stations)

# precip_yrly = pd.read_csv(
#     'precip_yearly.csv', index_col=[0], parse_dates=[0])
# precip_daily = pd.read_csv(
#     'precip_daily.csv', index_col=[0], parse_dates=[0])
# precip_normals = pd.read_csv(
#     'precip_normals_1981-2010.csv', index_col=[0])

# wl_daily_all = pd.read_csv(
#     'waterlevels_daily.csv', index_col=[0], parse_dates=[0])
# wl_daily_all['water_year'] = (
#     wl_daily_all.index.year
#     .where(wl_daily_all.index.month < 10, wl_daily_all.index.year + 1))

# yearly_mean_wl = wl_daily_all.groupby('water_year').mean()
# yearly_min_wl = wl_daily_all.groupby('water_year').min()
# yearly_max_wl = wl_daily_all.groupby('water_year').max()

# %%
# plt.close('all')
# plt.ioff()

# Color definitions.
COLORS = {
    'blue dark': '#0080FF',
    'blue light': '#CCCCFF',
    'Precip': 'orange'
    }
gridcolor = '0.66'
daily_wl_color = '#CCCCFF'
monthly_wl_color = '#0080FF'
max_wl_color = '#660000'
min_wl_color = '#FF9999'
mean_line_color = 'black'
precip_color = '#0080FF'

std_ind_max = 0
std_ind_min = 0


def plot_spli_overview(staname, wlvl_daily, precip_daily, std_indexes):
    print("Plotting results for station {}...".format(staname))
    # Nous calculons les SPI et SPLI seulement pour les données récentes.
    # wlvl_mly = wlvl_mly[wlvl_mly.index >= 1981].copy()
    # wlvl_mly.unstack(level=0)

    fig, axs = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)

    # We make sure there is at least one data entry for each year. This avoid
    # having long straigth lines between two data point that are on each side
    # of a big data gap.
    year_range = pd.Index(np.arange(
        wlvl_daily.index.year.min(), wlvl_daily.index.year.max() + 1))
    isin = year_range.isin(wlvl_daily.index.year.unique())
    for year in year_range[~isin]:
        wlvl_daily.loc[datetime(year, 1, 1)] = np.nan
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
                marker='.', ls='None', ms=2, zorder=1)
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
                marker='None', color=COLORS['blue dark'], ls='-', zorder=100,
                label="Niveaux moyens annuels")

    # Plot yearly total precipitation.
    precip = precip_daily.copy()
    precip['year'] = precip.index.year
    precip = precip.groupby(['year']).sum()
    precip = precip.reset_index()
    precip['month'] = 6
    precip['day'] = 15
    precip.index = pd.to_datetime(precip[['year', 'month', 'day']])

    axs[1].bar(precip.index, precip['precip'].values,
               width=150, zorder=10, color=COLORS['blue light'],
               )
    axs[1].plot(precip.index, precip['precip'].values,
                marker='o', color=COLORS['blue dark'], ls='--', zorder=100,
                label='Précipitations annuelles')
    axs[1].set_ylabel("Précipitations (mm)")
    axs[1].axis(ymin=500)

    # Plot SPI and SPLI results.
    spi = std_indexes['SPI_ref']
    spli_corr = std_indexes['SPLI_corr']

    axs[2].plot(spli_corr.index, spli_corr.values,
                marker='None', ls='-', color=COLORS['blue dark'],
                label='SPLI_3mois corrigés')
    axs[2].plot(spi.index, spi.values,
                marker='None', color=COLORS['Precip'], zorder=5,
                label='SPI_12mois (1981-2010)')

    axs[2].set_ylabel("Écart normalisé")

    mask = pd.notnull(std_indexes['SPLI_corr'].values)
    y_min = min(
        std_indexes[mask]['SPLI_corr'].dropna().min(),
        std_indexes[mask]['SPI_ref'].dropna().min())
    y_max = max(
        std_indexes[mask]['SPLI_corr'].dropna().max(),
        std_indexes[mask]['SPI_ref'].dropna().max())
    y_min += -0.05 * (y_max - y_min)
    y_max += 0.05 * (y_max - y_min)

    # yticks = [-ylim, -1.28, -0.84, -0.25, 0.25, 0.84, 1.28, ylim]
    # axs[2].set_yticks(yticks)
    # axs[2].set_yticklabels(yticks)
    axs[2].axis(ymin=y_min, ymax=y_max)
    axs[2].grid(visible=True, which='major', axis='y',
                linestyle='-', linewidth=0.5, color=gridcolor)

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
        top=0.9, bottom=0.1, hspace=0.25, left=0.1, right=0.975)

    # Setup grid.
    for ax in axs:
        ax.grid(visible=True, which='major', axis='x', color=gridcolor,
                linestyle='-', linewidth=0.5)
        if base > 1:
            ax.grid(visible=True, which='minor', axis='x', color=gridcolor,
                    linestyle='-', linewidth=0.5)

    # Setup legend.
    for ax in axs:
        ax.legend(
            bbox_to_anchor=[0, 1], loc='lower left', ncol=4,
            handletextpad=0.5, numpoints=1, fontsize=10, frameon=False,
            borderpad=0, labelspacing=0, borderaxespad=0.1)

    return fig

# print('Saving the results to the PDF file...')
# with PdfPages('standard_index_results (min-mean-max).pdf') as pdf:
#     for fig in figures:
#         pdf.savefig(fig)
