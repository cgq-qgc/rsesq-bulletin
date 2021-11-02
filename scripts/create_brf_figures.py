# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:38:46 2021
@author: User
"""

# A script to produce BRF figures for the RSESQ data sheet.

import os.path as osp
from gwhat.projet.reader_projet import ProjetReader
from gwhat.brf_mod.kgs_plot import BRFFigure

regions = [
    "capitale-nationale",
    "centre-quebec",
    "chau_app2",
    "monteregie",
    "montreal"
    ]

chosen_brf_indexes = {
    # ---- capitale-nationale
    'Clermont': 2,
    'La Malbaie': 0,
    'Lac-Jacques-Cartier': 0,
    'Lac-Jacques-Cartier_01': 0,
    'Lac-Jacques-Cartier_02': 0,
    'Lac-Pikauba': 2,
    'Pont-Rouge': 0,
    'Pont-Rouge_01': 0,
    'Québec': 0,
    'Québec_01': 2,
    'Saint-Ferréol-les-Neiges': 0,
    'Saint-Léonard-de-Portneuf': 0,
    'Saint-Raymond': 0,
    'Saint-Siméon': 0,
    "Sainte-Christine-d'Auvergne": 0,
    'Sainte-Famille': 1,
    # ---- centre-quebec
    'Asbestos': 3,
    'Baie-du-Febvre': 11,
    'Bécancour': 2,
    'Cookshire-Eaton': 3,
    'Drummondville': 26,
    'Drummondville_01': 8,
    'Dudswell': 0,
    'Leclercville': 4,
    'Manseau': 8,
    'Saint-Albert': 4,
    'Saint-Camille': 10,
    'Saint-Isidore-de-Clifton': 0,
    'Saint-Isidore-de-Clifton_01': 1,
    'Saint-Rémi-de-Tingwick': 11,
    'Saint-Édouard-de-Lotbinière': 10,
    'Sainte-Anne-du-Sault': 6,
    'Sainte-Marie-de-Blandford': 32,
    'Sainte-Marie-de-Blandford_01': 9,
    'Sainte-Monique': 3,
    'Ulverton': 1,
    'Victoriaville': 0,
    'Villeroy': 1,
    'Villeroy_01': 1,
    'Villeroy_02': 3,
    # ---- chau_app2
    'Armagh': None,
    'Armagh 2': None,
    'Berthier-sur-Mer': None,
    'Disraeli': 1,
    'Frampton': None,
    'Irlande': 0,
    'Irlande 2': 0,
    "L'Islet": 0,
    'Saint-Agapit': 5,
    'Saint-Anselme': 1,
    'Saint-Antoine-de-Tilly': 7,
    'Saint-Charles-de-Bellechasse': 0,
    'Saint-Georges': 0,
    'Saint-Gilles': 0,
    'Saint-Honoré-de-Shenley': 7,
    'Saint-Jacques-de-Leeds': 9,
    'Saint-Jean-Port-Joli': 5,
    'Saint-Luc-de-Bellechasse': None,
    'Saint-Magloire': None,
    'Saint-Martin': 0,
    'Saint-Pamphile': None,
    'Saint-Théophile': 0,
    'Saint-Vallier': None,
    'Saint-Zacharie': 2,
    'Sainte-Justine': 11,
    'Thetford Mines': 35,
    'Tourville': None,
    # ---- monteregie
    'Brome': 1,
    'Bromont': 1,
    'Bromont_01': 9,
    'Calixa-Lavallée': 2,
    'Cowansville': 7,
    'Cowansville_01': 4,
    'Eastman': 1,
    'Frelighsburg': 0,
    'Orford': 4,
    'Potton': 1,
    'Rougemont': 1,
    'Rougemont_01': 31,
    'Rougemont_02': 4,
    'Saint-Alphonse-de-Granby': 1,
    'Saint-Amable': 2,
    'Saint-Damase': 1,
    'Saint-Guillaume': 26,
    'Saint-Guillaume_01': 3,
    'Saint-Hugues': 1,
    'Saint-Hyacinthe': 5,
    'Saint-Ignace-de-Stanbridge': 2,
    'Saint-Ignace-de-Stanbridge_01': 5,
    'Saint-Jean-sur-Richelieu': 1,
    'Saint-Jean-sur-Richelieu_01': 15,
    'Saint-Marcel-sur-Richelieu': 2,
    'Saint-Mathias-sur-Richelieu': 5,
    'Saint-Ours': 1,
    "Saint-Paul-d'Abbotsford": 11,
    "Saint-Paul-de-l'Île-aux-Noix": 1,
    'Saint-Simon': 10,
    "Saint-Théodore-d'Acton": 2,
    'Saint-Valérien-de-Milton': 1,
    'Sainte-Christine': 1,
    'Sainte-Victoire-de-Sorel': 6,
    'Sutton': 5,
    'Sutton_01': 3,
    'Valcourt': 5,
    # ---- montreal
    'Brownsburg-Chatham': 0,
    'Brownsburg-Chatham_01': 2,
    'Elgin': 8,
    'Franklin': 10,
    'Franklin_01': 4,
    'Franklin_02': 0,
    'Godmanchester': 1,
    'Godmanchester_01': 1,
    'Grenville-sur-la-Rouge': 2,
    'Havelock': 4,
    'Lachute': 1,
    'Lachute_01': 0,
    'Lachute_02': 1,
    'Les Cèdres': 2,
    'Mercier': 2,
    'Mercier_01': 3,
    'Mercier_02': 1,
    'Mirabel': 0,
    'Mirabel_01': 0,
    'Mirabel_02': 0,
    'Mirabel_03': 2,
    'Mirabel_04': 7,
    'Oka': 0,
    'Ormstown': 21,
    'Saint-Anicet': 9,
    'Saint-Isidore': 8,
    'Saint-Michel': 14,
    'Saint-Michel_01': 3,
    'Saint-Patrice-de-Sherrington': 1,
    'Saint-Patrice-de-Sherrington_01': 4,
    'Saint-Rémi': 5,
    'Saint-Télesphore': 4,
    'Saint-Urbain-Premier': 5,
    'Sainte-Anne-des-Plaines': 7,
    'Sainte-Clotilde-de-Châteauguay': 2,
    'Sainte-Clotilde-de-Châteauguay_01': 17,
    'Sainte-Marthe': 13,
    'Sainte-Martine': 10,
    'Sainte-Martine_01': 27,
    'Sainte-Martine_02': 4,
    'Sainte-Martine_03': 11
    }

# %%

brffig = BRFFigure(lang='french', figsize=(8, 5))
projectdir = "D:/OneDrive/INRS/2021 - Bulletin/projets_gwhat_correction_baro"
figdir = "C:/Users/User/rsesq-bulletin/fiches/img_fonction_reponse_baro"
for region in regions:
    filename = "correction_baro_{}.gwt".format(region)
    projet = ProjetReader(osp.join(projectdir, filename))

    for name in projet.wldsets:
        index = chosen_brf_indexes[name]
        if index is None:
            continue
        wldset = projet.get_wldset(name)
        brfdata = wldset.get_brf(wldset.get_brfname_at(index))
        wellid = wldset['Well ID']
        ffname = "fonction_reponse_baro_{}.pdf".format(wellid)
        brffig.plot(brfdata, wellid, msize=5, draw_line=False)

        brffig.savefig(osp.join(figdir, ffname), dpi=300)

    projet.close()
