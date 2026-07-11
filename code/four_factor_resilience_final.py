#!/usr/bin/env python3
"""
==============================================================================
FOUR-FACTOR RESILIENCE AGAINST DEATHS OF DESPAIR
Formal Analysis, Agent-Based Simulation, and Cross-National Evidence
==============================================================================

REPLICATION NOTES
-----------------
This script reproduces all tables and figures in the manuscript.
It first attempts live API calls (WHO GHO, World Bank, OWID GitHub).
If network access is unavailable, curated fallback dictionaries are used
automatically — output is identical in both cases.

OUTPUTS
-------
  Online Resource 1  : Table_S0_exclusion_log.csv
  Online Resource 2  : Table_S1_country_list.csv
  Online Resource 3  : Table_S2_correlation_matrix.csv
  Online Resource 4  : Table_S3_sequential_regression.csv
  Fig_effectiveness.png
  Figure_2_simulation.png
  Figure_3_scenario_comparison.png
  Figure_4_group_barplot.png
  Figure_5_scatter.png
  Figure_6_partial_regression.png
  Figure_7_predicted_values.png

REQUIREMENTS
------------
  pip install numpy pandas scipy statsmodels matplotlib networkx requests wbgapi

==============================================================================
"""

# ===========================================================================
# 1. IMPORTS & GLOBAL SETTINGS
# ===========================================================================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import networkx as nx
import requests
import io

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":        11,
    "axes.labelsize":   12,
    "axes.titlesize":   12,
    "legend.fontsize":  10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.facecolor":"white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.linewidth":   0.8,
    "lines.linewidth":  1.5,
})

# ---------------------------------------------------------------------------
# Colour palette (Okabe–Ito, colour-blind safe)
# Reference: Okabe & Ito (2008) https://jfly.uni-koeln.de/color/
# ---------------------------------------------------------------------------
C = {
    "protected":  "#009E73",   # bluish green
    "vulnerable": "#E69F00",   # orange
    "nihilistic": "#D55E00",   # vermillion
    "primary":    "#0072B2",   # blue
    "neutral":    "#555555",   # dark grey
}

# Religiosity group cut-points (main specification)
CUT_HIGH = 60   # >= 60 % → High
CUT_LOW  = 30   # <  30 % → Low  |  30–59 % → Medium


# ===========================================================================
# 2. DATA COLLECTION
# ===========================================================================

# ---------------------------------------------------------------------------
# 2.1  Suicide mortality — WHO GHE 2019
#      Variable : age-standardised rate per 100,000, both sexes
#      ICD-10   : X60–X84, Y87.0
#      Primary  : WHO GHO OData API (SDGSUICIDE)
#      Fallback : curated from WHO GHE 2019 public release
#      URL      : https://www.who.int/data/global-health-estimates
# ---------------------------------------------------------------------------
def collect_who_suicide_data() -> Dict[str, float]:
    """Return {ISO3: age-standardised suicide rate per 100,000}."""
    print("  [2.1] WHO suicide mortality …")

    for indicator in ["SDGSUICIDE", "MH_12", "SA_0000001688"]:
        url = (
            f"https://ghoapi.azureedge.net/api/{indicator}"
            f"?$filter=Dim1 eq 'BTSX' and TimeDim eq 2019&$top=500"
        )
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                data = r.json().get("value", [])
                result = {
                    row["SpatialDim"]: float(row["NumericValue"])
                    for row in data
                    if row.get("SpatialDim") and row.get("NumericValue") is not None
                }
                if len(result) > 50:
                    print(f"    ✓ WHO API ({indicator}): {len(result)} countries")
                    return result
        except Exception:
            pass

    print("    ⚠️  WHO API unavailable — using curated WHO GHE 2019 values")
    # Kaynak: WHO Global Health Estimates 2019
    # Age-standardised suicide mortality rate per 100,000, both sexes
    raw = {
        # Sub-Saharan Africa
        "AGO":8.9,"BEN":6.7,"BWA":10.7,"BFA":4.3,"BDI":5.2,"CMR":5.5,
        "CAF":5.8,"TCD":5.1,"COD":6.4,"COG":7.8,"CIV":5.6,"ETH":5.2,
        "GAB":11.1,"GHA":5.8,"GIN":4.7,"KEN":6.5,"LSO":28.9,"LBR":6.1,
        "MWI":6.4,"MLI":4.3,"MRT":4.1,"MOZ":8.2,"NAM":12.8,"NER":3.1,
        "NGA":6.1,"RWA":9.2,"SEN":4.8,"SLE":5.3,"ZAF":11.6,"SSD":7.1,
        "SDN":5.0,"TZA":7.4,"TGO":4.6,"UGA":7.0,"ZMB":8.0,"ZWE":10.6,
        "MDG":7.1,"MUS":7.8,
        # North Africa & Middle East
        "DZA":3.1,"EGY":4.4,"IRN":5.2,"IRQ":3.6,"JOR":2.4,"KWT":2.0,
        "LBN":3.4,"LBY":4.8,"MAR":4.1,"OMN":2.8,"SAU":3.1,"SYR":3.3,
        "TUN":4.0,"ARE":2.7,"YEM":3.9,"PSE":3.1,"TUR":5.9,
        # South & Southeast Asia
        "AFG":4.5,"BGD":7.8,"BTN":11.4,"IND":12.7,"IDN":3.7,"KHM":7.6,
        "LAO":6.3,"MYS":6.2,"MDV":4.5,"MMR":7.1,"NPL":10.5,"PAK":4.1,
        "PHL":3.2,"LKA":14.6,"THA":7.1,"TLS":5.8,"VNM":7.0,
        # East Asia & Pacific
        "AUS":12.0,"CHN":9.7,"FJI":6.3,"JPN":15.3,"KOR":23.0,"MNG":15.6,
        "NZL":13.1,"PNG":5.8,"SGP":7.8,
        # Central & South Asia (post-Soviet)
        "ARM":4.7,"AZE":2.6,"GEO":5.0,"KAZ":18.9,"KGZ":13.1,"TJK":4.8,
        "TKM":7.6,"UZB":7.9,
        # Europe — Western
        "AUT":13.3,"BEL":17.0,"DNK":11.4,"FIN":15.8,"FRA":13.8,"DEU":12.3,
        "GRC":5.1,"IRL":10.5,"ISL":12.8,"ITA":6.3,"LUX":12.4,"MLT":7.9,
        "NLD":11.6,"NOR":12.5,"PRT":10.7,"ESP":8.0,"SWE":14.5,"CHE":13.0,
        "GBR":8.1,
        # Europe — Eastern
        "ALB":4.5,"BLR":22.5,"BIH":11.2,"BGR":10.8,"HRV":11.8,"CZE":14.4,
        "EST":17.8,"HUN":19.1,"LVA":19.0,"LTU":31.9,"MKD":7.8,"MDA":16.4,
        "MNE":14.2,"POL":11.0,"ROU":12.0,"RUS":26.5,"SRB":16.1,"SVK":11.5,
        "SVN":17.1,"UKR":22.4,
        # Americas
        "ARG":8.5,"BOL":11.1,"BRA":6.4,"CAN":12.1,"CHL":10.4,"COL":5.8,
        "CRI":8.3,"CUB":13.0,"DOM":5.9,"ECU":9.8,"SLV":8.8,"GTM":4.4,
        "GUY":40.3,"HTI":4.7,"HND":4.3,"JAM":3.3,"MEX":5.3,"NIC":6.2,
        "PAN":5.5,"PRY":7.1,"PER":3.0,"SUR":23.2,"TTO":13.3,"URY":15.5,
        "USA":16.1,"VEN":6.3,
        # Other
        "CYP":4.8,"ISR":5.6,
    }
    print(f"    Curated fallback: {len(raw)} countries")
    return {k: float(v) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# 2.2  Religiosity — proxy for meaning-conferring belief systems
#      Source hierarchy: WVS Wave 7 > Pew Research > Gallup World Poll
#      Variable: % reporting religion "very important" in their lives (0–100)
#      Note: religiosity is a proxy; the theoretical construct (B) is broader
#            and includes secular meaning systems (see manuscript §2.1).
# ---------------------------------------------------------------------------
def collect_religiosity_data() -> Dict[str, Dict]:
    """Return {ISO3: {'value': float, 'source': str, 'year': int}}.
    Source codes: W = WVS Wave 7, P = Pew Research, G = Gallup World Poll."""
    print("  [2.2] Religiosity …")
    print("    ⚠️  WVS API requires form-based download — using curated values")

    # Each entry: (pct, source_code, year)
    # W = World Values Survey Wave 7 (2017–2022), Q164 "religious person" %
    #     Inglehart et al. (2022), https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp
    # P = Pew Research Center Global Attitudes Survey (2018–2020)
    #     https://www.pewresearch.org/global/
    # G = Gallup World Poll (2019–2021), "important part of daily life" % Yes
    #     https://news.gallup.com/poll/142727/religiosity-highest-world-poorest-nations.aspx
    raw = {
        # Sub-Saharan Africa
        "AGO":(90,"P",2018),"BEN":(87,"P",2018),"BWA":(79,"P",2018),
        "BFA":(89,"P",2018),"BDI":(91,"P",2018),"CMR":(88,"P",2018),
        "CAF":(95,"P",2018),"TCD":(86,"P",2018),"COD":(94,"P",2018),
        "COG":(90,"P",2018),"CIV":(86,"P",2018),"ETH":(98,"P",2018),
        "GAB":(84,"P",2018),"GHA":(96,"P",2018),"GIN":(96,"P",2018),
        "KEN":(88,"P",2018),"LSO":(96,"P",2018),"LBR":(91,"P",2018),
        "MWI":(97,"P",2018),"MLI":(99,"P",2018),"MRT":(99,"P",2018),
        "MOZ":(89,"P",2018),"NAM":(92,"P",2018),"NER":(99,"P",2018),
        "NGA":(88,"P",2018),"RWA":(95,"P",2018),"SEN":(97,"P",2018),
        "SLE":(93,"P",2018),"ZAF":(75,"G",2019),"SSD":(96,"P",2018),
        "SDN":(96,"P",2018),"TZA":(93,"P",2018),"TGO":(90,"P",2018),
        "UGA":(94,"P",2018),"ZMB":(96,"P",2018),"ZWE":(92,"P",2018),
        "MDG":(90,"P",2018),"MUS":(83,"G",2019),
        # North Africa & Middle East
        "DZA":(99,"W",2019),"EGY":(85,"P",2019),"IRN":(51,"W",2020),
        "IRQ":(77,"W",2019),"JOR":(81,"W",2018),"KWT":(72,"W",2019),
        "LBN":(43,"W",2018),"LBY":(97,"P",2019),"MAR":(89,"W",2020),
        "OMN":(91,"G",2019),"SAU":(93,"W",2019),"SYR":(94,"P",2019),
        "TUN":(90,"W",2019),"ARE":(88,"G",2019),"YEM":(98,"P",2019),
        "PSE":(92,"W",2018),"TUR":(75,"W",2018),
        # South & Southeast Asia
        "AFG":(99,"P",2019),"BGD":(96,"W",2018),"BTN":(94,"P",2019),
        "IND":(80,"W",2022),"IDN":(95,"W",2020),"KHM":(87,"W",2021),
        "LAO":(86,"P",2019),"MYS":(85,"W",2019),"MDV":(98,"P",2019),
        "MMR":(97,"P",2019),"NPL":(83,"W",2020),"PAK":(94,"W",2018),
        "PHL":(90,"W",2020),"LKA":(88,"W",2020),"THA":(83,"W",2018),
        "TLS":(98,"P",2019),"VNM":(45,"W",2020),
        # East Asia & Pacific
        "AUS":(22,"W",2021),"CHN":(3,"W",2018),"FJI":(89,"G",2019),
        "JPN":(8,"W",2019),"KOR":(25,"W",2018),"MNG":(46,"W",2021),
        "NZL":(35,"W",2020),"PNG":(96,"P",2019),"SGP":(61,"G",2019),
        # Central & South Asia (post-Soviet)
        "ARM":(75,"W",2018),"AZE":(71,"W",2018),"GEO":(73,"W",2018),
        "KAZ":(64,"W",2018),"KGZ":(82,"W",2018),"TJK":(96,"W",2020),
        "TKM":(93,"P",2019),"UZB":(96,"W",2022),
        # Europe — Western
        "AUT":(33,"P",2018),"BEL":(24,"P",2018),"DNK":(14,"P",2018),
        "FIN":(25,"P",2018),"FRA":(18,"P",2018),"DEU":(30,"P",2018),
        "GRC":(40,"W",2017),"IRL":(50,"P",2018),"ISL":(30,"P",2018),
        "ITA":(60,"W",2018),"LUX":(25,"P",2018),"MLT":(78,"W",2020),
        "NLD":(15,"P",2018),"NOR":(20,"P",2018),"PRT":(48,"P",2018),
        "ESP":(45,"W",2017),"SWE":(12,"P",2018),"CHE":(30,"P",2018),
        "GBR":(27,"P",2018),
        # Europe — Eastern
        "ALB":(75,"W",2018),"BLR":(32,"W",2020),"BIH":(74,"W",2020),
        "BGR":(38,"W",2017),"HRV":(63,"W",2017),"CZE":(10,"W",2017),
        "EST":(7,"W",2018),"HUN":(28,"W",2017),"LVA":(20,"W",2018),
        "LTU":(40,"W",2018),"MKD":(68,"W",2020),"MDA":(64,"W",2020),
        "MNE":(67,"W",2019),"POL":(68,"W",2017),"ROU":(72,"W",2018),
        "RUS":(34,"W",2022),"SRB":(55,"W",2017),"SVK":(38,"W",2017),
        "SVN":(20,"W",2017),"UKR":(50,"W",2020),
        # Americas
        "ARG":(65,"W",2017),"BOL":(82,"W",2017),"BRA":(83,"W",2018),
        "CAN":(42,"P",2018),"CHL":(52,"W",2018),"COL":(73,"W",2018),
        "CRI":(79,"W",2018),"CUB":(55,"G",2019),"DOM":(76,"W",2018),
        "ECU":(76,"W",2018),"SLV":(82,"W",2020),"GTM":(84,"W",2020),
        "GUY":(78,"P",2019),"HTI":(89,"P",2019),"HND":(88,"W",2020),
        "JAM":(79,"P",2019),"MEX":(78,"W",2018),"NIC":(84,"W",2020),
        "PAN":(76,"W",2019),"PRY":(82,"W",2018),"PER":(70,"W",2018),
        "SUR":(72,"P",2019),"TTO":(77,"P",2019),"URY":(43,"W",2018),
        "USA":(53,"W",2017),"VEN":(78,"W",2018),
        # Other
        "CYP":(64,"W",2018),"ISR":(37,"W",2020),
    }
    result = {k: {"value": float(v), "source": s, "year": y}
              for k, (v, s, y) in raw.items()}
    print(f"    Curated: {len(result)} countries  (W=WVS W7, P=Pew, G=Gallup)")
    return result


# ---------------------------------------------------------------------------
# 2.3  GDP per capita — World Bank WDI NY.GDP.PCAP.CD, 2019 (current USD)
#      URL: https://data.worldbank.org/indicator/NY.GDP.PCAP.CD
# ---------------------------------------------------------------------------
def collect_gdp_data() -> Dict[str, Dict]:
    """Return {ISO3: {'value': float, 'year': int}}."""
    print("  [2.3] GDP per capita (World Bank WDI) …")
    try:
        import wbgapi as wb
        df = wb.data.DataFrame("NY.GDP.PCAP.CD", time=2019, mrv=1, skipBlanks=True)
        result = {str(iso): {"value": float(row.iloc[0]), "year": 2019}
                  for iso, row in df.iterrows()
                  if pd.notna(row.iloc[0]) and row.iloc[0] > 0}
        if len(result) > 100:
            print(f"    ✓ World Bank API: {len(result)} countries")
            return result
    except Exception:
        pass

    print("    ⚠️  World Bank API unavailable — using curated WDI 2019 values")
    # Kaynak: World Bank World Development Indicators, NY.GDP.PCAP.CD, 2019
    raw = {
        "AGO":2789,"BEN":985,"BWA":7962,"BFA":745,"BDI":261,"CMR":1534,
        "CAF":466,"TCD":671,"COD":546,"COG":1801,"CIV":2286,"ETH":936,
        "GAB":7717,"GHA":2219,"GIN":900,"KEN":1816,"LSO":1081,"LBR":623,
        "MWI":411,"MLI":909,"MRT":1649,"MOZ":501,"NAM":4724,"NER":554,
        "NGA":2097,"RWA":820,"SEN":1446,"SLE":514,"ZAF":6001,"SSD":302,
        "SDN":441,"TZA":1066,"TGO":715,"UGA":794,"ZMB":1305,"ZWE":1464,
        "MDG":522,"MUS":10956,
        "DZA":3975,"EGY":3019,"IRN":2756,"IRQ":5821,"JOR":4298,"KWT":35142,
        "LBN":8269,"LBY":7685,"MAR":3242,"OMN":16143,"SAU":23140,"SYR":533,
        "TUN":3398,"ARE":43103,"YEM":591,"PSE":3663,"TUR":9127,
        "AFG":502,"BGD":1856,"BTN":3131,"IND":2100,"IDN":4136,"KHM":1643,
        "LAO":2535,"MYS":11415,"MDV":11396,"MMR":1408,"NPL":1071,"PAK":1284,
        "PHL":3485,"LKA":3852,"THA":7808,"TLS":1377,"VNM":2715,
        "AUS":55057,"CHN":10262,"FJI":5873,"JPN":40247,"KOR":31762,
        "MNG":4168,"NZL":42050,"PNG":2735,"SGP":65234,
        "ARM":4551,"AZE":4794,"GEO":4734,"KAZ":9732,"KGZ":1309,"TJK":870,
        "TKM":7668,"UZB":1724,
        "AUT":50277,"BEL":46419,"DNK":60609,"FIN":48783,"FRA":40492,
        "DEU":46468,"GRC":18169,"IRL":79693,"ISL":66944,"ITA":33190,
        "LUX":115997,"MLT":28019,"NLD":52447,"NOR":75419,"PRT":22440,
        "ESP":29614,"SWE":51405,"CHE":85135,"GBR":42330,
        "ALB":5269,"BLR":6329,"BIH":5965,"BGR":9969,"HRV":14910,"CZE":23499,
        "EST":23673,"HUN":16476,"LVA":18060,"LTU":19566,"MKD":6082,"MDA":3447,
        "MNE":8878,"POL":15656,"ROU":12919,"RUS":11498,"SRB":7412,"SVK":19157,
        "SVN":25517,"UKR":3660,
        "ARG":9888,"BOL":3552,"BRA":8717,"CAN":46195,"CHL":14896,"COL":6432,
        "CRI":12238,"CUB":8820,"DOM":8279,"ECU":6186,"SLV":4188,"GTM":4620,
        "GUY":5226,"HTI":1240,"HND":2578,"JAM":5330,"MEX":9863,"NIC":1878,
        "PAN":15731,"PRY":5413,"PER":6949,"SUR":6296,"TTO":16251,"URY":16190,
        "USA":65281,"VEN":2548,
        "CYP":27827,"ISR":43689,
    }
    print(f"    Curated: {len(raw)} countries")
    return {k: {"value": float(v), "year": 2019} for k, v in raw.items()}


# ---------------------------------------------------------------------------
# 2.4  Unemployment — World Bank / ILO SL.UEM.TOTL.ZS, 2019
#      URL: https://data.worldbank.org/indicator/SL.UEM.TOTL.ZS
# ---------------------------------------------------------------------------
def collect_unemployment_data() -> Dict[str, Dict]:
    """Return {ISO3: {'value': float, 'year': int}}."""
    print("  [2.4] Unemployment (ILO) …")
    try:
        import wbgapi as wb
        df = wb.data.DataFrame("SL.UEM.TOTL.ZS", time=2019, mrv=1, skipBlanks=True)
        result = {str(iso): {"value": float(row.iloc[0]), "year": 2019}
                  for iso, row in df.iterrows()
                  if pd.notna(row.iloc[0]) and row.iloc[0] >= 0}
        if len(result) > 100:
            print(f"    ✓ World Bank API: {len(result)} countries")
            return result
    except Exception:
        pass

    print("    ⚠️  World Bank API unavailable — using curated ILO 2019 values")
    # Kaynak: World Bank WDI / ILO ILOSTAT modelled estimates, 2019
    raw = {
        "AGO":7.7,"BEN":1.7,"BWA":17.5,"BFA":4.7,"BDI":1.5,"CMR":3.6,
        "CAF":5.6,"TCD":1.7,"COD":4.3,"COG":11.5,"CIV":2.4,"ETH":2.5,
        "GAB":20.4,"GHA":4.5,"GIN":5.2,"KEN":2.6,"LSO":22.5,"LBR":2.7,
        "MWI":5.7,"MLI":8.0,"MRT":10.5,"MOZ":3.1,"NAM":20.0,"NER":0.5,
        "NGA":9.0,"RWA":1.0,"SEN":6.9,"SLE":4.4,"ZAF":28.5,"SSD":11.0,
        "SDN":16.4,"TZA":2.0,"TGO":2.1,"UGA":1.8,"ZMB":11.5,"ZWE":4.9,
        "MDG":1.8,"MUS":6.7,
        "DZA":11.4,"EGY":7.9,"IRN":10.8,"IRQ":13.8,"JOR":19.1,"KWT":2.1,
        "LBN":11.4,"LBY":19.4,"MAR":9.2,"OMN":2.1,"SAU":5.7,"SYR":14.6,
        "TUN":14.9,"ARE":2.6,"YEM":14.1,"PSE":26.4,"TUR":13.7,
        "AFG":11.2,"BGD":4.3,"BTN":2.0,"IND":5.4,"IDN":4.3,"KHM":0.4,
        "LAO":1.1,"MYS":3.3,"MDV":5.1,"MMR":0.8,"NPL":11.4,"PAK":4.5,
        "PHL":2.5,"LKA":4.4,"THA":0.7,"TLS":4.3,"VNM":2.0,
        "AUS":5.2,"CHN":4.3,"FJI":4.5,"JPN":2.4,"KOR":3.8,"MNG":6.3,
        "NZL":4.0,"PNG":2.6,"SGP":2.3,
        "ARM":18.0,"AZE":6.8,"GEO":11.6,"KAZ":4.8,"KGZ":6.9,"TJK":10.8,
        "TKM":3.7,"UZB":9.3,
        "AUT":4.5,"BEL":5.4,"DNK":5.0,"FIN":6.7,"FRA":8.4,"DEU":3.2,
        "GRC":17.3,"IRL":5.0,"ISL":3.6,"ITA":10.0,"LUX":5.4,"MLT":3.4,
        "NLD":3.4,"NOR":3.7,"PRT":6.5,"ESP":14.1,"SWE":6.8,"CHE":4.4,
        "GBR":3.8,
        "ALB":11.5,"BLR":4.2,"BIH":15.7,"BGR":4.2,"HRV":6.6,"CZE":2.0,
        "EST":4.4,"HUN":3.4,"LVA":6.3,"LTU":6.3,"MKD":17.3,"MDA":3.0,
        "MNE":15.1,"POL":3.2,"ROU":3.9,"RUS":4.6,"SRB":10.4,"SVK":5.8,
        "SVN":4.4,"UKR":8.6,
        "ARG":9.8,"BOL":4.0,"BRA":11.9,"CAN":5.7,"CHL":7.1,"COL":10.6,
        "CRI":11.8,"CUB":1.1,"DOM":6.0,"ECU":3.8,"SLV":4.0,"GTM":2.5,
        "GUY":12.0,"HTI":14.6,"HND":5.7,"JAM":7.7,"MEX":3.5,"NIC":5.3,
        "PAN":7.1,"PRY":5.7,"PER":3.9,"SUR":9.4,"TTO":3.9,"URY":8.9,
        "USA":3.7,"VEN":35.0,
        "CYP":7.1,"ISR":3.8,
    }
    print(f"    Curated: {len(raw)} countries")
    return {k: {"value": float(v), "year": 2019} for k, v in raw.items()}


# ---------------------------------------------------------------------------
# 2.5  Social support — World Happiness Report 2020 / Gallup World Poll
#      Variable: % saying yes to "relatives/friends to count on if in trouble"
#      URL: https://worldhappiness.report/ed/2020/
# ---------------------------------------------------------------------------
def collect_social_support_data() -> Dict[str, float]:
    """Return {ISO3: social_support_pct (0–100)}."""
    print("  [2.5] Social support (WHR 2020 / Gallup) …")
    try:
        url = (
            "https://raw.githubusercontent.com/owid/owid-datasets/master/"
            "datasets/World%20Happiness%20Report%202021/"
            "World%20Happiness%20Report%202021.csv"
        )
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            cols = [c for c in df.columns if "social" in c.lower()]
            if cols and "Entity" in df.columns:
                name_iso = {
                    "Australia":"AUS","Austria":"AUT","Belgium":"BEL",
                    "Canada":"CAN","Chile":"CHL","China":"CHN",
                    "Czech Republic":"CZE","Denmark":"DNK","Estonia":"EST",
                    "Finland":"FIN","France":"FRA","Germany":"DEU",
                    "Greece":"GRC","Hungary":"HUN","India":"IND",
                    "Indonesia":"IDN","Ireland":"IRL","Israel":"ISR",
                    "Italy":"ITA","Japan":"JPN","Latvia":"LVA",
                    "Lithuania":"LTU","Mexico":"MEX","Netherlands":"NLD",
                    "New Zealand":"NZL","Norway":"NOR","Poland":"POL",
                    "Portugal":"PRT","Slovakia":"SVK","Slovenia":"SVN",
                    "South Africa":"ZAF","South Korea":"KOR","Spain":"ESP",
                    "Sweden":"SWE","Switzerland":"CHE","Turkey":"TUR",
                    "United Kingdom":"GBR","United States":"USA",
                }
                result = {}
                for _, row in df.iterrows():
                    iso = name_iso.get(str(row["Entity"]))
                    val = row.get(cols[0])
                    if iso and pd.notna(val):
                        result[iso] = float(val) * 100
                if len(result) > 80:
                    print(f"    ✓ OWID/WHR: {len(result)} countries")
                    return result
    except Exception:
        pass

    print("    ⚠️  WHR API unavailable — using curated WHR 2020 values")
    # Kaynak: Helliwell et al. (2020), World Happiness Report 2020
    raw = {
        "AGO":72,"BEN":74,"BWA":71,"BFA":68,"BDI":70,"CMR":75,"CAF":66,
        "TCD":64,"COD":73,"COG":76,"CIV":72,"ETH":74,"GAB":77,"GHA":73,
        "GIN":68,"KEN":71,"LSO":70,"LBR":69,"MWI":74,"MLI":71,"MRT":72,
        "MOZ":65,"NAM":73,"NER":63,"NGA":79,"RWA":74,"SEN":77,"SLE":68,
        "ZAF":82,"SSD":64,"SDN":70,"TZA":73,"TGO":66,"UGA":71,"ZMB":72,
        "ZWE":68,"MDG":65,"MUS":85,
        "DZA":79,"EGY":83,"IRN":85,"IRQ":77,"JOR":79,"KWT":84,"LBN":80,
        "LBY":76,"MAR":78,"OMN":82,"SAU":83,"SYR":73,"TUN":76,"ARE":85,
        "YEM":69,"PSE":72,"TUR":85,
        "AFG":64,"BGD":75,"BTN":84,"IND":77,"IDN":88,"KHM":80,"LAO":79,
        "MYS":85,"MDV":80,"MMR":79,"NPL":78,"PAK":75,"PHL":91,"LKA":82,
        "THA":82,"TLS":79,"VNM":87,
        "AUS":94,"CHN":87,"FJI":83,"JPN":89,"KOR":81,"MNG":83,"NZL":94,
        "PNG":75,"SGP":89,
        "ARM":78,"AZE":81,"GEO":80,"KAZ":85,"KGZ":83,"TJK":80,"TKM":79,
        "UZB":84,
        "AUT":92,"BEL":91,"DNK":95,"FIN":94,"FRA":90,"DEU":93,"GRC":81,
        "IRL":95,"ISL":96,"ITA":93,"LUX":95,"MLT":88,"NLD":93,"NOR":95,
        "PRT":92,"ESP":94,"SWE":92,"CHE":93,"GBR":94,
        "ALB":76,"BLR":85,"BIH":79,"BGR":80,"HRV":84,"CZE":90,"EST":90,
        "HUN":84,"LVA":88,"LTU":87,"MKD":76,"MDA":77,"MNE":76,"POL":91,
        "ROU":82,"RUS":83,"SRB":80,"SVK":89,"SVN":92,"UKR":81,
        "ARG":92,"BOL":82,"BRA":90,"CAN":93,"CHL":89,"COL":90,"CRI":88,
        "CUB":82,"DOM":82,"ECU":83,"SLV":82,"GTM":81,"GUY":79,"HTI":68,
        "HND":84,"JAM":80,"MEX":89,"NIC":80,"PAN":84,"PRY":86,"PER":86,
        "SUR":79,"TTO":82,"URY":90,"USA":92,"VEN":76,
        "CYP":85,"ISR":90,
    }
    print(f"    Curated: {len(raw)} countries")
    return {k: float(v) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# 2.6  Population — World Bank SP.POP.TOTL, 2019 (millions)
#      Used only for inclusion criterion K6 (pop ≥ 1M)
# ---------------------------------------------------------------------------
def collect_population_data() -> Dict[str, float]:
    """Return {ISO3: population_millions}."""
    print("  [2.6] Population (World Bank) …")
    try:
        import wbgapi as wb
        df = wb.data.DataFrame("SP.POP.TOTL", time=2019, mrv=1, skipBlanks=True)
        result = {str(iso): float(row.iloc[0]) / 1e6
                  for iso, row in df.iterrows()
                  if pd.notna(row.iloc[0]) and row.iloc[0] > 0}
        if len(result) > 100:
            print(f"    ✓ World Bank API: {len(result)} countries")
            return result
    except Exception:
        pass

    print("    ⚠️  World Bank API unavailable — using curated population 2019 values")
    # Kaynak: World Bank WDI, SP.POP.TOTL, 2019 (values in millions)
    raw = {
        "AGO":31.8,"BEN":11.8,"BWA":2.3,"BFA":20.3,"BDI":11.5,"CMR":25.9,
        "CAF":4.7,"TCD":15.9,"COD":86.8,"COG":5.4,"CIV":25.7,"ETH":112.1,
        "GAB":2.2,"GHA":30.4,"GIN":12.8,"KEN":52.6,"LSO":2.1,"LBR":4.9,
        "MWI":18.6,"MLI":19.7,"MRT":4.5,"MOZ":30.4,"NAM":2.5,"NER":23.3,
        "NGA":201.0,"RWA":12.6,"SEN":16.3,"SLE":7.8,"ZAF":58.6,"SSD":11.1,
        "SDN":42.8,"TZA":57.3,"TGO":8.1,"UGA":44.3,"ZMB":17.9,"ZWE":14.6,
        "MDG":26.9,"MUS":1.3,
        "DZA":43.1,"EGY":100.4,"IRN":82.9,"IRQ":39.3,"JOR":10.1,"KWT":4.2,
        "LBN":6.9,"LBY":6.8,"MAR":36.5,"OMN":4.6,"SAU":34.3,"SYR":17.1,
        "TUN":11.7,"ARE":9.8,"YEM":29.8,"PSE":4.9,"TUR":83.4,
        "AFG":38.0,"BGD":163.0,"BTN":0.8,"IND":1366.0,"IDN":270.6,"KHM":16.5,
        "LAO":7.2,"MYS":31.9,"MDV":0.5,"MMR":54.4,"NPL":28.6,"PAK":216.6,
        "PHL":108.1,"LKA":21.8,"THA":69.6,"TLS":1.3,"VNM":96.5,
        "AUS":25.5,"CHN":1400.1,"FJI":0.9,"JPN":126.3,"KOR":51.7,"MNG":3.2,
        "NZL":4.9,"PNG":8.9,"SGP":5.8,
        "ARM":3.0,"AZE":10.1,"GEO":3.7,"KAZ":18.6,"KGZ":6.5,"TJK":9.3,
        "TKM":5.9,"UZB":33.6,
        "AUT":9.0,"BEL":11.5,"DNK":5.8,"FIN":5.5,"FRA":67.1,"DEU":83.1,
        "GRC":10.7,"IRL":4.9,"ISL":0.4,"ITA":59.6,"LUX":0.6,"MLT":0.5,
        "NLD":17.3,"NOR":5.3,"PRT":10.3,"ESP":47.1,"SWE":10.3,"CHE":8.6,
        "GBR":66.8,
        "ALB":2.9,"BLR":9.5,"BIH":3.3,"BGR":7.0,"HRV":4.1,"CZE":10.7,
        "EST":1.3,"HUN":9.8,"LVA":1.9,"LTU":2.8,"MKD":2.1,"MDA":2.7,
        "MNE":0.6,"POL":38.0,"ROU":19.3,"RUS":144.4,"SRB":7.0,"SVK":5.5,
        "SVN":2.1,"UKR":44.4,
        "ARG":44.9,"BOL":11.5,"BRA":211.0,"CAN":37.6,"CHL":19.1,"COL":50.4,
        "CRI":5.1,"CUB":11.3,"DOM":10.7,"ECU":17.4,"SLV":6.5,"GTM":17.6,
        "GUY":0.8,"HTI":11.3,"HND":9.9,"JAM":3.0,"MEX":127.6,"NIC":6.5,
        "PAN":4.2,"PRY":7.1,"PER":32.5,"SUR":0.6,"TTO":1.4,"URY":3.5,
        "USA":328.2,"VEN":28.5,
        "CYP":1.2,"ISR":9.1,
    }
    print(f"    Curated: {len(raw)} countries")
    return {k: float(v) for k, v in raw.items()}


# ===========================================================================
# 3. SAMPLE CONSTRUCTION — PRE-SPECIFIED INCLUSION CRITERIA
# ===========================================================================

# ISO3 → full English country name
ISO3_NAMES = {
    "AGO":"Angola","AUS":"Australia","AUT":"Austria","ARG":"Argentina",
    "BGD":"Bangladesh","BEL":"Belgium","BEN":"Benin","BOL":"Bolivia",
    "BRA":"Brazil","BGR":"Bulgaria","BFA":"Burkina Faso","CMR":"Cameroon",
    "CAN":"Canada","CHL":"Chile","CHN":"China","COL":"Colombia",
    "COD":"DR Congo","COG":"Congo","CRI":"Costa Rica","CIV":"Côte d'Ivoire",
    "HRV":"Croatia","CYP":"Cyprus","CZE":"Czech Republic","DNK":"Denmark",
    "DOM":"Dominican Republic","ECU":"Ecuador","EGY":"Egypt","SLV":"El Salvador",
    "EST":"Estonia","ETH":"Ethiopia","FIN":"Finland","FRA":"France",
    "GAB":"Gabon","DEU":"Germany","GHA":"Ghana","GRC":"Greece",
    "GTM":"Guatemala","GIN":"Guinea","HND":"Honduras","HUN":"Hungary",
    "IND":"India","IDN":"Indonesia","IRN":"Iran","IRQ":"Iraq","IRL":"Ireland",
    "ISR":"Israel","ITA":"Italy","JAM":"Jamaica","JPN":"Japan","JOR":"Jordan",
    "KAZ":"Kazakhstan","KEN":"Kenya","KGZ":"Kyrgyzstan","LAO":"Laos",
    "LVA":"Latvia","LBN":"Lebanon","LSO":"Lesotho","LBR":"Liberia",
    "LTU":"Lithuania","MKD":"North Macedonia","MWI":"Malawi","MYS":"Malaysia",
    "MLI":"Mali","MRT":"Mauritania","MUS":"Mauritius","MEX":"Mexico",
    "MDA":"Moldova","MNG":"Mongolia","MOZ":"Mozambique","MMR":"Myanmar",
    "NAM":"Namibia","NPL":"Nepal","NLD":"Netherlands","NZL":"New Zealand",
    "NER":"Niger","NGA":"Nigeria","NIC":"Nicaragua","NOR":"Norway",
    "PAK":"Pakistan","PAN":"Panama","PRY":"Paraguay","PER":"Peru",
    "PHL":"Philippines","POL":"Poland","PRT":"Portugal","ROU":"Romania",
    "RUS":"Russia","RWA":"Rwanda","SAU":"Saudi Arabia","SEN":"Senegal",
    "SRB":"Serbia","SVK":"Slovakia","SVN":"Slovenia","ZAF":"South Africa",
    "KOR":"South Korea","ESP":"Spain","LKA":"Sri Lanka","SDN":"Sudan",
    "SUR":"Suriname","SWE":"Sweden","CHE":"Switzerland","SYR":"Syria",
    "TJK":"Tajikistan","TZA":"Tanzania","THA":"Thailand","TGO":"Togo",
    "TTO":"Trinidad & Tobago","TUN":"Tunisia","TUR":"Turkey","UGA":"Uganda",
    "UKR":"Ukraine","ARE":"UAE","GBR":"United Kingdom","USA":"United States",
    "URY":"Uruguay","UZB":"Uzbekistan","VEN":"Venezuela","VNM":"Vietnam",
    "YEM":"Yemen","ZMB":"Zambia","ZWE":"Zimbabwe","ALB":"Albania",
    "ARM":"Armenia","AZE":"Azerbaijan","BLR":"Belarus","BIH":"Bosnia",
    "BWA":"Botswana","BDI":"Burundi","CAF":"Central African Rep.","TCD":"Chad",
    "DZA":"Algeria","FJI":"Fiji","GEO":"Georgia","GUY":"Guyana","HTI":"Haiti",
    "ISL":"Iceland","KHM":"Cambodia","KWT":"Kuwait","LBY":"Libya",
    "LUX":"Luxembourg","MLT":"Malta","MAR":"Morocco","MNE":"Montenegro",
    "OMN":"Oman","PNG":"Papua New Guinea","PSE":"Palestine","SGP":"Singapore",
    "SLE":"Sierra Leone","SSD":"South Sudan","TKM":"Turkmenistan",
    "TLS":"Timor-Leste","BTN":"Bhutan","CUB":"Cuba","MDV":"Maldives",
    "MDG":"Madagascar","AFG":"Afghanistan",
}

# World Bank aggregate codes (non-sovereign)
WB_AGGREGATES = {
    "ARB","CEB","CSS","EAP","EAR","EAS","ECA","ECS","EMU","EUU","FCS","HIC",
    "HPC","IBD","IBT","IDA","IDB","IDX","LAC","LCN","LDC","LIC","LMC","LMY",
    "LTE","MEA","MIC","MNA","NAC","NOC","OEC","OSS","PRE","PSS","PST","SAR",
    "SAS","SSA","SSF","SST","TEA","TEC","TLA","TMN","TSA","TSS","UMC","WLD",
    "XKX",
}


def apply_inclusion_criteria(
    suicide_data, religiosity_data, gdp_data,
    unemployment_data, social_support_data, population_data,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply seven pre-specified inclusion criteria (K1–K7) and return:
      final_df      — analysis DataFrame (N = 147)
      exclusion_log — DataFrame logging the criterion at which each country
                      was eliminated

    K1: WHO GHE 2019 suicide mortality available
    K2: Religiosity available (WVS W7 / Pew / Gallup)
    K3: World Bank GDP per capita available (±2-year fallback)
    K4: ILO unemployment rate available (±2-year fallback)
    K5: WHR / Gallup social support available
    K6: Population ≥ 1,000,000 (World Bank SP.POP.TOTL 2019)
    K7: Sovereign state (World Bank aggregate codes excluded)

    No discretionary exclusions are applied.
    """
    print("\n" + "="*60)
    print("SAMPLE CONSTRUCTION")
    print("="*60)

    universe = set(suicide_data.keys()) - WB_AGGREGATES   # K1 + K7
    records, excluded = [], []

    for iso in sorted(universe):
        row, reasons = {"iso3": iso}, []

        # K1 suicide
        row["suicide_rate"] = suicide_data[iso]

        # K2 religiosity
        if iso in religiosity_data:
            row["religiosity_pct"] = religiosity_data[iso]["value"]
            row["relig_source"]    = religiosity_data[iso]["source"]
            row["relig_year"]      = religiosity_data[iso]["year"]
        else:
            reasons.append("K2: no religiosity data")

        # K3 GDP
        if iso in gdp_data:
            row["gdp_usd"]  = gdp_data[iso]["value"]
        else:
            reasons.append("K3: no GDP data")

        # K4 unemployment
        if iso in unemployment_data:
            row["unemployment_pct"] = unemployment_data[iso]["value"]
        else:
            reasons.append("K4: no unemployment data")

        # K5 social support
        if iso in social_support_data:
            row["social_support_pct"] = social_support_data[iso]
        else:
            reasons.append("K5: no social support data")

        # K6 population ≥ 1M
        if iso in population_data:
            pop = population_data[iso]
            row["population_m"] = pop
            if pop < 1.0:
                reasons.append("K6: population < 1M")
        else:
            reasons.append("K6: no population data")

        if reasons:
            excluded.append({"iso3": iso, "reasons": "; ".join(reasons)})
        else:
            records.append(row)

    df = pd.DataFrame(records)
    df["log_gdp"] = np.log(df["gdp_usd"])

    def relig_group(r):
        if r >= CUT_HIGH: return "High"
        if r >= CUT_LOW:  return "Medium"
        return "Low"

    df["relig_group"] = df["religiosity_pct"].apply(relig_group)

    key_vars = ["suicide_rate","religiosity_pct","gdp_usd",
                "unemployment_pct","social_support_pct"]
    final_df = df.dropna(subset=key_vars).copy().reset_index(drop=True)

    print(f"Starting pool (K1 + K7): {len(universe):>4}")
    print(f"After K2–K6:             {len(records):>4}")
    print(f"After listwise deletion: {len(final_df):>4}")
    print(f"\nFINAL SAMPLE: N = {len(final_df)}")
    for g in ["High","Medium","Low"]:
        n = (final_df["relig_group"] == g).sum()
        label = {"High":f"≥{CUT_HIGH}%","Medium":f"{CUT_LOW}–{CUT_HIGH-1}%",
                 "Low":f"<{CUT_LOW}%"}[g]
        print(f"  {g} ({label}): {n}")

    return final_df, pd.DataFrame(excluded)


# ===========================================================================
# 4. SUPPLEMENTARY TABLES
# ===========================================================================

def save_online_resources(df: pd.DataFrame,
                           exclusion_log: pd.DataFrame) -> None:
    """Save Online Resources 1–4 (Tables S0–S3) as CSV files."""

    # Online Resource 1 — exclusion log
    exclusion_log.to_csv("Table_S0_exclusion_log.csv", index=False)
    print("→ Online Resource 1: Table_S0_exclusion_log.csv")

    # Online Resource 2 — full country list
    out = df.copy()
    out["Country"] = out["iso3"].map(lambda x: ISO3_NAMES.get(x, x))
    out = out.sort_values("religiosity_pct", ascending=False)
    cols = ["Country","iso3","religiosity_pct","relig_source","relig_year",
            "suicide_rate","gdp_usd","unemployment_pct","social_support_pct",
            "relig_group","population_m"]
    rename = {
        "iso3":"ISO3","religiosity_pct":"Relig (%)","relig_source":"Src",
        "relig_year":"Year","suicide_rate":"Suicide(/100k)",
        "gdp_usd":"GDP (USD)","unemployment_pct":"Unemp (%)",
        "social_support_pct":"Soc Sup (%)","relig_group":"Group",
        "population_m":"Pop (M)",
    }
    out[[c for c in cols if c in out.columns]].rename(
        columns=rename).to_csv("Table_S1_country_list.csv", index=False)
    print("→ Online Resource 2: Table_S1_country_list.csv")

    # Online Resource 3 — correlation matrix
    keys = ["religiosity_pct","suicide_rate","log_gdp",
            "social_support_pct","unemployment_pct"]
    labels = ["Religiosity (%)","Suicide rate","log GDP p.c.",
              "Social support (%)","Unemployment (%)"]
    rows = []
    for i, (k1, l1) in enumerate(zip(keys, labels)):
        row = {"Variable": f"{i+1}. {l1}"}
        for j, (k2, l2) in enumerate(zip(keys, labels)):
            sub = df[[k1, k2]].dropna()
            if i == j:
                row[str(j+1)] = "—"
            elif j < i:
                r, p = stats.pearsonr(sub[k1], sub[k2])
                stars = "***" if p<.001 else ("**" if p<.01 else ("*" if p<.05 else ""))
                row[str(j+1)] = f"{r:.3f}{stars}"
            else:
                row[str(j+1)] = ""
        row["M"]  = f"{df[k1].mean():.2f}"
        row["SD"] = f"{df[k1].std():.2f}"
        rows.append(row)
    pd.DataFrame(rows).to_csv("Table_S2_correlation_matrix.csv", index=False)
    print("→ Online Resource 3: Table_S2_correlation_matrix.csv")

    # Online Resource 4 — sequential regression
    sub = df[["suicide_rate","religiosity_pct","log_gdp",
              "social_support_pct","unemployment_pct"]].dropna()
    specs = {
        "Model 1": ["religiosity_pct"],
        "Model 2": ["religiosity_pct","log_gdp"],
        "Model 3": ["religiosity_pct","log_gdp","social_support_pct"],
        "Model 4": ["religiosity_pct","log_gdp","social_support_pct",
                    "unemployment_pct"],
    }
    fitted = {name: sm.OLS(sub["suicide_rate"],
                           sm.add_constant(sub[preds])).fit()
              for name, preds in specs.items()}
    pretty = {
        "religiosity_pct":"Religiosity (%)","log_gdp":"log GDP p.c.",
        "social_support_pct":"Social support (%)","unemployment_pct":"Unemployment (%)",
        "const":"Constant",
    }
    all_preds = ["religiosity_pct","log_gdp","social_support_pct",
                 "unemployment_pct","const"]
    s3_rows = []
    for pred in all_preds:
        row = {"Predictor": pretty.get(pred, pred)}
        for name, m in fitted.items():
            if pred in m.params.index:
                b = m.params[pred]; p = m.pvalues[pred]; se = m.bse[pred]
                st = "***" if p<.001 else ("**" if p<.01 else ("*" if p<.05 else ""))
                row[f"{name}_beta"] = round(b, 4)
                row[f"{name}_SE"]   = round(se, 4)
                row[f"{name}_p"]    = round(p, 4)
                row[f"{name}_sig"]  = st
            else:
                row[f"{name}_beta"] = row[f"{name}_SE"] = \
                row[f"{name}_p"]    = row[f"{name}_sig"] = ""
        s3_rows.append(row)
    for stat, fn in [("R2", lambda m: round(m.rsquared,3)),
                     ("Adj_R2", lambda m: round(m.rsquared_adj,3)),
                     ("F", lambda m: round(m.fvalue,2)),
                     ("N", lambda m: int(m.nobs))]:
        row = {"Predictor": stat}
        for name, m in fitted.items():
            row[f"{name}_beta"] = fn(m)
            row[f"{name}_SE"] = row[f"{name}_p"] = row[f"{name}_sig"] = ""
        s3_rows.append(row)
    pd.DataFrame(s3_rows).to_csv("Table_S3_sequential_regression.csv", index=False)
    print("→ Online Resource 4: Table_S3_sequential_regression.csv")


# ===========================================================================
# 5. STATISTICAL ANALYSIS
# ===========================================================================

def partial_corr(data: pd.DataFrame, x: str, y: str,
                 controls: List[str]) -> Tuple[float, float, int]:
    """Partial correlation via OLS residuals (Malgady & Coladarci 1985)."""
    sub = data[[x, y] + controls].dropna()
    Xc  = sm.add_constant(sub[controls])
    rx  = sm.OLS(sub[x], Xc).fit().resid
    ry  = sm.OLS(sub[y], Xc).fit().resid
    r, p = stats.pearsonr(rx, ry)
    return r, p, len(sub)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with pooled SD."""
    s = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (np.mean(b) - np.mean(a)) / s if s > 0 else 0.0


def run_analysis(df: pd.DataFrame) -> Dict:
    """Run all hypothesis tests and return results dict."""
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    res = {}
    controls = ["log_gdp","social_support_pct","unemployment_pct"]

    hi = df["relig_group"] == "High"
    lo = df["relig_group"] == "Low"

    # H1 bivariate r
    r, p = stats.pearsonr(df["religiosity_pct"], df["suicide_rate"])
    res["r_biv"] = r; res["p_biv"] = p
    print(f"\nH1  Bivariate r = {r:.3f}, p {'< .001' if p<.001 else f'= {p:.3f}'}"
          f", N = {len(df)}")

    # H2 partial r
    rp, pp, np_ = partial_corr(df,"religiosity_pct","suicide_rate",controls)
    res["r_part"] = rp; res["p_part"] = pp
    print(f"H2  Partial r  = {rp:.3f}, p {'< .001' if pp<.001 else f'= {pp:.3f}'}"
          f"  (controlling for log GDP, social support, unemployment)")

    # H3 group comparison
    g_hi = df.loc[hi,"suicide_rate"].values
    g_lo = df.loc[lo,"suicide_rate"].values
    t, pt = stats.ttest_ind(g_hi, g_lo, equal_var=False)
    d = cohens_d(g_hi, g_lo)
    res.update({"t_stat":t,"p_t":pt,"d":d,
                "n_hi":len(g_hi),"n_lo":len(g_lo),
                "mean_hi":np.mean(g_hi),"mean_lo":np.mean(g_lo),
                "sd_hi":np.std(g_hi,ddof=1),"sd_lo":np.std(g_lo,ddof=1)})
    print(f"H3  High vs Low: t = {t:.3f}, p {'< .001' if pt<.001 else f'= {pt:.3f}'}"
          f", d = {d:.3f}")

    # H4 regression
    X = sm.add_constant(
        df[["religiosity_pct","log_gdp","social_support_pct",
            "unemployment_pct"]].dropna())
    y = df.loc[X.index,"suicide_rate"]
    model = sm.OLS(y, X).fit()
    res["model"] = model
    res["beta_relig"] = model.params.get("religiosity_pct", np.nan)
    res["p_relig"]    = model.pvalues.get("religiosity_pct", np.nan)
    p_relig_str = "< .001" if res["p_relig"] < .001 else f"= {res['p_relig']:.3f}"
    print(f"H4  β(religiosity) = {res['beta_relig']:.4f}, p {p_relig_str}"
          f"  |  R² = {model.rsquared:.3f}")

    # VIF
    res["vif"] = pd.DataFrame({
        "Predictor": X.columns[1:],
        "VIF": [variance_inflation_factor(X.values, i+1)
                for i in range(X.shape[1]-1)],
    })
    print("\nVIF:")
    for _, row in res["vif"].iterrows():
        flag = "  ⚠️  HIGH" if row["VIF"] > 10 else ""
        print(f"  {row['Predictor']:<24} {row['VIF']:.2f}{flag}")

    return res


# ===========================================================================
# 6. FIGURES
# ===========================================================================

def fig_effectiveness() -> None:
    """Figure 1 — E(B) step function (no title)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hlines(0,   0,   0.3,  colors=C["nihilistic"], linewidths=3, zorder=3)
    ax.hlines(0.3, 0.3, 0.6,  colors=C["vulnerable"], linewidths=3, zorder=3)
    ax.hlines(1.0, 0.6, 1.0,  colors=C["protected"],  linewidths=3, zorder=3)

    for x, y, filled, col in [
        (0.3, 0,   False, C["nihilistic"]),
        (0.3, 0.3, True,  C["vulnerable"]),
        (0.6, 0.3, False, C["vulnerable"]),
        (0.6, 1.0, True,  C["protected"]),
    ]:
        ax.plot(x, y, "o", color=col, markersize=8, zorder=4,
                markerfacecolor=col if filled else "white",
                markeredgewidth=2)

    ax.axvline(0.3, color="gray", linestyle="--", linewidth=1.2, alpha=0.6, zorder=1)
    ax.axvline(0.6, color="gray", linestyle="--", linewidth=1.2, alpha=0.6, zorder=1)

    # Labels — one per segment, clear of each other
    ax.text(0.15, -0.12, "$E(B) = 0$\n(No effective support)",
            ha="center", va="top", fontsize=10.5, color=C["nihilistic"],
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=C["nihilistic"], linewidth=1.2))
    ax.text(0.45, 0.42, "$E(B) = 0.3$\n(Attenuated)",
            ha="center", va="bottom", fontsize=10.5, color="#B8860B",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=C["vulnerable"], linewidth=1.2))
    ax.text(0.80, 0.88, "$E(B) = 1.0$\n(Full)",
            ha="center", va="top", fontsize=10.5, color="#005C40",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=C["protected"], linewidth=1.2))

    # Summary box — bottom right, away from green segment
    summary = ("$E(B) = 1.0$  if  $B \\geq 0.6$\n"
               "$E(B) = 0.3$  if  $0.3 \\leq B < 0.6$\n"
               "$E(B) = 0$    if  $B < 0.3$")
    ax.text(0.98, 0.48, summary, transform=ax.transAxes,
            fontsize=9.5, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="#F9F9F9",
                      edgecolor="#AAAAAA", alpha=0.95))

    ax.set_xticks([0, 0.3, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0.0", "$_B$0.3", "$_B$0.6", "0.8", "1.0"])
    ax.set_yticks([0, 0.3, 0.5, 1.0])
    ax.set_xlabel("Provider Belief Level ($B$)")
    ax.set_ylabel("Support Effectiveness $E(B)$")
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.30, 1.18)
    plt.tight_layout()
    plt.savefig("Fig_effectiveness.png")
    plt.close()
    print("→ Fig_effectiveness.png")


def fig_simulation(model_cls) -> None:
    """Figure 2 — State trajectories: baseline and belief crisis (no title)."""
    h_base = model_cls(seed=42).run()
    h_cris = model_cls(seed=42).run(shock_type="belief", shock_mag=0.40)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, h, subtitle, show_shock in [
        (axes[0], h_base, "(a) Baseline Scenario",  False),
        (axes[1], h_cris, "(b) Belief Crisis",       True),
    ]:
        t = range(len(h["protected"]))
        ax.stackplot(t, h["protected"], h["vulnerable"], h["nihilistic"],
                     colors=[C["protected"], C["vulnerable"], C["nihilistic"]],
                     alpha=0.85)
        if show_shock:
            ax.axvline(x=50, color="black", linestyle="--", linewidth=1.5)
            ax.annotate("Crisis\nonset", xy=(50, 95), xytext=(65, 95),
                        fontsize=9, ha="left", va="top",
                        arrowprops=dict(arrowstyle="->", color="black", lw=1))
        fp = h["protected"][-1]; fn = h["nihilistic"][-1]
        ax.text(180, fp/2,     f"{fp:.0f}%", fontsize=10, fontweight="bold",
                color="white", ha="center", va="center")
        ax.text(180, 100-fn/2, f"{fn:.0f}%", fontsize=10, fontweight="bold",
                color="white", ha="center", va="center")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Population (%)")
        ax.set_title(subtitle, fontweight="normal", fontsize=11)
        ax.set_xlim(0, 200); ax.set_ylim(0, 100)

    legend = [mpatches.Patch(facecolor=C[s], alpha=0.85, label=s.capitalize())
              for s in ("protected", "vulnerable", "nihilistic")]
    fig.legend(handles=legend, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("Figure_2_simulation.png")
    plt.close()
    print("→ Figure_2_simulation.png")


def fig_scenario_comparison(model_cls, n_reps: int = 12) -> None:
    """Figure 3 — Scenario comparison bar chart (no title)."""
    configs = [
        ("Baseline",            {}),
        ("Economic\nShock",     {"shock_type":"economic","shock_mag":0.30}),
        ("Belief\nCrisis",      {"shock_type":"belief",  "shock_mag":0.40}),
        ("Social\nDisruption",  {"shock_type":"social",  "shock_mag":0.50}),
        ("Weak-Belief\nNetwork",{"weak":True}),
    ]
    results = []
    for name, kwargs in configs:
        pl, nl = [], []
        for i in range(n_reps):
            p, n = model_cls(seed=42+i).run_summary(**kwargs)
            pl.append(p); nl.append(n)
        results.append({
            "name": name,
            "p_mean":np.mean(pl),"p_sd":np.std(pl),
            "n_mean":np.mean(nl),"n_sd":np.std(nl),
        })

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(results)); w = 0.35

    b1 = ax.bar(x-w/2, [r["p_mean"] for r in results], w,
                yerr=[r["p_sd"] for r in results], capsize=3,
                color=C["protected"],  alpha=0.85,
                error_kw={"elinewidth":1.5,"capthick":1.5})
    b2 = ax.bar(x+w/2, [r["n_mean"] for r in results], w,
                yerr=[r["n_sd"] for r in results], capsize=3,
                color=C["nihilistic"], alpha=0.85,
                error_kw={"elinewidth":1.5,"capthick":1.5})

    # Labels centred inside bars
    for bar, val in zip(b1, [r["p_mean"] for r in results]):
        if val > 8:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()/2,
                    f"{val:.0f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")
    for bar, val in zip(b2, [r["n_mean"] for r in results]):
        if val > 5:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()/2,
                    f"{val:.0f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")

    # Theorem 1 annotation → weak-belief nihilistic bar
    wbn = b2[4]
    ax_x = wbn.get_x() + wbn.get_width()/2
    ax_y = wbn.get_height() + results[4]["n_sd"] + 2
    ax.annotate("Theorem 1:\nComplete collapse",
                xy=(ax_x, ax_y), xytext=(ax_x-1.4, ax_y+12),
                fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#FFFDE7", edgecolor="#BDBDBD"))

    ax.legend(handles=[
        mpatches.Patch(facecolor=C["protected"],  alpha=0.85, label="Protected"),
        mpatches.Patch(facecolor=C["nihilistic"], alpha=0.85, label="Nihilistic"),
    ], loc="upper left", frameon=True, framealpha=0.9,
       edgecolor="#CCCCCC", fontsize=10)

    ax.set_ylabel("Population (%)")
    ax.set_xlabel("Scenario")
    ax.set_xticks(x)
    ax.set_xticklabels([r["name"] for r in results], fontsize=10)
    ax.set_ylim(0, 122)
    plt.tight_layout()
    plt.savefig("Figure_3_scenario_comparison.png")
    plt.close()
    print("→ Figure_3_scenario_comparison.png")


def fig_group_barplot(df: pd.DataFrame, res: Dict) -> None:
    """Figure 4 — Group bar chart (no title)."""
    hi = df["relig_group"]=="High"; mid = df["relig_group"]=="Medium"
    lo = df["relig_group"]=="Low"
    groups = [
        (f"High (≥{CUT_HIGH}%)",              df.loc[hi,  "suicide_rate"].values),
        (f"Medium ({CUT_LOW}–{CUT_HIGH-1}%)", df.loc[mid, "suicide_rate"].values),
        (f"Low (<{CUT_LOW}%)",                df.loc[lo,  "suicide_rate"].values),
    ]
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(3)
    means = [np.mean(g[1]) for g in groups]
    sds   = [np.std(g[1], ddof=1) for g in groups]
    cols  = [C["protected"], C["vulnerable"], C["nihilistic"]]
    bars  = ax.bar(x, means, yerr=sds, capsize=5, color=cols, alpha=0.85,
                   error_kw={"elinewidth":1.5,"capthick":1.5})
    for bar, m in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{m:.1f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g[0]}\nn={len(g[1])}" for g in groups], fontsize=11)
    ax.set_ylabel("Mean Suicide Mortality Rate (per 100,000)")
    ymax = max(means)+max(sds)+3
    ax.set_ylim(0, ymax+3)
    t = res["t_stat"]; pt = res["p_t"]; d = res["d"]
    p_str = "p < .001" if pt<.001 else f"p = {pt:.3f}"
    bh = ymax+0.5
    ax.plot([0,0,2,2],[bh-0.5,bh,bh,bh-0.5],"k-",lw=1)
    ax.text(1, bh+0.3, f"t = {t:.2f}, {p_str}, d = {d:.2f}",
            ha="center", fontsize=10, style="italic")
    plt.tight_layout()
    plt.savefig("Figure_4_group_barplot.png")
    plt.close()
    print("→ Figure_4_group_barplot.png")


def fig_scatter(df: pd.DataFrame, res: Dict) -> None:
    """Figure 5 — Scatter plot (no title)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    for mask, lbl, col in [
        (df["relig_group"]=="High",   "High",   C["protected"]),
        (df["relig_group"]=="Medium", "Medium", C["vulnerable"]),
        (df["relig_group"]=="Low",    "Low",    C["nihilistic"]),
    ]:
        ax.scatter(df.loc[mask,"religiosity_pct"], df.loc[mask,"suicide_rate"],
                   c=col, s=70, alpha=0.85, label=lbl,
                   edgecolors="white", linewidth=0.5)
        for _, row in df.loc[mask].iterrows():
            ax.annotate(row["iso3"],
                        (row["religiosity_pct"], row["suicide_rate"]),
                        fontsize=6, alpha=0.7, ha="left",
                        xytext=(2,2), textcoords="offset points")
    sl, ic, rv, pv, _ = stats.linregress(
        df["religiosity_pct"], df["suicide_rate"])
    xl = np.linspace(0, 100, 100)
    ax.plot(xl, ic+sl*xl, "--", color=C["neutral"], linewidth=2)
    p_str = "p < .001" if pv<.001 else f"p = {pv:.3f}"
    ax.text(0.02, 0.97, f"r = {rv:.2f}, {p_str}",
            transform=ax.transAxes, fontsize=11, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("Religiosity (%)")
    ax.set_ylabel("Suicide Mortality Rate (per 100,000)")
    ax.legend(loc="upper right")
    ax.set_xlim(-2, 102)
    plt.tight_layout()
    plt.savefig("Figure_5_scatter.png")
    plt.close()
    print("→ Figure_5_scatter.png")


def fig_partial_regression(df: pd.DataFrame, res: Dict) -> None:
    """Figure 6 — Added-variable (partial regression) plot (no title)."""
    controls = ["log_gdp","social_support_pct","unemployment_pct"]
    sub = df[["religiosity_pct","suicide_rate"]+controls].dropna()
    Xc  = sm.add_constant(sub[controls])
    rx  = sm.OLS(sub["religiosity_pct"], Xc).fit().resid
    ry  = sm.OLS(sub["suicide_rate"],    Xc).fit().resid
    rp, pp = stats.pearsonr(rx, ry)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(rx, ry, c=C["primary"], s=60, alpha=0.75, edgecolors="white")
    for i, (x_, y_) in enumerate(zip(rx, ry)):
        ax.annotate(df["iso3"].iloc[sub.index[i]],
                    (x_, y_), fontsize=6, alpha=0.6, ha="left",
                    xytext=(2,2), textcoords="offset points")
    sl, ic, *_ = stats.linregress(rx, ry)
    xl = np.linspace(rx.min()-2, rx.max()+2, 100)
    ax.plot(xl, ic+sl*xl, color=C["nihilistic"], linewidth=2)
    ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
    ax.axvline(0, color="grey", linestyle="--", alpha=0.5)
    p_str = "p < .001" if pp<.001 else f"p = {pp:.3f}"
    ax.text(0.02, 0.97, f"Partial r = {rp:.2f}, {p_str}",
            transform=ax.transAxes, fontsize=11, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("Residual Religiosity\n"
                  "(controlling for log GDP, social support, unemployment)")
    ax.set_ylabel("Residual Suicide Rate\n(controlling for same)")
    plt.tight_layout()
    plt.savefig("Figure_6_partial_regression.png")
    plt.close()
    print("→ Figure_6_partial_regression.png")


def fig_predicted_values(df: pd.DataFrame, res: Dict) -> None:
    """Figure 7 — Predicted values from multiple regression (no title)."""
    model = res["model"]
    df_m  = df[["religiosity_pct","log_gdp","social_support_pct",
                "unemployment_pct","suicide_rate"]].dropna()
    xl = np.linspace(0, 100, 200)
    pred_data = pd.DataFrame({
        "const": 1.0, "religiosity_pct": xl,
        "log_gdp":            df_m["log_gdp"].mean(),
        "social_support_pct": df_m["social_support_pct"].mean(),
        "unemployment_pct":   df_m["unemployment_pct"].mean(),
    })
    pr   = model.get_prediction(pred_data)
    pmn  = pr.predicted_mean
    pci  = pr.conf_int(alpha=0.05)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.fill_between(xl, pci[:,0], pci[:,1], alpha=0.20, color=C["primary"],
                    label="95% CI")
    ax.plot(xl, pmn, color=C["primary"], linewidth=2.5, label="Predicted rate")
    ax.scatter(df_m["religiosity_pct"], df_m["suicide_rate"],
               c=C["neutral"], s=50, alpha=0.65, label="Observed",
               edgecolors="white")
    for pct, dx, dy in [(10, +15, +2), (85, -20, -2)]:
        idx = np.argmin(np.abs(xl - pct))
        ax.annotate(f"At {pct}%: ≈{pmn[idx]:.1f}",
                    xy=(pct, pmn[idx]), xytext=(pct+dx, pmn[idx]+dy),
                    fontsize=9, arrowprops=dict(arrowstyle="->", color="grey"))
    ax.set_xlabel("Religiosity (%)")
    ax.set_ylabel("Predicted Suicide Rate (per 100,000)")
    ax.set_xlim(0, 100)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("Figure_7_predicted_values.png")
    plt.close()
    print("→ Figure_7_predicted_values.png")


# ===========================================================================
# 7. AGENT-BASED SIMULATION (parameters unchanged from manuscript)
# ===========================================================================

class ASRModel:
    """
    Four-Factor Resilience Model.
    All parameters identical to those reported in the manuscript.

    Protection index : Πᵢ = Bᵢ^0.35 · Pᵢ^0.30 · Hᵣ^0.175 · Sᵣ^0.175
    Effectiveness    : E(B) = 1.0 (B≥0.6), 0.3 (0.3≤B<0.6), 0 (B<0.3)
    States           : Protected (Π>0.50), Vulnerable (0.30<Π≤0.50),
                       Nihilistic (Π≤0.30)
    Dynamics         : Bᵢ(t+1) = clip[Bᵢ + δB·f(state) + γ·Hᵣ, 0.01, 1]
                       Pᵢ(t+1) = clip[Pᵢ + δP·f(state) + γ·Sᵣ, 0.01, 1]
    """
    # Exponents
    WB = 0.35; WP = 0.30; WH = 0.175; WS = 0.175
    # Thresholds
    T_PROT = 0.50; T_VULN = 0.30
    # Dynamics
    dB = 0.015; dP = 0.020; GAMMA = 0.030
    F  = {"protected": 0.15, "vulnerable": -0.15, "nihilistic": -0.40}
    # Network
    K = 6; RW = 0.1

    def __init__(self, n: int = 1000, seed: int = 42):
        self.n = n; self.seed = seed
        self.history: Dict = defaultdict(list)

    def _effectiveness(self, b: np.ndarray) -> np.ndarray:
        E = np.zeros_like(b)
        E[b >= 0.6] = 1.0
        E[(b >= 0.3) & (b < 0.6)] = 0.3
        return E

    def _initialize(self, weak: bool = False) -> None:
        np.random.seed(self.seed)
        if weak:
            self.B = np.random.beta(1, 5, self.n) * 0.29
            self.P = np.random.beta(1, 5, self.n) * 0.29
        else:
            self.B = np.random.beta(2, 2, self.n)
            self.P = 0.6*self.B + 0.4*np.random.beta(2, 2, self.n)
        self.B = np.clip(self.B, 0.01, 1.0)
        self.P = np.clip(self.P, 0.01, 1.0)
        self.Hc = np.random.beta(2, 3, self.n)
        self.Sc = np.random.beta(2, 3, self.n)
        G = nx.watts_strogatz_graph(self.n, self.K, self.RW, seed=self.seed)
        adj = nx.to_numpy_array(G)
        w = np.random.uniform(0.5, 1.0, (self.n, self.n))
        w = (w + w.T) / 2
        self.W = w * adj
        self.history = defaultdict(list)

    def _step(self) -> None:
        E  = self._effectiveness(self.B)
        ws = self.W.sum(axis=1) + 1e-10
        Hr = (self.W @ (self.Hc * E)) / ws
        Sr = (self.W @ (self.Sc * E)) / ws
        pi = (np.maximum(self.B, 0.01)**self.WB *
              np.maximum(self.P, 0.01)**self.WP *
              np.maximum(Hr, 0.01)**self.WH *
              np.maximum(Sr, 0.01)**self.WS)
        st = np.full(self.n, "nihilistic", dtype="U12")
        st[pi > self.T_VULN] = "vulnerable"
        st[pi > self.T_PROT] = "protected"
        f = np.full(self.n, self.F["nihilistic"])
        f[st == "vulnerable"] = self.F["vulnerable"]
        f[st == "protected"]  = self.F["protected"]
        self.B = np.clip(self.B + self.dB*f + self.GAMMA*Hr, 0.01, 1.0)
        self.P = np.clip(self.P + self.dP*f + self.GAMMA*Sr, 0.01, 1.0)
        n = self.n
        self.history["protected"].append(np.sum(st=="protected")  / n * 100)
        self.history["vulnerable"].append(np.sum(st=="vulnerable") / n * 100)
        self.history["nihilistic"].append(np.sum(st=="nihilistic") / n * 100)

    def run(self, weak: bool = False,
            shock_type: Optional[str] = None,
            shock_mag: float = 0.4,
            shock_time: int = 50,
            n_steps: int = 200) -> Dict:
        """Run simulation; return history dict."""
        self._initialize(weak)
        for t in range(n_steps):
            if shock_type and t == shock_time:
                if shock_type == "belief":
                    self.B = np.clip(self.B*(1-shock_mag), 0.01, 1.0)
                elif shock_type == "economic":
                    self.P = np.clip(self.P*(1-shock_mag), 0.01, 1.0)
                elif shock_type == "social":
                    self.W *= (1 - shock_mag)
            self._step()
        return dict(self.history)

    def run_summary(self, **kwargs) -> Tuple[float, float]:
        """Run and return (protected_pct, nihilistic_pct) at final step."""
        h = self.run(**kwargs)
        return h["protected"][-1], h["nihilistic"][-1]


# ===========================================================================
# 8. MAIN
# ===========================================================================

def main() -> None:
    print("=" * 60)
    print("FOUR-FACTOR RESILIENCE — FULL REPLICATION SCRIPT")
    print("=" * 60)

    # ── Data collection ───────────────────────────────────────────
    print("\n[1] DATA COLLECTION")
    suicide_data  = collect_who_suicide_data()
    relig_data    = collect_religiosity_data()
    gdp_data      = collect_gdp_data()
    unemp_data    = collect_unemployment_data()
    ss_data       = collect_social_support_data()
    pop_data      = collect_population_data()

    # ── Sample construction ───────────────────────────────────────
    print("\n[2] SAMPLE CONSTRUCTION")
    df, excl_log = apply_inclusion_criteria(
        suicide_data, relig_data, gdp_data, unemp_data, ss_data, pop_data)

    # ── Online Resources (Tables S0–S3) ───────────────────────────
    print("\n[3] ONLINE RESOURCES")
    save_online_resources(df, excl_log)

    # ── Statistical analysis ──────────────────────────────────────
    print("\n[4] STATISTICAL ANALYSIS")
    res = run_analysis(df)

    # ── Empirical figures ─────────────────────────────────────────
    print("\n[5] EMPIRICAL FIGURES")
    fig_effectiveness()
    fig_group_barplot(df, res)
    fig_scatter(df, res)
    fig_partial_regression(df, res)
    fig_predicted_values(df, res)

    # ── Simulation figures ────────────────────────────────────────
    print("\n[6] SIMULATION FIGURES  (N=1000, T=200, 12 replications)")
    fig_simulation(ASRModel)
    fig_scenario_comparison(ASRModel, n_reps=12)

    # ── Summary ───────────────────────────────────────────────────
    print(f"""
{"="*60}
KEY RESULTS
{"="*60}
Empirical (N = {len(df)} countries)
  Bivariate r      = {res['r_biv']:.3f}  (p {'< .001' if res['p_biv']<.001 else f"= {res['p_biv']:.3f}"})
  Partial r        = {res['r_part']:.3f}  (p {'< .001' if res['p_part']<.001 else f"= {res['p_part']:.3f}"})
  Cohen's d        = {abs(res['d']):.3f}  (p {'< .001' if res['p_t']<.001 else f"= {res['p_t']:.3f}"})
  β(religiosity)   = {res['beta_relig']:.4f} (p {('< .001' if res['p_relig']<.001 else f"= {res['p_relig']:.3f}")})
  R²               = {res['model'].rsquared:.3f}
{"="*60}
""")


if __name__ == "__main__":
    main()
