#!/usr/bin/env python3
import numpy as np, pandas as pd, scipy.stats as stats, statsmodels.api as sm, re

icdr = pd.read_csv("icdr.csv")
icdr.columns = [c.strip().strip('"').lstrip('\ufeff') for c in icdr.columns]
# per country: 2019 completeness, else most recent 2015-2018
icdr = icdr.dropna(subset=["death_comp"])
icdr = icdr.sort_values("year")
latest = icdr.groupby("country_name").tail(1)[["country_name","year","death_comp"]]
comp2019 = icdr[icdr.year==2019][["country_name","death_comp"]]
comp = comp2019.set_index("country_name")["death_comp"].to_dict()
latestd = latest.set_index("country_name")["death_comp"].to_dict()
def get_comp(name):
    return comp.get(name, latestd.get(name, np.nan))

df = pd.read_csv("/mnt/user-data/uploads/Table_S1_country_list.csv").rename(columns={
    "Relig (%)":"relig","Suicide(/100k)":"suicide","GDP (USD)":"gdp",
    "Unemp (%)":"unemp","Soc Sup (%)":"soc","ISO3":"iso3","Country":"country"})
df["log_gdp"]=np.log(df["gdp"])

# name normalisation + manual aliases (S1 name -> ICDR country_name)
alias = {
 "United States":"United States of America","Russia":"Russian Federation",
 "South Korea":"Republic of Korea","North Macedonia":"North Macedonia",
 "Czechia":"Czechia","Czech Republic":"Czechia","Moldova":"Republic of Moldova",
 "Bolivia":"Bolivia (Plurinational State of)","Venezuela":"Venezuela (Bolivarian Republic of)",
 "Iran":"Iran (Islamic Republic of)","Tanzania":"United Republic of Tanzania",
 "Vietnam":"Viet Nam","Syria":"Syrian Arab Republic","Laos":"Lao People's Democratic Republic",
 "Brunei":"Brunei Darussalam","Cape Verde":"Cabo Verde","Ivory Coast":"Côte d'Ivoire",
 "Democratic Republic of the Congo":"Democratic Republic of the Congo","Congo":"Congo",
 "United Kingdom":"United Kingdom of Great Britain and Northern Ireland",
 "Turkey":"Türkiye","Kyrgyzstan":"Kyrgyzstan","Slovakia":"Slovakia",
 "Republic of the Congo":"Congo","Palestine":"State of Palestine","Trinidad & Tobago":"Trinidad and Tobago","Trinidad and Tobago":"Trinidad and Tobago","Czech Republic":"Czechia","UAE":"United Arab Emirates",
}
def norm(s): return re.sub(r'[^a-z]','',str(s).lower())
icdr_names = {norm(k):k for k in set(list(comp.keys())+list(latestd.keys()))}
def match_comp(row):
    nm = alias.get(row["country"], row["country"])
    if norm(nm) in icdr_names: return get_comp(icdr_names[norm(nm)])
    if norm(row["country"]) in icdr_names: return get_comp(icdr_names[norm(row["country"])])
    return np.nan
df["death_comp"] = df.apply(match_comp, axis=1)

print("matched completeness for", df["death_comp"].notna().sum(), "of", len(df), "countries")
missing = df[df["death_comp"].isna()]["country"].tolist()
print("unmatched (no ICDR completeness):", len(missing))

def pr(d,x,y,c):
    Z=sm.add_constant(d[c]); rx=sm.OLS(d[x],Z).fit().resid; ry=sm.OLS(d[y],Z).fit().resid
    r,p=stats.pearsonr(rx,ry); return r,p
def cohend(a,b):
    sp=np.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2))
    return (np.mean(a)-np.mean(b))/sp
def analyse(d,label):
    c=["log_gdp","soc","unemp"]; r,p=stats.pearsonr(d["relig"],d["suicide"])
    rp,pp=pr(d,"relig","suicide",c); hi=d[d.relig>=60]["suicide"].values; lo=d[d.relig<30]["suicide"].values
    dd=cohend(hi,lo) if len(hi)>1 and len(lo)>1 else float('nan')
    print(f"\n{label}  N={len(d)}  relig range {d.relig.min():.0f}-{d.relig.max():.0f} (mean {d.relig.mean():.0f})")
    print(f"   high(>=60%) n={len(hi)}  low(<30%) n={len(lo)}")
    print(f"   bivariate r={r:+.3f} (p{'<.001' if p<.001 else f'={p:.3f}'})   partial r={rp:+.3f} (p{'<.001' if pp<.001 else f'={pp:.3f}'})   d={dd:+.3f}")

analyse(df,"FULL SAMPLE")
for thr in [80,90]:
    sub=df[df["death_comp"]>=thr]
    analyse(sub, f"ICDR death-registration completeness >= {thr}%")
