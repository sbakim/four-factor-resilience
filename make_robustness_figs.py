#!/usr/bin/env python3
"""Generate the two robustness figures for the revised manuscript.
VR figure uses the objective ICDR death-registration completeness >=90% subsample."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, re
import scipy.stats as stats, statsmodels.api as sm

plt.rcParams.update({
    "font.family":"serif","font.serif":["DejaVu Serif"],
    "font.size":11,"axes.labelsize":12,"axes.titlesize":12,"legend.fontsize":10,
    "xtick.labelsize":10,"ytick.labelsize":10,"figure.dpi":200,"savefig.dpi":200,
    "savefig.bbox":"tight","savefig.facecolor":"white",
    "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.8})
C={"protected":"#009E73","vulnerable":"#E69F00","nihilistic":"#D55E00",
   "primary":"#0072B2","neutral":"#555555"}

# E(B) robustness figure
gates=["Belief $B$\n(original)","Purpose $P$","Mean$(B,P)$"]
belief=[1.92,1.13,1.37]; econ=[1.08,1.46,1.20]
x=np.arange(len(gates)); w=0.36
fig,ax=plt.subplots(figsize=(7.2,4.6))
b1=ax.bar(x-w/2,belief,w,color=C["primary"],alpha=0.9,label="Belief shock ($-40\\%$)")
b2=ax.bar(x+w/2,econ,w,color=C["vulnerable"],alpha=0.9,label="Economic shock ($-30\\%$)")
ax.axhline(1.0,color=C["neutral"],ls="--",lw=1.1,zorder=0)
ax.text(2.48,1.02,"no effect",fontsize=8.5,color=C["neutral"],va="bottom",ha="right")
for bars in (b1,b2):
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.03,
                f"{bar.get_height():.2f}$\\times$",ha="center",va="bottom",
                fontsize=9.5,fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(gates)
ax.set_ylabel("Nihilism ratio (crisis $/$ baseline)")
ax.set_xlabel("Resource gating support effectiveness $E(\\cdot)$")
ax.set_ylim(0,2.25)
ax.legend(loc="upper right",frameon=True,framealpha=0.95,edgecolor="#CCCCCC")
plt.tight_layout(); plt.savefig("figs/fig_effrobust.png"); plt.close()
print("wrote figs/fig_effrobust.png")

# VR figure (ICDR completeness >=90%)
df=pd.read_csv("/mnt/user-data/uploads/Table_S1_country_list.csv").rename(columns={
    "Relig (%)":"relig","Suicide(/100k)":"suicide","Country":"country"})
icdr=pd.read_csv("icdr.csv"); icdr.columns=[c.strip().strip('"').lstrip('\ufeff') for c in icdr.columns]
icdr=icdr.dropna(subset=["death_comp"]).sort_values("year")
latest=icdr.groupby("country_name").tail(1).set_index("country_name")["death_comp"].to_dict()
c2019=icdr[icdr.year==2019].set_index("country_name")["death_comp"].to_dict()
def comp(n): return c2019.get(n,latest.get(n,np.nan))
alias={"United States":"United States of America","Russia":"Russian Federation",
"South Korea":"Republic of Korea","Moldova":"Republic of Moldova",
"Venezuela":"Venezuela (Bolivarian Republic of)","Iran":"Iran (Islamic Republic of)",
"Vietnam":"Viet Nam","United Kingdom":"United Kingdom of Great Britain and Northern Ireland",
"Turkey":"Türkiye","Bolivia":"Bolivia (Plurinational State of)","Czech Republic":"Czechia",
"Trinidad & Tobago":"Trinidad and Tobago","UAE":"United Arab Emirates"}
def norm(s): return re.sub(r'[^a-z]','',str(s).lower())
idx={norm(k):k for k in set(list(c2019)+list(latest))}
def getc(row):
    nm=alias.get(row["country"],row["country"])
    if norm(nm) in idx: return comp(idx[norm(nm)])
    if norm(row["country"]) in idx: return comp(idx[norm(row["country"])])
    return np.nan
df["comp"]=df.apply(getc,axis=1); df["hq"]=df["comp"]>=90
sub=df[df["hq"]]
r_full=stats.pearsonr(df["relig"],df["suicide"])[0]
r_sub =stats.pearsonr(sub["relig"],sub["suicide"])[0]
fig,ax=plt.subplots(figsize=(7.2,4.8))
ax.scatter(df.loc[~df.hq,"relig"],df.loc[~df.hq,"suicide"],s=34,c="#BBBBBB",
           alpha=0.7,edgecolors="white",linewidth=0.4,label="Other countries")
ax.scatter(sub["relig"],sub["suicide"],s=42,c=C["primary"],alpha=0.85,
           edgecolors="white",linewidth=0.5,label="Complete registration ($\\geq$90%)")
xl=np.linspace(0,100,100)
for d,col,lab,ls in [(df,"#999999",f"Full sample ($r={r_full:+.2f}$)","--"),
                     (sub,C["primary"],f"Complete-registration ($r={r_sub:+.2f}$)","-")]:
    X=sm.add_constant(d["relig"]); m=sm.OLS(d["suicide"],X).fit()
    ax.plot(xl,m.params.iloc[0]+m.params.iloc[1]*xl,col,ls=ls,lw=2,label=lab)
ax.set_xlabel("Religiosity (%)"); ax.set_ylabel("Suicide Mortality Rate (per 100,000)")
ax.set_xlim(0,100); ax.set_ylim(0,df["suicide"].max()*1.05)
ax.legend(loc="upper right",frameon=True,framealpha=0.95,edgecolor="#CCCCCC",fontsize=9)
plt.tight_layout(); plt.savefig("figs/fig_vr.png"); plt.close()
print(f"wrote figs/fig_vr.png  (full r={r_full:.2f}, subsample N={len(sub)} r={r_sub:.2f})")
