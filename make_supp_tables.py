#!/usr/bin/env python3
"""Generate Table S4 (support-effectiveness robustness) and
Table S5 (vital-registration subsample) as CSVs, from the two tests."""
import numpy as np, pandas as pd, scipy.stats as stats, statsmodels.api as sm
import networkx as nx
from collections import defaultdict

# ------------------------------------------------------------------ S4
class M:
    WB=0.35; WP=0.30; WH=0.175; WS=0.175; T_PROT=0.50; T_VULN=0.30
    dB=0.015; dP=0.020; GAMMA=0.030
    F={"protected":0.15,"vulnerable":-0.15,"nihilistic":-0.40}; K=6; RW=0.1
    def __init__(s,n=1000,seed=42,mode="belief"): s.n=n; s.seed=seed; s.mode=mode; s.h=defaultdict(list)
    def _tier(s,x):
        E=np.zeros_like(x); E[x>=0.6]=1.0; E[(x>=0.3)&(x<0.6)]=0.3; return E
    def _init(s,weak=False):
        np.random.seed(s.seed)
        if weak: s.B=np.random.beta(1,5,s.n)*0.29; s.P=np.random.beta(1,5,s.n)*0.29
        else: s.B=np.random.beta(2,2,s.n); s.P=0.6*s.B+0.4*np.random.beta(2,2,s.n)
        s.B=np.clip(s.B,0.01,1); s.P=np.clip(s.P,0.01,1)
        s.Hc=np.random.beta(2,3,s.n); s.Sc=np.random.beta(2,3,s.n)
        G=nx.watts_strogatz_graph(s.n,s.K,s.RW,seed=s.seed); adj=nx.to_numpy_array(G)
        w=np.random.uniform(0.5,1,(s.n,s.n)); w=(w+w.T)/2; s.W=w*adj; s.h=defaultdict(list)
    def _step(s):
        if s.mode=="belief": E=s._tier(s.B)
        elif s.mode=="purpose": E=s._tier(s.P)
        else: E=s._tier(0.5*(s.B+s.P))
        ws=s.W.sum(1)+1e-10; Hr=(s.W@(s.Hc*E))/ws; Sr=(s.W@(s.Sc*E))/ws
        pi=(np.maximum(s.B,.01)**s.WB*np.maximum(s.P,.01)**s.WP*
            np.maximum(Hr,.01)**s.WH*np.maximum(Sr,.01)**s.WS)
        st=np.full(s.n,"nihilistic","U12"); st[pi>s.T_VULN]="vulnerable"; st[pi>s.T_PROT]="protected"
        f=np.full(s.n,s.F["nihilistic"]); f[st=="vulnerable"]=s.F["vulnerable"]; f[st=="protected"]=s.F["protected"]
        s.B=np.clip(s.B+s.dB*f+s.GAMMA*Hr,0.01,1); s.P=np.clip(s.P+s.dP*f+s.GAMMA*Sr,0.01,1)
        s.h["nih"].append(np.sum(st=="nihilistic")/s.n*100)
    def run(s,weak=False,shock=None,mag=0.4,tt=50,T=200):
        s._init(weak)
        for t in range(T):
            if shock and t==tt:
                if shock=="belief": s.B=np.clip(s.B*(1-mag),0.01,1)
                elif shock=="economic": s.P=np.clip(s.P*(1-mag),0.01,1)
                elif shock=="social": s.W*=(1-mag)
            s._step()
        return s.h["nih"][-1]

def mean_nih(mode,**kw):
    v=[M(seed=42+i,mode=mode).run(**kw) for i in range(12)]; return np.mean(v),np.std(v)

gates=[("Belief B (original)","belief"),("Purpose P","purpose"),("Mean(B,P)","meanBP")]
scen=[("Baseline",{}),("Economic -30%",{"shock":"economic","mag":0.30}),
      ("Belief -40%",{"shock":"belief","mag":0.40}),("Social -50%",{"shock":"social","mag":0.50})]
rows=[]
for gname,gmode in gates:
    base=mean_nih(gmode)[0]
    for sname,kw in scen:
        m,sd=mean_nih(gmode,**kw)
        rows.append({"Effectiveness gate":gname,"Scenario":sname,
                     "Nihilism_pct":round(m,2),"SD":round(sd,2),
                     "Ratio_vs_baseline":round(m/base,2)})
pd.DataFrame(rows).to_csv("Table_S4_effectiveness_robustness.csv",index=False)
print("wrote Table_S4_effectiveness_robustness.csv")

# ------------------------------------------------------------------ S5
df=pd.read_csv("/mnt/user-data/uploads/Table_S1_country_list.csv").rename(columns={
    "Relig (%)":"relig","Suicide(/100k)":"suicide","GDP (USD)":"gdp",
    "Unemp (%)":"unemp","Soc Sup (%)":"soc","ISO3":"iso3"})
df["log_gdp"]=np.log(df["gdp"])
HQ_VR={"ALB","AUT","BEL","BIH","BGR","HRV","CZE","DNK","EST","FIN","FRA","DEU","GRC",
"HUN","ISL","IRL","ITA","LVA","LTU","LUX","MLT","MDA","MNE","NLD","MKD","NOR","POL",
"PRT","ROU","RUS","SRB","SVK","SVN","ESP","SWE","CHE","UKR","GBR","BLR","ARG","BRA",
"CAN","CHL","COL","CRI","CUB","DOM","ECU","GUY","MEX","PAN","PRY","SUR","TTO","URY",
"USA","VEN","AUS","JPN","KOR","NZL","SGP","ISR","CYP","MUS","ARM","AZE","GEO","KAZ","KGZ"}
def pr(d,x,y,c):
    Z=sm.add_constant(d[c]); rx=sm.OLS(d[x],Z).fit().resid; ry=sm.OLS(d[y],Z).fit().resid
    r,p=stats.pearsonr(rx,ry); return r,p
def cohend(a,b):
    sp=np.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2))
    return (np.mean(a)-np.mean(b))/sp
def stats_row(d,label):
    c=["log_gdp","soc","unemp"]; r,p=stats.pearsonr(d["relig"],d["suicide"])
    rp,pp=pr(d,"relig","suicide",c); hi=d[d.relig>=60]["suicide"]; lo=d[d.relig<30]["suicide"]
    return {"Sample":label,"N":len(d),"Relig_range":f"{d.relig.min():.0f}-{d.relig.max():.0f}",
            "n_high":len(hi),"n_low":len(lo),"bivariate_r":round(r,3),
            "partial_r":round(rp,3),"cohens_d":round(cohend(hi.values,lo.values),3),
            "p":"<.001" if max(p,pp)<.001 else f"{max(p,pp):.3f}"}
s5=pd.DataFrame([stats_row(df,"Full sample"),
                 stats_row(df[df.iso3.isin(HQ_VR)],"High-quality vital registration")])
s5.to_csv("Table_S5_vitalregistration_subsample.csv",index=False)
print("wrote Table_S5_vitalregistration_subsample.csv")
print(s5.to_string(index=False))
