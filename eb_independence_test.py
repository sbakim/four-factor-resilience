#!/usr/bin/env python3
"""
E(B)-INDEPENDENCE ROBUSTNESS TEST
Reproduces the manuscript ASRModel exactly, then adds an alternative
support-effectiveness specification that keys off the provider's overall
protection state (lagged), instead of belief alone.

If 'belief crisis > economic/social crisis' survives the belief-neutral
specification, the asymmetry is not an artefact of E(B) singling out belief.
"""
import numpy as np
import networkx as nx
from collections import defaultdict

class ASRModel:
    WB = 0.35; WP = 0.30; WH = 0.175; WS = 0.175
    T_PROT = 0.50; T_VULN = 0.30
    dB = 0.015; dP = 0.020; GAMMA = 0.030
    F  = {"protected": 0.15, "vulnerable": -0.15, "nihilistic": -0.40}
    K = 6; RW = 0.1

    def __init__(self, n=1000, seed=42, eff_mode="belief"):
        # eff_mode: "belief" (original E(B)) or "state" (belief-neutral, lagged Pi)
        self.n = n; self.seed = seed; self.eff_mode = eff_mode
        self.history = defaultdict(list)
        self.pi_prev = None

    def _eff_from_belief(self, b):
        E = np.zeros_like(b)
        E[b >= 0.6] = 1.0
        E[(b >= 0.3) & (b < 0.6)] = 0.3
        return E

    def _eff_tiered(self, x):
        # Same 1.0 / 0.3 / 0 tiers and SAME thresholds (0.6 / 0.3) as E(B),
        # but applied to a belief-neutral gate variable x (keeps calibration).
        E = np.zeros_like(x)
        E[x >= 0.6] = 1.0
        E[(x >= 0.3) & (x < 0.6)] = 0.3
        return E

    def _initialize(self, weak=False):
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
        self.pi_prev = None

    def _step(self):
        # --- provider effectiveness ---
        if self.eff_mode == "belief":
            E = self._eff_from_belief(self.B)          # original E(B)
        elif self.eff_mode == "purpose":
            E = self._eff_tiered(self.P)               # belief-neutral: gate on P
        elif self.eff_mode == "meanBP":
            E = self._eff_tiered(0.5*(self.B+self.P))  # belief-neutral: gate on mean(B,P)
        else:
            raise ValueError(self.eff_mode)
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
        self.pi_prev = pi
        n = self.n
        self.history["nihilistic"].append(np.sum(st=="nihilistic") / n * 100)

    def run(self, weak=False, shock_type=None, shock_mag=0.4,
            shock_time=50, n_steps=200):
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

    def final_nih(self, **kw):
        return self.run(**kw)["nihilistic"][-1]


def mean_nih(eff_mode, n_reps=12, **kw):
    vals = [ASRModel(seed=42+i, eff_mode=eff_mode).final_nih(**kw)
            for i in range(n_reps)]
    return np.mean(vals), np.std(vals)

scenarios = [
    ("Baseline",        {}),
    ("Economic (-30%)", {"shock_type":"economic","shock_mag":0.30}),
    ("Belief (-40%)",   {"shock_type":"belief",  "shock_mag":0.40}),
    ("Social (-50%)",   {"shock_type":"social",  "shock_mag":0.50}),
]

for mode, label in [("belief","ORIGINAL   E gated on belief B  (thresholds 0.6/0.3)"),
                    ("purpose","ALT-1      E gated on purpose P (same thresholds) [belief-neutral]"),
                    ("meanBP","ALT-2      E gated on mean(B,P)   (same thresholds) [belief-neutral]")]:
    print("="*68)
    print(label)
    print("="*68)
    base = None
    print(f"{'Scenario':<18}{'Nihilism %':>12}{'SD':>8}{'Ratio vs base':>16}")
    for name, kw in scenarios:
        m, s = mean_nih(mode, **kw)
        if name == "Baseline":
            base = m
        ratio = m/base if base else float('nan')
        print(f"{name:<18}{m:>12.2f}{s:>8.2f}{ratio:>16.2f}")
    print()
