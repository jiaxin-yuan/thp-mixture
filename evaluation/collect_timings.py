#!/usr/bin/env python3
"""Training cost per (dataset, model) for Table 5: wall time, epochs, min/epoch.

Sources:
  THP-B / THP-M     logs_{baseline_act,mixture}_v2_UQ_<DS>.log, written by
                    run_thp.sh, summed over the per-epoch `Train ... t=<x>min`
  Hawkes            logs_hawkes_v2_UQ_<DS>.log, the same format, written by
                    hawkes_baseline.py under evaluation/
  SuTraN / ED-LSTM  paper_results/baselines/timings_sutran.csv, wall time from
                    the checkpoint mtime span of their own runs; where that span
                    was unavailable it is estimated as epochs times the mean
                    min/epoch over the datasets that do have one

The epoch budgets differ (THP 300 epochs / patience 24, the baselines 200), so
the total conflates per-epoch cost with convergence length; min/epoch is the
hardware-comparable column, as the report states.
"""
import os, re, sys, csv, glob, statistics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)               # THP_for_PPM/
THP = ROOT                                       # run_thp.sh writes its logs here
OUTDIR = os.path.join(ROOT, "paper_results")
# SuTraN / ED-LSTM training times, as captured in their own runs
SUT = os.path.join(OUTDIR, "baselines")

DATASETS = ["UQ_HelpDesk","UQ_Sepsis","UQ_BPIC13I","UQ_BPIC20DD","UQ_BPIC20RFP",
            "UQ_BPIC20PTC","UQ_BPIC20ID","UQ_BPIC20TPD","UQ_BPIC12","UQ_BPIC15_1"]

T_RE = re.compile(r"Train .*?t=([0-9.]+)min")

def thp_time(kind, ds):
    """Epochs and wall-clock minutes from one run's log, *kind* being its infix: baseline_act, mixture, or hawkes."""
    # run_thp.sh logs land in the repo root, hawkes_baseline.py's under evaluation/
    for d in (THP, SCRIPT_DIR):
        p = os.path.join(d, f"logs_{kind}_v2_{ds}.log")
        if os.path.exists(p):
            with open(p, errors="ignore") as fh:
                ts = [float(m) for m in T_RE.findall(fh.read())]
            if ts:
                return len(ts), round(sum(ts), 3), os.path.basename(p)
    return None, None, None

def load_sutran_csv():
    p = os.path.join(SUT, "timings_sutran.csv")
    rows = {}
    if not os.path.exists(p):
        return rows
    with open(p) as fh:
        for r in csv.DictReader(fh):
            rows[(r["dataset"], r["model"])] = r
    return rows

def f(x):
    try: return float(x)
    except: return None

def main():
    sc = load_sutran_csv()
    # mean min/epoch from the real ckpt spans, to estimate the pruned runs
    rates = {}
    for m in ("SUTRAN_NDA_results","ED_LSTM_results"):
        vals=[]
        for (ds,mm),r in sc.items():
            if mm==m and r["source"]=="ckpt_mtime_span":
                w,e=f(r["wall_min"]),f(r["epochs_trained"])
                if w and e and e>0: vals.append(w/e)
        rates[m]=statistics.mean(vals) if vals else None

    MODELS=[("THP-baseline","thp","baseline_act"),("THP-mixture","thp","mixture"),
            ("SuTraN","sut","SUTRAN_NDA_results"),("ED-LSTM","sut","ED_LSTM_results"),
            ("Hawkes","thp","hawkes")]
    table={}  # ds -> model -> (epochs, wall, src)
    for ds in DATASETS:
        table[ds]={}
        for name,fam,key in MODELS:
            if fam=="thp":
                ep,wall,src=thp_time(key,ds)
                table[ds][name]=(ep,wall,src)
            else:
                r=sc.get((ds,key))
                if not r:
                    table[ds][name]=(None,None,"missing"); continue
                ep=int(f(r["epochs_trained"])) if f(r["epochs_trained"]) else None
                w=f(r["wall_min"]); src=r["source"]
                if w is None and ep and rates[key]:
                    w=round(ep*rates[key],2); src="EST(ep*rate)"
                table[ds][name]=(ep,w,src)

    # without training logs the THP columns would be blank, so keep the shipped
    # measurements rather than overwrite them
    have_thp = any(table[ds][n][1] is not None
                   for ds in DATASETS for n in ("THP-baseline", "THP-mixture"))
    os.makedirs(OUTDIR, exist_ok=True)
    csv_path=os.path.join(OUTDIR,"time_consumption_comparison.csv")
    if not have_thp and os.path.exists(csv_path) and "--force" not in sys.argv:
        print(f"no training logs found under {THP}; keeping the existing "
              f"{os.path.basename(csv_path)} (use --force to overwrite)")
        return
    names=[m[0] for m in MODELS]
    with open(csv_path,"w",newline="") as fh:
        w=csv.writer(fh)
        head=["dataset"]
        for n in names: head+=[f"{n}_epochs",f"{n}_wall_min",f"{n}_min_per_epoch",f"{n}_src"]
        w.writerow(head)
        for ds in DATASETS:
            row=[ds]
            for n in names:
                ep,wall,src=table[ds][n]
                mpe=round(wall/ep,3) if (wall and ep) else ""
                row+=[ep if ep is not None else "",
                      wall if wall is not None else "",
                      mpe, src]
            w.writerow(row)

    # text report
    lines=[]
    lines.append("="*100)
    lines.append("TIME-CONSUMPTION COMPARISON  (training wall-time per dataset per model)")
    lines.append("="*100)
    lines.append("Caveat: epoch budgets differ -> THP early-stops up to 300 ep (patience 24),")
    lines.append("        SuTraN/ED-LSTM early-stop up to 200 ep. 'min/ep' is the hardware-cost-comparable column.")
    lines.append("        SuTraN/ED-LSTM wall from ckpt-mtime span; 'EST' = epochs*mean(min/ep). THP wall = sum of per-epoch train t.")
    lines.append("        Hawkes = classical MLE, single-pass fit (no epochs; covers uni+marked variants) -> 'epochs'=1, min/ep == wall.")
    lines.append("")
    hdr=f"{'dataset':<14}"+"".join(f"{n+' min':>16}" for n in names)
    lines.append(hdr); lines.append("-"*len(hdr))
    agg={n:[] for n in names}; aggmpe={n:[] for n in names}
    for ds in DATASETS:
        cells=f"{ds:<14}"
        for n in names:
            ep,wall,src=table[ds][n]
            if wall is not None:
                tag="*" if src in("EST(ep*rate)","pruned_no_span") else ""
                cells+=f"{f'{wall:.2f}{tag}':>16}"; agg[n].append(wall)
                if ep: aggmpe[n].append(wall/ep)
            else:
                cells+=f"{'NA':>16}"
        lines.append(cells)
    lines.append("-"*len(hdr))
    lines.append(f"{'TOTAL min':<14}"+"".join(f"{(f'{sum(agg[n]):.1f}' if agg[n] else 'NA'):>16}" for n in names))
    lines.append(f"{'mean min/ep':<14}"+"".join(f"{(f'{statistics.mean(aggmpe[n]):.3f}' if aggmpe[n] else 'NA'):>16}" for n in names))
    lines.append("(* = estimated/derived wall-time)")
    txt="\n".join(lines)
    with open(os.path.join(OUTDIR,"time_consumption_comparison.txt"),"w") as fh:
        fh.write(txt+"\n")
    print(txt)
    print(f"\nWrote {csv_path}")
    print(f"per-model mean min/epoch from spans: "+
          ", ".join(f"{k.split('_')[0]}={v:.3f}" if v else f"{k}=NA" for k,v in rates.items()))

if __name__=="__main__":
    main()
