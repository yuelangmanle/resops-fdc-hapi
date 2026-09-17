"""Check saved benchmark values and quantify the MAE/R-squared trade-off."""
from pathlib import Path
import json
import pandas as pd
R=Path(__file__).resolve().parents[2];P=R/'outputs/revision_20260916/phase2'
O=R/'outputs/revision_20260918';O.mkdir(exist_ok=True)
p=pd.read_csv(P/'chronological_predictions.csv');b=pd.read_csv(P/'chronological_benchmark.csv')
for r in b.itertuples():
    a=p[(p.task==r.task)&(p['sample']==r.sample)&(p.model==r.model)]
    assert abs((a.predicted-a.observed).abs().mean()-r.mae)<1e-10
rows=[]
for model in ['persistence','rf']:
    a=p[(p.task=='US_chronological_basin')&(p['sample']=='all_candidates')&(p.model==model)]
    e=(a.predicted-a.observed).abs()
    rows.append({'model':model,'n':len(a),'median_absolute_error':e.median(),'p90':e.quantile(.9),'p95':e.quantile(.95),'maximum_absolute_error':e.max(),'mean_squared_error':(e**2).mean(),'top10_fraction_squared_error':(e.nlargest(10)**2).sum()/(e**2).sum()})
pd.DataFrame(rows).to_csv(O/'error_distribution_diagnostics.csv',index=False)
(O/'saved_verification.json').write_text(json.dumps({'chronological_rows_checked':len(b),'mae_tolerance':1e-10,'all_passed':True,'scope':'Saved predictions; no model refitting or raw-data rerun'},indent=2))
print(pd.DataFrame(rows).to_string(index=False))
