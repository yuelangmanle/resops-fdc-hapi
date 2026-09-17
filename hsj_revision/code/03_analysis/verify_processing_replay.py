"""Verify saved stepwise reconstruction and add explicit day provenance labels."""
from pathlib import Path
import json,hashlib,ast
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[2];O=R/'outputs/revision_20260917/processing_replay'
checks=[]
for gid in [938,939]:
 d=pd.read_csv(O/f'{gid}_daily_stages.csv',parse_dates=['date']).set_index('date');s=d.loc['1975-01-01':'2019-12-31']
 for v in ['inflow','outflow']:
  a=s[f'full_replay_{v}'];b=s[f'CARS_{v}'];assert np.array_equal(np.isfinite(a),np.isfinite(b));m=np.isfinite(a)&np.isfinite(b);assert np.allclose(a[m],b[m],atol=1e-10,rtol=0)
  checks.append(dict(ID=gid,variable=v,n=int(m.sum()),max_abs_error=float(abs(a[m]-b[m]).max()),missing_mask_matches=True))
 raw=d.raw_inflow;mask=d.negative_mask_inflow;filtered=d.filtered_inflow;filled=d.linear_limit7_inflow;replay=d.full_replay_inflow
 missing=filtered.isna();runs=missing.ne(missing.shift()).cumsum();length=missing.groupby(runs).transform('sum')
 labels=pd.DataFrame({'upstream_inflow':raw,'upstream_negative':raw<0,'upstream_missing':raw.isna(),'filtered_inflow':filtered,'missing_run_length':length.where(missing,0),'linear_inflow':filled,'filled_in_linear_step':filtered.isna()&filled.notna(),'partially_filled_long_gap':filtered.isna()&filled.notna()&(length>7),'utc_replay_inflow':replay,'utc_step_new_finite_on_date_label':filled.isna()&replay.notna(),'CARS_inflow':d.CARS_inflow})
 labels.to_csv(O/f'{gid}_daily_processing_labels.csv',index_label='date')
# Direct long-gap semantics, including physical spikes bracketed by negative records.
x=pd.Series([0.]+[np.nan]*10+[110.]);y=x.interpolate('linear',limit=7)
assert y.iloc[1:8].notna().all() and y.iloc[8:11].isna().all() and y.iloc[7]==70
# Recompute focal value from reviewed, individually stored steps and known DST offset.
d=pd.read_csv(O/'939_daily_stages.csv',parse_dates=['date']).set_index('date')
expected=d.loc['2007-09-04','linear_limit7_inflow']*(5/24)+d.loc['2007-09-05','linear_limit7_inflow']*(19/24)
assert abs(expected-d.loc['2007-09-05','CARS_inflow'])<1e-10
# All original valid pairs must be preserved by the masking-only stage in annual metrics.
a=pd.read_csv(O/'annual_stage_metrics.csv');p=a.query("stage=='raw'").merge(a.query("stage=='negative_mask'"),on=['ID','wy'],suffixes=('_raw','_masked'))
assert (p.valid_days_raw==p.valid_days_masked).all();assert np.allclose(p.shape_raw,p.shape_masked)
out={'checks':checks,'linear_limit_does_partially_fill_long_gaps':True,'focal_2007_09_05_reconstructed':float(expected),'mask_only_retains_original_valid_pair_metrics':True,'version_claim':'historical candidate reproduces 938/939 inflow/outflow through 2019; exact production commit not attested; not a full-dataset replay'}
(O/'verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
print(a.query('ID==939 and wy==2007').to_string(index=False));print(a.query("wy>=2000").groupby(['ID','stage']).eligible.sum().to_string())
