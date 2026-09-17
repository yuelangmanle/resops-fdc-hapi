from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[2];O=R/'outputs/revision_20260917/cohort_processing_labels';s=pd.read_csv(O/'reservoir_summary.csv');a=pd.read_csv(O/'annual_labels.csv');counts=[];diffs=[]
for r in s[s.status=='processed'].itertuples():
 d=pd.read_csv(O/f'daily/{r.ID}.csv.gz',parse_dates=['date']);assert not d.date.duplicated().any()
 assert ((~d.long_gap_partial_fill)|(d.linear_fill&(d.missing_run_length>7))).all()
 for col in ['linear_fill','long_gap_partial_fill','gradient_removed']:
  assert int(d[col].sum())==int(getattr(r,col+'_days'))
 for v in ['inflow','outflow']:
  x=d['replay_'+v];y=d['cars_'+v];both=np.isfinite(x)&np.isfinite(y);assert np.allclose(x[both],y[both],atol=1e-8,rtol=0)
  bad=(np.isfinite(x)!=np.isfinite(y));assert bad.sum()==getattr(r,v+'_mask_disagreements')
  for dt in d.loc[bad,'date']:diffs.append(dict(ID=r.ID,variable=v,date=str(dt.date()),type='finite_mask'))
  assert not bad[d.date<='2019-09-30'].any()
 counts.append(len(d))
assert len(counts)==275 and len(s)==276
result={'reservoirs':276,'daily_files_verified':275,'daily_rows':sum(counts),'all_common_finite_values_match':True,'full_calendar_period_exact_reservoirs':int((s.exact_flow_replay==True).sum()),'remaining_differences':diffs,'all_275_exact_on_available_labels_through_2019_09_30':True,'coverage_warning':'Calendar labels begin 1980-01-01: WY1980 is partial; WY2020 also partial; no annual eligibility recomputed from these files.'}
(O/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
