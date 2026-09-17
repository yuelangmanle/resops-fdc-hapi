"""Fixed complete-gap sensitivity; no raw overwrites or test-set tuning."""
from pathlib import Path
import ast,io,json,zipfile,calendar,hashlib
import numpy as np,pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error
R=Path(__file__).resolve().parents[2];O=R/'outputs/revision_20260917/gap_sensitivity';O.mkdir(exist_ok=True,parents=True);P=R/'outputs/revision_20260916/phase2';RAW=R/'data/raw/ResOpsUS+CARS_v10/v1.0/time_series/csv'
for path,names in [('revision_phase2.py',['signature','pair','evaluate']),('cohort_source_sensitivity.py',['prepare','valid','eligible','predict'])]:
 for node in ast.parse((R/'code/03_analysis'/path).read_text()).body:
  if isinstance(node,ast.FunctionDef) and node.name in names:exec(compile(ast.Module(body=[node],type_ignores=[]),path,'exec'))
features=['early_shape','early_ratio','early_low10','early_cv','early_zero_delta','early_concentration','early_trend']
frozen=pd.read_csv(P/'frozen_window_pairs.csv');screen=pd.read_csv(P/'physical_screen.csv').query("country=='US'");meta=frozen.query("country=='US'").drop_duplicates('ID').set_index('ID');saved=pd.read_csv(P/'chronological_predictions.csv').query("task=='US_chronological_basin' and sample=='all_candidates' and model=='rf'");foldmap=saved.drop_duplicates('MAIN_BAS').set_index('MAIN_BAS').fold.astype(int).to_dict()

def fill_complete(series,limit):
 m=series.isna();groups=m.ne(m.shift()).cumsum();length=m.groupby(groups).transform('sum');interpolated=series.interpolate('linear',limit_area='inside')
 return interpolated.where(~m|(length<=limit))
# Boundary semantics: no partial filling of long gaps or extrapolation.
x=pd.Series([np.nan,0.,np.nan,np.nan,3.,np.nan]);assert fill_complete(x,1).isna().equals(x.isna());y=fill_complete(x,3);assert y.iloc[2]==1 and y.iloc[3]==2 and pd.isna(y.iloc[0]) and pd.isna(y.iloc[-1])
rows=[];counts=[];skips=[];flow_labels=['raw','gap1','gap3','gap7','cars']
with zipfile.ZipFile(R/'data/raw/phase2_resops_original/ResOpsUS 2.zip') as archive:
 for step,r in enumerate(screen.itertuples(),1):
  gid=int(r.ID);up=pd.read_csv(io.BytesIO(archive.read(f'ResOpsUS/time_series_all/ResOpsUS_{gid}.csv')),parse_dates=['date']).set_index('date')[['inflow','outflow']];ca=pd.read_csv(RAW/f'{gid}.csv',parse_dates=['date']).set_index('date')[['inflow','outflow']]
  dup=up.index.duplicated().any()
  if dup:skips.append(gid)
  datasets={'cars':ca}
  if not dup:
   up=up.reindex(pd.date_range('1979-10-01','2019-09-30',freq='D')).mask(lambda x:x<0);datasets['raw']=up
   for limit in [1,3,7]:
    d=up.copy()
    for start in [1980,1990,2000,2010]:
     ix=d.loc[f'{start-1}-10-01':f'{start+9}-09-30'].index;d.loc[ix,'inflow']=fill_complete(up.loc[ix,'inflow'],limit)
    datasets[f'gap{limit}']=d
    counts.append(dict(ID=gid,limit=limit,filled_days=int((up.inflow.isna()&d.inflow.notna()).sum())))
  frames={k:prepare(d.rename_axis('date').reset_index()) for k,d in datasets.items()}
  if not dup:
   common=set.intersection(*(set(d.loc[valid(d),'date']) for d in frames.values()))
   for k,d in list(frames.items()):frames[k+'_common_dates']=d[d.date.isin(common)].copy()
   for k in ['gap1','gap3','gap7']:
    a=frames[k+'_common_dates'].set_index('date');b=frames['raw_common_dates'].set_index('date');assert a[['inflow','outflow']].equals(b[['inflow','outflow']])
  if gid not in meta.index:continue
  for label,d in frames.items():
   d=eligible(d)
   for start in [1980,2000]:
    q=pair(d,start)
    if q:rows.append(dict(source=label,ID=gid,period='train' if start==1980 else 'test',MAIN_BAS=meta.loc[gid,'MAIN_BAS'],flagged=bool(r.flagged),**q))
  if step%60==0:print('processed',step,flush=True)
t=pd.DataFrame(rows);t.to_csv(O/'window_pairs.csv',index=False);pd.DataFrame(counts).to_csv(O/'filled_days.csv',index=False)
common=set.intersection(*(set(map(tuple,t[t.source==s][['ID','period']].to_numpy())) for s in flow_labels))
scenarios={k:t[t.source==k].copy() for k in t.source.unique()}
for k in flow_labels:scenarios[k+'_common_cohort']=scenarios[k][[tuple(x) in common for x in scenarios[k][['ID','period']].to_numpy()]].copy()
# Published replay must reproduce all frozen input and target metrics.
a=t[t.source=='cars'].merge(frozen.query("country=='US'"),on=['ID','period'],suffixes=('_new','_old'));assert len(a)==344
for col in features+['late_shape']:assert np.allclose(a[col+'_new'],a[col+'_old'],equal_nan=True,atol=1e-12,rtol=1e-10)
scores=[];predictions=[];trainlog=[]
for scenario,d in scenarios.items():
 tr=d[d.period=='train'];te=d[d.period=='test'].reset_index(drop=True);assert te.MAIN_BAS.isin(foldmap).all()
 for model in ['mean','persistence','ridge','rf']:
  pred=np.full(len(te),np.nan)
  for fold in range(5):
   held={b for b,f in foldmap.items() if f==fold};ix=np.flatnonzero(te.MAIN_BAS.isin(held));train=tr[~tr.MAIN_BAS.isin(held)]
   if not len(ix):continue
   assert len(train)>=10 and not set(train.MAIN_BAS)&held and train.late_end.max()<te.early_start.min()
   pred[ix]=predict(train,te.iloc[ix],model)
   if model=='rf':trainlog.append(dict(scenario=scenario,fold=fold,n_train=len(train),n_test=len(ix),train_ids=','.join(map(str,train.ID.astype(int))),test_ids=','.join(map(str,te.iloc[ix].ID.astype(int)))))
  assert np.isfinite(pred).all();scores.append(evaluate(te,pred,'US_chronological_basin',model,scenario))
  predictions.extend(dict(scenario=scenario,model=model,ID=int(r.ID),MAIN_BAS=int(r.MAIN_BAS),observed=float(r.late_shape),predicted=float(v)) for r,v in zip(te.itertuples(),pred))
 print('finished',scenario,'train',len(tr),'test',len(te),flush=True)
pd.DataFrame(scores).to_csv(O/'benchmark.csv',index=False);pd.DataFrame(predictions).to_csv(O/'predictions.csv',index=False);pd.DataFrame(trainlog).to_csv(O/'fold_membership.csv',index=False)
(O/'verification.json').write_text(json.dumps({'duplicate_excluded':skips,'frozen_cars_pairs_match':len(a),'complete_gap_boundary_checks':'passed','common_date_upstream_values_identical':'passed','temporal_and_basin_separation':'passed','scenarios':len(scenarios),'protocol_sha256':hashlib.sha256((R/'docs/revision_20260917/GAP_SENSITIVITY_PROTOCOL.md').read_bytes()).hexdigest()},indent=2))
print(pd.DataFrame(scores).query("model in ['persistence','rf']")[['sample','model','n','mae','difference_ci_low','difference_ci_high']].to_string(index=False))
