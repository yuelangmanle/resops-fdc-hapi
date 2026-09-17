"""Retrospective source sensitivity; original outputs and records remain untouched."""
from pathlib import Path
import ast,io,json,zipfile,hashlib,calendar
import numpy as np,pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.model_selection import GroupKFold
R=Path(__file__).resolve().parents[2];O=R/'outputs/revision_20260917/cohort_source';O.mkdir(parents=True,exist_ok=True)
P=R/'outputs/revision_20260916/phase2';RAW=R/'data/raw/ResOpsUS+CARS_v10/v1.0/time_series/csv'
# Reuse the exact frozen statistics and evaluation definitions without running its IO/main.
source=R/'code/03_analysis/revision_phase2.py';tree=ast.parse(source.read_text())
for node in tree.body:
 if isinstance(node,ast.FunctionDef) and node.name in ['signature','pair','evaluate']:
  exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'))
features=['early_shape','early_ratio','early_low10','early_cv','early_zero_delta','early_concentration','early_trend']
frozen=pd.read_csv(P/'frozen_window_pairs.csv');screen=pd.read_csv(P/'physical_screen.csv').query("country=='US'")
meta=frozen.query("country=='US'").drop_duplicates('ID').set_index('ID')
base_test=frozen.query("country=='US' and period=='test'").reset_index(drop=True)
# Reuse saved folds: rerunning GroupKFold across library versions can break ties differently.
saved=pd.read_csv(P/'chronological_predictions.csv').query("task=='US_chronological_basin' and sample=='all_candidates' and model=='rf'")
assert saved.groupby('MAIN_BAS').fold.nunique().max()==1
foldmap=saved.drop_duplicates('MAIN_BAS').set_index('MAIN_BAS').fold.astype(int).to_dict()

def prepare(d):
 d=d.copy();d['date']=pd.to_datetime(d.date,errors='coerce');assert not d.date.dropna().duplicated().any()
 d=d[d.date.notna()].copy();d['wy']=d.date.dt.year+(d.date.dt.month>=10).astype(int)
 return d[d.wy.between(1980,2019)].sort_values('date')
def valid(d):return np.isfinite(d[['inflow','outflow']]).all(axis=1)&(d[['inflow','outflow']]>=0).all(axis=1)
def eligible(d):
 d=d[valid(d)].copy();counts=d.groupby('wy').size();years=[int(y) for y,n in counts.items() if n/(366 if calendar.isleap(int(y)) else 365)>=.9]
 return d[d.wy.isin(years)]
audit=[];annual=[];pairs=[];concentration=[];evidence=[]
archive=R/'data/raw/phase2_resops_original/ResOpsUS 2.zip'
with zipfile.ZipFile(archive) as z:
 names=set(z.namelist())
 for step,r in enumerate(screen.itertuples(),1):
  gid=int(r.ID);member=f'ResOpsUS/time_series_all/ResOpsUS_{gid}.csv';cp=RAW/f'{gid}.csv'
  if member not in names:
   audit.append(dict(ID=gid,status='missing_upstream'));continue
  payload=z.read(member);upraw=pd.read_csv(io.BytesIO(payload));duplicate_days=int(pd.to_datetime(upraw.date,errors='coerce').dropna().duplicated(keep=False).sum())
  # Ambiguous same-date records cannot be resolved by choosing an arbitrary row.
  up=prepare(upraw.iloc[:0] if duplicate_days else upraw);ca=prepare(pd.read_csv(cp))
  j=ca.merge(up,on='date',how='outer',suffixes=('_cars','_upstream'),validate='one_to_one')
  ar=dict(ID=gid,name=r.name,status='duplicate_dates_excluded' if duplicate_days else 'matched',duplicate_date_rows=duplicate_days,upstream_member_sha256=hashlib.sha256(payload).hexdigest(),cars_sha256=hashlib.sha256(cp.read_bytes()).hexdigest())
  for col in ['inflow','outflow']:
   a=j[col+'_cars'];b=j[col+'_upstream'];finite=np.isfinite(a)&np.isfinite(b);err=(a[finite]-b[finite]).abs();v=valid(ca)
   ar.update({col+'_n_compared':int(finite.sum()),col+'_mae':float(err.mean()),col+'_negative_to_nonnegative':int(((b<0)&(a>=0)&np.isfinite(a)).sum()),col+'_missing_to_finite':int((~np.isfinite(b)&np.isfinite(a)).sum()),col+'_above_tolerance_days':int((err>.0015+1e-6*b[finite].abs()).sum())})
   if col=='inflow' and len(err):
    for idx in err.nlargest(3).index:evidence.append(dict(ID=gid,date=str(j.loc[idx,'date'].date()),cars=float(a[idx]),upstream=float(b[idx]),abs_difference=float(abs(a[idx]-b[idx]))))
  dates=ca.loc[valid(ca),'date'];dates=dates[dates.isin(up.loc[valid(up),'date'])]
  datasets={'cars':ca,'upstream':up,'cars_common_days':ca[ca.date.isin(dates)],'upstream_common_days':up[up.date.isin(dates)]}
  for label,d in datasets.items():
   e=eligible(d)
   if label in ['cars','upstream']:
    counts=d[valid(d)].groupby('wy').size()
    for y in range(1980,2020):
     a=d[d.wy==y];n=int(counts.get(y,0));annual.append(dict(ID=gid,source=label,wy=y,valid_days=n,eligible=n/(366 if calendar.isleap(y) else 365)>=.9,negative_inflow=int((a.inflow<0).sum())))
   if gid not in meta.index:continue
   for start in [1980,2000]:
    q=pair(e,start)
    if q:
     pairs.append(dict(source=label,country='US',ID=gid,period='train' if start==1980 else 'test',flagged=bool(r.flagged),MAIN_BAS=meta.loc[gid,'MAIN_BAS'],**q))
     if label in ['cars','upstream']:
      for tag,lo,hi in [('early',start,start+9),('late',start+10,start+19)]:
       a=e[e.wy.between(lo,hi)].inflow;concentration.append(dict(ID=gid,source=label,start=start,window=tag,n_days=len(a),top10_inflow_share=float(a.nlargest(10).sum()/a.sum())))
  if duplicate_days:
   for key in list(ar):
    if key.endswith('_missing_to_finite'):ar[key]=np.nan
  audit.append(ar)
  if step%40==0:print('Audited',step,'/',len(screen),flush=True)
pd.DataFrame(audit).to_csv(O/'source_audit.csv',index=False);pd.DataFrame(annual).to_csv(O/'annual_coverage.csv',index=False);pd.DataFrame(evidence).to_csv(O/'largest_source_differences.csv',index=False);pd.DataFrame(concentration).to_csv(O/'extreme_day_concentration.csv',index=False)
t=pd.DataFrame(pairs);t.to_csv(O/'source_window_pairs.csv',index=False)
# Verify replay against frozen CARS values before interpreting any sensitivity.
c=t[t.source=='cars'];check=c.merge(frozen.query("country=='US'"),on=['ID','period'],suffixes=('_new','_frozen'),validate='one_to_one')
assert len(c)==len(check)==len(frozen.query("country=='US'"))
for col in features+['late_shape','early_nyears','late_nyears']:
 assert np.allclose(check[col+'_new'],check[col+'_frozen'],equal_nan=True,rtol=1e-10,atol=1e-12),col
common=set(map(tuple,t[t.source=='cars'][['ID','period']].to_numpy()))&set(map(tuple,t[t.source=='upstream'][['ID','period']].to_numpy()))
scenarios={k:t[t.source==k].copy() for k in ['cars','upstream','cars_common_days','upstream_common_days']}
for k in ['cars','upstream']:scenarios[k+'_common_cohort']=scenarios[k][[tuple(x) in common for x in scenarios[k][['ID','period']].to_numpy()]].copy()
assert set(map(tuple,scenarios['cars_common_days'][['ID','period']].to_numpy()))==set(map(tuple,scenarios['upstream_common_days'][['ID','period']].to_numpy()))
models=['mean','persistence','ridge','rf'];scores=[];predictions=[];excluded=[]
def predict(tr,te,model):
 if model=='persistence':return te.early_shape.to_numpy()
 if model=='mean':return np.repeat(tr.late_shape.mean(),len(te))
 est=Ridge(alpha=10,solver='lsqr',tol=1e-10) if model=='ridge' else RandomForestRegressor(n_estimators=200,min_samples_leaf=8,max_features=.7,random_state=42,n_jobs=4)
 m=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),est);m.fit(tr[features],tr.late_shape);return m.predict(te[features])
for label,d in scenarios.items():
 tr=d[d.period=='train'];te=d[d.period=='test'].reset_index(drop=True)
 unknown=~te.MAIN_BAS.isin(foldmap);excluded.extend(dict(scenario=label,ID=int(r.ID),reason='no_original_fold') for r in te[unknown].itertuples());te=te[~unknown].reset_index(drop=True)
 if len(te)<10 or te.MAIN_BAS.nunique()<5:continue
 for model in models:
  p=np.full(len(te),np.nan)
  for fold in range(5):
   held={b for b,f in foldmap.items() if f==fold};ix=np.flatnonzero(te.MAIN_BAS.isin(held));train=tr[~tr.MAIN_BAS.isin(held)]
   if not len(ix):continue
   assert len(train)>=10 and not set(train.MAIN_BAS)&held
   assert train.late_end.max()<te.early_start.min()
   p[ix]=predict(train,te.iloc[ix],model)
   predictions.extend(dict(scenario=label,task='US',model=model,ID=int(te.iloc[j].ID),MAIN_BAS=int(te.iloc[j].MAIN_BAS),fold=fold,n_train=len(train),observed=float(te.iloc[j].late_shape),predicted=float(p[j])) for j in ix)
  assert np.isfinite(p).all()
  scores.append(evaluate(te,p,'US_chronological_basin',model,label))
  if label in ['cars','upstream','cars_common_cohort','upstream_common_cohort']:
   br=frozen.query("country=='BR' and period=='test'").reset_index(drop=True);bp=predict(tr,br,model);scores.append(evaluate(br,bp,'BR_unchanged_targets',model,label))
   predictions.extend(dict(scenario=label,task='BR',model=model,ID=int(r.ID),MAIN_BAS=int(r.MAIN_BAS),fold=-1,n_train=len(tr),observed=float(r.late_shape),predicted=float(v)) for r,v in zip(br.itertuples(),bp))
 print('Models complete:',label,'train',len(tr),'test',len(te),flush=True)
pd.DataFrame(scores).to_csv(O/'benchmark.csv',index=False);pd.DataFrame(predictions).to_csv(O/'predictions.csv',index=False);pd.DataFrame(excluded,columns=['scenario','ID','reason']).to_csv(O/'excluded_unknown_fold.csv',index=False)
q=t[t.source=='cars'].merge(t[t.source=='upstream'],on=['ID','period'],suffixes=('_cars','_upstream'));q.to_csv(O/'paired_source_metrics.csv',index=False)
summary=dict(audited=len(audit),matched=sum(a['status']=='matched' for a in audit),frozen_cars_replay_rows=len(check),all_frozen_metrics_match=True,unknown_fold_exclusions=len(excluded),source_counts=t.groupby(['source','period']).size().to_dict())
summary['source_counts']={str(k):int(v) for k,v in summary['source_counts'].items()}
(O/'verification.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));print(pd.DataFrame(scores)[['task','sample','model','n','mae','difference_ci_low','difference_ci_high']].round(5).to_string(index=False))
(O/'provenance.json').write_text(json.dumps({'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'frozen_function_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((R/'docs/revision_20260917/SOURCE_SENSITIVITY_PROTOCOL.md').read_bytes()).hexdigest(),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'note':'upstream is a release file, not ground truth; no values corrected; retrospective sensitivity'},indent=2))
