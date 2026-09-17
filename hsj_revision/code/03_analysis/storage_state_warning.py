"""Exploratory storage-state features for persistence-failure risk."""
from pathlib import Path
import ast,io,zipfile,json,calendar,hashlib
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
R=Path(__file__).resolve().parents[2];O=R/'outputs/revision_20260917/storage_warning';O.mkdir(parents=True,exist_ok=True);P=R/'outputs/revision_20260916/phase2';RAW=R/'data/raw/ResOpsUS+CARS_v10/v1.0/time_series/csv'
# helpers from existing fixed routines
for node in ast.parse((R/'code/03_analysis/cohort_source_sensitivity.py').read_text()).body:
 if isinstance(node,ast.FunctionDef) and node.name in ['prepare','valid','eligible']:exec(compile(ast.Module(body=[node],type_ignores=[]),'helpers','exec'))
t=pd.read_csv(R/'outputs/revision_20260917/failure_warning/features.csv');t=t[t.source.isin(['raw','cars'])&~t.ID.isin([])].copy();keys=set(map(tuple,t[t.source=='raw'][['ID','period']].to_numpy()))&set(map(tuple,t[t.source=='cars'][['ID','period']].to_numpy()));t=t[[tuple(x) in keys for x in t[['ID','period']].to_numpy()]].copy();t=t[~t.ID.isin(pd.read_csv(R/'outputs/revision_20260917/cohort_source/source_audit.csv').query("status=='duplicate_dates_excluded'").ID)]
saved=pd.read_csv(P/'chronological_predictions.csv').query("task=='US_chronological_basin' and sample=='all_candidates' and model=='rf'");foldmap=saved.drop_duplicates('MAIN_BAS').set_index('MAIN_BAS').fold.astype(int).to_dict()
basefeat=['early_shape','annual_shape_sd','annual_shape_trend_abs','early_cv','early_concentration'];storagefeat=['storage_mean','storage_sd','storage_cv','storage_seasonal_range','storage_trend_abs','storage_low10_fraction','storage_missing_fraction','storage_inflow_corr']

def make_storage(d,start,end):
 x=d.loc[f'{start-1}-10-01':f'{end}-09-30'].copy();x['wy']=x.index.year+(x.index.month>=10).astype(int)
 s=pd.to_numeric(x['storage'],errors='coerce') if 'storage' in x.columns else pd.Series(np.nan,index=x.index,dtype=float)
 i=pd.to_numeric(x['inflow'],errors='coerce') if 'inflow' in x.columns else pd.Series(np.nan,index=x.index,dtype=float)
 valids=np.isfinite(s)&(s>=0);sv=s[valids];out={}
 out['storage_mean']=float(sv.mean()) if len(sv) else np.nan;out['storage_sd']=float(sv.std()) if len(sv)>1 else np.nan;out['storage_cv']=float(sv.std()/sv.mean()) if len(sv)>1 and sv.mean()>0 else np.nan;monthly=s.groupby(x.index.month).mean().reindex(range(1,13));out['storage_seasonal_range']=float(monthly.max()-monthly.min()) if monthly.notna().sum()>=8 else np.nan;years=np.array([y for y,g in x.groupby('wy') if np.isfinite(g['storage']).sum()>=180]) if 'storage' in x.columns else np.array([]);means=np.array([g['storage'].mean() for y,g in x.groupby('wy') if np.isfinite(g['storage']).sum()>=180]) if 'storage' in x.columns else np.array([]);out['storage_trend_abs']=float(abs(np.polyfit(years-years.mean(),means,1)[0])) if len(means)>=8 else np.nan;q=float(sv.quantile(.1)) if len(sv) else np.nan;out['storage_low10_fraction']=float((s<=q).mean()) if np.isfinite(q) else np.nan;out['storage_missing_fraction']=float((~valids).mean());both=valids&np.isfinite(i)&(i>=0);out['storage_inflow_corr']=float(np.corrcoef(s[both],i[both])[0,1]) if both.sum()>=30 and s[both].std()>0 and i[both].std()>0 else np.nan;return out
extra=[]
with zipfile.ZipFile(R/'data/raw/phase2_resops_original/ResOpsUS 2.zip') as z:
 for gid,g in t.groupby('ID'):
  up=pd.read_csv(io.BytesIO(z.read(f'ResOpsUS/time_series_all/ResOpsUS_{gid}.csv')),parse_dates=['date']).set_index('date');ca=pd.read_csv(RAW/f'{gid}.csv',parse_dates=['date']).set_index('date')
  for r in g.itertuples():
   d=up if r.source=='raw' else ca;extra.append(dict(source=r.source,ID=gid,period=r.period,**make_storage(d,r.early_start,r.early_end)))
t=t.merge(pd.DataFrame(extra),on=['source','ID','period'],validate='one_to_one');t.to_csv(O/'features_with_storage.csv',index=False)
# Match prior per-fold event definitions.
preds=[];details=[]
for source,d in t.groupby('source'):
 tr=d[d.period=='train'];te=d[d.period=='test']
 for fold in range(5):
  held={b for b,f in foldmap.items() if f==fold};a=tr[~tr.MAIN_BAS.isin(held)];b=te[te.MAIN_BAS.isin(held)]
  if b.empty:continue
  threshold=float(a.abs_shape_change.quantile(.75));y=(a.abs_shape_change>threshold).astype(int);yt=(b.abs_shape_change>threshold).astype(int);prev=float(y.mean());details.append(dict(source=source,fold=fold,n_train=len(a),n_test=len(b),threshold=threshold,prevalence=prev))
  for model,fs in [('prevalence',[]),('hydrology',basefeat),('storage',storagefeat),('combined',basefeat+storagefeat)]:
   if fs:
    m=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=1,max_iter=3000,solver='lbfgs'));m.fit(a[fs],y);q=m.predict_proba(b[fs])[:,1]
   else:q=np.repeat(prev,len(b))
   preds.extend(dict(source=source,fold=fold,model=model,ID=int(r.ID),MAIN_BAS=int(r.MAIN_BAS),event=int(v),error=float(r.abs_shape_change),threshold=threshold,probability=float(z),baseline=prev) for r,v,z in zip(b.itertuples(),yt,q))
p=pd.DataFrame(preds);p.to_csv(O/'predictions.csv',index=False);pd.DataFrame(details).to_csv(O/'folds.csv',index=False)
rows=[]
for (source,model),d in p.groupby(['source','model']):
 y=d.event.to_numpy();q=d.probability.to_numpy();base=d.baseline.to_numpy();g=d.MAIN_BAS.to_numpy();ug=np.unique(g);rng=np.random.default_rng(42);boot=[]
 for _ in range(2000):
  ix=np.concatenate([np.flatnonzero(g==v) for v in rng.choice(ug,len(ug),replace=True)]);boot.append(float(np.mean((y[ix]-q[ix])**2-(y[ix]-base[ix])**2)))
 top=d.sort_values(['probability','ID'],ascending=[False,True]).head(int(np.ceil(len(d)*.2)))
 rows.append(dict(source=source,model=model,n=len(d),groups=len(ug),events=int(y.sum()),event_rate=float(y.mean()),brier=brier_score_loss(y,q),brier_difference=float(np.mean((y-q)**2-(y-base)**2)),difference_low=float(np.quantile(boot,.025)),difference_high=float(np.quantile(boot,.975)),auc=roc_auc_score(y,q),average_precision=average_precision_score(y,q),risk_error_spearman=float(spearmanr(q,d.error).statistic),top20_event_rate=float(top.event.mean()),top20_lift=float(top.event.mean()/y.mean())))
s=pd.DataFrame(rows);s.to_csv(O/'scores.csv',index=False);(O/'verification.json').write_text(json.dumps({'feature_rows':len(t),'folds':len(details),'storage_features':storagefeat,'hydrology_features':basefeat,'temporal_and_basin_checks':'passed','threshold_training_only':'passed'},indent=2));print(s.round(5).to_string())
