"""Exploratory past-information failure warning; protocol fixed before fitting."""
from pathlib import Path
import ast,io,zipfile,json,calendar
import numpy as np,pandas as pd
from scipy.stats import wasserstein_distance,spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
R=Path(__file__).resolve().parents[2];O=R/'outputs/revision_20260917/failure_warning';O.mkdir(exist_ok=True,parents=True);P=R/'outputs/revision_20260916/phase2';RAW=R/'data/raw/ResOpsUS+CARS_v10/v1.0/time_series/csv'
for node in ast.parse((R/'code/03_analysis/cohort_source_sensitivity.py').read_text()).body:
 if isinstance(node,ast.FunctionDef) and node.name in ['prepare','valid','eligible']:exec(compile(ast.Module(body=[node],type_ignores=[]),'helpers','exec'))
t=pd.read_csv(R/'outputs/revision_20260917/gap_sensitivity/window_pairs.csv');t=t[t.source.isin(['raw','cars'])&~t.flagged].copy();keys=set(map(tuple,t[t.source=='raw'][['ID','period']].to_numpy()))&set(map(tuple,t[t.source=='cars'][['ID','period']].to_numpy()));t=t[[tuple(x) in keys for x in t[['ID','period']].to_numpy()]].copy()
saved=pd.read_csv(P/'chronological_predictions.csv').query("task=='US_chronological_basin' and sample=='all_candidates' and model=='rf'");foldmap=saved.drop_duplicates('MAIN_BAS').set_index('MAIN_BAS').fold.astype(int).to_dict()
extra=[]
with zipfile.ZipFile(R/'data/raw/phase2_resops_original/ResOpsUS 2.zip') as z:
 for gid,group in t.groupby('ID'):
  up=pd.read_csv(io.BytesIO(z.read(f'ResOpsUS/time_series_all/ResOpsUS_{gid}.csv')),parse_dates=['date']).set_index('date');assert not up.index.duplicated().any();cars=pd.read_csv(RAW/f'{gid}.csv',parse_dates=['date']).set_index('date')
  for row in group.itertuples():
   lo=f'{row.early_start-1}-10-01';hi=f'{row.early_end}-09-30';ix=pd.date_range(lo,hi,freq='D');raw=up.reindex(ix);quality={'paired_invalid_fraction':float((~(np.isfinite(raw[['inflow','outflow']]).all(axis=1)&(raw[['inflow','outflow']]>=0).all(axis=1))).mean()),'negative_inflow_fraction':float((raw.inflow<0).mean()),'missing_inflow_fraction':float((~np.isfinite(raw.inflow)).mean())}
   source=up if row.source=='raw' else cars;d=eligible(prepare(source.loc[lo:hi].rename_axis('date').reset_index()));annual=[]
   for y,a in d.groupby('wy'):
    if min(a.inflow.mean(),a.outflow.mean())>0:annual.append((y,wasserstein_distance(a.inflow/a.inflow.mean(),a.outflow/a.outflow.mean())))
   years=np.array([a[0] for a in annual]);sh=np.array([a[1] for a in annual]);extra.append(dict(source=row.source,ID=gid,period=row.period,feature_last_date=hi,annual_shape_usable_years=len(annual),annual_shape_sd=float(np.std(sh)) if len(annual)>=8 else np.nan,annual_shape_trend_abs=float(abs(np.polyfit(years-years.mean(),sh,1)[0])) if len(annual)>=8 else np.nan,**quality))
t=t.merge(pd.DataFrame(extra),on=['source','ID','period'],validate='one_to_one');t.to_csv(O/'features.csv',index=False)
hydro=['early_shape','annual_shape_sd','annual_shape_trend_abs','early_cv','early_concentration'];quality=['paired_invalid_fraction','negative_inflow_fraction','missing_inflow_fraction'];pred=[];details=[];coefs=[]
for source,d in t.groupby('source'):
 train=d[d.period=='train'];test=d[d.period=='test']
 for fold in range(5):
  held={b for b,f in foldmap.items() if f==fold};tr=train[~train.MAIN_BAS.isin(held)];te=test[test.MAIN_BAS.isin(held)]
  if te.empty:continue
  assert tr.late_end.max()<te.early_start.min() and not set(tr.MAIN_BAS)&set(te.MAIN_BAS)
  threshold=float(tr.abs_shape_change.quantile(.75));y=(tr.abs_shape_change>threshold).astype(int);yt=(te.abs_shape_change>threshold).astype(int);prevalence=float(y.mean());assert y.nunique()==2
  details.append(dict(source=source,fold=fold,n_train=len(tr),n_test=len(te),threshold=threshold,train_event_rate=prevalence,train_ids=','.join(map(str,tr.ID)),test_ids=','.join(map(str,te.ID))))
  for model,features in [('prevalence',[]),('hydrology',hydro),('quality',quality),('combined',hydro+quality)]:
   if features:
    m=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=1,solver='lbfgs',max_iter=3000));m.fit(tr[features],y);prob=m.predict_proba(te[features])[:,1]
    coefs.extend(dict(source=source,fold=fold,model=model,feature=f,standardized_coefficient=float(c)) for f,c in zip(features,m[-1].coef_[0]))
   else:prob=np.repeat(prevalence,len(te))
   pred.extend(dict(source=source,fold=fold,model=model,ID=int(r.ID),MAIN_BAS=int(r.MAIN_BAS),error=float(r.abs_shape_change),threshold=threshold,event=int(v),probability=float(p),baseline=prevalence) for r,v,p in zip(te.itertuples(),yt,prob))
p=pd.DataFrame(pred);p.to_csv(O/'predictions.csv',index=False);pd.DataFrame(details).to_csv(O/'folds.csv',index=False);pd.DataFrame(coefs).to_csv(O/'coefficients.csv',index=False)
rows=[]
for (source,model),d in p.groupby(['source','model']):
 y=d.event.to_numpy();q=d.probability.to_numpy();base=d.baseline.to_numpy();g=d.MAIN_BAS.to_numpy();ug=np.unique(g);rng=np.random.default_rng(42);boot=[]
 for _ in range(2000):
  ix=np.concatenate([np.flatnonzero(g==v) for v in rng.choice(ug,len(ug),replace=True)]);boot.append(float(np.mean((y[ix]-q[ix])**2-(y[ix]-base[ix])**2)))
 top=d.sort_values(['probability','ID'],ascending=[False,True]).head(int(np.ceil(len(d)*.2)))
 rows.append(dict(source=source,model=model,n=len(d),groups=len(ug),events=int(y.sum()),event_rate=float(y.mean()),brier=brier_score_loss(y,q),brier_difference=float(np.mean((y-q)**2-(y-base)**2)),difference_low=float(np.quantile(boot,.025)),difference_high=float(np.quantile(boot,.975)),auc=roc_auc_score(y,q),average_precision=average_precision_score(y,q),risk_error_spearman=float(spearmanr(q,d.error).statistic),top20_event_rate=float(top.event.mean()),top20_lift=float(top.event.mean()/y.mean())))
s=pd.DataFrame(rows);s.to_csv(O/'scores.csv',index=False);print(s.round(5).to_string(index=False));(O/'verification.json').write_text(json.dumps({'feature_rows':len(t),'folds':len(details),'feature_cutoffs':'training features end 1989-09-30; testing features end 2009-09-30','threshold_source':'outer-training error 75th percentile only','temporal_and_basin_checks':'passed','hydrology_features':hydro,'quality_features':quality},indent=2))
