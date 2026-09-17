"""Exploratory nested main-basin temporal validation; see protocol."""
from pathlib import Path
import json, calendar, hashlib
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import wasserstein_distance, spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/revision_20260916/temporal_validation'
RAW=ROOT/'data/raw/ResOpsUS+CARS_v10/v1.0'; OUT.mkdir(exist_ok=True,parents=True)
SCENARIOS={'primary':(.9,10,False),'coverage80':(.8,10,False),'coverage95':(.95,10,False),'years8':(.9,8,False),'years15':(.9,15,False),'common1990_2019':(.9,10,True)}
TARGETS=['shape','low10','ratio']
def metrics(d):
 i=d.inflow.to_numpy();o=d.outflow.to_numpy();mi=i.mean();mo=o.mean()
 if mi<=0 or mo<=0:return None
 return dict(shape=wasserstein_distance(i/mi,o/mo),low10=(np.quantile(o,.1)-np.quantile(i,.1))/mi,ratio=mo/mi,zero_delta=np.mean(o==0)-np.mean(i==0),in_cv=i.std()/mi)
def build():
 rows=[];audit=[]
 for p in sorted((RAW/'time_series/csv').glob('*.csv')):
  d=pd.read_csv(p);gid=int(p.stem)
  if not {'date','inflow','outflow'}.issubset(d.columns):
   audit.append(dict(GRAND_ID=gid,reason='missing_observed_columns',sha256=hashlib.sha256(p.read_bytes()).hexdigest()));continue
  d['date']=pd.to_datetime(d.date,errors='coerce');gid=int(p.stem)
  duplicates=int(d.date.dropna().duplicated().sum());valid=d.date.notna()&np.isfinite(d[['inflow','outflow']]).all(axis=1)&(d[['inflow','outflow']]>=0).all(axis=1)
  audit.append(dict(GRAND_ID=gid,duplicates=duplicates,valid_days=int(valid.sum()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
  if duplicates:continue
  d=d[valid].sort_values('date').copy();d['wy']=d.date.dt.year+(d.date.dt.month>=10).astype(int)
  counts=d.groupby('wy').size();coverage={y:n/(366 if calendar.isleap(y) else 365) for y,n in counts.items()}
  for name,(cut,ny,common) in SCENARIOS.items():
   good=sorted(y for y,c in coverage.items() if c>=cut)
   if common:ey=[y for y in good if 1990<=y<=2004];ly=[y for y in good if 2005<=y<=2019]
   else:mid=len(good)//2;ey=good[:mid];ly=good[mid:]
   if len(good)<ny or min(len(ey),len(ly))<(10 if common else ny//2):continue
   e=metrics(d[d.wy.isin(ey)]);l=metrics(d[d.wy.isin(ly)])
   if e is None or l is None:continue
   rows.append(dict(scenario=name,GRAND_ID=gid,early_start=min(ey),early_end=max(ey),late_start=min(ly),late_end=max(ly),early_years=len(ey),late_years=len(ly),**{'early_'+k:v for k,v in e.items()},**{'late_'+k:v for k,v in l.items()}))
 t=pd.DataFrame(rows);excluded=t[(t.scenario=='primary')&~t.GRAND_ID.isin([144,1916])].copy();excluded['scenario']='exclude_flagged';t=pd.concat([t,excluded],ignore_index=True)
 pd.DataFrame(audit).to_csv(OUT/'source_audit.csv',index=False);t.to_csv(OUT/'temporal_metrics.csv',index=False)
 return t

def main():
 t=build();print('metrics',t.groupby('scenario').size().to_dict(),flush=True)
 grand=pd.read_csv(RAW/'attributes/grand.csv'); cols=['CAP_MCM','CATCH_SKM','DAM_HGT_M','AREA_SKM','LAT','LON']
 grand[cols]=grand[cols].apply(pd.to_numeric,errors='coerce');grand.loc[:,cols]=grand[cols].replace([-99,-999],np.nan)
 basin=gpd.read_file(ROOT/'data/enhancement_probe/basinatlas/BasinATLAS_v10.gdb',layer='BasinATLAS_v10_lev06',columns=['HYBAS_ID','MAIN_BAS'])
 pts=gpd.GeoDataFrame(grand[['GRAND_ID']],geometry=gpd.points_from_xy(grand.LON,grand.LAT),crs='EPSG:4326')
 match=gpd.sjoin(pts,basin,how='left',predicate='intersects')[['GRAND_ID','HYBAS_ID','MAIN_BAS']]
 match['ambiguous']=match.GRAND_ID.duplicated(keep=False);match.to_csv(OUT/'basin_assignment_audit.csv',index=False)
 match=match[~match.ambiguous].dropna(subset=['MAIN_BAS']);t=t.merge(match,on='GRAND_ID',validate='many_to_one').merge(grand[['GRAND_ID']+cols],on='GRAND_ID',validate='many_to_one')
 t.to_csv(OUT/'analysis_table.csv',index=False);foldrows=[];predrows=[];summaries=[];params=[]
 early=['early_'+k for k in ['shape','low10','ratio','zero_delta','in_cv']]
 for scenario,s in t.groupby('scenario'):
  s=s.reset_index(drop=True);groups=s.MAIN_BAS.to_numpy();ng=len(set(groups));splits=list(GroupKFold(min(5,ng)).split(s,groups=groups))
  for target in TARGETS:
   y=s['late_'+target].to_numpy();persist=s['early_'+target].to_numpy();models=['mean','persistence','early_ridge','additive_spline','rf','boosting','attributes_only_rf']
   for name in models:
    oof=np.zeros(len(s))
    for fold,(tr,te) in enumerate(splits):
     assert not set(groups[tr])&set(groups[te]);features=cols if name=='attributes_only_rf' else early if name=='early_ridge' else early+cols
     if name=='mean':p=np.repeat(y[tr].mean(),len(te))
     elif name=='persistence':p=persist[te]
     else:
      if name=='early_ridge':est=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge());grid={'ridge__alpha':[1.,100.]}
      elif name=='additive_spline':est=make_pipeline(SimpleImputer(strategy='median'),SplineTransformer(n_knots=3,degree=3,extrapolation='linear'),StandardScaler(),Ridge());grid={'ridge__alpha':[10.,100.]}
      elif name in ['rf','attributes_only_rf']:est=make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=1));grid={'randomforestregressor__min_samples_leaf':[5,15]}
      else:est=make_pipeline(SimpleImputer(strategy='median'),GradientBoostingRegressor(n_estimators=100,max_depth=2,learning_rate=.03,random_state=42,loss='huber'));grid={'gradientboostingregressor__min_samples_leaf':[5,15]}
      inner=GroupKFold(min(3,len(set(groups[tr]))));m=GridSearchCV(est,grid,cv=inner,scoring='neg_mean_absolute_error',n_jobs=4,error_score='raise');m.fit(s.iloc[tr][features],y[tr],groups=groups[tr]);p=m.predict(s.iloc[te][features]);params.append(dict(scenario=scenario,target=target,model=name,fold=fold,best=m.best_params_))
     oof[te]=p;foldrows.append(dict(scenario=scenario,target=target,model=name,fold=fold,n_test=len(te),r2=r2_score(y[te],p),mae=mean_absolute_error(y[te],p)))
     predrows.extend(dict(scenario=scenario,target=target,model=name,fold=fold,GRAND_ID=int(s.iloc[j].GRAND_ID),MAIN_BAS=int(groups[j]),observed=float(y[j]),predicted=float(v)) for j,v in zip(te,p))
    # Paired cluster bootstrap for prediction-error difference versus persistence.
    rng=np.random.default_rng(42);ugs=np.unique(groups);deltas=[]
    for _ in range(1000):
     ix=np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(ugs,len(ugs),replace=True)]);deltas.append(float(np.mean(abs(y[ix]-oof[ix])-abs(y[ix]-persist[ix]))))
    summaries.append(dict(scenario=scenario,target=target,model=name,n=len(s),n_main_basins=ng,largest_group=int(s.groupby('MAIN_BAS').size().max()),r2=r2_score(y,oof),mae=mean_absolute_error(y,oof),rmse=np.sqrt(mean_squared_error(y,oof)),rho=spearmanr(y,oof).statistic,mae_difference_vs_persistence=np.mean(abs(y-oof)-abs(y-persist)),difference_ci_low=np.quantile(deltas,.025),difference_ci_high=np.quantile(deltas,.975)))
   print(scenario,target,'done',flush=True)
  pd.DataFrame(summaries).to_csv(OUT/'benchmark_summary.csv',index=False);pd.DataFrame(predrows).to_csv(OUT/'oof_predictions.csv',index=False);pd.DataFrame(foldrows).to_csv(OUT/'fold_metrics.csv',index=False)
 (OUT/'selected_parameters.json').write_text(json.dumps(params,indent=2));print('COMPLETE',flush=True)
if __name__=='__main__':main()
