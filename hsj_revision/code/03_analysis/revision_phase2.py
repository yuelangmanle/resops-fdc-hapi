"""Frozen retrospective hypothesis/chronological validation. Not preregistered."""
from pathlib import Path
import json,calendar,hashlib
import numpy as np,pandas as pd,geopandas as gpd
from scipy.stats import wasserstein_distance,spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score,mean_absolute_error
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'outputs/revision_20260916/phase2';O.mkdir(exist_ok=True,parents=True)
US=ROOT/'data/raw/ResOpsUS+CARS_v10/v1.0';BR=ROOT/'data/raw/ResOpsBR+CARS_v10/v1.0'
ids=set(pd.read_csv(ROOT/'outputs/revision_20260916/temporal_validation/analysis_table.csv').query("scenario=='primary'").GRAND_ID)
features=['early_shape','early_ratio','early_low10','early_cv','early_zero_delta','early_concentration','early_trend']
def signature(d):
 i=d.inflow.to_numpy();o=d.outflow.to_numpy();mi=i.mean();mo=o.mean()
 if mi<=0 or mo<=0:return None
 monthly=d.groupby(d.date.dt.month).inflow.sum().reindex(range(1,13),fill_value=0).to_numpy();shares=monthly/monthly.sum()
 lo=d[d.wy<=(int(d.wy.min())+4)].inflow.mean();hi=d[d.wy>(int(d.wy.min())+4)].inflow.mean()
 return dict(shape=wasserstein_distance(i/mi,o/mo),ratio=mo/mi,low10=(np.quantile(o,.1)-np.quantile(i,.1))/mi,cv=i.std()/mi,zero_delta=np.mean(o==0)-np.mean(i==0),mean_in=mi,concentration=np.sum(shares**2),trend=np.log(hi/lo) if hi>0 and lo>0 else np.nan,**{'month'+str(k+1):v for k,v in enumerate(shares)})
def read(p):
 d=pd.read_csv(p);d['date']=pd.to_datetime(d.date,errors='coerce')
 if not {'inflow','outflow'}.issubset(d.columns):return None
 if d.date.dropna().duplicated().any():return None
 d=d[d.date.notna()&np.isfinite(d[['inflow','outflow']]).all(axis=1)&(d[['inflow','outflow']]>=0).all(axis=1)].copy();d['wy']=d.date.dt.year+(d.date.dt.month>=10).astype(int)
 counts=d.groupby('wy').size();good=[y for y,n in counts.items() if n/(366 if calendar.isleap(y) else 365)>=.9]
 return d[d.wy.isin(good)].sort_values('date')
def pair(d,start):
 e=d[d.wy.between(start,start+9)];l=d[d.wy.between(start+10,start+19)]
 if min(e.wy.nunique(),l.wy.nunique())<8:return None
 a=signature(e);b=signature(l)
 if a is None or b is None:return None
 return dict(early_start=start,early_end=start+9,late_start=start+10,late_end=start+19,early_nyears=e.wy.nunique(),late_nyears=l.wy.nunique(),mean_shift=abs(np.log(b['mean_in']/a['mean_in'])),season_shift=.5*sum(abs(b['month'+str(k)]-a['month'+str(k)]) for k in range(1,13)),abs_shape_change=abs(b['shape']-a['shape']),**{'early_'+k:v for k,v in a.items()},**{'late_'+k:v for k,v in b.items()})
def bootstrap_coef(s):
 X=np.c_[np.ones(len(s)),s[['mean_shift','season_shift']].to_numpy()];y=s.abs_shape_change.to_numpy();g=s.MAIN_BAS.to_numpy();ug=np.unique(g);rng=np.random.default_rng(42);coef=np.linalg.lstsq(X,y,rcond=None)[0];boot=[]
 for _ in range(2000):
  ix=np.concatenate([np.flatnonzero(g==v) for v in rng.choice(ug,len(ug),replace=True)]);boot.append(np.linalg.lstsq(X[ix],y[ix],rcond=None)[0])
 return [dict(term=k,coefficient=coef[j],ci95_low=np.quantile(boot,.025,axis=0)[j],ci95_high=np.quantile(boot,.975,axis=0)[j],simultaneous_low=np.quantile(boot,.0125,axis=0)[j],simultaneous_high=np.quantile(boot,.9875,axis=0)[j],n=len(s),groups=len(ug)) for j,k in enumerate(['intercept','H1_mean_shift','H2_season_shift'])]
def evaluate(s,p,label,model,sample):
 y=s.late_shape.to_numpy();base=s.early_shape.to_numpy();groups=s.MAIN_BAS.to_numpy();ug=np.unique(groups);rng=np.random.default_rng(42);diff=[]
 for _ in range(2000):
  ix=np.concatenate([np.flatnonzero(groups==v) for v in rng.choice(ug,len(ug),replace=True)]);diff.append(np.mean(abs(y[ix]-p[ix])-abs(y[ix]-base[ix])))
 return dict(task=label,sample=sample,model=model,n=len(y),groups=len(ug),r2=r2_score(y,p),mae=mean_absolute_error(y,p),equal_basin_mae=pd.DataFrame({'g':groups,'err':abs(y-p)}).groupby('g').err.mean().mean(),difference_ci_low=np.quantile(diff,.025),difference_ci_high=np.quantile(diff,.975))
def main():
 grand=pd.read_csv(US/'attributes/grand.csv').set_index('GRAND_ID');gdw=pd.read_csv(BR/'attributes/gdw.csv').set_index('GDW_ID');audit=[];rows=[];traces=[]
 for country,base,meta in [('US',US,grand),('BR',BR,gdw)]:
  for p in sorted((base/'time_series/csv').glob('*.csv')):
   gid=int(p.stem)
   if country=='US' and gid not in ids:continue
   d=read(p)
   if d is None or len(d)==0:continue
   cap=float(meta.loc[gid,'CAP_MCM']);ratio=d.outflow.mean()/d.inflow.mean() if d.inflow.mean()>0 else np.nan
   maxdaily=d.outflow.max();turnover=maxdaily*86400/(cap*1e6) if cap>0 else np.nan
   reasons=[]
   if ratio<.1 or ratio>10:reasons.append('meanratio_outside_0.1_10')
   if maxdaily>1e6:reasons.append('daily_outflow_above_1e6')
   if turnover>100:reasons.append('daily_outflow_volume_above_100_capacities')
   flagged=bool(reasons)
   audit.append(dict(country=country,ID=gid,name=meta.loc[gid,'RES_NAME'],ratio=ratio,max_outflow=maxdaily,max_daily_capacity_turnovers=turnover,flagged=flagged,reasons=';'.join(reasons),eligible_years=d.wy.nunique(),first_date=str(d.date.min().date()),last_date=str(d.date.max().date()),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
   if flagged:
    for dt in d.nlargest(3,'outflow').date:
     q=d[abs((d.date-dt).dt.days)<=3][['date','inflow','outflow']].copy();q['country']=country;q['ID']=gid;traces.append(q)
   for start in [1980,2000]:
    z=pair(d,start)
    if z:rows.append(dict(country=country,ID=gid,period='train' if start==1980 else 'test',flagged=flagged,LAT=meta.loc[gid,'LAT'],LON=meta.loc[gid,'LON'],**z))
 pd.DataFrame(audit).to_csv(O/'physical_screen.csv',index=False);pd.concat(traces,ignore_index=True).drop_duplicates().to_csv(O/'flagged_event_context.csv',index=False)
 t=pd.DataFrame(rows);basin=gpd.read_file(ROOT/'data/enhancement_probe/basinatlas/BasinATLAS_v10.gdb',layer='BasinATLAS_v10_lev06',columns=['MAIN_BAS'])
 pts=gpd.GeoDataFrame(t[['country','ID']].drop_duplicates(),geometry=gpd.points_from_xy(t.drop_duplicates(['country','ID']).LON,t.drop_duplicates(['country','ID']).LAT),crs='EPSG:4326')
 joined=gpd.sjoin(pts,basin,how='left',predicate='intersects');joined['ambiguous']=joined.duplicated(['country','ID'],keep=False);joined.drop(columns='geometry').to_csv(O/'basin_matches.csv',index=False)
 joined=joined[~joined.ambiguous].dropna(subset=['MAIN_BAS']);t=t.merge(joined[['country','ID','MAIN_BAS']],on=['country','ID'],validate='many_to_one');t.to_csv(O/'frozen_window_pairs.csv',index=False)
 print(t.groupby(['country','period']).size(),flush=True);print('FLAGS',pd.DataFrame(audit).query('flagged')[['country','ID','reasons']].to_string(index=False),flush=True)
 hy=[];summaries=[];predictions=[]
 for sample in ['all_candidates','exclude_screen_flags']:
  d=t if sample=='all_candidates' else t[~t.flagged]
  for (country,period),s in d.groupby(['country','period']):
   if len(s)<10 or s.MAIN_BAS.nunique()<5:continue
   hy.extend(dict(sample=sample,country=country,period=period,**r) for r in bootstrap_coef(s))
  train=d[(d.country=='US')&(d.period=='train')].copy();test=d[(d.country=='US')&(d.period=='test')].copy().reset_index(drop=True)
  assert train.late_end.max()<test.early_start.min()
  def fitpredict(train,test,model):
   if model=='persistence':return test.early_shape.to_numpy()
   if model=='mean':return np.repeat(train.late_shape.mean(),len(test))
   est=Ridge(alpha=10,solver='lsqr',tol=1e-10) if model=='ridge' else RandomForestRegressor(n_estimators=200,min_samples_leaf=8,max_features=.7,random_state=42,n_jobs=4)
   m=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),est);m.fit(train[features],train.late_shape);return m.predict(test[features])
  for model in ['mean','persistence','ridge','rf']:
   p=np.zeros(len(test))
   for fold,(_,te) in enumerate(GroupKFold(5).split(test,groups=test.MAIN_BAS)):
    held=set(test.iloc[te].MAIN_BAS);tr=train[~train.MAIN_BAS.isin(held)];assert not set(tr.MAIN_BAS)&held
    p[te]=fitpredict(tr,test.iloc[te],model)
    for j in te:predictions.append(dict(task='US_chronological_basin',sample=sample,model=model,fold=fold,ID=int(test.iloc[j].ID),MAIN_BAS=int(test.iloc[j].MAIN_BAS),observed=float(test.iloc[j].late_shape),predicted=float(p[j]),n_train=len(tr),train_max_target_year=int(tr.late_end.max()),test_min_target_year=int(test.iloc[j].late_start)))
   summaries.append(evaluate(test,p,'US_chronological_basin',model,sample))
   external=d[(d.country=='BR')&(d.period=='test')].reset_index(drop=True)
   if len(external)>=10:
    ep=fitpredict(train,external,model);summaries.append(evaluate(external,ep,'US_to_BR_retrospective',model,sample))
    predictions.extend(dict(task='US_to_BR_retrospective',sample=sample,model=model,ID=int(row.ID),MAIN_BAS=int(row.MAIN_BAS),observed=float(row.late_shape),predicted=float(v),n_train=len(train),train_max_target_year=int(train.late_end.max()),test_min_target_year=int(row.late_start)) for row,v in zip(external.itertuples(),ep))
 pd.DataFrame(hy).to_csv(O/'hypothesis_coefficients.csv',index=False);pd.DataFrame(summaries).to_csv(O/'chronological_benchmark.csv',index=False);pd.DataFrame(predictions).to_csv(O/'chronological_predictions.csv',index=False)
 print(pd.DataFrame(summaries).round(4).to_string(index=False),flush=True)
 (O/'provenance.json').write_text(json.dumps({'status':'exploratory chronological and geographic replay, not untouched confirmation','features':features,'protocol_sha256':hashlib.sha256((ROOT/'docs/revision_20260916/PHASE2_FROZEN_PROTOCOL.md').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
if __name__=='__main__':main()
