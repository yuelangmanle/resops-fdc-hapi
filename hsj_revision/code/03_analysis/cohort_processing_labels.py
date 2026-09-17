"""Full cohort processing provenance labels, without correcting source records."""
from pathlib import Path
import io,json,zipfile,hashlib
import numpy as np,pandas as pd
from timezonefinder import TimezoneFinder
R=Path(__file__).resolve().parents[2];O=R/'outputs/revision_20260917/cohort_processing_labels';O.mkdir(parents=True,exist_ok=True);(O/'daily').mkdir(exist_ok=True)
base=R/'data/raw/ResOpsUS+CARS_v10/v1.0';screen=pd.read_csv(R/'outputs/revision_20260916/phase2/physical_screen.csv').query("country=='US'");grand=pd.read_csv(base/'attributes/grand.csv').set_index('GRAND_ID');tf=TimezoneFinder()
inv=pd.read_csv(R/'outputs/revision_20260916/phase2/upstream_evidence/time_series_inventory.csv').set_index('DAM_ID');agency=pd.read_csv(R/'outputs/revision_20260916/phase2/upstream_evidence/agency_attributes.csv').set_index('Agency_Code')
rows=[];annual=[];evidence=[]
def utc(d,tz):
 x=d.copy();x.index=x.index.tz_localize(tz,ambiguous=False,nonexistent='shift_forward').tz_convert('UTC');offset=x.index.hour.min();start=x.index.min().floor('D') if offset<12 else x.index.min().ceil('D');end=x.index.max().floor('D') if offset<12 else x.index.max().ceil('D');grid=pd.date_range(start,end,freq='D',tz='UTC');x=x.reindex(grid.union(x.index)).interpolate('time',limit=1,limit_direction='both').loc[grid];x.index=x.index.tz_localize(None);return x
with zipfile.ZipFile(R/'data/raw/phase2_resops_original/ResOpsUS 2.zip') as z:
 for n,r in enumerate(screen.itertuples(),1):
  gid=int(r.ID);raw=pd.read_csv(io.BytesIO(z.read(f'ResOpsUS/time_series_all/ResOpsUS_{gid}.csv')),parse_dates=['date']).set_index('date');ar={'ID':gid,'name':r.name,'status':'processed'}
  for variable,col in [('inflow','DATA_SOURCE.1'),('outflow','DATA_SOURCE.2')]:
   code=inv.loc[gid,col] if gid in inv.index else None;note=agency.loc[code,'Additional_Data_Notes'] if code in agency.index else None;kind=agency.loc[code,'Inflow/Outflow '] if code in agency.index else None
   evidence.append(dict(ID=gid,variable=variable,agency=code,inventory_description=kind,notes=note,time_support='agency_note_daily_average' if isinstance(note,str) and 'average over the day' in note.lower() else 'unresolved'))
  if raw.index.duplicated().any():ar['status']='duplicate_dates_excluded';rows.append(ar);continue
  raw=raw.loc['1975-01-01':,['inflow','outflow']];raw=raw.reindex(pd.date_range(raw.index.min(),raw.index.max(),freq='D'));tz=tf.timezone_at(lng=float(grand.loc[gid,'LON']),lat=float(grand.loc[gid,'LAT']));ar['timezone']=tz
  clean=raw.mask(raw<0);before=clean.copy();clean.loc[clean.inflow>=1e5,'inflow']=np.nan;grad=clean.inflow.diff().abs()>1e4;clean.loc[grad,'inflow']=np.nan
  miss=clean.inflow.isna();runs=miss.ne(miss.shift()).cumsum();length=miss.groupby(runs).transform('sum');linear=clean.copy();linear.inflow=linear.inflow.interpolate('linear',limit=7,order=2)
  cars=pd.read_csv(base/f'time_series/csv/{gid}.csv',parse_dates=['date']).set_index('date')[['inflow','outflow']];full=utc(linear,tz)
  labels=pd.DataFrame({'raw_inflow':raw.inflow,'raw_outflow':raw.outflow,'upstream_negative_inflow':raw.inflow<0,'upstream_missing_inflow':raw.inflow.isna(),'gradient_removed':grad,'missing_run_length':length.where(miss,0),'linear_fill':miss&linear.inflow.notna(),'long_gap_partial_fill':miss&linear.inflow.notna()&(length>7),'linear_inflow':linear.inflow,'linear_outflow':linear.outflow}).join(full.add_prefix('replay_')).join(cars.add_prefix('cars_'),how='outer').loc['1980-01-01':'2019-12-31']
  labels['wy']=labels.index.year+(labels.index.month>=10).astype(int)
  for v in ['inflow','outflow']:
   a=labels['replay_'+v];b=labels['cars_'+v];both=np.isfinite(a)&np.isfinite(b);err=abs(a-b);labels[v+'_replay_matches']=both&(err<=1e-8);labels[v+'_mask_disagrees']=np.isfinite(a)!=np.isfinite(b)
   ar.update({v+'_n_common':int(both.sum()),v+'_max_abs_error':float(err[both].max()),v+'_different_values':int((err[both]>1e-8).sum()),v+'_mask_disagreements':int(labels[v+'_mask_disagrees'].sum())})
  ar['exact_flow_replay']=all(ar[v+'_different_values']==0 and ar[v+'_mask_disagreements']==0 for v in ['inflow','outflow'])
  for col in ['upstream_negative_inflow','upstream_missing_inflow','linear_fill','long_gap_partial_fill','gradient_removed']:ar[col+'_days']=int(labels[col].fillna(False).sum())
  for y,a in labels.groupby('wy'):
   annual.append(dict(ID=gid,wy=int(y),days=len(a),**{col:int(a[col].fillna(False).sum()) for col in ['linear_fill','long_gap_partial_fill','upstream_negative_inflow','inflow_mask_disagrees','outflow_mask_disagrees']}))
  labels.to_csv(O/f'daily/{gid}.csv.gz',index_label='date',compression='gzip');rows.append(ar)
  if n%50==0:print(n,'/',len(screen),flush=True)
pd.DataFrame(rows).to_csv(O/'reservoir_summary.csv',index=False);pd.DataFrame(annual).to_csv(O/'annual_labels.csv',index=False);pd.DataFrame(evidence).to_csv(O/'time_support_evidence.csv',index=False)
t=pd.DataFrame(rows);print(t.groupby('status').size());print(t.exact_flow_replay.value_counts());print(t[t.exact_flow_replay==False][['ID','inflow_different_values','outflow_different_values','inflow_mask_disagreements','outflow_mask_disagreements']].to_string(index=False))
(O/'scope.json').write_text(json.dumps({'label_period':'calendar 1980-01-01 through 2019-12-31; WY1980 and WY2020 partial; not used to recompute eligibility','method':'historical pipeline transcription; timezonefinder current geographic mapping; exact reproduction tested per reservoir','caution':'labels describe modeled processing stages; CARS attribution accepted only for matching replay, otherwise candidate labels','daily_files':len(list((O/'daily').glob('*.gz')))},indent=2))
