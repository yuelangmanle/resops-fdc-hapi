"""Independent transcription of inspected July-2025 source operations for 938/939.
No downloaded notebook or module is executed. Timezone is an explicit case assumption.
"""
from pathlib import Path
import io,json,zipfile,calendar,hashlib
import numpy as np,pandas as pd
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[2];O=R/'outputs/revision_20260917/processing_replay';O.mkdir(exist_ok=True,parents=True)
TZ='America/Chicago'
def aligned(d):
 x=d.copy();x.index=x.index.tz_localize(TZ).tz_convert('UTC')
 grid=pd.date_range(x.index.min().floor('D'),x.index.max().floor('D'),freq='D',tz='UTC')
 x=x.reindex(grid.union(x.index)).interpolate('time',limit=1,limit_direction='both').loc[grid]
 x.index=x.index.tz_localize(None);return x

def stats(d):
 z=d[np.isfinite(d[['inflow','outflow']]).all(axis=1)&(d[['inflow','outflow']]>=0).all(axis=1)]
 if len(z)==0 or min(z.inflow.mean(),z.outflow.mean())<=0:return None
 i=z.inflow.to_numpy();o=z.outflow.to_numpy()
 return dict(valid_days=len(z),mean_in=float(i.mean()),mean_out=float(o.mean()),shape=float(wasserstein_distance(i/i.mean(),o/o.mean())),inflow_volume_available_mcm=float(i.sum()*86400/1e6))
summary=[];annual=[];comparisons=[];provenance={};stages_939=None
with zipfile.ZipFile(R/'data/raw/phase2_resops_original/ResOpsUS 2.zip') as z:
 for gid in [938,939]:
  member=f'ResOpsUS/time_series_all/ResOpsUS_{gid}.csv';payload=z.read(member);raw=pd.read_csv(io.BytesIO(payload),parse_dates=['date']).set_index('date');assert not raw.index.duplicated().any()
  raw=raw.loc['1975-01-01':];raw=raw.reindex(pd.date_range(raw.index.min(),raw.index.max(),freq='D'));raw.index.name='date';raw=raw[['inflow','outflow']]
  nonnegative=raw.mask(raw<0)
  filtered=nonnegative.copy();filtered.loc[filtered.inflow>=1e5,'inflow']=np.nan
  # Historical notebook passes 'outlfow', not 'outflow'; **kwargs absorbs it.
  # Thus outflow defaults to None and the gradient criterion operates alone.
  gradient=filtered.inflow.diff().abs()>1e4;filtered.loc[gradient,'inflow']=np.nan
  linear=filtered.copy();linear.inflow=linear.inflow.interpolate('linear',limit=7,order=2)
  strict=filtered.copy();missing=strict.inflow.isna();run_id=missing.ne(missing.shift()).cumsum();length=missing.groupby(run_id).transform('sum');interp=strict.inflow.interpolate('linear',limit_area='inside');strict.inflow=interp.where(~missing|(length<=7))
  cars=pd.read_csv(R/f'data/raw/ResOpsUS+CARS_v10/v1.0/time_series/csv/{gid}.csv',parse_dates=['date']).set_index('date')[['inflow','outflow']]
  stages={'raw':raw,'negative_mask':nonnegative,'filtered':filtered,'linear_limit7':linear,'negative_mask_utc':aligned(nonnegative),'full_replay':aligned(linear),'strict_gap7_utc':aligned(strict),'CARS':cars}
  provenance[str(gid)]={'upstream_member_sha256':hashlib.sha256(payload).hexdigest(),'timezone_assumption':TZ,'gradient_filtered_days':int(gradient.sum()),'case_scope':'1975 onward; common assessment dates through 2019'}
  allstage=pd.concat({k:d.add_prefix(k+'_') for k,d in stages.items()},axis=1);allstage.columns=allstage.columns.get_level_values(1);allstage.to_csv(O/f'{gid}_daily_stages.csv',index_label='date')
  for label,d in stages.items():
   for y in range(1980,2020):
    a=d.loc[f'{y-1}-10-01':f'{y}-09-30'];s=stats(a)
    if s:annual.append(dict(ID=gid,stage=label,wy=y,eligible=s['valid_days']/(366 if calendar.isleap(y) else 365)>=.9,**s))
   j=d.join(cars,lsuffix='_replay',rsuffix='_cars').loc['1975-01-01':'2019-12-31']
   for variable in ['inflow','outflow']:
    a=j[variable+'_replay'];b=j[variable+'_cars'];both=np.isfinite(a)&np.isfinite(b);err=abs(a[both]-b[both]);comparisons.append(dict(ID=gid,stage=label,variable=variable,n_common=int(both.sum()),mae=float(err.mean()),max_abs_error=float(err.max()),values_within_1e_8=int((err<1e-8).sum()),finite_mask_disagreements=int((np.isfinite(a)!=np.isfinite(b)).sum())))
  if gid==939:stages_939=stages
pd.DataFrame(annual).to_csv(O/'annual_stage_metrics.csv',index=False);pd.DataFrame(comparisons).to_csv(O/'replay_agreement.csv',index=False)
a=stages_939['raw'].loc['2007-08-20':'2007-09-15'].copy()
for label,d in stages_939.items():a[label+'_inflow']=d.inflow.reindex(a.index)
a.to_csv(O/'939_event_stage_trace.csv',index_label='date')
# Show both partial long-gap filling and the complete observed peak.
fig,axs=plt.subplots(2,1,figsize=(11,7),layout='constrained')
for label,col,style in [('raw','#777777','.-'),('linear_limit7','#bf8f32','.-'),('full_replay','#245c86','.-'),('CARS','#ba4b35','x')]:axs[0].plot(a.index,a[label+'_inflow'],style,label=label,color=col)
axs[0].set_ylabel('Inflow (m³/s)');axs[0].set_title('939: stepwise reconstruction of the August–September 2007 episode');axs[0].legend(ncol=4);axs[0].grid(alpha=.2)
for label,col in [('full_replay','#245c86'),('strict_gap7_utc','#bf8f32')]:axs[1].plot(a.index,a[label+'_inflow'],'.-',label=label,color=col)
axs[1].set_ylabel('Inflow (m³/s)');axs[1].set_title('Diagnostic alternative: fill only complete interior gaps of at most 7 days');axs[1].legend();axs[1].grid(alpha=.2)
fig.savefig(O/'processing_steps.png',dpi=170);fig.savefig(O/'processing_steps.svg');plt.close(fig)
(O/'provenance.json').write_text(json.dumps(provenance,indent=2));print(pd.DataFrame(comparisons).query("stage=='full_replay'").to_string(index=False));print(pd.DataFrame(annual).query('ID==939 and wy==2007').to_string(index=False));print(a.loc['2007-09-04':'2007-09-06'].to_string())
