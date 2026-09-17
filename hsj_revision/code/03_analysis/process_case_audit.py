"""Process-level comparison for stable/unstable shape cases and flagged records."""
from pathlib import Path
import json,hashlib,calendar
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[2];OUT=R/'outputs/revision_20260917/process_audit';OUT.mkdir(exist_ok=True,parents=True);FIG=R/'manuscript/revision_20260916/figures';FIG.mkdir(exist_ok=True,parents=True)
P=R/'outputs/revision_20260916/phase2/frozen_window_pairs.csv';pair=pd.read_csv(P);g=pd.read_csv(R/'data/raw/ResOpsUS+CARS_v10/v1.0/attributes/grand.csv');d=pair.query("country=='US' and period=='test'").merge(g,left_on='ID',right_on='GRAND_ID',how='left');d=d[~d.flagged].copy();stable=d.nsmallest(4,'abs_shape_change');unstable=d.nlargest(4,'abs_shape_change');cases=pd.concat([stable.assign(case='stable'),unstable.assign(case='unstable')],ignore_index=True);cases.to_csv(OUT/'case_selection.csv',index=False)
RAW=R/'data/raw/ResOpsUS+CARS_v10/v1.0/time_series/csv';ann=[];monthly=[]
for row in cases.itertuples():
 x=pd.read_csv(RAW/f'{int(row.ID)}.csv');x.date=pd.to_datetime(x.date);x=x[np.isfinite(x[['inflow','outflow']]).all(axis=1)&(x[['inflow','outflow']]>=0).all(axis=1)].copy();x['wy']=x.date.dt.year+(x.date.dt.month>=10).astype(int);x['month']=x.date.dt.month;x['period']=np.where(x.wy.between(row.early_start,row.early_end),'early',np.where(x.wy.between(row.late_start,row.late_end),'late','gap'))
 counts=x.groupby('wy').size();good=[y for y,n in counts.items() if n/(366 if calendar.isleap(int(y)) else 365)>=.9];x=x[x.wy.isin(good)].copy()
 x=x[x.period!='gap'];x['ID']=row.ID;x['case']=row.case;x['RES_NAME']=row.RES_NAME;x['RIVER']=row.RIVER
 for period,z in x.groupby('period'):
  if len(z)==0:continue
  mi=z.inflow.mean();mo=z.outflow.mean();ann.append(dict(ID=row.ID,case=row.case,name=row.RES_NAME,river=row.RIVER,period=period,years=z.wy.nunique(),mean_in=mi,mean_out=mo,ratio=mo/mi if mi>0 else np.nan,zero_in=np.mean(z.inflow==0),zero_out=np.mean(z.outflow==0),q10_in=np.quantile(z.inflow,.1),q10_out=np.quantile(z.outflow,.1),q90_in=np.quantile(z.inflow,.9),q90_out=np.quantile(z.outflow,.9),in_cv=z.inflow.std()/mi if mi else np.nan,out_cv=z.outflow.std()/mo if mo else np.nan))
 for period,z in x.groupby('period'):
  sh=z.groupby('month').inflow.sum().reindex(range(1,13),fill_value=0);so=z.groupby('month').outflow.sum().reindex(range(1,13),fill_value=0);sh=sh/sh.sum();so=so/so.sum();
  for month in range(1,13):monthly.append(dict(ID=row.ID,case=row.case,name=row.RES_NAME,period=period,month=month,inflow_share=sh.loc[month],outflow_share=so.loc[month]))
pd.DataFrame(ann).to_csv(OUT/'case_annual_summary.csv',index=False);pd.DataFrame(monthly).to_csv(OUT/'case_monthly_shares.csv',index=False)
# plot monthly shares and ratio/shape transitions
m=pd.DataFrame(monthly);fig,axs=plt.subplots(2,4,figsize=(13,6),sharex=True,sharey='row',layout='constrained');
for ax,(_,r) in zip(axs.flat,cases.iterrows()):
 z=m[m.ID==r.ID];
 for p,c in [('early','#567a9e'),('late','#d06b5d')]:
  zz=z[z.period==p];ax.plot(zz.month,zz.inflow_share,marker='o',color=c,label=p+' inflow');ax.plot(zz.month,zz.outflow_share,marker='x',ls='--',color=c,alpha=.75,label=p+' outflow')
 ax.set_title(f"{int(r.ID)} {r.RES_NAME or ''}\n{r.case}, ΔS={r.abs_shape_change:.3f}");ax.set_xticks([1,4,7,10]);ax.grid(alpha=.2)
axs[0,0].legend(fontsize=7,frameon=False);fig.suptitle('Monthly shares of archived inflow-rate sums in selected stable and unstable cases',fontsize=12);fig.savefig(FIG/'fig3_case_process_monthly.png',dpi=300);fig.savefig(FIG/'fig3_case_process_monthly.svg');plt.close(fig)
# data provenance
(OUT/'provenance.json').write_text(json.dumps({'source_pair_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'case_rule':'four smallest/four largest absolute shape changes among unflagged US 2000-09→2010-19 pairs','warning':'case selection is descriptive and not independent; no process causality claimed'},indent=2))
print(cases[['ID','RES_NAME','case','abs_shape_change','mean_shift','season_shift']].to_string(index=False))
