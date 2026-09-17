"""Render saved results; no model fitting or selection by performance."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[2]
O=R/'outputs/revision_20260917/hsj_core';O.mkdir(exist_ok=True)
F=R/'manuscript/revision_20260916/figures'
S=R/'outputs/revision_20260917/cohort_source'
G=R/'outputs/revision_20260917/gap_sensitivity'
p=pd.read_csv(S/'predictions.csv').query("task=='US' and model=='persistence'")
b=pd.read_csv(S/'benchmark.csv').query("task=='US_chronological_basin'")
g=pd.read_csv(G/'benchmark.csv').query("task=='US_chronological_basin'")
c=p.query("scenario=='cars'").copy();u=p.query("scenario=='upstream'")
assert not c.ID.duplicated().any() and set(u.ID)<=set(c.ID)
c['group']=np.where(c.ID.isin(u.ID),'Common cohort','CARS-only')
c['absolute_error']=(c.predicted-c.observed).abs()
common=c[c.group=='Common cohort'];lost=c[c.group=='CARS-only']
assert (len(c),len(common),len(lost))==(197,154,43)
assert np.isclose(c.absolute_error.mean(),(154*common.absolute_error.mean()+43*lost.absolute_error.mean())/197)
for name,t,pp in [('source',b,pd.read_csv(S/'predictions.csv').query("task=='US'")),('gap',g,pd.read_csv(G/'predictions.csv'))]:
 if name=='gap':
  # Gap predictions use scenario and task columns; verify selected US scenarios.
  if 'task' in pp: pp=pp[pp.task.isin(['US','US_chronological_basin'])]
 for row in t[t.model.isin(['persistence','rf'])].itertuples():
  label='scenario' if 'scenario' in pp else 'sample'
  z=pp[(pp[label]==row.sample)&(pp.model==row.model)]
  assert len(z)==row.n,(name,row.sample,len(z),row.n)
  assert np.isclose((z.predicted-z.observed).abs().mean(),row.mae)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False})
fig,axs=plt.subplots(2,2,figsize=(8.5,7.5),layout='constrained')
blue='#32688A';orange='#B56C43';gray='#8A939B'
rows=[]
ax=axs[0,0]
for j,(label,d,col) in enumerate([('All CARS\n(n=197)',c,gray),('Common\n(n=154)',common,blue),('CARS-only\n(n=43)',lost,orange)]):
 v=d.absolute_error.mean();ax.bar(j,v,color=col,width=.6);ax.text(j,v+.005,f'{v:.5f}',ha='center',fontsize=8)
 rows.append(dict(panel='a',scenario=label.replace('\n',' '),model='persistence',n=len(d),mae=v))
ax.set_xticks(range(3),['All CARS\n(n=197)','Common\n(n=154)','CARS-only\n(n=43)']);ax.set_ylim(0,.18);ax.set_ylabel('MAE on CARS targets');ax.set_title('a  Population composition changes error',loc='left')
ax.text(.03,.94,'Same source and persistence rule',transform=ax.transAxes,va='top',fontsize=9)
ax=axs[0,1]
scenarios=['cars_common_cohort','upstream_common_cohort']
for k,(model,col,label) in enumerate([('persistence',blue,'Persistence'),('rf',orange,'Random forest')]):
 vals=[b[(b['sample']==s)&(b.model==model)].iloc[0].mae for s in scenarios]
 ax.bar(np.arange(2)+(k-.5)*.32,vals,width=.32,color=col,label=label,hatch='' if k==0 else '///',edgecolor='white')
 for j,v in enumerate(vals):
  ax.text(j+(k-.5)*.32,v+.003,f'{v:.4f}',ha='center',fontsize=9)
  rows.append(dict(panel='b',scenario=scenarios[j],model=model,n=154,mae=v))
ax.set_xticks(range(2),['CARS','Upstream']);ax.set_ylim(0,.15);ax.set_ylabel('MAE on source-specific targets');ax.set_title('b  Source comparison (n=154)',loc='left');ax.legend(loc='upper left',fontsize=9)
ax=axs[1,0];scenarios=['raw_common_cohort','gap1_common_cohort','gap3_common_cohort','gap7_common_cohort']
for model,col,label in [('persistence',blue,'Persistence'),('rf',orange,'Random forest')]:
 vals=[g[(g['sample']==s)&(g.model==model)].iloc[0].mae for s in scenarios]
 ax.plot(range(4),vals,'o-' if model=='persistence' else 's--',color=col,label=label,ms=4)
 rows.extend(dict(panel='c',scenario=s,model=model,n=154,mae=v) for s,v in zip(scenarios,vals))
ax.set_xticks(range(4),['None','≤1 day','≤3 days','≤7 days']);ax.set_ylim(0,.13);ax.set_ylabel('MAE on scenario-specific targets');ax.set_xlabel('Complete internal inflow gaps filled');ax.set_title('c  Gap rules on the same 154 reservoirs',loc='left')
ax.text(.03,.94,'Eligible: 154 / 163 / 174 / 186\nCommon cohort: n=154',transform=ax.transAxes,va='top',fontsize=9)
ax=axs[1,1]
for j,(tab,s,label) in enumerate([(b,'cars_common_cohort','CARS'),(b,'upstream_common_cohort','Upstream'),(g,'gap1_common_cohort','≤1 day'),(g,'gap3_common_cohort','≤3 days'),(g,'gap7_common_cohort','≤7 days')]):
 rr=tab[(tab['sample']==s)&(tab.model=='rf')].iloc[0];pr=tab[(tab['sample']==s)&(tab.model=='persistence')].iloc[0];v=rr.mae-pr.mae
 ax.plot([rr.difference_ci_low,rr.difference_ci_high],[j,j],color=gray,lw=1.6);ax.plot(v,j,'o',color=orange,ms=4)
 rows.append(dict(panel='d',scenario=s,model='rf-minus-persistence',n=154,mae=v,ci_low=rr.difference_ci_low,ci_high=rr.difference_ci_high))
ax.axvline(0,color='black',ls='--',lw=.7);ax.set_yticks(range(5),['CARS','Upstream','≤1 day','≤3 days','≤7 days']);ax.invert_yaxis();ax.set_xlim(-.008,.065);ax.set_xlabel('RF minus persistence MAE\nPositive values favour persistence');ax.set_title('d  Paired differences and 95% intervals',loc='left')
for ax in axs.flat: ax.grid(axis='x' if ax is axs[1,1] else 'y',alpha=.16);ax.set_axisbelow(True)
for ext in ['png','svg','pdf','tiff']:fig.savefig(F/f'fig2_population_processing.{ext}',dpi=600)
plt.close(fig)
pd.DataFrame(rows).to_csv(O/'figure_source_data.csv',index=False);c.to_csv(O/'population_errors.csv',index=False)
(O/'verification.json').write_text(json.dumps({'population_n':[197,154,43],'population_mae':[float(d.absolute_error.mean()) for d in [c,common,lost]],'weighted_identity_passed':True,'source_and_gap_mae_recomputed':True,'uncertainty':'Saved paired main-basin bootstrap intervals, conditional on fitted predictions; panels a-c show point estimates','no_raw_data_modified':True},indent=2))
print((O/'verification.json').read_text())
