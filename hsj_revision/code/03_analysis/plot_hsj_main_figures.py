"""Create publication-oriented main figures from saved predictions and summaries."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import textwrap
from matplotlib.lines import Line2D

R=Path(__file__).resolve().parents[2]
P=R/'outputs/revision_20260916/phase2'
O=R/'manuscript/revision_20260916/figures'
O.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':11,
    'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
colors={'persistence':'#32688A','ridge':'#8A939B','rf':'#B56C43'}
labels={'persistence':'Persistence','ridge':'Ridge','rf':'Random forest'}
pred=pd.read_csv(P/'chronological_predictions.csv')
bench=pd.read_csv(P/'chronological_benchmark.csv')

def ecdf(ax, frame, title):
    for model in ['persistence','ridge','rf']:
        z=frame[frame.model==model]
        e=np.sort(np.abs(z.predicted-z.observed)); y=np.arange(1,len(e)+1)/len(e)
        ax.step(e,y,where='post',color=colors[model],lw=1.8,ls={'persistence':'-','ridge':':','rf':'--'}[model],label=labels[model])
    ax.set_xscale('symlog',linthresh=.01)
    ax.set_xlim(left=0)
    ax.set_title(title,loc='left'); ax.set_xlabel('Absolute shape error (symlog scale)'); ax.set_ylabel('Reservoir fraction')
    ax.grid(alpha=.18); ax.set_axisbelow(True)

fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
us=pred[(pred.task=='US_chronological_basin')&(pred['sample']=='all_candidates')]
br=pred[(pred.task=='US_to_BR_retrospective')&(pred['sample']=='all_candidates')]
ecdf(axs[0,0],us,'a  US test-error distributions'); axs[0,0].legend(frameon=False,fontsize=8,loc='lower right')
piv=us[us.model.isin(['persistence','rf'])].pivot(index=['ID','MAIN_BAS'],columns='model',values=['predicted','observed'])
assert np.allclose(piv['observed']['rf'],piv['observed']['persistence'])
errors=(piv['predicted']-piv['observed']).abs()
diff=(errors['rf']-errors['persistence']).reset_index(name='difference')
bas=diff.groupby('MAIN_BAS').difference.mean().sort_values()
axs[0,1].axvline(0,color='black',ls='--',lw=.8)
axs[0,1].barh(np.arange(len(bas)),bas,color=np.where(bas.values>0,'#B56C43','#32688A'),height=.72)
axs[0,1].set_yticks(np.arange(len(bas)),[str(int(x)) for x in bas.index],fontsize=9)
axs[0,1].set_xlabel('RF minus persistence mean absolute error\nPositive values favour persistence')
axs[0,1].set_title('b  Basin-level direction of the comparison',loc='left');axs[0,1].grid(axis='x',alpha=.18);axs[0,1].set_axisbelow(True)
ecdf(axs[1,0],br,'c  Brazil transfer-error distributions'); axs[1,0].legend(frameon=False,fontsize=8,loc='lower right')
for j,(task,title) in enumerate([('US_chronological_basin','US'),('US_to_BR_retrospective','Brazil')]):
    vals=bench[(bench.task==task)&(bench['sample']=='all_candidates')].set_index('model').loc[['persistence','ridge','rf']]
    x=np.arange(3)+(j-.5)*.24
    axs[1,1].scatter(x,vals.mae,s=55,marker=['o','D'][j],color=[colors[m] for m in ['persistence','ridge','rf']],zorder=3)
    for xx,v in zip(x,vals.mae): axs[1,1].annotate(f'{v:.3f}',(xx,v),xytext=(0,9 if j==0 else -14),textcoords='offset points',ha='center',fontsize=8)
axs[1,1].set_xticks(np.arange(3),[labels[m] for m in ['persistence','ridge','rf']],rotation=15)
axs[1,1].set_ylabel('Pooled MAE in shape');axs[1,1].set_title('d  Pooled point estimates',loc='left');axs[1,1].grid(axis='y',alpha=.18);axs[1,1].set_axisbelow(True)
axs[1,1].margins(y=.3)
axs[1,1].legend(handles=[Line2D([],[],color='0.3',marker=m,ls='',label=l) for m,l in [('o','US'),('D','Brazil')]],frameon=False,fontsize=8)
fig.suptitle('Temporal persistence across reservoirs and basins',fontsize=12)
for ext in ['png','svg','pdf','tiff']: fig.savefig(O/f'fig1_chronological_errors.{ext}',dpi=600,bbox_inches='tight')
plt.close(fig)

# Reuse the audited case tables; this plotting step needs no raw-data downloads.
C=R/'outputs/revision_20260917/process_audit'
cases=pd.read_csv(C/'case_selection.csv');monthly=pd.read_csv(C/'case_monthly_shares.csv')
fig,axs=plt.subplots(4,2,figsize=(9,10),sharex=True,sharey=True,layout='constrained')
for col,kind in enumerate(['stable','unstable']):
    for row,(_,case) in enumerate(cases[cases['case']==kind].iterrows()):
        ax=axs[row,col];z=monthly[monthly.ID==case.ID]
        for period,color in [('early','#32688A'),('late','#B56C43')]:
            zz=z[z.period==period].sort_values('month')
            marker='o' if period=='early' else 's'
            ax.plot(zz.month,zz.inflow_share,color=color,lw=1.6,marker=marker,ms=3)
            ax.plot(zz.month,zz.outflow_share,color=color,lw=1.4,ls='--',marker=marker,ms=3)
        name=textwrap.fill(str(case.RES_NAME),30)
        ax.set_title(f'{chr(97+row*2+col)}  {int(case.ID)} | {name}',loc='left',fontsize=10)
        ax.text(.97,.94,f'|ΔS| = {case.abs_shape_change:.4f}',transform=ax.transAxes,ha='right',va='top',fontsize=9)
        ax.set_xticks([1,4,7,10],['Jan','Apr','Jul','Oct']);ax.set_xlim(1,12)
        ax.set_ylim(0,max(monthly.inflow_share.max(),monthly.outflow_share.max())*1.22)
        ax.grid(alpha=.15);ax.set_axisbelow(True)
        if col==0:ax.set_ylabel('Monthly share')
        if row==3:ax.set_xlabel('Calendar month')
fig.suptitle('Stable shape (left) and unstable shape (right)\nMonthly shares of archived flow-rate sums',fontsize=12)
fig.legend(handles=[Line2D([],[],color=c,ls=ls,marker='o' if c=='#32688A' else 's',ms=3,label=label) for c,ls,label in [('#32688A','-','2000-2009 inflow'),('#32688A','--','2000-2009 outflow'),('#B56C43','-','2010-2019 inflow'),('#B56C43','--','2010-2019 outflow')]],loc='outside lower center',ncol=2,frameon=False,fontsize=9)
for ext in ['png','svg','pdf','tiff']:fig.savefig(O/f'fig3_case_process_monthly.{ext}',dpi=600,bbox_inches='tight')
plt.close(fig)

h=pd.read_csv(P/'hypothesis_coefficients.csv');fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,term,title in zip(axs,['H1_mean_shift','H2_season_shift'],['H1: absolute inflow mean change','H2: monthly-share change proxy']):
    for i,(country,period) in enumerate([('US','train'),('US','test'),('BR','test')]):
        row=h[(h['sample']=='all_candidates')&(h.country==country)&(h.period==period)&(h.term==term)].iloc[0]
        ax.errorbar(row.coefficient,i,xerr=[[row.coefficient-row.simultaneous_low],[row.simultaneous_high-row.coefficient]],fmt='o',color='#32688A',capsize=4)
    ax.axvline(0,color='grey',ls='--');ax.set_yticks(range(3),['US early pair','US late pair','Brazil late pair']);ax.set_title(title,loc='left');ax.set_xlabel('Coefficient (97.5% basin-bootstrap interval)');ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
fig.suptitle('Post-hoc diagnostic associations; not causal or forecast inputs',fontsize=11)
for ext in ['png','svg','pdf','tiff']: fig.savefig(O/f'fig4_hypothesis_intervals.{ext}',dpi=600,bbox_inches='tight')
plt.close(fig)
