"""Figure 4(b), revised: per-method gap to the within-city upper bound on the 9-cell grid.
Values read from results/32_method_comparison_profile_tgtfda.json (profile, scope 9_cell,
no access restriction). Hollow marker = mean gap with 95% cell-level bootstrap CI (M2);
filled marker = worst single-cell gap (M1). Style follows the submitted fig4_right.ipynb.
Output: panel PNG and the composited Figure 4 (panel (a) cropped unchanged from the submitted JPEG)."""
import json, sys
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
import os
ROOT=os.environ['PROJECT_ROOT']
d=json.load(open(f'{ROOT}/results/32_method_comparison_profile_tgtfda.json'))
rows=[r for r in d['profile'] if r['scope']=='9_cell' and r['access_restriction'] is None]
assert len(rows)==10
LAB={'sib':'SIB (ours)','fda':'FDA','segdesic':'SegDeSiC','sar':'SAR','vanilla':'Vanilla','bsm':'BSM','fada':'FADA','iim':'IIM','mrfp_plus':'MRFP+','isw':'ISW'}
rows.sort(key=lambda r:r['M2_mean_gap'])
van=[r for r in rows if r['method']=='vanilla'][0]['M2_mean_gap']
SIB_COLOR='#C9302C'; OTHER='#3F4858'; HL='#FFF1F0'; GRID='#D9D9D9'; SPINE='#9AA0A6'; BASE='#8c8c8c'
mpl.rcParams.update({'font.family':'DejaVu Sans','axes.unicode_minus':True,'pdf.fonttype':42})
W,H=2020,1540; DPI=W/3.5
fig,ax=plt.subplots(figsize=(W/DPI,H/DPI),dpi=DPI); fig.patch.set_facecolor('white')
n=len(rows); y=np.arange(n)[::-1]
for i,r in enumerate(rows):
    sib=r['method']=='sib'; c=SIB_COLOR if sib else OTHER
    if sib: ax.axhspan(y[i]-0.42,y[i]+0.42,color=HL,zorder=0)
    m=r['M2_mean_gap']; lo,hi=r['M2_mean_gap_ci95']; w=r['M1_worst_case_gap']
    ax.plot([m,w],[y[i],y[i]],color=c,lw=2.2 if sib else 1.4,alpha=1 if sib else .78,solid_capstyle='round',zorder=2)
    ax.errorbar(m,y[i],xerr=[[m-lo],[hi-m]],fmt='none',ecolor='#333333',elinewidth=.8,capsize=1.8,zorder=3)
    ax.scatter(m,y[i],s=40 if sib else 26,facecolor='white',edgecolor=c,linewidth=1.4,zorder=4)
    ax.scatter(w,y[i],s=40 if sib else 26,color=c,zorder=4)
ax.axvline(van,color=BASE,ls=(0,(4,2.5)),lw=.9,zorder=1)
ax.set_yticks(y); ax.set_yticklabels([LAB[r['method']]+(' (n = 8)' if r['n_cells']==8 else '') for r in rows],fontsize=6.3)
for t,r in zip(ax.get_yticklabels(),rows):
    if r['method']=='sib': t.set_color(SIB_COLOR); t.set_fontweight('bold')
ax.set_xlabel('Gap to within-city upper bound (mIoU points)',fontsize=6.3)
ax.tick_params(axis='x',labelsize=6); ax.tick_params(axis='y',length=0)
ax.set_xlim(0,26); ax.set_ylim(-0.7,n-0.3)
ax.grid(axis='x',color=GRID,lw=.5); ax.set_axisbelow(True)
for s in ['top','right','left']: ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color(SPINE)
h=[Line2D([],[],marker='o',ls='none',mfc='white',mec=OTHER,ms=4,label='Mean gap, 95% CI (n = 9 cells)'),
   Line2D([],[],marker='o',ls='none',color=OTHER,ms=4,label='Worst cell'),
   Line2D([],[],color=BASE,ls=(0,(4,2.5)),lw=.9,label='Vanilla mean gap')]
ax.legend(handles=h,loc='upper right',fontsize=5.3,frameon=True,framealpha=.95,edgecolor=GRID,handletextpad=.4,borderpad=.5)
fig.tight_layout(pad=0.3)
out_panel=sys.argv[1] if len(sys.argv)>1 else f'{ROOT}/plots/20_sib_method_profile.png'
fig.savefig(out_panel,dpi=DPI,facecolor='white'); plt.close(fig)
# composite: panel (a) unchanged, divider kept, new (b) pasted right of the divider
base=Image.open(f'{ROOT}/final_paper/fig4_final.001.jpeg').convert('RGB'); assert base.size==(5900,1540)
pb=Image.open(out_panel).convert('RGB').resize((W,H))
canvas=base.copy(); canvas.paste((255,255,255),(3880,0,5900,1540)); canvas.paste(pb,(3880,0))
# re-draw the panel label "(b)" area is part of pb? add label
from PIL import ImageDraw, ImageFont
try: f=ImageFont.truetype('DejaVuSans-Bold.ttf',70)
except Exception: f=ImageFont.load_default()
ImageDraw.Draw(canvas).text((3905,15),'(b)',fill='black',font=f)
out_full=sys.argv[2] if len(sys.argv)>2 else f'{ROOT}/analysis/fig04/fig4_sib.png'
canvas.save(out_full,dpi=(300,300)); print('saved',out_panel,out_full)
for r in rows: print(r['method'],round(r['M2_mean_gap'],2),[round(x,2) for x in r['M2_mean_gap_ci95']],round(r['M1_worst_case_gap'],2),r['n_cells'])
