#!/usr/bin/env python3
"""
Regenerate paper figures 1-5 (file numbers match the paper) from the stored results (Figure 6 = results/part1/SSPathNet_main/gradcam.png,
produced by the training script).

    python analysis/make_figures.py --results-dir results --out figures
"""
import argparse, shutil
from pathlib import Path
_ap = argparse.ArgumentParser(); _ap.add_argument("--results-dir", default="results"); _ap.add_argument("--out", default="figures")
_a = _ap.parse_args(); RES = Path(_a.results_dir); FIG = Path(_a.out); FIG.mkdir(parents=True, exist_ok=True)
import json, math, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.size":9,"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False,"figure.dpi":100})
r1=json.load(open(RES/'part1'/'results.json')); r2=json.load(open(RES/'part2'/'results.json'))
N=8146
def wil(k,n=N,z=1.96):
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); a=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return (c-a)/d*100,(c+a)/d*100
acc=lambda v:v['test']['accuracy']*100
BLUE="#1f5fa8"; GREY="#8a8a8a"; RED="#b5362b"; LG="#d9d9d9"

# ---------- Fig 2 efficiency ----------
pts=[("SSPathNet",r1['SSPathNet_main'],BLUE),("MobileNetV2",r1['baseline_mobilenet_v2'],GREY),("ResNet-18",r1['baseline_resnet18'],GREY),
     ("EfficientNet-B0",r1['baseline_efficientnet_b0'],GREY),("DenseNet-121",r1['baseline_densenet121'],GREY),
     ("A4 (no deformable)",r2['A4_no_deformable'],"#7aa6d6")]
fig,ax=plt.subplots(1,2,figsize=(9.6,3.8))
for a,(key,xl,log) in zip(ax,[('throughput_img_s','Inference throughput, batch 128 (img/s, T4, fp16)',True),('latency_ms_bs1','Single-image latency, batch 1 (ms, T4, fp16)',True)]):
    for nm,v,c in pts:
        err=round((1-v['test']['accuracy'])*N); lo,hi=wil(N-err); y=100*(1-err/N)
        a.errorbar(v[key],y,yerr=[[y-lo],[hi-y]],fmt='o',ms=4+v['params_M']*1.1,color=c,ecolor=c,capsize=2,alpha=.9,mec='white')
        off={"ResNet-18":(6,-12),"SSPathNet":(-10,13),"A4 (no deformable)":(9,-15)}.get(nm,(6,5))
        a.annotate(nm,(v[key],y),xytext=off,textcoords='offset points',fontsize=8,color="#222")
    a.set_xscale('log'); a.set_xlabel(xl,fontsize=8.5); a.set_ylim(99.0,100.0); a.grid(alpha=.25)
ax[0].set_ylabel("Test accuracy (%) with 95% Wilson CI")
ax[0].set_title("(a) Accuracy vs. throughput",loc='left',fontsize=9); ax[1].set_title("(b) Accuracy vs. latency",loc='left',fontsize=9)
fig.text(0.5,-0.02,"Marker area scales with parameter count (2.3\u201311.2 M). Higher and further right (a) / left (b) is better.",ha='center',fontsize=8,color="#444")
plt.tight_layout(); plt.savefig(str(FIG)+'/fig3_efficiency.png',dpi=220,bbox_inches='tight'); plt.close()

# ---------- Fig 3 ablation ----------
names=[("A1_stem_rgb_only","A1  RGB-only stem",False),("A2_stem_rgb_hsv","A2  RGB+HSV stem",False),("A3_stem_rgb_dct","A3  RGB+DCT stem",True),
 ("A4_no_deformable","A4  no deformable sampling",True),("A5_no_attention","A5  no attention",True),("A6_agg_none","A6  no aggregation (last stage)",True),
 ("A7_agg_concat","A7  concatenation instead of TCFA",True),("A8_no_center_loss","A8  no center loss",True),("A9_gelu_head","A9  GELU instead of adaptive act.",True)]
a0=r2['A0_full_SSPathNet']; p0=a0['test']['accuracy']
fig,ax=plt.subplots(1,2,figsize=(10,3.9),gridspec_kw={'width_ratios':[1.25,1]})
y=np.arange(len(names))[::-1]
for yi,(k,lab,dct) in zip(y,names):
    p=r2[k]['test']['accuracy']; d=(p-p0)*100; se=math.sqrt(p*(1-p)/N+p0*(1-p0)/N)*100; 
    ax[0].errorbar(d,yi,xerr=1.96*se,fmt='o',color=BLUE if abs(d)/se>1.96 else GREY,capsize=2.5,ms=5)
    ax[0].text(0.62,yi,f"{d:+.2f}",va='center',fontsize=8,color="#333")
ax[0].axvline(0,color="#444",lw=.8); ax[0].set_yticks(y); ax[0].set_yticklabels([n[1] for n in names]); ax[0].set_xlim(-0.75,0.75)
ax[0].set_xlabel("\u0394 test accuracy vs. full SSPathNet (percentage points, unpaired 95% CI)"); ax[0].grid(axis='x',alpha=.25)
ax[0].set_title("(a) Clean accuracy: wide intervals; none survives Bonferroni correction",loc='left',fontsize=8.5)
ax[0].text(-0.73,-1.15,"Blue: 95% CI excludes 0 before multiplicity correction (9 comparisons → |z| > 2.77 needed).",fontsize=7.2,color='#444')
rows=[("A0 full",r2['A0_full_SSPathNet'],True),*[(n[1].split("  ")[0]+" "+n[1].split("  ")[1][:18],r2[n[0]],n[2]) for n in names]]
rows=[("A0  full SSPathNet",a0,True)]+[(n[1],r2[n[0]],n[2]) for n in names]
yy=np.arange(len(rows))[::-1]
for yi,(lab,v,dct) in zip(yy,rows):
    val=v['robustness']['gauss_noise']*100
    ax[1].barh(yi,val,color=RED if dct else BLUE,alpha=.85,height=.62); ax[1].text(val+1.5,yi,f"{val:.1f}",va='center',fontsize=8)
ax[1].set_yticks(yy); ax[1].set_yticklabels([r[0].split("  ")[0] for r in rows]); ax[1].set_xlim(0,105)
ax[1].set_xlabel("Accuracy under Gaussian noise, \u03c3 = 0.08 (%)"); ax[1].grid(axis='x',alpha=.25)
ax[1].set_title("(b) Noise robustness splits on the DCT stream",loc='left',fontsize=8.5)
from matplotlib.patches import Patch
ax[1].legend(handles=[Patch(color=RED,alpha=.85,label="contains DCT stream"),Patch(color=BLUE,alpha=.85,label="no DCT stream")],loc='lower right',fontsize=8,frameon=False)
plt.tight_layout(); plt.savefig(str(FIG)+'/fig4_ablation.png',dpi=220,bbox_inches='tight'); plt.close()

# ---------- Fig 4 robustness heatmap ----------
cols=["clean","dark","bright","low_contrast","gauss_noise","blur","hue_shift","occlusion"]
cl=["Clean","Dark\n(\u00d70.55)","Bright\n(\u00d71.6)","Low\ncontrast","Gaussian\nnoise","Blur","Hue shift\n(40\u00b0)","Occlusion\n(3 patches)"]
order=[("SSPathNet (main run)",r1['SSPathNet_main']),("MobileNetV2",r1['baseline_mobilenet_v2']),("ResNet-18",r1['baseline_resnet18']),("EfficientNet-B0",r1['baseline_efficientnet_b0']),("DenseNet-121",r1['baseline_densenet121']),
 ("A0 full",r2['A0_full_SSPathNet']),("A1 RGB-only",r2['A1_stem_rgb_only']),("A2 RGB+HSV",r2['A2_stem_rgb_hsv']),("A3 RGB+DCT",r2['A3_stem_rgb_dct']),("A4 no deform.",r2['A4_no_deformable']),
 ("A5 no attention",r2['A5_no_attention']),("A6 no aggregation",r2['A6_agg_none']),("A7 concat agg.",r2['A7_agg_concat']),("A8 no center loss",r2['A8_no_center_loss']),("A9 GELU head",r2['A9_gelu_head'])]
M=np.array([[v['robustness'][c]*100 for c in cols] for _,v in order])
fig,ax=plt.subplots(figsize=(8.6,5.6))
im=ax.imshow(M,cmap="Blues",vmin=0,vmax=100,aspect='auto')
for i in range(M.shape[0]):
    for j in range(M.shape[1]): ax.text(j,i,f"{M[i,j]:.1f}",ha='center',va='center',fontsize=7.8,color="white" if M[i,j]>62 else "#111")
ax.set_xticks(range(len(cl))); ax.set_xticklabels(cl,fontsize=8); ax.set_yticks(range(len(order))); ax.set_yticklabels([o[0] for o in order],fontsize=8.5)
ax.xaxis.tick_top(); [s.set_visible(False) for s in ax.spines.values()]
for yline in (4.5,5.5): ax.axhline(yline,color="white",lw=3)
ax.set_title("Test accuracy (%) under synthetic distribution shift",loc='left',fontsize=9.5,pad=34)
cb=plt.colorbar(im,fraction=.03,pad=.02); cb.set_label("accuracy (%)",fontsize=8)
plt.tight_layout(); plt.savefig(str(FIG)+'/fig5_robustness.png',dpi=220,bbox_inches='tight'); plt.close()

# ---------- Fig 5 curves ----------
fig,ax=plt.subplots(1,2,figsize=(9.6,3.5))
sty={'SSPathNet_main':("SSPathNet",BLUE,2.0),'baseline_mobilenet_v2':("MobileNetV2","#999",1.2),'baseline_resnet18':("ResNet-18","#555",1.2),'baseline_efficientnet_b0':("EfficientNet-B0","#c07a1d",1.2),'baseline_densenet121':("DenseNet-121","#3a8f5c",1.2)}
for k,(lab,c,lw) in sty.items():
    h=pd.read_csv(RES/'part1'/k/'history.csv'); ax[0].plot(h.epoch,h.val_acc*100,label=lab,color=c,lw=lw); ax[1].plot(h.epoch,h.train_loss,color=c,lw=lw,label=lab)
ax[0].set_ylim(90,100); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("validation accuracy (%)"); ax[0].grid(alpha=.25); ax[0].legend(frameon=False,fontsize=8,loc='lower right'); ax[0].set_title("(a) Validation accuracy",loc='left',fontsize=9)
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("training loss (label-smoothed CE [+ center loss])"); ax[1].grid(alpha=.25); ax[1].set_title("(b) Training loss",loc='left',fontsize=9)
plt.tight_layout(); plt.savefig(str(FIG)+'/fig2_curves.png',dpi=220,bbox_inches='tight'); plt.close()

# ---------- Fig 1 architecture ----------
fig=plt.figure(figsize=(11.5,5.4)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,115); ax.set_ylim(-1,53); ax.axis('off')
def box(x,y,w,h,t,fc="#f2f2f2",ec="#555",fs=7.6,bold=False,tc="#111"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.25,rounding_size=0.8",fc=fc,ec=ec,lw=.9)); ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,color=tc,fontweight='bold' if bold else 'normal',linespacing=1.25)
def arr(x1,y1,x2,y2,c="#444",st="-|>",lw=1.0,ls='-',rad=0): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=st,mutation_scale=9,color=c,lw=lw,ls=ls,connectionstyle=f"arc3,rad={rad}"))
box(1,26,8,10,"Input\nRGB\n160\u00d7160\u00d73",fc="white")
box(13,40,19,8,"RGB stream\n2\u00d7[Conv3\u00d73, s2]+BN+GELU",fc="#e8eef7"); box(13,29.5,19,8,"HSV stream\n[s\u00b7cos\u03b8, s\u00b7sin\u03b8, s, v]\n2\u00d7[Conv3\u00d73, s2]",fc="#e8eef7"); box(13,17.5,19,10.5,"DCT stream\n4\u00d74 block-DCT (Y,Cb,Cr)\nsign(z)\u00b7log(1+4|z|) \u2192 BN\n1\u00d71 + DW3\u00d73",fc="#f6e4e1",ec=RED)
box(36,26,12,10,"Stream-gated\nfusion\n(softmax over\nstreams)",fc="#dfe9f6")
for y0 in (44,33.5,22.7): arr(9.3,31,12.7,y0,rad=0); arr(32.3,y0,35.7,31)
ax.text(1,50,"Dual-stream chromatic\u2013frequency stem (stride 4, 64 ch)",fontsize=8.5,fontweight='bold')
sx=[53,67,81,95]; dims=[("Stage 1","64 ch, 40\u00d740","2 blocks"),("Stage 2","128 ch, 20\u00d720","3 blocks"),("Stage 3","256 ch, 10\u00d710","8 blocks"),("Stage 4","384 ch, 5\u00d75","2 blocks")]
for i,(x,(a,b,c)) in enumerate(zip(sx,dims)):
    box(x,26,10.5,10,f"{a}\n{b}\n{c}",fc="#e6ecf4",fs=7.3,bold=False)
    if i>0: arr(sx[i-1]+10.8,31,x-0.3,31); ax.text(x-2.3,37.2,"Conv3\u00d73 s2",fontsize=6,ha='center',color="#555")
arr(48.3,31,52.7,31)
ax.text(53,50,"Hierarchical Deformable Inception-Transformer (DIT) backbone",fontsize=8.5,fontweight='bold')
box(53,41,52,6.5,"DIT block = BN \u2192 [conv block: deformable-inception mixer | odd block: windowed cross-attention (5\u00d75)] \u2192 BN \u2192 MLP(\u00d73)\nlayer-scale 0.1, stochastic depth \u2264 0.1; conv and attention blocks alternate",fc="white",fs=7)
box(53,8,52,10.5,"TCFA  (Topological Cross-Layer Feature Aggregation)\na\u2081 = f\u2081 ;  a\u1d62 = f\u1d62 + G\u1d62(f\u1d62) \u2299 D\u1d62(a\u1d62\u208b\u2081),  i = 2..4\nD\u1d62: DW3\u00d73 s2 + 1\u00d71 + BN ;  G\u1d62: channel gate \u00d7 spatial gate (7\u00d77)",fc="#e3f0e6",ec="#3a8f5c",fs=7.4)
for x in sx: arr(x+5.2,25.7,x+5.2,18.9,c="#3a8f5c",ls='--')
box(31,3.2,17,11,"Head\nGAP \u2192 FC 1024 \u2192 BN\n\u2192 adaptive act.\n\u2192 FC 256 (embedding)\n\u2192 FC 38",fc="#fbf0d9",ec="#b58a1b",fs=7)
arr(52.7,11.5,48.4,11.5,c="#3a8f5c")
box(1,3.2,26.5,11,"Loss = label-smoothed CE\n+ 0.005 \u00b7 center loss on the\n256-d embedding (centres updated\nby moving average)",fc="#fbf0d9",ec="#b58a1b",fs=7.2)
arr(30.7,8.7,27.8,8.7,c="#b58a1b")
ax.text(1,0.2,"SSPathNet: 9.80 M parameters \u00b7 1.96 GFLOPs (forward, 160\u00d7160; grid sampling not counted)",fontsize=8,color="#333")
plt.savefig(str(FIG)+'/fig1_architecture.png',dpi=220,bbox_inches='tight'); plt.close()
_g = RES/'part1'/'SSPathNet_main'/'gradcam.png'
if _g.exists(): shutil.copy(_g, FIG/'fig6_gradcam.png')
print("figures written to", FIG)
