"""Relabel two values baked into the submitted Figure 3 composite (fig3_final.001.jpeg).
(a) Delta-AURC label +0.31 -> +0.26  (results/06: AURC gap under the paper's definition is +0.2615;
    +0.31 was the full-coverage risk gap).
(b) "~50x asymmetry" -> "~24x asymmetry" (results/43 item 2b: ratio of pooled foreground to
    background SP gap over 63 cells = 23.75).
Only the digits are replaced; glyph boxes were located by column/row segmentation of black pixels.
Usage: python fig03_relabel.py <in.jpeg> <out.jpeg>
"""
import sys, numpy as np
from PIL import Image, ImageDraw, ImageFont
from numpy.lib.stride_tricks import sliding_window_view
F='/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf'
im=Image.open(sys.argv[1]).convert('RGB')
def font_for_digit_height(h):
    for sz in range(20,200):
        f=ImageFont.truetype(F,sz); b=f.getbbox('0')
        if b[3]-b[1]>=h: return f
out=im.copy(); D=ImageDraw.Draw(out)
# (b) digits "50" at x 2351..2425, y 322..373 on white
D.rectangle((2349,318,2428,377),fill=(255,255,255))
f=font_for_digit_height(373-322+1); b=f.getbbox('24'); D.text((2351-b[0],373-b[3]),'24',font=f,fill=(0,0,0))
# (a) rotated digits "31": glyph x 1790..1835, y 381..445; diffusion fill, then draw "26" rotated
arr=np.asarray(out).astype(float); y0,y1,x0,x1=376,449,1785,1840; reg=arr[y0:y1,x0:x1]
nb=(reg.min(2)<225)&~((reg[...,2]-reg[...,0])>40)
dil=sliding_window_view(np.pad(nb,2),(5,5)).any((2,3))
M=np.zeros(arr.shape[:2],bool); M[y0:y1,x0:x1]=dil; known=~M
for _ in range(200):
    todo=np.argwhere(M&~known)
    if len(todo)==0: break
    nv={}
    for y,x in todo:
        k=known[y-1:y+2,x-1:x+2]
        if k.any(): nv[(y,x)]=arr[y-1:y+2,x-1:x+2][k].mean(0)
    for (y,x),v in nv.items(): arr[y,x]=v; known[y,x]=True
out=Image.fromarray(arr.clip(0,255).astype(np.uint8))
f2=font_for_digit_height(1835-1790+1); bb=f2.getbbox('26')
mask=Image.new('L',(bb[2]-bb[0]+4,bb[3]-bb[1]+4),0); ImageDraw.Draw(mask).text((2-bb[0],2-bb[1]),'26',font=f2,fill=255)
mr=mask.rotate(90,expand=True); rw,rh=mr.size
out.paste(Image.new('RGB',mr.size,(0,0,0)),(1835-rw+2,445-rh+2),mr)
out.save(sys.argv[2],quality=95)
