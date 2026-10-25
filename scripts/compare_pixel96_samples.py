"""Arrange existing experiment PNGs for matched-seed visual comparisons."""
from pathlib import Path
import json
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
TAGS=['baseline','b1024_lr5e5','b1024_lr1e4','b2048_lr5e5','b2048_lr1e4']

def main():
    source=ROOT/'runs/pixel96_refine_20261025/eval_artifacts'
    im=Image.new('RGB',(1322,860),'white');draw=ImageDraw.Draw(im)
    draw.text((8,8),'Same seed3333 | g2, Heun50, guide interval [0.1,0.8] | first8, unfiltered',fill='black')
    for row,tag in enumerate(TAGS):
        label='baseline_g2' if tag=='baseline' else tag+'_8192000_g2'
        p=source/label/'independent_seed_grid.png';y=32+row*164;draw.text((4,y+50),tag,fill='black')
        if not p.exists():draw.text((180,y+60),'Pending',fill='black');continue
        with Image.open(p) as grid:
            for c in range(8):im.paste(grid.crop((c*294,0,c*294+288,288)).resize((144,144),Image.Resampling.NEAREST),(170+c*144,y))
    im.save(ROOT/'reports/pixel96_refine_guided_comparison_20261025.png')
    rows=[]
    for tag in TAGS:
        root=ROOT/('runs/pixel96_bestofn_20261025' if tag=='baseline' else 'runs/pixel96_refine_bestofn_20261025')
        directory='g2_h50' if tag=='baseline' else tag
        manifest=root/directory/'selection.json'
        if not manifest.exists():continue
        selection=json.loads(manifest.read_text())
        for n in [1,8]:rows.append((tag,n,root,directory,selection['selections'][str(n)]['candidate_ids'][:8]))
    im=Image.new('RGB',(1322,42+164*len(rows)),'white');draw=ImageDraw.Draw(im)
    draw.text((8,8),'g2 / Heun50 | matched seed120000 candidate groups | first 8 groups | original 96px samples',fill='black')
    for row,(tag,n,root,directory,ids) in enumerate(rows):
        y=32+row*164;draw.text((4,y+50),f'{tag}\nBest of {n}',fill='black')
        for col,i in enumerate(ids):
            with Image.open(root/'pools'/directory/'images'/f'{i:05d}.png') as face:
                im.paste(face.resize((144,144),Image.Resampling.NEAREST),(170+col*144,y))
            draw.text((170+col*144,y+145),str(i),fill='black')
    im.save(ROOT/'reports/pixel96_refine_bestofn_comparison_20261025.png')

if __name__=='__main__':main()
