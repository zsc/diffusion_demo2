"""Scientific contact sheets of fixed, unfiltered generated sample indices."""
from pathlib import Path
from PIL import Image,ImageDraw
root=Path(__file__).resolve().parents[1]
for seed in [3333,7777]:
 rows=[('parent | g2',root/f'runs/pixel96_next_20261025/previews/cleaner_24576000_g2_seed{seed}/independent_seed_grid.png')]
 for tag in ['continue_cleaner','balanced','narrow','stratified']:
  rows.append((f'{tag} | g2',root/f'runs/pixel96_cycle2_20261025/previews/{tag}_32768000_g2_seed{seed}/independent_seed_grid.png'))
 image=Image.new('RGB',(1720,40+212*len(rows)),'white');draw=ImageDraw.Draw(image)
 draw.text((8,8),f'Seed {seed} | first 8 fixed samples | Heun50 | native96px, nearest-neighbor display | no selection',fill='black')
 for i,(name,p) in enumerate(rows):
  y=40+i*212;draw.text((8,y+80),name,fill='black')
  with Image.open(p) as grid:
   assert grid.size==(2352,2352),grid.size
   for c in range(8):image.paste(grid.crop((294*c,0,294*c+288,288)).resize((192,192),Image.Resampling.NEAREST),(160+c*194,y))
 image.save(root/f'reports/pixel96_cycle2_comparison_seed{seed}_20261025.png')
