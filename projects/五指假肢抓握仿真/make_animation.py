"""Assemble captured MuJoCo frames; requires Pillow only for this optional export."""
from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent;DEST=ROOT/'evidence'
sequence=json.loads((DEST/'sequence.json').read_text(encoding='utf-8'))
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13)
frames=[]
for row in sequence:
    source=Image.open(DEST/'frames'/f"{row['frame']:04d}.png").resize((840,588))
    canvas=Image.new('RGB',(840,664),'#173244');canvas.paste(source,(0,76))
    draw=ImageDraw.Draw(canvas)
    draw.text((18,10),'PBL | '+row['label'],font=font,fill='white')
    draw.text((18,36),'ACTUAL MUJOCO RENDER | TH / INDEX / MIDDLE / RING / LITTLE: '+' / '.join(f'{f:.1f}' for f in row['normal'])+' N',font=small,fill='#bce0e6')
    draw.text((18,56),f"t={row['time']:.2f}s | lift={row['lift_mm']:.0f}mm | Slip={row['slip_events']} | Protection={row['force_events']} | Release refills={row['release_slip_events']}",font=small,fill='#bce0e6')
    frames.append(canvas.quantize(colors=160,dither=Image.Dither.NONE))
frames[0].save(DEST/'五指实际运行.gif',save_all=True,append_images=frames[1:],duration=125,loop=0,optimize=True)
with Image.open(DEST/'五指实际运行.gif') as animation:
    assert animation.n_frames==len(sequence)
print('Verified animation:',len(sequence),'actual frames,',(DEST/'五指实际运行.gif').stat().st_size,'bytes')
