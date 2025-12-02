from PIL import Image, ImageDraw, ImageFont
import os

letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
output_dir = 'dataset/'
os.makedirs(output_dir, exist_ok=True)

font = ImageFont.truetype("arial.ttf", 28)

for l in letters:
    folder = os.path.join(output_dir, l)
    os.makedirs(folder, exist_ok=True)
    for i in range(20):  # 20 images per letter
        img = Image.new('L', (28, 28), color=255)
        draw = ImageDraw.Draw(img)
        draw.text((4,0), l, font=font, fill=0)
        img.save(os.path.join(folder, f'{i}.png'))
