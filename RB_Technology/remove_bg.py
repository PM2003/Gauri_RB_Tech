import os
from io import BytesIO

from PIL import Image
from rembg import remove


INPUT_DIR  = '/content/inputs/test/image'
OUTPUT_DIR = '/content/inputs/test/image'


def remove_background(input_dir: str, output_dir: str) -> None:
    """
    Remove backgrounds from all person images using rembg (u2net).
    Images are saved back as white-background PNGs.
    """
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        src_path = os.path.join(input_dir, fname)
        img      = Image.open(src_path).convert('RGBA')

        # rembg removes the background and returns an RGBA image
        result = remove(img)

        # Composite onto white background
        bg    = Image.new('RGBA', result.size, (255, 255, 255, 255))
        final = Image.alpha_composite(bg, result).convert('RGB')

        out_name = fname.rsplit('.', 1)[0] + '.png'
        out_path = os.path.join(output_dir, out_name)
        final.save(out_path)
        print(f'[INFO] BG removed: {out_path}')


if __name__ == '__main__':
    remove_background(INPUT_DIR, OUTPUT_DIR)
