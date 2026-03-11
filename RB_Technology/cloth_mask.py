import os
from collections import OrderedDict

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

# Paths — override these or pass via CLI when running on Colab
IMAGE_DIR      = '/content/inputs/test/cloth'
RESULT_DIR     = '/content/inputs/test/cloth-mask'
CHECKPOINT    = 'cloth_segm_u2net_latest.pth'


def load_checkpoint(model, path):
    if not os.path.exists(path):
        print(f'[WARNING] Checkpoint not found: {path}')
        return
    state = torch.load(path, map_location='cpu')
    new_state = OrderedDict((k[7:] if k.startswith('module.') else k, v)
                             for k, v in state.items())
    model.load_state_dict(new_state)
    print(f'[INFO] Loaded: {path}')


class NormalizeImage:
    """Normalize image tensor to zero mean and unit std."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def __call__(self, img_tensor):
        return (img_tensor - self.mean) / self.std


def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        NormalizeImage(0.5, 0.5),
    ])


def generate_cloth_mask(model, image_dir, result_dir, device='cuda'):
    """
    Run U2NET to generate binary masks for all cloth images in image_dir
    and save them to result_dir.
    """
    os.makedirs(result_dir, exist_ok=True)
    transform = get_transform()
    model.eval()

    for fname in os.listdir(image_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(image_dir, fname)
        img = Image.open(img_path).convert('RGB').resize((768, 1024))
        inp = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            # U2NET returns multiple side outputs; use the first (d0)
            d0 = model(inp)[0]
            pred = d0[:, 0, :, :]
            # Normalize to [0, 1]
            mn, mx = pred.min(), pred.max()
            pred = (pred - mn) / (mx - mn + 1e-8)

        mask = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)
        mask = Image.fromarray(mask).convert('L')
        save_path = os.path.join(result_dir, fname.replace('.jpg', '.png'))
        mask.save(save_path)
        print(f'[INFO] Mask saved: {save_path}')


if __name__ == '__main__':
    # Import U2NET from the networks/ folder
    from networks.u2net import U2NET
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = U2NET(in_ch=3, out_ch=1).to(device)
    load_checkpoint(model, CHECKPOINT)
    generate_cloth_mask(model, IMAGE_DIR, RESULT_DIR, device)
