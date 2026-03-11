import os
import torch
from torchvision.utils import save_image


def gen_noise(shape):
    """Generate Gaussian noise tensor of given shape."""
    return torch.randn(shape)


def load_checkpoint(model, checkpoint_path):
    """Load model weights from a checkpoint file."""
    if not os.path.exists(checkpoint_path):
        print(f'[WARNING] No checkpoint found at {checkpoint_path}')
        return
    state = torch.load(checkpoint_path, map_location='cpu')
    # Handle DataParallel-wrapped checkpoints (keys start with 'module.')
    from collections import OrderedDict
    new_state = OrderedDict()
    for k, v in state.items():
        new_state[k[7:] if k.startswith('module.') else k] = v
    model.load_state_dict(new_state)
    print(f'[INFO] Loaded checkpoint: {checkpoint_path}')


def save_images(output_tensor, names, save_dir):
    """Save a batch of output tensors as PNG images."""
    os.makedirs(save_dir, exist_ok=True)
    # De-normalize from [-1, 1] to [0, 1]
    output_tensor = (output_tensor + 1) / 2
    for img, name in zip(output_tensor, names):
        out_path = os.path.join(save_dir, name if name.endswith('.png') else name + '.png')
        save_image(img, out_path)
        print(f'[INFO] Saved: {out_path}')
