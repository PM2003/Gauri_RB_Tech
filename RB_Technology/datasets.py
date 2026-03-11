import json
from os import path as osp

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils import data
from torchvision import transforms


class VITONDataset(data.Dataset):
    """
    Dataset class for the VITON virtual try-on benchmark.
    Loads person images, clothing items, pose keypoints and parsing maps.
    """
    def __init__(self, opt):
        super(VITONDataset, self).__init__()
        self.load_height  = opt.load_height
        self.load_width   = opt.load_width
        self.semantic_nc  = opt.semantic_nc
        self.data_path    = osp.join(opt.dataset_dir, opt.dataset_mode)
        self.transform    = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        img_names, c_names = [], []
        with open(osp.join(opt.dataset_dir, opt.dataset_list)) as f:
            for line in f:
                img, cloth = line.strip().split()
                img_names.append(img)
                c_names.append(cloth)
        self.img_names = img_names
        self.c_names   = {'unpaired': c_names}

    # ------------------------------------------------------------------
    def _parse_agnostic(self, parse, pose_data):
        """Remove upper-body region from parsing map (clothing-agnostic)."""
        arr = np.array(parse)
        upper  = sum((arr == i).astype(np.float32) for i in [5, 6, 7])
        neck   = (arr == 10).astype(np.float32)
        r      = 10
        agn    = parse.copy()
        for pid, joints in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
            mask = Image.new('L', (self.load_width, self.load_height), 'black')
            draw = ImageDraw.Draw(mask)
            prev = joints[0]
            for j in joints[1:]:
                if (pose_data[prev, 0] == 0 and pose_data[prev, 1] == 0) or \
                   (pose_data[j,    0] == 0 and pose_data[j,    1] == 0):
                    continue
                draw.line([tuple(pose_data[k]) for k in [prev, j]], 'white', width=r * 10)
                px, py = pose_data[j]
                rad = r * 4 if j == joints[-1] else r * 15
                draw.ellipse((px - rad, py - rad, px + rad, py + rad), 'white')
                prev = j
            arm = (np.array(mask) / 255) * (arr == pid).astype(np.float32)
            agn.paste(0, None, Image.fromarray((arm * 255).astype(np.uint8), 'L'))
        agn.paste(0, None, Image.fromarray((upper * 255).astype(np.uint8), 'L'))
        agn.paste(0, None, Image.fromarray((neck  * 255).astype(np.uint8), 'L'))
        return agn

    def _img_agnostic(self, img, parse, pose_data):
        """Build clothing-agnostic person image (gray out torso & arms)."""
        arr  = np.array(parse)
        head  = sum((arr == i).astype(np.float32) for i in [4, 13])
        lower = sum((arr == i).astype(np.float32) for i in [9, 12, 16, 17, 18, 19])
        r     = 20
        agn   = img.copy()
        draw  = ImageDraw.Draw(agn)
        la = np.linalg.norm(pose_data[5] - pose_data[2])
        lb = np.linalg.norm(pose_data[12] - pose_data[9])
        mid = (pose_data[9] + pose_data[12]) / 2
        pose_data[9]  = mid + (pose_data[9]  - mid) / lb * la
        pose_data[12] = mid + (pose_data[12] - mid) / lb * la
        draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r * 10)
        for i in [2, 5]:
            px, py = pose_data[i]
            draw.ellipse((px - r*5, py - r*5, px + r*5, py + r*5), 'gray')
        for i in [3, 4, 6, 7]:
            if (pose_data[i-1, 0] == 0 and pose_data[i-1, 1] == 0) or \
               (pose_data[i,   0] == 0 and pose_data[i,   1] == 0):
                continue
            draw.line([tuple(pose_data[j]) for j in [i-1, i]], 'gray', width=r * 10)
            px, py = pose_data[i]
            draw.ellipse((px - r*5, py - r*5, px + r*5, py + r*5), 'gray')
        for i in [9, 12]:
            px, py = pose_data[i]
            draw.ellipse((px - r*3, py - r*6, px + r*3, py + r*6), 'gray')
        draw.line([tuple(pose_data[i]) for i in [2,  9]],  'gray', width=r*6)
        draw.line([tuple(pose_data[i]) for i in [5,  12]], 'gray', width=r*6)
        draw.line([tuple(pose_data[i]) for i in [9,  12]], 'gray', width=r*12)
        draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray')
        px, py = pose_data[1]
        draw.rectangle((px - r*7, py - r*7, px + r*7, py + r*7), 'gray')
        agn.paste(img, None, Image.fromarray((head  * 255).astype(np.uint8), 'L'))
        agn.paste(img, None, Image.fromarray((lower * 255).astype(np.uint8), 'L'))
        return agn

    # ------------------------------------------------------------------
    LABELS = {
        0:  ('background', [0, 10]),
        1:  ('hair',       [1, 2]),
        2:  ('face',       [4, 13]),
        3:  ('upper',      [5, 6, 7]),
        4:  ('bottom',     [9, 12]),
        5:  ('left_arm',   [14]),
        6:  ('right_arm',  [15]),
        7:  ('left_leg',   [16]),
        8:  ('right_leg',  [17]),
        9:  ('left_shoe',  [18]),
        10: ('right_shoe', [19]),
        11: ('socks',      [8]),
        12: ('noise',      [3, 11]),
    }

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        c_name, c, cm = {}, {}, {}
        for key in self.c_names:
            c_name[key] = self.c_names[key][idx]
            cloth = Image.open(osp.join(self.data_path, 'cloth', c_name[key])).convert('RGB')
            cloth = transforms.Resize(self.load_width, interpolation=2)(cloth)
            mask  = Image.open(osp.join(self.data_path, 'cloth-mask', c_name[key]))
            mask  = transforms.Resize(self.load_width, interpolation=0)(mask)
            c[key]  = self.transform(cloth)
            cm_arr  = (np.array(mask) >= 128).astype(np.float32)
            cm[key] = torch.from_numpy(cm_arr).unsqueeze(0)

        # Pose
        pose_rgb = self.transform(transforms.Resize(self.load_width, interpolation=2)(
            Image.open(osp.join(self.data_path, 'openpose-img', img_name.replace('.jpg', '_rendered.png')))))
        with open(osp.join(self.data_path, 'openpose-json', img_name.replace('.jpg', '_keypoints.json'))) as f:
            kp = json.load(f)['people'][0]['pose_keypoints_2d']
        pose_data = np.array(kp).reshape(-1, 3)[:, :2]

        # Parsing
        parse = transforms.Resize(self.load_width, interpolation=0)(
            Image.open(osp.join(self.data_path, 'image-parse', img_name.replace('.jpg', '.png'))))
        parse_agn = torch.from_numpy(np.array(self._parse_agnostic(parse, pose_data))[None]).long()
        pa_map = torch.zeros(20, self.load_height, self.load_width)
        pa_map.scatter_(0, parse_agn, 1.0)
        new_pa = torch.zeros(self.semantic_nc, self.load_height, self.load_width)
        for i, (_, ids) in self.LABELS.items():
            for lid in ids:
                new_pa[i] += pa_map[lid]

        # Person image
        img = transforms.Resize(self.load_width, interpolation=2)(
            Image.open(osp.join(self.data_path, 'image', img_name)))
        img_agn = self._img_agnostic(img, parse, pose_data)
        return {
            'img_name': img_name, 'c_name': c_name,
            'img': self.transform(img), 'img_agnostic': self.transform(img_agn),
            'parse_agnostic': new_pa, 'pose': pose_rgb,
            'cloth': c, 'cloth_mask': cm,
        }

    def __len__(self):
        return len(self.img_names)


class VITONDataLoader:
    def __init__(self, opt, dataset):
        sampler = data.sampler.RandomSampler(dataset) if opt.shuffle else None
        self.data_loader = data.DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=(sampler is None),
            num_workers=opt.workers, pin_memory=True, drop_last=True, sampler=sampler
        )
        self.dataset   = dataset
        self.data_iter = iter(self.data_loader)

    def next_batch(self):
        try:
            return next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            return next(self.data_iter)
