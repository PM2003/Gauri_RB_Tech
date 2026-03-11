import argparse
import os

import torch
from torch import nn
from torch.nn import functional as F
import torchgeometry as tgm

from datasets import VITONDataset, VITONDataLoader
from network import SegGenerator, GMM, ALIASGenerator
from utils import gen_noise, load_checkpoint, save_images


def get_opt():
    p = argparse.ArgumentParser(description='RB_Technology Virtual Try-On Inference')
    p.add_argument('--name',              type=str, required=True,   help='Experiment name')
    p.add_argument('-b', '--batch_size',  type=int, default=1)
    p.add_argument('-j', '--workers',     type=int, default=1)
    p.add_argument('--load_height',       type=int, default=1024)
    p.add_argument('--load_width',        type=int, default=768)
    p.add_argument('--shuffle',           action='store_true')
    p.add_argument('--dataset_dir',       type=str, default='./datasets/')
    p.add_argument('--dataset_mode',      type=str, default='test')
    p.add_argument('--dataset_list',      type=str, default='test_pairs.txt')
    p.add_argument('--checkpoint_dir',    type=str, default='./checkpoints/')
    p.add_argument('--save_dir',          type=str, default='./results/')
    p.add_argument('--display_freq',      type=int, default=1)
    p.add_argument('--seg_checkpoint',    type=str, default='seg_final.pth')
    p.add_argument('--gmm_checkpoint',    type=str, default='gmm_final.pth')
    p.add_argument('--alias_checkpoint',  type=str, default='alias_final.pth')
    p.add_argument('--semantic_nc',       type=int, default=13)
    p.add_argument('--init_type',         type=str, default='xavier',
                   choices=['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'none'])
    p.add_argument('--init_variance',     type=float, default=0.02)
    p.add_argument('--grid_size',         type=int, default=5)
    p.add_argument('--norm_G',            type=str, default='spectralaliasinstance')
    p.add_argument('--ngf',               type=int, default=64)
    p.add_argument('--num_upsampling_layers', default='most',
                   choices=['normal', 'more', 'most'])
    return p.parse_args()


def test(opt, seg, gmm, alias):
    up    = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear')
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3)).cuda()

    loader = VITONDataLoader(opt, VITONDataset(opt))

    with torch.no_grad():
        for step, inputs in enumerate(loader.data_loader, 1):
            img_names = inputs['img_name']
            c_names   = inputs['c_name']['unpaired']

            img_agn      = inputs['img_agnostic'].cuda()
            parse_agn    = inputs['parse_agnostic'].cuda()
            pose         = inputs['pose'].cuda()
            c            = inputs['cloth']['unpaired'].cuda()
            cm           = inputs['cloth_mask']['unpaired'].cuda()

            # ── Stage 1: Segmentation ──────────────────────────────────
            pa_down = F.interpolate(parse_agn, (256, 192), mode='bilinear')
            po_down = F.interpolate(pose,      (256, 192), mode='bilinear')
            c_down  = F.interpolate(c * cm,    (256, 192), mode='bilinear')
            cm_down = F.interpolate(cm,        (256, 192), mode='bilinear')
            seg_in  = torch.cat((cm_down, c_down, pa_down, po_down,
                                  gen_noise(cm_down.size()).cuda()), dim=1)
            pred_down = seg(seg_in)
            pred      = gauss(up(pred_down)).argmax(dim=1, keepdim=True)

            parse_old = torch.zeros(pred.size(0), 13,
                                    opt.load_height, opt.load_width).cuda()
            parse_old.scatter_(1, pred, 1.0)

            LABELS = {
                0: [0], 1: [2, 4, 7, 8, 9, 10, 11],
                2: [3], 3: [1], 4: [5], 5: [6], 6: [12]
            }
            parse = torch.zeros(pred.size(0), 7,
                                opt.load_height, opt.load_width).cuda()
            for j, ids in LABELS.items():
                for lid in ids:
                    parse[:, j] += parse_old[:, lid]

            # ── Stage 2: Cloth Deformation ─────────────────────────────
            agn_gmm  = F.interpolate(img_agn,          (256, 192), mode='nearest')
            pc_gmm   = F.interpolate(parse[:, 2:3],    (256, 192), mode='nearest')
            pose_gmm = F.interpolate(pose,             (256, 192), mode='nearest')
            c_gmm    = F.interpolate(c,                (256, 192), mode='nearest')
            _, warped_grid = gmm(torch.cat((pc_gmm, pose_gmm, agn_gmm), 1), c_gmm)
            warped_c  = F.grid_sample(c,  warped_grid, padding_mode='border')
            warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')

            # ── Stage 3: Try-On Synthesis ──────────────────────────────
            mm = (parse[:, 2:3] - warped_cm).clamp(min=0)
            parse_div = torch.cat((parse, mm), dim=1)
            parse_div[:, 2:3] -= mm

            output = alias(
                torch.cat((img_agn, pose, warped_c), 1),
                parse, parse_div, mm
            )

            names = [f"{i.split('_')[0]}_{c}" for i, c in zip(img_names, c_names)]
            save_images(output, names, os.path.join(opt.save_dir, opt.name))

            if step % opt.display_freq == 0:
                print(f'Step {step} done.')


def main():
    opt = get_opt()
    print(opt)
    os.makedirs(os.path.join(opt.save_dir, opt.name), exist_ok=True)

    seg   = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc)
    gmm   = GMM(opt, inputA_nc=7, inputB_nc=3)
    opt.semantic_nc = 7
    alias = ALIASGenerator(opt, input_nc=9)
    opt.semantic_nc = 13

    load_checkpoint(seg,   os.path.join(opt.checkpoint_dir, opt.seg_checkpoint))
    load_checkpoint(gmm,   os.path.join(opt.checkpoint_dir, opt.gmm_checkpoint))
    load_checkpoint(alias, os.path.join(opt.checkpoint_dir, opt.alias_checkpoint))

    seg.cuda().eval()
    gmm.cuda().eval()
    alias.cuda().eval()
    test(opt, seg, gmm, alias)


if __name__ == '__main__':
    main()
