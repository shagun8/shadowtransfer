"""
Feature extraction for linear probe diagnostic (1d).
Extracts penultimate encoder features at boundary-band pixels and saves them.

Model variants supported:
  base / vanilla / fda / isw  — base architecture
  segdesic                    — SegDesic geographic adaptation
  iim                         — Illumination-Invariant Module
  mrfp_plus                   — Multi-Resolution Feature Perturbation+
  fada                        — Frequency-Adapted Domain Adaptation

Usage:
    python extract_features.py \\
        --model_type mamnet \\
        --model_variant vanilla \\
        --checkpoint_path /path/to/checkpoint_best.pth \\
        --data_root /path/to/Final_data_test/chicago/highres \\
        --city chicago \\
        --res highres \\
        --checkpoint_id upper_chicago_highres \\
        --output_dir /path/to/extracted_features
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

IMG_SIZE       = 384
BOUNDARY_WIDTH = 5

SCRIPT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAMNET_PATH = os.path.join(SCRIPT_DIR, 'mamnet')
OGLANET_PATH = os.path.join(SCRIPT_DIR, 'oglanet')
DINOV3_PATH = os.path.join(SCRIPT_DIR, 'dinov3', 'dinov3')


# ================================================================
# MODEL LOADING
# ================================================================

def _clean_sys_path():
    for p in [MAMNET_PATH, OGLANET_PATH, DINOV3_PATH]:
        if p in sys.path:
            sys.path.remove(p)


def load_model(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    _clean_sys_path()

    if args.model_type == 'mamnet':
        sys.path.insert(0, SCRIPT_DIR)
        from mamnet.models.mamnet         import MAMNet
        from mamnet.models.mamnet_segdesic import MAMNetSegDesic
        from mamnet.models.mamnet_iim      import MAMNetIIM
        from mamnet.models.mamnet_mrfp     import MAMNetMRFP
        from mamnet.models.mamnet_fada     import MAMNetFADA

        if args.model_variant in ('base', 'vanilla', 'fda', 'isw'):
            model = MAMNet(num_classes=2, pretrained=False,
                           use_aux=True, use_contrast=True)
        elif args.model_variant == 'segdesic':
            model = MAMNetSegDesic(num_classes=2, pretrained=False, use_aux=True,
                                   segdesic_hidden_dim=256, segdesic_num_scales=10,
                                   use_contrast=True)
        elif args.model_variant == 'iim':
            model = MAMNetIIM(num_classes=2, pretrained=False, use_aux=True,
                              use_contrast=True, num_kernels=8, kernel_size=5)
        elif args.model_variant == 'mrfp_plus':
            model = MAMNetMRFP(num_classes=2, pretrained=False, use_aux=True,
                               use_contrast=True, use_mrfp_plus=True)
        elif args.model_variant == 'fada':
            model = MAMNetFADA(num_classes=2, pretrained=False, use_aux=True,
                               use_contrast=True, fada_rank=16, fada_token_length=100,
                               fada_stages=(3, 4, 5))
        else:
            raise ValueError(f"Unknown variant for mamnet: {args.model_variant}")

    elif args.model_type == 'oglanet':
        sys.path.insert(0, SCRIPT_DIR)
        from oglanet.models.oglanet         import OGLANet
        from oglanet.models.oglanet_segdesic import OGLANetSegDesic
        from oglanet.models.oglanet_iim      import OGLANetIIM
        from oglanet.models.oglanet_mrfp     import OGLANetMRFP
        from oglanet.models.oglanet_fada     import OGLANetFADA

        if args.model_variant in ('base', 'vanilla', 'fda', 'isw'):
            model = OGLANet(num_classes=2, pretrained=False,
                            img_size=IMG_SIZE, use_contrast=True)
        elif args.model_variant == 'segdesic':
            model = OGLANetSegDesic(num_classes=2, pretrained=False,
                                    img_size=IMG_SIZE, use_contrast=True,
                                    segdesic_hidden_dim=256, segdesic_num_scales=10)
        elif args.model_variant == 'iim':
            model = OGLANetIIM(num_classes=2, pretrained=False,
                               img_size=IMG_SIZE, use_contrast=True,
                               num_kernels=8, kernel_size=5)
        elif args.model_variant == 'mrfp_plus':
            model = OGLANetMRFP(num_classes=2, pretrained=False,
                                img_size=IMG_SIZE, use_contrast=True,
                                use_mrfp_plus=True)
        elif args.model_variant == 'fada':
            model = OGLANetFADA(num_classes=2, pretrained=False,
                                img_size=IMG_SIZE, use_contrast=True,
                                fada_rank=16, fada_token_length=100,
                                fada_stages=(3, 4, 5))
        else:
            raise ValueError(f"Unknown variant for oglanet: {args.model_variant}")

    elif args.model_type == 'dinov3':
        sys.path.insert(0, os.path.join(SCRIPT_DIR, 'dinov3'))
        sys.path.insert(0, os.path.join(SCRIPT_DIR, 'dinov3', 'dinov3'))
        from dinov3_model      import DINOv3ShadowDetector
        from dinov3_segdesic   import DINOv3SegDesic
        from dinov3_iim_model  import DINOv3ShadowDetectorIIM
        from dinov3_model_mrfp import DINOv3ShadowDetectorMRFP
        from dinov3_model_fada import DINOv3FADAShadowDetector

        kw = dict(num_classes=2, model_name=args.dinov3_model_name,
                  weights_path=args.dinov3_weights_path,
                  pretrained=args.dinov3_pretrained,
                  frozen_stages=args.dinov3_frozen_stages)

        if args.model_variant in ('base', 'vanilla', 'fda', 'isw'):
            model = DINOv3ShadowDetector(**kw)
        elif args.model_variant == 'segdesic':
            model = DINOv3SegDesic(**kw, segdesic_hidden_dim=256, segdesic_num_scales=10)
        elif args.model_variant == 'iim':
            model = DINOv3ShadowDetectorIIM(**kw, num_kernels=8, kernel_size=5)
        elif args.model_variant == 'mrfp_plus':
            model = DINOv3ShadowDetectorMRFP(**kw, use_mrfp_plus=True)
        elif args.model_variant == 'fada':
            model = DINOv3FADAShadowDetector(
                num_classes=2, model_name=args.dinov3_model_name,
                weights_path=args.dinov3_weights_path,
                pretrained=args.dinov3_pretrained,
                fada_rank=16, fada_token_length=100,
                fada_stages=(3, 6, 9, 11))
        else:
            raise ValueError(f"Unknown variant for dinov3: {args.model_variant}")
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    model = model.to(device)
    ckpt  = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    print(f"  Model loaded: {args.model_type}/{args.model_variant}")
    return model, device


# ================================================================
# HOOK HELPERS
# ================================================================

def find_hook_layer(model, model_type, override=None):
    modules = dict(model.named_modules())
    if override:
        if override in modules:
            return override, modules[override]
        raise RuntimeError(
            f"--hook_layer '{override}' not found. Options:\n" +
            "\n".join(f"  {n}" for n in modules if n))

    if model_type in ('mamnet', 'oglanet'):
        cands = ['encoder.resnet_encoder.layer4',
                 'encoder.layer4', 'backbone.layer4', 'resnet.layer4',
                 'encoder.resnet.layer4', 'layer4',
                 'encoder.resnet_encoder.layer3',
                 'encoder.layer3', 'backbone.layer3']
    else:
        block_names = sorted(
            [n for n in modules
             if 'block' in n.lower() and n.count('.') <= 3
             and not any(s in n for s in ('norm', 'attn', 'mlp', 'proj', 'drop', 'ls'))],
        )
        cands = list(reversed(block_names))
        cands += ['backbone.dinov3.blocks.11', 'backbone.dinov3.blocks.10',
                  'backbone.blocks.11',        'backbone.blocks.10',
                  'encoder.blocks.11',         'encoder.blocks.10']

    for n in cands:
        if n in modules:
            print(f"  Hook layer: {n}")
            return n, modules[n]

    print("  Available modules (depth<=3):")
    for n in modules:
        if n and n.count('.') <= 3:
            print(f"    {n}")
    raise RuntimeError("Cannot auto-detect encoder layer. Use --hook_layer.")


def register_hook(module):
    store = {}
    def fn(mod, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(t, torch.Tensor):
            store['feat'] = t.detach()
    handle = module.register_forward_hook(fn)
    return store, handle


def feat_to_spatial(feat, model_type):
    if feat.dim() == 4:
        return feat
    if feat.dim() == 3:
        B, N, D = feat.shape
        patch   = 16
        g       = IMG_SIZE // patch
        expected = g * g
        if N == expected + 1:
            tok = feat[:, 1:, :]
        elif N == expected:
            tok = feat
        else:
            side = int(np.sqrt(N - 1))
            if side * side == N - 1:
                tok, g = feat[:, 1:, :], side
            else:
                side = int(np.sqrt(N))
                tok, g = feat, side
        return tok.permute(0, 2, 1).reshape(B, D, g, g)
    raise ValueError(f"Unexpected feature shape {feat.shape}")


# ================================================================
# BOUNDARY BAND + METADATA
# ================================================================

def eval_mask(gt, width=BOUNDARY_WIDTH):
    if gt.sum() == 0 or gt.sum() == gt.size:
        return np.ones_like(gt, dtype=bool), gt.astype(bool)
    dist_in  = distance_transform_edt(gt)
    dist_out = distance_transform_edt(1 - gt)
    dont_care = (((dist_in  > 0) & (dist_in  <= width)) |
                 ((dist_out > 0) & (dist_out <= width)))
    valid      = ~dont_care
    shadow_int = (dist_in > width)
    return valid, shadow_int


def downsample_masks_to_feature_res(valid, shadow_int, fh, fw, min_valid_frac=0.5):
    H, W = valid.shape
    ch, cw = H / fh, W / fw
    cell_valid       = np.zeros((fh, fw), dtype=bool)
    cell_shadow      = np.zeros((fh, fw), dtype=bool)
    cell_shadow_frac = np.zeros((fh, fw), dtype=np.float32)

    for r in range(fh):
        r0, r1 = int(r * ch), int(min((r+1)*ch, H))
        for c in range(fw):
            c0, c1 = int(c * cw), int(min((c+1)*cw, W))
            pv = valid[r0:r1, c0:c1]
            n_pix   = pv.size
            n_valid = pv.sum()
            if n_valid / n_pix >= min_valid_frac:
                cell_valid[r, c] = True
                ps     = shadow_int[r0:r1, c0:c1]
                sfrac  = ps[pv].mean() if n_valid > 0 else 0
                cell_shadow_frac[r, c] = sfrac
                cell_shadow[r, c]      = sfrac >= 0.5
    return cell_valid, cell_shadow, cell_shadow_frac


def cell_metadata(valid, gt, gray, pred, fh, fw):
    H, W = gt.shape
    ch, cw = H / fh, W / fw
    intensity = np.zeros((fh, fw), dtype=np.float32)
    pred_frac = np.zeros((fh, fw), dtype=np.float32)

    for r in range(fh):
        r0, r1 = int(r*ch), int(min((r+1)*ch, H))
        for c in range(fw):
            c0, c1 = int(c*cw), int(min((c+1)*cw, W))
            v = valid[r0:r1, c0:c1]
            if v.sum() > 0:
                intensity[r, c] = gray[r0:r1, c0:c1][v].mean()
                pred_frac[r, c] = pred[r0:r1, c0:c1][v].mean()
            else:
                intensity[r, c] = gray[r0:r1, c0:c1].mean()
                pred_frac[r, c] = pred[r0:r1, c0:c1].mean()
    return intensity, pred_frac


# ================================================================
# MAIN EXTRACTION
# ================================================================

def run_extraction(model, device, loader, gt_mask_dir, feat_store,
                   model_type, out_path):
    all_feat, all_int, all_gt, all_pred, all_pos, all_img = [], [], [], [], [], []

    img_dir     = os.path.join(os.path.dirname(gt_mask_dir), 'images')
    has_img_dir = os.path.isdir(img_dir)
    image_idx   = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting"):
            images = batch['image'].to(device)
            fnames = batch['filename']
            B      = images.size(0)

            feat_store.clear()
            outputs = model(images)

            if 'feat' not in feat_store:
                image_idx += B
                continue

            spatial = feat_to_spatial(feat_store['feat'], model_type).cpu().numpy()
            _, C, fh, fw = spatial.shape

            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if isinstance(outputs, dict):
                logits = outputs.get('main', outputs.get('p6', list(outputs.values())[0]))
            else:
                logits = outputs
            preds_np = torch.argmax(logits, dim=1).cpu().numpy()

            for b in range(B):
                fname   = fnames[b]
                gt_path = os.path.join(gt_mask_dir, fname)
                if not os.path.exists(gt_path):
                    base = os.path.splitext(fname)[0]
                    for ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
                        alt = os.path.join(gt_mask_dir, base + ext)
                        if os.path.exists(alt):
                            gt_path = alt
                            break
                if not os.path.exists(gt_path):
                    image_idx += 1
                    continue

                gt_pil = Image.open(gt_path).convert('L').resize(
                    (IMG_SIZE, IMG_SIZE), Image.NEAREST)
                gt_bin = (np.array(gt_pil, dtype=np.uint8) > 127).astype(np.uint8)

                gray = None
                if has_img_dir:
                    base = os.path.splitext(fname)[0]
                    for ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
                        ip = os.path.join(img_dir, base + ext)
                        if os.path.exists(ip):
                            pil  = Image.open(ip).convert('RGB').resize(
                                (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                            gray = np.array(pil, dtype=np.float32).mean(axis=2)
                            break
                if gray is None:
                    t    = images[b].cpu()
                    gray = t[:3].mean(0).numpy() * 255.0

                pred = preds_np[b]

                valid, shadow_int = eval_mask(gt_bin)
                if valid.sum() == 0:
                    image_idx += 1
                    continue

                cell_valid, cell_shadow, _ = downsample_masks_to_feature_res(
                    valid, shadow_int, fh, fw)
                if cell_valid.sum() == 0:
                    image_idx += 1
                    continue

                inten, pf = cell_metadata(valid, gt_bin, gray, pred, fh, fw)

                rows, cols = np.where(cell_valid)
                n    = len(rows)
                raw  = spatial[b, :, rows, cols]
                vecs = raw.reshape(C, n).T.astype(np.float16)

                all_feat.append(vecs)
                all_int.append(inten[rows, cols])
                all_gt.append(cell_shadow[rows, cols].astype(np.int8))
                all_pred.append((pf[rows, cols] >= 0.5).astype(np.int8))
                all_pos.append(np.stack([rows, cols], axis=1).astype(np.int16))
                all_img.append(np.full(n, image_idx, dtype=np.int32))

                image_idx += 1

    if not all_feat:
        print("  WARNING: no features extracted!")
        return

    os.makedirs(out_path, exist_ok=True)
    save = os.path.join(out_path, "features.npz")
    np.savez_compressed(save,
                        features      = np.concatenate(all_feat),
                        intensities   = np.concatenate(all_int),
                        gt_labels     = np.concatenate(all_gt),
                        pred_labels   = np.concatenate(all_pred),
                        positions     = np.concatenate(all_pos),
                        image_indices = np.concatenate(all_img))
    total = np.concatenate(all_feat).shape[0]
    dim   = np.concatenate(all_feat).shape[1]
    print(f"\n  Saved {total} vectors ({dim}D) to {save}")
    print(f"  Feature resolution: {fh}x{fw}, images processed: {image_idx}")


# ================================================================
# ARGS
# ================================================================

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_type',    required=True,
                   choices=['mamnet', 'oglanet', 'dinov3'])
    p.add_argument('--model_variant', required=True,
                   choices=['base', 'vanilla', 'fda', 'segdesic',
                            'iim', 'isw', 'mrfp_plus', 'fada'])
    p.add_argument('--checkpoint_path', required=True)
    p.add_argument('--data_root', required=True,
                   help='e.g. /path/to/Final_data_test/chicago/highres')
    p.add_argument('--city',          required=True)
    p.add_argument('--res',           required=True, choices=['highres', 'midres'])
    p.add_argument('--checkpoint_id', required=True,
                   help='Unique label, e.g. upper_chicago_highres')
    # --- Gilbreth ---
    # p.add_argument('--output_dir',
    #                default=os.path.join(os.environ["PROJECT_ROOT"], 'data', 'extracted_features'))
    # --- NCSA Delta --- (path resolves from the PROJECT_ROOT env var)
    p.add_argument('--output_dir',
                   default=os.path.join(os.environ["PROJECT_ROOT"], 'data', 'extracted_features'))
    p.add_argument('--hook_layer',    default=None, help='Override auto-detect')
    p.add_argument('--img_size',      type=int, default=384)
    p.add_argument('--batch_size',    type=int, default=4)
    p.add_argument('--num_workers',   type=int, default=4)
    p.add_argument('--device',        default='cuda')
    p.add_argument('--dinov3_model_name',     default='dinov3_vits16')
    p.add_argument('--dinov3_weights_path',   default=None)
    p.add_argument('--dinov3_pretrained',     action='store_true', default=True)
    p.add_argument('--dinov3_frozen_stages',  type=int, default=-1)
    return p.parse_args()


def main():
    args   = get_args()
    model, device = load_model(args)

    layer_name, layer_mod  = find_hook_layer(model, args.model_type, args.hook_layer)
    feat_store, handle     = register_hook(layer_mod)

    _clean_sys_path()
    sys.path.insert(0, SCRIPT_DIR)

    if args.model_type in ('mamnet', 'oglanet'):
        from mamnet.data.dataset_enhanced import ShadowDatasetEnhanced
        ds = ShadowDatasetEnhanced(root_dir=[args.data_root], split='test',
                                   img_size=IMG_SIZE, task_id=2,
                                   augment=False, geo_metadata_path=None)
    else:
        from mamnet.data.dataset import ShadowDataset
        ds = ShadowDataset(root_dir=args.data_root, split='test',
                           img_size=IMG_SIZE, augment=False)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"  Dataset: {len(ds)} images")

    gt_mask_dir = os.path.join(args.data_root, 'test', 'masks')
    if not os.path.isdir(gt_mask_dir):
        gt_mask_dir = os.path.join(args.data_root, 'masks')
    assert os.path.isdir(gt_mask_dir), f"GT masks not found at {gt_mask_dir}"
    print(f"  GT masks: {gt_mask_dir}")

    out = os.path.join(args.output_dir, args.model_type,
                       args.checkpoint_id, f"{args.city}_{args.res}")
    print(f"  Output: {out}")

    run_extraction(model, device, loader, gt_mask_dir,
                   feat_store, args.model_type, out)
    handle.remove()
    print("Done!")


if __name__ == '__main__':
    main()