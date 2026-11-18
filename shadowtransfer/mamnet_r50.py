"""
Arm L — MAMNet with a ResNet-50 encoder.

Why this file exists
--------------------
zNBu-W5 says the §4.4 failure-locus claim confounds ARCHITECTURE (ViT vs CNN)
with REGIME (self-supervised pretraining + low-LR fine-tuning), because the only
SSL-pretrained, low-LR model in the paper is also the only transformer. Arm L
separates the two at fixed capacity: one CNN, four cells, {supervised, SSL} x
{1e-4, 1e-5}. The encoder is ResNet-50 because that is the capacity DINO
publishes a self-supervised CNN checkpoint for; ResNet-34 has none.

The published MAMNet is NOT edited. This is a separate model file that mirrors
`python/mamnet/models/mamnet.py` stage for stage, with the bottleneck channel
widths substituted. The published modules are imported, not copied, so MSCAF,
CCA, the decoder block and the auxiliary head are bit-identical to the ones that
produced Tables 30-32 and only their channel arguments differ.

Stage map, published R34 -> arm L R50, at 384x384 input
-------------------------------------------------------
    stage      stride   R34 ch   R50 ch    consumed by
    conv1/bn1/relu  1      64       64      layer1
    feat1 (layer1)  1      64      256      decoder stage 4 skip
    feat2 (layer2)  2     128      512      decoder stage 3 skip
    feat3 (layer3)  4     256     1024      decoder stage 2 skip
    feat4 (layer4)  8     512     2048      decoder stage 1 skip
    feat5 (downsm) 16     512     2048      MSCAF, then decoder stage 1

conv1 is 7x7 stride 1 and maxpool is dropped, exactly as
`models/encoder_4ch.py` does it, so the stride schedule is the published one and
feat5 lands at 24x24 for a 384 input in both encoders.

Decoder stage outputs are 1024 / 512 / 256 / 64. The three auxiliary branches
attach at the stages carrying 512 / 256 / 64 channels, per the arm spec — that
is decoder stages 2, 3 and 4. The published R34 model attaches at stages 1, 2
and 3 (256 / 128 / 64). The spec states the channel widths, not the stage
indices, so the widths are what is honoured here and
`AUX_ATTACH_NOTE` records the resulting difference in placement.

Checkpoint loading
------------------
Both pretrainings load through ONE code path: build a stock torchvision
`resnet50` with no weights, then `load_state_dict(..., strict=False)` from a
file. `audit_load` refuses to continue when anything outside `fc.*` is missing
or unexpected. This project has two live errata caused by silently dropped
weights (E-8, and the DINOv3 LayerScale drop); a third is not acceptable.

The 4-channel stem is built AFTER that load, from the loaded 3-channel
`conv1.weight`: RGB copied, the 4th (contrast) channel initialised to the
channel-mean of the RGB weights. That is `models/encoder_4ch.py:44` verbatim,
and it applies to both checkpoints.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.checkpoint  # noqa: F401  (not pulled in by `import torch`)

BASE = Path(__file__).resolve().parents[2]
_MAMNET = BASE / "mamnet"
if str(_MAMNET) not in sys.path:
    sys.path.insert(0, str(_MAMNET))

from models.mscaf import MSCAF                    # noqa: E402
from models.decoder import DecoderBlock           # noqa: E402
from models.auxiliary import AuxiliaryBranch      # noqa: E402

# Where the two pretrainings live. Both are fetched once on a login node and
# read from disk by every cell; a training job never reaches the network.
SUPERVISED_CKPT = BASE / "data/pretrained/resnet50-0676ba61.pth"
SSL_CKPT = BASE / "data/pretrained/dino_resnet50_pretrain.pth"

PRETRAIN_FILES = {"supervised": SUPERVISED_CKPT, "ssl": SSL_CKPT}
PRETRAIN_DESC = {
    "supervised": "torchvision ResNet50_Weights.IMAGENET1K_V1 "
                  "(resnet50-0676ba61.pth)",
    "ssl": "DINO self-supervised ResNet-50, dino_resnet50_pretrain.pth "
           "(https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/)",
}

# Keys allowed to be absent from a backbone checkpoint. The classifier head is
# not part of the encoder and neither pretraining is expected to carry one that
# we would use.
LOAD_EXEMPT_PREFIXES = ("fc.",)

R50_STAGE_CH = {"feat1": 256, "feat2": 512, "feat3": 1024,
                "feat4": 2048, "feat5": 2048}
DEC_OUT_CH = (1024, 512, 256, 64)

AUX_ATTACH_NOTE = (
    "aux branches attach at the decoder stages carrying 512 / 256 / 64 "
    "channels, i.e. stages 2, 3 and 4. The published ResNet-34 MAMNet attaches "
    "at stages 1, 2 and 3 (256 / 128 / 64). The arm L spec fixes the channel "
    "widths rather than the stage indices; honouring the widths moves the "
    "attachment one stage later, so aux3 shares its input tensor with the main "
    "classifier. Recorded rather than silently resolved.")


def audit_load(report, where):
    """Fail loudly on any dropped or surplus weight outside `fc.*`.

    Returns the verbatim (missing, unexpected) lists for args.json. The caller
    writes them whether or not they are empty, so the artifact records what was
    checked and not merely that a check ran.
    """
    missing = [k for k in report.missing_keys
               if not k.startswith(LOAD_EXEMPT_PREFIXES)]
    unexpected = [k for k in report.unexpected_keys
                  if not k.startswith(LOAD_EXEMPT_PREFIXES)]
    print(f"[mamnet_r50] {where}: missing_keys={list(report.missing_keys)}",
          flush=True)
    print(f"[mamnet_r50] {where}: unexpected_keys={list(report.unexpected_keys)}",
          flush=True)
    if missing or unexpected:
        raise SystemExit(
            f"FATAL {where}: state_dict load dropped weights.\n"
            f"  missing (excluding fc.*)    : {missing}\n"
            f"  unexpected (excluding fc.*) : {unexpected}\n"
            f"This is the failure mode behind errata E-8 and the DINOv3 "
            f"LayerScale drop. Not training on a partially loaded encoder.")
    return list(report.missing_keys), list(report.unexpected_keys)


def _load_backbone(pretraining, ckpt_path=None):
    """A torchvision resnet50 carrying `pretraining`, plus its load report.

    `weights=None` builds the architecture only; every weight then comes from
    the file, so the supervised and SSL cells differ in exactly one input.
    """
    from torchvision.models import resnet50

    net = resnet50(weights=None)
    if pretraining == "none":
        return net, {"pretraining": "none", "ckpt_path": None,
                     "missing_keys": [], "unexpected_keys": []}

    p = Path(ckpt_path) if ckpt_path else PRETRAIN_FILES[pretraining]
    if not p.exists():
        raise SystemExit(f"FATAL pretrained checkpoint missing: {p}")
    sd = torch.load(p, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k[len("module."):] if k.startswith("module.") else k: v
          for k, v in sd.items()}

    report = net.load_state_dict(sd, strict=False)
    missing, unexpected = audit_load(report, f"{pretraining} <- {p.name}")
    return net, {
        "pretraining": pretraining,
        "ckpt_path": str(p),
        "ckpt_name": p.name,
        "ckpt_desc": PRETRAIN_DESC[pretraining],
        "n_keys_in_file": len(sd),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


class _BNStatsGuard:
    """Restores BatchNorm running statistics on exit.

    Used as `checkpoint()`'s RECOMPUTE context. The recomputed forward runs in
    train mode on batch statistics -- identical math to the original forward,
    so the gradients are exact -- and this guard undoes the SECOND update of
    the running buffers, which would otherwise apply BN's momentum twice per
    optimiser step and silently change what the model does at eval time.

    Without it, gradient checkpointing over a block containing BatchNorm is not
    a memory optimisation, it is a different model.
    """

    def __init__(self, module):
        self.module = module
        self.saved = []

    def __enter__(self):
        self.saved = [(m, m.running_mean.clone(), m.running_var.clone(),
                       m.num_batches_tracked.clone())
                      for m in self.module.modules()
                      if isinstance(m, nn.modules.batchnorm._BatchNorm)
                      and m.running_mean is not None]
        return self

    def __exit__(self, *_exc):
        with torch.no_grad():
            for m, mean, var, nbt in self.saved:
                m.running_mean.copy_(mean)
                m.running_var.copy_(var)
                m.num_batches_tracked.copy_(nbt)
        self.saved = []
        return False


def _ckpt_stage(stage, x, enabled):
    """Run one encoder stage, gradient-checkpointed when `enabled`.

    The recompute context restores BatchNorm running statistics, so the only
    difference from an uncheckpointed run is peak memory and compute time.
    """
    if not (enabled and torch.is_grad_enabled() and x.requires_grad):
        return stage(x)
    import contextlib
    return torch.utils.checkpoint.checkpoint(
        stage, x, use_reentrant=False,
        context_fn=lambda: (contextlib.nullcontext(), _BNStatsGuard(stage)))


class ResNet50Encoder4Ch(nn.Module):
    """ResNet-50 encoder, 4-channel RGBC stem, published stride schedule.

    `checkpoint_stages` gradient-checkpoints layer1..layer4. layer1 runs at the
    full 384x384 with 256 channels -- ResNet-50's bottleneck width, four times
    ResNet-34's -- and its retained activations are what keeps this model off a
    40 GiB card even after the attention blocks are checkpointed. See
    `_BNStatsGuard` for why this is exact rather than approximately exact.
    """

    def __init__(self, pretraining="supervised", ckpt_path=None,
                 checkpoint_stages=True):
        super().__init__()
        self.checkpoint_stages = checkpoint_stages
        resnet, self.load_report = _load_backbone(pretraining, ckpt_path)

        # 4-channel stem at stride 1, built from the LOADED conv1 weights.
        # models/encoder_4ch.py:30-44, unchanged except for the input width.
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
        w = resnet.conv1.weight.data
        with torch.no_grad():
            self.conv1.weight[:, :3, :, :] = w
            self.conv1.weight[:, 3:4, :, :] = w.mean(dim=1, keepdim=True)
        self.load_report["conv1_expansion"] = (
            "4-ch stem: RGB copied from the loaded conv1, channel 4 set to the "
            "channel-mean of the RGB weights (models/encoder_4ch.py:44)")

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = nn.Identity()          # dropped, as in the published R34

        self.layer1 = resnet.layer1           # 256 ch, stride 1
        self.layer2 = resnet.layer2           # 512 ch, stride 2
        self.layer3 = resnet.layer3           # 1024 ch, stride 4
        self.layer4 = resnet.layer4           # 2048 ch, stride 8

        self.downsample = nn.Sequential(
            nn.Conv2d(2048, 2048, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        ck = self.checkpoint_stages
        feat1 = _ckpt_stage(self.layer1, x, ck)
        feat2 = _ckpt_stage(self.layer2, feat1, ck)
        feat3 = _ckpt_stage(self.layer3, feat2, ck)
        feat4 = _ckpt_stage(self.layer4, feat3, ck)
        feat5 = self.downsample(feat4)
        return {"feat1": feat1, "feat2": feat2, "feat3": feat3,
                "feat4": feat4, "feat5": feat5}


class DecoderR50(nn.Module):
    """Published DecoderBlock stack at ResNet-50 widths: 2048 -> 1024 -> 512
    -> 256 -> 64. Blocks are the published class; only the widths change.

    Gradient checkpointing over the Criss-Cross Attention
    -----------------------------------------------------
    At ResNet-50 widths this model does not fit an A100-40GB at batch 8 and
    384 px: the stage-1 smoke (job 20643780) died at 39.45 GiB of 39.49 GiB
    inside `decoder3.cca`. CCA's cost is set by the SPATIAL size, not the
    channel count -- `energy_H` is [B*H, W, W], which is 1.81 GB in fp32 for
    the finest decoder stage alone, and the softmax over it saves another.

    `checkpoint_cca` recomputes those tensors in the backward pass instead of
    storing them. It is applied ONLY to CrissCrossAttention, which contains
    convolutions, a softmax and a learnable scalar -- no BatchNorm and no
    dropout. Recomputation is therefore exact and carries no side effect. The
    same trick over a block containing BatchNorm would update that layer's
    running statistics twice per step, which is a silent change to the model
    and is not done here.

    The block bodies below are `models/decoder.py:50-59` verbatim, unrolled so
    the checkpoint boundary can sit around `cca` alone. Nothing published is
    edited and every state_dict key is unchanged, so a checkpoint written with
    checkpointing on loads into a model with it off.
    """

    def __init__(self, num_classes=2, checkpoint_cca=True):
        super().__init__()
        self.checkpoint_cca = checkpoint_cca
        self.decoder1 = DecoderBlock(2048, 2048, 1024)   # skip feat4, in mscaf
        self.decoder2 = DecoderBlock(1024, 1024, 512)    # skip feat3
        self.decoder3 = DecoderBlock(512, 512, 256)      # skip feat2
        self.decoder4 = DecoderBlock(256, 256, 64)       # skip feat1
        self.classifier = nn.Conv2d(64, num_classes, 1)

    def _stage(self, block, enc_feat, dec_feat):
        _, _, H, W = enc_feat.size()
        dec_up = nn.functional.interpolate(dec_feat, size=(H, W),
                                           mode="bilinear", align_corners=False)
        if self.checkpoint_cca and self.training and enc_feat.requires_grad:
            enc_cca = torch.utils.checkpoint.checkpoint(
                block.cca, enc_feat, use_reentrant=False)
        else:
            enc_cca = block.cca(enc_feat)
        return block.fusion(torch.cat([enc_cca, dec_up], dim=1))

    def forward(self, mscaf_out, enc):
        dec1 = self._stage(self.decoder1, enc["feat4"], mscaf_out)
        dec2 = self._stage(self.decoder2, enc["feat3"], dec1)
        dec3 = self._stage(self.decoder3, enc["feat2"], dec2)
        dec4 = self._stage(self.decoder4, enc["feat1"], dec3)
        return {"main": self.classifier(dec4),
                "dec_feat1": dec1, "dec_feat2": dec2,
                "dec_feat3": dec3, "dec_feat4": dec4}


class AuxiliaryModuleR50(nn.Module):
    """Three published AuxiliaryBranch heads at 512 / 256 / 64 channels."""

    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.aux1 = AuxiliaryBranch(512, num_classes, dropout_rate)
        self.aux2 = AuxiliaryBranch(256, num_classes, dropout_rate)
        self.aux3 = AuxiliaryBranch(64, num_classes, dropout_rate)

    def forward(self, d2, d3, d4, target_size):
        return {"aux1": self.aux1(d2, target_size),
                "aux2": self.aux2(d3, target_size),
                "aux3": self.aux3(d4, target_size)}


class MAMNetR50(nn.Module):
    """MAMNet with a ResNet-50 encoder. Submodule names match the published
    MAMNet (`encoder`, `mscaf`, `decoder`, `aux_module`) so
    `armc_probes_common`'s encoder/decoder split, feat5 tap and freeze helpers
    apply unchanged under arch='mamnet'."""

    def __init__(self, num_classes=2, pretraining="supervised",
                 ckpt_path=None, use_aux=True, checkpoint_cca=True):
        super().__init__()
        self.num_classes = num_classes
        self.use_aux = use_aux
        self.checkpoint_cca = checkpoint_cca
        self.encoder = ResNet50Encoder4Ch(pretraining, ckpt_path,
                                          checkpoint_stages=checkpoint_cca)
        self.mscaf = MSCAF(in_channels=R50_STAGE_CH["feat5"])
        self.decoder = DecoderR50(num_classes=num_classes,
                                  checkpoint_cca=checkpoint_cca)
        if use_aux:
            self.aux_module = AuxiliaryModuleR50(num_classes=num_classes,
                                                 dropout_rate=0.3)

    @property
    def load_report(self):
        return self.encoder.load_report

    def forward(self, x):
        _, _, H, W = x.size()
        enc = self.encoder(x)
        mscaf_out = self.mscaf(enc["feat5"])
        dec = self.decoder(mscaf_out, enc)
        if self.use_aux and self.training:
            aux = self.aux_module(dec["dec_feat2"], dec["dec_feat3"],
                                  dec["dec_feat4"], target_size=(H, W))
            return {"main": dec["main"], **aux}
        return dec["main"]


def feat5_cosine_check(images, pretraining, ckpt_path=None, device="cpu"):
    """Mean cosine similarity of feat5 between a fresh-init and a loaded
    encoder, on the SAME fixed images.

    The point is not the value but that it is far from 1.0: if a checkpoint
    silently failed to take, the loaded encoder is the fresh one and the
    similarity is exactly 1.0. `audit_load` catches a dropped KEY; this catches
    a load that reported clean but changed nothing.
    """
    torch.manual_seed(0)
    fresh = ResNet50Encoder4Ch("none").to(device).eval()
    loaded = ResNet50Encoder4Ch(pretraining, ckpt_path).to(device).eval()
    with torch.no_grad():
        a = fresh(images.to(device))["feat5"].flatten(1)
        b = loaded(images.to(device))["feat5"].flatten(1)
        cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
    del fresh, loaded
    return float(cos.mean()), float(cos.min()), float(cos.max())


def param_counts(model):
    enc = sum(p.numel() for n, p in model.named_parameters()
              if n.startswith("encoder."))
    tot = sum(p.numel() for p in model.parameters())
    return {"total": int(tot), "encoder": int(enc), "non_encoder": int(tot - enc)}
