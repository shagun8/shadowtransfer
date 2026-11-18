"""
n450 retrain -- load a published trainer and resolve the config it will run
with, using the TRAINER'S OWN argparse, then diff it against the original.

CHECK 4 (operator): print every key on which the new invocation's resolved
config differs from the original's args.json / config.json / train.log Args.
The only permitted differences are training count / per-city draw (not a
config key: done by n450_inject), validation count (not a config key either,
and unchanged at 300), output paths and run_id suffix, and pure bookkeeping.
Anything else refuses the cell.

How the resolution works. argparse.ArgumentParser.parse_args is patched for
the duration of ONE call to the trainer's own get_args()/parse_args(): the
parser parses only its required flags, every destination the original config
recorded is then set to the original value, the permitted overrides are
applied, and control returns to the trainer's function -- which then computes
its own derived fields (train_oglanet_sib.py: use_sib, in_channels,
kl_warmup_epochs, fold_names, test_city). The resolved config is therefore
what the trainer's code produced, not a reimplementation of it.
"""

import argparse
import ast
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import re
import sys

SAVE_FREQ_OFF = 10 ** 9

# key -> why changing it cannot change the trained model
PERMITTED_CHANGED = {
    "output_dir": "output path (new run writes under ${ST_SCRATCH_ROOT}/n450/{run_id})",
    "save_freq": "periodic-checkpoint frequency; best checkpoint is not gated "
                 "on it, so only extra epoch_*.pth files are suppressed "
                 "(CLAUDE.md best-only rule)",
}
SMOKE_CHANGED = {
    "epochs": "stage-1 smoke truncation (2 epochs); never in a real cell",
    "kl_warmup_epochs": "derived by train_oglanet_sib.parse_args from epochs; "
                        "smoke only",
}
DUMP_ONLY_CHANGED = {
    "epochs": "dump-only re-entry (0 epochs: load best_model.pth, test only)",
    "kl_warmup_epochs": "derived from epochs; dump-only re-entry only",
}
# Keys present in the trainer's CURRENT parser but absent from the original's
# record (arguments added after the published run). Each is listed with the
# reason its default reproduces the published behaviour. Filled from the
# 2026-10-26 dry run; an unlisted new key refuses the cell.
PERMITTED_NEW = {}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_trainer(arch_tree, modname, fake_dir):
    """Import python/{arch}/{modname}.py with __file__ pointed at fake_dir.

    The source executed is the published file, byte for byte. Only the
    module's __file__ differs, so __file__-relative WRITES land on /scratch:
    train_isw.py appends to os.path.dirname(__file__)/train_isw_debug.log at
    import time, which would otherwise write into /projects. Every other
    __file__ use in the eight trainers is `sys.path.append(dirname(__file__))`,
    harmless because the arch tree is already first on sys.path.
    """
    path = os.path.join(arch_tree, modname + ".py")
    os.makedirs(fake_dir, exist_ok=True)
    loader = importlib.machinery.SourceFileLoader(modname, path)
    spec = importlib.util.spec_from_file_location(modname, path, loader=loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = os.path.join(fake_dir, modname + ".py")
    sys.modules[modname] = mod
    loader.exec_module(mod)
    return mod, sha256_file(path)


def read_orig_config(orig_dir, source):
    """(cfg, path, sha256)."""
    if source == "args.json":
        p = os.path.join(orig_dir, "args.json")
        with open(p) as f:
            return json.load(f), p, sha256_file(p)
    p = os.path.join(orig_dir, "train.log")
    with open(p, errors="replace") as f:
        for line in f.read().splitlines()[:200]:
            m = re.search(r"Args:\s*(\{.*\})\s*$", line)
            if m:
                return ast.literal_eval(m.group(1)), p + ":Args", sha256_file(p)
    raise RuntimeError(f"no Args line in {p}")


def read_ckpt_args(ckpt_path):
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck.get("args")
    d = a if isinstance(a, dict) else vars(a)
    return {k: v for k, v in d.items()
            if isinstance(v, (str, int, float, bool, list, tuple, type(None)))}


def _norm(v):
    if isinstance(v, tuple):
        return [_norm(x) for x in v]
    if isinstance(v, list):
        return [_norm(x) for x in v]
    return v


def resolve(mod, fn_name, orig, overrides):
    """Call mod.<fn_name>() under a patched parse_args. Returns the Namespace."""
    real = argparse.ArgumentParser.parse_args
    seen = {}

    def fake(self, args=None, namespace=None):
        argv = []
        for a in self._actions:
            if a.required and a.option_strings:
                if a.dest not in orig and a.dest not in overrides:
                    raise RuntimeError(f"required arg {a.dest} has no value")
                argv += [a.option_strings[0],
                         str(overrides.get(a.dest, orig.get(a.dest)))]
        ns = real(self, argv)
        dests = [a.dest for a in self._actions if a.dest != "help"]
        seen["dests"] = dests
        for d in dests:
            if d in orig:
                setattr(ns, d, orig[d])
        for k, v in overrides.items():
            if k not in dests:
                raise RuntimeError(f"override {k} is not a trainer argument")
            setattr(ns, k, v)
        return ns

    argparse.ArgumentParser.parse_args = fake
    try:
        ns = getattr(mod, fn_name)()
    finally:
        argparse.ArgumentParser.parse_args = real
    return ns, seen.get("dests", [])


def diff(orig, resolved, permitted_changed):
    o = {k: _norm(v) for k, v in orig.items()}
    r = {k: _norm(v) for k, v in resolved.items()}
    changed = {k: {"orig": o[k], "new": r[k]} for k in sorted(set(o) & set(r))
               if o[k] != r[k]}
    orig_only = {k: o[k] for k in sorted(set(o) - set(r))}
    new_only = {k: r[k] for k in sorted(set(r) - set(o))}
    refusals = []
    for k in changed:
        if k not in permitted_changed:
            refusals.append(f"changed key {k}: {changed[k]}")
    for k in orig_only:
        refusals.append(f"original key {k}={orig_only[k]!r} not accepted by the "
                        f"current trainer")
    for k, v in new_only.items():
        if k not in PERMITTED_NEW:
            refusals.append(f"new key {k}={v!r} not in PERMITTED_NEW")
        elif PERMITTED_NEW[k][0] != v:
            refusals.append(f"new key {k}={v!r} != vetted default "
                            f"{PERMITTED_NEW[k][0]!r}")
    return {
        "changed": changed,
        "changed_reasons": {k: permitted_changed.get(k) for k in changed},
        "orig_only": orig_only,
        "new_only": new_only,
        "new_only_reasons": {k: PERMITTED_NEW.get(k, (None, None))[1]
                             for k in new_only},
        "refusals": refusals,
        "verdict": "PERMITTED" if not refusals else "REFUSED",
    }


def config_sha(d):
    return hashlib.sha256(json.dumps({k: _norm(v) for k, v in d.items()},
                                     sort_keys=True, default=str).encode()
                          ).hexdigest()
