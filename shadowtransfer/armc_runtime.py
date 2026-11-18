"""
Arm C worker runtime: switch off published-trainer side effects, and time-box
everything that runs after training.

Why this module exists
----------------------
Job 20540648 trained cleanly (early stop at epoch 26, best at 16) and then hung
for ~90 minutes on a 0-byte `metrics_curves.png`, against a 2-hour wall clock.
The dump directory was never created, so the cell produced nothing at all. The
figure call is `python/mamnet/train.py:694` (`plot_metrics_curves`), inside
`Trainer.train()`.

That file is published code: it produced the numbers in Tables 30-32 and other
things in the tree depend on its behaviour. It is NOT edited. Instead the arm C
wrapper rebinds the plotting names, and the tensorboard writer, on the
architecture's own train MODULE after importing it. `train.py` refers to
`plot_loss_curves`, `plot_metrics_curves` and `SummaryWriter` as module globals
(imported at `:41-45` and `:25`), so rebinding the module attribute changes what
`Trainer.train()` and `Trainer.__init__` resolve, and nothing on disk changes.

Rebinding is used rather than forcing `matplotlib.use("Agg")` because Agg only
changes the backend: the call still runs, still walks the font cache, and still
writes a PNG to a loaded parallel filesystem. It would have made the hang less
likely, not impossible. Rebinding removes the call. This box does not build
figures at all (CLAUDE.md; CONVENTIONS.md puts them on the SSH box, built from
results JSONs), so nothing is lost.

Periodic checkpointing is switched off through the published config's own
`save_freq` knob rather than by patching, because `train.py:564` already gates
on it: `if epoch % self.args.save_freq == 0`. Setting it beyond any reachable
epoch leaves `checkpoint_best.pth` (written on the `is_best` branch at `:559`,
which is not gated on `save_freq`) as the only checkpoint a cell keeps.
"""

import errno
import faulthandler
import json
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# `epoch % SAVE_FREQ_OFF` is never 0 for any epoch a run can reach. Epochs are
# 1-based here (`save_checkpoint(epoch + 1)` at train.py:667), so epoch 0 -- the
# one value that would still match -- never occurs.
SAVE_FREQ_OFF = 10 ** 9


# ---------------------------------------------------------------------------
# Side effects
# ---------------------------------------------------------------------------

def _noop(*_a, **_k):
    return None


class NullWriter:
    """Stands in for torch.utils.tensorboard.SummaryWriter.

    Accepts and discards every add_*/flush call. `close()` is defined so it does
    not go through __getattr__, which is only consulted on failed lookups.
    """

    def __init__(self, *a, **_k):
        self.log_dir = a[0] if a else None

    def __getattr__(self, _name):
        return _noop

    def close(self):
        return None


def silence_worker_side_effects(archtrain, verbose=True):
    """Disable figures and tensorboard on an architecture's train module.

    Call AFTER importing the module and BEFORE constructing its Trainer -- the
    writer is built in `Trainer.__init__` (train.py:172), so a later rebind
    would be too late for it.

    Returns the list of names actually rebound, for run.json.
    """
    disabled = []

    for name in ("plot_loss_curves",
                 "plot_metrics_curves",
                 "save_best_worst_visualizations",
                 "plot_confusion_matrix",
                 "visualize_predictions"):
        if hasattr(archtrain, name):
            setattr(archtrain, name, _noop)
            disabled.append(name)

    if hasattr(archtrain, "SummaryWriter"):
        setattr(archtrain, "SummaryWriter", NullWriter)
        disabled.append("SummaryWriter")

    # Collapse guard. Installed here because this is the one call every arm's
    # trainer path already makes on the architecture's train module, after the
    # import and before the Trainer is built. It is not a "side effect" and is
    # deliberately NOT appended to `disabled`; run.json records it separately
    # from armc_runtime.COLLAPSE_GUARD_STATE.
    install_collapse_guard(archtrain, verbose=verbose)

    if verbose:
        print(f"[armc_runtime] disabled on {archtrain.__name__}: "
              f"{disabled or 'nothing found'}", flush=True)
    return disabled


def disable_periodic_checkpoints(cfg, verbose=True):
    """Mutate a config dict so only checkpoint_best.pth is kept.

    Three 509 MB checkpoints per cell (best + epoch_10 + epoch_20) is ~1.5 GB;
    the 81-cell grid would need ~122 GB against ~120 GB of headroom below the
    550G hard quota. Best-only brings the grid to ~41 GB.

    Note the cost: `armc_train.newest_epoch_checkpoint` resumes a killed cell
    from its newest periodic checkpoint, and with none written a killed cell
    restarts from epoch 0. That is the intended trade -- at ~27 s/epoch a rung
    225 cell is ~12 minutes of training.
    """
    was = cfg.get("save_freq")
    cfg["save_freq"] = SAVE_FREQ_OFF
    if verbose:
        print(f"[armc_runtime] save_freq {was} -> {SAVE_FREQ_OFF} "
              f"(checkpoint_best.pth only)", flush=True)
    return was


# ---------------------------------------------------------------------------
# Failure marker
# ---------------------------------------------------------------------------

def mark_failed(dump_dir, phase, reason):
    """Touch .FAILED with enough context to tell it apart from a cell that
    never ran. Never raises: a marker that can fail is not a marker.

    A partial parquet without a .FAILED beside it is the one state CLAUDE.md
    forbids, so this is also what the timeout handler calls before exiting.
    """
    try:
        d = Path(dump_dir)
        d.mkdir(parents=True, exist_ok=True)
        payload = {
            "phase": phase,
            "reason": reason,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "node": os.environ.get("SLURMD_NODENAME"),
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        (d / ".FAILED").write_text(json.dumps(payload, indent=2) + "\n")
        print(f"[armc_runtime] .FAILED written at {d/'.FAILED'} "
              f"(phase={phase})", flush=True)
    except Exception:
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Time-box
# ---------------------------------------------------------------------------

class PhaseTimeout:
    """Hard SIGALRM time-box over the post-training phase.

    On fire it dumps the stack of every thread (so the next hang names its own
    line rather than needing another 90-minute cell to find it), touches
    .FAILED, and leaves via os._exit so no `except`/`finally` anywhere can
    swallow it. That matters here: the failure mode being guarded against is a
    hang inside a C call, where raising a Python exception may never unwind.

    Usage:
        with PhaseTimeout(1200, dump_dir) as tb:
            tb.phase("inference")
            ...
            tb.phase("write_dump")
            ...
    """

    def __init__(self, seconds, dump_dir, label="post-training"):
        self.seconds = int(seconds)
        self.dump_dir = dump_dir
        self.label = label
        self.current = "not started"
        self._prev = None

    def phase(self, name):
        self.current = name
        print(f"[armc_runtime] phase -> {name}", flush=True)

    def _fire(self, _signum, _frame):
        print("\n" + "!" * 78, flush=True)
        print(f"[armc_runtime] TIMEOUT after {self.seconds}s during "
              f"{self.label} phase '{self.current}'", flush=True)
        print("!" * 78, flush=True)
        try:
            faulthandler.dump_traceback(file=sys.stdout, all_threads=True)
        except Exception:
            traceback.print_exc()
        sys.stdout.flush()
        mark_failed(self.dump_dir, self.current,
                    f"timeout after {self.seconds}s in {self.label}")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(2)

    def __enter__(self):
        if self.seconds > 0:
            self._prev = signal.signal(signal.SIGALRM, self._fire)
            signal.alarm(self.seconds)
            print(f"[armc_runtime] {self.label} time-box armed: "
                  f"{self.seconds}s", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.seconds > 0:
            signal.alarm(0)
            if self._prev is not None:
                signal.signal(signal.SIGALRM, self._prev)
        if exc_type is not None:
            mark_failed(self.dump_dir, self.current,
                        f"{exc_type.__name__}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Quarantined paths
# ---------------------------------------------------------------------------
#
# A path lands here when a writer was killed part-way through writing it and
# left a stale lock behind. The signature is specific and worth recognising:
# opening the path succeeds, metadata operations (ls, stat) succeed, reads
# usually succeed -- and the WRITE blocks forever instead of returning an
# error. Nothing surfaces; the job just stops making progress.
#
# This is not a sick filesystem, and diagnosing it as one wastes real time.
# Job 20556487 hung 1200 s writing a 150-row parquet at 17:00 after writing a
# 509 MB checkpoint to the same filesystem at 16:52, and job 20554422 wrote an
# equivalent parquet to a fresh path in 4 seconds. The filesystem was healthy
# throughout. It is the individual path that is poisoned, by an earlier
# SIGKILL, and it stays poisoned.
#
# Paths are relative to BASE_PATH. Add to this list, never remove from it
# without evidence the lock is gone. Keep it in step with CLAUDE.md's
# "Quarantined paths" section.

QUARANTINE = (
    # heredoc write and `wc -l` read both hung, login01, 2026-07-28
    "python/armc_n_backfill_submit.sh",
    # job 20551713 computed its whole 18-cell table, then hung writing
    # n_train/n_val into this cell's args.json, gpua043, 2026-07-28.
    # READS still work: mamnet/chicago cells load this as their published
    # config and did so successfully on 2026-07-28. Reads are time-boxed by
    # ReadGuard rather than refused.
    "data/mamnet/outputs/mamnet_loco_holdout_chicago_highres_1/args.json",
    # job 20546171 was SIGKILLed on the SLURM time limit mid-write to this
    # exact path; job 20556487 then hung 1200 s on the same path, 2026-07-28
    "dumps/st/mamnet-vanilla-loco-r225-phoenix-hr-s0/perimage.parquet",
    # append blocked in Lustre and never returned, writer (cat, PID 3580217)
    # stuck in cl_sync_io_wait, dt-login02, 2026-07-31. Sibling writes and a
    # write under summaries/ both completed instantly, so the path is
    # poisoned, not the filesystem. Arm F's decisions were routed to
    # DECISIONS_armF.md instead; see that file's header for the merge hazard.
    "DECISIONS.md",
    # ---- 2026-10-17: the 85 outage entries of 2026-10-16 were REMOVED -----
    # They were added 2026-10-16 (30 at gate 2, 55 at regen gate A) as
    # read-poisoned dump paths under the scratch dump roots. The
    # cause was NOT path poison: it was an NCSA outage of OST006a on dltawork
    # (announced 2026-10-16; `lfs check servers` showed the OST006a import in
    # "Cannot send after transport endpoint shutdown (108)"). `lfs getstripe`
    # 2026-10-17: the 74 that cleared first all sit on OST0069 (idx 105), the
    # 70 that cleared 2026-10-17 all on OST006a (idx 106); 200 never-hung
    # control files have objects on neither. Evidence for the
    # removal, all three required by the operator:
    #   1. NCSA's 2026-10-16 outage announcement for OST006a (operator-supplied);
    #   2. the OST006a transport error in `lfs check servers`;
    #   3. 2026-10-17 14:02, dt-login02: full census 1008 responsive / 0
    #      unresponsive / 99 absent of 1107, then the 70 still-hung-at-11:45
    #      paths re-read SEQUENTIALLY, abandoned child, 20 s budget, stat + FULL
    #      read with bytes == st_size: 70/70 OK, 21.6 MiB, 0 failures. The other
    #      15 had answered since 2026-10-16 ~22:20 (regen_gateA_reprobe30.json).
    # Caveat at removal time (14:15): one of login02's two OST006a client
    # imports still printed error 108; the other was active and every read
    # succeeded. If reads hang again, check `lfs check servers` for OST006a
    # BEFORE re-quarantining -- a dead OST is not a poisoned path.
    # Removal authorized by the operator 2026-10-17; the list of 85 is in
    # logs/quarantine_removed_2026-10-17.json and in git history (6ab2dbc).
)


def lexical_abspath(path, base=None):
    """Absolute, normalised path STRING, without touching the filesystem.

    os.path.normpath collapses ".", ".." and duplicate separators purely
    lexically -- no stat, no readlink, no resolve. A relative path is joined
    to `base` when one is given, else to the cwd, matching what resolve() did
    for relative inputs.
    """
    s = os.fspath(path)
    if not os.path.isabs(s):
        s = os.path.join(os.fspath(base), s) if base is not None \
            else os.path.abspath(s)
    return os.path.normpath(s)


def quarantined_targets(base, targets):
    """Return the subset of `targets` that is quarantined. LEXICAL ONLY.

    Compares normalised absolute path STRINGS. It does not stat, open,
    readlink or resolve anything, so it returns in microseconds even when
    every path in the list is poisoned.

    WHY -- this is the whole point of the 2026-10-16 change
    ------------------------------------------------------
    This used to compare `Path(p).resolve()`, and resolve() STATS the path. On
    2026-10-16, quarantining
      <scratch>/dumps/seo/oglanet-fda-loco-r3500-oma-hr-s1/inference_repeat.json
    -- a path whose `stat` does not return, on dt-login01 AND dt-login03,
    under budgets of 6 s, 6 s and 90 s, and which `timeout` cannot kill
    because the thread sits in uninterruptible Lustre I/O -- made the
    quarantine CHECK hang on its own list. preflight_write_targets is called
    by every cell, so the single mechanism that defends against the poison was
    disabled by the act of recording a poisoned path in it.

    A guard that must touch the hazard in order to report it is not a guard.
    That is the defect this fixes, and it is why the comparison is now purely
    lexical and must stay that way. Do not reintroduce resolve(), realpath(),
    exists() or any stat into this function or into QUARANTINE handling.

    WHAT THIS GIVES UP, stated rather than glossed
    ----------------------------------------------
    resolve() also collapsed SYMLINKS, so two spellings of one file compared
    equal. Lexical comparison does not: a symlink or bind-mount alias of a
    quarantined path will NOT match. That loss is accepted deliberately,
    because the alternative is a check that cannot be run at all on the paths
    that need it most.

    It has a concrete consequence here. CLAUDE.md records (2026-09-09) that
    <archive>/data/{arch}/outputs/{run_dir} ARE symlinks into <scratch>. So
    when a poisoned file is reachable by more than one spelling, BOTH spellings
    must be listed in QUARANTINE explicitly. "..", "." and duplicate
    separators are still collapsed, lexically, so a ../ detour cannot slip
    past -- only a symlink can.
    """
    banned = {lexical_abspath(q, base) for q in QUARANTINE}
    hits = []
    for t in targets:
        a = lexical_abspath(t, None)
        if a in banned:
            hits.append(a)
    return hits


def preflight_write_targets(base, targets, label="write"):
    """Refuse to run if any intended write target is quarantined.

    Called BEFORE the expensive part of a job rather than at the moment of
    writing. That ordering is the whole point: a quarantined cell should cost
    seconds, not a full training run followed by a hung write and a lost
    wall clock, which is exactly what job 20556487 cost.
    """
    targets = list(targets)
    hits = quarantined_targets(base, targets)
    if hits:
        print("!" * 78, flush=True)
        print(f"[armc_runtime] QUARANTINED {label} TARGET -- refusing to run",
              flush=True)
        for h in hits:
            print(f"    {h}", flush=True)
        print("  A write to these hangs rather than failing. See CLAUDE.md,",
              flush=True)
        print("  'Quarantined paths', for why and for what to do instead.",
              flush=True)
        print("!" * 78, flush=True)
        raise SystemExit(f"FATAL quarantined {label} target: {hits}")
    print(f"[armc_runtime] preflight OK: {len(targets)} {label} target(s), "
          f"none quarantined", flush=True)
    return hits


def preflight_read_targets(paths, label="read"):
    """Refuse early, and by name, on any input path that has been archived.

    Call it beside preflight_write_targets at the top of a cell, so an archived
    checkpoint costs seconds rather than a full training run followed by a
    confusing FileNotFoundError deep in a loader.

    Applied 2026-09-22, when the 812-file Fortress archive set was deleted from
    /scratch. A missing path that is covered by an .ARCHIVED_FORTRESS marker
    raises fortress_recall.ArchivedToFortress carrying the recall instruction;
    a missing path with no marker is collected and reported as genuinely
    missing, which is the pre-existing behaviour.

    NOTE the staged 2026-09-09 version of this patch inserted the function
    immediately after preflight_write_targets' `def` line, which would have
    broken that function. It is inserted after its `return hits` instead.
    """
    from shadowtransfer import fortress_recall
    paths = list(paths)
    missing = []
    for p in paths:
        msg = fortress_recall.check(p, what=label)   # raises if archived
        if msg is not None:
            missing.append(msg)
    if missing:
        raise FileNotFoundError(
            f"[armc_runtime] {len(missing)} {label} target(s) missing and not "
            f"marked archived:\n  " + "\n  ".join(missing))
    print(f"[armc_runtime] read preflight OK: {len(paths)} {label} target(s), "
          f"none archived", flush=True)
    return []


class ReadGuard:
    """Time-box a read that might touch a quarantined path.

    Reads from a quarantined path have been observed to work while its writes
    hang, so reads are allowed rather than refused -- some quarantined paths
    are published configs a cell genuinely needs. The guard exists so that a
    read which does stick fails in seconds with a name attached, instead of
    consuming the cell's wall clock in silence.
    """

    def __init__(self, seconds, what):
        self.seconds = int(seconds)
        self.what = what
        self._prev = None

    def _fire(self, _signum, _frame):
        raise TimeoutError(
            f"read of {self.what} did not return within {self.seconds}s -- "
            f"treat the path as quarantined (CLAUDE.md, 'Quarantined paths')")

    def __enter__(self):
        if self.seconds > 0:
            self._prev = signal.signal(signal.SIGALRM, self._fire)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.seconds > 0:
            signal.alarm(0)
            if self._prev is not None:
                signal.signal(signal.SIGALRM, self._prev)
        return False


def read_with_retry(seconds, what, fn, retry_after=120, _sleep=None):
    """Time-box a read, and retry it ONCE after `retry_after` seconds.

    Added 2026-10-28. On 2026-10-27 05:51-06:41 Lustre went unresponsive for
    about 50 minutes and destroyed 26 jobs. Two distinct signatures were seen on
    reads of published checkpoints:

      * ``OSError`` with ``errno == EIO`` (Errno 5) from a ``stat`` or ``open``
        on a path that reads fine minutes later;
      * a read that never returns, caught here by ``ReadGuard``'s SIGALRM
        (n450 job 23209696 died exactly this way, on the 300 s guard around
        the original checkpoint args, and took its two fan cells with it).

    Both are transient. A single retry after a short wait turns a lost cell
    into a delayed one. A second failure still raises, so a genuinely
    quarantined or dead path still fails loudly with a name attached rather
    than eating the wall clock.

    Only EIO and ReadGuard's TimeoutError are retried. Every other exception
    propagates on the first attempt -- a missing file or an unpickleable
    checkpoint is not a filesystem blip and must not be retried into silence.
    """
    sleep = _sleep if _sleep is not None else time.sleep
    last = None
    for attempt in (1, 2):
        try:
            with ReadGuard(seconds, what):
                return fn()
        except TimeoutError as e:
            last = e
            reason = f"read timeout after {seconds}s"
        except OSError as e:
            if getattr(e, "errno", None) != errno.EIO:
                raise
            last = e
            reason = f"OSError errno {e.errno} (EIO)"
        if attempt == 1:
            print(f"[read_with_retry] {what}: {reason}; retrying once in "
                  f"{retry_after}s (transient-Lustre retry, 2026-10-28)",
                  flush=True)
            sleep(retry_after)
        else:
            print(f"[read_with_retry] {what}: {reason} on the retry as well; "
                  f"failing", flush=True)
    raise last


# ---------------------------------------------------------------------------
# Collapse guard
# ---------------------------------------------------------------------------
#
# Why this exists
# ---------------
# oglanet-augs3-upper-r450-phoenix-hr-s1-e200 (job 20636133) reached its best
# validation tolerant mIoU of 1.16% at epoch 1, never improved, and early-stopped
# at epoch 21 with a final 2-class mIoU of 44.16 against 63.57 and 62.80 on its
# two sibling seeds. It is a failed optimisation, not a data point, and it alone
# drove that (arch, city)'s aug-upper mean.
#
# Nothing caught it. The existing sanity flags in armd_train/armd_e200 check
# epochs_run < 10, ~zero per-image variance, and a headline landing on its own
# baseline. A run that collapses at epoch 1 and then early-stops at 21 trips none
# of the three: it ran 21 epochs, its per-image IoU varies, and 44.16 is nowhere
# near its baseline.
#
# The guard fires DURING training, at the first epoch >= COLLAPSE_MIN_EPOCH whose
# running best decision metric is still below COLLAPSE_MIN_BEST. It raises, so
# the worker's EXIT trap writes .FAILED and the job exits nonzero; a collapsed
# cell then costs ten epochs instead of a full patience window plus a dump that
# looks like a number.
#
# It is installed from silence_worker_side_effects, which every arm calls on the
# architecture's train module immediately before constructing its Trainer, so
# every later arm inherits it without touching a published trainer or a
# per-arm script.
#
# The hook is Trainer.save_checkpoint(epoch, is_best), which all three published
# trainers call exactly once per epoch AFTER updating self.best_decision_miou
# (mamnet train.py:536/652, oglanet train.py:628/644, dinov3
# train_dinov3.py:570/673). Patching the class, not an instance, means the guard
# is live however the Trainer is built.

COLLAPSE_METRIC_ATTR = "best_decision_miou"
COLLAPSE_MIN_EPOCH_DEFAULT = 10
COLLAPSE_MIN_BEST_DEFAULT = 10.0

#: Filled by install_collapse_guard so a run.json can record what was live.
COLLAPSE_GUARD_STATE = {"installed": False, "reason": "not attempted"}


class CollapsedRun(RuntimeError):
    """Training collapsed: the decision metric never left the floor."""


def _env_num(name, default, cast):
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default, "default"
    try:
        return cast(raw), f"env {name}={raw}"
    except (TypeError, ValueError):
        print(f"[armc_runtime] WARNING {name}={raw!r} is not a number; "
              f"using {default}", flush=True)
        return default, f"default (env {name}={raw!r} unparseable)"


def install_collapse_guard(archtrain, min_epoch=None, min_best=None,
                           attr=COLLAPSE_METRIC_ATTR, verbose=True):
    """Patch archtrain.Trainer.save_checkpoint so a collapsed run fails loudly.

    Returns the state dict it also stores in COLLAPSE_GUARD_STATE. Never raises
    on an architecture it cannot hook: it records why and returns, because a
    guard that can break an otherwise healthy arm is worse than no guard.

    Env overrides, so an arm with a genuinely slow-starting recipe can move the
    thresholds without editing code and with the change visible in the log:
      ST_COLLAPSE_GUARD=0        switch the guard off entirely
      ST_COLLAPSE_MIN_EPOCH=N    first epoch the check is evaluated on
      ST_COLLAPSE_MIN_BEST=X     floor on the running best decision metric
    """
    if os.environ.get("ST_COLLAPSE_GUARD", "1").strip() == "0":
        state = {"installed": False, "reason": "ST_COLLAPSE_GUARD=0",
                 "metric_attr": attr}
        COLLAPSE_GUARD_STATE.clear()
        COLLAPSE_GUARD_STATE.update(state)
        if verbose:
            print("[armc_runtime] COLLAPSE GUARD OFF (ST_COLLAPSE_GUARD=0)",
                  flush=True)
        return state

    if min_epoch is None:
        min_epoch, epoch_src = _env_num("ST_COLLAPSE_MIN_EPOCH",
                                        COLLAPSE_MIN_EPOCH_DEFAULT, int)
    else:
        epoch_src = "caller"
    if min_best is None:
        min_best, best_src = _env_num("ST_COLLAPSE_MIN_BEST",
                                      COLLAPSE_MIN_BEST_DEFAULT, float)
    else:
        best_src = "caller"

    state = {
        "installed": False,
        "metric_attr": attr,
        "min_epoch": int(min_epoch),
        "min_best": float(min_best),
        "min_epoch_source": epoch_src,
        "min_best_source": best_src,
        "hook": f"{getattr(archtrain, '__name__', '?')}.Trainer.save_checkpoint",
        "rule": (f"raise CollapsedRun when epoch >= {int(min_epoch)} and the "
                 f"running {attr} is still < {float(min_best)}"),
        "reason": None,
    }

    T = getattr(archtrain, "Trainer", None)
    if T is None or not hasattr(T, "save_checkpoint"):
        state["reason"] = ("no Trainer.save_checkpoint on "
                           f"{getattr(archtrain, '__name__', archtrain)}")
        COLLAPSE_GUARD_STATE.clear()
        COLLAPSE_GUARD_STATE.update(state)
        if verbose:
            print(f"[armc_runtime] COLLAPSE GUARD NOT INSTALLED: "
                  f"{state['reason']}", flush=True)
        return state

    orig = T.save_checkpoint
    if getattr(orig, "_st_collapse_guard", False):
        state["installed"] = True
        state["reason"] = "already installed on this class"
        COLLAPSE_GUARD_STATE.clear()
        COLLAPSE_GUARD_STATE.update(state)
        return state

    def guarded_save_checkpoint(self, epoch, is_best=False, *a, **k):
        out = orig(self, epoch, is_best, *a, **k)
        best = getattr(self, attr, None)
        try:
            best = float(best)
        except (TypeError, ValueError):
            best = None
        if best is None:
            return out
        try:
            ep = int(epoch)
        except (TypeError, ValueError):
            return out
        if ep >= int(min_epoch) and best < float(min_best):
            raise CollapsedRun(
                f"COLLAPSE GUARD: at epoch {ep} the running best {attr} is "
                f"{best:.4f}, below the floor {float(min_best):.4f}. This is a "
                f"failed optimisation, not a data point -- the run that "
                f"prompted this guard "
                f"(oglanet-augs3-upper-r450-phoenix-hr-s1-e200) sat at 1.16 "
                f"for 20 epochs and then early-stopped at 21, producing a "
                f"dump that looked like a number. Failing here instead. "
                f"Override with ST_COLLAPSE_MIN_BEST / ST_COLLAPSE_MIN_EPOCH, "
                f"or ST_COLLAPSE_GUARD=0, only with a recorded reason.")
        return out

    guarded_save_checkpoint._st_collapse_guard = True
    guarded_save_checkpoint._st_wrapped = orig
    T.save_checkpoint = guarded_save_checkpoint
    state["installed"] = True
    state["reason"] = "ok"
    COLLAPSE_GUARD_STATE.clear()
    COLLAPSE_GUARD_STATE.update(state)
    if verbose:
        print(f"[armc_runtime] COLLAPSE GUARD LIVE on {state['hook']}: "
              f"{state['rule']} (min_epoch {state['min_epoch_source']}, "
              f"min_best {state['min_best_source']})", flush=True)
    return state
