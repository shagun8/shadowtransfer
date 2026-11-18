"""Loud failure for a path that has been archived to Fortress.

LIVE since 2026-09-22. Imported by armc_runtime.preflight_read_targets and
by armf_sar_cells.resolve_cell. It went live when the operator confirmed
Fortress-side verification (Globus task 29bd16e3-b699-11f1-bb62-02ffe792127d,
812/812 files, 0 faults, --verify-checksum on) and the 812 archived copies
were deleted from /scratch.

The problem it solves: once the archive set leaves /scratch, a script that
resolves one of those checkpoints gets a bare `FileNotFoundError`, or -- worse
-- `armf_sar_cells.resolve_cell()`'s `SystemExit(f"FATAL missing {p}")`, which
reads as data loss rather than as a recall step. This turns that into an
instruction.

A marker file `.ARCHIVED_FORTRESS` is left in each archived directory naming
the tree root and the manifest; this module reads it and reports it.
"""
from pathlib import Path

MARKER = ".ARCHIVED_FORTRESS"
MANIFEST = "logs/fortress_2026-09-09/archive_manifest_2026-09-09.csv"


class ArchivedToFortress(RuntimeError):
    """Raised when a needed path is absent but marked archived."""


def find_marker(path, max_up=6):
    """The nearest .ARCHIVED_FORTRESS at or above `path`, or None."""
    p = Path(path)
    d = p if p.is_dir() else p.parent
    for _ in range(max_up):
        m = d / MARKER
        try:
            if m.is_file():
                return m
        except OSError:
            return None
        if d.parent == d:
            break
        d = d.parent
    return None


def check(path, what="file"):
    """Return None if `path` exists.

    If it does not exist and an .ARCHIVED_FORTRESS marker covers it, raise
    ArchivedToFortress carrying the recall instruction. If it does not exist
    and no marker covers it, return a plain 'missing, not archived' string so
    the caller can keep its own error.
    """
    p = Path(path)
    if p.exists():
        return None
    m = find_marker(p)
    if m is None:
        return f"missing and NOT marked archived: {p}"
    try:
        body = m.read_text().strip()
    except OSError as e:  # noqa: BLE001
        body = f"(marker unreadable: {e})"
    raise ArchivedToFortress(
        f"\n{'=' * 74}\n"
        f"ARCHIVED TO FORTRESS -- this {what} is not lost, it was moved.\n"
        f"  wanted   {p}\n"
        f"  marker   {m}\n"
        f"{body}\n"
        f"  RECALL: pull this path back from Fortress with Globus, using the\n"
        f"          endpoint and tree_root named in the marker above, confirm\n"
        f"          the recalled size matches the byte count the marker gives\n"
        f"          for this file, then delete the {MARKER} marker.\n"
        f"          (The manifest's sha256 column is EMPTY -- the local\n"
        f"          checksum job was never submitted. Integrity evidence is\n"
        f"          the Globus --verify-checksum result plus the size match.\n"
        f"          Manifest, for the file list: {MANIFEST})\n"
        f"  Do NOT retrain to work around this.\n"
        f"{'=' * 74}"
    )


def require(path, what="file"):
    """`check`, but also raise on a plain missing path. Use at a hard gate."""
    msg = check(path, what)
    if msg is not None:
        raise FileNotFoundError(msg)
