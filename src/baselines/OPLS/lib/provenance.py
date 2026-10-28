"""Run provenance for the OPLS baseline: what GROMACS, which topology files, which commit.

Written into eq_meta.json / rep_meta.json so that every run directory records
exactly which force-field files and code produced it. A reviewer (or a later
us) can then tell a run made with the buggy PF6 topology from a run made with
the corrected one by hash alone, without diffing files.
"""
import os
import glob
import hashlib
import subprocess


def gmx_version(gmxbin):
    """'2025.3' etc. from `gmx --version`, or 'unknown'."""
    try:
        out = subprocess.run([gmxbin, "--version"], capture_output=True, text=True).stdout
    except OSError:
        return "unknown"
    for line in out.splitlines():
        if line.startswith("GROMACS version"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def gmx_build_info(gmxbin):
    """Precision / GPU / SIMD lines from `gmx --version`, as a dict."""
    keys = ("Precision", "GPU support", "SIMD instructions", "CUDA driver", "CUDA runtime")
    info = {}
    try:
        out = subprocess.run([gmxbin, "--version"], capture_output=True, text=True).stdout
    except OSError:
        return info
    for line in out.splitlines():
        for k in keys:
            if line.startswith(k + ":"):
                info[k.lower().replace(" ", "_")] = line.split(":", 1)[1].strip()
    return info


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def topology_hashes(run_dir):
    """{filename: sha256} for every .itp/.top in run_dir."""
    files = sorted(glob.glob(os.path.join(run_dir, "*.itp")) + glob.glob(os.path.join(run_dir, "*.top")))
    return {os.path.basename(f): sha256(f) for f in files}


def git_sha(start=None):
    """Commit of the repository containing `start` (default: this file), or None."""
    # A run launched from a code snapshot outside git passes the commit in.
    if os.environ.get("OPLS_CODE_SHA"):
        return os.environ["OPLS_CODE_SHA"]
    cwd = os.path.dirname(os.path.abspath(start or __file__))
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True)
        if r.returncode != 0:
            return None
        sha = r.stdout.strip()
        return sha + ("-dirty" if dirty.stdout.strip() else "")
    except OSError:
        return None


def record(gmxbin, run_dir):
    """Everything the drivers store under the 'provenance' key."""
    return {
        "gromacs_version": gmx_version(gmxbin),
        "gromacs_build": gmx_build_info(gmxbin),
        "gromacs_binary": gmxbin,
        "topology_sha256": topology_hashes(run_dir),
        "code_git_sha": git_sha(),
    }
