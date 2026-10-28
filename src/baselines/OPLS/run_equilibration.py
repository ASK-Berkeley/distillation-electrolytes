"""
Equilibration for the molarity-audited OPLS boxes.

Per box, each stage starts from the previous one's final state:

    em    steepest-descent geometric relaxation   (removes packmol overlaps)
    npt   NPT relaxation at T, 1 bar              (default 1 ns, C-rescale)

The NPT stage draws a fresh Maxwell-Boltzmann velocity field at the target
temperature (`gen-vel = yes`), so it heats the box itself -- there is no
separate NVT stage. `--heat` inserts one (an NVT annealing ramp, default
10 K -> T over 200 ps) between em and npt if a gentler start is wanted.

The packed boxes come straight out of packmol and contain contacts that would
blow up an unminimised MD step, so `em` is mandatory either way.

Input:  <CM_BOX_ROOT>/<box>/<box>.gro (+ .top + .itp)
Output: <CM_EQ_ROOT>/<box>/{em,npt}.{mdp,tpr,log,gro,cpt,edr,xtc}
        <CM_EQ_ROOT>/<box>/eq_meta.json   -- protocol actually used

The final `npt.gro` + `npt.cpt` are the hand-off to production
(OPLS_code/02_production/run_production_npt_nvt.py).

Idempotent / resumable: a finished stage is skipped; an interrupted MD stage
resumes from its checkpoint (mdrun -cpi). Re-run to continue; never delete
outputs to "retry".

Usage:
    source config.sh
    python run_equilibration.py --gpus 0,1        # all boxes, spread over 2 GPUs
    python run_equilibration.py --per-gpu 9       # 9 boxes at once on one GPU
    python run_equilibration.py --sys lipf6       # only matching boxes
    python run_equilibration.py --dry-run         # write dirs + mdps, run nothing
"""

import os
import re
import glob
import json
import shutil
import argparse
import subprocess
from threading import Thread

from lib import provenance

GMXBIN = os.environ.get("GMXBIN", "gmx")

_HERE = os.path.dirname(os.path.abspath(__file__))
# Boxes built by build_opls_boxes.py (this dir) land in opls_ready_boxes/.
BOX_ROOT = os.environ.get("CM_BOX_ROOT", os.path.join(_HERE, "opls_ready_boxes"))
EQ_ROOT = os.environ.get("CM_EQ_ROOT", os.path.join(_HERE, "runs", "equilibrated"))

# Protocol defaults. dt / constraints match OPLS_code/02_production so the
# density this stage produces is equilibrated under the *same* Hamiltonian the
# production run uses (flexible hydrogens, 1 fs).
DT_PS = 0.001
CONSTRAINTS = "none"
START_TEMP = float(os.environ.get("CM_START_TEMP", 10))
HEAT_PS = float(os.environ.get("CM_HEAT_PS", 200))
NPT_NS = float(os.environ.get("CM_NPT_NS", 1))

# ----------------------------------------------------------------------------
# MDP templates
# ----------------------------------------------------------------------------
EM_MDP = """\
; Stage: em -- geometric relaxation (steepest descent)
integrator = steep
nsteps     = {nsteps}
emtol      = {emtol}
emstep     = 0.005

cutoff-scheme = Verlet
nstlist       = 10
pbc           = xyz
rlist         = 1.2
coulombtype   = PME
rcoulomb      = 1.2
vdwtype       = Cut-off
rvdw          = 1.2
DispCorr      = EnerPres

nstxout   = 0
nstvout   = 0
nstenergy = 100
nstlog    = 100
"""

# Heating: V-rescale (robust for a just-minimised box; Nose-Hoover is not
# meant to be driven through a temperature ramp) + simulated annealing from
# START_TEMP to the target. No pressure coupling -- the volume is whatever the
# packing gave us and is corrected in the NPT stage that follows.
HEAT_MDP = """\
; Stage: heat (optional, --heat) -- NVT heating ramp {t0} K -> {temp} K over {ps} ps
integrator = md
nsteps     = {nsteps}
dt         = {dt}

tcoupl  = v-rescale
tc-grps = System
tau_t   = 0.1
ref_t   = {temp}

; linear anneal; ref_t is reached at the end of the ramp and held there
annealing        = single
annealing-npoints = 2
annealing-time    = 0 {ps}
annealing-temp    = {t0} {temp}

pcoupl = no

constraints          = {constraints}
constraint-algorithm = LINCS
lincs-iter           = 1
lincs-order          = 4

cutoff-scheme = Verlet
nstlist       = 20
; 1e-3 (GROMACS default 5e-3): keeps the pair-list drift error below 0.1% in T
verlet-buffer-tolerance = 1e-3
pbc           = xyz
rlist         = 1.2
coulombtype   = PME
rcoulomb      = 1.2
vdwtype       = Cut-off
rvdw          = 1.2
DispCorr      = EnerPres

nstxout-compressed = 1000
compressed-x-grps  = System
nstenergy          = 500
nstlog             = 500
nstxout            = 0
nstvout            = 0

; fresh velocities at the bottom of the ramp
gen-vel      = yes
gen-temp     = {t0}
gen-seed     = -1
continuation = no
"""

# NPT: C-rescale is the stochastic barostat GROMACS recommends for
# equilibration -- it samples the correct ensemble and, unlike
# Parrinello-Rahman/MTTK, tolerates a box that is still far from its
# equilibrium volume. Production switches to MTTK from here.
NPT_MDP = """\
; Stage: npt -- NPT relaxation, {ns} ns at {temp} K / 1 bar
integrator = md
nsteps     = {nsteps}
dt         = {dt}

tcoupl  = v-rescale
tc-grps = System
tau_t   = 0.1
ref_t   = {temp}

pcoupl          = c-rescale
pcoupltype      = isotropic
tau_p           = 2.0
ref_p           = 1.0
compressibility = 4.5e-5

constraints          = {constraints}
constraint-algorithm = LINCS
lincs-iter           = 1
lincs-order          = 4

cutoff-scheme = Verlet
nstlist       = 20
; 1e-3 (GROMACS default 5e-3): keeps the pair-list drift error below 0.1% in T
verlet-buffer-tolerance = 1e-3
pbc           = xyz
rlist         = 1.2
coulombtype   = PME
rcoulomb      = 1.2
vdwtype       = Cut-off
rvdw          = 1.2
DispCorr      = EnerPres

nstxout-compressed = 1000
compressed-x-grps  = System
nstenergy          = 500
nstlog             = 500
nstxout            = 0
nstvout            = 0

{velocities}
"""

# Straight from EM: draw a fresh Maxwell-Boltzmann field at the target
# temperature. v-rescale + c-rescale absorb the initial transient.
VEL_FRESH = """\
gen-vel      = yes
gen-temp     = {temp}
gen-seed     = -1
continuation = no"""

# With --heat: continue from the ramp (coords via -c, velocities via -t cpt).
VEL_CONTINUE = """\
gen-vel      = no
continuation = yes"""


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def get_temperature(box_rel):
    """Target temperature (K) from the box dir name, e.g. lipf6_dme_0_5M_273.

    Same convention as OPLS_code/02_production/run_production_npt_nvt.py, so a
    box equilibrated here can be handed straight to production. Falls back to
    the `temperature_K` field of meta.json, then to 298 K.
    """
    m = re.search(r"_(\d{3})$", box_rel.split(os.sep)[-1])
    if m:
        return float(m.group(1))
    meta = os.path.join(BOX_ROOT, box_rel, "meta.json")
    if os.path.exists(meta):
        with open(meta) as f:
            t = json.load(f).get("temperature_K")
        if t:
            return float(t)
    return 298.0


def is_tegdme(name):
    n = name.lower()
    return "tegdme" in n or "tgdme" in n


def md_gpu_flags(name, ntomp):
    # No -update gpu: the production stage cannot use it (Nose-Hoover/MTTK), and
    # keeping the same execution path here avoids a second set of GPU surprises.
    # -pin off / -notunepme: with several mdruns stacked on one GPU they would
    # otherwise all pin to the same cores and each re-tune PME against a load
    # that is not theirs.
    common = ["-ntmpi", "1", "-ntomp", str(ntomp), "-pin", "off", "-notunepme"]
    # CM_TEGDME_PME=gpu lifts the TEGDME workaround on machines where GPU PME is
    # fine for it (verified on Perlmutter A100 / GROMACS 2025.3 CUDA). Where PME
    # runs is a performance setting only; it does not change the physics.
    if is_tegdme(name) and os.environ.get("CM_TEGDME_PME", "cpu") != "gpu":
        # TEGDME PME grid is touchy on GPU here -> keep PME on CPU too.
        return ["-nb", "gpu", "-pme", "cpu", "-update", "cpu"] + common
    return ["-nb", "gpu", "-pme", "gpu", "-update", "cpu"] + common


def discover_boxes():
    """Return [(relpath, system_name)] for every box dir with a matching .top/.gro."""
    out = []
    for top in glob.glob(f"{BOX_ROOT}/**/*.top", recursive=True):
        d = os.path.dirname(top)
        name = os.path.splitext(os.path.basename(top))[0]
        if os.path.exists(os.path.join(d, f"{name}.gro")):
            out.append((os.path.relpath(d, BOX_ROOT), name))
    return sorted(out)


def setup_run_dir(run_dir, box_dir, name):
    """Copy the box + its topology into the run dir so the run is self-contained."""
    os.makedirs(run_dir, exist_ok=True)
    srcs = (glob.glob(f"{box_dir}/*.itp") + glob.glob(f"{box_dir}/*.top")
            + [f"{box_dir}/{name}.gro"] + glob.glob(f"{box_dir}/meta.json"))
    for f in srcs:
        dst = os.path.join(run_dir, os.path.basename(f))
        if not os.path.exists(dst):
            shutil.copy2(f, dst)


def tpr_nsteps(tpr):
    r = subprocess.run([GMXBIN, "dump", "-s", tpr], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("nsteps"):
            return int(line.split("=")[1].strip())
    return None


def stage_done(run_dir, mode, target_nsteps):
    """True only if the stage finished *and* was run with the current nsteps."""
    gro = os.path.join(run_dir, f"{mode}.gro")
    log = os.path.join(run_dir, f"{mode}.log")
    if not os.path.exists(gro) or not os.path.exists(log):
        return False
    tpr = os.path.join(run_dir, f"{mode}.tpr")
    if not os.path.exists(tpr) or tpr_nsteps(tpr) != target_nsteps:
        return False
    with open(log) as f:
        text = f.read()
    if "Finished mdrun" not in text:
        return False
    # EM may legitimately stop early on emtol. An MD stage must have reached its
    # last step: an mdrun stopped by SIGTERM (scancel, wall-time limit) also
    # writes a .gro and logs "Finished mdrun", and must be resumed, not accepted.
    if mode == "em":
        return True
    return last_logged_step(text) >= target_nsteps


def last_logged_step(log_text):
    """Step number of the last energy block in an mdrun log (-1 if none)."""
    last = -1
    lines = log_text.splitlines()
    for i, line in enumerate(lines[:-1]):
        if line.split()[:2] == ["Step", "Time"]:
            try:
                last = max(last, int(lines[i + 1].split()[0]))
            except (ValueError, IndexError):
                pass
    return last


def run(cmd, run_dir, logname, env=None):
    log = os.path.join(run_dir, logname)
    with open(log, "a") as lf:
        lf.write("\n# " + " ".join(cmd) + "\n")
        lf.flush()
        r = subprocess.run(cmd, cwd=run_dir, stdout=lf, stderr=subprocess.STDOUT, env=env)
    return r.returncode == 0


def run_stage(run_dir, mode, mdp_text, gro_in, top, name, gpu, target_nsteps,
              cpt_in=None, minimise=False, dry_run=False, ntomp=4):
    """grompp (if needed) + mdrun (resumable). Returns the stage .gro, or None on failure."""
    tag = f"[{mode.upper()}] {os.path.relpath(run_dir, EQ_ROOT)}"
    if not dry_run and stage_done(run_dir, mode, target_nsteps):
        print(f"  {tag}: done, skipping")
        return os.path.join(run_dir, f"{mode}.gro")

    mdp_file = os.path.join(run_dir, f"{mode}.mdp")
    if not os.path.exists(mdp_file):
        with open(mdp_file, "w") as f:
            f.write(mdp_text)

    if dry_run:
        print(f"  {tag}: dry-run, wrote {os.path.basename(mdp_file)}")
        return os.path.join(run_dir, f"{mode}.gro")

    tpr = os.path.join(run_dir, f"{mode}.tpr")
    if not os.path.exists(tpr):
        # -maxwarn 0: a grompp WARNING means something is wrong with the
        # topology or the mdp and must be fixed, not waved through.
        cmd = [GMXBIN, "grompp", "-f", mdp_file, "-c", gro_in, "-p", top,
               "-o", tpr, "-maxwarn", "0"]
        if cpt_in and os.path.exists(cpt_in):
            cmd += ["-t", cpt_in]
        print(f"  {tag}: grompp...")
        if not run(cmd, run_dir, f"{mode}.log"):
            print(f"  {tag}: grompp FAILED -- see {mode}.log")
            return None

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    # Minimisation on the GPU buys nothing and is a common source of trouble;
    # run it on the CPU.
    flags = (["-ntmpi", "1", "-ntomp", str(ntomp), "-pin", "off"] if minimise
             else md_gpu_flags(name, ntomp))
    cmd = [GMXBIN, "mdrun", "-deffnm", mode, "-s", tpr] + flags
    cpt = os.path.join(run_dir, f"{mode}.cpt")
    if not minimise and os.path.exists(cpt):
        cmd += ["-cpi", cpt]   # resume an interrupted run
    print(f"  {tag}: mdrun (gpu {gpu})..." if not minimise else f"  {tag}: mdrun (cpu)...")
    if not run(cmd, run_dir, f"{mode}.log", env=env):
        print(f"  {tag}: mdrun FAILED -- see {mode}.log")
        return None
    return os.path.join(run_dir, f"{mode}.gro")


def run_box(box_rel, name, gpu, args):
    box_dir = os.path.join(BOX_ROOT, box_rel)
    run_dir = os.path.join(EQ_ROOT, box_rel)
    temp = get_temperature(box_rel)

    heat_nsteps = int(round(args.heat_ps / args.dt))
    npt_nsteps = int(round(args.npt_ns * 1000 / args.dt))

    print(f"\n=== {box_rel}  (T={temp} K, gpu {gpu}) ===")
    setup_run_dir(run_dir, box_dir, name)
    top = glob.glob(f"{run_dir}/*.top")[0]
    start_gro = os.path.join(run_dir, f"{name}.gro")

    # 1 -- geometric relaxation
    em_mdp = EM_MDP.format(nsteps=args.em_steps, emtol=args.emtol)
    em_gro = run_stage(run_dir, "em", em_mdp, start_gro, top, name, gpu,
                       target_nsteps=args.em_steps, minimise=True, dry_run=args.dry_run,
                       ntomp=args.ntomp)
    if em_gro is None or args.stop_after == "em":
        return

    # 2 -- optional NVT heating ramp START_TEMP -> temp (off by default)
    npt_in, npt_cpt = em_gro, None
    if args.heat:
        heat_mdp = HEAT_MDP.format(t0=args.start_temp, temp=temp, ps=args.heat_ps,
                                   nsteps=heat_nsteps, dt=args.dt,
                                   constraints=args.constraints)
        heat_gro = run_stage(run_dir, "heat", heat_mdp, em_gro, top, name, gpu,
                             target_nsteps=heat_nsteps, dry_run=args.dry_run, ntomp=args.ntomp)
        if heat_gro is None or args.stop_after == "heat":
            return
        npt_in, npt_cpt = heat_gro, os.path.join(run_dir, "heat.cpt")

    # 3 -- NPT relaxation at temp; velocities from the heat cpt, or fresh at temp
    velocities = (VEL_CONTINUE if args.heat else VEL_FRESH).format(temp=temp)
    npt_mdp = NPT_MDP.format(ns=args.npt_ns, temp=temp, nsteps=npt_nsteps,
                             dt=args.dt, constraints=args.constraints,
                             velocities=velocities)
    npt_gro = run_stage(run_dir, "npt", npt_mdp, npt_in, top, name, gpu,
                        target_nsteps=npt_nsteps, cpt_in=npt_cpt, dry_run=args.dry_run,
                        ntomp=args.ntomp)
    if npt_gro is None or args.dry_run:
        return

    with open(os.path.join(run_dir, "eq_meta.json"), "w") as f:
        json.dump({
            "box": box_rel,
            "name": name,
            "temperature_K": temp,
            "dt_ps": args.dt,
            "constraints": args.constraints,
            "em_steps": args.em_steps,
            "em_emtol": args.emtol,
            "heat_stage": bool(args.heat),
            "heat_from_K": args.start_temp if args.heat else None,
            "heat_ps": args.heat_ps if args.heat else None,
            "npt_ns": args.npt_ns,
            "barostat": "c-rescale",
            "verlet_buffer_tolerance": 1e-3,
            "source_box": box_dir,
            "provenance": provenance.record(GMXBIN, run_dir),
        }, f, indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpus", default=os.environ.get("CUDA_DEVICE", "0"),
                   help="comma-separated GPU ids; boxes are spread across them "
                        "(one active mdrun per GPU). Default: $CUDA_DEVICE or 0")
    p.add_argument("--sys", nargs="*",
                   help="only run boxes whose relative path contains one of these substrings")
    p.add_argument("--stop-after", choices=["em", "heat"], default=None,
                   help="stop after this stage (default: run through NPT)")
    p.add_argument("--heat", action="store_true",
                   help="insert the NVT heating ramp between em and npt; off by "
                        "default, in which case npt draws its own velocities at T")
    p.add_argument("--dt", type=float, default=DT_PS,
                   help=f"timestep in ps (default {DT_PS}, matching production)")
    p.add_argument("--constraints", default=CONSTRAINTS, choices=["none", "h-bonds", "all-bonds"],
                   help=f"bond constraints (default {CONSTRAINTS}, matching production)")
    p.add_argument("--em-steps", type=int, default=50000, help="max steepest-descent steps")
    p.add_argument("--emtol", type=float, default=100.0, help="EM force tolerance, kJ/mol/nm")
    p.add_argument("--start-temp", type=float, default=START_TEMP,
                   help=f"heating ramp start temperature, K (default {START_TEMP})")
    p.add_argument("--heat-ps", type=float, default=HEAT_PS,
                   help=f"heating ramp length, ps (default {HEAT_PS})")
    p.add_argument("--npt-ns", type=float, default=NPT_NS,
                   help=f"NPT relaxation length, ns (default {NPT_NS})")
    p.add_argument("--per-gpu", type=int, default=6,
                   help="concurrent boxes per GPU (default 6). These systems are "
                        "tiny; one mdrun does not come close to filling a card.")
    p.add_argument("--ntomp", type=int, default=4,
                   help="OpenMP threads per mdrun (default 4). per-gpu x ntomp x "
                        "n_gpus must fit the CPUs the job actually holds.")
    p.add_argument("--shard", default=None, metavar="I/N",
                   help="run only shard I of N over the sorted box list (e.g. 0/2). "
                        "Lets two SLURM jobs split the set without overlapping.")
    p.add_argument("--dry-run", action="store_true",
                   help="create run dirs and write mdps, but call no gmx")
    args = p.parse_args()

    if not os.path.isdir(BOX_ROOT):
        raise SystemExit(f"box root not found: {BOX_ROOT} (set CM_BOX_ROOT)")
    heat_str = (f"heat({args.start_temp} K -> T, {args.heat_ps} ps) -> "
                if args.heat else "")
    print(f"Boxes: {BOX_ROOT}\nOutput: {EQ_ROOT}")
    print(f"Protocol: em(<={args.em_steps} steps) -> {heat_str}"
          f"npt({args.npt_ns} ns, gen-vel={'no' if args.heat else 'yes'}), "
          f"dt={args.dt} ps, constraints={args.constraints}")

    boxes = discover_boxes()
    if args.sys:
        boxes = [b for b in boxes if any(s in b[0] for s in args.sys)]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        if not 0 <= i < n:
            raise SystemExit(f"bad --shard {args.shard}: need 0 <= I < N")
        # Stride, not contiguous blocks: the box list is sorted, so a stride
        # interleaves cheap and expensive solvents across the two shards
        # instead of piling every TEGDME box into one of them.
        boxes = boxes[i::n]
        print(f"Shard {i}/{n}")
    print(f"Found {len(boxes)} box(es):")
    for rel, _ in boxes:
        print(f"  {rel}  (T={get_temperature(rel)} K)")
    if not boxes:
        return

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]

    # These boxes are ~3000 atoms: a single mdrun leaves an RTX 6000 Ada almost
    # idle, so stack --per-gpu of them on each card. Slots = gpus x per_gpu
    # worker threads, each chewing through its stride of the box list; every
    # slot on a GPU shares that card via the CUDA scheduler.
    slots = [(gpu, k) for gpu in gpus for k in range(args.per_gpu)]
    print(f"{len(slots)} concurrent slot(s): {len(gpus)} GPU(s) x {args.per_gpu} "
          f"per GPU, {args.ntomp} OpenMP thread(s) each "
          f"({len(slots) * args.ntomp} cores total)")

    def worker(my_boxes, gpu):
        for rel, name in my_boxes:
            run_box(rel, name, gpu, args)

    threads = []
    for si, (gpu, _) in enumerate(slots):
        my_boxes = boxes[si::len(slots)]
        if not my_boxes:
            continue
        t = Thread(target=worker, args=(my_boxes, gpu))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    print("\nAll done.")


if __name__ == "__main__":
    main()
