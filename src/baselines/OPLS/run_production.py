"""
Production NVT transport runs for the molarity-audited OPLS boxes.

Four independent replicas per box, each started from the SAME production seed
frame (`production_ready/<box>/seed.gro`, the NPT-equilibrium frame chosen by
prepare_production.py) but with a different Maxwell-Boltzmann velocity draw.
The box is therefore identical across replicas -- NVT holds the volume that the
NPT stage equilibrated, which is what makes the V/(kB T) Green-Kubo prefactor
well defined -- and only the phase-space trajectory differs.

Per replica, a single stage:

    prod   NVT   20 ns at T   fresh velocities at T, tau_t = 10 ps (weak)

The four replicas are distinguished by nothing but their `gen-seed`, which sets
the Maxwell-Boltzmann draw: seeds 1007 / 2007 / 3007 / 4007, fixed rather than
`-1` so a replica is reproducible.

Note that the run therefore starts on an uncorrelated velocity field, exactly
as launch_transport.sh does. Positions come from the NPT-equilibrium seed so
there is no structural relaxation, but the first stretch of the stress log is
still a thermalisation transient rather than an equilibrium sample -- discard a
lead-in when integrating the Green-Kubo ACF.

Ensemble/thermostat follow electrolytes/src/simulate/launch_transport.sh:
NVT with a *weak global* velocity-rescale (v-rescale is GROMACS' name for
Bussi/CSVR; GAMMA_THZ=0.1 there is tau = 10 ps here). Weak matters twice over:
tau = 10 ps is short against the ns-scale energy drift so the run holds a real
298 K, and long against the sub-ps stress ACF that Green-Kubo integrates, so it
barely biases the transport coefficient. v-rescale rescales velocities by a
single global scalar, so unlike Langevin it adds no per-atom momentum sink that
would drag the viscosity.

Observables written, one file per replica:

    <run>.stress   the Green-Kubo stress log, every 2 fs (see edr2stress.py for
                   the column format; GROMACS has all of it in the .edr, it just
                   does not write that file itself)
    prod.part*.xtc trajectory every 1 ps, for MSD self-diffusivity and the
                   Onsager/charge-MSD conductivity

The 20 ns is run as segments (default 2 ns each) with `mdrun -noappend`. Each
segment's .edr is converted to the stress log and then DELETED: a 2 fs energy
dump is ~500 B/frame, so a single un-segmented 20 ns .edr would be ~5 GB per
replica (~360 GB over all 72) purely as an intermediate. Segmenting caps that
intermediate at ~0.5 GB per running replica and makes the run resumable at
segment granularity for free.

Input:  <CM_PROD_ROOT>/<box>/{seed.gro,<box>.top,*.itp}
Output: <CM_REP_ROOT>/<box>/r<k>/prod.* + <box>_r<k>.stress + rep_meta.json

Idempotent / resumable: finished segments are skipped, an interrupted mdrun
resumes from prod.cpt. Re-run to continue; never delete outputs to "retry".

Usage:
    source config.sh
    python run_production.py --dry-run          # build replica dirs + mdps only
    python run_production.py --gpus 0 --per-gpu 6
    python run_production.py --shard 0/4        # split 72 runs over 4 SLURM jobs
    python run_production.py --sys lipf6
"""

import os
import re
import glob
import json
import shutil
import argparse
import subprocess
from threading import Thread

from lib import edr2stress, provenance

GMXBIN = os.environ.get("GMXBIN", "gmx")

_HERE = os.path.dirname(os.path.abspath(__file__))
PROD_ROOT = os.environ.get(
    "CM_PROD_ROOT", os.path.join(_HERE, "runs", "production_ready"))
REP_ROOT = os.environ.get(
    "CM_REP_ROOT", os.path.join(_HERE, "runs", "production_replicas"))

# Protocol defaults -- dt / constraints match the equilibration stage, so the
# production Hamiltonian is the one the seed density was equilibrated under.
DT_PS = 0.001                                            # 1 fs
CONSTRAINTS = "none"
NREPLICAS = int(os.environ.get("CM_NREPLICAS", 4))
PROD_NS = float(os.environ.get("CM_PROD_NS", 20))
SEG_NS = float(os.environ.get("CM_SEG_NS", 2))
STRESS_INTERVAL = int(os.environ.get("CM_STRESS_INTERVAL", 2))   # steps -> 2 fs
TRAJ_INTERVAL = int(os.environ.get("CM_TRAJ_INTERVAL", 1000))    # steps -> 1 ps
TAU_T_PROD = float(os.environ.get("CM_TAU_T", 10.0))             # ps, weak (0.1 THz)

# ----------------------------------------------------------------------------
# MDP templates
# ----------------------------------------------------------------------------
# Production. nstcalcenergy = nstenergy = STRESS_INTERVAL is what makes the
# .edr carry a full pressure tensor every 2 fs; without nstcalcenergy GROMACS
# would only evaluate the virial on its own (much coarser) schedule and the
# .edr would repeat stale values.
#
# DispCorr = EnerPres is kept for continuity with equilibration. It adds an
# isotropic term, so it shifts Pxx/Pyy/Pzz but leaves the off-diagonals -- the
# only components the shear-viscosity Green-Kubo integral uses -- untouched.
PROD_MDP = """\
; Stage: prod -- NVT transport production, {ns} ns at {temp} K (replica {rep}, seed {seed})
integrator = md
nsteps     = {nsteps}
dt         = {dt}

; weak global Bussi/CSVR: tau = {tau} ps, i.e. gamma = {gamma:.3g} THz
tcoupl  = v-rescale
tc-grps = System
tau_t   = {tau}
ref_t   = {temp}

pcoupl = no

constraints          = {constraints}
constraint-algorithm = LINCS
lincs-iter           = 1
lincs-order          = 4

cutoff-scheme = Verlet
nstlist       = 20
; 1e-3 (GROMACS default 5e-3): with the weak tau_t = 10 ps thermostat the default
; buffer lets the pair-list drift bias T by up to 0.5% (grompp warns); 1e-3 keeps it < 0.1%
verlet-buffer-tolerance = 1e-3
pbc           = xyz
rlist         = 1.2
coulombtype   = PME
rcoulomb      = 1.2
vdwtype       = Cut-off
rvdw          = 1.2
DispCorr      = EnerPres

; stress tensor every {stress} step(s) -> the Green-Kubo log
nstcalcenergy      = {stress}
nstenergy          = {stress}
; trajectory every {traj} step(s) -> MSD diffusivity + Onsager conductivity
nstxout-compressed = {traj}
compressed-x-grps  = System
nstlog             = 50000
nstxout            = 0
nstvout            = 0

; fresh Maxwell-Boltzmann draw at T -- the ONLY thing that distinguishes the
; four replicas, which share the seed frame and therefore the cell
gen-vel      = yes
gen-temp     = {temp}
gen-seed     = {seed}
continuation = no
"""


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def get_temperature(box):
    """Target temperature (K) from the box name, e.g. lipf6_dme_0_5M_273."""
    m = re.search(r"_(\d{3})$", box)
    if m:
        return float(m.group(1))
    meta = os.path.join(PROD_ROOT, box, "seed_meta.json")
    if os.path.exists(meta):
        t = json.load(open(meta)).get("temperature_K")
        if t:
            return float(t)
    return 298.0


def gro_volume_nm3(gro):
    """Cell volume from the last line of a .gro.

    NVT keeps this fixed for the whole run and GROMACS therefore writes no
    Volume term to the .edr, so it has to be handed to the stress converter.
    """
    with open(gro) as f:
        v = [float(x) for x in f.read().rstrip("\n").splitlines()[-1].split()]
    return v[0] * v[1] * v[2]


def is_tegdme(name):
    n = name.lower()
    return "tegdme" in n or "tgdme" in n


def md_gpu_flags(name, ntomp):
    # Same choices as the equilibration driver: no -update gpu, and PME stays
    # on the CPU for TEGDME where the GPU PME grid misbehaves on this machine.
    common = ["-ntmpi", "1", "-ntomp", str(ntomp), "-pin", "off", "-notunepme"]
    # CM_TEGDME_PME=gpu lifts the TEGDME workaround on machines where GPU PME is
    # fine for it (verified on Perlmutter A100 / GROMACS 2025.3 CUDA). Where PME
    # runs is a performance setting only; it does not change the physics.
    if is_tegdme(name) and os.environ.get("CM_TEGDME_PME", "cpu") != "gpu":
        return ["-nb", "gpu", "-pme", "cpu", "-update", "cpu"] + common
    return ["-nb", "gpu", "-pme", "gpu", "-update", "cpu"] + common


def discover_seeds():
    """[(box, topfile)] for every production_ready box with a seed + topology."""
    out = []
    for d in sorted(glob.glob(os.path.join(PROD_ROOT, "*"))):
        if not os.path.isdir(d):
            continue
        box = os.path.basename(d)
        top = os.path.join(d, f"{box}.top")
        if os.path.exists(top) and os.path.exists(os.path.join(d, "seed.gro")):
            out.append((box, top))
    return out


def setup_rep_dir(rep_dir, box):
    """Copy the seed + topology in, so each replica is a self-contained run."""
    os.makedirs(rep_dir, exist_ok=True)
    src = os.path.join(PROD_ROOT, box)
    for f in (glob.glob(f"{src}/*.itp") + glob.glob(f"{src}/*.top")
              + [f"{src}/seed.gro", f"{src}/seed_meta.json"]):
        dst = os.path.join(rep_dir, os.path.basename(f))
        if not os.path.exists(dst):
            shutil.copy2(f, dst)


def run(cmd, cwd, logname, env=None):
    with open(os.path.join(cwd, logname), "a") as lf:
        lf.write("\n# " + " ".join(cmd) + "\n")
        lf.flush()
        r = subprocess.run(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    return r.returncode == 0


def state_path(rep_dir):
    return os.path.join(rep_dir, "prod_state.json")


def load_state(rep_dir):
    p = state_path(rep_dir)
    return json.load(open(p)) if os.path.exists(p) else {"segments_done": 0,
                                                         "stress_frames": 0}


def save_state(rep_dir, st):
    json.dump(st, open(state_path(rep_dir), "w"), indent=2)


# ----------------------------------------------------------------------------
# One replica
# ----------------------------------------------------------------------------
def run_replica(box, top_name, rep, gpu, args):
    temp = get_temperature(box)
    rep_dir = os.path.join(REP_ROOT, box, f"r{rep}")
    tag = f"[{box} r{rep}]"
    seed = 1000 * (rep + 1) + 7        # deterministic, distinct per replica

    prod_nsteps = int(round(args.prod_ns * 1000 / args.dt))
    seg_nsteps = int(round(args.seg_ns * 1000 / args.dt))
    nseg = -(-prod_nsteps // seg_nsteps)          # ceil

    setup_rep_dir(rep_dir, box)
    top = os.path.join(rep_dir, os.path.basename(top_name))

    prod_mdp = PROD_MDP.format(ns=args.prod_ns, temp=temp, rep=rep, seed=seed,
                               nsteps=prod_nsteps, dt=args.dt,
                               constraints=args.constraints,
                               tau=args.tau_t, gamma=1.0 / args.tau_t,
                               stress=args.stress_interval, traj=args.traj_interval)
    mdp_path = os.path.join(rep_dir, "prod.mdp")
    if not os.path.exists(mdp_path):
        open(mdp_path, "w").write(prod_mdp)

    json.dump({
        "box": box, "replica": rep, "gen_seed": seed,
        "temperature_K": temp, "ensemble": "NVT",
        "thermostat": "v-rescale (Bussi/CSVR), global",
        "tau_t_ps": args.tau_t, "dt_ps": args.dt,
        "constraints": args.constraints,
        "start_from": "fresh Maxwell-Boltzmann velocities at T (gen-vel = yes)",
        "prod_ns": args.prod_ns,
        "segment_ns": args.seg_ns, "n_segments": nseg,
        "stress_interval_steps": args.stress_interval,
        "stress_interval_fs": args.stress_interval * args.dt * 1000,
        "traj_interval_steps": args.traj_interval,
        "stress_log": f"{box}_r{rep}.stress",
        "seed_frame": os.path.join(PROD_ROOT, box, "seed.gro"),
        "verlet_buffer_tolerance": 1e-3,
        "provenance": provenance.record(GMXBIN, rep_dir),
    }, open(os.path.join(rep_dir, "rep_meta.json"), "w"), indent=2)

    if args.dry_run:
        print(f"  {tag}: dry-run -- dir + mdps written ({nseg} segment(s))")
        return

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}

    # Straight from the seed frame: grompp draws the velocities itself from
    # gen-seed, so there is no -t checkpoint and no preceding stage.
    tpr = os.path.join(rep_dir, "prod.tpr")
    if not os.path.exists(tpr):
        # -maxwarn 0: any grompp WARNING aborts; fix the cause, do not raise maxwarn.
        if not run([GMXBIN, "grompp", "-f", "prod.mdp", "-c", "seed.gro",
                    "-p", top, "-o", "prod.tpr", "-maxwarn", "0"],
                   rep_dir, "prod.log"):
            print(f"  {tag}: prod grompp FAILED"); return

    stress = os.path.join(rep_dir, f"{box}_r{rep}.stress")
    volume = gro_volume_nm3(os.path.join(rep_dir, "seed.gro"))
    st = load_state(rep_dir)

    total_ps = args.prod_ns * 1000
    for seg in range(st["segments_done"], nseg):
        # -nsteps with -cpi is *additional* steps, not a cumulative target, so
        # each segment asks for exactly one segment's worth. The loop is driven
        # by the time actually reached (below) rather than by this count, so a
        # short or restarted segment cannot silently truncate the run.
        cmd = ([GMXBIN, "mdrun", "-deffnm", "prod", "-s", "prod.tpr",
                "-noappend", "-nsteps", str(seg_nsteps)]
               + md_gpu_flags(box, args.ntomp))
        if os.path.exists(os.path.join(rep_dir, "prod.cpt")):
            cmd += ["-cpi", "prod.cpt"]
        print(f"  {tag}: prod segment {seg + 1}/{nseg} (gpu {gpu})...")
        if not run(cmd, rep_dir, "prod_run.log", env=env):
            print(f"  {tag}: prod segment {seg + 1} mdrun FAILED"); return

        # convert this segment's energy file, then drop it -- see module docstring
        parts = sorted(glob.glob(os.path.join(rep_dir, "prod.part*.edr")))
        if not parts:
            print(f"  {tag}: no .edr produced by segment {seg + 1}"); return
        for edr in parts:
            n, last_t = edr2stress.convert(
                edr, stress, interval=args.stress_interval,
                dt_fs=args.dt * 1000, append=st["stress_frames"] > 0,
                volume_nm3=volume, t_min=st.get("last_time_ps"))
            st["stress_frames"] += n
            if last_t is not None:
                st["last_time_ps"] = last_t
            if not args.keep_edr:
                os.remove(edr)
        st["segments_done"] = seg + 1
        save_state(rep_dir, st)
        print(f"  {tag}: segment {seg + 1}/{nseg} done, "
              f"{st['stress_frames']} stress frame(s) to t = {st.get('last_time_ps')} ps")
        if st.get("last_time_ps", 0) >= total_ps - 0.5 * args.dt:
            break

    reached = st.get("last_time_ps", 0)
    if reached < total_ps - 0.5 * args.dt:
        print(f"  {tag}: INCOMPLETE -- reached {reached} of {total_ps} ps; re-run to continue")
        return

    print(f"  {tag}: COMPLETE -- {os.path.basename(stress)} "
          f"({st['stress_frames']} frames)")


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpus", default=os.environ.get("CUDA_DEVICE", "0"))
    p.add_argument("--sys", nargs="*", help="only boxes matching one of these substrings")
    p.add_argument("--replicas", type=int, default=NREPLICAS)
    p.add_argument("--prod-ns", type=float, default=PROD_NS)
    p.add_argument("--seg-ns", type=float, default=SEG_NS,
                   help=f"segment length, ns (default {SEG_NS}); caps the "
                        "intermediate .edr size and the resume granularity")
    p.add_argument("--dt", type=float, default=DT_PS)
    p.add_argument("--constraints", default=CONSTRAINTS,
                   choices=["none", "h-bonds", "all-bonds"])
    p.add_argument("--tau-t", type=float, default=TAU_T_PROD)
    p.add_argument("--stress-interval", type=int, default=STRESS_INTERVAL,
                   help=f"stress dump interval in steps (default {STRESS_INTERVAL} = 2 fs)")
    p.add_argument("--traj-interval", type=int, default=TRAJ_INTERVAL)
    p.add_argument("--keep-edr", action="store_true",
                   help="keep the per-segment .edr after conversion (large)")
    p.add_argument("--per-gpu", type=int, default=6)
    p.add_argument("--ntomp", type=int, default=4)
    p.add_argument("--shard", default=None, metavar="I/N")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not os.path.isdir(PROD_ROOT):
        raise SystemExit(f"seed root not found: {PROD_ROOT} (set CM_PROD_ROOT)")

    seeds = discover_seeds()
    if args.sys:
        seeds = [s for s in seeds if any(x in s[0] for x in args.sys)]
    jobs = [(box, top, rep) for box, top in seeds for rep in range(args.replicas)]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        if not 0 <= i < n:
            raise SystemExit(f"bad --shard {args.shard}: need 0 <= I < N")
        jobs = jobs[i::n]

    print(f"Seeds:  {PROD_ROOT}\nOutput: {REP_ROOT}")
    print(f"Protocol: prod({args.prod_ns} ns NVT, v-rescale tau_t={args.tau_t} ps) "
          f"from fresh velocities at T, dt={args.dt} ps, "
          f"constraints={args.constraints}")
    print(f"Stress every {args.stress_interval} step(s) = "
          f"{args.stress_interval * args.dt * 1000:g} fs; "
          f"traj every {args.traj_interval} step(s) = "
          f"{args.traj_interval * args.dt:g} ps")
    print(f"{len(seeds)} box(es) x {args.replicas} replica(s) = {len(jobs)} run(s)"
          + (f", shard {args.shard}" if args.shard else ""))

    n_frames = args.prod_ns * 1000 / (args.stress_interval * args.dt)
    print(f"Projected: {n_frames / 1e6:.1f}M stress frames/run, "
          f"~{n_frames * 112 / 1e9:.2f} GB/run, "
          f"~{len(seeds) * args.replicas * n_frames * 112 / 1e9:.0f} GB total")
    if not jobs:
        return

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    slots = [(g, k) for g in gpus for k in range(args.per_gpu)]
    print(f"{len(slots)} slot(s): {len(gpus)} GPU(s) x {args.per_gpu}, "
          f"{args.ntomp} OpenMP thread(s) each")

    def worker(my_jobs, gpu):
        for box, top, rep in my_jobs:
            run_replica(box, top, rep, gpu, args)

    threads = []
    for si, (gpu, _) in enumerate(slots):
        mine = jobs[si::len(slots)]
        if not mine:
            continue
        t = Thread(target=worker, args=(mine, gpu))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    print("\nAll done.")


if __name__ == "__main__":
    main()
