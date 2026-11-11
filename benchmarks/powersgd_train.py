"""Real training with PyTorch's PowerSGD DDP hook: does the orthogonalization routine change the training result?

Pre-registration: benchmarks/prereg/PREREG_POWERSGD_TRAIN_2026-11-12.md (predictions and judge in this file, committed
before any run). CPU, torch.distributed with gloo, 2 processes. CIFAR-10 (binary version, read as raw bytes), a small
ResNet (3 stages x 2 basic blocks, widths 32/64/128), SGD with Nesterov momentum.

Arms (only the orthogonalization of the PowerSGD factor P differs; same seeds, data order, init and PowerSGD seed):
  torch      PyTorch's own powerSGD_hook._orthogonalize (FP32 Householder QR for rank > 2)
  contrib    cholqr.contrib.powersgd.patch() (batches: FP32 CholeskyQR2 with a 1e-3 check + Householder fallback;
             single matrices: FP32 Householder)
  verify     cholesky_qr(X, verify=True) (default scqr3, FP64 certificate tol 1e-4, default fallback) for every batch
  inj1e-3    injected orthogonality error: Q = U (I + F)^{1/2}, U = FP64 Householder basis of P, F random symmetric with
  inj1e-2    ||F||_F = delta (1e-3, 1e-2) per matrix; range unchanged (construction of the note's Sec. 6.11, "random F").
             F is drawn from a generator seeded by (seed, call index), so all ranks apply the same F.

Usage:
  python benchmarks/powersgd_train.py --data DIR --arm torch --seed 0 --out results.jsonl
  python benchmarks/powersgd_train.py --data DIR --smoke              (timings only; writes nothing)
  python benchmarks/powersgd_train.py --judge results.jsonl           (registered judge)
DIR contains cifar-10-batches-bin/ (https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz).
"""
import argparse, json, math, os, socket, statistics, sys, time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as Fn

ARMS = ("torch", "contrib", "verify", "inj1e-3", "inj1e-2")
STEPS, BATCH, WORLD, THREADS, RANK_R, START_PSGD = 500, 64, 2, 4, 4, 10
LR, MOM, WD, WARM = 0.05, 0.9, 5e-4, 50


# ----------------------------------------------------------------------------------------------------------- data
def load_cifar(d):
    """CIFAR-10 binary version: 1 label byte + 3072 pixel bytes per record. No pickle."""
    def rd(names):
        raw = b"".join(open(os.path.join(d, "cifar-10-batches-bin", n), "rb").read() for n in names)
        t = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(-1, 3073)
        return t[:, 1:].reshape(-1, 3, 32, 32).clone(), t[:, 0].long().clone()
    xtr, ytr = rd([f"data_batch_{i}.bin" for i in range(1, 6)])
    xte, yte = rd(["test_batch.bin"])
    assert xtr.shape[0] == 50000 and xte.shape[0] == 10000
    mean = xtr.float().mean((0, 2, 3), keepdim=True) / 255; std = xtr.float().std((0, 2, 3), keepdim=True) / 255
    norm = lambda x: (x.float() / 255 - mean) / std
    return norm(xtr), ytr, norm(xte), yte


def batch_indices(seed, step, rank):
    """Global batch of WORLD*BATCH examples from a per-epoch permutation (seeded); rank takes its shard."""
    per_epoch = 50000 // (WORLD * BATCH)
    ep, k = divmod(step, per_epoch)
    perm = torch.randperm(50000, generator=torch.Generator().manual_seed(1000 * seed + ep))
    g = perm[k * WORLD * BATCH:(k + 1) * WORLD * BATCH]
    return g[rank * BATCH:(rank + 1) * BATCH]


def augment(x, seed, step, rank):
    g = torch.Generator().manual_seed(10 ** 6 * (seed + 1) + 10 * step + rank)
    flip = torch.rand(x.shape[0], generator=g) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    xp = Fn.pad(x, (4, 4, 4, 4), mode="reflect")
    dx = torch.randint(0, 9, (x.shape[0],), generator=g); dy = torch.randint(0, 9, (x.shape[0],), generator=g)
    return torch.stack([xp[i, :, dy[i]:dy[i] + 32, dx[i]:dx[i] + 32] for i in range(x.shape[0])])


# ---------------------------------------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.sc = None if (stride == 1 and cin == cout) else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                                                           nn.BatchNorm2d(cout))

    def forward(self, x):
        y = Fn.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return Fn.relu(y + (x if self.sc is None else self.sc(x)))


def make_model():
    w = (32, 64, 128)
    layers = [nn.Conv2d(3, w[0], 3, 1, 1, bias=False), nn.BatchNorm2d(w[0]), nn.ReLU()]
    cin = w[0]
    for i, c in enumerate(w):
        layers += [Block(cin, c, 1 if i == 0 else 2), Block(c, c, 1)]; cin = c
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(cin, 10)]
    return nn.Sequential(*layers)


# ------------------------------------------------------------------------------------------- orthogonalizations
def orth_err64(Q):
    Q = Q.double()
    return (Q.mT @ Q - torch.eye(Q.shape[-1], dtype=torch.float64)).norm(dim=(-2, -1))


def make_orth(arm, seed, log):
    from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as psh
    from cholqr.contrib import powersgd
    import cholqr
    original = powersgd._ORIGINAL
    calls = [0]

    if arm == "torch":
        inner = original
    elif arm == "contrib":
        inner = powersgd.orthogonalize_
    elif arm == "verify":
        def inner(M, epsilon=0):
            M.copy_(cholqr.cholesky_qr(M.float(), verify=True))
            return M
    else:
        delta = float(arm[3:])

        def inner(M, epsilon=0):
            U = torch.linalg.qr(M.double(), mode="reduced")[0]
            b, _, r = U.shape
            g = torch.Generator().manual_seed(10 ** 7 * (seed + 1) + calls[0])
            S = torch.randn(b, r, r, dtype=torch.float64, generator=g); S = S + S.mT
            F = delta * S / S.norm(dim=(-2, -1), keepdim=True)
            ev, V = torch.linalg.eigh(torch.eye(r, dtype=torch.float64) + F)
            M.copy_(U @ (V * ev.sqrt()[..., None, :]) @ V.mT)
            return M

    def wrapped(M, epsilon=0):
        t0 = time.perf_counter()
        inner(M, epsilon)
        dt = time.perf_counter() - t0
        calls[0] += 1
        log["orth_time"] += dt; log["orth_calls"] += 1
        e = orth_err64(M)
        log["orth_max"] = max(log["orth_max"], float(e.max()))
        log["orth_sum"] += float(e.sum()); log["orth_n"] += e.numel()
        return M

    psh._orthogonalize = wrapped
    return original


# ------------------------------------------------------------------------------------------------------- training
def lr_at(step):
    return LR * (step + 1) / WARM if step < WARM else LR * max(0.0, (STEPS - step) / (STEPS - WARM))


def worker(rank, a, port, q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = str(port)
    torch.set_num_threads(THREADS)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as psh
    import cholqr
    xtr, ytr, xte, yte = load_cifar(a.data)
    log = dict(orth_time=0.0, orth_calls=0, orth_max=0.0, orth_sum=0.0, orth_n=0)
    make_orth(a.arm, a.seed, log)
    torch.manual_seed(a.seed)
    model = DDP(make_model())
    st = psh.PowerSGDState(process_group=None, matrix_approximation_rank=RANK_R, start_powerSGD_iter=START_PSGD,
                           min_compression_rate=2, use_error_feedback=True, warm_start=True,
                           batch_tensors_with_same_shape=True, random_seed=a.seed)
    model.register_comm_hook(st, psh.powerSGD_hook)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOM, nesterov=True, weight_decay=WD)
    steps = a.steps
    losses, times = [], []
    for step in range(steps):
        idx = batch_indices(a.seed, step, rank)
        x = augment(xtr[idx], a.seed, step, rank); y = ytr[idx]
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step)
        dist.barrier()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = Fn.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        times.append(time.perf_counter() - t0)
        lt = loss.detach().clone(); dist.all_reduce(lt); losses.append(float(lt) / WORLD)
    # all ranks must hold identical weights (the injected F and any fallback must be rank-consistent)
    flat = torch.cat([p.detach().reshape(-1) for p in model.module.parameters()])
    mx, mn = flat.clone(), flat.clone()
    dist.all_reduce(mx, op=dist.ReduceOp.MAX); dist.all_reduce(mn, op=dist.ReduceOp.MIN)
    rank_dev = float((mx - mn).abs().max())
    if rank == 0:
        m = model.module.eval(); tl, corr = 0.0, 0
        with torch.no_grad():
            for i in range(0, 10000, 500):
                out = m(xte[i:i + 500]); tl += float(Fn.cross_entropy(out, yte[i:i + 500], reduction="sum"))
                corr += int((out.argmax(1) == yte[i:i + 500]).sum())
        q.put(dict(arm=a.arm, seed=a.seed, rep=a.rep, steps=steps, losses=losses,
                   final_train_loss=statistics.fmean(losses[-50:]), test_loss=tl / 10000, test_acc=corr / 10000,
                   step_time_median=statistics.median(times[START_PSGD + 5:]),
                   orth_time_per_step=log["orth_time"] / steps, orth_calls=log["orth_calls"],
                   orth_max=log["orth_max"], orth_mean=log["orth_sum"] / max(1, log["orth_n"]),
                   rank_param_maxdev=rank_dev, param_checksum=float(flat.double().abs().sum()),
                   torch=torch.__version__, cholqr=cholqr.__version__, world=WORLD, threads=THREADS))
    dist.destroy_process_group()


def run(a):
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0)); port = sk.getsockname()[1]
    ctx = mp.get_context("spawn"); q = ctx.Queue()
    mp.start_processes(worker, args=(a, port, q), nprocs=WORLD, start_method="spawn", join=True)
    return q.get()


# ---------------------------------------------------------------------------------------------------------- judge
def judge(path):
    R = [json.loads(l) for l in open(path)]
    by = {}
    for r in R:
        by.setdefault((r["arm"], r["rep"]), {})[r["seed"]] = r
    base = by[("torch", 0)]; seeds = sorted(base)
    res = []

    def chk(name, ok, detail):
        res.append(ok); print(f"{name}: {'PASS' if ok else 'FAIL'}  {detail}")

    S = {k: statistics.stdev([base[s][k] for s in seeds]) for k in ("test_acc", "final_train_loss")}
    print(f"seed-to-seed sd of arm torch over seeds {seeds}: test_acc {S['test_acc']:.4f}, "
          f"final_train_loss {S['final_train_loss']:.4f}")
    for arm in ARMS[1:]:
        run_ = by[(arm, 0)]
        for k in ("test_acc", "final_train_loss"):
            d = [run_[s][k] - base[s][k] for s in seeds]
            md = statistics.fmean(d)
            print(f"  {arm:8s} {k:16s} paired diffs {['%+.4f' % x for x in d]} mean {md:+.4f} "
                  f"(|mean|/sd {abs(md) / S[k]:.2f}, max|d|/sd {max(map(abs, d)) / S[k]:.2f})")
    # P0 determinism
    if ("torch", 1) in by:
        rep = by[("torch", 1)][0]
        chk("P0 rerun of (torch, seed 0) bit-identical loss curve", rep["losses"] == base[0]["losses"],
            f"max |dloss| {max(abs(x - y) for x, y in zip(rep['losses'], base[0]['losses'])):.3g}")
    # P1-P4 within seed noise
    for tag, arm in (("P1", "contrib"), ("P2", "verify"), ("P3", "inj1e-3"), ("P4", "inj1e-2")):
        ok, det = True, []
        for k in ("test_acc", "final_train_loss"):
            d = [by[(arm, 0)][s][k] - base[s][k] for s in seeds]
            ok &= abs(statistics.fmean(d)) <= S[k] and max(map(abs, d)) <= 3 * S[k]
            det.append(f"{k}: |mean d| {abs(statistics.fmean(d)):.4f} vs sd {S[k]:.4f}, max|d| {max(map(abs, d)):.4f}")
        chk(f"{tag} {arm} within seed noise of torch", ok, "; ".join(det))
    # P5 trajectory: paired |dloss| over last 100 steps smaller than across-seed |dloss|
    def tail_absdiff(a_, b_):
        return statistics.fmean(abs(x - y) for x, y in zip(a_["losses"][-100:], b_["losses"][-100:]))
    cross = statistics.fmean(tail_absdiff(base[s], base[t]) for s in seeds for t in seeds if s < t)
    for arm in ARMS[1:]:
        p = statistics.fmean(tail_absdiff(by[(arm, 0)][s], base[s]) for s in seeds)
        chk(f"P5 {arm} paired loss-curve deviation (last 100 steps) < across-seed deviation", p < cross,
            f"paired {p:.4f} vs across-seed {cross:.4f} (ratio {p / cross:.2f})")
    # P6 orthogonality actually delivered
    o = {arm: max(by[(arm, 0)][s]["orth_max"] for s in seeds) for arm in ARMS}
    chk("P6a torch/contrib/verify max orthogonality <= 1e-4", max(o["torch"], o["contrib"], o["verify"]) <= 1e-4,
        f"torch {o['torch']:.2e}, contrib {o['contrib']:.2e}, verify {o['verify']:.2e}")
    chk("P6b injected max orthogonality within 1% of delta",
        all(abs(o[f"inj{d}"] / float(d) - 1) <= 0.01 for d in ("1e-3", "1e-2")),
        f"inj1e-3 {o['inj1e-3']:.4e}, inj1e-2 {o['inj1e-2']:.4e}")
    # P7 ranks consistent
    dev = max(r["rank_param_maxdev"] for r in R)
    chk("P7 identical weights on all ranks in every run", dev == 0.0, f"max deviation {dev:.3g}")
    # P8 step time
    for arm in ("contrib", "verify"):
        t = [by[(arm, 0)][s]["step_time_median"] / base[s]["step_time_median"] for s in seeds]
        chk(f"P8 {arm} median step time within 15% of torch", all(abs(x - 1) <= 0.15 for x in t),
            f"ratios {['%.3f' % x for x in t]}")
    print(f"\n{sum(res)} PASS, {len(res) - sum(res)} FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--arm", choices=ARMS, default="torch")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--out"); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--judge")
    ap.add_argument("--steps", type=int, default=STEPS)
    a = ap.parse_args()
    if a.judge:
        return judge(a.judge)
    if a.smoke:                       # timings only: no losses, accuracies or orthogonality are printed or written
        for arm in ARMS:
            a.arm, a.steps = arm, 30
            t0 = time.time(); r = run(a)
            print(f"smoke {arm}: {time.time() - t0:.1f}s wall for 30 steps, median step {r['step_time_median']:.3f}s, "
                  f"orth time/step {r['orth_time_per_step'] * 1e3:.2f} ms")
        return
    r = run(a)
    with open(a.out, "a") as f:
        f.write(json.dumps(r) + "\n")
    print(f"{a.arm} seed {a.seed} rep {a.rep}: done", file=sys.stderr)


if __name__ == "__main__":
    main()
