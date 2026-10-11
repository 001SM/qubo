
---

### A.1 一元松弛 / 二元松弛 + 严格下三角装配 → QUBO

源码：`tests\miplib_qubo\qubo_build.py`

```python
"""
MIPLIB 实例 → 完整 QUBO/Ising 模型（一元松弛 / 二元松弛 + λ 罚项装配）
=====================================================================
流程：
  1) 解析 MPS（复用 miplib_sa_test.parse_mps / Instance：固定变量折叠、整数展开、
     行归一化，得到全 0/1 变量空间）
  2) 每条约束加松弛变量：
       L 行 (lhs ≤ ub):  lhs + s = ub,  s ∈ [0, S],  S = ub − Σ_j min(0, a_rj)
       G 行 (lhs ≥ lb):  lhs − s = lb,  s ∈ [0, S],  S = Σ_j max(0, a_rj) − lb
       E 行:              lhs = rhs（无松弛）
       R 行:              拆成两条不等式
     松弛编码二选一：
       一元(unary):  s = Σ_{k=1..S} y_k                    （S 个辅助变量，系数全 1）
       二元(binary): s = Σ_k w_k y_k,  w=[1,2,4,...,余量]   （⌈log2(S+1)⌉ 个）
  3) 罚项装配（下三角/上三角规则，利用 z² = z）：
       λ(Σ a_i z_i + d)²  =  λ[ Σ_i a_i² z_i + 2Σ_{i<j} a_i a_j z_i z_j
                               + 2d Σ_i a_i z_i + d² ]
       → 线性项 h_i += λ(a_i² + 2 d a_i)
       → 上三角 Q_ij += 2λ a_i a_j  (i<j)
       → 常数项  += λ d²
  4) 输出 (h, Q, const)，交给迭代消元全局退火（sa_iterative_v4/v5）求解

罚权：λ_r = 2·max(1,cmax) / viol_unit_r²，viol_unit_r = min(min_j|a_rj|, 1)
      （只需 > Δmax_obj/viol_unit²；过大反而冻结搜索）
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, r"D:\玻色\下三角算法\tests\miplib_hard")
import miplib_sa_test as core

def build_qubo(inst: core.Instance, slack: str = "unary",
               lam_scale: float = 1.0, lam_cap_mult: float = 1e4):
    n_orig = inst.n
    cmax = max(1.0, float(inst.cmax))
    const = float(inst.obj_const)
    ri, ci, av = inst.ri_r, inst.ci_r, inst.av_r
    row_slice = np.searchsorted(ri, np.arange(inst.m))
    row_end = np.searchsorted(ri, np.arange(inst.m), side="right")

    slack_rows = []
    nxt = n_orig
    n_slack = 0
    for r in range(inst.m):
        s, e = row_slice[r], row_end[r]
        if e <= s:
            continue
        cols = ci[s:e]
        vals = av[s:e]
        lo, hi = inst.lb[r], inst.ub[r]
        lo_f, hi_f = np.isfinite(lo), np.isfinite(hi)
        viol_unit = min(float(np.abs(vals).min()), 1.0)
        lam = min(lam_scale * 2.0 * cmax / max(viol_unit ** 2, 1e-18),
                  lam_cap_mult * cmax)

        def add_slack(bound, sign):
            nonlocal nxt, n_slack
            coefs = [(int(c), float(v)) for c, v in zip(cols, vals)]
            if sign > 0:
                S = bound - float(np.minimum(vals, 0).sum())
            else:
                S = float(np.maximum(vals, 0).sum()) - bound
            S = int(np.floor(S + 1e-9))
            if S < 0:
                return None
            if slack == "unary":
                weights = [1] * S
            else:
                if S == 0:
                    weights = []
                else:
                    bits = max(1, int(np.ceil(np.log2(S + 1))))
                    weights = [2 ** k for k in range(bits - 1)]
                    weights.append(S - (2 ** (bits - 1) - 1))
            for w in weights:
                coefs.append((nxt, float(sign * w)))
                nxt += 1
                n_slack += 1
            return coefs, -float(bound)

        if lo_f and hi_f and abs(hi - lo) < 1e-12:
            coefs = [(int(c), float(v)) for c, v in zip(cols, vals)]
            slack_rows.append((r, coefs, -float(hi), lam))
        else:
            if hi_f:
                got = add_slack(hi, +1)
                if got:
                    slack_rows.append((r, got[0], got[1], lam))
            if lo_f:
                got = add_slack(lo, -1)
                if got:
                    slack_rows.append((r, got[0], got[1], lam))

    n_total = nxt
    h = np.zeros(n_total, dtype=np.float64)
    h[:n_orig] = inst.c
    from scipy.sparse import coo_matrix
    rows_i, cols_j, vals_q = [], [], []
    for _, coefs, d, lam in slack_rows:
        idx = np.array([c[0] for c in coefs], dtype=np.int64)
        aa = np.array([c[1] for c in coefs], dtype=np.float64)
        h[idx] += lam * (aa * aa + 2.0 * d * aa)
        k = idx.size
        if k < 2:
            const += lam * d * d
            continue
        ii, jj = np.triu_indices(k, k=1)
        pi, pj = idx[ii], idx[jj]
        lo_ = np.minimum(pi, pj)
        hi_ = np.maximum(pi, pj)
        rows_i.append(lo_)
        cols_j.append(hi_)
        vals_q.append(2.0 * lam * aa[ii] * aa[jj])
        const += lam * d * d
    if rows_i:
        ri_cat = np.concatenate(rows_i)
        ci_cat = np.concatenate(cols_j)
        vv_cat = np.concatenate(vals_q)
        Q = coo_matrix((vv_cat, (ri_cat, ci_cat)),
                       shape=(n_total, n_total)).tocsr()
        Q.sum_duplicates()
    else:
        Q = coo_matrix((n_total, n_total)).tocsr()
    return dict(h=h, Q=Q, const=const, n_orig=n_orig, n_total=n_total,
                n_slack=n_slack, n_cons=len(slack_rows), slack=slack,
                nnz=int(Q.nnz))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=r"D:\玻色\下三角算法\tests\miplib_qubo")
    ap.add_argument("--instances", required=True)
    ap.add_argument("--slack", default="both", choices=["unary", "binary", "both"])
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    for name in [x.strip() for x in a.instances.split(",") if x.strip()]:
        inst = core.Instance(name, os.path.join(a.dir, name + ".mps"))
        for mode in (["unary", "binary"] if a.slack == "both" else [a.slack]):
            q = build_qubo(inst, slack=mode)
            print("%-16s %-6s 原变量=%5d 松弛变量=%7d 约束=%6d → QUBO 规模=%6d "
                  "非零耦合=%8d" % (name, mode, q["n_orig"], q["n_slack"],
                                    q["n_cons"], q["n_total"], q["nnz"]))
            if a.save:
                np.savez_compressed(
                    os.path.join(a.dir, "%s_%s.npz" % (name, mode)),
                    h=q["h"], data=q["Q"].data, indices=q["Q"].indices,
                    indptr=q["Q"].indptr, shape=q["Q"].shape,
                    const=q["const"], n_orig=q["n_orig"], n_slack=q["n_slack"])

if __name__ == "__main__":
    main()
```

---

### A.2 混合编码：严格 e_K（超立方体下三角）+ 松弛兜底

源码：`tests\miplib_qubo\hybrid_build.py`

```python
"""
混合编码：严格 e_K（超立方体下三角）+ 松弛兜底
================================================
策略（"松弛兜底"）：
  对每条约束的每一"侧"（≤ / ≥ / =）：
    1) 先尝试严格编码：canonical（GCD 归一化 + 补变量）后用加权初等对称
       多项式 e_K 给出**精确违反指示**（无松弛变量、无编码误差）。
    2) 任一环节失败（系数非整数 / 右端非整数 / K 超限 / 项数超 cap）→
       该侧回退到松弛编码（一元 unary / 二元 binary 松弛 + 平方罚项，
       与 qubo_build.py 完全同构），保证**每行每侧必有罚项**，不再跳行。

变量布局：[0, n_orig) 原变量 | [n_orig, n_orig+n_slack) 松弛变量 | 之后 Rosenberg 辅助变量。
罚权：严格侧 lam_con = lam_scale·2·cmax；兜底侧 λ_r = lam_scale·2·cmax/viol_unit²
      （归一化系数空间，与 qubo_build 一致）。

strict 全覆盖时与 simplex_build.build_simplex_qubo 等价（同一 e_K 实现）。
"""
from __future__ import annotations

import os
import sys
import time
from heapq import heappop, heappush, heapify
from itertools import combinations
from math import comb, gcd

import numpy as np
from scipy.sparse import coo_matrix

sys.path.insert(0, r"D:\玻色\下三角算法\tests\miplib_hard")
import miplib_sa_test as core
from simplex_build import build_e_K, quadratize

def _canon_side(vals, bound, is_le, use_gcd=True):
    vals = np.asarray(vals, dtype=float)
    nz = np.abs(vals) > 1e-9
    if not nz.any():
        return []
    wa = np.abs(vals[nz])
    if np.max(np.abs(wa - np.round(wa))) > 1e-6:
        return None
    if abs(bound - round(bound)) > 1e-6:
        return None
    w = np.round(wa).astype(np.int64)
    if (w <= 0).any():
        return None
    sgn = np.where(vals[nz] > 0, 1, -1).astype(np.int8)
    B = int(round(bound))
    S = int(w.sum())
    g = 1
    if use_gcd:
        for x in w.tolist():
            g = gcd(g, x)
            if g == 1:
                break
    if is_le:
        if B < 0:
            return None
        K = B // g + 1
        return (w // g, sgn.copy(), K)
    else:
        if S - B + 1 <= 0:
            return None
        Rq = -((-B) // g)
        K = S // g - Rq + 1
        if K <= 0:
            return None
        return (w // g, (-sgn).astype(np.int8), K)

def _count_monomials(w, K, cap=100_000):
    K = int(K)
    if K <= 0:
        return 0
    dp = [0] * (K + 1)
    dp[0] = 1
    for wj in w.tolist():
        wj = int(wj)
        ndp = [0] * (K + 1)
        pref = 0
        for k in range(K + 1):
            pref += dp[k]
            lo = k - wj - 1
            if lo >= 0:
                pref -= dp[lo]
            ndp[k] = pref if pref <= cap else cap + 1
        dp = ndp
        if min(dp) > cap:
            return None
    c = dp[K]
    return None if c > cap else c

def _quadratize_fast(terms, lam_quad, n_base):
    pending = {}

    def bump(d, key, val):
        d[key] = d.get(key, 0.0) + val

    in_heap = set()
    heap = []
    for k, v in terms.items():
        if abs(v) > 1e-12:
            bump(pending, k, v)
            if len(k) > 2:
                heappush(heap, (-len(k), k))
                in_heap.add(k)
    extra = {}
    aux_defs = {}
    n_aux = 0
    while heap:
        _, key = heappop(heap)
        in_heap.discard(key)
        c = pending.pop(key, None)
        if c is None or abs(c) <= 1e-12:
            continue
        n1, n2 = key[0], key[1]
        rest = key[2:]
        wnode = (n_base + n_aux, 1)
        aux_defs[n_aux] = (n1, n2)
        n_aux += 1
        nk = tuple(sorted((wnode,) + rest))
        bump(pending, nk, c)
        if len(nk) > 2 and nk not in in_heap:
            heappush(heap, (-len(nk), nk))
            in_heap.add(nk)
        bump(extra, tuple(sorted((n1, n2))), lam_quad)
        bump(extra, tuple(sorted((n1, wnode))), -2.0 * lam_quad)
        bump(extra, tuple(sorted((n2, wnode))), -2.0 * lam_quad)
        bump(extra, (wnode,), 3.0 * lam_quad)
    for k, v in extra.items():
        bump(pending, k, v)
    return ({k: v for k, v in pending.items() if abs(v) > 1e-12}, n_aux,
            aux_defs)

def _slack_weights(S, slack):
    if S == 0:
        return []
    if slack == "unary":
        return [1] * S
    bits = max(1, int(np.ceil(np.log2(S + 1))))
    ws = [2 ** k for k in range(bits - 1)]
    ws.append(S - (2 ** (bits - 1) - 1))
    return ws

def build_hybrid_qubo(inst, mode="strict", slack="binary", lam_scale=1.0,
                      cap_terms=2_000_000, K_full_cap=12, use_gcd=True,
                      fallback=True, node_cap=1_000_000, K_strict_cap=25,
                      terms_side_cap=100, aux_side_cap=8, progress=None):
    n_orig = inst.n
    cmax = max(1.0, float(inst.cmax))
    lam_con = lam_scale * 2.0 * cmax
    lam_quad = lam_scale * 4.0 * cmax

    use_raw = hasattr(inst, "av_orig")
    if use_raw:
        order = np.lexsort((inst.ci, inst.ri))
        ri_, ci_, av_ = inst.ri[order], inst.ci[order], inst.av_orig[order]
        sc_all = inst.scale
        lb_, ub_ = inst.lb * sc_all, inst.ub * sc_all
    else:
        ri_, ci_, av_ = inst.ri_r, inst.ci_r, inst.av_r
        sc_all = np.ones(inst.m)
        lb_, ub_ = inst.lb, inst.ub
    rs = np.searchsorted(ri_, np.arange(inst.m))
    re_ = np.searchsorted(ri_, np.arange(inst.m), side="right")
    t_build = time.time()

    strict_terms = {}
    slack_rows = []
    n_slack = 0
    n_strict = n_fb = n_skip = 0
    n_ovf = 0
    Ks = []
    NODE_CAP = node_cap
    nxt = n_orig

    for r in range(inst.m):
        if progress and r and r % 5000 == 0:
            progress(f"    行 {r}/{inst.m}  strict={n_strict} 兜底={n_fb} "
                     f"ovf={n_ovf} 用时{time.time()-t_build:.0f}s")
        s, e = rs[r], re_[r]
        if e <= s:
            continue
        cols = ci_[s:e]
        vals = av_[s:e]
        nz = np.abs(vals) > 1e-9
        cols_nz = cols[nz]
        sc = float(sc_all[r])
        lo, hi = lb_[r], ub_[r]
        negsum = float(-vals[vals < -1e-9].sum())
        eq = np.isfinite(lo) and np.isfinite(hi) and abs(hi - lo) < 1e-12
        if eq:
            sides = [("le", hi + negsum), ("ge", lo + negsum)]
        else:
            sides = []
            if np.isfinite(hi):
                sides.append(("le", hi + negsum))
            if np.isfinite(lo):
                sides.append(("ge", lo + negsum))

        a_n = vals / sc
        au = np.abs(a_n)
        vu = float(au.min()) if au.size else 1.0
        lam_r = min(lam_scale * 2.0 * cmax / max(vu * vu, 1e-18),
                    1e4 * cmax)

        for typ, bound in sides:
            handled = False
            cr = _canon_side(vals, bound, is_le=(typ == "le"),
                             use_gcd=use_gcd)
            if cr is not None and len(cr) == 0:
                handled = True
            elif cr:
                w, sgn, K = cr
                kcap = K_full_cap if mode == "full" else K_strict_cap
                ok_k = K <= kcap
                t = None
                if ok_k:
                    if mode == "strict":
                        if _count_monomials(w, K, terms_side_cap) is None:
                            n_ovf += 1
                        else:
                            t = build_e_K(w, K, terms_side_cap,
                                          max_nodes=NODE_CAP)
                            if t is None:
                                n_ovf += 1
                    else:
                        t = _e_K_from_h(w, K, cap_terms)
                        if t is None:
                            n_ovf += 1
                    if t is not None:
                        if sum(len(k) - 1 for k in t) > aux_side_cap:
                            n_ovf += 1
                        else:
                            for key, coef in t.items():
                                node = tuple(sorted(
                                    (int(cols_nz[j]), int(sgn[j]))
                                    for j in key))
                                strict_terms[node] = \
                                    strict_terms.get(node, 0.0) + lam_con * coef
                            n_strict += 1
                            Ks.append(K)
                            handled = True
                else:
                    n_ovf += 1
            if handled or not fallback:
                continue

            lo_n, hi_n = lo / sc, hi / sc
            if eq:
                ids = cols.astype(np.int64)
                slack_rows.append((lam_r, ids, a_n.copy(), -float(hi_n)))
            elif typ == "le":
                S = int(np.floor(hi_n - float(np.minimum(a_n, 0).sum()) + 1e-9))
                if S < 0:
                    n_skip += 1
                    continue
                ws = _slack_weights(S, slack)
                ids = np.concatenate([cols.astype(np.int64),
                                      np.arange(nxt, nxt + len(ws),
                                                dtype=np.int64)])
                a_vec = np.concatenate([a_n, np.array(ws, dtype=float)])
                nxt += len(ws)
                n_slack += len(ws)
                slack_rows.append((lam_r, ids, a_vec, -float(hi_n)))
            else:
                S = int(np.floor(float(np.maximum(a_n, 0).sum()) - lo_n + 1e-9))
                if S < 0:
                    n_skip += 1
                    continue
                ws = _slack_weights(S, slack)
                ids = np.concatenate([cols.astype(np.int64),
                                      np.arange(nxt, nxt + len(ws),
                                                dtype=np.int64)])
                a_vec = np.concatenate([a_n, -np.array(ws, dtype=float)])
                nxt += len(ws)
                n_slack += len(ws)
                slack_rows.append((lam_r, ids, a_vec, -float(lo_n)))
            n_fb += 1

    n_base = nxt
    if strict_terms:
        quad_terms, n_aux, aux_defs = _quadratize_fast(strict_terms, lam_quad,
                                                       n_base)
    else:
        quad_terms, n_aux, aux_defs = {}, 0, {}
    n_total = n_base + n_aux

    h = np.zeros(n_total, dtype=np.float64)
    h[:n_orig] = np.asarray(inst.c, dtype=np.float64)
    const = float(inst.obj_const)
    qd = {}

    def add_expanded(nodes, coef):
        nonlocal const
        neg = [g for (g, sg) in nodes if g < n_orig and sg < 0]
        pos = [g for (g, sg) in nodes if not (g < n_orig and sg < 0)]
        for rr in range(len(neg) + 1):
            for Sset in combinations(neg, rr):
                c = coef * ((-1) ** rr)
                idx = sorted(pos + list(Sset))
                if len(idx) == 0:
                    const += c
                elif len(idx) == 1:
                    h[idx[0]] += c
                elif len(idx) == 2:
                    i, j = idx
                    qd[(i, j)] = qd.get((i, j), 0.0) + c
                else:
                    raise RuntimeError("装配时出现次数 > 2 的项")

    for k, c in quad_terms.items():
        add_expanded(list(k), c)

    for lam_r, ids, a, d in slack_rows:
        h[ids] += lam_r * (a * a + 2.0 * d * a)
        k = ids.size
        if k < 2:
            const += lam_r * d * d
            continue
        ii, jj = np.triu_indices(k, k=1)
        pi, pj = np.minimum(ids[ii], ids[jj]), np.maximum(ids[ii], ids[jj])
        for i, j, v in zip(pi, pj, 2.0 * lam_r * a[ii] * a[jj]):
            if abs(v) > 1e-12:
                qd[(int(i), int(j))] = qd.get((int(i), int(j)), 0.0) + v
        const += lam_r * d * d

    ri_q, ci_q, vv_q = [], [], []
    for (i, j), c in qd.items():
        if abs(c) < 1e-12:
            continue
        ri_q.append(i)
        ci_q.append(j)
        vv_q.append(c)
    if ri_q:
        Q = coo_matrix((vv_q, (ri_q, ci_q)),
                       shape=(n_total, n_total)).tocsr()
        Q.sum_duplicates()
    else:
        Q = coo_matrix((n_total, n_total)).tocsr()

    return dict(h=h, Q=Q, const=const, n_orig=n_orig, n_aux=n_aux,
                n_slack=n_slack, n_total=n_total, nnz=int(Q.nnz),
                rows_ok=n_strict, rows_fb=n_fb, rows_skip=n_skip,
                rows_ovf=n_ovf, n_base=n_base,
                slack_rows=slack_rows, aux_defs=aux_defs,
                mode=mode, fallback=fallback,
                stats=dict(lam_con=lam_con, lam_quad=lam_quad,
                           K_med=(float(np.median(Ks)) if Ks else 0.0),
                           K_max=(int(max(Ks)) if Ks else 0),
                           n_terms=len(strict_terms)))

def _e_K_from_h(w, K, cap_terms=2_000_000):
    from simplex_build import build_h_m
    acc = {}
    for j in range(1, K + 1):
        hj = build_h_m(w, j, cap_terms)
        if hj is None:
            return None
        c = ((-1) ** (K - j)) * comb(K - 1, j - 1)
        for k, v in hj.items():
            acc[k] = acc.get(k, 0.0) + c * v
        if len(acc) > cap_terms:
            return None
    return {k: v for k, v in acc.items() if abs(v) > 1e-12}

def main():
    import argparse
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=r"D:\玻色\下三角算法\tests\miplib_qubo")
    ap.add_argument("--instances", required=True)
    ap.add_argument("--mode", default="strict", choices=["strict", "full"])
    ap.add_argument("--slack", default="binary", choices=["unary", "binary"])
    ap.add_argument("--lam-scale", type=float, default=1.0)
    a = ap.parse_args()
    for name in [x.strip() for x in a.instances.split(",") if x.strip()]:
        inst = core.Instance(name, os.path.join(a.dir, name + ".mps"))
        q = build_hybrid_qubo(inst, mode=a.mode, slack=a.slack,
                              lam_scale=a.lam_scale)
        print("%-16s %-7s 原变量=%-6d 松弛=%-7d 辅助=%-7d QUBO=%-7d 耦合=%-9d "
              "侧: strict=%d 兜底=%d 跳过=%d Kmax=%d"
              % (name, a.mode, q["n_orig"], q["n_slack"], q["n_aux"],
                 q["n_total"], q["nnz"], q["rows_ok"], q["rows_fb"],
                 q["rows_skip"], q["stats"]["K_max"]))

if __name__ == "__main__":
    main()
```

---

### A.3 HUBO 编码：混合罚项，不做 Rosenberg 二次化

源码：`tests\miplib_qubo\hubo_build.py`

```python
"""
HUBO 编码：混合罚项（strict e_K + 松弛兜底），**不做 Rosenberg 二次化**
======================================================================
目的：
  绕过二次化，直接得到高阶无约束二值多项式（HUBO）：
      E(x) = c·x + obj_const + Σ_strict侧 λ_con·e_K(v(x)) + Σ_兜底侧 λ_r·(a·x̂+d)²
  变量布局：[0, n_orig) 原变量 | [n_orig, n_base) 松弛变量（仅兜底行需要）。
  **没有辅助变量** —— 这是 HUBO 相对 QUBO 的结构优势（QUBO 版每个高阶项
  需要 len-1 个 Rosenberg 辅助变量，且要保证 aux 取到 lit(n1)·lit(n2)）。

  strict 侧的 e_K 项按 (全局变量号, 符号) node-key 收集后，v→x 展开
  （负号节点 v = 1-x，一个 k 阶 v-monomial 最多展开成 2^k 个 x-monomial），
  兜底松弛行本身就是二次式，直接展开进同一张项表。

  本文件只负责"构造"；求解（穷举 / 经典 SA）见 hubo_solver.py，
  罚项性质逐点验证见 verify_hubo.py。
"""
from __future__ import annotations

import os
import sys
import time
from itertools import combinations

import numpy as np

sys.path.insert(0, r"D:\玻色\下三角算法\tests\miplib_hard")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import miplib_sa_test as core
from simplex_build import build_e_K
from hybrid_build import (_canon_side, _count_monomials,
                          _slack_weights)

def _expand_cost(t, sgn, cols_nz, n_orig):
    tot = 0
    for key in t:
        e = 0
        for j in key:
            if cols_nz[j] < n_orig and sgn[j] < 0:
                e += 1
        tot += 1 << e
    return tot

def build_hybrid_hubo(inst, mode="strict", slack="binary", lam_scale=1.0,
                      K_strict_cap=25, terms_side_cap=100, use_gcd=True,
                      node_cap=1_000_000, expand_cap=200_000):
    n_orig = inst.n
    cmax = max(1.0, float(inst.cmax))
    lam_con = lam_scale * 2.0 * cmax

    use_raw = hasattr(inst, "av_orig")
    if use_raw:
        order = np.lexsort((inst.ci, inst.ri))
        ri_, ci_, av_ = inst.ri[order], inst.ci[order], inst.av_orig[order]
        sc_all = inst.scale
        lb_, ub_ = inst.lb * sc_all, inst.ub * sc_all
    else:
        ri_, ci_, av_ = inst.ri_r, inst.ci_r, inst.av_r
        sc_all = np.ones(inst.m)
        lb_, ub_ = inst.lb, inst.ub
    rs = np.searchsorted(ri_, np.arange(inst.m))
    re_ = np.searchsorted(ri_, np.arange(inst.m), side="right")
    t_build = time.time()

    strict_terms = {}
    strict_cost = 0
    slack_rows = []
    n_slack = 0
    n_strict = n_fb = n_skip = n_ovf = 0
    Ks = []
    nxt = n_orig

    for r in range(inst.m):
        s, e = rs[r], re_[r]
        if e <= s:
            continue
        cols = ci_[s:e]
        vals = av_[s:e]
        nz = np.abs(vals) > 1e-9
        cols_nz = cols[nz]
        sc = float(sc_all[r])
        lo, hi = lb_[r], ub_[r]
        negsum = float(-vals[vals < -1e-9].sum())
        eq = np.isfinite(lo) and np.isfinite(hi) and abs(hi - lo) < 1e-12
        if eq:
            sides = [("le", hi + negsum), ("ge", lo + negsum)]
        else:
            sides = []
            if np.isfinite(hi):
                sides.append(("le", hi + negsum))
            if np.isfinite(lo):
                sides.append(("ge", lo + negsum))

        a_n = vals / sc
        au = np.abs(a_n)
        vu = float(au.min()) if au.size else 1.0
        lam_r = min(lam_scale * 2.0 * cmax / max(vu * vu, 1e-18),
                    1e4 * cmax)

        for typ, bound in sides:
            handled = False
            cr = _canon_side(vals, bound, is_le=(typ == "le"),
                             use_gcd=use_gcd)
            if cr is not None and len(cr) == 0:
                handled = True
            elif cr:
                w, sgn, K = cr
                if K <= K_strict_cap and mode == "strict":
                    if _count_monomials(w, K, terms_side_cap) is None:
                        n_ovf += 1
                    else:
                        t = build_e_K(w, K, terms_side_cap,
                                      max_nodes=node_cap)
                        if t is None:
                            n_ovf += 1
                        else:
                            cost = _expand_cost(t, sgn, cols_nz, n_orig)
                            if strict_cost + cost > expand_cap:
                                n_ovf += 1
                            else:
                                strict_cost += cost
                                for key, coef in t.items():
                                    node = tuple(sorted(
                                        (int(cols_nz[j]), int(sgn[j]))
                                        for j in key))
                                    strict_terms[node] = \
                                        strict_terms.get(node, 0.0) \
                                        + lam_con * coef
                                n_strict += 1
                                Ks.append(K)
                                handled = True
                else:
                    n_ovf += 1
            if handled:
                continue

            lo_n, hi_n = lo / sc, hi / sc
            if eq:
                ids = cols.astype(np.int64)
                slack_rows.append((lam_r, ids, a_n.copy(), -float(hi_n)))
            elif typ == "le":
                S = int(np.floor(hi_n - float(np.minimum(a_n, 0).sum()) + 1e-9))
                if S < 0:
                    n_skip += 1
                    continue
                ws = _slack_weights(S, slack)
                ids = np.concatenate([cols.astype(np.int64),
                                      np.arange(nxt, nxt + len(ws),
                                                dtype=np.int64)])
                a_vec = np.concatenate([a_n, np.array(ws, dtype=float)])
                nxt += len(ws)
                n_slack += len(ws)
                slack_rows.append((lam_r, ids, a_vec, -float(hi_n)))
            else:
                S = int(np.floor(float(np.maximum(a_n, 0).sum()) - lo_n + 1e-9))
                if S < 0:
                    n_skip += 1
                    continue
                ws = _slack_weights(S, slack)
                ids = np.concatenate([cols.astype(np.int64),
                                      np.arange(nxt, nxt + len(ws),
                                                dtype=np.int64)])
                a_vec = np.concatenate([a_n, -np.array(ws, dtype=float)])
                nxt += len(ws)
                n_slack += len(ws)
                slack_rows.append((lam_r, ids, a_vec, -float(lo_n)))
            n_fb += 1

    n_base = nxt

    terms = {}
    const = float(inst.obj_const)

    def bump(key, val):
        terms[key] = terms.get(key, 0.0) + val

    def add_expanded(nodes, coef):
        nonlocal const
        neg = [g for (g, sg) in nodes if g < n_orig and sg < 0]
        pos = [g for (g, sg) in nodes if not (g < n_orig and sg < 0)]
        for rr in range(len(neg) + 1):
            for Sset in combinations(neg, rr):
                c = coef * ((-1) ** rr)
                idx = tuple(sorted(pos + list(Sset)))
                if len(idx) == 0:
                    const += c
                else:
                    bump(idx, c)

    for i in range(n_orig):
        ci_ = float(inst.c[i])
        if abs(ci_) > 0:
            bump((i,), ci_)

    for k, c in strict_terms.items():
        add_expanded(list(k), c)
        if len(terms) > expand_cap:
            raise RuntimeError("HUBO 项数超过 expand_cap=%d" % expand_cap)

    for lam_r, ids, a, d in slack_rows:
        const += lam_r * d * d
        lin = lam_r * (a * a + 2.0 * d * a)
        for g, lv in zip(ids.tolist(), lin.tolist()):
            bump((g,), lv)
        k = ids.size
        for ii in range(k):
            for jj in range(ii + 1, k):
                v = 2.0 * lam_r * a[ii] * a[jj]
                if abs(v) > 1e-12:
                    i2, j2 = ids[ii], ids[jj]
                    bump((int(min(i2, j2)), int(max(i2, j2))), v)

    terms = {k: v for k, v in terms.items() if abs(v) > 1e-12}
    degs = [len(k) for k in terms]
    dh = {}
    for d in degs:
        dh[d] = dh.get(d, 0) + 1
    return dict(terms=terms, const=const, n_orig=n_orig, n_slack=n_slack,
                n_total=n_base, rows_ok=n_strict, rows_fb=n_fb,
                rows_skip=n_skip, rows_ovf=n_ovf, slack_rows=slack_rows,
                stats=dict(lam_con=lam_con, n_terms=len(terms),
                           deg_max=(max(degs) if degs else 0),
                           deg_hist=dh,
                           K_med=(float(np.median(Ks)) if Ks else 0.0),
                           K_max=(int(max(Ks)) if Ks else 0),
                           t_build=time.time() - t_build))

def main():
    import argparse
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=r"D:\玻色\下三角算法\tests\miplib_qubo")
    ap.add_argument("--instances", required=True)
    ap.add_argument("--slack", default="binary", choices=["unary", "binary"])
    a = ap.parse_args()
    for name in [x.strip() for x in a.instances.split(",") if x.strip()]:
        inst = core.Instance(name, os.path.join(a.dir, name + ".mps"))
        hb = build_hybrid_hubo(inst, slack=a.slack)
        st = hb["stats"]
        print("%-16s 原变量=%-6d 松弛=%-6d HUBO变量=%-6d 项数=%-8d 最高阶=%-3d "
              "侧: strict=%d 兜底=%d 跳过=%d Kmax=%d 构建耗时=%.1fs"
              % (name, hb["n_orig"], hb["n_slack"], hb["n_total"],
                 st["n_terms"], st["deg_max"], hb["rows_ok"], hb["rows_fb"],
                 hb["rows_skip"], st["K_max"], st["t_build"]))
        print("   阶分布:", dict(sorted(st["deg_hist"].items())))

if __name__ == "__main__":
    main()
```

---

### A.4 QAPLIB → HUBO：one-hot 的 strict/兜底 + λ 规则 + 起跑自检

源码：`tests\qaplib\qap_hubo.py`

```python
"""
QAPLIB → HUBO 构建器（不二次化，直接任意阶项表）

QAP: min_{π} Σ_{i,k} A_ik · B_{π(i),π(k)}
二进制化：x_ij = 1 ⟺ 设施 i 放到位置 j（n² 个变量）
  · 目标：Σ_{i,k,j,l} A_ik B_jl x_ij x_kl（二次；互斥对 (i=k 或 j=l) 在可行点上
    恒为 0，为省项数直接跳过——可行点能量不变）
  · one-hot 约束（每设施一行、每位置一行，共 2n 行 Σx = 1）：
      ≤1 侧：strict e_2 = λ·Σ_{a<b} x_a x_b（二次，精确指示，无松弛）
      ≥1 侧：松弛兜底 λ·(Σx - Σ_b 2^b s_b - 1)²（二次，精确指示）
    —— 与 MIPLIB 混合编码同一套"strict 优先 + 兜底"哲学
λ 规则（lam_rule）：
    λ = max_{i,j} ( Σ_k|A_ik| · Σ_l|B_jl| )
推导：目标对 x_ij 单调（系数非负），删掉"设施 i → 位置 j"这一个赋值，
      目标损失 Δobj = Σ_{k,l} A_ik B_jl x_kl ≤ (Σ_k A_ik)(Σ_l B_jl)；
      同时设施行 i、位置列 j 各违 1 行 ≥1 约束，罚增 2λ。
      ⇒ 只要 2λ > Δobj_max，HUBO 全局最优就不可能落在"丢 1 格"的点上。
      λ 取上式（而非 MIPLIB 的 2·cmax）：QAP 目标是 n² 项非负求和，
      丢一格能省 O(n) 项，2·cmax 少了 n 因子 —— 这正是首轮 12/17 实例
      HUBO 全局最优不可行的原因（见 2026-09-27 记忆）。
      另需 2nλ > OPT（压住全零点），本规则实测对 17 实例均满足。
跑前自检：preflight() 以认证最优排列为基准，逐点验证
      "全零点 / 丢 1 格 / 随机扰动点"能量都不低于 OPT，不通过拒绝起跑。
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

QAP_DIR = os.path.dirname(os.path.abspath(__file__))

INSTANCES = {
    "nug12": 578, "nug14": 1014, "nug15": 1150, "nug18": 1930,
    "nug20": 2570, "nug22": 3596, "nug24": 3488, "nug25": 3744,
    "nug27": 5234, "nug30": 6124, "chr18a": 11098, "tai20a": 703482,
    "tai25a": 1167256, "scr20": 110030, "rou20": 725522,
    "had20": 6922, "tho30": 149936,
}

_OPT_F = os.path.join(QAP_DIR, "qap_opt.json")
if os.path.exists(_OPT_F):
    try:
        with open(_OPT_F, encoding="utf-8") as _f:
            INSTANCES.update({k: float(v) for k, v in json.load(_f).items()})
    except Exception:
        pass

def parse_dat(path):
    toks = open(path).read().replace(",", " ").split()
    n = int(toks[0])
    vals = np.array([int(t) for t in toks[1:]], dtype=np.float64)
    A = vals[:n * n].reshape(n, n)
    B = vals[n * n:n * n + n * n].reshape(n, n)
    return n, A, B

def parse_sln(path):
    toks = open(path).read().replace(",", " ").split()
    n = int(toks[0])
    obj = float(toks[1])
    perm = np.array([int(t) - 1 for t in toks[2:2 + n]], dtype=np.int64)
    return obj, perm

def qap_obj(A, B, perm):
    return float((A * B[np.ix_(perm, perm)]).sum())

def lam_rule(A, B, opt=None):
    ra = np.abs(A).sum(axis=1)
    rb = np.abs(B).sum(axis=1)
    lam = float(ra.max() * rb.max())
    if opt is not None:
        n = A.shape[0]
        lam = max(lam, 1.1 * float(opt) / (2.0 * n))
    return lam

def build_qap_hubo(A, B, lam=None):
    n = A.shape[0]
    n2 = n * n
    terms = {}
    const = 0.0

    def add(key, coef):
        if abs(coef) < 1e-12:
            return
        terms[key] = terms.get(key, 0.0) + coef

    nzA = [(i, k, A[i, k]) for i in range(n) for k in range(n)
           if A[i, k] != 0]
    nzB = [(j, l, B[j, l]) for j in range(n) for l in range(n)
           if B[j, l] != 0]
    cmax = 1.0
    for i, k, a in nzA:
        for j, l, b in nzB:
            c = a * b
            u, v = i * n + j, k * n + l
            if u == v:
                add((u,), c)
                cmax = max(cmax, abs(c))
            elif i == k or j == l:
                continue
            else:
                add((min(u, v), max(u, v)), c)
                cmax = max(cmax, abs(c))
    n_obj_terms = len(terms)

    if lam is None:
        lam = lam_rule(A, B)

    rows = []
    for i in range(n):
        rows.append([i * n + j for j in range(n)])
    for j in range(n):
        rows.append([i * n + j for i in range(n)])

    nbits = max(1, (n - 1).bit_length())
    n_slack = 2 * n * nbits
    sid0 = n2
    row_slack = []

    for rv in rows:
        for a_i in range(len(rv)):
            for b_i in range(a_i + 1, len(rv)):
                add((rv[a_i], rv[b_i]), lam)

    for r_i, rv in enumerate(rows):
        sl = list(range(sid0 + r_i * nbits, sid0 + (r_i + 1) * nbits))
        row_slack.append(sl)
        zu = [(u, 1.0) for u in rv] + \
             [(s, -float(1 << b)) for b, s in enumerate(sl)]
        for u, cu in zu:
            add((u,), lam * (cu * cu - 2.0 * cu))
            for v, cv in zu:
                if v > u:
                    add((u, v), lam * 2.0 * cu * cv)
        const += lam
    deg = max(len(k) for k in terms)
    return dict(terms=terms, const=const, n_total=n2 + n_slack,
                n2=n2, n=n, n_slack=n_slack, lam=lam, cmax=cmax,
                stats=dict(n_terms=len(terms), n_obj=n_obj_terms,
                           deg_max=deg, n_rows=2 * n, nbits=nbits))

def x_of_perm(perm, n_total):
    x = np.zeros(n_total, dtype=np.int8)
    n = len(perm)
    for i, j in enumerate(perm):
        x[i * n + j] = 1
    return x

def sln_perm(name, A, B, ret_obj=False):
    sobj, perm = parse_sln(os.path.join(QAP_DIR, name + ".sln"))
    ref = float(INSTANCES.get(name, sobj))
    if abs(qap_obj(A, B, perm) - ref) > 1e-6:
        alt = np.argsort(perm)
        if abs(qap_obj(A, B, alt) - ref) <= 1e-6:
            perm = alt
    return (ref, perm) if ret_obj else perm

def feasible_rows(x, n):
    X = x[:n * n].reshape(n, n)
    return np.concatenate([X.sum(axis=1), X.sum(axis=0)])

def preflight(A, B, m, perm, lam, n_total, n_sample=120, seed=0):
    n = A.shape[0]
    opt = qap_obj(A, B, perm)
    tol = 1e-9 * max(abs(opt), 1.0)
    x_opt = x_of_perm(perm, n_total)
    E_opt = float(m.energy(x_opt))
    ok1 = abs(E_opt - opt) <= tol

    E_zero = float(m.energy(np.zeros(n_total, dtype=np.int8)))
    ok2 = E_zero > opt + tol

    dmin, darg, nbad3 = np.inf, None, 0
    for i in range(n):
        x = x_opt.copy()
        x[i * n + perm[i]] = 0
        E = float(m.energy(x))
        if E < dmin:
            dmin, darg = E, (i, int(perm[i]))
        if E <= opt + tol:
            nbad3 += 1
    ok3 = nbad3 == 0

    rng = np.random.RandomState(seed)
    smin, nbad4, nfeas, nfeas_bad = np.inf, 0, 0, 0
    for _ in range(n_sample):
        k = int(rng.randint(1, n + 1))
        facs = rng.choice(n, size=k, replace=False)
        x = x_opt.copy()
        for i in facs:
            x[i * n + perm[i]] = 0
            x[i * n + int(rng.randint(n))] = 1
        if (feasible_rows(x, n) == 1).all():
            nfeas += 1
            if qap_obj(A, B, np.array([int(np.argmax(x[i * n:(i + 1) * n]))
                                       for i in range(n)])) < opt - tol:
                nfeas_bad += 1
            continue
        E = float(m.energy(x))
        if E < smin:
            smin = E
        if E <= opt + tol:
            nbad4 += 1
    ok4 = (nbad4 == 0 and nfeas_bad == 0)

    info = dict(ok=bool(ok1 and ok2 and ok3 and ok4),
                E_opt=E_opt, opt=opt, E_zero=E_zero, zero_over_opt=E_zero - opt,
                drop1_min=dmin, drop1_arg=darg, drop1_over_opt=dmin - opt,
                drop1_bad=nbad3, samp_min=smin, samp_over_opt=smin - opt,
                samp_bad=nbad4, samp_feas=nfeas, samp_feas_bad=nfeas_bad,
                lam=float(lam))
    return info["ok"], info

def main():
    import sys
    sys.path.insert(0, r"D:\玻色\下三角算法\tests\miplib_qubo")
    from hubo_solver import hubo_energy
    print("数据完整性 + 罚项性质验证（17 实例）")
    print("%-9s %-4s %-12s %-12s %-8s %-7s %-6s %s"
          % ("实例", "n", "官方最优", "sln复算", "E(sln)", "项数", "变量", "检查"))
    for name, opt in INSTANCES.items():
        n, A, B = parse_dat(os.path.join(QAP_DIR, name + ".dat"))
        sobj, perm = parse_sln(os.path.join(QAP_DIR, name + ".sln"))
        o = qap_obj(A, B, perm)
        if abs(o - sobj) >= 1e-6:
            perm = np.argsort(perm)
            o = qap_obj(A, B, perm)
        ok1 = abs(o - sobj) < 1e-6 and abs(sobj - opt) < 1e-6
        hb = build_qap_hubo(A, B)
        x = x_of_perm(perm, hb["n_total"])
        E = hubo_energy(hb["terms"], hb["const"], x)
        ok2 = abs(E - opt) < 1e-6
        rng = np.random.RandomState(0)
        ok3 = True
        for _ in range(20):
            p = rng.permutation(n)
            xp = x_of_perm(p, hb["n_total"])
            if abs(hubo_energy(hb["terms"], hb["const"], xp)
                   - qap_obj(A, B, p)) > 1e-6:
                ok3 = False
                break
        ok4 = True
        for _ in range(50):
            xq = (rng.rand(hb["n_total"]) < 0.5 / n).astype(np.int8)
            rs = feasible_rows(xq, n)
            if (rs == 1).all():
                continue
            Eq = hubo_energy(hb["terms"], hb["const"], xq)
            if Eq <= 0:
                ok4 = False
                break
        warn = "罚权警告(2nλ≤OPT)" if 2 * n * hb["lam"] <= opt else ""
        st = hb["stats"]
        print("%-9s %-4d %-12.6g %-12.6g %-8.6g %-7d %-6d %s %s"
              % (name, n, opt, o, E, st["n_terms"], hb["n_total"],
                 "通过" if (ok1 and ok2 and ok3 and ok4) else
                 "失败[%s%s%s%s]" % ("" if ok1 else "1", "" if ok2 else "2",
                                     "" if ok3 else "3", "" if ok4 else "4"),
                 warn))

if __name__ == "__main__":
    main()
```

---

### A.5 独立解校验器：自带解析器复算，违反行数 = 0 才判定可行

源码：`tests\miplib_hard\verify_sol.py`

```python
"""
独立解校验器（不复用求解器任何代码）
====================================
输入：MPS 实例文件 + 原始变量空间解文件（.sol，格式 "=obj= v" + "变量名 值"）
输出：逐行复算约束，报告 违反行数 / 最大违约 / 总违约 / 独立复算目标值，并给出判定

设计要点：
- 自带独立的最小 MPS 解析器（ROWS/COLUMNS/RHS/RANGES/BOUNDS），与求解器的解析器
  完全分离，避免"同一处 bug 被两边共用"而误判可行
- 独立处理 RANGES 与 E/L/G 行型，缺失变量按 0 处理
- 若 MPS 目标方向为 MAX，复算值按原始方向报告，同时给出最小化口径

用法：
  python verify_sol.py <inst.mps> <inst.sol> [--tol 1e-6]
退出码：0=严格可行，1=不可行，2=错误
"""

import argparse
import gzip
import io
import sys

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

def parse_mps_min(path):
    opener = gzip.open if path.endswith(".gz") else open
    section = None
    row_type, row_order = {}, []
    obj_row = None
    cols = []
    colset = set()
    obj = {}
    rows = []
    rhs, ranges = {}, {}
    bounds = {}
    objsense = "min"
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("*"):
                continue
            if line[0] not in " \t":
                section = line.split()[0].upper()
                continue
            p = line.split()
            if section == "OBJSENSE":
                objsense = p[0].lower()
            elif section == "ROWS":
                if p[0] == "N":
                    if obj_row is None:
                        obj_row = p[1]
                else:
                    row_type[p[1]] = p[0]
                    row_order.append(p[1])
            elif section == "COLUMNS":
                if "MARKER" in line.upper():
                    continue
                c = p[0]
                if c not in colset:
                    colset.add(c)
                    cols.append(c)
                for k in range(1, len(p) - 1, 2):
                    rn, val = p[k], float(p[k + 1])
                    if rn == obj_row:
                        obj[c] = obj.get(c, 0.0) + val
                    else:
                        rows.append((rn, c, val))
            elif section == "RHS":
                for k in range(1, len(p) - 1, 2):
                    if p[k] != obj_row:
                        rhs[p[k]] = rhs.get(p[k], 0.0) + float(p[k + 1])
            elif section == "RANGES":
                for k in range(1, len(p) - 1, 2):
                    ranges[p[k]] = float(p[k + 1])
            elif section == "BOUNDS":
                b = bounds.setdefault(p[2] if len(p) > 2 else p[1], {})
                t = p[0].upper()
                v = None
                if len(p) > 3:
                    try:
                        v = float(p[3])
                    except ValueError:
                        v = None
                if t == "FX":
                    b["fx"] = v
                elif t == "LO":
                    b["lo"] = v
                elif t == "UP":
                    b["up"] = v
                elif t == "BV":
                    b["lo"], b["up"] = 0.0, 1.0
                elif t == "MI":
                    b["lo"] = float("-inf")
                elif t == "PL":
                    b["up"] = float("inf")
    return objsense, row_type, row_order, rows, rhs, ranges, obj, bounds, cols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mps")
    ap.add_argument("sol")
    ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args()

    objsense, row_type, row_order, trips, rhs, ranges, obj, bounds, cols = \
        parse_mps_min(a.mps)

    xval, sol_obj = {}, None
    for line in open(a.sol, encoding="utf-8", errors="ignore"):
        s = line.strip()
        if not s:
            continue
        if s.startswith("="):
            toks = s.replace("=", " ").split()
            for i, t in enumerate(toks):
                if t.lower() in ("obj", "objective") and i + 1 < len(toks):
                    sol_obj = float(toks[i + 1])
                    break
            continue
        p = s.split()
        if len(p) == 2:
            try:
                xval[p[0]] = float(p[1])
            except ValueError:
                pass

    lhs = {r: 0.0 for r in row_order}
    for rn, c, v in trips:
        xv = xval.get(c)
        if xv:
            lhs[rn] = lhs.get(rn, 0.0) + v * xv

    nbad, max_viol, tot_viol = 0, 0.0, 0.0
    worst = None
    for rn in row_order:
        v = lhs.get(rn, 0.0)
        R = rhs.get(rn, 0.0)
        t = row_type[rn]
        rg = ranges.get(rn)
        if t == "E":
            lo = hi = R
            if rg:
                lo, hi = (R, R + abs(rg)) if rg > 0 else (R - abs(rg), R)
        elif t == "L":
            hi = R
            lo = R - abs(rg) if rg else float("-inf")
        else:
            lo = R
            hi = R + abs(rg) if rg else float("inf")
        viol = max(0.0, v - hi) + max(0.0, lo - v)
        if viol > a.tol:
            nbad += 1
            tot_viol += viol
            if viol > max_viol:
                max_viol, worst = viol, (rn, v, lo, hi)

    obj_val = sum(obj.get(c, 0.0) * xval.get(c, 0.0) for c in cols)
    nb_bad = 0
    for c in cols:
        xv = xval.get(c, 0.0)
        b = bounds.get(c, {})
        lo = b.get("lo", 0.0)
        hi = b.get("up", 1.0)
        if "fx" in b:
            lo = hi = b["fx"]
        if xv < lo - a.tol or xv > hi + a.tol:
            nb_bad += 1

    print(f"实例: {a.mps}")
    print(f"解文件: {a.sol}")
    print(f"变量数(文件出现): {len(cols)}  解中非零变量: {len(xval)}")
    print(f"目标方向: {objsense.upper()}")
    print(f"解文件声明目标: {sol_obj}")
    print(f"独立复算目标:   {obj_val:.10g}  差异: "
          f"{'N/A' if sol_obj is None else f'{abs(obj_val - sol_obj):.3g}'}")
    print(f"越界变量数: {nb_bad}")
    print(f"违反行数: {nbad} / {len(row_order)}")
    print(f"最大单行违约: {max_viol:.6g}  总违约: {tot_viol:.6g}")
    if worst:
        print(f"  最差行: {worst[0]}  lhs={worst[1]:.6g} 区间=[{worst[2]}, {worst[3]}]")
    ok = (nbad == 0 and nb_bad == 0)
    print("判定:", "严格可行（FEASIBLE）" if ok else "不可行（INFEASIBLE）")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
```

---

### A.6 一元松弛闭式罚项 SA 求解器（关键函数）

源码：`tests\miplib_hard\miplib_sa_test.py`（全文件 1257 行，此处只附编码与搜索直接相关的部分）

```python
"""
MIPLIB hard 实例 × 一元松弛不等式编码（x 空间闭式实现）SA 测试
============================================================

编码原理（docs/qubo_inequality_encoding.md）：
  线性约束 Σ a_i x_i <= / >= / = C 编码为 QUBO 罚项
      P · (Σ a_i x_i + y − C)²   (le, y∈{0,1}^C 一元松弛)
      P · (Σ a_i x_i − y − C)²   (ge)
  一元松弛的 min_y 有闭式 max(0, viol)²，经典求解器可整体消去 y 块，
  在 x 空间直接搜索（本项目 2026-09-11 高维验证 26/26 PASS 的最优实践）。
  本脚本即该闭式实现：
      E(x) = c·x + Σ_r λ_r · viol_r(x)²
      viol_r(x) = max(0, lhs_r − ub_r) + max(0, lb_r − lhs_r)
  （双向违约之和，同时覆盖 le/ge/eq 与 RANGES 行）

流程：MPS 解析 → 行归一化（谱宽处理：除以行尺度，避免 O(C²) 谱宽压垮温度）
      → 两阶段 SA（修复阶段求可行 + 优化阶段带自适应罚权退火）
      → 与 MIPLIB 最优已知值对比 gap

用法：python miplib_sa_test.py [每实例时间预算秒] [实例名...]
日志：run_log.txt（追加）；进度：progress.json（周期覆盖）；结果：results.md
"""
```

**A.6.1 `Instance.viol`** —— 双向违约量 viol_r(x)（闭式罚项的核心一行）

```python
def viol(self, lhs):
        v = np.maximum(0.0, lhs - self.ub) + np.maximum(0.0, self.lb - lhs)
        v[~np.isfinite(self.lb) & ~np.isfinite(self.ub)] = 0.0
        return v
```

**A.6.2 `Instance.feasible`** —— 可行性判定

```python
def feasible(self, x):
        return float(self.viol(self.lhs_init(x)).sum()) <= self.tol
```

**A.6.3 `sa_cycle`** —— 多周期回热 SA 单周期：单翻转 / 交换移动 + 增量 ΔE

```python
def sa_cycle(inst, x, lhs, v, obj, lamv, T0, Tmin, iters, deadline, rng,
             best_f_obj, best_f_x, prog, t_start, flips_total, name):
    cs, ce = inst.col_start[:-1], inst.col_start[1:]
    Tmin = max(1e-9, Tmin)
    alpha = (Tmin / T0) ** (1.0 / max(1, iters))
    T = T0
    total_v = float(v.sum())
    ones = IdxSet(x > 0)
    zeros = IdxSet(x < 1)
    obj0 = best_f_obj[0]
    it = 0
    while it < iters and time.time() < deadline:
        it += 1
        if it % 2048 == 0 and prog is not None:
            prog(name, "anneal", flips_total[0], time.time() - t_start,
                 best_f_obj[0], total_v, float(lamv.mean()), T)
        if (SWAP_PROB > 0 and ones.n > 0 and zeros.n > 0
                and rng.random() < SWAP_PROB):
            k1 = int(ones.idx[rng.integers(ones.n)])
            k2 = int(zeros.idx[rng.integers(zeros.n)])
            s1, s2 = slice(cs[k1], ce[k1]), slice(cs[k2], ce[k2])
            rws = np.concatenate([inst.ri[s1], inst.ri[s2]])
            aws = np.concatenate([-inst.av[s1], inst.av[s2]])
            uniq, inv = np.unique(rws, return_inverse=True)
            agg = np.zeros(uniq.size)
            np.add.at(agg, inv, aws)
            old = v[uniq]
            nv = _viol_at(inst, lhs[uniq] + agg, uniq)
            d_obj = inst.c[k2] - inst.c[k1]
            dE = d_obj + float((lamv[uniq] * (nv * nv - old * old)).sum())
            if dE <= 0 or rng.random() < math.exp(-dE / T):
                x[k1], x[k2] = 0, 1
                lhs[uniq] += agg
                v[uniq] = nv
                obj[0] += d_obj
                total_v = float(v.sum())
                flips_total[0] += 1
                ones.update(k1, False)
                zeros.update(k1, True)
                zeros.update(k2, False)
                ones.update(k2, True)
                if total_v <= inst.tol and (best_f_obj[0] is None
                                            or obj[0] < best_f_obj[0]):
                    best_f_obj[0] = obj[0]
                    best_f_x[0] = x.copy()
        else:
            k = int(rng.integers(inst.n))
            sgn = 1 - 2 * x[k]
            seg = slice(cs[k], ce[k])
            rws, aws = inst.ri[seg], inst.av[seg] * sgn
            old = v[rws]
            nv = _viol_at(inst, lhs[rws] + aws, rws)
            dE = inst.c[k] * sgn + float((lamv[rws] * (nv * nv - old * old)).sum())
            if dE <= 0 or rng.random() < math.exp(-dE / T):
                x[k] ^= 1
                lhs[rws] += aws
                v[rws] = nv
                obj[0] += inst.c[k] * sgn
                total_v = float(v.sum())
                flips_total[0] += 1
                if sgn > 0:
                    zeros.update(k, False)
                    ones.update(k, True)
                else:
                    ones.update(k, False)
                    zeros.update(k, True)
                if total_v <= inst.tol and (best_f_obj[0] is None
                                            or obj[0] < best_f_obj[0]):
                    best_f_obj[0] = obj[0]
                    best_f_x[0] = x.copy()
        T *= alpha
    improved = (best_f_obj[0] is not None
                and (obj0 is None or best_f_obj[0] < obj0 - 1e-12))
    return total_v, improved
```

**A.6.4 `sa_solve`** —— 两阶段主流程：初值竞选 → 回热循环 → 抛光

```python
def sa_solve(inst, budget_s, seed=12345, prog=None, x_init=None, x_warm=None,
             x_known=None):
    rng = np.random.default_rng(seed)
    t_start = time.time()
    t_end = t_start + budget_s
    t_rep = min(t_end, t_start + budget_s * REPAIR_SHARE)
    inits = []
    if x_known is not None:
        inits.append(np.asarray(x_known, dtype=np.int8).copy())
    if x_warm is not None:
        inits.append(np.asarray(x_warm, dtype=np.int8).copy())
    if x_init is not None:
        inits.append(np.asarray(x_init, dtype=np.int8).copy())
        xl = np.asarray(x_init, dtype=float)
        for thr in (0.25, 0.5, 0.75):
            inits.append((rng.random(inst.n) < xl + (thr - 0.5) * 0.8
                          ).astype(np.int8))
    inits.append(np.ones(inst.n, dtype=np.int8))
    inits.append(np.zeros(inst.n, dtype=np.int8))
    inits.append((rng.random(inst.n) < 0.5).astype(np.int8))

    feas = False
    best_repair_v, best_repair_x = np.inf, None
    for init in inits:
        if feas or time.time() >= t_rep:
            break
        x = init
        lhs = inst.lhs_init(x)
        v = inst.viol(lhs)
        if greedy_repair(inst, x, lhs, v, t_rep, rng):
            feas, bv, bx = True, 0.0, x.copy()
        else:
            feas, wf, bv, bx = walksat(inst, x, lhs, v, t_rep, rng)
        if bv < best_repair_v:
            best_repair_v, best_repair_x = bv, bx
    log(f"  {inst.name} WalkSAT修复: 可行={feas} "
        f"最好viol={best_repair_v:.4g} 耗时={time.time()-t_start:.1f}s")
    if feas:
        x = bx
    else:
        x = best_repair_x.copy()
    lhs = inst.lhs_init(x)
    v = inst.viol(lhs)
    obj = [float(inst.c @ x)]
    best_f_obj, best_f_x = [None], [None]
    flips_total = [0]
    if feas:
        best_f_obj[0] = obj[0]
        best_f_x[0] = x.copy()

    lam_scale = 1.0
    Tmin_cycle = 0.01 * max(1.0, inst.cmax)
    T0_full = max(1.0, inst.cmax)
    rep_cap = max(10.0, min(60.0, 0.08 * budget_s))
    pol_cap = max(3.0, min(20.0, 0.03 * budget_s))
    cyc = 0
    stall = 0
    t_sa_end = t_end - budget_s * LNS_FRAC if LNS_FRAC > 0 else t_end
    while time.time() < t_sa_end:
        cyc += 1
        lamv = lam_scale * inst.lam_row
        T0 = T0_full * (1.0 + 0.5 * stall) if stall else T0_full * 0.5
        T0 = min(T0, 4.0 * T0_full)
        iters = 200_000
        dl = min(t_sa_end, time.time() + max(5.0, budget_s * 0.25))
        if best_f_x[0] is not None and stall >= 1:
            x[:] = best_f_x[0]
            if stall >= 3:
                pert = rng.choice(inst.n, size=max(1, inst.n // (20 * stall)),
                                  replace=False)
                x[pert] ^= 1
            lhs[:] = inst.lhs_init(x)
            v[:] = inst.viol(lhs)
            obj[0] = float(inst.c @ x)
        tv, improved = sa_cycle(inst, x, lhs, v, obj, lamv, T0, Tmin_cycle,
                                iters, dl, rng, best_f_obj, best_f_x, prog,
                                t_start, flips_total, inst.name)
        if improved:
            stall = 0
        else:
            stall += 1
        if tv > inst.tol and time.time() < t_sa_end:
            if best_f_x[0] is not None and tv > best_repair_v:
                x[:] = best_f_x[0]
                lhs[:] = inst.lhs_init(x)
                v[:] = inst.viol(lhs)
                obj[0] = float(inst.c @ x)
                tv = float(v.sum())
            elif (best_f_x[0] is None and best_repair_x is not None
                  and tv > best_repair_v):
                x[:] = best_repair_x
                lhs[:] = inst.lhs_init(x)
                v[:] = inst.viol(lhs)
                tv = best_repair_v
            if tv > inst.tol:
                dl2 = min(t_sa_end, time.time() + rep_cap)
                if greedy_repair(inst, x, lhs, v, dl2, rng):
                    feas2, bv2, bx2 = True, 0.0, x.copy()
                else:
                    feas2, _, bv2, bx2 = walksat(inst, x, lhs, v, dl2, rng)
                if bv2 < best_repair_v:
                    best_repair_v, best_repair_x = bv2, bx2
                if feas2:
                    obj[0] = float(inst.c @ x)
                    if best_f_obj[0] is None or obj[0] < best_f_obj[0]:
                        best_f_obj[0], best_f_x[0] = obj[0], x.copy()
                    log(f"  {inst.name} 周期{cyc} 补修复成功 obj={obj[0]:.6g}")
                else:
                    lam_scale *= LAMBDA_GROW
        if best_f_x[0] is not None and time.time() < t_end:
            x[:] = best_f_x[0]
            lhs[:] = inst.lhs_init(x)
            v[:] = inst.viol(lhs)
            new_obj, pol_n = polish(inst, x, lhs, v,
                                    min(t_end, time.time() + pol_cap), rng)
            if (pol_n and float(v.sum()) <= inst.tol
                    and new_obj < best_f_obj[0] - 1e-12):
                best_f_obj[0], best_f_x[0] = new_obj, x.copy()
                log(f"  {inst.name} 周期{cyc} 抛光改进 -> {new_obj:.8g}"
                    f"（{pol_n} 步）")
        if cyc % 4 == 0 or tv > inst.tol or improved:
            log(f"  {inst.name} 周期{cyc} lam_scale={lam_scale:.3g} T0={T0:.3g} "
                f"viol={tv:.3g} best={best_f_obj[0]} stall={stall} "
                f"翻转={flips_total[0]/1e6:.1f}M elapsed={time.time()-t_start:.0f}s")
    lns_it = 0
    if LNS_FRAC > 0 and best_f_x[0] is not None and time.time() < t_end:
        x[:] = best_f_x[0]
        lhs[:] = inst.lhs_init(x)
        v[:] = inst.viol(lhs)
        obj[0] = float(inst.c @ x)
        before = best_f_obj[0]
        lns_refine(inst, x, 0, rng, lam_scale * inst.lam_row,
                   best_f_obj, best_f_x, deadline=t_end, prog=prog,
                   t_start=t_start, name=inst.name)
        if best_f_obj[0] is not None and best_f_obj[0] < before - 1e-12:
            log(f"  {inst.name} LNS 精修改进: {before:.8g} -> "
                f"{best_f_obj[0]:.8g}")

    pol_moves = 0
    if best_f_x[0] is not None and time.time() < t_end:
        x[:] = best_f_x[0]
        lhs[:] = inst.lhs_init(x)
        v[:] = inst.viol(lhs)
        obj[0] = float(inst.c @ x)
        new_obj, pol_moves = polish(inst, x, lhs, v, t_end, rng)
        if new_obj < best_f_obj[0] - 1e-12 and float(v.sum()) <= inst.tol:
            best_f_obj[0], best_f_x[0] = new_obj, x.copy()
            log(f"  {inst.name} 抛光改进: {new_obj:.8g}（移动 {pol_moves} 次）")
    return dict(feasible=best_f_obj[0] is not None, obj=best_f_obj[0],
                x=best_f_x[0], flips=flips_total[0], lam=lam_scale,
                polish_moves=pol_moves, repair_ok=feas)
```
