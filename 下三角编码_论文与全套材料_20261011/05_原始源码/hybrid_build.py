# -*- coding: utf-8 -*-
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
import miplib_sa_test as core  # noqa: E402
from simplex_build import build_e_K, quadratize  # noqa: E402


# ══════════════════════════════════════════════════════════════
# 单侧规范化（独立于 canonical_rows，允许单侧兜底）
# ══════════════════════════════════════════════════════════════

def _canon_side(vals, bound, is_le, use_gcd=True):
    """
    对一侧约束做规范化，返回 (w, sgn, K) 或 None（该侧转兜底）。
    bound = 原始右端 + Σ|负系数|（≤ 型）或 lb + Σ|负系数|（≥ 型），需近似整数。
    is_le=True : 违反 ⟺ Σ w·v ≥ C+1        → K = C+1（补变量 v=x/1-x, sgn 原号）
    is_le=False: 违反 ⟺ Σ w·(1-v) ≤ R-1    → K = S-R+1（sgn 取反）
    """
    vals = np.asarray(vals, dtype=float)
    nz = np.abs(vals) > 1e-9
    if not nz.any():
        return []                       # 空行：无需罚项
    wa = np.abs(vals[nz])
    if np.max(np.abs(wa - np.round(wa))) > 1e-6:
        return None
    if abs(bound - round(bound)) > 1e-6:
        return None                     # 右端非整数（安全保护，canonical_rows 原版未查）
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
        Rq = -((-B) // g)               # ⌈B/g⌉
        K = S // g - Rq + 1
        if K <= 0:
            return None
        return (w // g, (-sgn).astype(np.int8), K)


def _count_monomials(w, K, cap=100_000):
    """e_K 单项式个数：Π(1+...+z^{w_j}) 的 z^K 系数（DP，O(n·K)）；
    超 cap 返回 None。用于预筛，避免 build_e_K 递归节点爆炸。"""
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
    """Rosenberg 二次化（堆实现，避免原版每步 O(|pending|) 的 max 扫描）。
    语义与 simplex_build.quadratize 等价（辅助变量编号可能不同，最终多项式相同）。
    返回 (二次化后项表, 辅助变量数, aux_defs)；
    aux_defs[k] = (node1, node2) 表示第 k 个辅助变量的取值应为 lit(node1)·lit(node2)
    （用于热启动时给辅助变量赋正确初值）。"""
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
    """S≥0 的松弛权重向量（binary: 1,2,4,..,余量；unary: 全 1）"""
    if S == 0:
        return []
    if slack == "unary":
        return [1] * S
    bits = max(1, int(np.ceil(np.log2(S + 1))))
    ws = [2 ** k for k in range(bits - 1)]
    ws.append(S - (2 ** (bits - 1) - 1))
    return ws


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════

def build_hybrid_qubo(inst, mode="strict", slack="binary", lam_scale=1.0,
                      cap_terms=2_000_000, K_full_cap=12, use_gcd=True,
                      fallback=True, node_cap=1_000_000, K_strict_cap=25,
                      terms_side_cap=100, aux_side_cap=8, progress=None):
    """
    mode      : "strict"（e_K 直构）或 "full"（h 基 Pascal 逆，K 受 K_full_cap 限制）
    slack     : 兜底松弛类型 "binary" / "unary"
    fallback  : False 时等价于纯 strict（跳过行不兜底），供 A/B 对比
    """
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

    strict_terms = {}       # node-key(全局变量号,符号) -> 罚系数
    slack_rows = []         # (lam_r, var_ids, a_vec, d)
    n_slack = 0
    n_strict = n_fb = n_skip = 0        # 侧计数
    n_ovf = 0                           # 节点超限转兜底的侧数
    Ks = []
    NODE_CAP = node_cap
    nxt = n_orig                       # 松弛变量分配指针

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
        cols_nz = cols[nz]           # 非零系数列（与 _canon_side 的 w/sgn 对齐）
        sc = float(sc_all[r])
        lo, hi = lb_[r], ub_[r]
        negsum = float(-vals[vals < -1e-9].sum())   # Σ|负系数|，与 canonical_rows 一致
        eq = np.isfinite(lo) and np.isfinite(hi) and abs(hi - lo) < 1e-12
        if eq:
            sides = [("le", hi + negsum), ("ge", lo + negsum)]
        else:
            sides = []
            if np.isfinite(hi):
                sides.append(("le", hi + negsum))
            if np.isfinite(lo):
                sides.append(("ge", lo + negsum))

        a_n = vals / sc                                  # 归一化系数（兜底用）
        au = np.abs(a_n)
        vu = float(au.min()) if au.size else 1.0
        lam_r = min(lam_scale * 2.0 * cmax / max(vu * vu, 1e-18),
                    1e4 * cmax)

        for typ, bound in sides:
            handled = False
            cr = _canon_side(vals, bound, is_le=(typ == "le"),
                             use_gcd=use_gcd)
            if cr is not None and len(cr) == 0:          # 全零行：无需罚项
                handled = True
            elif cr:                                     # (w,sgn,K)
                w, sgn, K = cr
                kcap = K_full_cap if mode == "full" else K_strict_cap
                ok_k = K <= kcap
                t = None
                if ok_k:
                    if mode == "strict":
                        if _count_monomials(w, K, terms_side_cap) is None:
                            n_ovf += 1          # DP 预筛：项数超限，直接兜底
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
                            n_ovf += 1          # 辅助变量预算超限，转兜底
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

            # ── 松弛兜底（归一化空间）──
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

    # ── Rosenberg 二次化（严格项，堆实现）──
    n_base = nxt
    if strict_terms:
        quad_terms, n_aux, aux_defs = _quadratize_fast(strict_terms, lam_quad,
                                                       n_base)
    else:
        quad_terms, n_aux, aux_defs = {}, 0, {}
    n_total = n_base + n_aux

    # ── 装配 ──
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
    """完整下三角路线（供 mode='full'）：h_1..h_K + Pascal 逆变换"""
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
