# -*- coding: utf-8 -*-
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

import miplib_sa_test as core                       # noqa: E402
from simplex_build import build_e_K                 # noqa: E402
from hybrid_build import (_canon_side, _count_monomials,   # noqa: E402
                          _slack_weights)


# ══════════════════════════════════════════════════════════════
# strict 展开项数预估（预筛，防止 2^负号节点 爆炸）
# ══════════════════════════════════════════════════════════════

def _expand_cost(t, sgn, cols_nz, n_orig):
    """
    估计 e_K 的 node-key 表 t 展开成 x-单项式后的项数上界。
    每个含 e 个负号节点的 key 展开成 2^e 个 x-monomial（v = 1-x 代入）。
    这一步必须在真正展开**之前**做：原实现只在展开后检查 expand_cap，
    单个 20 个负号的 key 会先撑出 2^20 条项再报错（内存风险）。
    """
    tot = 0
    for key in t:
        e = 0
        for j in key:
            if cols_nz[j] < n_orig and sgn[j] < 0:
                e += 1
        tot += 1 << e
    return tot


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════

def build_hybrid_hubo(inst, mode="strict", slack="binary", lam_scale=1.0,
                      K_strict_cap=25, terms_side_cap=100, use_gcd=True,
                      node_cap=1_000_000, expand_cap=200_000):
    """
    与 build_hybrid_qubo 相同的行级策略（strict 优先 + 松弛兜底），
    但 strict 高阶项**不二次化**，直接进 HUBO 项表。
    返回：
      dict(terms={tuple(sorted var ids): coef}  # 任意阶（1..deg_max）
           const=..., n_orig, n_slack, n_total=n_orig+n_slack,
           rows_ok=n_strict, rows_fb, rows_skip, rows_ovf,
           slack_rows=[...],  # 兜底行（热启动/调试用，同 QUBO 版）
           stats=dict(lam_con, n_terms, deg_max, deg_hist, K_med, K_max))
    """
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

    strict_terms = {}       # node-key(变量号,符号) -> 罚系数（含 λ_con）
    strict_cost = 0         # strict 展开后的 x-单项式数上界（预算 = expand_cap）
    slack_rows = []         # (lam_r, var_ids, a_vec, d)
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
                                # 展开后项数会超预算 → 该行改走松弛兜底，
                                # 而不是把整个子问题抛错作废
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

            # ── 松弛兜底（归一化空间，二次式）──
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

    # ── 装配 HUBO 项表（不做二次化）──
    terms = {}
    const = float(inst.obj_const)

    def bump(key, val):
        terms[key] = terms.get(key, 0.0) + val

    def add_expanded(nodes, coef):
        """v→x 展开：负号节点 (g,-1)(g<n_orig) 以 1-x 代入，任意阶"""
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

    # 目标（线性）
    for i in range(n_orig):
        ci_ = float(inst.c[i])
        if abs(ci_) > 0:
            bump((i,), ci_)

    # strict 高阶罚项
    for k, c in strict_terms.items():
        add_expanded(list(k), c)
        if len(terms) > expand_cap:
            raise RuntimeError("HUBO 项数超过 expand_cap=%d" % expand_cap)

    # 兜底行（二次）
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
