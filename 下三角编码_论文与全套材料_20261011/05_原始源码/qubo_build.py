# -*- coding: utf-8 -*-
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
import miplib_sa_test as core  # noqa: E402


def build_qubo(inst: core.Instance, slack: str = "unary",
               lam_scale: float = 1.0, lam_cap_mult: float = 1e4):
    """返回 dict(h, Q, const, n_orig, n_total, slack_vars, rows_info)"""
    n_orig = inst.n
    cmax = max(1.0, float(inst.cmax))
    const = float(inst.obj_const)
    ri, ci, av = inst.ri_r, inst.ci_r, inst.av_r
    row_slice = np.searchsorted(ri, np.arange(inst.m))
    row_end = np.searchsorted(ri, np.arange(inst.m), side="right")

    slack_rows = []      # (行号, 编码类型, 系数向量 [(var, coef)], 常数 d, λ)
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
            """sign=+1: lhs + s = bound (≤行); sign=-1: lhs − s = bound (≥行)"""
            nonlocal nxt, n_slack
            coefs = [(int(c), float(v)) for c, v in zip(cols, vals)]
            if sign > 0:
                S = bound - float(np.minimum(vals, 0).sum())
            else:
                S = float(np.maximum(vals, 0).sum()) - bound
            S = int(np.floor(S + 1e-9))
            if S < 0:
                return None                      # 该行本身不可行
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

        if lo_f and hi_f and abs(hi - lo) < 1e-12:        # 等式
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
    h[:n_orig] = inst.c                              # 目标线性项
    # 稀疏上三角：逐约束累加（避免 n² 稠密矩阵）
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
        # 上三角配对（i<j），向量化
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
