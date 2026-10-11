# -*- coding: utf-8 -*-
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

# 17 个认证最优实例（QAPLIB 标注 OPT）——首轮基准
INSTANCES = {
    "nug12": 578, "nug14": 1014, "nug15": 1150, "nug18": 1930,
    "nug20": 2570, "nug22": 3596, "nug24": 3488, "nug25": 3744,
    "nug27": 5234, "nug30": 6124, "chr18a": 11098, "tai20a": 703482,
    "tai25a": 1167256, "scr20": 110030, "rou20": 725522,
    "had20": 6922, "tho30": 149936,
}

# 扩充实例（QAPLIB 页面标 (OPT)，下载后经 .sln 复算核验）
# 由 qap_fetch_wide.py 生成，缺失时静默跳过
_OPT_F = os.path.join(QAP_DIR, "qap_opt.json")
if os.path.exists(_OPT_F):
    try:
        with open(_OPT_F, encoding="utf-8") as _f:
            INSTANCES.update({k: float(v) for k, v in json.load(_f).items()})
    except Exception:
        pass


def parse_dat(path):
    """QAPLIB .dat：n，随后 n² 个 A，再 n² 个 B（空白分隔；
    个别实例（ste36a 等）用逗号分隔，统一替换掉）"""
    toks = open(path).read().replace(",", " ").split()
    n = int(toks[0])
    vals = np.array([int(t) for t in toks[1:]], dtype=np.float64)
    A = vals[:n * n].reshape(n, n)
    B = vals[n * n:n * n + n * n].reshape(n, n)
    return n, A, B


def parse_sln(path):
    """QAPLIB .sln：n，目标值，随后 n 个 1-based 排列（可跨行；容忍逗号分隔）"""
    toks = open(path).read().replace(",", " ").split()
    n = int(toks[0])
    obj = float(toks[1])
    perm = np.array([int(t) - 1 for t in toks[2:2 + n]], dtype=np.int64)
    return obj, perm


def qap_obj(A, B, perm):
    """obj(π) = Σ_{i,k} A_ik · B_{π(i),π(k)}"""
    return float((A * B[np.ix_(perm, perm)]).sum())


def lam_rule(A, B, opt=None):
    """
    罚权规则：λ = max_{i,j}(Σ_k|A_ik| · Σ_l|B_jl|)
    A、B 非负时 = max_i Σ_k|A_ik| × max_j Σ_l|B_jl|（行和乘积上界）。
    opt 给定时再与 1.1·OPT/(2n) 取大（压住全零点的下界，正常情况下不生效）。
    """
    ra = np.abs(A).sum(axis=1)
    rb = np.abs(B).sum(axis=1)
    lam = float(ra.max() * rb.max())
    if opt is not None:
        n = A.shape[0]
        lam = max(lam, 1.1 * float(opt) / (2.0 * n))
    return lam


def build_qap_hubo(A, B, lam=None):
    """
    返回 dict(terms, const, n_total, n2, n_rows, row_vars, stats)
    terms: {(tuple ids 升序): coef}；x 变量 id = i*n+j ∈ [0, n²)
    """
    n = A.shape[0]
    n2 = n * n
    terms = {}
    const = 0.0

    def add(key, coef):
        if abs(coef) < 1e-12:
            return
        terms[key] = terms.get(key, 0.0) + coef

    # ── 目标：二次双和，跳过互斥对 ──
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
                continue                      # 互斥对：可行点恒 0
            else:
                add((min(u, v), max(u, v)), c)
                cmax = max(cmax, abs(c))
    n_obj_terms = len(terms)

    if lam is None:
        lam = lam_rule(A, B)

    # ── one-hot 行（2n 行）──
    rows = []
    for i in range(n):                        # 设施 i
        rows.append([i * n + j for j in range(n)])
    for j in range(n):                        # 位置 j
        rows.append([i * n + j for i in range(n)])

    # ≥1 侧松弛位数：s ∈ [0, n-1]
    nbits = max(1, (n - 1).bit_length())
    n_slack = 2 * n * nbits
    sid0 = n2
    row_slack = []

    # ≤1 侧：strict e_2
    for rv in rows:
        for a_i in range(len(rv)):
            for b_i in range(a_i + 1, len(rv)):
                add((rv[a_i], rv[b_i]), lam)

    # ≥1 侧：λ(Σx - Σ 2^b s_b - 1)²
    for r_i, rv in enumerate(rows):
        sl = list(range(sid0 + r_i * nbits, sid0 + (r_i + 1) * nbits))
        row_slack.append(sl)
        zu = [(u, 1.0) for u in rv] + \
             [(s, -float(1 << b)) for b, s in enumerate(sl)]
        for u, cu in zu:
            add((u,), lam * (cu * cu - 2.0 * cu))   # (Σc z -1)² 线性部
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
    """
    读官方 .sln 并统一成"设施 i → 位置 π(i)"约定。
    QAPLIB 个别实例（tho30）的 .sln 用逆排列给出，需取逆才能复算出认证最优值。
    参照值优先用 INSTANCES 里的权威 OPT（个别 .sln 的头值有误，如 kra32）。
    """
    sobj, perm = parse_sln(os.path.join(QAP_DIR, name + ".sln"))
    ref = float(INSTANCES.get(name, sobj))
    if abs(qap_obj(A, B, perm) - ref) > 1e-6:
        alt = np.argsort(perm)
        if abs(qap_obj(A, B, alt) - ref) <= 1e-6:
            perm = alt
    return (ref, perm) if ret_obj else perm


def feasible_rows(x, n):
    """返回每行 Σx（前 n 行设施、后 n 行位置）"""
    X = x[:n * n].reshape(n, n)
    return np.concatenate([X.sum(axis=1), X.sum(axis=0)])


def preflight(A, B, m, perm, lam, n_total, n_sample=120, seed=0):
    """
    λ 跑前自检（断言）：以认证最优排列为基准，
      ① 基准点 E(π*) 是否等于 OPT（编码自洽）
      ② 全零点 E(0) = 2nλ 是否 > OPT（压住"全不指派"这个最便宜不可行点）
      ③ 丢 1 格：n 个点（设施行、位置列各空 1 行）能量是否都 > OPT
      ④ 随机扰动：k 个设施重新随机放置（k=1..n，n_sample 次）
         · 若扰动恰好仍是可行排列 → 只校验 obj ≥ OPT（认证最优性质）
         · 若不可行 → 必须 E > OPT（否则 HUBO 全局最优不在可行域上）
    m 为 HuboModel（仅用 m.energy，避免纯 Python 逐项评估）。
    返回 (ok, info)
    """
    n = A.shape[0]
    opt = qap_obj(A, B, perm)
    tol = 1e-9 * max(abs(opt), 1.0)
    x_opt = x_of_perm(perm, n_total)
    E_opt = float(m.energy(x_opt))
    ok1 = abs(E_opt - opt) <= tol

    E_zero = float(m.energy(np.zeros(n_total, dtype=np.int8)))
    ok2 = E_zero > opt + tol

    # ③ 丢 1 格（必然不可行）
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

    # ④ 随机扰动
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
        # ① 官方解复算 == 官方最优 == 记录值（QAPLIB 个别实例 .sln 用逆排列
        #    约定，如 tho30；两种取向都试，最优值不受约定影响）
        o = qap_obj(A, B, perm)
        if abs(o - sobj) >= 1e-6:
            perm = np.argsort(perm)
            o = qap_obj(A, B, perm)
        ok1 = abs(o - sobj) < 1e-6 and abs(sobj - opt) < 1e-6
        hb = build_qap_hubo(A, B)
        x = x_of_perm(perm, hb["n_total"])
        # 松弛按"恰好消违"回填：每行 Σx=1 → s = Σx-1 = 0（已全零）
        E = hubo_energy(hb["terms"], hb["const"], x)
        ok2 = abs(E - opt) < 1e-6
        # ③ 随机排列 20 个：E == obj
        rng = np.random.RandomState(0)
        ok3 = True
        for _ in range(20):
            p = rng.permutation(n)
            xp = x_of_perm(p, hb["n_total"])
            if abs(hubo_energy(hb["terms"], hb["const"], xp)
                   - qap_obj(A, B, p)) > 1e-6:
                ok3 = False
                break
        # ④ 随机破坏点：罚 > 0（E > 去掉罚的同点下界）
        ok4 = True
        for _ in range(50):
            xq = (rng.rand(hb["n_total"]) < 0.5 / n).astype(np.int8)
            rs = feasible_rows(xq, n)
            if (rs == 1).all():
                continue
            Eq = hubo_energy(hb["terms"], hb["const"], xq)
            # 罚>0 ⟺ E > 纯目标部分；纯目标 >= 0（A,B 非负）
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
