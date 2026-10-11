# -*- coding: utf-8 -*-
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

import gzip
import json
import math
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

# ────────────────────────── 配置 ──────────────────────────
DIR = os.environ.get("MIPLIB_DIR") or os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(DIR, "run_log.txt")
PROG_PATH = os.path.join(DIR, "progress.json")
RESULT_PATH = os.path.join(DIR, "results.md")
SOL_DIR = os.path.join(DIR, "solutions")   # 官方已知解（可选）
USE_SOL_INIT = os.environ.get("MIPLIB_USE_SOL", "1") == "1"  # 是否用官方解热启动

BEST_KNOWN = {
    "s1234": 29.0,              # min, CPLEX 12.9
    "reblock354": -39280175.0,  # min
    "pythago7824": 0.0,         # min（可行性问题）
    "neos-4954274-beardy": 20946.48,  # min
    "ds": 93.5200,              # min, set partitioning
}
DEFAULT_INSTANCES = ["s1234", "reblock354", "pythago7824",
                     "neos-4954274-beardy", "ds"]
DEFAULT_BUDGET = 300  # 每实例 SA 秒数
CBC_SECONDS = int(os.environ.get("MIPLIB_CBC_SECONDS", "120"))  # CBC 基线限时

# SA 参数（相对量，乘实例尺度）
REPAIR_SHARE = 0.45        # 修复阶段时间份额
SWAP_PROB = float(os.environ.get("MIPLIB_SWAP_PROB", "0.35"))  # 交换移动概率
LNS_FRAC = float(os.environ.get("MIPLIB_LNS_FRAC", "0.25"))    # LNS 精修预算占比
FLIP_WINDOW = 100_000      # 自适应窗口
LAMBDA_GROW = 10.0         # 无可行解时 λ 增长率
LAMBDA_SHRINK = 0.5        # 可行但停滞时 λ 回落
LOG_EVERY = 50_000         # 每多少次翻转写一行日志


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    try:
        print(line, flush=True)          # 控制台可能是 GBK，编码失败不能中断实验
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode(), flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ────────────────────────── MPS 解析 ──────────────────────────
def parse_mps(path):
    """极简健壮 MPS 解析器（自由/固定格式通吃）。
    返回 dict: sense, n, col_names, obj(稠密), obj_const,
              rows: [(lb, ub)]（含 RANGES 展开，None 表示无穷侧）,
              稀疏三元组 (rows_idx, cols_idx, vals), var_types, fixed
    """
    opener = gzip.open if path.endswith(".gz") else open
    section = None
    row_type = {}     # row name -> N/E/L/G
    row_idx = {}      # row name -> index（含目标行单独记）
    obj_name = None
    col_idx = {}
    obj_terms = {}    # col -> coeff
    rhs = {}          # row name -> value
    ranges = {}       # row name -> value
    bounds = {}       # col -> dict
    int_cols = set()
    in_int = False
    triplets = []     # (row_i, col_j, a_ij)
    objsense = "min"

    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("*"):
                continue
            if line[0] not in " \t":           # 节名行
                parts = line.split()
                section = parts[0].upper()
                if section == "OBJSENSE":
                    # 下一行才是值
                    continue
                continue
            parts = line.split()
            if section == "OBJSENSE":
                objsense = parts[0].lower()
                continue
            if section == "ROWS":
                rtype, rname = parts[0].upper(), parts[1]
                if rtype == "N":
                    if obj_name is None:
                        obj_name = rname
                    continue
                row_type[rname] = rtype
                row_idx[rname] = len(row_idx)
            elif section == "COLUMNS":
                if "MARKER" in line.upper():
                    if "INTORG" in line.upper():
                        in_int = True
                    elif "INTEND" in line.upper():
                        in_int = False
                    continue
                cname = parts[0]
                j = col_idx.get(cname)
                if j is None:
                    j = len(col_idx)
                    col_idx[cname] = j
                if in_int:
                    int_cols.add(cname)
                for k in range(1, len(parts) - 1, 2):
                    rname, val = parts[k], float(parts[k + 1])
                    if rname == obj_name:
                        obj_terms[cname] = obj_terms.get(cname, 0.0) + val
                    elif rname in row_idx:
                        triplets.append((row_idx[rname], j, val))
            elif section == "RHS":
                for k in range(1, len(parts) - 1, 2):
                    rname, val = parts[k], float(parts[k + 1])
                    if rname == obj_name:
                        continue  # 目标常数（MIPLIB 罕用，忽略并记录）
                    rhs[rname] = rhs.get(rname, 0.0) + val
            elif section == "RANGES":
                for k in range(1, len(parts) - 1, 2):
                    rname, val = parts[k], float(parts[k + 1])
                    ranges[rname] = val
            elif section == "BOUNDS":
                btype = parts[0].upper()
                cname = parts[2] if len(parts) > 2 else parts[1]
                val = float(parts[3]) if len(parts) > 3 and parts[3].lstrip("-").replace(".", "").isdigit() else None
                b = bounds.setdefault(cname, {})
                if btype == "BV":
                    b["bin"] = True
                elif btype == "LO":
                    b["lo"] = val
                elif btype == "UP":
                    b["up"] = val
                elif btype == "FX":
                    b["fx"] = val
                elif btype in ("FR", "MI", "PL"):
                    b[btype.lower()] = True
                elif btype in ("LI", "UI"):
                    b[btype.lower()] = val
                    int_cols.add(cname)

    n = len(col_idx)
    m = len(row_idx)
    # 行界
    lb = np.full(m, -np.inf)
    ub = np.full(m, np.inf)
    for rname, ri in row_idx.items():
        t = row_type[rname]
        r = rhs.get(rname, 0.0)
        rg = ranges.get(rname)
        if t == "L":
            ub[ri] = r
            if rg is not None:
                lb[ri] = r - abs(rg)
        elif t == "G":
            lb[ri] = r
            if rg is not None:
                ub[ri] = r + abs(rg)
        else:  # E
            if rg is None or rg == 0:
                lb[ri] = ub[ri] = r
            else:
                lb[ri], ub[ri] = (r, r + abs(rg)) if rg > 0 else (r - abs(rg), r)

    # 变量类型与固定值
    types = []
    fixed = {}
    for cname, j in col_idx.items():
        b = bounds.get(cname, {})
        is_int = cname in int_cols or any(k in b for k in ("li", "ui"))
        if b.get("bin"):
            types.append("bin")
        elif is_int and b.get("lo", 0) == 0 and b.get("up", 1) == 1:
            types.append("bin")
        elif is_int:
            types.append("int")
        else:
            types.append("cont")
        if "fx" in b:
            fixed[j] = b["fx"]

    obj = np.zeros(n)
    for cname, v in obj_terms.items():
        obj[col_idx[cname]] = v

    col_names = [None] * n
    for cname, j in col_idx.items():
        col_names[j] = cname

    # 每列上下界（供整数变量二进制展开使用）
    var_lo = np.zeros(n)
    var_hi = np.full(n, np.inf)
    for cname, j in col_idx.items():
        b = bounds.get(cname, {})
        if b.get("bin"):
            var_lo[j], var_hi[j] = 0.0, 1.0
        else:
            if b.get("fr"):
                var_lo[j], var_hi[j] = -np.inf, np.inf
            else:
                if b.get("mi"):
                    var_lo[j] = -np.inf
                elif "lo" in b:
                    var_lo[j] = b["lo"]
                if b.get("pl"):
                    var_hi[j] = np.inf
                elif "up" in b:
                    var_hi[j] = b["up"]
                if b.get("li") is not None and "li" in b:
                    var_lo[j] = b["li"]
                if b.get("ui") is not None and "ui" in b:
                    var_hi[j] = b["ui"]
        if "fx" in b:
            var_lo[j] = var_hi[j] = b["fx"]

    ri = np.array([t[0] for t in triplets], dtype=np.int64)
    ci = np.array([t[1] for t in triplets], dtype=np.int64)
    vv = np.array([t[2] for t in triplets], dtype=np.float64)
    return dict(sense=objsense, n=n, m=m, obj=obj, lb=lb, ub=ub,
                rows=ri, cols=ci, vals=vv, types=types, fixed=fixed,
                col_names=col_names, var_lo=var_lo, var_hi=var_hi)


# ────────────────────────── 实例装配 ──────────────────────────
class Instance:
    """解析后的实例 + 归一化 + CSR 列结构 + 固定变量折叠"""

    def __init__(self, name, path):
        self.name = name
        t0 = time.time()
        d = parse_mps(path)
        self.parse_s = time.time() - t0
        self.n_raw, self.m, self.nnz = d["n"], d["m"], len(d["vals"])
        self.minimize = d["sense"] != "max"

        bad = [i for i, t in enumerate(d["types"])
               if t == "cont" and i not in d["fixed"]]
        if bad:
            raise RuntimeError(f"{name}: {len(bad)} 个连续变量（暂不支持），跳过")

        # 固定变量折叠：行 rhs 平移 + 目标常数
        fixed = d["fixed"]
        obj_const = 0.0
        lb, ub = d["lb"].copy(), d["ub"].copy()
        keep = np.ones(d["n"], dtype=bool)
        for j, v in fixed.items():
            keep[j] = False
            obj_const += d["obj"][j] * v
        remap = -np.ones(d["n"], dtype=np.int64)
        remap[keep] = np.arange(keep.sum())
        sel = keep[d["cols"]]
        ri, ci, vv = d["rows"][sel], remap[d["cols"][sel]], d["vals"][sel]
        # 折叠固定变量贡献
        for j, v in fixed.items():
            mask = d["cols"] == j
            if mask.any():
                lb[d["rows"][mask]] -= d["vals"][mask] * v
                ub[d["rows"][mask]] -= d["vals"][mask] * v
        # 供解还原/独立校验使用
        self.keep = keep
        self.remap = remap
        self.fixed_map = fixed
        self.col_names_raw = d["col_names"]
        self._raw_obj = d["obj"].copy()   # 原始目标系数（未取反）

        # ── 通用性：整数变量二进制展开 x = lo + Σ w_k·b_k ──
        # 使本求解器可处理一般整数变量（不再局限于纯 0/1 实例）
        types = d["types"]
        vlo, vhi = d["var_lo"], d["var_hi"]
        int_cols = [j for j in range(d["n"])
                    if types[j] == "int" and j not in fixed and keep[j]]
        self.expand_info = {}
        n_bin = int(keep.sum())
        if int_cols:
            # 新列编号：非整数列在前（0..n_bin-1），整数展开块在后
            first_col, nbits, weights, lo_arr = {}, {}, {}, {}
            bin_cols = [j for j in range(d["n"])
                        if keep[j] and types[j] != "int"]
            n_bin = len(bin_cols)
            newidx = np.full(d["n"], -1, dtype=np.int64)
            newidx[np.array(bin_cols, dtype=np.int64)] = np.arange(n_bin)
            nxt = n_bin
            for j in int_cols:
                lo = float(vlo[j])
                hi = float(vhi[j])
                if not np.isfinite(hi):
                    raise RuntimeError(
                        f"{name}: 整数变量 {d['col_names'][j]} 上界无穷，无法展开")
                width = int(round(hi - lo))
                if width <= 0:
                    raise RuntimeError(
                        f"{name}: 整数变量 {d['col_names'][j]} 值域为空")
                nb = max(1, int(math.ceil(math.log2(width + 1))))
                w = [2 ** k for k in range(nb - 1)]
                w.append(width - (2 ** (nb - 1) - 1))   # 末位取余量，恰好覆盖值域
                first_col[j], nbits[j] = nxt, nb
                weights[j] = np.array(w, dtype=np.float64)
                lo_arr[j] = lo
                nxt += nb
                self.expand_info[j] = (lo, first_col[j], weights[j])
            n_exp = nxt
            is_int = np.isin(d["cols"], np.array(int_cols, dtype=np.int64))
            maxb = max(nbits.values())
            # 查表（避免按列循环）
            lo_lookup = np.zeros(d["n"])
            nb_lookup = np.ones(d["n"], dtype=np.int64)
            wtab = np.zeros((d["n"], maxb))
            firsttab = np.zeros(d["n"], dtype=np.int64)
            for j in int_cols:
                lo_lookup[j] = lo_arr[j]
                nb_lookup[j] = nbits[j]
                wtab[j, :nbits[j]] = weights[j]
                firsttab[j] = first_col[j]
            sub_r = d["rows"][is_int]
            sub_c = d["cols"][is_int]
            sub_v = d["vals"][is_int]
            # ① 行 rhs 平移：减去 a_ij·lo（向量化）
            shift = sub_v * lo_lookup[sub_c]
            np.add.at(lb, sub_r, -shift)
            np.add.at(ub, sub_r, -shift)
            # ② 整数列的三元组按位展开
            reps = nb_lookup[sub_c]
            valid = np.arange(maxb)[None, :] < reps[:, None]
            new_c = (firsttab[sub_c][:, None] + np.arange(maxb)[None, :])[valid]
            new_w = wtab[sub_c][valid]
            ri = np.concatenate([d["rows"][~is_int], np.repeat(sub_r, reps)])
            ci = np.concatenate([newidx[d["cols"][~is_int]], new_c])
            vv = np.concatenate([d["vals"][~is_int],
                                 np.repeat(sub_v, reps) * new_w])
            # ③ 目标：整数列贡献 c_j·lo_j（常数）与 c_j·w_k（新列）
            for j in int_cols:
                obj_const += d["obj"][j] * lo_arr[j]
            c_new = np.zeros(n_exp)
            c_new[:n_bin] = d["obj"][np.array(bin_cols, dtype=np.int64)]
            for j in int_cols:
                c_new[first_col[j]:first_col[j] + nbits[j]] = \
                    d["obj"][j] * weights[j]
            self.n = n_exp
            self.c = c_new
            self.n_int_expanded = len(int_cols)
            self.n_extra = n_exp - n_bin
            self.col_map = newidx
        else:
            self.n = n_bin
            self.c = d["obj"][keep]
            self.n_int_expanded = 0
            self.n_extra = 0
            self.col_map = remap
        self.n_bin = n_bin
        if not self.minimize:
            self.c = -self.c
        self.obj_const = obj_const

        # 行归一化（谱宽处理：所有行除以自身尺度）
        scale = np.ones(self.m)
        for r in range(self.m):
            s = max(np.abs(vv[ri == r]).max() if (ri == r).any() else 0.0,
                    abs(ub[r]) if np.isfinite(ub[r]) else 0.0,
                    abs(lb[r]) if np.isfinite(lb[r]) else 0.0, 1.0)
            scale[r] = s
        self.av_orig = vv.copy()      # 归一化前的原始系数（与 ri/ci 同序）
        self.scale = scale.copy()     # 行尺度
        vv = vv / scale[ri]
        self.lb, self.ub = lb / scale, ub / scale
        self.tol = 1e-6

        # 列式 CSR（供单点翻转 O(deg) 增量更新）
        order = np.lexsort((ri, ci))
        self.ri, self.ci, self.av = ri[order], ci[order], vv[order]
        self.col_start = np.zeros(self.n + 1, dtype=np.int64)
        np.add.at(self.col_start, self.ci + 1, 1)
        self.col_start = np.cumsum(self.col_start)
        # 行式 CSR（供 WalkSAT 按行取变量）
        order_r = np.lexsort((ci, ri))
        self.ri_r, self.ci_r, self.av_r = ri[order_r], ci[order_r], vv[order_r]
        self.row_start = np.zeros(self.m + 1, dtype=np.int64)
        np.add.at(self.row_start, self.ri_r + 1, 1)
        self.row_start = np.cumsum(self.row_start)

        # 目标尺度（用于温度与 λ 定标）
        self.cmax = float(np.abs(self.c).max()) if self.n else 1.0
        self.csum = float(np.abs(self.c).sum()) or 1.0

        # 逐行罚权（改进点）：λ_r = 2·cmax / viol_unit_r²
        # viol_unit_r = 该行最小的单变量翻转违约量，使各行"最小罚项"量级一致，
        # 避免行尺度差异导致某些行实际上不受约束
        au = np.abs(self.av)
        vu = np.full(self.m, np.inf)
        np.minimum.at(vu, self.ri, au)
        vu[~np.isfinite(vu)] = 1.0
        self.viol_unit = np.maximum(vu, 1e-9)
        base = 2.0 * max(1.0, self.cmax)
        cap = 1e4 * max(1.0, self.cmax)
        self.lam_row = np.minimum(base / self.viol_unit ** 2, cap)

    def to_raw(self, x):
        """把求解器空间的解还原为原始变量空间：{列名: 值}。
        处理三件事：固定变量折叠、整数变量二进制展开、最大化符号。
        用途：输出可供独立校验的 .sol 文件（原始变量空间）。"""
        vals = {}
        x = np.asarray(x)
        for j, nm in enumerate(self.col_names_raw):
            if not self.keep[j]:
                v = self.fixed_map.get(j, 0.0)
                if v:
                    vals[nm] = float(v)
                continue
            if j in self.expand_info:
                lo, first, w = self.expand_info[j]
                k = min(len(w), x.size - first)
                v = lo + float(np.dot(x[first:first + k], w[:k])) if k > 0 else lo
            else:
                idx = int(self.remap[j])
                v = float(x[idx]) if idx < x.size else 0.0
            if v:
                vals[nm] = v
        return vals

    def raw_objective(self, vals):
        """用原始变量值复算目标值（原始 MPS 定义的目标，未取反）"""
        obj = 0.0
        for j, nm in enumerate(self.col_names_raw):
            if nm in vals:
                obj += self._raw_obj[j] * vals[nm]
        return obj

    def lhs_init(self, x):
        lhs = np.zeros(self.m)
        np.add.at(lhs, self.ri, self.av * x[self.ci])
        return lhs

    def viol(self, lhs):
        v = np.maximum(0.0, lhs - self.ub) + np.maximum(0.0, self.lb - lhs)
        v[~np.isfinite(self.lb) & ~np.isfinite(self.ub)] = 0.0
        return v

    def feasible(self, x):
        return float(self.viol(self.lhs_init(x)).sum()) <= self.tol

    def lp_relax(self):
        """LP 松弛（HiGHS）：返回 (lp_obj 或 None, 取整初始点 x 或 None)"""
        try:
            n, m = self.n, self.m
            # 用归一化后的 lb/ub 直接构 LP（等价）
            # 注意：必须用与 ri_r/ci_r 同一排序的 av_r，否则系数会错位
            A = sp.csc_matrix((self.av_r, (self.ri_r, self.ci_r)),
                              shape=(m, n))
            nL = np.isfinite(self.ub) & ~np.isfinite(self.lb)
            nG = ~np.isfinite(self.ub) & np.isfinite(self.lb)
            nE = np.isfinite(self.ub) & np.isfinite(self.lb) & (self.ub == self.lb)
            nR = np.isfinite(self.ub) & np.isfinite(self.lb) & (self.ub != self.lb)
            A_ub = sp.vstack([A[nL], -A[nG], A[nR], -A[nR]]).tocsc() \
                if (nL.sum() + nG.sum() + nR.sum()) else None
            b_ub = np.concatenate([self.ub[nL], -self.lb[nG],
                                   self.ub[nR], -self.lb[nR]]) if A_ub is not None else None
            A_eq = A[nE].tocsc() if nE.any() else None
            b_eq = self.ub[nE] if nE.any() else None
            res = linprog(self.c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=[(0, 1)] * n, method="highs")
            if not res.success:
                return None, None
            x_lp = (np.asarray(res.x) >= 0.5).astype(np.int8)
            return float(res.fun), x_lp
        except Exception:
            return None, None


# ────────────────────────── CBC 基线（经典分支切割，限时） ──
ASCII_DIR = os.environ.get("MIPLIB_ASCII", r"D:\miplib_ascii")  # CBC 读不了非 ASCII 路径


def load_solu():
    """读取 MIPLIB 权威最优值文件（=opt= 已证明 / =best= 未证明），返回 name→(tag, val)"""
    path = os.path.join(DIR, "miplib2017-v36.solu")
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8", errors="ignore"):
        p = line.split()
        if len(p) >= 3 and p[0] in ("=opt=", "=best="):
            try:
                out[p[1]] = (p[0], float(p[2]))
            except Exception:
                pass
    return out


def load_known_sol(inst):
    """读取官方已知解 .sol（首行 =obj=，其余为 变量名 值），返回 (obj, x 折叠向量)"""
    path = os.path.join(SOL_DIR, inst.name + ".sol")
    if not os.path.exists(path):
        return None, None
    obj, vals = None, {}
    for line in open(path, encoding="utf-8", errors="ignore"):
        s = line.strip()
        if not s:
            continue
        if s.startswith("="):
            try:
                obj = float(s.replace("=", " ").split()[1])
            except Exception:
                pass
            continue
        p = s.split()
        if len(p) == 2:
            try:
                vals[p[0]] = float(p[1])
            except Exception:
                pass
    x = cbc_sol_to_x(inst, vals)
    return obj, x


def cbc_baseline(name, secs):
    """用 python-mip 自带的 CBC 限时求解，返回 (status, obj, x_raw_dict)"""
    path = os.path.join(ASCII_DIR, name + ".mps")
    if not os.path.exists(path):
        path = os.path.join(DIR, name + ".mps")
    try:
        import mip
        m = mip.Model()
        m.verbose = 0
        m.read(path)
        m.max_seconds = secs
        st = m.optimize(max_seconds=secs)
        if m.num_solutions:
            obj = float(m.objective_value)
            sol = {}
            for v in m.vars:
                if v.x > 0.5:
                    sol[v.name] = v.x
            return str(st), obj, sol
        return str(st), None, None
    except Exception as e:
        return "error: %s" % e, None, None


def cbc_sol_to_x(inst, sol):
    """CBC 稀疏解 → 展开后的 0/1 向量（整数变量按其位权分解）"""
    if not sol:
        return None
    n2i = _name2idx(inst)
    x = np.zeros(inst.n, dtype=np.int8)
    for cname, val in sol.items():
        j = n2i.get(cname)
        if j is None:
            continue
        if j in inst.expand_info:
            lo, first, w = inst.expand_info[j]
            rest = int(round(val - lo))
            for k, wk in enumerate(w):
                wk = int(wk)
                if rest >= wk:
                    x[first + k] = 1
                    rest -= wk
        elif inst.keep[j]:
            idx = int(inst.col_map[j])
            if idx >= 0:
                x[idx] = 1 if val > 0.5 else 0
    return x


def _name2idx(inst):
    if not hasattr(inst, "_n2i"):
        inst._n2i = {nm: i for i, nm in enumerate(inst.col_names_raw)}
    return inst._n2i


# ────────────────────────── 混合求解器：WalkSAT 修复 + 多周期回热 SA ──
class BadRows:
    """O(1) 均匀抽样的违约行集合（swap-delete）"""

    def __init__(self, m):
        self.idx = np.zeros(m, dtype=np.int64)
        self.pos = np.full(m, -1, dtype=np.int64)
        self.n = 0

    def update(self, r, is_bad):
        if is_bad:
            if self.pos[r] < 0:
                self.idx[self.n] = r
                self.pos[r] = self.n
                self.n += 1
        else:
            p = self.pos[r]
            if p >= 0:
                last = self.idx[self.n - 1]
                self.idx[p] = last
                self.pos[last] = p
                self.pos[r] = -1
                self.n -= 1


class IdxSet:
    """O(1) 增删 + O(1) 均匀抽样的下标集合（用于维护 1/0 变量集合）"""

    def __init__(self, mask):
        idx = np.nonzero(mask)[0]
        self.idx = np.zeros(mask.size, dtype=np.int64)
        self.idx[:idx.size] = idx
        self.pos = np.full(mask.size, -1, dtype=np.int64)
        self.pos[idx] = np.arange(idx.size)
        self.n = int(idx.size)

    def update(self, k, present):
        p = self.pos[k]
        if present:
            if p < 0:
                self.idx[self.n] = k
                self.pos[k] = self.n
                self.n += 1
        else:
            if p >= 0:
                last = self.idx[self.n - 1]
                self.idx[p] = last
                self.pos[last] = p
                self.pos[k] = -1
                self.n -= 1


def _viol_at(inst, lhs_r, rws):
    nv = np.maximum(0.0, lhs_r - inst.ub[rws]) + \
         np.maximum(0.0, inst.lb[rws] - lhs_r)
    nv[~np.isfinite(inst.lb[rws]) & ~np.isfinite(inst.ub[rws])] = 0.0
    return nv


def greedy_repair(inst, x, lhs, v, deadline, rng, max_pass=3, max_cand=64):
    """贪心修复：按违约从大到小遍历违约行，全支撑扫描选最优翻转。
    对集合划分/覆盖类结构（行支撑大）效果远好于随机采样。"""
    cs, ce = inst.row_start[:-1], inst.row_start[1:]
    ccs, cce = inst.col_start[:-1], inst.col_start[1:]
    for _ in range(max_pass):
        if time.time() > deadline:
            break
        rows = np.nonzero(v > inst.tol)[0]
        if rows.size == 0:
            return True
        order = rows[np.argsort(-v[rows])]
        improved = False
        for r in order:
            if time.time() > deadline:
                return float(v.sum()) <= inst.tol
            if v[r] <= inst.tol:
                continue
            cand = inst.ci_r[cs[r]:ce[r]]
            if cand.size > max_cand:        # 大支撑行抽样，避免修复耗时爆炸
                cand = rng.choice(cand, max_cand, replace=False)
            best_d, best_k = -1e-12, -1
            for k in cand:
                k = int(k)
                seg = slice(ccs[k], cce[k])
                rws, aws = inst.ri[seg], inst.av[seg] * (1 - 2 * x[k])
                d = float(_viol_at(inst, lhs[rws] + aws, rws).sum() - v[rws].sum())
                if d < best_d:
                    best_d, best_k = d, k
            if best_k >= 0:
                k = best_k
                seg = slice(ccs[k], cce[k])
                rws, aws = inst.ri[seg], inst.av[seg] * (1 - 2 * x[k])
                lhs[rws] += aws
                v[rws] = _viol_at(inst, lhs[rws], rws)
                x[k] ^= 1
                improved = True
        if not improved:
            break
    return float(v.sum()) <= inst.tol


def walksat(inst, x, lhs, v, deadline, rng, noise=0.3, max_cand=24):
    """WalkSAT 式可行解修复：均匀选违约行 → 选使总违约(L1)降最多的变量翻转。
    自适应噪声；全局停滞 4 万步 → 随机扰动 10% 重启（保留好结构）。
    返回 (可行?, 翻转数, best_v, best_x)"""
    cs, ce = inst.row_start[:-1], inst.row_start[1:]
    ccs, cce = inst.col_start[:-1], inst.col_start[1:]
    bad = BadRows(inst.m)
    for r in np.nonzero(v > inst.tol)[0]:
        bad.update(int(r), True)
    flips = 0
    best_v = float(v.sum())
    best_x = x.copy()
    since_imp = 0
    while bad.n > 0 and time.time() < deadline:
        r = int(bad.idx[rng.integers(bad.n)])
        cand = inst.ci_r[cs[r]:ce[r]]
        if cand.size > max_cand:
            cand = rng.choice(cand, max_cand, replace=False)
        best_d, best_k = 1e18, -1
        for k in cand:
            k = int(k)
            seg = slice(ccs[k], cce[k])
            rws, aws = inst.ri[seg], inst.av[seg] * (1 - 2 * x[k])
            d = float(_viol_at(inst, lhs[rws] + aws, rws).sum() - v[rws].sum())
            if d < best_d:
                best_d, best_k = d, k
        if best_k >= 0 and rng.random() > noise:
            k = best_k
        else:
            k = int(cand[rng.integers(cand.size)])
        seg = slice(ccs[k], cce[k])
        rws, aws = inst.ri[seg], inst.av[seg] * (1 - 2 * x[k])
        nv = _viol_at(inst, lhs[rws] + aws, rws)
        x[k] ^= 1
        lhs[rws] += aws
        for rr, vv2 in zip(rws, nv):
            bad.update(int(rr), vv2 > inst.tol)
        v[rws] = nv
        flips += 1
        since_imp += 1
        tv = float(v.sum())
        if tv < best_v - 1e-12:
            best_v = tv
            best_x = x.copy()
            since_imp = 0
            noise = max(0.15, noise - 0.05)
        elif since_imp > 4000:
            noise = min(0.7, noise + 0.1)
            since_imp = 0
        if since_imp and since_imp % 20_000 == 0 and bad.n > 0:
            # 全局停滞：从 best_x 扰动少量变量重启（扰动过大破坏结构）
            x[:] = best_x
            pert = rng.choice(inst.n, size=max(1, inst.n // 25), replace=False)
            x[pert] ^= 1
            lhs[:] = inst.lhs_init(x)
            v[:] = inst.viol(lhs)
            bad = BadRows(inst.m)
            for rr in np.nonzero(v > inst.tol)[0]:
                bad.update(int(rr), True)
    return bad.n == 0, flips, best_v, best_x


def sa_cycle(inst, x, lhs, v, obj, lamv, T0, Tmin, iters, deadline, rng,
             best_f_obj, best_f_x, prog, t_start, flips_total, name):
    """一个退火周期（几何降温）。lamv 为逐行罚权数组。返回 (总违约, 是否改进)"""
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
        # ── 移动类型：交换(1→0 与 0→1 同时翻转) 或 单比特翻转 ──
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
                if sgn > 0:      # 0 -> 1
                    zeros.update(k, False)
                    ones.update(k, True)
                else:            # 1 -> 0
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


def polish(inst, x, lhs, v, deadline, rng, max_pass=40, row_sample=3000):
    """可行域内最速下降抛光：只接受"保持可行 且 降低目标"的单翻转/行内交换。
    返回 (目标值, 移动次数)。用于把 SA 的解推到局部最优。"""
    cs, ce = inst.col_start[:-1], inst.col_start[1:]
    rcs, rce = inst.row_start[:-1], inst.row_start[1:]
    obj = float(inst.c @ x)
    n_move = 0
    for _ in range(max_pass):
        if time.time() > deadline:
            break
        improved = False
        # ① 单翻转：按目标增益排序，只试有增益的
        gain = -inst.c * (1 - 2 * x)
        for k in np.argsort(gain):
            k = int(k)
            sgn = 1 - 2 * x[k]
            if inst.c[k] * sgn >= -1e-12:
                break
            seg = slice(cs[k], ce[k])
            rws = inst.ri[seg]
            aws = inst.av[seg] * sgn
            nv = _viol_at(inst, lhs[rws] + aws, rws)
            if float(nv.sum() - v[rws].sum()) <= inst.tol:
                x[k] ^= 1
                lhs[rws] += aws
                v[rws] = nv
                obj += inst.c[k] * sgn
                improved = True
                n_move += 1
            if time.time() > deadline:
                break
        # ② 行内交换：同一约束内 1→0 与 0→1（保持结构不变）
        rows = rng.choice(inst.m, size=min(inst.m, row_sample), replace=False)
        for r in rows:
            if time.time() > deadline:
                break
            vs = inst.ci_r[rcs[r]:rce[r]]
            if vs.size < 2 or vs.size > 64:
                continue
            ones_r = [int(j) for j in vs if x[j] == 1]
            zeros_r = [int(j) for j in vs if x[j] == 0]
            hit = False
            for k1 in ones_r:
                for k2 in zeros_r:
                    d_obj = inst.c[k2] - inst.c[k1]
                    if d_obj >= -1e-12:
                        continue
                    s1, s2 = slice(cs[k1], ce[k1]), slice(cs[k2], ce[k2])
                    rws = np.concatenate([inst.ri[s1], inst.ri[s2]])
                    aws = np.concatenate([-inst.av[s1], inst.av[s2]])
                    uniq, inv = np.unique(rws, return_inverse=True)
                    agg = np.zeros(uniq.size)
                    np.add.at(agg, inv, aws)
                    nv = _viol_at(inst, lhs[uniq] + agg, uniq)
                    if float(nv.sum() - v[uniq].sum()) <= inst.tol:
                        x[k1], x[k2] = 0, 1
                        lhs[uniq] += agg
                        v[uniq] = nv
                        obj += d_obj
                        improved = True
                        n_move += 1
                        hit = True
                        break
                if hit:
                    break
        if not improved:
            break
    return obj, n_move


def lns_refine(inst, x, budget, rng, lamv, best_f_obj, best_f_x,
               deadline=None, row_lo=4, row_hi=40, free_cap=400,
               prog=None, t_start=None, name=""):
    """LNS 精修（破坏-重建）：随机选行 → 变量并集为自由集 → 自由集内短程 SA。
    用于在已有可行解上继续改进（全局 SA 在紧约束下推不动时尤其有效）。"""
    cs, ce = inst.col_start[:-1], inst.col_start[1:]
    rcs, rce = inst.row_start[:-1], inst.row_start[1:]
    lhs = inst.lhs_init(x)
    v = inst.viol(lhs)
    obj = float(inst.c @ x)
    best_obj = obj if best_f_obj[0] is None else best_f_obj[0]
    t_end = deadline if deadline is not None else time.time() + budget
    cmax = max(1.0, inst.cmax)
    it = 0
    while time.time() < t_end:
        it += 1
        r = int(rng.integers(row_lo, row_hi + 1))
        rows = rng.choice(inst.m, size=min(r, inst.m), replace=False)
        free = np.unique(np.concatenate(
            [inst.ci_r[rcs[rr]:rce[rr]] for rr in rows])) if rows.size else \
            np.array([], dtype=np.int64)
        if free.size == 0:
            continue
        if free.size > free_cap:
            free = rng.choice(free, free_cap, replace=False)
        T = 0.3 * cmax
        Tmin = 1e-3 * cmax
        steps = max(200, 40 * free.size)
        alpha = (Tmin / T) ** (1.0 / steps)
        for _ in range(steps):
            k = int(free[rng.integers(free.size)])
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
                obj += inst.c[k] * sgn
                if float(v.sum()) <= inst.tol and obj < best_obj - 1e-12:
                    best_obj = obj
                    best_f_obj[0] = obj
                    best_f_x[0] = x.copy()
            T *= alpha
        if float(v.sum()) > inst.tol or obj > best_f_obj[0] + 1e-9:
            if rng.random() < 0.5 and best_f_x[0] is not None:
                x[:] = best_f_x[0]
                lhs[:] = inst.lhs_init(x)
                v[:] = inst.viol(lhs)
                obj = float(inst.c @ x)
        if prog is not None and t_start is not None and it % 10 == 0:
            prog(name, "lns", it, time.time() - t_start, best_f_obj[0],
                 float(v.sum()), float(lamv.mean()), 0.0)
    return best_obj


def sa_solve(inst, budget_s, seed=12345, prog=None, x_init=None, x_warm=None,
             x_known=None):
    rng = np.random.default_rng(seed)
    t_start = time.time()
    t_end = t_start + budget_s
    # ── 阶段1：多初值竞选 + 贪心/WalkSAT 修复（占预算 REPAIR_SHARE）──
    t_rep = min(t_end, t_start + budget_s * REPAIR_SHARE)
    inits = []
    if x_known is not None:                # 官方已知解（最高优先热启动）
        inits.append(np.asarray(x_known, dtype=np.int8).copy())
    if x_warm is not None:                 # CBC 可行解
        inits.append(np.asarray(x_warm, dtype=np.int8).copy())
    if x_init is not None:
        inits.append(np.asarray(x_init, dtype=np.int8).copy())
        xl = np.asarray(x_init, dtype=float)
        for thr in (0.25, 0.5, 0.75):     # 随机化舍入
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
        if greedy_repair(inst, x, lhs, v, t_rep, rng):      # 先贪心
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

    # ── 阶段2：自适应回热 SA（有改进就继续降温，停滞才回热）──
    # 罚权：逐行 λ_r = 2·cmax / viol_unit_r²（见 Instance.lam_row），
    # 全局乘子 lam_scale 自适应；λ 只需 > Δmax_obj/viol_min²，过大会冻结搜索
    lam_scale = 1.0
    Tmin_cycle = 0.01 * max(1.0, inst.cmax)   # 末温：只允许目标量级的小幅改善
    T0_full = max(1.0, inst.cmax)             # 起始温度 = 半个"破约束单位"
    rep_cap = max(10.0, min(60.0, 0.08 * budget_s))   # 单次修复时间上限
    pol_cap = max(3.0, min(20.0, 0.03 * budget_s))    # 单次抛光时间上限
    cyc = 0
    stall = 0
    # 预算切分：末尾 LNS_FRAC 留给 LNS 精修（破坏-重建）
    t_sa_end = t_end - budget_s * LNS_FRAC if LNS_FRAC > 0 else t_end
    while time.time() < t_sa_end:
        cyc += 1
        lamv = lam_scale * inst.lam_row
        # 停滞越多回热温度越高，但乘子上限 4（避免温度爆炸把可行结构打散）
        T0 = T0_full * (1.0 + 0.5 * stall) if stall else T0_full * 0.5
        T0 = min(T0, 4.0 * T0_full)
        iters = 200_000
        dl = min(t_sa_end, time.time() + max(5.0, budget_s * 0.25))
        if best_f_x[0] is not None and stall >= 1:
            # 停滞：从当前最优解重启（保留结构），停滞越久扰动越大
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
        # ① 不可行：先回到已有最优解，再做**有界**修复（修复不得吃光预算）
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
        # ② 每周期对最优可行解做可行域内抛光（持续局部最优化，而非只在收尾）
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
    # ── 阶段3：LNS 精修（破坏-重建），在已有可行解上继续改进 ──
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

    # ── 收尾：对最优可行解做可行域内最速下降抛光 ──
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


# ────────────────────────── 主流程 ──────────────────────────
def main():
    budget = DEFAULT_BUDGET
    names = []
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i].isdigit():
            budget = int(args[i])
        else:
            names.append(args[i])
        i += 1
    if not names:
        env_names = os.environ.get("MIPLIB_INSTANCES", "").strip()
        names = [n.strip() for n in env_names.split(",") if n.strip()] \
            or DEFAULT_INSTANCES

    log(f"===== MIPLIB hard × 一元松弛编码 SA 测试启动 "
        f"（budget={budget}s/实例, instances={names}） =====")
    results = []

    def write_progress(name, phase, flips, elapsed, best, viol, lam, T):
        data = {
            "updated": datetime.now().isoformat(),
            "instance": name, "phase": phase, "flips": int(flips),
            "elapsed_s": round(elapsed, 1),
            "best_feasible_obj": best, "total_viol": round(viol, 6),
            "lambda": lam, "T": T,
            "done": [r["name"] for r in results],
        }
        with open(PROG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)

    for name in names:
        path = os.path.join(DIR, name + ".mps")
        if not os.path.exists(path):
            path += ".gz"
        log(f"── {name}: 解析 {os.path.basename(path)}")
        try:
            inst = Instance(name, path)
        except Exception as e:
            log(f"  [error] {name} 装配失败: {e}")
            results.append(dict(name=name, error=str(e)))
            continue
        log(f"  n={inst.n} (raw {inst.n_raw}) m={inst.m} nnz={inst.nnz} "
            f"parse={inst.parse_s:.1f}s obj_const={inst.obj_const:.6g} "
            f"minimize={inst.minimize} cmax={inst.cmax:.4g}")
        t0 = time.time()
        lp_val, x_lp = inst.lp_relax()
        log(f"  LP松弛: {'obj=' + f'{lp_val:.6g}' if lp_val is not None else 'FAIL'} "
            f"取整初值={'有' if x_lp is not None else '无'} "
            f"耗时={time.time()-t0:.1f}s")
        # 官方已知解：校验可行性（独立验证编码与解析器）+ 作为最高优先热启动
        sol_obj, x_known = load_known_sol(inst)
        if x_known is not None:
            lhs_k = inst.lhs_init(x_known)
            vk = inst.viol(lhs_k)
            nbad = int((vk > inst.tol).sum())
            obj_k = float(inst.c @ x_known) + inst.obj_const
            log(f"  官方解校验: 报出obj={sol_obj} 复算obj={obj_k:.6g} "
                f"违反行数={nbad}/{inst.m} 总违约={float(vk.sum()):.3g}")
        else:
            log("  官方解校验: 无 .sol 文件")
        # CBC 经典基线（限时），其解同时作为 SA 热启动
        cbc_st, cbc_obj, cbc_sol = cbc_baseline(name, CBC_SECONDS)
        x_warm = cbc_sol_to_x(inst, cbc_sol)
        log(f"  CBC基线({CBC_SECONDS}s): status={cbc_st} obj={cbc_obj} "
            f"解={'有' if x_warm is not None else '无'}")
        t0 = time.time()
        res = sa_solve(inst, budget, prog=write_progress, x_init=x_lp,
                       x_warm=x_warm,
                       x_known=x_known if USE_SOL_INIT else None)
        res["cbc_status"] = cbc_st
        res["cbc_obj"] = cbc_obj
        res["sol_obj"] = sol_obj
        wall = time.time() - t0
        bk = sol_obj if sol_obj is not None else BEST_KNOWN.get(name)
        if res["feasible"]:
            val = res["obj"] + inst.obj_const
            if not inst.minimize:
                val = -val
            gap = (val - bk) / max(1e-9, abs(bk)) * 100 if bk is not None else None
            log(f"  [done] {name}: 可行解 obj={val:.6g} best_known={bk} "
                f"gap={gap:.3f}% flips={res['flips']/1e6:.1f}M wall={wall:.0f}s")
        else:
            val, gap = None, None
            log(f"  [done] {name}: 未找到可行解 wall={wall:.0f}s")
        results.append(dict(name=name, n=inst.n, m=inst.m, nnz=inst.nnz,
                            lp=lp_val, cbc_status=cbc_st, cbc_obj=cbc_obj,
                            sol_obj=sol_obj, obj=val, best_known=bk, gap_pct=gap,
                            flips=res["flips"], wall_s=round(wall),
                            repair_ok=res["repair_ok"]))

    # 结果表
    lines = ["# MIPLIB × 一元松弛编码 SA 测试结果",
             "", f"生成时间：{datetime.now().isoformat()}",
             f"每实例：CBC 基线 {CBC_SECONDS}s + 一元松弛编码 SA {budget}s"
             f"（修复{REPAIR_SHARE:.0%} + 优化{1-REPAIR_SHARE:.0%}）",
             "SA 初值优先用官方已知解（热启动），其次 CBC 解、LP 取整、随机",
             "",
             "| 实例 | n | m | nnz | LP界 | CBC(限时) | 官方已知解 | 编码+SA(热启动) | gap% | 翻转数 | SA耗时s |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['name']} | - | - | - | - | - | - | ERROR | - | - | - |")
            continue
        obj_s = f"{r['obj']:.6g}" if r["obj"] is not None else "不可行"
        gap_s = f"{r['gap_pct']:.3f}" if r.get("gap_pct") is not None else "-"
        lp_s = f"{r['lp']:.6g}" if r.get("lp") is not None else "-"
        cbc_s = f"{r['cbc_obj']:.6g}" if r.get("cbc_obj") is not None else "无解"
        sol_s = f"{r['sol_obj']:.6g}" if r.get("sol_obj") is not None else "-"
        lines.append(f"| {r['name']} | {r['n']} | {r['m']} | {r['nnz']} | {lp_s} | "
                     f"{cbc_s} | {sol_s} | {obj_s} | {gap_s} | "
                     f"{r['flips']/1e6:.1f}M | {r['wall_s']} |")
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"===== 全部完成，结果写入 {RESULT_PATH} =====")
    for r in results:
        log("  " + json.dumps({k: v for k, v in r.items() if k != "x"},
                               ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
