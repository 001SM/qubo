# -*- coding: utf-8 -*-
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

if sys.platform == "win32":     # 强制 UTF-8 输出，避免被调用方按 GBK 误读
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")


def parse_mps_min(path):
    """独立最小 MPS 解析器：返回 (objsense, rows, rhs, ranges, obj, bounds, cols)"""
    opener = gzip.open if path.endswith(".gz") else open
    section = None
    row_type, row_order = {}, []
    obj_row = None
    cols = []
    colset = set()
    obj = {}
    rows = []           # (row_name, col_name, coef)
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

    # 读解（原始变量空间）
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

    # 独立复算每行活动值
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
        else:  # G
            lo = R
            hi = R + abs(rg) if rg else float("inf")
        viol = max(0.0, v - hi) + max(0.0, lo - v)
        if viol > a.tol:
            nbad += 1
            tot_viol += viol
            if viol > max_viol:
                max_viol, worst = viol, (rn, v, lo, hi)

    # 独立复算目标
    obj_val = sum(obj.get(c, 0.0) * xval.get(c, 0.0) for c in cols)
    # 变量界检查
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
