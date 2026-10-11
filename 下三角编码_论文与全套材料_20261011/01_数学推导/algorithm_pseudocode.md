# 一元松弛编码 + 混合 SA/LNS 求解器：算法伪代码

生成日期：2026-09-12
对应实现：`D:\玻色\下三角算法\tests\miplib_hard\miplib_sa_test.py`（核心）、`miplib_parallel.py`（并行调度）、`verify_sol.py`（独立校验）、`lns_improve.py`（LNS 改进器）

---

## 0. 总体结构

```
┌─────────────────────────────────────────────────────────────┐
│ ① 编码层：约束 → 罚项（一元松弛 + 下三角装配 + min_y 闭式消元）│
├─────────────────────────────────────────────────────────────┤
│ ② 预处理：MPS 解析 → 固定变量折叠 → 整数展开 → 行归一化 → CSR │
├─────────────────────────────────────────────────────────────┤
│ ③ 初值：LP 松弛（界 + 取整）+ CBC 限时（基线 + 热启动）        │
├─────────────────────────────────────────────────────────────┤
│ ④ 可行化：贪心修复 + WalkSAT（自适应噪声 / 停滞重启）          │
├─────────────────────────────────────────────────────────────┤
│ ⑤ 优化：多周期回热 SA（单翻转 + 交换移动，逐行 λ）             │
│         + 每周期可行域内抛光                                   │
├─────────────────────────────────────────────────────────────┤
│ ⑥ 精修（可选）：LNS 破坏-重建                                  │
├─────────────────────────────────────────────────────────────┤
│ ⑦ 独立校验：还原原始变量空间 → 独立解析器复算 → 违反行数=0 才报告│
└─────────────────────────────────────────────────────────────┘
```

---

## 1. 编码层：一元松弛 → 闭式罚项

```
输入：约束 r:  lb_r ≤ Σ_j a_rj·x_j ≤ ub_r     (x ∈ {0,1}^n)

# 1.1 一元松弛（unary slack）
#     引入 y 块（整数松弛量 s ∈ [0, C] 用 s 个二进制变量表示）
#     le:  P·(Σ a x + Σ y − C)²        ge:  P·(Σ a x − Σ y − C)²

# 1.2 min_y 闭式消元（一元松弛独有，二进制松弛做不到）
#     min_y (Σ a x ± |y| − C)²  =  max(0, viol)²
#     ⇒ y 块被解析消去，等价罚项直接写成违约量平方

viol_r(x) = max(0, lhs_r − ub_r) + max(0, lb_r − lhs_r)
E(x)      = c·x + Σ_r λ_r · viol_r(x)²

# 1.3 下三角装配（需要显式 QUBO 矩阵时）
#     E = Σ_k Q_kk·z_k + 2·Σ_{k<l} Q_kl·z_k·z_l + const
#     Q_kk = w·(a_k² + 2·const·a_k)      ← 对角（利用 z² = z）
#     Q_kl = w·2·a_k·a_l   (k < l)        ← 严格下三角，不重复计
```

### 1.4 两个定标（关键，错了会让搜索冻结或失稳）

```
# 行归一化（控制谱宽 O(C²)）
s_r = max( max_j |a_rj|, |lb_r|, |ub_r|, 1 )
a_rj ← a_rj / s_r ,  lb_r ← lb_r / s_r ,  ub_r ← ub_r / s_r

# 逐行罚权：只需 > Δmax_obj / viol_unit²，过大反而冻结搜索
viol_unit_r = min_j |a_rj|                      # 该行最小单翻转违约量
λ_r = min( 2·cmax / viol_unit_r² , 1e4·cmax )   # cmax = max_j |c_j|
```

---

## 2. 预处理（通用性）

```
parse_mps(path):
    ROWS / COLUMNS / RHS / RANGES / BOUNDS  → 行型(E/L/G)、系数、界、变量类型

Instance(parse 结果):
    # ① 固定变量折叠：x_j = v → 行 rhs 平移、目标常数累加
    for j in fixed:  lb_r −= a_rj·v ;  ub_r −= a_rj·v ;  obj_const += c_j·v

    # ② 整数变量二进制展开（使方法不限于纯 0/1）
    for j in 整数变量:
        width = hi_j − lo_j
        bits  = ⌈log2(width+1)⌉
        w = [1,2,4,...,2^(bits−2), width − (2^(bits−1) − 1)]   # 末位取余量
        x_j = lo_j + Σ_k w_k·b_k
        # 行：系数 a_rj·w_k 挂到新列；rhs −= a_rj·lo_j
        # 目标：常数 += c_j·lo_j；新列系数 = c_j·w_k

    # ③ 行归一化（见 1.4）
    # ④ CSR：列式（供单点翻转增量更新）+ 行式（供修复算子按行取变量）
```

---

## 3. 初值

```
lp_relax():
    松弛 x ∈ [0,1]，用 HiGHS 解 LP
    → (LP 界, 取整初值)     # 注意：系数矩阵必须用与行列下标同一排序的数组

cbc_baseline(path, 秒数):
    CBC 限时求解 → (状态, 目标值, 解)      # 既是基线，也作 SA 热启动
```

---

## 4. 可行化：贪心修复 + WalkSAT

```
greedy_repair(x, lhs, v, deadline, max_cand=64):
    按违约从大到小遍历违约行 r：
        cand = r 的变量（支撑 > max_cand 时随机抽样）
        k* = argmin_k Δ(Σ viol)        # 全支撑/抽样扫描，选使总违约下降最多的
        翻转 k*（若严格改善）
    重复至无改善或超时

walksat(x, lhs, v, deadline, noise=0.3):
    维护违约行集合（O(1) 增删/抽样）
    while 有违约行 and 未超时:
        r = 均匀随机取一个违约行
        cand = r 的变量（抽样 ≤ 24）
        k* = argmin_k Δ(Σ viol)
        k = k*（概率 1−noise） 或 随机候选（概率 noise）
        翻转 k，更新 lhs / v / 违约行集合
        if 总违约下降: noise ← max(0.15, noise−0.05)
        elif 停滞 4000 步: noise ← min(0.7, noise+0.1)
        if 停滞 20000 步: 从 best_x 扰动 n/25 个变量重启
```

---

## 5. 优化：多周期回热 SA

```
sa_solve(inst, budget):
    # ── 阶段1：多初值竞选（45% 预算）──
    inits = [官方解?, CBC解?, LP取整, LP随机化舍入×3, 全1, 全0, 随机]
    for init in inits:
        if greedy_repair(init) 成功: 得到可行解; break
        else: walksat(init)
        # 全程跟踪"最好违约量"的解，供阶段2回退

    # ── 阶段2：多周期回热 SA（55% 预算）──
    lam_scale = 1 ;  stall = 0 ;  T0_full = cmax ;  Tmin = 0.01·cmax
    while 未超时:
        lamv = lam_scale · λ_row                 # 逐行罚权
        T0 = min( T0_full·(1 + 0.5·stall), 4·T0_full )   # 停滞越久越热，但封顶
        if 有可行解 and stall ≥ 1:
            x ← best_x                           # 从最优解重启
            if stall ≥ 3: 随机翻转 n/(20·stall) 个变量   # 扰动保持多样性

        sa_cycle(T0 → Tmin, 步数 200k):
            维护 ones/zeros 集合（O(1) 抽样）
            for step in 1..200k:
                if rand < 0.35 and ones 和 zeros 非空:
                    # 交换移动：1→0 与 0→1 同时翻转（紧约束下唯一可行的大范围移动）
                    k1 ∈ ones, k2 ∈ zeros
                    ΔE = (c_k2 − c_k1) + Σ_{行∈col(k1)∪col(k2)} λ_r·Δviol_r²
                else:
                    # 单比特翻转
                    k ∈ [n]
                    ΔE = c_k·(1−2x_k) + Σ_{行∈col(k)} λ_r·Δviol_r²
                接受 if ΔE ≤ 0 or rand < exp(−ΔE / T)
                更新 x / lhs / v / obj（列式 CSR，O(deg) 增量）
                if 可行 and obj < best: best ← (obj, x)
                T ← T · α                          # α = (Tmin/T0)^(1/200k)

        # 周期后处理
        if 不可行:
            x ← best_x 或 best_repair_x            # 先回退，再从被打散的点硬修是浪费
            有界修复：greedy_repair / walksat，限时 ≤ min(60s, 8%·预算)
            仍不可行 → lam_scale ← lam_scale · 10  # 加压强
        else:
            每周期抛光：polish(best_x)              # 可行域内局部最优化
    收尾：再抛光一次
    return (best_obj, best_x, 统计量)

polish(x):      # 只接受"既降低目标、又保持可行"的移动
    repeat:
        for k in 按目标增益排序的变量: 若翻转后仍可行且降目标 → 翻转
        for 随机抽样的行 r:           若行内交换后仍可行且降目标 → 交换
    until 无改善
```

---

## 6. LNS 精修（在已有可行解上继续改进）

```
lns(x, budget):
    while 未超时:
        # ① 破坏：随机选 r ∈ [4, 40] 行，取变量并集作自由集（限幅 400）
        free = ∪_{r∈rows} vars(r)
        # ② 重建：只在 free 上做短程 SA（其他变量冻结）
        T = 0.3·cmax → 1e-3·cmax，步数 = 40·|free|
            单比特翻转，ΔE 同 SA，罚项用逐行 λ
        # ③ 精英保留：更优可行解则更新 best；否则 50% 概率回退到 best
```

---

## 7. 独立校验（先验证，再下结论）

```
report(解 x):
    vals = to_raw(x)          # 还原到原始变量空间：
                              #   固定变量折叠 → 补回固定值
                              #   整数展开 → x_j = lo_j + Σ w_k·b_k
    write_sol(vals)           # MIPLIB .sol 格式："=obj= v" + "变量名 值"
    verify_sol(mps, sol):     # 完全独立的解析器 + 独立复算（不复用求解器代码）
        逐行重算 lhs = Σ a_rj·x_j（缺失变量按 0）
        检查 [lb_r, ub_r]（含 RANGES）→ 违反行数 / 最大违约 / 总违约
        检查变量界 → 越界变量数
        独立复算目标值，与解文件声明值比对
    if 违反行数 > 0: 该解不是解，目标值无意义，不报告 gap
```

---

## 8. 关键设计取舍（踩过的坑）

| 设计点 | 错误做法 | 正确做法 | 后果 |
|--------|---------|---------|------|
| 罚权 λ | 1000·cmax | 2·cmax / viol_unit² | λ 过大会让**所有移动被拒**，SA 完全冻结（实测 60s 零接受） |
| 温度定标 | T0 随 λ 缩放 | T0 ≈ 半个"破约束单位"，Tmin = 0.01·cmax | 温度与罚权脱节 → 可行结构被打散 |
| 回热温度 | 停滞乘子无上界 | 乘子封顶 4 | 无界时温度爆炸，viol 冲到数百 |
| 修复预算 | 按比例给 35% 预算 | 限时 ≤ min(60s, 8%)，且先回退再修复 | 修复吃光预算，1800s 只跑 5 个周期 |
| 移动类型 | 只有单比特翻转 | + 交换移动（1↔0 同时翻） | 紧约束下单翻转几乎全被拒 |
| 抛光时机 | 只在收尾跑一次 | 每周期都跑 | 局部优化能力浪费 |
| LP 系数矩阵 | 用不同排序的数组混搭 | 与行列下标同一排序 | 系数错位 → LP 界错 119 倍 |
| 并行 worker | 运行中编辑核心模块 | 编辑前后不动在跑的模块 | spawn 子进程重新导入中间态 → 崩溃 |
