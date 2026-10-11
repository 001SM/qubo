# 一元松弛不等式编码（下三角装配）相关工作检索报告

检索日期：2026-09-12
检索范围：arXiv / Nature Scientific Reports / Springer / IOP / JOCO / DAM / Math Programming / 工具文档
检索词：unary slack QUBO inequality encoding、binary vs unary slack encoding、quadratization、exact penalty function、slack-free penalty、penalty weight tuning

---

## 0. 结论速览

我们的方法可拆成三个技术要素，**每一项在文献中都有对应工作，但没有一篇把它们合并成同一套流程**：

| 我们的要素 | 文献状况 | 最接近的对标 |
|---|---|---|
| 一元松弛（unary slack）编码不等式约束 | 标准做法，2014 起广泛使用 | Lucas 2014；Ceselli & Premoli 2023（系统对比） |
| 严格下三角装配（Q = diag(c²−2Cc) + 下三角 c_kc_l） | QUBO 教程的标准"展开平方 + 用 x²=x"流程，无独立命名 | Glover/Kochenberger/Du 2019 QUBO Tutorial |
| min_y 闭式消元（等价罚项 = max(0, viol)²） | 量子侧：slack-free / unbalanced penalization（2024-2025）；经典侧：精确罚函数（1960s-70s） | Montanez-Barrera et al. 2024；arXiv:2504.12611 |

**没有检索到**：把"一元松弛 + 下三角装配 + 闭式消元 + 谱宽归一化"作为一套完整方法论、并配等价性验证协议的论文。我们的高维验证三件套（矩阵-表达式逐点一致性 / 构造性零能 / 解析最小罚项）在文献中未见同类表述。

---

## 1. 直接对标文献

| # | 文献 | 年份/出处 | 核心内容 | 与我们方法的关系 |
|---|---|---|---|---|
| 1 | Ceselli & Premoli, *On good encodings for quantum annealer and digital optimization solvers* | Sci Rep 13:5628 (2023) | 系统对比 binary / unary / linear 三种松弛方案 | **最接近**：其 "unary" 即我们的一元松弛，其 "linear" 即我们的闭式消元思想 |
| 2 | Montanez-Barrera et al., *Unbalanced penalization* | Quantum Sci. Technol. 9(2) (2024), arXiv:2211.13914 | 不用松弛变量，用非平衡惩罚函数编码不等式 | 与我们的"消元"目标一致（去松弛变量），路径不同（非平衡 vs 闭式精确） |
| 3 | *Implementing Slack-Free Custom Penalty Function for QUBO on Gate-Based Quantum Computers* | arXiv:2504.12611 (2025) | 无松弛变量的自定义罚函数，面向门电路量子机 | 与我们的闭式罚项 max(0,viol)² 思路最接近的近期工作 |
| 4 | Boros, Crama, Rodríguez-Heck, *Compact quadratizations for pseudo-Boolean functions* | J. Comb. Optim. 39:687-707 (2020) | 二次化（quadratization）的辅助变量紧致构造 | 理论支撑：一元松弛是最朴素的二次化，辅助变量数有紧致构造 |
| 5 | Anthony, Boros, Crama, Gruber, *Quadratization of symmetric pseudo-Boolean functions* | Discrete Appl. Math. 203:1-12 (2016) | 对称函数二次化的辅助变量下界 Ω(√n) | 理论支撑：为何某些约束无法用少量辅助变量精确二次化 |
| 6 | Glover, Kochenberger, Du, *QUBO Tutorial* | arXiv:1811.11538 (2019) | QUBO 建模标准流程与罚项构造 | 下三角装配的教科书出处（展开平方 + x²=x） |
| 7 | *Lagrangian Duality in Quantum Optimization: Overcoming QUBO Limitations* | arXiv:2310.04542 | 用拉格朗日对偶绕过 QUBO 的约束局限 | 与 Ceselli 的 "linear" 方案同源，是"不引入松弛"的另一条路线 |
| 8 | *Decomposition of Large-Scale QUBO Problems for Quantum Annealers* | MDPI Eng. Proc. 134:29 | SVRS（松弛值域拆分）与 1W1H（one-weight-one-hot）编码 | 一元松弛在超大规模约束下的工程化改造（值域分块） |
| 9 | Lucas, *Ising formulations of many NP problems* | Front. Phys. 2:5 (2014) | 经典 Ising/QUBO 建模综述 | 一元松弛编码的原始出处之一 |

---

## 2. 关键文献要点

### 2.1 Ceselli & Premoli 2023（最直接的对照）

测试问题为基数约束二次背包（CQKP），容量约束 Σa_i x_i ≤ b：

| 方案 | 松弛变量个数 | 系数特征 | 缺点 |
|---|---|---|---|
| binary | ⌊log₂(Σa_i − b)⌋+1 | 二次项系数 c_m 很大 | 数值问题风险，需谨慎选罚权 |
| unary（= 我们的方案） | b 个（系数全 1） | 二次项系数小 | 变量数大、**高度退化** |
| linear（= 我们的消元思想） | 0 | 稀疏 | 非精确（需调 λ） |

作者原文关键判断：

- 一元编码："Unary coefficients result in lower value of quadratic coefficients; however, unary encodings requires a high number of additional variables and a **highly degenerate** formulation."（一元系数使二次项系数更低，但需要大量额外变量且表述高度退化）
- 二进制编码："the use of coefficients c_m results in high quadratic coefficients, hence requiring a careful choice of the penalty multiplier for the QUBO, and with **risk of numerical issues**."
- one-hot 编码被明确排除：同时继承一元（变量多、系数大）和二进制（需再松弛一个额外约束）的缺点。

**与我们结论的一致性/差异**：

| 维度 | Ceselli & Premoli | 我们 |
|---|---|---|
| 一元松弛的退化性 | 视为缺点（表示不唯一） | 视为无害（2026-09-11 结论：零能态多，单比特翻转即可在可行域内移动，测量后映射到同一 x） |
| 求解器侧重 | D-Wave 量子退火 | 经典 SA（x 空间） |
| 最优方案 | linear（线性罚，最稀疏，D-Wave 上最好） | 闭式消元版（等价于 linear 的精确化） |
| 罚权/能标 | 承认"最优 λ 需寻找"，未系统扫描 | 给出 P > Δmax 下界 + 除以 C² 归一化 |

### 2.2 Montanez-Barrera et al. 2024（非平衡惩罚）

- 不用任何松弛变量，用非平衡惩罚函数（约束未满足时惩罚更大）把最优解推向基态附近。
- 效果：BPP 可解规模从 11 项（松弛变量法）提升到 29 项。
- 局限：只能保证最优解"接近"基态（near the ground state），不是严格精确编码。
- **与我们的关系**：目标相同（去松弛变量），但我们的闭式 max(0,viol)² 是**严格精确**的（前提 P > Δmax），非平衡惩罚是近似的。

### 2.3 经典精确罚函数脉络（我们的数学根基）

- Courant (1943)：等式约束的二次罚 f(x) + t·g(x)²，t→∞ 收敛。
- Zangwill (1967)：精确罚函数 E(x,t) —— 存在 t₀ > 0，当 t ≥ t₀ 时无约束最优解即原问题最优解（**我们"罚权足够大则精确"的理论源头**）。
- 不等式约束的标准做法：罚 max(0, g_i(x))² 或 max(0, g_i(x))（ℓ2 / ℓ1 精确罚）。
- **对应关系**：我们的 min_y (Σa_i x_i ± |y| − C)² 闭式解 = max(0, viol)²，正是经典 ℓ2 精确罚函数在二元线性约束上的特例。文献里叫 exact penalty，我们叫"一元松弛的 min_y 闭式"。

### 2.4 二次化理论（辅助变量下界）

- Anthony/Boros/Crama/Gruber (2016)：对称函数二次化的辅助变量下界 Ω(√n)；某些函数无线性数量的二次化。
- Boros/Crama/Rodríguez-Heck (2020)：紧致二次化构造（JOCO 39:687-707）。
- **意义**：为"至少选一在 n≥3 时不存在二次精确罚项、必须引入松弛"这一结论提供了文献支撑（我们文档第 8 节的论断）。

### 2.5 工程侧参考

| 来源 | 内容 | 用途 |
|---|---|---|
| Glover/Kochenberger/Du, arXiv:1811.11538 | QUBO 建模标准流程；罚项构造规则 | 下三角装配的引用出处 |
| MQT QUBOMaker 文档（penalty weight tuning） | 罚权 λ 的分层设置策略 | 我们的 P > Δmax 下界可与之对照 |
| 玻色 Kaiwu SDK（自动确定惩罚系数） | 按"目标函数最大变化量 < 罚项变化量"估计罚系数 | 与我们的罚权定标思路一致，是国产工具链的实现 |
| QUBO++ 文档 | 原生支持 `relu(f,2)` = max(0,f)² 作为约束函数 | 说明"闭式违约罚"已进入工具层，但缺方法论总结 |
| arXiv:2503.03518 | MILP→QUBO 罚参数的构造式调优方法 | 与我们 MIPLIB 实验的罚权设定直接相关 |

---

## 3. 我们的差异点（可作为写作的创新性论述）

1. **三要素合并**：一元松弛 + 严格下三角装配 + min_y 闭式消元，作为一套完整流程（文献里分别出现，未合并）。
2. **经典/量子分工结论**：明确"经典求解器（SA/禁忌/局部搜索）应在 x 空间消元搜索；只有需要真正 QUBO 形式（量子退火/QAOA/D-Wave）时才引入 y 块，且必须做罚项/C² 归一化"。Ceselli 的 linear 方案隐含这个结论但未区分经典与量子的使用场景。
3. **退化性重新评价**：一元松弛的 C(b,v) 退化性，我们论证为**无害**（对应能隙均匀为 1·P，单比特翻转可在可行域内移动），文献普遍视为缺点。
4. **等价性验证协议**：高维无法穷举时，用「矩阵-表达式逐点一致性 + 构造性零能 + 解析最小罚项」三件套验证（26/26 PASS），文献未见同类验证方法。
5. **谱宽的实证量化**：完整 QUBO 谱宽 O(P·C²) 导致朴素 SA gap 39%（n=60 背包），消元后 gap=0；这是把"罚权过大会淹没目标"的定性说法变成可测的定量结论。

---

## 4. 建议引用结构（若撰写论文）

- **引言/相关工作**：Ising/QUBO 建模综述 [Lucas 2014]，QUBO 教程 [Glover 2019]，松弛编码对比 [Ceselli & Premoli 2023]，slack-free 路线 [Montanez-Barrera 2024；arXiv:2504.12611]
- **理论部分**：二次化下界 [Anthony 2016；Boros 2020]，精确罚函数 [Zangwill 1967]，罚权下界 [MQT/玻色/arXiv:2503.03518]
- **实验部分**：经典基线 CBC/Gurobi（见 MIPLIB 实验）；量子侧对比对象 D-Wave Advantage [Ceselli 2023]
- **差异化定位**：强调"经典求解器侧的消元实现 + 等价性验证协议 + 谱宽量化"

---

## 5. 检索局限与后续方向

- 未系统检索中文期刊（CNKI 类）与学位论文，可能存在同思路工作。
- 未检索 "unary encoding degeneracy" 的专门理论分析（是否有论文证明退化对 SA 的影响）。
- 建议补充检索词：`unary slack degeneracy QUBO spectral gap`、`constraint violation penalty simulated annealing knapsack exact`、`MIPLIB QUBO benchmark`。
- 与项目另一条线（V4/V5/V6MP 分区 SA / MAX-CUT）的相关工作检索尚未做，需要时另行检索。
