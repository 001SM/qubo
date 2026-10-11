# 超立方体下三角编码：相关工作检索报告

检索日期：2026-09-13
检索范围：arXiv / Nature / IEEE / ACM / EPJ QT / JOCO / INSPIRE / DBLP
检索词：violation indicator QUBO、cardinality constraint encoding、higher-order encoding、
exponential penalization、at-most-one QUBO、quadratization、slack-free penalty、
elementary symmetric polynomial QUBO

---

## 0. 结论速览

**你的思路** = 用超立方体下三角（严格单形）得到初等对称多项式
`e_{C+1}(x) = C(S, C+1)` 作为"Σx ≤ C"的**精确违反指示**，再二次化降到 QUBO。

**检索结论**：**没有找到完全相同的方案**。但有 **3 条高度相关的路线**，
其中 2 条与你高度重合但有关键差异。

| 相似度 | 工作 | 相同点 | 关键差异 |
|---|---|---|---|
| ★★★★★ | **Koch et al. 2025**（arXiv:2511.17545，EPJ QT 13:59, 2026） | **"violation indicator"** 概念；不加松弛变量；用高阶项直接编码约束；HUBO 框架 | 保留高次项（不做二次化）；未给出 `e_{C+1}` 的显式组合数形式；QAOA 上 λ 有限只是近似 |
| ★★★★☆ | **Kanatbekova et al. 2025**（arXiv:2509.08080） | 不用松弛变量；展开后天然是**初等对称多项式**结构（系数 (e^c−1)^k）；qubit 减少 BPP 57% / TSP 83% | **近似**（软惩罚，准确率 6%/21%），非精确；指数函数把各阶混合打包，没有"只留 C+1 阶" |
| ★★★☆☆ | **Codognet 2023**（GECCO'23, pp.2195-2202） | 专门研究 AMO（Σx ≤ 1）的 QUBO 编码；用 pairwise 乘积 | 只做 C=1 特例，未推广到一般 C；未讨论二次化 |
| ★★★☆☆ | **Glover/Kochenberger/Du 2019**（QUBO Tutorial, arXiv:1811.11538） | 罚项表里含 AMO 的 pairwise 罚项 | 教科书标准做法，止步于 C=1 |
| ★★☆☆☆ | **Montañez-Barrera et al. 2024**（QST 9(2)） | 无松弛变量（unbalanced penalization） | 近似；不是多项式/对称结构 |
| ★★☆☆☆ | **Ceselli & Premoli 2023**（Sci Rep 13:5628） | 系统对比多种编码 | 只比 binary / unary / linear，不含超立方体路线 |
| ★★☆☆☆ | **Warkentin 2026**（JLESC Workshop） | 改进基数约束编码 | 用 binary search + adders（电路式），不是多项式罚项 |

**你的差异化点（文献里没找到对应）**：
1. 把 AMO 的 `Σ_{i<j} x_i x_j` **显式推广到一般 C**：`e_{C+1} = C(S, C+1)`
2. 明确它是**精确**违反指示（Σx ≤ C 时恒为 0），而不是近似软惩罚
3. **加权推广**：`e_K^(a)(v) = Σ_{Σm_j=K, m_j≤a_j} [Π_j C(a_j, m_j)] · Π_j v_j`
4. **严格下三角 vs 完整下三角**的口径分析（`e_K = Σ(-1)^(K-j)C(K-1,j-1)h_j`）
5. 完整的**二次化管线**：把 HUBO 精确降到 QUBO（Koch 是保留 HUBO 不降）

---

## 1. 最接近的工作详解

### 1.1 Koch, Panahiyan, Mukherjee, Doetsch, Jaksch (2025/2026)

**Resource-Efficient Quantum Optimization via Higher-Order Encoding**
arXiv:2511.17545 [quant-ph]，发表于 EPJ Quantum Technology 13 (2026) 59

- **核心主张**：QUBO 编码因罚项而膨胀电路、增加 qubit 和门数；改用 **HUBO**
  （Higher-Order Unconstrained Binary Optimization）直接incorporate 约束，可减少资源需求
- **关键句**：*"Constraints can be enforced by adding λ times a **violation indicator** to
  the Hamiltonian, with λ large enough that the ground state is feasible"*
  —— 与你的"违反指示"是同一个概念
- 同时构建 QUBO 和 HUBO 哈密顿量
- 已有演示：蛋白质折叠、TSP
- 局限：作者自述 *"prior work has lacked a general framework"*（此前缺通用框架）；
  QAOA 上有限 λ 只能近似保证可行性，λ 靠经验调

**与你的关系**：概念最接近（violation indicator + 不用松弛变量），
但他们是**保留高次项**（HUBO），你是**二次化降到 QUBO**——
在只能跑 QUBO 的求解器（如迭代消元 SA、D-Wave）上，你的路线是必需的补充。

### 1.2 Kanatbekova, De Maio, Brandic (2025)

**Qubit-Efficient QUBO Formulation for Constrained Optimization Problems**
arXiv:2509.08080 [cs.ET]（TU Wien + Leicester）

- **广义指数惩罚框架**：用 `exp(c·Σx)` 类函数替代松弛变量
- 数学上：`exp(c Σx) = Π_i (1 + (e^c−1) x_i)`，展开后每项都是变量乘积，
  系数按**初等对称多项式** `e_k` 组织，系数为 `(e^c−1)^k`
  —— 也就是**把所有阶的乘积项打包进一个表达式**，从而不需要松弛变量
- 量化结果：qubit 减少 **BPP 57% / TSP 83%**；解质量"与经典可比"，
  但 BPP 8 qubit 下概率 6%、TSP 12 qubit 下准确率 21%
- **性质**：**近似软惩罚，不是精确编码**（摘要用概率/准确率表述，未声称精确）

**与你的关系**：数学结构同源（都是初等对称多项式），但：
- 他们**混合所有阶**（系数 (e^c−1)^k 全非零），你**只取 C+1 阶**
- 因此他们**不精确**，你**精确**（Σx ≤ C 时罚项恒为 0）

### 1.3 Codognet (2023)

**Encoding the At-Most-One Constraint for QUBO and Quantum Annealing: Experiments with the N-Queens problem**
GECCO'23 Companion, pp.2195-2202, DOI 10.1145/3583133.3596394

- 专门研究 **AMO（Σx ≤ 1）** 的 QUBO 编码，用 pairwise 乘积罚项
- 相关工作：Codognet, *Domain-Wall / Unary Encoding in QUBO for Permutation Problems*
  (QCE22)；Richoux/Baffier/Codognet, *Comparing Learned and Handmade one-hot QUBO Models*

**与你的关系**：这是你 C=1 甜点的**特例**，但没有推广到一般 C。

### 1.4 Glover, Kochenberger, Du (2019)

**A Tutorial on Formulating and Using QUBO Models**（arXiv:1811.11538）

- 提供"常用约束 → 罚项"对照表，含 AMO 的 `P·Σ_{i<j} x_i x_j`
- 是 QUBO 建模的标准教科书出处

---

## 2. 其他相关方向（供参考）

| 方向 | 代表工作 | 说明 |
|---|---|---|
| 二次化（扁平化） | Rosenberg 1975；Boros/Crama/Rodríguez-Heck 2020 (JOCO 39:687-707)；Anthony et al. 2016 (DAM 203:1-12) | 高次项 → 二次项的理论与紧致构造；**Dattani & Chau 2019**（4 变量函数只用 1 个辅助变量完美二次化） |
| 二次化最新 | Crama/Elloumi/Lambert/Rodríguez-Heck 2025 (JOCO 50:3) | 三阶段法：递归分解 → 二次化 → 凸化 |
| HUBO 求解 | Koch et al. 2025 | 保留高次项，直接上能处理多体交互的硬件 |
| one-hot 编码 | Nature 2026, s41598-026-55521-w | 更小的 one-hot 编码，对标 SAT 的 exactly-one 编码 |
| 基数约束电路式编码 | Warkentin 2026 (JLESC) | binary search + adders |
| slack-free 路线 | Montañez-Barrera et al. 2024 (QST 9(2))；arXiv:2504.12611 | 非平衡惩罚，近似 |
| 编码方案对比 | Ceselli & Premoli 2023 (Sci Rep 13:5628) | binary / unary / linear 三方对比 |
| QUDO/HOBO 综述 | arXiv:2508.01958 | 含 counting constraint 的多种表述 |

---

## 3. 结论与建议

### 3.1 定位

你的思路处于**"无松弛变量 + 精确罚项"**这个交叉点上：

```
        精确                近似
      ┌──────────────┬──────────────┐
无松  │  ★ 你的思路   │ Kanatbekova  │
弛变  │  (e_{C+1})    │ Montañez     │
量    ├──────────────┼──────────────┤
      │ 传统松弛变量  │   —          │
      │ (unary/binary)│              │
      └──────────────┴──────────────┘
```

- **左上格（精确 + 无松弛变量）目前基本是空的** —— 你的 `e_{C+1}` 构造正好落在这里
- Koch et al. 占了"无松弛变量 + 高次项"但没做二次化，也没给出精确的组合数形式

### 3.2 建议的写作/汇报角度

1. **定位**："首个把 AMO 的 pairwise 罚项显式推广到一般基数约束、且保持精确性的构造"
2. **对标**：与 Kanatbekova（指数惩罚，近似）做精度对比 —— 用你已有的小规模穷举协议
3. **理论支撑**：引 Anthony 2016 / Boros 2020 的二次化下界，说明何时必须引入辅助变量
4. **工程价值**：与 Koch 的 HUBO 互补 —— 他们面向多体交互硬件，你面向只吃 QUBO 的求解器

### 3.3 待补充检索

- 中文期刊（CNKI）与学位论文
- SAT 领域的基数约束编码（sequential counter / totalizer / sorting network）是否有对应 QUBO 版本
- "polynomial penalty cardinality" 的更早文献（1990s 伪布尔优化）
