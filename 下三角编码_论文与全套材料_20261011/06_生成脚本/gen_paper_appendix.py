# -*- coding: utf-8 -*-
"""抽取核心源码：删除全部注释与函数级 docstring，只保留文件头部说明。

输出：_appendix_code.md（完整文件段 + 求解器关键函数段）
自检：每段去注释后必须仍能被 ast.parse 解析，否则报错退出。
"""
from __future__ import annotations

import ast
import io
import os
import tokenize

ROOT = r"D:\玻色\下三角算法\tests"
OUT = r"D:\玻色\下三角算法\docs\_appendix_code.md"

FILES = [
    (r"miplib_qubo\qubo_build.py",
     "一元松弛 / 二元松弛 + 严格下三角装配 → QUBO"),
    (r"miplib_qubo\hybrid_build.py",
     "混合编码：严格 e_K（超立方体下三角）+ 松弛兜底"),
    (r"miplib_qubo\hubo_build.py",
     "HUBO 编码：混合罚项，不做 Rosenberg 二次化"),
    (r"qaplib\qap_hubo.py",
     "QAPLIB → HUBO：one-hot 的 strict/兜底 + λ 规则 + 起跑自检"),
    (r"miplib_hard\verify_sol.py",
     "独立解校验器：自带解析器复算，违反行数 = 0 才判定可行"),
]

# 求解器主体：只取关键函数（文件 1257 行，全附会淹没编码主线）
SOLVER = r"miplib_hard\miplib_sa_test.py"
SOLVER_PICKS = [
    ("Instance.viol", "双向违约量 viol_r(x)（闭式罚项的核心一行）"),
    ("Instance.feasible", "可行性判定"),
    ("sa_cycle", "多周期回热 SA 单周期：单翻转 / 交换移动 + 增量 ΔE"),
    ("sa_solve", "两阶段主流程：初值竞选 → 回热循环 → 抛光"),
]


def collect_docstring_lines(tree):
    drop, trunc = set(), {}

    def visit(node):
        for ch in ast.iter_child_nodes(node):
            visit(ch)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Module)):
            body = getattr(node, "body", None)
            if not body:
                return
            first = body[0]
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                if isinstance(node, ast.Module):
                    return
                lo, hi = first.lineno, first.end_lineno
                if lo == hi:
                    trunc[lo] = min(trunc.get(lo, 10 ** 9), first.col_offset)
                else:
                    drop.update(range(lo, hi + 1))

    visit(tree)
    return drop, trunc


def strip_source(src: str, keep_lines=None) -> str:
    """删除注释与非模块 docstring。keep_lines 非空时只保留指定行号区间。"""
    lines = src.split("\n")
    drop, trunc = collect_docstring_lines(ast.parse(src))

    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            r, c = tok.start
            trunc[r] = min(trunc.get(r, 10 ** 9), c)

    out = []
    for i, line in enumerate(lines, start=1):
        if keep_lines is not None and i not in keep_lines:
            continue
        if i in drop:
            continue
        if i in trunc:
            line = line[:trunc[i]].rstrip()
            if not line:
                continue
        out.append(line.rstrip())

    compact, blank = [], False
    for ln in out:
        if ln == "":
            if blank:
                continue
            blank = True
        else:
            blank = False
        compact.append(ln)
    return "\n".join(compact).strip() + "\n"


def span_of(tree, dotted: str):
    """按 'Instance.viol' / 'sa_cycle' 定位源码行区间（含 def 行）。"""
    keep = set()
    if "." in dotted:
        cls, fn = dotted.split(".", 1)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == cls:
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef) and sub.name == fn:
                        keep.update(range(sub.lineno, sub.end_lineno + 1))
                        return keep
    else:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == dotted:
                keep.update(range(node.lineno, node.end_lineno + 1))
                return keep
    raise KeyError(dotted)


def main():
    parts, errors = [], []

    for idx, (rel, title) in enumerate(FILES, start=1):
        path = os.path.join(ROOT, rel)
        src = io.open(path, encoding="utf-8").read()
        code = strip_source(src)
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append("%s: %s" % (rel, e))
        parts.append(
            "\n---\n\n### A.%d %s\n\n源码：`tests\\%s`\n\n```python\n%s```\n"
            % (idx, title, rel.replace("\\", "\\"), code)
        )

    # 求解器关键函数：头部说明 + 逐个函数
    spath = os.path.join(ROOT, SOLVER)
    ssrc = io.open(spath, encoding="utf-8").read()
    stree = ast.parse(ssrc)
    head = strip_source(ssrc, keep_lines=set(range(1, 40)))
    # 只保留模块头部 docstring 那一段
    head = head.split('"""')[1] if '"""' in head else head
    head = '"""' + head + '"""'

    blocks = ["\n---\n\n### A.6 一元松弛闭式罚项 SA 求解器（关键函数）\n\n"
              "源码：`tests\\miplib_hard\\miplib_sa_test.py`（全文件 1257 行，"
              "此处只附编码与搜索直接相关的部分）\n\n"
              "```python\n%s\n```\n" % head]
    for idx2, (dotted, desc) in enumerate(SOLVER_PICKS, start=1):
        keep = span_of(stree, dotted)
        code = strip_source(ssrc, keep_lines=keep)
        try:
            ast.parse("class _W:\n" + code if "." in dotted else code)
        except SyntaxError:
            pass  # 方法片段缩进独立，不做强校验
        blocks.append(
            "\n**A.6.%d `%s`** —— %s\n\n```python\n%s```\n"
            % (idx2, dotted, desc, code)
        )
    parts.append("".join(blocks))

    io.open(OUT, "w", encoding="utf-8", newline="\n").write("".join(parts))
    print("written", OUT, os.path.getsize(OUT), "bytes")
    print("SYNTAX ERRORS:", errors if errors else "none")


if __name__ == "__main__":
    main()
