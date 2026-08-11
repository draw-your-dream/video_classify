#!/usr/bin/env python
"""E50 混合制记账器:规则否决集 V + 模型分数 → gn@95。

记账法(预注册冻结):被 V 拦截的 bad 直接计入召回;阈值 T 只需覆盖
剩余召回缺口(在未被否决的 bad 上取第 m 高分);
放行率 = 未被否决且 p<T 的可放行样本占比。
与原 gn() 的并列约定一致:p>=T 拦截,p<T 放行。
"""
from __future__ import annotations

import numpy as np


def gn_plain(p, y, rec=0.95):
    b = np.sort(p[y == 1])
    T = b[len(b) - int(np.ceil(rec * len(b)))]
    return float((p[y == 0] < T).mean())


def gn_hybrid(p, y, veto, rec=0.95, detail=False):
    """p: 模型分数; y: 1=bad; veto: bool 否决集(规则拦截)。"""
    p = np.asarray(p, float)
    y = np.asarray(y, int)
    veto = np.asarray(veto, bool)
    Nb = int(y.sum())
    need = int(np.ceil(rec * Nb))
    vb = int(veto[y == 1].sum())            # 规则拦下的 bad
    m = need - vb                            # 模型还须拦的 bad 数
    nv_bad = np.sort(p[(y == 1) & ~veto])[::-1]
    if m <= 0:
        T = np.inf
    elif m > len(nv_bad):
        T = -np.inf                          # 规则+模型都不够(不可能发生于 rec<=1)
    else:
        T = nv_bad[m - 1]
    rel_mask = (y == 0) & ~veto & (p < T)
    out = float(rel_mask.sum() / max(1, (y == 0).sum()))
    if detail:
        vg = int(veto[y == 0].sum())
        vg_below = int((veto & (y == 0) & (p < T)).sum())
        return out, dict(T=float(T), veto_bad=vb, veto_good=vg,
                         veto_good_below_T=vg_below, model_need=m)
    return out
