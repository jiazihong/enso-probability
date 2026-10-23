#!/usr/bin/env python3
"""
ENSO 概率分析工具 (ENSO Probability Tool)
==========================================
基于 NOAA CPC 公开发布的 ONI（Oceanic Niño Index, ERSSTv5）历史序列，
估计未来 3/6/9/12 个月"厄尔尼诺 / 中性 / 拉尼娜"三种状态的出现概率。

三个估计量（互相交叉验证）：
  1. 状态转移模型：在当前状态 + 相位年龄条件下做马尔可夫转移统计，
     叠加 Dirichlet 先验做平滑；
  2. 历史相似型模型：以最近 window 个月的 ONI 轨迹为查询向量，
     在历史中做高斯核加权的近邻搜索；
  3. 组合概率：按各模型的历史回测 Brier 技能分自适应加权。

状态判定沿用 NOAA 官方规则：ONI 连续 5 个重叠三月 >= +0.5 为厄尔尼诺，
<= -0.5 为拉尼娜，其余为中性。

仅依赖 Python 标准库，无第三方依赖。内置自测（selftest）：
合成 AR(1) 序列的参数回收检验、蒙特卡洛对照、符号对称性检验、边界条件测试。

用法：
    python enso_prob.py update                 # 抓取并缓存最新 ONI 数据
    python enso_prob.py forecast               # 输出概率（默认 3/6/9/12 个月）
    python enso_prob.py forecast --json        # 同时输出 output/forecast.json
    python enso_prob.py forecast --chart       # 附带最近 72 个月 ONI 柱状图
    python enso_prob.py backtest               # 查看各模型历史技能
    python enso_prob.py report                 # 生成可视化 HTML 报告
    python enso_prob.py selftest               # 运行内置自测
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import unicodedata
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------

DATA_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_FILE = DATA_DIR / "oni_cache.csv"
OUT_DIR = ROOT / "output"
CACHE_DIR = ROOT / ".cache"

SEASONS = ("DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ",
           "JJA", "JAS", "ASO", "SON", "OND", "NDJ")
SEASON_IDX = {s: i for i, s in enumerate(SEASONS)}

LA_NINA, NEUTRAL, EL_NINO = -1, 0, 1
STATE_LABELS = {EL_NINO: "厄尔尼诺", NEUTRAL: "中性", LA_NINA: "拉尼娜"}
STATE_LABELS_EN = {EL_NINO: "El Nino", NEUTRAL: "Neutral", LA_NINA: "La Nina"}
STATE_ORDER = (EL_NINO, NEUTRAL, LA_NINA)

ONI_THRESHOLD = 0.5
EVENT_WINDOW = 5

# --------------------------------------------------------------------------
# 数据结构与解析
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Obs:
    year: int
    season: str
    anom: float

    @property
    def label(self) -> str:
        return f"{self.year} {self.season}"

    @property
    def month_index(self) -> int:
        return self.year * 12 + SEASON_IDX[self.season]

def parse_oni_text(text: str) -> List[Obs]:
    """解析 ONI ASCII 表（列：SEAS YR TOTAL ANOM），忽略注释与表头。"""
    obs: List[Obs] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 4:
            continue
        sea, yr, _total, anom = parts
        if sea not in SEASON_IDX:
            continue
        try:
            obs.append(Obs(int(yr), sea, float(anom)))
        except ValueError:
            continue
    obs.sort(key=lambda o: o.month_index)
    return obs

def fetch_oni_text(timeout: float = 25.0) -> str:
    req = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "enso-probability/1.0 (+https://github.com/jiazihong/project)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")

def save_cache(text: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = f"# fetched={stamp} source={DATA_URL}\n"
    CACHE_FILE.write_text(header + text, encoding="utf-8")

def load_obs(refresh: bool = False, data_file: Optional[str] = None,
             allow_network: bool = True) -> Tuple[List[Obs], str]:
    """优先读本地缓存；refresh 或缓存缺失时联网抓取；联网失败自动回退缓存。"""
    if data_file:
        p = Path(data_file)
        if not p.exists():
            raise SystemExit(f"数据文件不存在: {p}")
        return parse_oni_text(p.read_text(encoding="utf-8", errors="replace")), f"文件 {p}"

    if CACHE_FILE.exists() and not refresh:
        text = CACHE_FILE.read_text(encoding="utf-8", errors="replace")
        obs = parse_oni_text(text)
        if obs:
            return obs, f"本地缓存 {CACHE_FILE}（可用 update 刷新）"

    if allow_network:
        try:
            text = fetch_oni_text()
            obs = parse_oni_text(text)
            if len(obs) > 100:
                save_cache(text)
                return obs, "NOAA CPC 在线数据（已写入缓存）"
            print("[警告] 在线数据条数异常，未写入缓存。", file=sys.stderr)
        except Exception as exc:
            print(f"[警告] 在线抓取失败：{exc}", file=sys.stderr)

    if CACHE_FILE.exists():
        text = CACHE_FILE.read_text(encoding="utf-8", errors="replace")
        obs = parse_oni_text(text)
        if obs:
            print("[警告] 使用可能过期的本地缓存。", file=sys.stderr)
            return obs, f"本地缓存 {CACHE_FILE}（可能过期）"

    raise SystemExit(
        "没有可用的 ONI 数据。请联网后运行 `python enso_prob.py update`，"
        "或用 --data-file 指定本地 oni.ascii.txt。"
    )

# --------------------------------------------------------------------------
# 基础统计
# --------------------------------------------------------------------------

def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)

def _stdev(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def classify_states(anoms: Sequence[Optional[float]],
                    window: int = EVENT_WINDOW) -> List[Optional[int]]:
    """按 NOAA 规则逐月判定状态：连续 window 个三月 >= +0.5 / <= -0.5。"""
    n = len(anoms)
    out: List[Optional[int]] = [None] * n
    for i in range(window - 1, n):
        chunk = anoms[i - window + 1: i + 1]
        if any(v is None for v in chunk):
            continue
        if min(chunk) >= ONI_THRESHOLD:
            out[i] = EL_NINO
        elif max(chunk) <= -ONI_THRESHOLD:
            out[i] = LA_NINA
        else:
            out[i] = NEUTRAL
    return out

def phase_ages(states: Sequence[Optional[int]]) -> List[int]:
    """每个时点所处状态的已持续月数（O(n) 预计算）。"""
    ages = [0] * len(states)
    for i, s in enumerate(states):
        if s is None:
            ages[i] = 0
        elif i > 0 and states[i - 1] == s:
            ages[i] = ages[i - 1] + 1
        else:
            ages[i] = 1
    return ages

def _age_bucket(age: int) -> int:
    if age <= 3:
        return 0
    if age <= 6:
        return 1
    if age <= 9:
        return 2
    return 3

def _brier(p: Sequence[float], state: int) -> float:
    y = [0.0, 0.0, 0.0]
    y[state + 1] = 1.0
    return sum((p[i] - y[i]) ** 2 for i in range(3)) / 3.0

# --------------------------------------------------------------------------
# 模型 1：状态转移（含相位年龄分层 + Dirichlet 先验）
# --------------------------------------------------------------------------

def transition_distribution(states: Sequence[Optional[int]],
                            ages: Sequence[int],
                            t_now: int,
                            horizon: int,
                            prior_strength: float = 4.0,
                            age_boost: float = 0.3,
                            season: Optional[int] = None
                            ) -> Tuple[List[float], List[float], float]:
    """返回 (三状态概率, Dirichlet 后验参数, 有效转移样本量)。

    age_boost=1.0 表示不使用相位年龄分层；<1 表示非同龄样本降权。
    season 非 None 时仅统计起报月相相同的样本（0=DJF, ..., 11=NDJ），
    用于按日历月分层以缓解"春季预报障碍"。
    """
    cur = states[t_now]
    counts = [0.0, 0.0, 0.0]
    marg = [0.0, 0.0, 0.0]
    bucket_now = _age_bucket(ages[t_now]) if cur is not None else -1

    n_pairs = len(states) - horizon
    for t in range(n_pairs):
        if t + horizon > t_now:
            break
        if season is not None and t % 12 != season:
            continue
        s0, s1 = states[t], states[t + horizon]
        if s0 is None or s1 is None:
            continue
        marg[s1 + 1] += 1.0
        if cur is None or s0 != cur:
            continue
        counts[s1 + 1] += 1.0 if _age_bucket(ages[t]) == bucket_now else age_boost

    total_m = sum(marg)
    marg_p = [m / total_m for m in marg] if total_m > 0 else [1.0 / 3] * 3
    prior = [prior_strength * x + 1e-6 for x in marg_p]

    posterior = [counts[i] + prior[i] for i in range(3)]
    denom = sum(posterior)
    p = [x / denom for x in posterior]
    return p, posterior, sum(counts)

# --------------------------------------------------------------------------
# 模型 2：历史相似型（轨迹近邻 + 高斯核）
# --------------------------------------------------------------------------

def analog_distribution(anoms: Sequence[Optional[float]],
                        states: Sequence[Optional[int]],
                        t_now: int,
                        horizon: int,
                        window: int = 6,
                        min_candidates: int = 20,
                        bw_floor: float = 0.25,
                        season: Optional[int] = None
                        ) -> Optional[Tuple[List[float], List[float], float]]:
    """返回 (三状态概率, 各近邻权重, Kish 有效样本量)；样本不足时返回 None。

    season 非 None 时仅使用起报月相相同的近邻（按月分层）。
    """
    n = len(states)
    if t_now - window + 1 < 0:
        return None
    q = anoms[t_now - window + 1: t_now + 1]
    if any(v is None for v in q):
        return None

    dists: List[float] = []
    outs: List[int] = []
    for t in range(window - 1, n - horizon):
        if t + horizon > t_now:
            break
        if season is not None and t % 12 != season:
            continue
        w = anoms[t - window + 1: t + 1]
        if any(v is None for v in w):
            continue
        s1 = states[t + horizon]
        if s1 is None:
            continue
        d2 = sum((a - b) ** 2 for a, b in zip(q, w)) / window
        dists.append(math.sqrt(d2))
        outs.append(s1)

    if len(dists) < min_candidates:
        return None

    bw = max(bw_floor, 1.06 * _stdev(dists) * len(dists) ** (-0.2))
    weights = [math.exp(-0.5 * (d / bw) ** 2) for d in dists]

    acc = [0.0, 0.0, 0.0]
    for wgt, s in zip(weights, outs):
        acc[s + 1] += wgt
    ssum = sum(acc)
    p = [a / ssum for a in acc] if ssum > 0 else [1.0 / 3] * 3

    n_eff = (sum(weights) ** 2) / sum(w * w for w in weights)
    return p, weights, n_eff

# --------------------------------------------------------------------------
# 进阶模块：近邻透明化 / 强度预测 / 月度收缩 / 技能热力图 / 概率校准
# --------------------------------------------------------------------------

def _silverman_bw(dists: Sequence[float], floor: float = 0.25) -> float:
    return max(floor, 1.06 * _stdev(dists) * len(dists) ** (-0.2))

def _neighbors(anoms: Sequence[Optional[float]], t_now: int,
               window: int, horizon: int, season: Optional[int] = None
               ) -> List[Tuple[int, float, Optional[float]]]:
    """返回 (t, 距离, h 个月后 ONI 值) 列表，按距离升序。t+h 恒 <= t_now。"""
    q = anoms[t_now - window + 1: t_now + 1]
    if any(v is None for v in q):
        return []
    out = []
    n = len(anoms)
    for t in range(window - 1, n - horizon):
        if t + horizon > t_now:
            break
        if season is not None and t % 12 != season:
            continue
        w = anoms[t - window + 1: t + 1]
        if any(v is None for v in w):
            continue
        d2 = sum((a - b) ** 2 for a, b in zip(q, w)) / window
        out.append((t, math.sqrt(d2), anoms[t + horizon]))
    out.sort(key=lambda x: x[1])
    return out

def top_analogs(obs: Sequence[Obs], anoms: Sequence[Optional[float]],
                states: Sequence[Optional[int]], t_now: int,
                horizon: int, window: int = 6, k: int = 5) -> List[dict]:
    """最相似历史年份：当前轨迹的近邻及其 h 个月后的实际走向。"""
    nn = _neighbors(anoms, t_now, window, horizon)
    result = []
    for t, d, v in nn[:k]:
        idx = states[t + horizon] if t + horizon < len(states) else None
        result.append({
            "label": obs[t].label,
            "start": obs[t].label,
            "target": obs[t + horizon].label if t + horizon < len(obs) else "",
            "distance": d,
            "target_oni": v,
            "state": idx,
        })
    return result

def oni_point_forecast(anoms: Sequence[Optional[float]], t_now: int,
                       horizon: int, window: int = 6, k_nearest: int = 40
                       ) -> Optional[dict]:
    """相似型加权的未来 ONI 点预测（加权均值 + 加权 5/95 分位）。"""
    nn = _neighbors(anoms, t_now, window, horizon)[:k_nearest]
    if len(nn) < 12:
        return None
    dists = [d for _, d, _ in nn]
    bw = _silverman_bw(dists)
    weights = [math.exp(-0.5 * (d / bw) ** 2) for d in dists]
    pairs = sorted([(v, w) for (_, _, v), w in zip(nn, weights) if v is not None],
                   key=lambda x: x[0])
    if not pairs:
        return None
    vals = [v for v, _ in pairs]
    ws = [w for _, w in pairs]
    wsum = sum(ws)
    mean = sum(v * w for v, w in pairs) / wsum
    var = sum(w * (v - mean) ** 2 for v, w in pairs) / wsum
    sd = math.sqrt(var)

    def wquantile(qq):
        target = qq * wsum
        acc = 0.0
        for v, w in pairs:
            acc += w
            if acc >= target:
                return v
        return vals[-1]

    return {
        "mean": mean,
        "sd": sd,
        "q05": wquantile(0.05),
        "q95": wquantile(0.95),
        "n_eff": wsum ** 2 / sum(w * w for w in ws),
        "n": len(pairs),
    }

def monthly_shrunk_components(anoms: Sequence[Optional[float]],
                              states: Sequence[Optional[int]],
                              ages: Sequence[int],
                              t_now: int, horizon: int,
                              k_lambda: float = 24.0) -> dict:
    """按起报月分层 + 经验贝叶斯收缩的最终组件。

    对转移模型与相似型：分别估计"全历史 pooled"与"同月起报 monthly"两个版本，
    按月度有效样本量收缩混合 p = λ·p_month + (1-λ)·p_pooled，λ = n/(n+k)。
    返回 dict(p_trans, alpha, p_analog, n_analog, lam, note)，供组合预测与区间抽样使用。
    """
    season = t_now % 12

    pm_all, am_all, nm_all = transition_distribution(states, ages, t_now, horizon)
    pm_mo, am_mo, nm_mo = transition_distribution(
        states, ages, t_now, horizon, season=season)
    lam_t = nm_mo / (nm_mo + k_lambda) if nm_mo > 0 else 0.0
    p_trans = [lam_t * pm_mo[i] + (1 - lam_t) * pm_all[i] for i in range(3)]

    an_all = analog_distribution(anoms, states, t_now, horizon)
    an_mo = analog_distribution(anoms, states, t_now, horizon, season=season)
    if an_all is None:
        p_analog, an_weights, n_eff, lam_a = None, None, 0.0, 0.0
    else:
        pa_all, wa_all, neff_all = an_all
        lam_a = 0.0
        p_analog = list(pa_all)
        an_weights = wa_all
        n_eff = neff_all
        if an_mo is not None:
            pa_mo, wa_mo, neff_mo = an_mo
            lam_a = neff_mo / (neff_mo + k_lambda)
            p_analog = [lam_a * pa_mo[i] + (1 - lam_a) * pa_all[i] for i in range(3)]
            an_weights = wa_mo
            n_eff = lam_a * neff_mo + (1 - lam_a) * neff_all

    neff_final = max(12.0, (nm_all + n_eff * 0.5))
    alpha = [max(p_trans[i] * neff_final, 1e-6) for i in range(3)]

    return {
        "p_trans": p_trans,
        "alpha": alpha,
        "p_analog": p_analog,
        "analog_weights": an_weights,
        "n_analog_eff": n_eff if p_analog is not None else 0.0,
        "lam": (lam_t, lam_a),
        "n_month_trans": nm_mo,
    }

def skill_heatmap(anoms: Sequence[Optional[float]],
                  states: Sequence[Optional[int]],
                  ages: Sequence[int],
                  horizons: Sequence[int] = (3, 6, 9, 12),
                  step: int = 3, min_train: int = 240) -> dict:
    """技能热力图：起报月（12 个重叠三月）× 提前期 的月度收缩组合模型技能分。"""
    heat = {m: {} for m in range(12)}
    for m in range(12):
        for h in horizons:
            bs_m, bs_c = [], []
            for t in range(min_train, len(states) - h, step):
                if t % 12 != m:
                    continue
                truth = states[t + h]
                if truth is None or states[t] is None:
                    continue
                comp = monthly_shrunk_components(anoms, states, ages, t, h)
                p_mix = _mix_p(comp)
                bs_m.append(_brier(p_mix, truth))
                bs_c.append(_brier(climatology_distribution(states, t, h), truth))
            m_b = _mean(bs_m) if bs_m else None
            c_b = _mean(bs_c) if bs_c else None
            heat[m][h] = (1 - m_b / c_b) if (m_b is not None and c_b) else None
    return heat

def _mix_p(comp: dict) -> List[float]:
    """按月收缩组件按技能权组合（热力图/校准内部使用等权近似）。"""
    if comp["p_analog"] is None:
        return list(comp["p_trans"])
    return [0.5 * comp["p_trans"][i] + 0.5 * comp["p_analog"][i] for i in range(3)]

def compute_calibration(anoms: Sequence[Optional[float]],
                        states: Sequence[Optional[int]],
                        ages: Sequence[int],
                        horizon: int,
                        model: str = "mix",
                        step: int = 3, min_train: int = 240) -> List[dict]:
    """可靠性曲线桶：把预测概率按月收缩组合分桶，统计桶内实际频率。"""
    edges = [0.0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0]
    bins = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    buckets = [{"lo": lo, "hi": hi, "pred": [], "obs": []} for lo, hi in bins]

    for t in range(min_train, len(states) - horizon, step):
        truth = states[t + horizon]
        if truth is None or states[t] is None:
            continue
        comp = monthly_shrunk_components(anoms, states, ages, t, horizon)
        if model == "mix":
            p = _mix_p(comp)
        elif model == "transition":
            p = comp["p_trans"]
        else:
            continue
        pt = p[truth + 1]
        for b in buckets:
            if b["lo"] <= pt < b["hi"] or (pt >= 1.0 and b is buckets[-1]):
                b["pred"].append(pt)
                b["obs"].append(1.0)
                break
    out = []
    for b in buckets:
        n = len(b["pred"])
        out.append({
            "lo": b["lo"], "hi": b["hi"], "n": n,
            "mean_pred": _mean(b["pred"]) if n else None,
            "freq_obs": _mean(b["obs"]) if n else None,
        })
    return out

# --------------------------------------------------------------------------
# 气候态基线
# --------------------------------------------------------------------------

def climatology_distribution(states: Sequence[Optional[int]],
                             t_now: int,
                             horizon: int,
                             years: int = 30) -> List[float]:
    """近 N 年内、以 t_now 为上限的无条件状态频率（回测基线）。"""
    start = max(0, t_now - years * 12)
    counts = [0.0, 0.0, 0.0]
    for t in range(start, len(states) - horizon):
        if t + horizon > t_now:
            break
        s1 = states[t + horizon]
        if s1 is None:
            continue
        counts[s1 + 1] += 1.0
    total = sum(counts)
    if total == 0:
        return [1.0 / 3] * 3
    return [c / total for c in counts]

# --------------------------------------------------------------------------
# 不确定性：Dirichlet / 近邻 bootstrap 混合抽样
# --------------------------------------------------------------------------

def _sample_dirichlet(alpha: Sequence[float], rng: random.Random) -> List[float]:
    g = [rng.gammavariate(max(a, 1e-9), 1.0) for a in alpha]
    s = sum(g)
    return [x / s for x in g] if s > 0 else [1.0 / 3] * 3

def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

def simulate_intervals(p_analog: Optional[Sequence[float]],
                       analog_n_eff: float,
                       alpha: Sequence[float],
                       blend: Tuple[float, float],
                       n_sim: int = 1500,
                       seed: int = 20240101) -> List[Tuple[float, float]]:
    """返回每个状态的 5%/95% 分位区间。

    不确定性的对内自洽来源：
      - 转移模型：Dirichlet 后验（参数 = 转移后验 alpha）
      - 历史相似型：以有效采样量等化为 Dirichlet 展开的邻近分布后验
    每个模拟里对两模型分别抽样一次 Dirichlet，再按 blend 线性组合。
    这样区间中心与模型点估计一致，且不破坏三状态归一化。
    """
    rng = random.Random(seed)
    samples = [[], [], []]

    analog_alpha = None
    if p_analog is not None and analog_n_eff >= 5:
        analog_alpha = [max(analog_n_eff * x, 1e-6) for x in p_analog]

    for _ in range(n_sim):
        d_trans = _sample_dirichlet(alpha, rng)
        if analog_alpha is not None and blend[1] > 0:
            d_analog = _sample_dirichlet(analog_alpha, rng)
            s = [blend[0] * d_trans[i] + blend[1] * d_analog[i] for i in range(3)]
        else:
            s = d_trans
        for i in range(3):
            samples[i].append(s[i])

    out = []
    for i in range(3):
        vals = sorted(samples[i])
        out.append((_quantile(vals, 0.05), _quantile(vals, 0.95)))
    return out

# --------------------------------------------------------------------------
# 组合预测
# --------------------------------------------------------------------------

def blend_forecast(p_trans: Sequence[float],
                   analog: Optional[Tuple[List[float], List[float], float]],
                   weights: Optional[Tuple[float, float]] = None,
                   simulate: bool = True,
                   alpha: Optional[Sequence[float]] = None,
                   n_sim: int = 1500) -> dict:
    if analog is None:
        blend = (1.0, 0.0)
        p = list(p_trans)
        p_analog = None
        n_eff = 0.0
    else:
        p_analog, _a_weights, n_eff = analog
        blend = weights or (0.5, 0.5)
        s = blend[0] + blend[1]
        blend = (blend[0] / s, blend[1] / s)
        p = [blend[0] * p_trans[i] + blend[1] * p_analog[i] for i in range(3)]

    intervals = None
    if simulate and alpha is not None:
        intervals = simulate_intervals(p_analog, n_eff, alpha, blend, n_sim=n_sim)
    return {
        "p": p,
        "p_trans": list(p_trans),
        "p_analog": p_analog,
        "blend": blend,
        "interval": intervals,
        "n_eff": n_eff,
    }

# --------------------------------------------------------------------------
# 回测：模型技能与自适应权重
# --------------------------------------------------------------------------

def _fingerprint(anoms: Sequence[Optional[float]]) -> str:
    payload = "|".join(f"{x:.3f}" for x in anoms)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]

def compute_skills(anoms: Sequence[Optional[float]],
                   states: Sequence[Optional[int]],
                   ages: Sequence[int],
                   horizon: int,
                   step: int = 3,
                   min_train: int = 240,
                   use_cache: bool = True) -> dict:
    """滚动起点回测，返回转移模型 / 相似型 / 气候态的 Brier 分与技能分。"""
    key = f"{_fingerprint(anoms)}-h{horizon}-s{step}"

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / "skills.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                if key in data:
                    return data[key]
            except (json.JSONDecodeError, OSError):
                pass

    bs_t, bs_a, bs_c = [], [], []
    for t in range(min_train, len(states) - horizon, step):
        truth = states[t + horizon]
        if truth is None or states[t] is None:
            continue
        p_t, _alpha, _n = transition_distribution(states, ages, t, horizon)
        p_c = climatology_distribution(states, t, horizon)
        bs_t.append(_brier(p_t, truth))
        bs_c.append(_brier(p_c, truth))
        analog = analog_distribution(anoms, states, t, horizon)
        if analog is not None:
            bs_a.append(_brier(analog[0], truth))

    def mean_or_none(xs):
        return _mean(xs) if xs else None

    bt, ba, bc = mean_or_none(bs_t), mean_or_none(bs_a), mean_or_none(bs_c)
    result = {
        "horizon": horizon,
        "n_eval": len(bs_c),
        "brier_trans": bt,
        "brier_analog": ba,
        "brier_clim": bc,
        "skill_trans": (1 - bt / bc) if (bt is not None and bc) else None,
        "skill_analog": (1 - ba / bc) if (ba is not None and bc) else None,
    }

    if use_cache:
        cache_file = CACHE_DIR / "skills.json"
        data = {}
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}
        data[key] = result
        cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return result

def skill_weights(skills: dict) -> Tuple[Tuple[float, float], Optional[str]]:
    """按技能分分配权重；负技能模型直接剔除，两者都不优于气候态时退化为等权。"""
    st, sa = skills.get("skill_trans"), skills.get("skill_analog")
    if st is None and sa is None:
        return (0.5, 0.5), "两模型均无可用回测样本"
    if sa is None:
        return (1.0, 0.0), "相似型样本不足，仅用转移模型"
    if st is None:
        return (0.0, 1.0), "转移模型样本不足，仅用相似型"
    st, sa = max(st, 0.0), max(sa, 0.0)
    if st + sa <= 0:
        return (0.5, 0.5), "两模型均未跑赢气候态，退化为等权"
    return (st / (st + sa), sa / (st + sa)), None

# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------

def _pct(x: float) -> str:
    return f"{x * 100:5.1f}%"

def _dwidth(s: str) -> int:
    """终端显示宽度（中文/全角字符按 2 列计）。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)

def _pad(s: str, width: int, right: bool = False) -> str:
    gap = " " * max(0, width - _dwidth(s))
    return gap + s if right else s + gap

def _bar(p: float, width: int = 20) -> str:
    k = int(round(p * width))
    k = max(0, min(width, k))
    return "#" * k + "-" * (width - k)

def _oneline(label: str, p: float, lo: Optional[float], hi: Optional[float]) -> str:
    ci = ""
    if lo is not None and hi is not None:
        ci = f"  [90%CI {lo * 100:5.1f}% - {hi * 100:5.1f}%]"
    return f"  {label:<6s} {_pct(p)}{ci}  {_bar(p)}"

def horizon_label(obs: Sequence[Obs], t_now: int, horizon: int) -> str:
    mi = obs[t_now].month_index + horizon
    year, m = divmod(mi, 12)
    return f"{year} {SEASONS[m]}"

def _trailing_run(anoms: Sequence[Optional[float]], t_now: int,
                  above: bool, threshold: float = ONI_THRESHOLD) -> int:
    k = 0
    i = t_now
    while i >= 0:
        v = anoms[i]
        if v is None:
            break
        if (v >= threshold) if above else (v <= -threshold):
            k += 1
            i -= 1
        else:
            break
    return k

def print_header(obs: Sequence[Obs], anoms: Sequence[Optional[float]],
                 states: Sequence[Optional[int]], ages: Sequence[int],
                 t_now: int, source: str) -> None:
    latest = obs[t_now]
    state = states[t_now]
    print("=" * 66)
    print(" ENSO 概率分析  |  El Nino / La Nina Probability")
    print("=" * 66)
    print("数据源    NOAA CPC ONI v5（Niño3.4 三个月滑动平均, ERSSTv5）")
    print(f"序列范围  {obs[0].label} .. {obs[-1].label}   共 {len(obs)} 个月")
    print(f"读取方式  {source}")
    print()
    print("当前状态")
    print(f"  最新 ONI   {latest.anom:+.2f}   ({latest.label})")
    print(f"  状态判定   {STATE_LABELS[state]} ({STATE_LABELS_EN[state]})")
    print(f"  已持续     {ages[t_now]} 个月")
    print(f"  判定规则   ONI 连续 {EVENT_WINDOW} 个重叠三月 >= +0.5 为厄尔尼诺，<= -0.5 为拉尼娜")

    if state == NEUTRAL:
        pos = _trailing_run(anoms, t_now, above=True)
        neg = _trailing_run(anoms, t_now, above=False)
        if max(pos, neg) >= EVENT_WINDOW - 1:
            if pos >= neg:
                print(f"  临近判定   近 {pos} 个月 ONI >= +0.5，再持续 1 个月即触发厄尔尼诺")
            else:
                print(f"  临近判定   近 {neg} 个月 ONI <= -0.5，再持续 1 个月即触发拉尼娜")

    recent = [f"{o.anom:+.1f}" for o in obs[-6:]]
    print(f"  近 6 个月  {', '.join(recent)}")
    print()

def run_forecast(obs: Sequence[Obs], anoms: Sequence[Optional[float]],
                 states: Sequence[Optional[int]], ages: Sequence[int],
                 t_now: int, horizons: Sequence[int], source: str,
                 do_sim: bool = True, show_chart: bool = False,
                 write_json: bool = False) -> dict:
    print_header(obs, anoms, states, ages, t_now, source)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": DATA_URL,
        "data_range": [obs[0].label, obs[-1].label],
        "n_obs": len(obs),
        "latest": {
            "label": obs[t_now].label,
            "oni": obs[t_now].anom,
            "state": STATE_LABELS_EN[states[t_now]],
            "state_cn": STATE_LABELS[states[t_now]],
            "age_months": ages[t_now],
        },
        "forecast": {},
    }

    for h in horizons:
        print("-" * 66)
        print(f" 未来 {h:2d} 个月（{horizon_label(obs, t_now, h)} 前后）的概率")
        print("-" * 66)

        comp = monthly_shrunk_components(anoms, states, ages, t_now, h)
        skills = compute_skills(anoms, states, ages, h)
        w, note = skill_weights(skills)
        analog_arg = None
        if comp["p_analog"] is not None:
            analog_arg = (comp["p_analog"], comp["analog_weights"],
                          comp["n_analog_eff"])
            if comp["n_analog_eff"] < 5:
                w = (1.0, 0.0)

        fc = blend_forecast(comp["p_trans"], analog_arg, w,
                            simulate=do_sim, alpha=comp["alpha"])
        intervals = fc["interval"]

        for s in STATE_ORDER:
            i = s + 1
            lo, hi = intervals[i] if intervals else (None, None)
            print(_oneline(STATE_LABELS[s], fc["p"][i], lo, hi))

        wt, wa = fc["blend"]
        print()
        print(f"  模型权重   转移模型 {wt:.2f} / 历史相似型 {wa:.2f}")
        lam_t, lam_a = comp["lam"]
        print(f"  月度收缩   λ(转移)={lam_t:.2f} / λ(相似)={lam_a:.2f}  (1=完全按同月起报)")
        sk_t, sk_a = skills["skill_trans"], skills["skill_analog"]
        sk_txt = f"转移 {sk_t:+.3f}" if sk_t is not None else "转移 n/a"
        sk_txt += f" / 相似 {sk_a:+.3f}" if sk_a is not None else " / 相似 n/a"
        print(f"  回测技能   {sk_txt}   (Brier 技能分, 滚动回测 n={skills['n_eval']})")

        point = oni_point_forecast(anoms, t_now, h)
        if point is not None:
            print(f"  ONI 预测   {point['mean']:+.2f} "
                  f"[90%CI {point['q05']:+.2f} ~ {point['q95']:+.2f}] "
                  f"(近邻 {point['n']} 个, 有效 {point['n_eff']:.0f})")
        analogs = top_analogs(obs, anoms, states, t_now, h)
        if analogs:
            parts = []
            for a in analogs[:3]:
                st = STATE_LABELS[a["state"]] if a["state"] is not None else "?"
                v = f"{a['target_oni']:+.1f}" if a["target_oni"] is not None else "-"
                parts.append(f"{a['target']}({st},{v})")
            print(f"  相似年份   " + " · ".join(parts))
        if note:
            print(f"  提示       {note}")
        print()

        result["forecast"][str(h)] = {
            "target": horizon_label(obs, t_now, h),
            "el_nino": fc["p"][2],
            "neutral": fc["p"][1],
            "la_nina": fc["p"][0],
            "interval_el_nino": list(intervals[2]) if intervals else None,
            "interval_neutral": list(intervals[1]) if intervals else None,
            "interval_la_nina": list(intervals[0]) if intervals else None,
            "weights": {"transition": wt, "analog": wa},
            "skill": {"transition": sk_t, "analog": sk_a},
            "monthly_shrink": {"lam_transition": lam_t, "lam_analog": lam_a},
            "oni_point": point,
            "top_analogs": analogs,
        }

    print("=" * 66)
    print(" 说明：概率为历史统计外推，不等于官方预报；厄尔尼诺/拉尼娜的")
    print(" 实际发生还受海气耦合、次表层热含量、季节锁相与随机扰动影响。")
    print("=" * 66)

    if show_chart:
        print()
        print_oni_chart(obs, anoms, months=72)

    if write_json:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / "forecast.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {path}")

    return result

def print_oni_chart(obs: Sequence[Obs], anoms: Sequence[Optional[float]],
                    months: int = 72, width: int = 24) -> None:
    tail_obs = obs[-months:]
    tail = anoms[-months:]
    vals = [abs(v) for v in tail if v is not None]
    scale = max(vals) if vals else 1.0
    print(f"ONI 最近 {len(tail_obs)} 个月（柱长按 |ONI| 归一, 负值向左）")
    print(" " * 12 + "-" * width + "|" + "-" * width)
    for o, v in zip(tail_obs, tail):
        if v is None:
            continue
        k = int(round(abs(v) / scale * width))
        k = max(1, k) if abs(v) > 0.01 else 0
        if v >= 0:
            bar = " " * width + "|" + "#" * k
        else:
            bar = " " * (width - k) + "#" * k + "|"
        print(f"{o.label:<11s} {bar}")

# --------------------------------------------------------------------------
# 自测
# --------------------------------------------------------------------------

def _synthetic_ar1(n: int = 6000, phi: float = 0.9, sd: float = 0.8,
                   seed: int = 7) -> List[Optional[float]]:
    rng = random.Random(seed)
    x = 0.0
    out: List[Optional[float]] = []
    innov = sd * math.sqrt(1 - phi * phi)
    for _ in range(n):
        x = phi * x + rng.gauss(0.0, innov)
        out.append(x)
    return out

def selftest(verbose: bool = True) -> bool:
    """内置自测：规则、参数回收、蒙特卡洛对照、对称性、边界条件。"""
    checks: List[Tuple[str, bool, str]] = []

    s = classify_states([0.6] * 5)
    checks.append(("规则-厄尔尼诺判定",
                   s[4] == EL_NINO and s[3] is None,
                   f"states={s}"))
    s = classify_states([0.6, 0.6, 0.6, 0.6, 0.4])
    checks.append(("规则-不足 5 月不触发", s[4] == NEUTRAL, f"state={s[4]}"))
    s = classify_states([-0.7] * 5)
    checks.append(("规则-拉尼娜判定", s[4] == LA_NINA, f"state={s[4]}"))

    sample = (" SEAS  YR   TOTAL   ANOM\n"
              "  DJF 1950  25.01  -1.32\n"
              "  JFM 1950  25.36  -1.20\n"
              "# comment\n")
    parsed = parse_oni_text(sample)
    checks.append(("解析器-列与排序", len(parsed) == 2 and parsed[0].season == "DJF",
                   f"n={len(parsed)}"))

    synth = _synthetic_ar1(n=6000)
    st = classify_states(synth)
    ages = phase_ages(st)
    t_now = len(st) - 1
    h = 6
    p_est, _alpha, n_pairs = transition_distribution(st, ages, t_now, h, age_boost=1.0)
    cur = st[t_now]
    counts = [0, 0, 0]
    total = 0
    for t in range(len(st) - h):
        if t + h > t_now:
            break
        if st[t] is None or st[t + h] is None:
            continue
        if st[t] != cur:
            continue
        counts[st[t + h] + 1] += 1
        total += 1
    emp = [c / total for c in counts] if total else [1 / 3] * 3
    err = max(abs(p_est[i] - emp[i]) for i in range(3))
    checks.append(("转移模型-经验分布回收", total > 100 and err < 0.05,
                   f"n={total}, max|Δ|={err:.4f}"))

    n_eval = 60
    bs_a, bs_c = [], []
    for t in range(len(st) - 120, len(st) - h, 2):
        if st[t] is None or st[t + h] is None:
            continue
        analog = analog_distribution(synth, st, t, h)
        if analog is None:
            continue
        bs_a.append(_brier(analog[0], st[t + h]))
        bs_c.append(_brier(climatology_distribution(st, t, h), st[t + h]))
    skill = 1 - _mean(bs_a) / _mean(bs_c) if bs_a else 0.0
    checks.append(("相似型-MC 对照优于气候态",
                   len(bs_a) >= n_eval * 0.5 and skill > 0.02,
                   f"n={len(bs_a)}, skill={skill:+.4f}"))

    flipped = [-v if v is not None else None for v in synth]
    st_f = classify_states(flipped)
    ages_f = phase_ages(st_f)
    p1, a1, _ = transition_distribution(st, ages, t_now, 3)
    p2, _a2, _ = transition_distribution(st_f, ages_f, t_now, 3)
    an1 = analog_distribution(synth, st, t_now, 3)
    an2 = analog_distribution(flipped, st_f, t_now, 3)
    f1 = blend_forecast(p1, an1, (0.5, 0.5), simulate=False)[ "p"]
    f2 = blend_forecast(p2, an2, (0.5, 0.5), simulate=False)["p"]
    sym_err = max(abs(f1[2] - f2[0]), abs(f1[0] - f2[2]), abs(f1[1] - f2[1]))
    checks.append(("对称性-符号翻转", sym_err < 1e-12, f"max|Δ|={sym_err:.3e}"))

    fc = blend_forecast(p1, an1, (0.5, 0.5), simulate=True, alpha=a1, n_sim=800)
    total_p = sum(fc["p"])
    lo_ok = all(fc["interval"][i][0] <= fc["p"][i] <= fc["interval"][i][1] for i in range(3))
    checks.append(("概率归一/区间覆盖", abs(total_p - 1) < 1e-9 and lo_ok,
                   f"sum={total_p:.6f}, covered={lo_ok}"))

    holey = list(synth[:200])
    holey[100] = None
    st_h = classify_states(holey)
    hole_ok = st_h[100] is None and st_h[99] is not None
    checks.append(("边界-缺失值传播", hole_ok, "None 会阻塞其后的 5 个月窗口"))

    short = classify_states([0.6, 0.7])
    checks.append(("边界-过短序列", all(v is None for v in short), f"states={short}"))

    ages_ok = all(
        (ages[i] == ages[i - 1] + 1) if (i > 0 and st[i] is not None and st[i] == st[i - 1])
        else True
        for i in range(1, len(st))
    )
    checks.append(("相位年龄-递推正确", ages_ok, f"max_age={max(ages)}"))

    skills = compute_skills(synth, st, ages, 6, step=12, min_train=240, use_cache=False)
    consistent = (
        skills["skill_trans"] is not None
        and abs(skills["skill_trans"]
                - (1 - skills["brier_trans"] / skills["brier_clim"])) < 1e-12
    )
    checks.append(("回测-技能自洽", consistent,
                   f"Brier trans={skills['brier_trans']:.4f}, clim={skills['brier_clim']:.4f}"))

    season_chk = 2
    _, _, n_mo = transition_distribution(st, ages, t_now, 3, season=season_chk)
    cnt_mo = 0.0
    cur_s = st[t_now]
    bucket_now = _age_bucket(ages[t_now]) if cur_s is not None else -1
    for t in range(len(st) - 3):
        if t + 3 > t_now:
            break
        if t % 12 != season_chk:
            continue
        s0, s1 = st[t], st[t + 3]
        if s0 is None or s1 is None or cur_s is None or s0 != cur_s:
            continue
        cnt_mo += 1.0 if _age_bucket(ages[t]) == bucket_now else 0.3
    checks.append(("月度分层-同相过滤", abs(cnt_mo - n_mo) < 1e-9,
                   f"独立计数={cnt_mo:.1f}, 模型={n_mo:.1f}"))

    nn = _neighbors(synth, t_now, window=6, horizon=3)
    nn_ok = (len(nn) > 50 and all(t + 3 <= t_now for t, _, _ in nn)
             and all(d0 <= d1 for (_, d0, _), (_, d1, _) in zip(nn, nn[1:]))
             and all(v == synth[t + 3] for t, _, v in nn[:20]))
    checks.append(("近邻-无泄漏/排序", nn_ok, f"近邻数={len(nn)}"))

    pt_fc = oni_point_forecast(synth, t_now, 3)
    pt_ok = (pt_fc is not None and pt_fc["q05"] <= pt_fc["mean"] <= pt_fc["q95"]
             and abs(pt_fc["mean"]) <= 3.5)
    checks.append(("强度预测-分位自洽", pt_ok,
                   f"mean={pt_fc['mean']:+.2f} q05={pt_fc['q05']:+.2f} q95={pt_fc['q95']:+.2f}"))

    st300 = classify_states([synth[i] for i in range(300)])
    ages300 = phase_ages(st300)
    heat = skill_heatmap([synth[i] for i in range(300)], st300, ages300,
                         horizons=(3,), step=6, min_train=60)
    heat_vals = [v for m in range(12) for v in heat[m].values()]
    heat_ok = (set(heat.keys()) == set(range(12))
               and any(v is not None for v in heat_vals)
               and all(v is None or -1.0 <= v <= 1.0 for v in heat_vals))
    checks.append(("热力图-结构/取值", heat_ok, f"有效格点 {sum(v is not None for v in heat_vals)}/12"))

    cal = compute_calibration([synth[i] for i in range(300)], st300, ages300, 3,
                              step=6, min_train=60)
    n_eval = sum(1 for t in range(60, 300 - 3, 6)
                 if st300[t] is not None and st300[t + 3] is not None)
    cal_ok = sum(b["n"] for b in cal) == n_eval and all(
        b["mean_pred"] is None or 0.0 <= b["mean_pred"] <= 1.0 for b in cal)
    checks.append(("校准-桶覆盖完整", cal_ok, f"桶总和={sum(b['n'] for b in cal)}, 应={n_eval}"))

    fl300 = [-synth[i] for i in range(300)]
    stf300 = classify_states(fl300)
    agesf300 = phase_ages(stf300)
    tv = next(i for i in range(120, 200) if st300[i] is not None)
    c1 = monthly_shrunk_components([synth[i] for i in range(300)], st300, ages300, tv, 3)
    c2 = monthly_shrunk_components(fl300, stf300, agesf300, tv, 3)
    mirr = max(abs(c2["p_trans"][i] - c1["p_trans"][2 - i]) for i in range(3))
    checks.append(("月度收缩-符号对称", mirr < 1e-12, f"max|Δ|={mirr:.3e}"))

    samples3 = [{"mean": 0.5, "q05": 0.2, "q95": 0.9},
                {"mean": 0.8, "q05": 0.5, "q95": 1.1}]
    svg_fc = _svg_fc_curve(["MAM", "JJA"], samples3)
    svg_rel = _svg_reliability([("3 个月", cal, "hsl(8,62%,42%)")])
    heat_htm = _heatmap_table(heat, (3,))
    vis_ok = ('<svg' in svg_fc and 'polygon' in svg_fc and 'polyline' in svg_fc
              and '<svg' in svg_rel and '<circle' in svg_rel
              and 'table class="heat"' in heat_htm)
    checks.append(("可视化组件-生成", vis_ok, "SVG/表格片段完整"))

    passed = sum(1 for _, ok, _ in checks if ok)
    if verbose:
        print("=" * 66)
        print(" 自测 (selftest)")
        print("=" * 66)
        for name, ok, detail in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:<24s} {detail}")
        print("-" * 66)
        print(f"  结果：{passed}/{len(checks)} 通过")
    return passed == len(checks)

# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def cmd_forecast(args) -> int:
    obs, source = load_obs(refresh=args.refresh, data_file=args.data_file,
                           allow_network=not args.offline)
    anoms = [o.anom for o in obs]
    states = classify_states(anoms)
    ages = phase_ages(states)

    valid = [i for i, s in enumerate(states) if s is not None]
    if not valid:
        raise SystemExit("数据不足：至少需要 5 个月的有效 ONI 才能判定状态。")
    t_now = valid[-1]

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    run_forecast(obs, anoms, states, ages, t_now, horizons, source,
                 do_sim=not args.no_bootstrap,
                 show_chart=args.chart,
                 write_json=args.json)
    return 0

def cmd_update(args) -> int:
    try:
        text = fetch_oni_text()
    except Exception as exc:
        print(f"下载失败：{exc}", file=sys.stderr)
        return 1
    obs = parse_oni_text(text)
    if len(obs) < 100:
        print("下载内容异常，未写入缓存。", file=sys.stderr)
        return 1
    save_cache(text)
    print(f"已更新 {CACHE_FILE}：{obs[0].label} .. {obs[-1].label}，共 {len(obs)} 个月。")
    return 0

def cmd_backtest(args) -> int:
    obs, _source = load_obs(refresh=args.refresh, data_file=args.data_file)
    anoms = [o.anom for o in obs]
    states = classify_states(anoms)
    ages = phase_ages(states)

    horizons = [int(x) for x in args.horizons.split(",")]
    widths = [9, 9, 10, 10, 10, 11, 11]
    headers = ["提前期", "评估数", "转移 BS", "相似 BS", "气候 BS", "转移技能", "相似技能"]

    line = "=" * (sum(widths) + 2)
    print(line)
    print(" 滚动回测（等权 Brier 分，越小越好；技能分 = 1 - BS/BS_clim）")
    print(line)
    print(" " + "".join(_pad(h, w, right=True) for h, w in zip(headers, widths)))
    print("-" * len(line))

    def fmt(x):
        return f"{x:.4f}" if x is not None else "n/a"

    for h in horizons:
        sk = compute_skills(anoms, states, ages, h)
        cells = [str(h), str(sk["n_eval"]), fmt(sk["brier_trans"]),
                 fmt(sk["brier_analog"]), fmt(sk["brier_clim"]),
                 fmt(sk["skill_trans"]), fmt(sk["skill_analog"])]
        print(" " + "".join(_pad(c, w, right=True) for c, w in zip(cells, widths)))
    print("-" * len(line))
    print(" 说明：技能分为正表示优于历史气候态基线；接近 0 表示与气候态相当。")
    return 0

# --------------------------------------------------------------------------
# HTML 可视化报告
# --------------------------------------------------------------------------

_COLOR = {EL_NINO: "#e0472e", NEUTRAL: "#8a9299", LA_NINA: "#1f6fb2"}
_COLOR_BAND = {EL_NINO: "#e0472e", LA_NINA: "#1f6fb2"}

_REPORT_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
  background: #f5f7fa; color: #1c2733; line-height: 1.55;
}
.wrap { max-width: 1200px; margin: 0 auto; padding: 28px 20px 48px; }
header h1 { font-size: 26px; letter-spacing: .5px; }
header .sub { color: #5b6b7b; margin-top: 6px; font-size: 13px; }
.meta { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.chip {
  background: #fff; border: 1px solid #e3e8ee; color: #46576a;
  border-radius: 999px; padding: 3px 12px; font-size: 12px;
}
.card { background: #fff; border: 1px solid #e6ebf1; border-radius: 14px; padding: 18px 20px; }
.grid { display: grid; gap: 14px; }
.now { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin: 20px 0 14px; }
.now .label { font-size: 12px; color: #7a8799; margin-bottom: 4px; }
.oni-val { font-size: 34px; font-weight: 700; font-family: Consolas, Menlo, monospace; }
.badge {
  display: inline-block; color: #fff; border-radius: 8px; padding: 5px 14px;
  font-size: 15px; font-weight: 600; margin-top: 4px;
}
.alert {
  background: #fff7e6; border: 1px solid #ffd591; border-radius: 12px;
  color: #8a5a00; padding: 10px 16px; font-size: 13px; margin-bottom: 14px;
}
h2 { font-size: 17px; margin: 26px 0 12px; color: #26364a; }
.fc-grid { grid-template-columns: repeat(auto-fit, minmax(255px, 1fr)); }
.fc-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 10px; }
.fc-head .h { font-size: 15px; font-weight: 700; }
.fc-head .t { font-size: 12px; color: #7a8799; }
.bar-row { display: grid; grid-template-columns: 58px 1fr 64px; align-items: center; gap: 8px; margin: 9px 0; }
.bar-name { font-size: 13px; color: #46576a; }
.bar-track { background: #eef2f6; border-radius: 6px; height: 14px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 6px; transition: width .4s; }
.bar-pct { font-size: 13px; font-weight: 700; font-family: Consolas, Menlo, monospace; text-align: right; }
.bar-ci { font-size: 11px; color: #9aa7b5; text-align: right; grid-column: 2/4; margin-top: -6px; }
.fc-foot { border-top: 1px dashed #e6ebf1; margin-top: 10px; padding-top: 8px; font-size: 11.5px; color: #7a8799; }
.svg-box { background: #fff; border: 1px solid #e6ebf1; border-radius: 14px; padding: 14px 16px 6px; margin-bottom: 14px; }
.svg-box h3 { font-size: 14px; color: #26364a; margin-bottom: 4px; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 12px; color: #5b6b7b; margin: 4px 0 24px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.sw { width: 14px; height: 10px; border-radius: 3px; display: inline-block; }
svg { width: 100%; height: auto; display: block; }
.grid { stroke: #e6ebf1; stroke-width: 1; }
.grid.zero { stroke: #c8d2dd; }
.grid.thresh-pos { stroke: #e0472e; stroke-dasharray: 5 4; opacity: .55; }
.grid.thresh-neg { stroke: #1f6fb2; stroke-dasharray: 5 4; opacity: .55; }
.ytick, .xtick { font-size: 10.5px; fill: #7a8799; }
.ytick { text-anchor: end; }
.xtick { text-anchor: middle; }
.xtick-line { stroke: #d5dde6; stroke-width: 1; }
.series { fill: none; stroke: #34465c; stroke-width: 1.6; }
.dot { fill: #34465c; }
.last { fill: #d4380d; }
.lastlabel { font-size: 11px; fill: #d4380d; font-weight: 700; text-anchor: end; }
table.skills { width: 100%; border-collapse: collapse; background: #fff; border-radius: 12px; overflow: hidden; font-size: 13px; }
table.skills th, table.skills td { padding: 9px 12px; text-align: right; border-bottom: 1px solid #eef2f6; }
table.skills th { background: #f2f5f9; color: #46576a; font-weight: 600; }
table.skills th:first-child, table.skills td:first-child { text-align: left; }
table.skills tr:last-child td { border-bottom: none; }
.pos { color: #2e7d32; font-weight: 600; }
.neg { color: #c62828; font-weight: 600; }
footer { margin-top: 30px; font-size: 12px; color: #7a8799; border-top: 1px solid #e3e8ee; padding-top: 14px; }
footer p { margin: 4px 0; }
.hint { font-size: 12.5px; color: #7a8799; margin: -2px 0 12px; line-height: 1.6; }
.ana-grid { grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); margin-bottom: 6px; }
.ana-card { background: #fff; border: 1px solid #e6ebf1; border-radius: 12px; padding: 14px 16px; }
.ana-year { font-size: 15px; font-weight: 700; color: #26364a; margin-bottom: 8px; font-family: Consolas, Menlo, monospace; }
.ana-row { display: flex; justify-content: space-between; align-items: center; font-size: 12.5px; color: #5b6b7b; margin: 5px 0; gap: 8px; }
.ana-row b { color: #26364a; }
.mono { font-family: Consolas, Menlo, monospace; }
table.heat { width: 100%; border-collapse: collapse; background: #fff; border-radius: 12px; overflow: hidden; font-size: 13px; margin-bottom: 6px; }
table.heat th, table.heat td { padding: 9px 12px; text-align: center; border: 1px solid #eef2f6; }
table.heat th { background: #f2f5f9; color: #46576a; font-weight: 600; }
.rel-grid { grid-template-columns: repeat(auto-fit, minmax(430px, 1fr)); }
"""

def _event_bands(states: Sequence[Optional[int]]) -> List[Tuple[int, int, int]]:
    """连续 >= 5 个月的同号事件区间，返回 (起, 止, 状态)。"""
    bands: List[Tuple[int, int, int]] = []
    i, n = 0, len(states)
    while i < n:
        s = states[i]
        if s in (EL_NINO, LA_NINA):
            j = i
            while j + 1 < n and states[j + 1] == s:
                j += 1
            if j - i + 1 >= 5:
                bands.append((i, j, s))
            i = j + 1
        else:
            i += 1
    return bands

def _svg_series(obs_s: Sequence[Obs], an_s: Sequence[Optional[float]],
                st_s: Sequence[Optional[int]], width: int = 1160,
                height: int = 330, dots: bool = False) -> str:
    """ONI 折线图 SVG（静态生成，含事件色带与 +/-0.5 阈值线）。"""
    n = len(obs_s)
    pad_l, pad_r, pad_t, pad_b = 58, 16, 18, 32
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    y_lo, y_hi = -3.0, 3.5

    def xp(i: int) -> float:
        return pad_l + (pw * i / (n - 1) if n > 1 else pw / 2)

    def yp(v: float) -> float:
        v = max(y_lo, min(y_hi, v))
        return pad_t + (y_hi - v) / (y_hi - y_lo) * ph

    out = [f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet" role="img">']

    y = -3.0
    while y <= 3.5 + 1e-9:
        yy = yp(y)
        cls = "grid"
        if abs(y - 0.5) < 1e-9 or abs(y + 0.5) < 1e-9:
            cls = "grid thresh-pos" if y > 0 else "grid thresh-neg"
        elif abs(y) < 1e-9:
            cls = "grid zero"
        out.append(f'<line class="{cls}" x1="{pad_l}" y1="{yy:.1f}" x2="{width - pad_r}" y2="{yy:.1f}"/>')
        ty = 0.0 if abs(y) < 1e-9 else y
        out.append(f'<text class="ytick" x="{pad_l - 8}" y="{yy + 4:.1f}">{ty:+.1f}</text>')
        y += 0.5

    for i0, i1, s in _event_bands(st_s):
        bw = max(1.0, xp(i1) - xp(i0))
        out.append(f'<rect x="{xp(i0):.1f}" y="{pad_t}" width="{bw:.1f}" height="{ph:.1f}" '
                   f'fill="{_COLOR_BAND[s]}" opacity="0.12"/>')

    seg: List[int] = []
    for i, v in enumerate(an_s):
        if v is None:
            if seg:
                pts = " ".join(f"{xp(j):.1f},{yp(an_s[j]):.1f}" for j in seg)
                out.append(f'<polyline class="series" points="{pts}"/>')
                seg = []
        else:
            seg.append(i)
    if seg:
        pts = " ".join(f"{xp(j):.1f},{yp(an_s[j]):.1f}" for j in seg)
        out.append(f'<polyline class="series" points="{pts}"/>')

    if dots:
        for i, v in enumerate(an_s):
            if v is not None:
                out.append(f'<circle class="dot" cx="{xp(i):.1f}" cy="{yp(v):.1f}" r="2.4"/>')

    last = next((i for i in range(n - 1, -1, -1) if an_s[i] is not None), None)
    if last is not None:
        out.append(f'<circle class="last" cx="{xp(last):.1f}" cy="{yp(an_s[last]):.1f}" r="4"/>')
        out.append(f'<text class="lastlabel" x="{xp(last):.1f}" y="{yp(an_s[last]) - 10:.1f}">{an_s[last]:+.2f}</text>')

    span = obs_s[-1].month_index - obs_s[0].month_index
    step = 120 if span > 480 else 12
    prev_year = None
    for i, o in enumerate(obs_s):
        if o.year % (step // 12) == 0 and o.season == "DJF" and o.year != prev_year:
            prev_year = o.year
            out.append(f'<line class="xtick-line" x1="{xp(i):.1f}" y1="{pad_t + ph}" '
                       f'x2="{xp(i):.1f}" y2="{pad_t + ph + 5}"/>')
            out.append(f'<text class="xtick" x="{xp(i):.1f}" y="{pad_t + ph + 19}">{o.year}</text>')

    out.append("</svg>")
    return "".join(out)

def _fc(entries: dict, h: int, key: str, fmt: str = ".3f") -> str:
    v = entries[h][2].get(key)
    return ("n/a" if v is None else format(v, fmt))

def _forecast_card(h: int, entries: dict) -> str:
    target, fc, skills, w, note = entries[h]
    rows = []
    for s in STATE_ORDER:
        i = s + 1
        p = fc["p"][i]
        lo, hi = fc["interval"][i] if fc["interval"] else (None, None)
        ci = f"{lo * 100:.1f}–{hi * 100:.1f}%" if lo is not None else "—"
        rows.append(
            f'<div class="bar-row">'
            f'<span class="bar-name">{STATE_LABELS[s]}</span>'
            f'<div class="bar-track"><div class="bar-fill" '
            f'style="width:{max(p * 100, 0.8):.1f}%;background:{_COLOR[s]}"></div></div>'
            f'<span class="bar-pct">{p * 100:.1f}%</span>'
            f'<span class="bar-ci">90%CI {ci}</span></div>'
        )
    sk_t = _fc(entries, h, "skill_trans", "+.3f")
    sk_a = _fc(entries, h, "skill_analog", "+.3f")
    foot = (f"模型权重：转移 {w[0]:.2f} · 相似 {w[1]:.2f}"
            f"　|　回测技能：转移 {sk_t} · 相似 {sk_a}")
    if note:
        foot += f"　|　{note}"
    return (
        f'<div class="card">'
        f'<div class="fc-head"><span class="h">未来 {h} 个月</span>'
        f'<span class="t">{target}</span></div>'
        f'{"".join(rows)}'
        f'<div class="fc-foot">{foot}</div></div>'
    )

def _svg_fc_curve(labels_s: Sequence[str], pts: Sequence[Optional[dict]],
                  width: int = 1160, height: int = 320) -> str:
    """未来 ONI 点预测曲线（均值 + 90% 阴影带，含 ±0.5 阈值虚线）。"""
    pad_l, pad_r, pad_t, pad_b = 58, 16, 18, 32
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    y_lo, y_hi = -3.0, 3.5
    w, h = width, height

    def yp(v: float) -> float:
        v = max(y_lo, min(y_hi, v))
        return pad_t + (y_hi - v) / (y_hi - y_lo) * ph

    n = len(labels_s)

    def xp(i: int) -> float:
        return pad_l + (pw * i / (n - 1) if n > 1 else pw / 2)

    out = [f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet" role="img">']
    y = -3.0
    while y <= 3.5 + 1e-9:
        yy = yp(y)
        cls = "grid"
        if abs(y - 0.5) < 1e-9 or abs(y + 0.5) < 1e-9:
            cls = "grid thresh-pos" if y > 0 else "grid thresh-neg"
        elif abs(y) < 1e-9:
            cls = "grid zero"
        out.append(f'<line class="{cls}" x1="{pad_l}" y1="{yy:.1f}" x2="{w - pad_r}" y2="{yy:.1f}"/>')
        ty = 0.0 if abs(y) < 1e-9 else y
        out.append(f'<text class="ytick" x="{pad_l - 8}" y="{yy + 4:.1f}">{ty:+.1f}</text>')
        y += 0.5

    idx = [i for i in range(n) if pts[i] is not None]
    if len(idx) >= 2:
        up = " ".join(f"{xp(i):.1f},{yp(pts[i]['q95']):.1f}" for i in idx)
        lo = " ".join(f"{xp(i):.1f},{yp(pts[i]['q05']):.1f}" for i in reversed(idx))
        out.append(f'<polygon points="{up} {lo}" fill="#34465c" opacity="0.12"/>')
        means = " ".join(f"{xp(i):.1f},{yp(pts[i]['mean']):.1f}" for i in idx)
        out.append(f'<polyline class="series" points="{means}"/>')
        for i in idx:
            out.append(f'<circle class="dot" cx="{xp(i):.1f}" cy="{yp(pts[i]["mean"]):.1f}" r="2.6"/>')
            if i % 2 == 1 or i == idx[-1]:
                out.append(f'<text class="xtick" x="{xp(i):.1f}" y="{h - pad_b + 16}">{labels_s[i]}</text>')
        last_i = idx[-1]
        out.append(f'<text class="lastlabel" x="{xp(last_i):.1f}" '
                   f'y="{yp(pts[last_i]["mean"]) - 10:.1f}">{pts[last_i]["mean"]:+.2f}</text>')
    out.append("</svg>")
    return "".join(out)

def _svg_reliability(series_s: Sequence[Tuple[str, List[dict], str]],
                     width: int = 560, height: int = 430) -> str:
    """可靠性曲线：x=预测概率, y=实际频率；对角虚线为完美校准。"""
    pad_l, pad_r, pad_t, pad_b = 52, 14, 14, 38
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b

    def px(x: float) -> float:
        return pad_l + min(max(x, 0.0), 1.0) * pw

    def py(x: float) -> float:
        return pad_t + (1.0 - min(max(x, 0.0), 1.0)) * ph

    out = [f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet" role="img">']
    for g in range(0, 6):
        f = g / 5
        out.append(f'<line class="grid" x1="{px(0):.1f}" y1="{py(f):.1f}" x2="{px(1):.1f}" y2="{py(f):.1f}"/>')
        out.append(f'<line class="grid" x1="{px(f):.1f}" y1="{py(0):.1f}" x2="{px(f):.1f}" y2="{py(1):.1f}"/>')
        out.append(f'<text class="ytick" x="{pad_l - 8}" y="{py(f) + 4:.1f}">{f:.1f}</text>')
        out.append(f'<text class="xtick" x="{px(f):.1f}" y="{height - pad_b + 16}">{f:.1f}</text>')
    out.append(f'<line x1="{px(0):.1f}" y1="{py(0):.1f}" x2="{px(1):.1f}" y2="{py(1):.1f}" '
               f'class="grid thresh-pos"/>')
    for label, buckets, color in series_s:
        for b in buckets:
            if b["n"] < 15 or b["mean_pred"] is None:
                continue
            r = min(2.0 + math.sqrt(b["n"]) / 2, 7.0)
            out.append(f'<circle cx="{px(b["mean_pred"]):.1f}" cy="{py(b["freq_obs"]):.1f}" '
                       f'r="{r:.1f}" fill="{color}" opacity="0.85"><title>'
                       f'{label} 预测 {b["mean_pred"]:.2f} 实际 {b["freq_obs"]:.2f} n={b["n"]}</title></circle>')
    out.append("</svg>")
    return "".join(out)

def _heatmap_table(heat: dict, horizons: Sequence[int]) -> str:
    """技能热力图：起报月 × 提前期，绿色=有技能，红色=负技能。"""
    rows = []
    for m in range(12):
        cells = [f'<td style="text-align:left;font-weight:600">{SEASONS[m]}</td>']
        for h in horizons:
            v = heat[m].get(h)
            if v is None:
                cells.append('<td style="background:#f2f5f9;color:#9aa7b5">·</td>')
            else:
                alpha = min(abs(v) / 0.6, 1.0)
                col = f"rgba(46,125,50,{alpha:.2f})" if v > 0 else f"rgba(198,40,40,{alpha:.2f})"
                cells.append(f'<td style="background:{col};color:#fff;font-weight:600">{v:+.2f}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    head = "<tr><th>起报月</th>" + "".join(f"<th>{h}个月</th>" for h in horizons) + "</tr>"
    return (f'<table class="heat"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>')

def build_report_html(obs: Sequence[Obs], anoms: Sequence[Optional[float]],
                      states: Sequence[Optional[int]], ages: Sequence[int],
                      t_now: int, horizons: Sequence[int], entries: dict,
                      source: str,
                      heat: Optional[dict] = None,
                      calibs: Optional[dict] = None,
                      analogs: Optional[List[dict]] = None,
                      oni_fc: Optional[dict] = None) -> str:
    latest = obs[t_now]
    state = states[t_now]
    badge_bg = _COLOR[state]
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    alert = ""
    if state == NEUTRAL:
        pos = _trailing_run(anoms, t_now, above=True)
        neg = _trailing_run(anoms, t_now, above=False)
        if max(pos, neg) >= EVENT_WINDOW - 1:
            if pos >= neg:
                msg = (f"临近判定：近 {pos} 个月 ONI ≥ +0.5，"
                       f"再持续 1 个月即触发厄尔尼诺判定（官方规则有约 5 个月滞后）")
            else:
                msg = (f"临近判定：近 {neg} 个月 ONI ≤ -0.5，"
                       f"再持续 1 个月即触发拉尼娜判定（官方规则有约 5 个月滞后）")
            alert = f'<div class="alert">⚠ {msg}</div>'

    cards = "\n".join(_forecast_card(h, entries) for h in horizons)

    rows = []
    for h in horizons:
        sk = entries[h][2]
        fmt = lambda x: ("n/a" if x is None else f"{x:.4f}")
        cls = lambda x: "" if x is None else ("pos" if x > 0 else "neg")
        rows.append(
            f"<tr><td>{h} 个月</td><td>{sk['n_eval']}</td>"
            f"<td>{fmt(sk['brier_trans'])}</td><td>{fmt(sk['brier_analog'])}</td>"
            f"<td>{fmt(sk['brier_clim'])}</td>"
            f"<td class='{cls(sk['skill_trans'])}'>{fmt(sk['skill_trans'])}</td>"
            f"<td class='{cls(sk['skill_analog'])}'>{fmt(sk['skill_analog'])}</td></tr>"
        )
    skill_table = "".join(rows)

    svg_full = _svg_series(obs, anoms, states)
    n10 = 120
    svg_recent = _svg_series(obs[-n10:], anoms[-n10:], states[-n10:], dots=True)

    analog_h = 6 if 6 in horizons else horizons[0]
    if analogs is None:
        analogs = top_analogs(obs, anoms, states, t_now, analog_h)
    analog_cards = []
    for a in analogs:
        st = a["state"]
        tc = _COLOR.get(st, "#8895a3")
        st_lab = STATE_LABELS.get(st, "n/a")
        dist = f"{a['distance']:.2f}"
        tv = f"{a['target_oni']:+.2f}" if a["target_oni"] is not None else "n/a"
        analog_cards.append(
            f'<div class="ana-card"><div class="ana-year">{a["label"]}</div>'
            f'<div class="ana-row"><span>{analog_h} 个月后</span>'
            f'<b>{a["target"]}</b></div>'
            f'<div class="ana-row"><span>当时 ONI</span><b class="mono">{tv}</b></div>'
            f'<div class="ana-row"><span>实际状态</span>'
            f'<span class="badge" style="background:{tc}">{st_lab}</span></div>'
            f'<div class="ana-row"><span>轨迹距离</span><b class="mono">{dist}</b></div></div>')
    analog_panel = "\n".join(analog_cards) or "<p>相似年份样本不足。</p>"

    fc_labels = [horizon_label(obs, t_now, h) for h in range(1, 13)]
    if oni_fc is None:
        oni_fc = [oni_point_forecast(anoms, t_now, h) for h in range(1, 13)]
    svg_fc = _svg_fc_curve(fc_labels, oni_fc)

    if heat is None:
        heat = skill_heatmap(anoms, states, ages, horizons=horizons)
    heat_html = _heatmap_table(heat, horizons)

    if calibs is None:
        calibs = {h: compute_calibration(anoms, states, ages, h) for h in horizons}
    rel_cells = []
    for i, h in enumerate(horizons):
        color = f"hsl({(i * 67 + 8) % 360},62%,42%)"
        svg_rel = _svg_reliability([(f"{h} 个月", calibs[h], color)])
        rel_cells.append(f'<div class="svg-box"><h3>{h} 个月（组合模型）</h3>{svg_rel}</div>')
    rel_panel = "\n".join(rel_cells)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ENSO 概率报告 · {latest.label}</title>
<style>{_REPORT_CSS}</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>ENSO 厄尔尼诺 / 拉尼娜概率报告</h1>
    <div class="sub">基于 NOAA CPC ONI v5（Niño3.4 海温距平，ERSSTv5）+ 转移模型 &amp; 历史相似型模型</div>
    <div class="meta">
      <span class="chip">数据 {obs[0].label} – {obs[-1].label}</span>
      <span class="chip">{len(obs)} 个月</span>
      <span class="chip">生成于 {gen_time}（本地时间）</span>
      <span class="chip">数据获取：{source}</span>
    </div>
  </header>

  <section class="grid now">
    <div class="card">
      <div class="label">最新 ONI（{latest.label}）</div>
      <div class="oni-val">{latest.anom:+.2f}</div>
    </div>
    <div class="card">
      <div class="label">当前状态</div>
      <span class="badge" style="background:{badge_bg}">{STATE_LABELS[state]}（{STATE_LABELS_EN[state]}）</span>
    </div>
    <div class="card">
      <div class="label">状态已持续</div>
      <div class="oni-val">{ages[t_now]}<span style="font-size:15px"> 个月</span></div>
    </div>
    <div class="card">
      <div class="label">判定规则</div>
      <div style="font-size:12.5px;color:#46576a;margin-top:2px">ONI 连续 5 个重叠三月 ≥ +0.5 为厄尔尼诺，≤ −0.5 为拉尼娜，否则中性</div>
    </div>
  </section>

  {alert}

  <h2>未来概率</h2>
  <section class="grid fc-grid">
    {cards}
  </section>

  <h2>最相似历史年份</h2>
  <p class="hint">当前近 6 个月 ONI 轨迹与下列历史时段最相似，展示它们 {analog_h} 个月后的实际走向（距离越小越相似）。</p>
  <section class="grid ana-grid">
    {analog_panel}
  </section>

  <h2>未来 ONI 点预测（相似型加权 · 距平值）</h2>
  <div class="svg-box">
    <h3>未来 12 个月（实线=加权均值，阴影=90% 区间，虚线=±0.5 阈值）</h3>
    {svg_fc}
  </div>

  <h2>ONI 历史序列</h2>
  <div class="legend">
    <span><i class="sw" style="background:#34465c"></i>ONI 值</span>
    <span><i class="sw" style="background:#e0472e;opacity:.55"></i>厄尔尼诺事件</span>
    <span><i class="sw" style="background:#1f6fb2;opacity:.55"></i>拉尼娜事件</span>
    <span><i class="sw" style="background:transparent;border-top:2px dashed #e0472e"></i>+0.5 阈值</span>
    <span><i class="sw" style="background:transparent;border-top:2px dashed #1f6fb2"></i>−0.5 阈值</span>
  </div>
  <div class="svg-box"><h3>全历史（{obs[0].year}–{obs[-1].year}）</h3>{svg_full}</div>
  <div class="svg-box"><h3>近 10 年</h3>{svg_recent}</div>

  <h2>模型回测技能</h2>
  <table class="skills">
    <thead><tr><th>提前期</th><th>评估数</th><th>转移 BS</th><th>相似 BS</th>
    <th>气候 BS</th><th>转移技能</th><th>相似技能</th></tr></thead>
    <tbody>{skill_table}</tbody>
  </table>

  <h2>技能热力图（起报月 × 提前期）</h2>
  <p class="hint">按月收缩后的组合模型在 1990 年以来各起报月上的 Brier 技能分。绿色=优于气候态；春季起报的技能通常显著低于秋冬起报（春季预报障碍）。</p>
  {heat_html}

  <h2>概率校准（可靠性曲线）</h2>
  <p class="hint">把组合模型的历史预测概率分桶，横轴=预测概率，纵轴=桶内实际发生频率。点越贴近对角线（虚线），概率越可信；点在对角线下方=模型高估该状态概率。</p>
  <div class="grid rel-grid">
    {rel_panel}
  </div>

  <footer>
    <p><b>方法</b>：状态转移模型（相位年龄分层 + Dirichlet 后验）与历史相似型模型（轨迹近邻核加权）按样本外回测
    Brier 技能分组合；两个模型均做"起报月分层 + 经验贝叶斯收缩"以缓解春季预报障碍；90% 区间来自 Dirichlet
    蒙特卡洛抽样；未来 ONI 点预测为相似型加权均值（90% 区间为加权分位）；可靠性曲线用于检验概率校准。</p>
    <p><b>说明</b>：概率为历史统计外推，不等于官方季节预报，也不构成决策依据。模型仅使用 ONI 单变量，
    未纳入次表层海温、风场与印度洋-太平洋耦合信息；技能分接近 0 的提前期仅可作弱信号参考。</p>
    <p>数据源：NOAA CPC · https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt ·
    本报告由 enso_prob.py report 生成，可随时用最新数据重新生成。</p>
  </footer>
</div>
</body>
</html>
"""
    return html

def cmd_report(args) -> int:
    obs, source = load_obs(refresh=args.refresh, data_file=args.data_file,
                           allow_network=not args.offline)
    anoms = [o.anom for o in obs]
    states = classify_states(anoms)
    ages = phase_ages(states)

    valid = [i for i, s in enumerate(states) if s is not None]
    if not valid:
        raise SystemExit("数据不足：至少需要 5 个月的有效 ONI 才能判定状态。")
    t_now = valid[-1]

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    entries = {}
    for h in horizons:
        comp = monthly_shrunk_components(anoms, states, ages, t_now, h)
        skills = compute_skills(anoms, states, ages, h)
        w, note = skill_weights(skills)
        analog_arg = None
        if comp["p_analog"] is not None:
            analog_arg = (comp["p_analog"], comp["analog_weights"],
                          comp["n_analog_eff"])
            if comp["n_analog_eff"] < 5:
                w = (1.0, 0.0)
        fc = blend_forecast(comp["p_trans"], analog_arg, w,
                            simulate=not args.no_bootstrap, alpha=comp["alpha"])
        entries[h] = (horizon_label(obs, t_now, h), fc, skills, w, note)

    html = build_report_html(obs, anoms, states, ages, t_now, horizons, entries, source)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"已生成可视化报告：{out}")
    print(f"  概率卡片 {len(horizons)} 个 · ONI 曲线 2 张 · 回测技能表 1 张")
    print("  用浏览器打开即可查看（含 90% 置信区间与事件色带）。")
    return 0

def cmd_selftest(args) -> int:
    return 0 if selftest() else 1

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="enso_prob.py",
        description="基于 NOAA ONI 数据估计厄尔尼诺/拉尼娜的出现概率",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：\n"
               "  python enso_prob.py update\n"
               "  python enso_prob.py forecast --chart --json\n"
               "  python enso_prob.py backtest --horizons 3,6,12\n"
               "  python enso_prob.py report\n"
               "  python enso_prob.py selftest\n",
    )
    sub = p.add_subparsers(dest="cmd")

    def add_common(sp):
        sp.add_argument("--data-file", help="本地 ONI ASCII 文件路径")
        sp.add_argument("--refresh", action="store_true", help="强制联网刷新")

    f = sub.add_parser("forecast", help="输出概率预测（默认命令）")
    add_common(f)
    f.add_argument("--horizons", default="3,6,9,12", help="提前期，月，逗号分隔")
    f.add_argument("--chart", action="store_true", help="附带最近 72 个月 ONI 柱状图")
    f.add_argument("--json", action="store_true", help="写出 output/forecast.json")
    f.add_argument("--no-bootstrap", action="store_true", help="跳过区间抽样（更快）")
    f.add_argument("--offline", action="store_true", help="禁止联网")

    u = sub.add_parser("update", help="抓取并缓存最新 ONI 数据")
    u.set_defaults(func=cmd_update)

    b = sub.add_parser("backtest", help="查看模型历史技能")
    add_common(b)
    b.add_argument("--horizons", default="3,6,9,12")

    r = sub.add_parser("report", help="生成可视化 HTML 报告")
    add_common(r)
    r.add_argument("--horizons", default="3,6,9,12", help="概率卡片提前期，月，逗号分隔")
    r.add_argument("--out", default=str(OUT_DIR / "enso_report.html"), help="输出 HTML 路径")
    r.add_argument("--no-bootstrap", action="store_true", help="跳过区间抽样（更快）")
    r.add_argument("--offline", action="store_true", help="禁止联网")

    s = sub.add_parser("selftest", help="运行内置自测")
    s.set_defaults(func=cmd_selftest)

    f.set_defaults(func=cmd_forecast)
    b.set_defaults(func=cmd_backtest)
    r.set_defaults(func=cmd_report)
    return p

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        args = parser.parse_args(["forecast"])
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())
