#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

# 状态编码：-1 拉尼娜，0 中性，1 厄尔尼诺；数组下标 = 状态 + 1
LA_NINA, NEUTRAL, EL_NINO = -1, 0, 1
STATE_LABELS = {EL_NINO: "厄尔尼诺", NEUTRAL: "中性", LA_NINA: "拉尼娜"}
STATE_LABELS_EN = {EL_NINO: "El Nino", NEUTRAL: "Neutral", LA_NINA: "La Nina"}
STATE_ORDER = (EL_NINO, NEUTRAL, LA_NINA)  # 展示顺序

ONI_THRESHOLD = 0.5
EVENT_WINDOW = 5  # 官方规则：连续 5 个重叠三月


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
        except Exception as exc:  # 网络异常时退回缓存
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
                            age_boost: float = 0.3) -> Tuple[List[float], List[float], float]:
    """返回 (三状态概率, Dirichlet 后验参数, 有效转移样本量)。

    age_boost=1.0 表示不使用相位年龄分层；<1 表示非同龄样本降权。
    """
    cur = states[t_now]
    counts = [0.0, 0.0, 0.0]      # 同龄样本
    marg = [0.0, 0.0, 0.0]        # 无条件边际（用作先验）
    bucket_now = _age_bucket(ages[t_now]) if cur is not None else -1

    n_pairs = len(states) - horizon
    for t in range(n_pairs):
        if t + horizon > t_now:
            break
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

    # Dirichlet 后验参数 = 观测计数 + 先验参数
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
                        bw_floor: float = 0.25
                        ) -> Optional[Tuple[List[float], List[float], float]]:
    """返回 (三状态概率, 各近邻权重, Kish 有效样本量)；样本不足时返回 None。"""
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

    # Silverman 带宽（取历史距离的标准差），并设下限避免退化
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

    # 阈值临近提示：官方规则有 5 个月滞后，当前状态常落后于实际演变
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

        p_trans, alpha, n_trans = transition_distribution(states, ages, t_now, h)
        analog = analog_distribution(anoms, states, t_now, h)
        skills = compute_skills(anoms, states, ages, h)
        w, note = skill_weights(skills)
        if analog is None:
            w = (1.0, 0.0)

        fc = blend_forecast(p_trans, analog, w, simulate=do_sim, alpha=alpha)
        intervals = fc["interval"]

        for s in STATE_ORDER:
            i = s + 1  # 概率数组下标 = 状态编码 + 1
            lo, hi = intervals[i] if intervals else (None, None)
            print(_oneline(STATE_LABELS[s], fc["p"][i], lo, hi))

        wt, wa = fc["blend"]
        print()
        print(f"  模型权重   转移模型 {wt:.2f} / 历史相似型 {wa:.2f}")
        sk_t, sk_a = skills["skill_trans"], skills["skill_analog"]
        sk_txt = f"转移 {sk_t:+.3f}" if sk_t is not None else "转移 n/a"
        sk_txt += f" / 相似 {sk_a:+.3f}" if sk_a is not None else " / 相似 n/a"
        print(f"  回测技能   {sk_txt}   (Brier 技能分, 滚动回测 n={skills['n_eval']})")
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

    # 1. 官方规则：连续 5 个月 >= 0.5
    s = classify_states([0.6] * 5)
    checks.append(("规则-厄尔尼诺判定",
                   s[4] == EL_NINO and s[3] is None,
                   f"states={s}"))
    s = classify_states([0.6, 0.6, 0.6, 0.6, 0.4])
    checks.append(("规则-不足 5 月不触发", s[4] == NEUTRAL, f"state={s[4]}"))
    s = classify_states([-0.7] * 5)
    checks.append(("规则-拉尼娜判定", s[4] == LA_NINA, f"state={s[4]}"))

    # 2. 解析器
    sample = (" SEAS  YR   TOTAL   ANOM\n"
              "  DJF 1950  25.01  -1.32\n"
              "  JFM 1950  25.36  -1.20\n"
              "# comment\n")
    parsed = parse_oni_text(sample)
    checks.append(("解析器-列与排序", len(parsed) == 2 and parsed[0].season == "DJF",
                   f"n={len(parsed)}"))

    # 3. 合成 AR(1) 上的转移概率参数回收
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

    # 4. 蒙特卡洛对照：相似型模型应优于气候态
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

    # 5. 结构对称性：翻转 ONI 符号，厄尔尼诺与拉尼娜概率互换
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

    # 6. 概率归一 + 区间包住点估计
    fc = blend_forecast(p1, an1, (0.5, 0.5), simulate=True, alpha=a1, n_sim=800)
    total_p = sum(fc["p"])
    lo_ok = all(fc["interval"][i][0] <= fc["p"][i] <= fc["interval"][i][1] for i in range(3))
    checks.append(("概率归一/区间覆盖", abs(total_p - 1) < 1e-9 and lo_ok,
                   f"sum={total_p:.6f}, covered={lo_ok}"))

    # 7. 边界条件：缺失值与过短序列
    holey = list(synth[:200])
    holey[100] = None
    st_h = classify_states(holey)
    hole_ok = st_h[100] is None and st_h[99] is not None
    checks.append(("边界-缺失值传播", hole_ok, "None 会阻塞其后的 5 个月窗口"))

    short = classify_states([0.6, 0.7])
    checks.append(("边界-过短序列", all(v is None for v in short), f"states={short}"))

    # 8. 相位年龄单调性
    ages_ok = all(
        (ages[i] == ages[i - 1] + 1) if (i > 0 and st[i] is not None and st[i] == st[i - 1])
        else True
        for i in range(1, len(st))
    )
    checks.append(("相位年龄-递推正确", ages_ok, f"max_age={max(ages)}"))

    # 9. 回测技能自洽：skill = 1 - BS / BS_clim
    skills = compute_skills(synth, st, ages, 6, step=12, min_train=240, use_cache=False)
    consistent = (
        skills["skill_trans"] is not None
        and abs(skills["skill_trans"]
                - (1 - skills["brier_trans"] / skills["brier_clim"])) < 1e-12
    )
    checks.append(("回测-技能自洽", consistent,
                   f"Brier trans={skills['brier_trans']:.4f}, clim={skills['brier_clim']:.4f}"))

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

    s = sub.add_parser("selftest", help="运行内置自测")
    s.set_defaults(func=cmd_selftest)

    f.set_defaults(func=cmd_forecast)
    b.set_defaults(func=cmd_backtest)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        args = parser.parse_args(["forecast"])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
