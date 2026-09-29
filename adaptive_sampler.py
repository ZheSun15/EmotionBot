#!/usr/bin/env python3
"""
基于类别高斯过程 + UCB 的自适应 Trial 选择
按照《情感机器人实验方案》实现

目标：在尽量少的用户实验次数内，找到所有全局最优的参数组合。

核心流程（对应方案步骤 1-8）：
  步骤1  建立 10 维类别候选空间（2304 组合）
  步骤2  Hamming 距离空间填充选择种子 trial（不依赖评分）
  步骤3  完成种子 trial，收集多维度评分
  步骤4  定义类别核函数
  步骤5  自适应循环：训练 GP → 设动态 κ → 扫描未测试组合 → 选 UCB 最高 → 收集评分
  步骤6  用全部数据重训 GP，预测全空间
  步骤7  Top-K + 高反应区域聚类（基于 Hamming 距离）
  步骤8  输出结果

依赖：numpy, scipy（无 scipy 时回退到 numpy.linalg.solve，超参数用经验值）
"""

import itertools
import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# scipy 可选：有则用 Cholesky 求解 + L-BFGS-B 超参数优化
try:
    from scipy.linalg import cho_factor, cho_solve
    from scipy.optimize import minimize
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] scipy 未安装，使用 numpy 回退（超参数将用经验值，不做优化）")


def _json_default(obj):
    """numpy 类型 → Python 原生，用于 JSON 序列化。"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ════════════════════════════════════════════════════════════════
# 步骤1：实验变量定义与候选空间
# ════════════════════════════════════════════════════════════════

# 10 个类别变量及其水平（顺序对应向量维度 0..9）
VARIABLES = [
    ("Appearance",      ["毛绒", "金属"]),          # 2
    ("Movement",        ["静止", "轮式", "四足"]),  # 3
    ("Pitch",           ["高音调", "低音调"]),       # 2
    ("Speech_Rate",     ["快", "慢"]),              # 2
    ("Gaze_Following",  ["有", "无"]),              # 2
    ("Pupil_Change",    ["有", "无"]),              # 2
    ("Emotion_Display", ["有", "无"]),              # 2
    ("Agency",          ["退缩", "中性", "主动"]),  # 3
    ("Memory",          ["有", "无"]),              # 2
    ("Vulnerability",   ["有", "无"]),              # 2
]

VAR_NAMES = [v[0] for v in VARIABLES]
VAR_LEVELS = [v[1] for v in VARIABLES]
VAR_SIZES = [len(v[1]) for v in VARIABLES]
NDIM = len(VARIABLES)
TOTAL_COMBINATIONS = math.prod(VAR_SIZES)  # 2304

# 主动学习主目标维度；其余作为次要结果记录
SCORE_KEYS = [
    "positive", "companionship", "closeness", "understood",
    "presence", "cuteness", "intention",
]
TARGET_KEY = "positive"


def build_candidate_set() -> list[tuple[int, ...]]:
    """返回全部 2304 个组合（整数向量编码，每维 0..size-1）。"""
    return list(itertools.product(*[range(s) for s in VAR_SIZES]))


def vector_to_dict(vec: tuple[int, ...]) -> dict:
    """整数向量 → {变量名: 水平标签}。"""
    return {VAR_NAMES[i]: VAR_LEVELS[i][vec[i]] for i in range(NDIM)}


def vector_to_label(vec: tuple[int, ...]) -> str:
    """人类可读标签。"""
    d = vector_to_dict(vec)
    return " | ".join(f"{k}={v}" for k, v in d.items())


# ════════════════════════════════════════════════════════════════
# 步骤2：Hamming 距离空间填充种子选择
# ════════════════════════════════════════════════════════════════

def hamming_distance(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    """两个组合之间不同特征的个数。"""
    return sum(1 for i in range(NDIM) if a[i] != b[i])


def select_seeds(
    candidates: list[tuple[int, ...]],
    n_seed: int,
    rng: random.Random,
) -> list[tuple[int, ...]]:
    """
    贪心最大化最小 Hamming 距离的空间填充种子选择。

    准则（对应方案步骤2）：
      1. 每个变量的各个水平尽量均衡出现
      2. 不同种子组合之间尽可能不同（最大化最小 Hamming 距离）
      3. 尽量覆盖二阶变量组合
      4. 不根据预期评分选择

    实现：贪心。每一步从候选中选「最大化已选集合的最小 Hamming 距离」
    且「最能补齐水平/二阶覆盖均衡性」的点。综合打分排序。
    """
    if n_seed >= len(candidates):
        return list(candidates)

    selected: list[tuple[int, ...]] = []
    # 已选集合在各维度各水平上的出现次数
    level_counts = [np.zeros(s, dtype=int) for s in VAR_SIZES]
    # 已选集合覆盖的二阶组合集合
    covered_pairs: set[tuple] = set()
    # 候选索引（避免反复修改大列表）
    remaining = set(range(len(candidates)))

    # 第一个点：随机选（不依赖评分）
    first = rng.choice(list(remaining))
    selected.append(candidates[first])
    remaining.discard(first)
    _update_coverage(candidates[first], level_counts, covered_pairs, add=True)

    while len(selected) < n_seed and remaining:
        best_idx = None
        best_score = -1.0

        for idx in remaining:
            cand = candidates[idx]
            # 最小 Hamming 距离到已选集合
            min_hamming = min(hamming_distance(cand, s) for s in selected)
            # 水平均衡性：选此点后各维度水平分布的最大不均衡度（越小越好）
            balance_penalty = 0
            for d in range(NDIM):
                counts = level_counts[d].copy()
                counts[cand[d]] += 1
                balance_penalty += max(counts) - min(counts)
            # 二阶覆盖增益
            new_pairs = _count_new_pairs(cand, covered_pairs)

            # 综合打分：Hamming 距离为主，均衡与覆盖为辅
            score = min_hamming * 10.0 - balance_penalty * 0.5 + new_pairs * 0.3
            if score > best_score:
                best_score = score
                best_idx = idx

        chosen = candidates[best_idx]
        selected.append(chosen)
        remaining.discard(best_idx)
        _update_coverage(chosen, level_counts, covered_pairs, add=True)

    return selected


def _update_coverage(vec, level_counts, covered_pairs, add=True):
    for d in range(NDIM):
        if add:
            level_counts[d][vec[d]] += 1
    for d1 in range(NDIM):
        for d2 in range(d1 + 1, NDIM):
            pair = (d1, vec[d1], d2, vec[d2])
            if add:
                covered_pairs.add(pair)


def _count_new_pairs(vec, covered_pairs) -> int:
    count = 0
    for d1 in range(NDIM):
        for d2 in range(d1 + 1, NDIM):
            if (d1, vec[d1], d2, vec[d2]) not in covered_pairs:
                count += 1
    return count


# ════════════════════════════════════════════════════════════════
# 步骤4 + 步骤5.1：类别核高斯过程
# ════════════════════════════════════════════════════════════════

class CategoricalGP:
    """
    类别核高斯过程（对应方案步骤4的核函数 + 步骤5.1的预测公式）。

    核函数：
        k(x_i, x_j) = σ_f² · exp(-Σ_d θ_d · I(x_i[d] ≠ x_j[d]))

    预测（先验均值 m = ȳ）：
        μ(x*)  = m + k*ᵀ (K + σ_n²I)⁻¹ (y - m)
        σ²(x*) = k(x*,x*) - k*ᵀ (K + σ_n²I)⁻¹ k*

    超参数 θ_d, σ_f², σ_n² 通过最大化边际似然估计（有 scipy 时）。
    数值稳定性用 Cholesky 分解，不直接求矩阵逆。
    """

    def __init__(
        self,
        theta: Optional[np.ndarray] = None,
        sigma_f2: float = 1.0,
        sigma_n2: float = 0.1,
    ):
        self.theta = theta if theta is not None else np.ones(NDIM)
        self.sigma_f2 = sigma_f2
        self.sigma_n2 = sigma_n2
        self._X = None
        self._y = None
        self._mean = 0.0
        self._cho = None  # Cholesky 分解缓存

    # ── 核函数 ──
    def _kernel(self, Xa: np.ndarray, Xb: np.ndarray) -> np.ndarray:
        """计算核矩阵。Xa, Xb 形状 (n, D), (m, D)，返回 (n, m)。"""
        # Hamming 距离矩阵：不同特征个数
        # 利用广播：(n,1,D) vs (1,m,D) → (n,m,D)
        diff = (Xa[:, None, :] != Xb[None, :, :])  # bool (n,m,D)
        hamming = diff.astype(float) @ self.theta  # (n,m)
        return self.sigma_f2 * np.exp(-hamming)

    # ── 拟合 ──
    def fit(self, X: np.ndarray, y: np.ndarray):
        """X: (n,D) int, y: (n,) float."""
        self._X = X
        self._y = y
        self._mean = float(np.mean(y)) if len(y) > 0 else 0.0
        K = self._kernel(X, X)
        K_noise = K + self.sigma_n2 * np.eye(len(X))
        # Cholesky 分解
        try:
            if HAS_SCIPY:
                self._cho = cho_factor(K_noise, lower=True)
            else:
                self._L = np.linalg.cholesky(K_noise)
                self._cho = None
        except np.linalg.LinAlgError:
            # 数值不稳定时加抖动
            K_noise += 1e-6 * np.eye(len(X))
            if HAS_SCIPY:
                self._cho = cho_factor(K_noise, lower=True)
            else:
                self._L = np.linalg.cholesky(K_noise)

    # ── 预测 ──
    def predict(self, X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回 (μ, σ)，形状均为 (m,)。"""
        if self._X is None or len(self._X) == 0:
            return np.full(len(X_new), self._mean), np.full(len(X_new), math.sqrt(self.sigma_f2))

        k_star = self._kernel(X_new, self._X)  # (m, n)
        y_centered = self._y - self._mean

        if HAS_SCIPY and self._cho is not None:
            alpha = cho_solve(self._cho, y_centered)         # (K+σ²I)⁻¹(y-m)
            mu = self._mean + k_star @ alpha                 # (m,)
            v = cho_solve(self._cho, k_star.T)               # (K+σ²I)⁻¹ k*  → (n,m)
            k_ss = np.array([self.sigma_f2] * len(X_new))    # 对角：k(x*,x*)
            var = k_ss - np.einsum("mi,im->m", k_star, v)    # (m,)
        else:
            K_noise = self._kernel(self._X, self._X) + self.sigma_n2 * np.eye(len(self._X))
            solved = np.linalg.solve(K_noise, y_centered)
            mu = self._mean + k_star @ solved
            v = np.linalg.solve(K_noise, k_star.T)
            k_ss = np.array([self.sigma_f2] * len(X_new))
            var = k_ss - np.einsum("mi,im->m", k_star, v)

        var = np.maximum(var, 0.0)  # 方差非负
        sigma = np.sqrt(var)
        return mu, sigma

    # ── 超参数优化（最大化边际似然）──
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, n_restarts: int = 3):
        """通过最大化 log marginal likelihood 估计 θ, σ_f², σ_n²。"""
        if not HAS_SCIPY or len(X) < 5:
            return  # 数据太少或无 scipy，用经验值

        def neg_lml(log_params):
            theta = np.exp(log_params[:NDIM])
            sf2 = math.exp(log_params[NDIM])
            sn2 = math.exp(log_params[NDIM + 1])
            K = sf2 * np.exp(-(X[:, None, :] != X[None, :, :]).astype(float) @ theta)
            K_noise = K + sn2 * np.eye(len(X)) + 1e-6 * np.eye(len(X))
            try:
                L = np.linalg.cholesky(K_noise)
            except np.linalg.LinAlgError:
                return 1e10
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y - y.mean()))
            lml = -0.5 * (y - y.mean()) @ alpha - np.sum(np.log(np.diag(L))) - len(X) / 2 * math.log(2 * math.pi)
            return -lml

        best_val = np.inf
        best_params = None
        rng = np.random.default_rng(42)
        for _ in range(n_restarts):
            x0 = rng.normal(0, 0.5, size=NDIM + 2)
            try:
                res = minimize(neg_lml, x0, method="L-BFGS-B",
                               bounds=[(-3, 3)] * (NDIM + 2))
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x
            except Exception:
                continue

        if best_params is not None:
            self.theta = np.exp(best_params[:NDIM])
            self.sigma_f2 = math.exp(best_params[NDIM])
            self.sigma_n2 = math.exp(best_params[NDIM + 1])

        # 重新拟合
        self.fit(X, y)


# ════════════════════════════════════════════════════════════════
# 数据结构
# ════════════════════════════════════════════════════════════════

@dataclass
class TrialRecord:
    """单次 trial 的完整记录。"""
    trial_id: int
    vector: tuple[int, ...]
    phase: str  # "seed" | "adaptive"
    scores: dict  # 所有评分维度
    mu_pred: float = 0.0       # 选择此 trial 时模型的预测分
    sigma_pred: float = 0.0    # 选择时的不确定性
    priority: float = 0.0      # 选择时的 UCB 优先级
    kappa: float = 0.0         # 选择时的 κ
    timestamp: str = ""


# ════════════════════════════════════════════════════════════════
# 步骤5.2-5.5：自适应循环
# ════════════════════════════════════════════════════════════════

def get_kappa(t: int, n_adaptive: int, k_high=2.0, k_middle=1.0, k_low=0.5) -> float:
    """动态 κ：前期探索 → 中期兼顾 → 后期利用。"""
    if t <= n_adaptive / 3:
        return k_high
    elif t <= 2 * n_adaptive / 3:
        return k_middle
    else:
        return k_low


def run_adaptive(
    candidates: list[tuple[int, ...]],
    trial_data: list[TrialRecord],
    ask_fn: Callable[[tuple[int, ...]], dict],
    n_adaptive: int,
    participant_info: Optional[dict] = None,
    verbose: bool = True,
) -> list[TrialRecord]:
    """
    自适应 trial 循环（方案步骤5）。

    每轮：
      5.1 用现有数据训练类别高斯过程（含超参数优化）
      5.2 设置当前阶段 κ
      5.3 扫描所有未测试组合，计算 μ, σ, UCB 优先级
      5.4 选优先级最高的组合，呈现给被试，收集评分
      5.5 更新数据
    """
    tested_vectors = {r.vector for r in trial_data}
    all_vectors = candidates
    trial_counter = len(trial_data)

    for t in range(1, n_adaptive + 1):
        # 5.1 训练 GP
        X = np.array([list(r.vector) for r in trial_data], dtype=int)
        y = np.array([r.scores[TARGET_KEY] for r in trial_data], dtype=float)

        gp = CategoricalGP()
        gp.optimize_hyperparameters(X, y)
        gp.fit(X, y)

        # 5.2 动态 κ
        kappa = get_kappa(t, n_adaptive)

        # 5.3 扫描未测试组合
        untested = [v for v in all_vectors if v not in tested_vectors]
        if not untested:
            if verbose:
                print(f"  [Adaptive {t}] 所有组合已测试完毕，提前结束。")
            break

        X_untested = np.array([list(v) for v in untested], dtype=int)
        mu, sigma = gp.predict(X_untested)
        priority = mu + kappa * sigma

        # 选优先级最高
        best_idx = int(np.argmax(priority))
        best_vec = untested[best_idx]

        if verbose:
            print(f"  [Adaptive {t:2d}/{n_adaptive}] κ={kappa:.1f} | "
                  f"选中: μ={mu[best_idx]:.3f} σ={sigma[best_idx]:.3f} "
                  f"UCB={priority[best_idx]:.3f} | {vector_to_label(best_vec)}")

        # 5.4 呈现并收集评分
        scores = ask_fn(best_vec, participant_info)

        # 5.5 记录
        trial_counter += 1
        record = TrialRecord(
            trial_id=trial_counter,
            vector=best_vec,
            phase="adaptive",
            scores=scores,
            mu_pred=float(mu[best_idx]),
            sigma_pred=float(sigma[best_idx]),
            priority=float(priority[best_idx]),
            kappa=kappa,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        trial_data.append(record)
        tested_vectors.add(best_vec)

    return trial_data


# ════════════════════════════════════════════════════════════════
# 步骤6-7：最终模型 + 高反应区域聚类
# ════════════════════════════════════════════════════════════════

def finalize_and_rank(
    candidates: list[tuple[int, ...]],
    trial_data: list[TrialRecord],
) -> tuple[list[tuple[tuple[int, ...], float, float]], CategoricalGP]:
    """
    步骤6：用全部数据重训 GP，预测全部 2304 组合。
    返回 [(vector, mu_final, sigma_final), ...] 按 mu_final 降序。
    """
    X = np.array([list(r.vector) for r in trial_data], dtype=int)
    y = np.array([r.scores[TARGET_KEY] for r in trial_data], dtype=float)

    gp = CategoricalGP()
    gp.optimize_hyperparameters(X, y)
    gp.fit(X, y)

    X_all = np.array([list(v) for v in candidates], dtype=int)
    mu_all, sigma_all = gp.predict(X_all)

    ranked = sorted(
        zip(candidates, mu_all.tolist(), sigma_all.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked, gp


def cluster_high_response(
    ranked: list[tuple[tuple[int, ...], float, float]],
    top_percent: float = 5.0,
    max_clusters: int = 6,
) -> list[dict]:
    """
    步骤7：取预测评分前 top_percent 的组合，用 Hamming 距离层次聚类，
    输出多个高反应设计模式。
    """
    n_high = max(1, int(len(ranked) * top_percent / 100))
    high_set = ranked[:n_high]
    high_vecs = [r[0] for r in high_set]
    high_scores = [r[1] for r in high_set]
    high_sigmas = [r[2] for r in high_set]

    if len(high_vecs) <= 1:
        return [_summarize_cluster(high_vecs, high_scores, high_sigmas, 0)]

    # Hamming 距离矩阵 → 层次聚类
    vecs_arr = np.array([list(v) for v in high_vecs])
    if HAS_SCIPY:
        dists = pdist(vecs_arr, metric="hamming") * NDIM  # Hamming 距离
        Z = linkage(dists, method="average")
        labels = fcluster(Z, t=max_clusters, criterion="maxclust")
    else:
        # 回退：贪心聚类
        labels = _greedy_cluster(high_vecs, max_clusters)

    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append(i)

    results = []
    for cid, indices in sorted(clusters.items(), key=lambda x: -np.mean([high_scores[i] for i in x[1]])):
        c_vecs = [high_vecs[i] for i in indices]
        c_scores = [high_scores[i] for i in indices]
        c_sigmas = [high_sigmas[i] for i in indices]
        results.append(_summarize_cluster(c_vecs, c_scores, c_sigmas, cid))

    return results


def _greedy_cluster(vecs, max_clusters):
    """无 scipy 时的简单聚类：按 Hamming 距离贪心分组，合并到最近簇。"""
    if not vecs:
        return []
    labels = [1] * len(vecs)
    cluster_centers = [vecs[0]]
    for i in range(1, len(vecs)):
        # 找到距离最近的已有簇中心
        dists = [hamming_distance(vecs[i], c) for c in cluster_centers]
        nearest = int(np.argmin(dists))
        # 只有簇数已达上限，或者距离足够近时才合并；否则新建簇
        if len(cluster_centers) >= max_clusters or dists[nearest] <= 3:
            labels[i] = nearest + 1
        else:
            cluster_centers.append(vecs[i])
            labels[i] = len(cluster_centers)
    return labels


def _summarize_cluster(vecs, scores, sigmas, cid) -> dict:
    """汇总一个高反应聚类的特征。"""
    best_idx = int(np.argmax(scores))
    # 找反复出现的特征（所有组合一致的水平）
    consistent = {}
    for d in range(NDIM):
        vals = [v[d] for v in vecs]
        if len(set(vals)) == 1:
            consistent[VAR_NAMES[d]] = VAR_LEVELS[d][vals[0]]

    return {
        "cluster_id": cid,
        "size": len(vecs),
        "representative": vector_to_dict(vecs[best_idx]),
        "representative_score": scores[best_idx],
        "mean_score": float(np.mean(scores)),
        "mean_sigma": float(np.mean(sigmas)),
        "consistent_features": consistent,  # 该类别中所有组合共有的特征
        "all_combinations": [vector_to_dict(v) for v in vecs],
    }


# ════════════════════════════════════════════════════════════════
# 评分接口
# ════════════════════════════════════════════════════════════════

def ask_score_human(vec: tuple[int, ...], participant_info: Optional[dict] = None) -> dict:
    """真实人类实验评分接口。"""
    print(f"\n{'─'*50}")
    print(f"  机器人组合：")
    for name, levels, val in zip(VAR_NAMES, VAR_LEVELS, vec):
        print(f"    {name:18s} = {levels[val]}")
    print(f"{'─'*50}")

    scores = {}
    prompts = {
        "positive":      "积极情绪评分 (1-7): ",
        "companionship": "陪伴感评分 (1-7): ",
        "closeness":     "亲近感评分 (1-7): ",
        "understood":    "被理解感评分 (1-7): ",
        "presence":      "社交存在感评分 (1-7): ",
        "cuteness":      "可爱感评分 (1-7): ",
        "intention":     "使用意愿评分 (1-7): ",
    }
    for key in SCORE_KEYS:
        while True:
            try:
                val = float(input(prompts[key]))
                if 1 <= val <= 7:
                    scores[key] = val
                    break
                print("  请输入 1-7 之间的数字。")
            except ValueError:
                print("  无效输入，请输入数字。")
    return scores


def true_response(vec: tuple[int, ...]) -> float:
    """
    确定性响应面（无噪声）——验证用的 ground truth。

    设计原则：效应适度、高分区域窄且分离，避免天花板效应。
    3 个分离的高反应区域：
      - 毛绒 + 四足 → 高分区域 1（强交互，窄峰）
      - 高音调 + 有眼神跟随 + 有记忆 → 高分区域 2（三阶交互，窄峰）
      - 主动 Agency + 有情感显示 → 高分区域 3（二阶交互，窄峰）
    其余区域为低分或中等。
    """
    base = 3.8
    # 主效应（适度，单独不会触顶）
    if vec[0] == 0:  base += 0.25  # 毛绒
    if vec[1] == 2:  base += 0.30  # 四足
    if vec[2] == 0:  base += 0.15  # 高音调
    if vec[3] == 1:  base += 0.10  # 慢语速
    if vec[4] == 0:  base += 0.15  # 有眼神跟随
    if vec[5] == 0:  base += 0.10  # 有瞳孔变化
    if vec[6] == 0:  base += 0.20  # 有情感显示
    if vec[7] == 2:  base += 0.25  # 主动
    if vec[8] == 0:  base += 0.20  # 有记忆
    if vec[9] == 0:  base += 0.10  # 有脆弱性
    # 交互效应（只有同时命中交互 + 多个主效应才接近顶部）
    if vec[0] == 0 and vec[1] == 2:       base += 0.8   # 毛绒×四足：区域1 核心
    if vec[2] == 0 and vec[4] == 0 and vec[8] == 0:  base += 0.7  # 高音×眼神×记忆：区域2 核心
    if vec[7] == 2 and vec[6] == 0:       base += 0.6   # 主动×情感显示：区域3 核心
    # 负交互
    if vec[0] == 1 and vec[1] == 0:       base -= 0.5   # 金属×静止
    if vec[7] == 0:                       base -= 0.3   # 退缩
    return max(1.0, min(7.0, base))


def mock_score(vec: tuple[int, ...], participant_info: Optional[dict] = None,
               noise_std: float = 0.3, rng: Optional[np.random.Generator] = None) -> dict:
    """
    模拟有噪声的被试评分（用于离线验证算法，正式实验替换为 ask_score_human）。
    在 true_response 基础上加高斯噪声，模拟被试评分的随机波动。
    """
    if rng is None:
        rng = np.random.default_rng()
    true_val = true_response(vec)
    observed = max(1.0, min(7.0, true_val + rng.normal(0, noise_std)))
    scores = {key: max(1.0, min(7.0, true_val + rng.normal(0, noise_std))) for key in SCORE_KEYS}
    scores[TARGET_KEY] = observed
    return scores


# ════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════

def run_experiment(
    ask_fn: Callable,
    n_seed: int = 24,
    n_adaptive: int = 56,
    participant_info: Optional[dict] = None,
    output_path: str = "D:/BIGAI/Code/EmotionBot/experiment_results.json",
    verbose: bool = True,
) -> dict:
    """运行完整实验流程（步骤1-8）。"""

    print("=" * 60)
    print("  情感机器人自适应实验 — 类别GP + UCB")
    print("=" * 60)
    print(f"  变量数: {NDIM} | 候选组合: {TOTAL_COMBINATIONS}")
    print(f"  种子 trial: {n_seed} | 自适应 trial: {n_adaptive}")
    print(f"  总实验次数: {n_seed + n_adaptive} / {TOTAL_COMBINATIONS}")
    print(f"  评分接口: {'模拟' if ask_fn is mock_score else '人类'}")
    print()

    # 步骤1：候选空间
    candidates = build_candidate_set()
    print(f"  [步骤1] 候选空间构建完成：{len(candidates)} 组合")

    # 步骤2：种子选择
    rng = random.Random(42)
    seeds = select_seeds(candidates, n_seed, rng)
    print(f"  [步骤2] 种子选择完成（Hamming 空间填充）：{len(seeds)} 个")
    if verbose:
        # 验证种子间的最小 Hamming 距离
        min_d = min(hamming_distance(seeds[i], seeds[j])
                    for i in range(len(seeds)) for j in range(i + 1, len(seeds)))
        print(f"          种子间最小 Hamming 距离 = {min_d}")

    # 步骤3：种子 trial
    print(f"\n  [步骤3] 种子 trial 阶段")
    trial_data: list[TrialRecord] = []
    for i, vec in enumerate(seeds):
        scores = ask_fn(vec, participant_info)
        trial_data.append(TrialRecord(
            trial_id=i + 1, vector=vec, phase="seed", scores=scores,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        ))
        if verbose:
            print(f"    Seed {i+1:2d}/{n_seed}: {vector_to_label(vec)} → "
                  f"positive={scores[TARGET_KEY]:.2f}")

    # 步骤5：自适应循环
    print(f"\n  [步骤5] 自适应 trial 阶段")
    trial_data = run_adaptive(
        candidates, trial_data, ask_fn, n_adaptive,
        participant_info, verbose,
    )

    # 步骤6：最终模型 + 全空间预测
    print(f"\n  [步骤6] 最终模型预测全空间")
    ranked, final_gp = finalize_and_rank(candidates, trial_data)
    print(f"          GP 超参数: θ={[f'{t:.2f}' for t in final_gp.theta]}, "
          f"σ_f²={final_gp.sigma_f2:.3f}, σ_n²={final_gp.sigma_n2:.3f}")

    # 步骤7：高反应区域聚类
    print(f"\n  [步骤7] 高反应区域聚类")
    clusters = cluster_high_response(ranked, top_percent=5.0, max_clusters=6)
    print(f"          发现 {len(clusters)} 个高反应设计模式")
    for c in clusters:
        print(f"            ─ 聚类 {c['cluster_id']}: {c['size']} 组合, "
              f"均分={c['mean_score']:.3f}, 共有特征={c['consistent_features']}")

    # 步骤8：输出
    print(f"\n  [步骤8] 输出结果")

    # 真实最优（如果测试过的话）
    tested_best = max(trial_data, key=lambda r: r.scores[TARGET_KEY])

    result = {
        "config": {
            "n_seed": n_seed,
            "n_adaptive": n_adaptive,
            "total_trials": len(trial_data),
            "total_combinations": TOTAL_COMBINATIONS,
            "coverage_ratio": len(trial_data) / TOTAL_COMBINATIONS,
            "gp_hyperparams": {
                "theta": final_gp.theta.tolist(),
                "sigma_f2": final_gp.sigma_f2,
                "sigma_n2": final_gp.sigma_n2,
            },
        },
        "trial_data": [
            {
                "trial_id": r.trial_id,
                "phase": r.phase,
                "vector": list(r.vector),
                "combo": vector_to_dict(r.vector),
                "scores": r.scores,
                "mu_pred": r.mu_pred,
                "sigma_pred": r.sigma_pred,
                "priority": r.priority,
                "kappa": r.kappa,
                "timestamp": r.timestamp,
            }
            for r in trial_data
        ],
        "top_combinations": [
            {"combo": vector_to_dict(v), "mu": mu, "sigma": sig}
            for v, mu, sig in ranked[:20]
        ],
        "best_tested": {
            "combo": vector_to_dict(tested_best.vector),
            "score": tested_best.scores[TARGET_KEY],
            "trial_id": tested_best.trial_id,
        },
        "high_response_modes": clusters,
        "full_ranking": [
            {"vector": list(v), "mu": mu, "sigma": sig}
            for v, mu, sig in ranked
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=_json_default)

    print(f"\n  结果已保存: {output_path}")
    print(f"\n{'='*60}")
    print(f"  实验完成")
    print(f"{'='*60}")
    print(f"  总 trial 数: {len(trial_data)} / {TOTAL_COMBINATIONS} "
          f"({len(trial_data)/TOTAL_COMBINATIONS*100:.1f}% 覆盖)")
    print(f"  实测最高分: {tested_best.scores[TARGET_KEY]:.3f}")
    print(f"    → {vector_to_label(tested_best.vector)}")
    print(f"  预测最高分: {ranked[0][1]:.3f}")
    print(f"    → {vector_to_label(ranked[0][0])}")
    print(f"  高反应模式数: {len(clusters)}")
    print(f"{'='*60}")

    return result


# ── 命令行入口 ──

def main():
    import sys

    use_mock = "--mock" in sys.argv
    n_seed = 24
    n_adaptive = 56

    # 解析参数
    for arg in sys.argv[1:]:
        if arg.startswith("--seed="):
            n_seed = int(arg.split("=")[1])
        elif arg.startswith("--adaptive="):
            n_adaptive = int(arg.split("=")[1])

    ask_fn = mock_score if use_mock else ask_score_human

    if use_mock:
        print("  [MOCK 模式] 使用模拟评分函数验证算法\n")
    else:
        print("  [人类实验模式] 将向被试收集真实评分\n")

    run_experiment(
        ask_fn=ask_fn,
        n_seed=n_seed,
        n_adaptive=n_adaptive,
        output_path="D:/BIGAI/Code/EmotionBot/experiment_results.json",
    )


if __name__ == "__main__":
    main()
