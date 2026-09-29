#!/usr/bin/env python3
"""
验证 adaptive_sampler.py 的算法正确性。

验证逻辑：
  1. 穷举全部 2304 组合的 true_response（无噪声）→ ground truth ranking
  2. 多次运行算法（不同随机种子，有噪声观测），每次对比预测 vs ground truth
  3. 量化指标：
     - Top-1 准确率：预测最高分组合是否在真实 Top-1% 内
     - Top-K 召回率：真实 Top-20 里找回了多少
     - 全空间预测误差 RMSE
     - Spearman 秩相关：预测排序 vs 真实排序
     - 高反应区域识别率：3 个预设高分区域是否都被聚类覆盖
  4. 消融实验：
     - 不同 κ 策略（动态 vs 固定 vs 随机）
     - 不同 budget（种子数 × 自适应数）
     - 不同噪声水平
"""

import itertools
import sys
import warnings
from typing import Callable

import numpy as np
from scipy.stats import spearmanr

from adaptive_sampler import (
    build_candidate_set, true_response, mock_score, run_experiment,
    CategoricalGP, TOTAL_COMBINATIONS, VAR_NAMES, VAR_LEVELS, NDIM,
    TARGET_KEY, SCORE_KEYS, select_seeds, hamming_distance,
    vector_to_dict, vector_to_label,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ════════════════════════════════════════════════════════════════
# 1. 构建 ground truth
# ════════════════════════════════════════════════════════════════

def build_ground_truth() -> dict:
    """穷举全空间，返回 ground truth 排序和相关统计。"""
    candidates = build_candidate_set()
    true_scores = np.array([true_response(v) for v in candidates])
    true_ranking = np.argsort(-true_scores)  # 降序索引

    # 识别 3 个预设高分区域的 ground truth
    # 区域1: 毛绒(0) + 四足(2)
    region1 = [i for i, v in enumerate(candidates) if v[0] == 0 and v[1] == 2]
    # 区域2: 高音调(0) + 有眼神(0) + 有记忆(0)
    region2 = [i for i, v in enumerate(candidates) if v[2] == 0 and v[4] == 0 and v[8] == 0]
    # 区域3: 主动(2) + 有情感显示(0)
    region3 = [i for i, v in enumerate(candidates) if v[7] == 2 and v[6] == 0]

    return {
        "candidates": candidates,
        "true_scores": true_scores,
        "true_ranking": true_ranking,
        "true_top1": true_ranking[0],
        "true_top20": set(true_ranking[:20].tolist()),
        "true_top5pct": set(true_ranking[:int(TOTAL_COMBINATIONS * 0.05)].tolist()),
        "regions": {"R1_毛绒四足": region1, "R2_高音眼神记忆": region2, "R3_主动情感": region3},
        "region_scores": {
            "R1": float(np.mean([true_scores[i] for i in region1])),
            "R2": float(np.mean([true_scores[i] for i in region2])),
            "R3": float(np.mean([true_scores[i] for i in region3])),
        },
    }


# ════════════════════════════════════════════════════════════════
# 2. 单次验证运行
# ════════════════════════════════════════════════════════════════

def make_noisy_ask_fn(seed: int, noise_std: float = 0.3) -> Callable:
    """生成一个带固定随机种子的噪声评分函数。"""
    rng = np.random.default_rng(seed)
    def ask_fn(vec, participant_info=None):
        return mock_score(vec, participant_info, noise_std=noise_std, rng=rng)
    return ask_fn


def single_run_eval(
    gt: dict,
    seed: int = 42,
    n_seed: int = 24,
    n_adaptive: int = 56,
    noise_std: float = 0.3,
    kappa_strategy: str = "dynamic",  # dynamic | fixed_high | fixed_low | random
    verbose: bool = False,
) -> dict:
    """
    跑一次完整实验，返回量化指标。
    通过 monkey-patch get_kappa 来测试不同 κ 策略。
    """
    import adaptive_sampler as AS

    # Monkey-patch κ 策略
    original_get_kappa = AS.get_kappa
    if kappa_strategy == "fixed_high":
        AS.get_kappa = lambda t, n, **kw: 2.0
    elif kappa_strategy == "fixed_low":
        AS.get_kappa = lambda t, n, **kw: 0.5
    elif kappa_strategy == "fixed_mid":
        AS.get_kappa = lambda t, n, **kw: 1.0
    elif kappa_strategy == "random":
        rng_k = np.random.default_rng(seed + 999)
        AS.get_kappa = lambda t, n, **kw: float(rng_k.choice([0.5, 1.0, 2.0]))
    # dynamic: 不改

    ask_fn = make_noisy_ask_fn(seed, noise_std)
    result = run_experiment(
        ask_fn=ask_fn,
        n_seed=n_seed,
        n_adaptive=n_adaptive,
        output_path="D:/BIGAI/Code/EmotionBot/_validation_tmp.json",
        verbose=verbose,
    )

    AS.get_kappa = original_get_kappa  # 恢复

    # 计算指标
    return _compute_metrics(result, gt)


def _compute_metrics(result: dict, gt: dict) -> dict:
    """对比算法预测与 ground truth，返回各项指标。"""
    candidates = gt["candidates"]
    true_scores = gt["true_scores"]
    true_ranking = gt["true_ranking"]

    # 从 result 中提取预测
    full_ranking = result["full_ranking"]
    pred_vectors = [tuple(v["vector"]) for v in full_ranking]
    pred_mu = np.array([v["mu"] for v in full_ranking])
    pred_sigma = np.array([v["sigma"] for v in full_ranking])

    # 建立预测分到候选索引的映射
    cand_to_idx = {v: i for i, v in enumerate(candidates)}
    pred_true_scores = np.array([true_scores[cand_to_idx[v]] for v in pred_vectors])

    # ── 指标1: Top-1 准确率 ──
    # 预测最高分组合是否在真实 Top-1% 内
    pred_top1_idx = int(np.argmax(pred_mu))
    pred_top1_true_score = pred_true_scores[pred_top1_idx]
    pred_top1_in_top5pct = cand_to_idx[pred_vectors[pred_top1_idx]] in gt["true_top5pct"]

    # ── 指标2: Top-K 召回率（用 Top-5% 更稳定）──
    top_k = int(TOTAL_COMBINATIONS * 0.05)  # 115
    pred_topk_indices = set()
    for v in pred_vectors[:top_k]:
        pred_topk_indices.add(cand_to_idx[v])
    true_topk = set(gt["true_ranking"][:top_k].tolist())
    topk_recall = len(pred_topk_indices & true_topk) / top_k

    # Top-20 也保留（窄范围）
    pred_top20_indices = set()
    for v in pred_vectors[:20]:
        pred_top20_indices.add(cand_to_idx[v])
    top20_recall = len(pred_top20_indices & gt["true_top20"]) / 20

    # ── 指标3: 全空间预测 RMSE ──
    rmse = float(np.sqrt(np.mean((pred_mu - pred_true_scores) ** 2)))

    # ── 指标4: Spearman 秩相关 ──
    pred_rank = np.argsort(-pred_mu)
    rho, _ = spearmanr(pred_rank, true_ranking)

    # ── 指标5: 高反应区域识别率 ──
    # 检查每个预设高分区域是否有至少一个组合出现在预测 Top-5% 内
    pred_top5pct_indices = set()
    for v in pred_vectors[:int(TOTAL_COMBINATIONS * 0.05)]:
        pred_top5pct_indices.add(cand_to_idx[v])
    regions_hit = 0
    region_details = {}
    for rname, region_indices in gt["regions"].items():
        hit = any(i in pred_top5pct_indices for i in region_indices)
        region_details[rname] = hit
        if hit:
            regions_hit += 1
    region_recall = regions_hit / len(gt["regions"])

    # ── 指标6: 实测最优 vs 真实最优 ──
    best_tested = result["best_tested"]
    best_tested_vec = tuple(best_tested["combo"].get(None, []))  # 不用这个
    # 从 trial_data 里找
    trial_scores = [t["scores"][TARGET_KEY] for t in result["trial_data"]]
    best_trial_score = max(trial_scores)
    true_optimal = float(true_scores[gt["true_top1"]])
    optimality_gap = (true_optimal - best_trial_score) / true_optimal

    return {
        "top1_in_top5pct": pred_top1_in_top5pct,
        "top1_true_score": pred_top1_true_score,
        "top20_recall": top20_recall,
        "top5pct_recall": topk_recall,
        "rmse": rmse,
        "spearman_rho": float(rho),
        "region_recall": region_recall,
        "region_details": region_details,
        "best_tested_score": best_trial_score,
        "true_optimal": true_optimal,
        "optimality_gap": optimality_gap,
        "n_trials": len(result["trial_data"]),
    }


# ════════════════════════════════════════════════════════════════
# 3. 批量验证 + 统计
# ════════════════════════════════════════════════════════════════

def batch_eval(
    gt: dict,
    n_runs: int = 10,
    seeds: list[int] = None,
    **kwargs,
) -> dict:
    """多次运行取统计。"""
    if seeds is None:
        seeds = list(range(42, 42 + n_runs))

    results = []
    for i, seed in enumerate(seeds):
        r = single_run_eval(gt, seed=seed, verbose=False, **kwargs)
        results.append(r)
        print(f"    Run {i+1:2d}/{n_runs} (seed={seed}): "
              f"top5%={r['top5pct_recall']:.2f} top20={r['top20_recall']:.2f} "
              f"RMSE={r['rmse']:.3f} ρ={r['spearman_rho']:.3f} gap={r['optimality_gap']:.3f} "
              f"regions={r['region_recall']:.2f}")

    # 汇总统计
    metrics = ["top1_in_top5pct", "top20_recall", "top5pct_recall", "rmse", "spearman_rho",
               "region_recall", "optimality_gap"]
    summary = {}
    for m in metrics:
        vals = [r[m] for r in results]
        if m == "top1_in_top5pct":
            summary[m] = {"mean": float(np.mean(vals)), "hits": int(sum(vals))}
        else:
            summary[m] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
    summary["n_runs"] = n_runs
    summary["n_trials_per_run"] = results[0]["n_trials"]
    return {"summary": summary, "runs": results}


# ════════════════════════════════════════════════════════════════
# 4. 消融实验
# ════════════════════════════════════════════════════════════════

def ablation_study(gt: dict, n_runs: int = 5):
    """消融实验：κ 策略、budget、噪声水平。"""

    print("\n" + "═" * 60)
    print("  消融实验 1: κ 策略对比")
    print("═" * 60)
    kappa_results = {}
    for strategy in ["dynamic", "fixed_high", "fixed_mid", "fixed_low", "random"]:
        print(f"\n  ─ κ 策略: {strategy}")
        r = batch_eval(gt, n_runs=n_runs, kappa_strategy=strategy)
        kappa_results[strategy] = r["summary"]
        s = r["summary"]
        print(f"    → top5%={s['top5pct_recall']['mean']:.3f}±{s['top5pct_recall']['std']:.3f}  "
              f"RMSE={s['rmse']['mean']:.3f}  ρ={s['spearman_rho']['mean']:.3f}  "
              f"gap={s['optimality_gap']['mean']:.4f}  "
              f"regions={s['region_recall']['mean']:.2f}")

    print("\n" + "═" * 60)
    print("  消融实验 2: Budget（种子+自适应 trial 数）")
    print("═" * 60)
    budgets = [(12, 28), (24, 56), (36, 84)]
    budget_results = {}
    for ns, na in budgets:
        print(f"\n  ─ seed={ns}, adaptive={na}, total={ns+na}")
        r = batch_eval(gt, n_runs=n_runs, n_seed=ns, n_adaptive=na)
        budget_results[f"{ns}+{na}"] = r["summary"]
        s = r["summary"]
        print(f"    → top5%={s['top5pct_recall']['mean']:.3f}  "
              f"RMSE={s['rmse']['mean']:.3f}  ρ={s['spearman_rho']['mean']:.3f}  "
              f"gap={s['optimality_gap']['mean']:.4f}")

    print("\n" + "═" * 60)
    print("  消融实验 3: 噪声水平")
    print("═" * 60)
    noise_results = {}
    for noise in [0.0, 0.15, 0.3, 0.5, 0.8]:
        print(f"\n  ─ noise_std={noise}")
        r = batch_eval(gt, n_runs=n_runs, noise_std=noise)
        noise_results[f"{noise}"] = r["summary"]
        s = r["summary"]
        print(f"    → top5%={s['top5pct_recall']['mean']:.3f}  "
              f"RMSE={s['rmse']['mean']:.3f}  ρ={s['spearman_rho']['mean']:.3f}  "
              f"gap={s['optimality_gap']['mean']:.4f}")

    return {"kappa": kappa_results, "budget": budget_results, "noise": noise_results}


# ════════════════════════════════════════════════════════════════
# 5. 随机基线对照
# ════════════════════════════════════════════════════════════════

def random_baseline(gt: dict, n_trials: int = 80, n_runs: int = 10) -> dict:
    """
    随机采样基线：等价于不用 GP，纯随机选 trial。
    对比 adaptive sampler 的提升。
    """
    print("\n" + "═" * 60)
    print(f"  随机基线对照（{n_trials} trials, {n_runs} runs）")
    print("═" * 60)

    candidates = gt["candidates"]
    true_scores = gt["true_scores"]
    results = []

    for run_i in range(n_runs):
        rng = np.random.default_rng(100 + run_i)
        # 随机选 n_trials 个组合
        indices = rng.choice(len(candidates), size=n_trials, replace=False)
        tested_vectors = [candidates[i] for i in indices]
        observed_scores = true_scores[indices]  # 无噪声（最乐观基线）

        # 简单用均值预测
        pred_mu = np.full(len(candidates), float(np.mean(observed_scores)))
        pred_rank = np.argsort(-pred_mu)

        # Top-20 召回：随机基线碰运气
        pred_top20 = set(pred_rank[:20].tolist())
        # 但随机基线没有排序能力，用实测最高的来代替
        best_observed_idx = indices[np.argmax(observed_scores)]
        top20_hit = len(pred_top20 & gt["true_top20"]) / 20

        # 随机基线的 Top-20 召回 = 在测试的 n_trials 里碰中 Top-20 的概率
        n_top20_tested = len(set(indices.tolist()) & gt["true_top20"])
        top20_recall_random = n_top20_tested / 20

        rmse = float(np.sqrt(np.mean((pred_mu - true_scores) ** 2)))
        best_tested_score = float(np.max(observed_scores))
        true_optimal = float(true_scores[gt["true_top1"]])
        gap = (true_optimal - best_tested_score) / true_optimal

        results.append({
            "top20_recall": top20_recall_random,
            "rmse": rmse,
            "spearman_rho": 0.0,  # 随机无排序能力
            "optimality_gap": gap,
            "best_tested_score": best_tested_score,
        })
        print(f"    Run {run_i+1:2d}: random_recall={top20_recall_random:.2f} "
              f"gap={gap:.4f}")

    summary = {}
    for m in ["top20_recall", "rmse", "optimality_gap"]:
        vals = [r[m] for r in results]
        summary[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return {"summary": summary, "runs": results}


# ════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════

def main():
    import json

    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    print("=" * 60)
    print("  算法验证 — Ground Truth 对比 + 消融实验")
    print("=" * 60)

    # 1. 构建 ground truth
    print("\n  [1] 构建地面真值（穷举 2304 组合）...")
    gt = build_ground_truth()
    print(f"      真实最优分: {gt['true_scores'][gt['true_top1']]:.3f}")
    print(f"      → {vector_to_label(gt['candidates'][gt['true_top1']])}")
    print(f"      Top-20 真实分范围: {gt['true_scores'][gt['true_ranking'][19]]:.3f} ~ "
          f"{gt['true_scores'][gt['true_top1']]:.3f}")
    print(f"      3 个高分区域均分: {gt['region_scores']}")

    if mode in ("full", "basic"):
        # 2. 基础验证
        print(f"\n  [2] 基础验证（动态 κ, 24+56=80 trials, 10 次重复）")
        basic = batch_eval(gt, n_runs=10)
        s = basic["summary"]
        print(f"\n  ──────────────────────────────────────")
        print(f"  基础验证结果汇总（{s['n_runs']} runs, {s['n_trials_per_run']} trials/run）:")
        print(f"    Top-1 在 Top-5% 内: {s['top1_in_top5pct']['hits']}/{s['n_runs']} 次")
        print(f"    Top-5% 召回率:       {s['top5pct_recall']['mean']:.3f} ± {s['top5pct_recall']['std']:.3f}")
        print(f"    Top-20 召回率:       {s['top20_recall']['mean']:.3f} ± {s['top20_recall']['std']:.3f}")
        print(f"    全空间 RMSE:         {s['rmse']['mean']:.3f} ± {s['rmse']['std']:.3f}")
        print(f"    Spearman ρ:         {s['spearman_rho']['mean']:.3f} ± {s['spearman_rho']['std']:.3f}")
        print(f"    高反应区域识别率:    {s['region_recall']['mean']:.2f} ± {s['region_recall']['std']:.2f}")
        print(f"    最优性差距:          {s['optimality_gap']['mean']:.4f} ± {s['optimality_gap']['std']:.4f}")

        # 3. 随机基线
        random_res = random_baseline(gt, n_trials=80, n_runs=10)
        rs = random_res["summary"]
        print(f"\n  随机基线对照（同样 80 trials）:")
        print(f"    Top-20 召回率:       {rs['top20_recall']['mean']:.3f} ± {rs['top20_recall']['std']:.3f}")
        print(f"    最优性差距:          {rs['optimality_gap']['mean']:.4f} ± {rs['optimality_gap']['std']:.4f}")
        print(f"\n  算法提升: 召回率 +{s['top20_recall']['mean'] - rs['top20_recall']['mean']:.3f}, "
              f"gap 改善 {rs['optimality_gap']['mean'] - s['optimality_gap']['mean']:.4f}")

    if mode in ("full", "ablation"):
        # 4. 消融实验
        ablation = ablation_study(gt, n_runs=3)

        # 保存消融结果
        with open("D:/BIGAI/Code/EmotionBot/validation_ablation.json", "w", encoding="utf-8") as f:
            json.dump({"ground_truth": {
                "true_optimal": float(gt["true_scores"][gt["true_top1"]]),
                "top20_range": [float(gt["true_scores"][gt["true_ranking"][19]]),
                                float(gt["true_scores"][gt["true_top1"]])],
                "region_scores": gt["region_scores"],
            }, "ablation": ablation}, f, ensure_ascii=False, indent=2, default=float)
        print(f"\n  消融结果已保存: validation_ablation.json")

    print(f"\n{'='*60}")
    print("  验证完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
