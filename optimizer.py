#!/usr/bin/env python3
"""
离散空间贝叶斯优化 + 爬山法
8 维离散空间，每维有 n_i 个整数点，目标是找到人类评分最高的点。
策略：初始采样 → 代理模型(UCB)探索 → 爬山局部优化 → 随机重启
"""

import sys
import random
import math
import itertools
from collections import namedtuple
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ── Data structures ──────────────────────────────────────────────

ObservedPoint = namedtuple("ObservedPoint", ["vector", "score"])


# ── User interaction ─────────────────────────────────────────────

def ask_dimensions() -> list[int]:
    """交互式获取每个维度的采样点数。"""
    while True:
        try:
            raw = input("Enter number of points for each dimension (8 integers, comma-separated): ")
            dims = [int(x.strip()) for x in raw.split(",")]
            if len(dims) != 8:
                print(f"  Expected 8 integers, got {len(dims)}. Try again.")
                continue
            if any(d < 2 for d in dims):
                print("  Each dimension must have at least 2 points. Try again.")
                continue
            return dims
        except ValueError:
            print("  Invalid input. Example: 2,3,4,2,2,5,3,2")


def ask_score(vector: tuple[int, ...], dim_names: Optional[list[str]] = None) -> float:
    """向用户询问某个采样点的人类评分。"""
    if dim_names:
        desc = "  ".join(f"{name}={v}" for name, v in zip(dim_names, vector))
        print(f"\n  Point: [{desc}]")
    else:
        print(f"\n  Point: {list(vector)}")

    while True:
        try:
            raw = input("  Score (float, higher = better): ")
            return float(raw)
        except ValueError:
            print("  Invalid. Enter a number.")


def ask_budget(default: int) -> int:
    """询问用户愿意接受的最大查询次数。"""
    raw = input(f"Max total queries? (press Enter for default={default}): ")
    if raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def ask_dim_names() -> list[str]:
    """可选：询问维度名称。"""
    raw = input("Dimension names? (8 names, comma-separated, or Enter to skip): ")
    if raw.strip() == "":
        return [f"dim{i}" for i in range(8)]
    names = [x.strip() for x in raw.split(",")]
    if len(names) != 8:
        print("  Didn't get 8 names, using defaults.")
        return [f"dim{i}" for i in range(8)]
    return names


# ── Space utilities ──────────────────────────────────────────────

class DiscreteSpace:
    """8 维离散采样空间。"""

    def __init__(self, dim_sizes: list[int]):
        self.dim_sizes = tuple(dim_sizes)
        self.ndim = len(dim_sizes)
        self.total_size = math.prod(dim_sizes)

    def random_point(self) -> tuple[int, ...]:
        return tuple(random.randrange(0, d) for d in self.dim_sizes)

    def random_batch(self, n: int) -> list[tuple[int, ...]]:
        """生成 n 个不重复的随机点。"""
        if n > self.total_size:
            n = self.total_size
        seen: set[tuple[int, ...]] = set()
        pts = []
        while len(pts) < n:
            p = self.random_point()
            if p not in seen:
                seen.add(p)
                pts.append(p)
        return pts

    def stratified_sample(self, n: int) -> list[tuple[int, ...]]:
        """
        分层随机采样：把每个维度分成大致 sqrt(n) 个 strata，每个 strata 随机选一个点。
        适合初始采样阶段，保证空间覆盖。
        """
        if n >= self.total_size:
            return list(itertools.product(*[range(d) for d in self.dim_sizes]))

        # 按重要维度（点数多的）做分层
        strata_per_dim = max(2, int(n ** (1 / self.ndim)))
        partitions = []
        for d in self.dim_sizes:
            chunk = max(1, d // strata_per_dim)
            parts = []
            for start in range(0, d, chunk):
                parts.append((start, min(start + chunk, d)))
            partitions.append(parts)

        # 生成所有 strata 组合，每个 strata 内随机选点
        candidates = []
        for indices in itertools.product(*[range(len(p)) for p in partitions]):
            if len(candidates) >= n * 3:
                break
            pt = tuple(
                random.randrange(partitions[dim][idx][0], partitions[dim][idx][1])
                for dim, idx in enumerate(indices)
            )
            candidates.append(pt)

        # 从候选中随机挑 n 个不重复的
        random.shuffle(candidates)
        seen: set[tuple[int, ...]] = set()
        result = []
        for p in candidates:
            if p not in seen:
                seen.add(p)
                result.append(p)
                if len(result) >= n:
                    break
        return result

    def neighbors(self, point: tuple[int, ...]) -> list[tuple[int, ...]]:
        """生成一个点的所有邻居（每个维度 ±1，共最多 16 个）。"""
        result = []
        for dim in range(self.ndim):
            val = point[dim]
            if val > 0:
                nb = list(point)
                nb[dim] = val - 1
                result.append(tuple(nb))
            if val < self.dim_sizes[dim] - 1:
                nb = list(point)
                nb[dim] = val + 1
                result.append(tuple(nb))
        return result

    def manhattan_distance(self, a: tuple[int, ...], b: tuple[int, ...]) -> int:
        return sum(abs(a[i] - b[i]) for i in range(self.ndim))

    def normalized_distance(self, a: tuple[int, ...], b: tuple[int, ...]) -> float:
        """归一化曼哈顿距离，范围 [0, 1]。"""
        max_dist = sum(d - 1 for d in self.dim_sizes)
        if max_dist == 0:
            return 0.0
        return self.manhattan_distance(a, b) / max_dist


# ── Surrogate model (KNN regression) ─────────────────────────────

class KNNSurrogate:
    """K-近邻加权回归代理模型。"""

    def __init__(self, space: DiscreteSpace, k: int = 5):
        self.space = space
        self.k = k

    def predict(self, point: tuple[int, ...],
                observed: list[ObservedPoint]) -> tuple[float, float]:
        """
        返回 (预测均值, 不确定性)。
        不确定性 = 到最近已观测点的归一化距离。
        """
        if not observed:
            return 0.0, 1.0

        # 按距离排序
        distances = []
        for obs in observed:
            d = self.space.manhattan_distance(point, obs.vector)
            distances.append((d, obs.score))
        distances.sort(key=lambda x: x[0])

        # 取前 k 个
        k = min(self.k, len(observed))
        nearest = distances[:k]

        # 加权平均（用 1/(d+1) 权重）
        weights = [1.0 / (d + 1.0) for d, _ in nearest]
        total_w = sum(weights)
        if total_w > 0:
            mean = sum(w * score for w, (_, score) in zip(weights, nearest)) / total_w
        else:
            mean = sum(score for _, score in nearest) / k

        # 不确定性 = 到最近观测点的归一化距离
        nearest_d = distances[0][0]
        max_dist = sum(d - 1 for d in self.space.dim_sizes)
        if max_dist == 0:
            uncertainty = 0.0
        else:
            uncertainty = min(1.0, nearest_d / max_dist)

        return mean, uncertainty

    def ucb(self, point: tuple[int, ...],
            observed: list[ObservedPoint], beta: float = 2.0) -> float:
        """上置信界采集函数。"""
        mean, uncertainty = self.predict(point, observed)
        return mean + beta * uncertainty


# ── Neighbor set for UCB candidates ──────────────────────────────

def ucb_candidates(space: DiscreteSpace, best_point: tuple[int, ...],
                   observed: list[ObservedPoint], n_random: int = 50) -> list[tuple[int, ...]]:
    """生成 UCB 阶段候选点：已观测点的邻居 + 随机点。"""
    candidates: set[tuple[int, ...]] = set()

    # 所有已观测点的邻居（1-hop + 2-hop）
    observed_pts = {obs.vector for obs in observed}
    for obs in observed:
        for nb in space.neighbors(obs.vector):
            if nb not in observed_pts:
                candidates.add(nb)
            # 2-hop
            for nb2 in space.neighbors(nb):
                if nb2 not in observed_pts:
                    candidates.add(nb2)

    # 补充随机点
    rand_pts = space.random_batch(n_random)
    for p in rand_pts:
        if p not in observed_pts:
            candidates.add(p)

    return list(candidates)


# ── Phase 3: Hill Climbing ──────────────────────────────────────

def hill_climb(space: DiscreteSpace, start: tuple[int, ...],
               observed: list[ObservedPoint],
               ask_fn, dim_names: list[str],
               query_count: list[int], budget: int) -> tuple[tuple[int, ...], float]:
    """
    离散爬山法：从 start 出发，每次评估所有邻居，移动到更好的邻居。
    收敛到局部最优。
    """
    current = start
    # 找 current 的评分
    current_score = None
    for obs in observed:
        if obs.vector == current:
            current_score = obs.score
            break

    if current_score is None:
        if query_count[0] >= budget:
            return current, -float("inf")
        current_score = ask_fn(current, dim_names)
        observed.append(ObservedPoint(current, current_score))
        query_count[0] += 1

    print(f"\n  [Hill Climb] Starting from {list(current)}, score={current_score:.4f}")

    while True:
        if query_count[0] >= budget:
            print(f"  [Hill Climb] Budget exhausted ({budget} queries).")
            break

        nbs = space.neighbors(current)
        best_nb, best_nb_score = None, current_score

        for nb in nbs:
            # 检查是否已经观测过
            nb_score = None
            for obs in observed:
                if obs.vector == nb:
                    nb_score = obs.score
                    break

            if nb_score is None:
                if query_count[0] >= budget:
                    break
                nb_score = ask_fn(nb, dim_names)
                observed.append(ObservedPoint(nb, nb_score))
                query_count[0] += 1

            if nb_score > best_nb_score:
                best_nb = nb
                best_nb_score = nb_score

        if best_nb is None or best_nb_score <= current_score:
            print(f"  [Hill Climb] Local optimum reached: {list(current)}, score={current_score:.4f}")
            break
        else:
            # 输出爬山路径
            move_dim = next(i for i in range(space.ndim) if current[i] != best_nb[i])
            direction = "↑" if best_nb[move_dim] > current[move_dim] else "↓"
            print(f"  [Hill Climb] Move dim{move_dim} {direction}: "
                  f"{list(current)} -> {list(best_nb)}, score {current_score:.4f} -> {best_nb_score:.4f}")
            current = best_nb
            current_score = best_nb_score

    return current, current_score


# ── Phase 2: UCB-guided search ──────────────────────────────────

def ucb_search(space: DiscreteSpace,
               observed: list[ObservedPoint],
               ask_fn, dim_names: list[str],
               query_count: list[int], budget: int,
               n_iter: int = 12,
               beta: float = 2.0) -> None:
    """
    UCB 引导的全局探索。
    每次选 UCB 最高的候选点询问。当逐步降低 beta 时，后期更侧重利用。
    """
    surrogate = KNNSurrogate(space, k=5)
    betas = np.linspace(beta, 0.5, n_iter)

    for iteration in range(n_iter):
        if query_count[0] >= budget:
            print(f"  [UCB] Budget exhausted at iteration {iteration + 1}.")
            break

        best_point = max(observed, key=lambda o: o.score).vector
        candidates = ucb_candidates(space, best_point, observed, n_random=40)

        if not candidates:
            print(f"  [UCB] No unexplored candidates left at iteration {iteration + 1}.")
            break

        # 选择 UCB 最高的候选点
        b = betas[iteration]
        best_candidate = max(candidates, key=lambda p: surrogate.ucb(p, observed, b))

        score = ask_fn(best_candidate, dim_names)
        observed.append(ObservedPoint(best_candidate, score))
        query_count[0] += 1

        # 报告进展
        current_best = max(observed, key=lambda o: o.score)
        mean, unc = surrogate.predict(best_candidate, observed[:-1])
        print(f"    [UCB iter {iteration + 1}/{n_iter}] beta={b:.2f} | "
              f"predicted={mean:.3f}±{unc:.3f} | actual={score:.4f} | "
              f"best so far={current_best.score:.4f} | queries={query_count[0]}")


# ── Phase 4: Random Restart ─────────────────────────────────────

def random_restarts(space: DiscreteSpace,
                    observed: list[ObservedPoint],
                    ask_fn, dim_names: list[str],
                    query_count: list[int], budget: int,
                    n_restarts: int = 3) -> list[tuple[tuple[int, ...], float]]:
    """
    从随机未探索点启动多次爬山。
    """
    results = []
    for i in range(n_restarts):
        if query_count[0] >= budget:
            print(f"  [Restart] Budget exhausted before restart {i + 1}.")
            break

        observed_pts = {obs.vector for obs in observed}
        unobserved_candidates = space.random_batch(50)
        unobserved_candidates = [p for p in unobserved_candidates if p not in observed_pts]

        if not unobserved_candidates:
            print(f"  [Restart] No unexplored points left.")
            break

        start = random.choice(unobserved_candidates)
        print(f"\n  [Restart {i + 1}/{n_restarts}] Starting from {list(start)}")
        local_opt, local_score = hill_climb(
            space, start, observed, ask_fn, dim_names, query_count, budget
        )
        results.append((local_opt, local_score))
    return results


# ── Visualization ────────────────────────────────────────────────

def visualize(space: DiscreteSpace, observed: list[ObservedPoint],
              best_point: tuple[int, ...], dim_names: list[str],
              output_path: str):
    """
    用 PCA 将 8 维观测点降维到 2 维可视化。
    - 点颜色 = 评分（越深越好）
    - 箭头 = 搜索路径（按查询顺序）
    - 星标 = 最优目标
    """
    if len(observed) < 2:
        print("  Not enough points to visualize.")
        return

    vectors = np.array([obs.vector for obs in observed])
    scores = np.array([obs.score for obs in observed])

    # PCA 降维
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(10, 8))

    # 散点：颜色映射评分
    sc = ax.scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c=scores, cmap="RdYlGn", s=80, edgecolors="black", linewidths=0.5,
        zorder=3
    )

    # 搜索路径箭头（按查询顺序）
    for i in range(len(coords_2d) - 1):
        dx = coords_2d[i + 1, 0] - coords_2d[i, 0]
        dy = coords_2d[i + 1, 1] - coords_2d[i, 1]
        if dx != 0 or dy != 0:
            arrow = FancyArrowPatch(
                coords_2d[i], coords_2d[i + 1],
                arrowstyle="->", color="gray", alpha=0.5,
                linewidth=0.8, mutation_scale=10,
                zorder=1
            )
            ax.add_patch(arrow)

    # 标出最优目标
    best_vec = np.array(best_point)
    best_2d = pca.transform(best_vec.reshape(1, -1))[0]
    ax.scatter(
        best_2d[0], best_2d[1],
        marker="*", color="gold", s=400, edgecolors="black", linewidths=1.5,
        zorder=5, label="Best"
    )

    # 在最优目标旁边标注评分
    best_score = max(obs.score for obs in observed if obs.vector == best_point)
    ax.annotate(
        f"Best: {best_score:.4f}\n{list(best_point)}",
        xy=(best_2d[0], best_2d[1]),
        xytext=(10, 10), textcoords="offset points",
        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9)
    )

    # 颜色条
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("Human Score", fontsize=11)

    ax.set_title(
        f"Search Path in 8D Space (PCA 2D projection)\n"
        f"{len(observed)} queries, Best score = {best_score:.4f}",
        fontsize=12
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\n  Visualization saved to: {output_path}")
    plt.close(fig)


# ── Main pipeline ────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Discrete Bayesian Optimizer — 8D Space")
    print("=" * 60)

    # ── Setup ──
    dim_sizes = ask_dimensions()
    dim_names = ask_dim_names()
    space = DiscreteSpace(dim_sizes)
    total = space.total_size
    print(f"\n  Space size: {' × '.join(str(d) for d in dim_sizes)} = {total:,} points")

    default_budget = min(50, max(20, total // 5))
    budget = ask_budget(default_budget)
    print(f"  Budget: {budget} queries")

    observed: list[ObservedPoint] = []
    query_count = [0]  # mutable counter

    ask_fn = ask_score  # for future extensibility

    # ── Phase 1: Initial Sampling ──
    n_init = min(5, max(3, budget // 8), total)
    print(f"\n{'─' * 40}")
    print(f"  Phase 1: Initial Sampling ({n_init} points)")
    print(f"{'─' * 40}")
    init_points = space.stratified_sample(n_init)
    for i, pt in enumerate(init_points):
        if query_count[0] >= budget:
            break
        print(f"\n  [Init {i + 1}/{n_init}]")
        score = ask_fn(pt, dim_names)
        observed.append(ObservedPoint(pt, score))
        query_count[0] += 1

    best = max(observed, key=lambda o: o.score)
    print(f"\n  Phase 1 best: {list(best.vector)}, score={best.score:.4f}")

    # ── Phase 2: UCB Search ──
    n_ucb = min(12, budget - query_count[0])
    if n_ucb > 0:
        print(f"\n{'─' * 40}")
        print(f"  Phase 2: UCB-Guided Search (~{n_ucb} iterations)")
        print(f"{'─' * 40}")
        ucb_search(space, observed, ask_fn, dim_names,
                   query_count, budget, n_iter=n_ucb, beta=2.0)

    # ── Phase 3: Hill Climb ──
    best_before_hc = max(observed, key=lambda o: o.score)
    print(f"\n{'─' * 40}")
    print(f"  Phase 3: Hill Climbing")
    print(f"{'─' * 40}")
    local_opt, local_opt_score = hill_climb(
        space, best_before_hc.vector, observed, ask_fn, dim_names,
        query_count, budget
    )

    # ── Phase 4: Random Restart ──
    remaining = budget - query_count[0]
    if remaining > 10:
        n_restarts = min(3, remaining // 10)
        if n_restarts > 0:
            print(f"\n{'─' * 40}")
            print(f"  Phase 4: Random Restarts ({n_restarts} restarts)")
            print(f"{'─' * 40}")
            random_restarts(space, observed, ask_fn, dim_names,
                            query_count, budget, n_restarts=n_restarts)
    else:
        print(f"\n  Budget remaining ({remaining}) < 10, skipping restarts.")

    # ── Final Report ──
    best = max(observed, key=lambda o: o.score)
    print(f"\n{'=' * 60}")
    print(f"  FINAL RESULT")
    print(f"{'=' * 60}")
    print(f"  Best point:   {list(best.vector)}")
    print(f"  Best score:   {best.score:.4f}")
    print(f"  Total queries: {query_count[0]}/{budget}")
    print(f"  Points visited: {len(observed)}")

    print(f"\n  All visited points:")
    for i, obs in enumerate(observed):
        marker = " ★" if obs.vector == best.vector else "  "
        print(f"  [{i + 1:2d}]{marker} {list(obs.vector)} → {obs.score:.4f}")

    # ── Visualization ──
    vis_path = "D:/BIGAI/Code/EmotionBot/search_path.png"
    visualize(space, observed, best.vector, dim_names, vis_path)

    # ── Save data ──
    data_path = "D:/BIGAI/Code/EmotionBot/search_data.tsv"
    with open(data_path, "w") as f:
        f.write("query_order\t" + "\t".join(dim_names) + "\tscore\n")
        for i, obs in enumerate(observed):
            f.write(f"{i + 1}\t" + "\t".join(str(v) for v in obs.vector) + f"\t{obs.score:.4f}\n")
    print(f"  Data saved to: {data_path}")

    return best.vector, best.score


if __name__ == "__main__":
    main()
