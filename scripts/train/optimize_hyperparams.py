from typing import Dict, List, Any
from sara_engine.models.snn_transformer import SpikingTransformerModel, SNNTransformerConfig
import time
import json
import random
import sys
import os
# ディレクトリパス: scripts/train/optimize_hyperparams.py
# ファイルの日本語タイトル: 遺伝的アルゴリズムによるハイパーパラメータ最適化
# ファイルの目的や内容: 誤差逆伝播法や行列演算、外部の最適化ライブラリに依存せず、生物学的な進化計算（遺伝的アルゴリズム）を用いてSNNモデルのハイパーパラメータを自律的に探索・最適化する。


# srcディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'src')))


# =====================================================================
# [1] 探索空間の定義 (Hyperparameter Search Space)
# =====================================================================
SEARCH_SPACE: Dict[str, List[Any]] = {
    "embed_dim": [32, 64, 128],
    "num_layers": [1, 2, 3],
    "ffn_dim": [64, 128, 256],
    "dropout_p": [0.0, 0.1, 0.2],
    "target_spikes_ratio": [0.1, 0.15, 0.2, 0.25, 0.3],
    "use_fuzzy": [True, False]
}

# =====================================================================
# [2] 遺伝的アルゴリズムのコアロジック
# =====================================================================


def generate_random_genome() -> dict:
    """探索空間からランダムに遺伝子（ハイパーパラメータ設定）を生成する"""
    return {key: random.choice(values) for key, values in SEARCH_SPACE.items()}


def crossover(parent1: dict, parent2: dict) -> dict:
    """2つの親から交叉（一様交叉）により子を生成する"""
    child = {}
    for key in SEARCH_SPACE.keys():
        child[key] = parent1[key] if random.random() < 0.5 else parent2[key]
    return child


def mutate(genome: dict, mutation_rate: float = 0.2) -> dict:
    """一定確率で遺伝子を突然変異させる"""
    mutated = genome.copy()
    for key, values in SEARCH_SPACE.items():
        if random.random() < mutation_rate:
            mutated[key] = random.choice(values)
    return mutated


def evaluate_genome(genome: dict) -> float:
    """
    指定されたハイパーパラメータでSNNモデルを構築し、
    自然言語の文字レベル予測タスクでACC（精度）を評価する。
    """
    config = SNNTransformerConfig(
        vocab_size=256,  # ASCII文字用
        embed_dim=genome["embed_dim"],
        num_layers=genome["num_layers"],
        ffn_dim=genome["ffn_dim"],
        dropout_p=genome["dropout_p"],
        target_spikes_ratio=genome["target_spikes_ratio"],
        use_fuzzy=genome["use_fuzzy"]
    )
    model = SpikingTransformerModel(config)

    # より複雑な評価用データ: コンテキスト（文脈）の理解が必要なテキスト
    text = (
        "SARA-Engine is a biological spiking neural network. "
        "It uses predictive coding and scalable SDR memory. "
        "We do not use backpropagation, matrix operations, or GPU. "
        "Rust and Python are perfectly integrated."
    )
    # 文字をASCIIのIDに変換
    tokens = [ord(c) for c in text]

    # 学習フェーズ（学習回数をあえて減らし、パラメータによる学習効率の差を出す）
    train_sequence = tokens * 3
    model.learn_sequence(train_sequence)

    # 評価フェーズ（未知の文脈からの予測能力を測るため元のテキストでテスト）
    model.reset_state()
    correct = 0
    total = len(tokens) - 1

    current_token = tokens[0]
    for i in range(total):
        target = tokens[i+1]
        pred, _info = model.forward_step(current_token, learning=False)
        if pred == target:
            correct += 1
        # 教師強制（Teacher Forcing）: 正解データを次の入力として与える
        current_token = target

    return correct / total if total > 0 else 0.0

# =====================================================================
# [3] 進化プロセスの実行
# =====================================================================


def run_evolution(population_size: int = 10, generations: int = 5):
    print("=" * 60)
    print("SNN Hyperparameter Optimization (Evolutionary Algorithm)")
    print("=" * 60)

    # 初期集団の生成
    population = [generate_random_genome() for _ in range(population_size)]
    best_overall_genome = None
    best_overall_fitness = -1.0

    for gen in range(generations):
        print(f"\n--- Generation {gen + 1}/{generations} ---")
        start_time = time.time()

        # 評価
        fitness_scores = []
        for i, genome in enumerate(population):
            score = evaluate_genome(genome)
            fitness_scores.append((score, genome))
            print(f"  Genome {i+1}: ACC {score:.4f} | {genome}")

        # 降順（適応度が高い順）にソート
        fitness_scores.sort(key=lambda x: x[0], reverse=True)

        current_best_score, current_best_genome = fitness_scores[0]
        print(f">> Generation Best ACC: {current_best_score:.4f}")

        if current_best_score > best_overall_fitness:
            best_overall_fitness = current_best_score
            best_overall_genome = current_best_genome

        # 次世代の生成（エリート選択 + 交叉 + 突然変異）
        next_population = []

        # エリート保存（上位2個体をそのまま残す）
        next_population.append(fitness_scores[0][1])
        next_population.append(fitness_scores[1][1])

        # 残りの個体を交叉と突然変異で生成
        while len(next_population) < population_size:
            # ルーレット選択ではなく、上位50%からランダムに親を選ぶ（簡易トーナメント）
            parents_pool = [genome for _,
                            genome in fitness_scores[:population_size // 2]]
            p1 = random.choice(parents_pool)
            p2 = random.choice(parents_pool)

            child = crossover(p1, p2)
            child = mutate(child, mutation_rate=0.2)
            next_population.append(child)

        population = next_population
        elapsed = time.time() - start_time
        print(f"Generation {gen + 1} completed in {elapsed:.2f} seconds.")

    print("\n" + "=" * 60)
    print("Optimization Finished!")
    print(f"Best ACC: {best_overall_fitness:.4f}")
    print("Best Hyperparameters:")
    print(json.dumps(best_overall_genome, indent=4))
    print("=" * 60)

    # 結果の保存
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/best_hyperparams.json", "w", encoding="utf-8") as f:
        json.dump(best_overall_genome, f, indent=4)


if __name__ == "__main__":
    # テスト用途のため小さな世代数で実行。実運用の際は数値を増やしてください。
    run_evolution(population_size=8, generations=5)
