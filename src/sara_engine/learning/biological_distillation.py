# ディレクトリパス: src/sara_engine/learning/biological_distillation.py
# ファイルの日本語タイトル: 生物学的知識蒸留モジュール
# ファイルの目的や内容: 教師モデルの連続的出力（レート）をポアソンスパイク列に変換し、生徒SNNのシナプスを局所的学習則で更新する。

import random


class PoissonSpikeGenerator:
    """
    教師モデルからの出力（発火率）をポアソンスパイク列に変換するジェネレータ。
    """

    def __init__(self, max_rate: float = 1.0) -> None:
        self.max_rate = max_rate

    def generate_spikes(self, rates: list[float], time_steps: int) -> list[list[int]]:
        """
        与えられた発火率プロファイルに基づき、指定ステップ数のスパイク列を生成する。

        Args:
            rates: 各ニューロンの発火率 (0.0 ~ 1.0の範囲を想定)
            time_steps: 生成する時間ステップ数

        Returns:
            各ステップごとのスパイクベクトル（[[0, 1, 0, ...], ...]）
        """
        spike_trains = []
        for _ in range(time_steps):
            step_spikes = []
            for rate in rates:
                # rateに比例する確率で発火(1)を生成
                normalized_rate = min(max(rate * self.max_rate, 0.0), 1.0)
                if random.random() < normalized_rate:
                    step_spikes.append(1)
                else:
                    step_spikes.append(0)
            spike_trains.append(step_spikes)

        return spike_trains


class BiologicalDistillationManager:
    """
    Teacherのポアソンスパイク列を用いて、Student SNNのシナプス結合重みを更新するマネージャー。
    """

    def __init__(self, num_inputs: int, num_outputs: int, learning_rate: float = 0.01) -> None:
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate

        # [num_outputs][num_inputs] の結合重み行列 (密結合を仮定)
        self.weights = []
        for _ in range(num_outputs):
            self.weights.append([random.uniform(0.1, 0.5)
                                for _ in range(num_inputs)])

        self.w_max = 5.0
        self.w_min = 0.0

    def step(self, student_input_spikes: list[int], student_output_spikes: list[int], teacher_target_spikes: list[int]) -> None:
        """
        1ステップの学習を実行する（スパイクベースのデルタ則）。

        Args:
            student_input_spikes: 生徒モデルへの入力スパイク
            student_output_spikes: 生徒モデルの実際の出力スパイク
            teacher_target_spikes: 期待される教師の目標スパイク
        """
        for j in range(self.num_outputs):
            # 目標と実際の出力の差分 (デルタ)
            error = teacher_target_spikes[j] - student_output_spikes[j]

            # error > 0: 目標は発火すべきだったのに、生徒は発火しなかった -> 重みを増やす (LTP)
            # error < 0: 目標は発火すべきでないのに、生徒は発火した -> 重みを減らす (LTD)
            if error != 0:
                for i in range(self.num_inputs):
                    if student_input_spikes[i] == 1:
                        delta_w = self.learning_rate * error
                        self.weights[j][i] += delta_w

                        # 重みのクリッピング
                        self.weights[j][i] = max(self.w_min, min(
                            self.w_max, self.weights[j][i]))

    def batch_train(self, inputs_seq: list[list[int]], student_outputs_seq: list[list[int]], teacher_targets_seq: list[list[int]]) -> None:
        """
        ステップシーケンス全体で学習を適用する。
        """
        for t in range(len(inputs_seq)):
            self.step(inputs_seq[t], student_outputs_seq[t],
                      teacher_targets_seq[t])
