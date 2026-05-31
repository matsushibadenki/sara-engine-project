# ディレクトリパス: tests/test_biological_distillation.py
# ファイルの日本語タイトル: 生物学的知識蒸留のテスト
# ファイルの目的や内容: PoissonSpikeGeneratorとBiologicalDistillationManagerの動作検証

from sara_engine.learning.biological_distillation import PoissonSpikeGenerator, BiologicalDistillationManager


def test_poisson_spike_generation() -> None:
    generator = PoissonSpikeGenerator(max_rate=1.0)
    rates = [0.1, 0.5, 0.9]
    time_steps = 1000

    spike_trains = generator.generate_spikes(rates, time_steps)

    assert len(spike_trains) == time_steps
    assert len(spike_trains[0]) == len(rates)

    # 1000ステップの発火回数を集計
    spike_counts = [0] * len(rates)
    for step_spikes in spike_trains:
        for i, spike in enumerate(step_spikes):
            spike_counts[i] += spike

    # 平均発火率が入力の rate に近いことを確認（許容誤差0.05）
    for i, rate in enumerate(rates):
        actual_rate = spike_counts[i] / time_steps
        assert abs(actual_rate - rate) < 0.05


def test_distillation_weight_update() -> None:
    num_inputs = 2
    num_outputs = 1
    manager = BiologicalDistillationManager(
        num_inputs, num_outputs, learning_rate=0.1)

    # 意図的に初期重みを固定
    manager.weights = [[0.5, 0.5]]

    # シナリオ1: 教師が発火し、生徒は発火しなかった -> 重み増加
    manager.step(
        student_input_spikes=[1, 0],
        student_output_spikes=[0],
        teacher_target_spikes=[1]
    )

    # index 0の入力のみアクティブなので、weights[0][0]のみ増加する
    assert manager.weights[0][0] > 0.5
    assert manager.weights[0][1] == 0.5  # 変化なし

    # シナリオ2: 教師が発火せず、生徒が発火した -> 重み減少
    manager.weights = [[0.8, 0.8]]
    manager.step(
        student_input_spikes=[0, 1],
        student_output_spikes=[1],
        teacher_target_spikes=[0]
    )

    # index 1の入力のみアクティブなので、weights[0][1]のみ減少する
    assert manager.weights[0][0] == 0.8  # 変化なし
    assert manager.weights[0][1] < 0.8
