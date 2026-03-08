# ディレクトリパス: tests/test_spatio_temporal_binding_layer.py
# ファイルの日本語タイトル: 時空間バインディング層のテストスクリプト
# ファイルの目的や内容: SNNのモダリティ間での非同期なスパイク入力が、適格度トレースを通じて適切に結合を形成（LTP）し、離れすぎた入力には結合が形成されない（減衰）ことを検証する。

from sara_engine.nn.spatio_temporal_binding import SpatioTemporalBindingLayer
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))


def test_spatio_temporal_binding():
    print("--- 時空間バインディング層 テスト ---")

    # モダリティ設定
    modalities = ["vision", "language"]
    dim_per_modality = 100

    # バインディング層の初期化
    layer = SpatioTemporalBindingLayer(
        modality_names=modalities,
        dim_per_modality=dim_per_modality,
        trace_decay=0.8,
        learning_rate=0.5,
        max_weight=3.0,
        prune_threshold=0.01
    )

    # 初期状態でのRecallは何も返さないことを確認
    inputs = {"vision": [10, 20]}
    recalls = layer.forward(inputs, learning=False, threshold=0.1)
    assert len(recalls["language"]) == 0, "初期状態では何もRecallされないはずです"

    print("1. 非同期スパイクの入力と学習 (Vision -> (Delay) -> Language)")
    # ステップ1: Vision入力
    layer.forward({"vision": [10, 20]}, learning=True)

    # ステップ2: 時間経過 (何もない)
    layer.forward({}, learning=True)

    # ステップ3: Language入力
    # 直前のVisionのトレースが残っているので、Vision -> Languageの結合が強化されるはず
    layer.forward({"language": [30, 40]}, learning=True)

    # 確認１: VisionからのRecall
    # Vision [10, 20] を入力すると Language [30, 40] がRecallされるか
    recalls_from_vision = layer.forward(
        {"vision": [10, 20]}, learning=False, threshold=0.1)
    print(
        f"Vision(10, 20)からのRecall in Language: {recalls_from_vision['language']}")
    assert 30 in recalls_from_vision["language"] and 40 in recalls_from_vision[
        "language"], "VisionからLanguageへのバインディングが失敗しています"

    print("2. 逆方向のRecall確認 (LanguageからのRecallは学習されていない)")
    recalls_from_lang = layer.forward(
        {"language": [30, 40]}, learning=False, threshold=0.1)
    print(
        f"Language(30, 40)からのRecall in Vision: {recalls_from_lang['vision']}")
    assert len(recalls_from_lang["vision"]) == 0, "逆方向のバインディングはまだ行われていないはずです"

    print("3. 長時間経過後のバインディング減衰テスト")
    # ステップ1: Vision入力
    layer.forward({"vision": [50]}, learning=True)

    # 長時間ループ (トレースの自然消滅を待つ)
    for _ in range(15):
        layer.forward({}, learning=True)

    # ステップ2: Language入力
    layer.forward({"language": [60]}, learning=True)

    # 確認２: VisionからLanguageへのRecall
    # トレースが減衰しているので、バインディングは形成されないはず
    recalls_delayed = layer.forward(
        {"vision": [50]}, learning=False, threshold=0.1)
    print(
        f"Vision(50) (長時間遅延) からのRecall in Language: {recalls_delayed['language']}")
    assert 60 not in recalls_delayed["language"], "時間が離れすぎている入力同士が誤ってバインディングされています"

    print("✅ テスト完了: 時空間バインディング層は正常に動作しています。")


if __name__ == "__main__":
    test_spatio_temporal_binding()
