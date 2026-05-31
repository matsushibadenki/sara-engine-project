# ディレクトリパス: scripts/eval/health_check.py
# ファイルの日本語タイトル: SARA-Engine 統合アーキテクチャ・ヘルスチェック
# ファイルの目的や内容: SARA-Engineの主要コンポーネントが正常に機能しているか診断するスクリプト。Rustコアで実装されているSDR連想メモリ、WTAルーティング、Direct Synaptic Wiringなどの最新機能の正常性確認を追加。

import os
import random
import sys
import time

# srcディレクトリをパスに追加（インポート前に実行する必要がある）
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from sara_engine.core.layers import DynamicLiquidLayer  # noqa: E402
from sara_engine.learning.stdp import STDPLayer  # noqa: E402


class SARAHealthCheck:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []
        print("=" * 50)
        print("SARA-Engine: Biological Architecture Health Check")
        print("=" * 50)

    def log(self, category: str, status: bool, message: str) -> None:
        icon = "✅" if status else "❌"
        self.results.append((category, status, message))
        print(f"[{icon}] {category}: {message}")

    def run_checks(self) -> None:
        self.check_event_driven_integrity()
        self.check_structural_plasticity()
        self.check_homeostatic_stability()
        self.check_energy_efficiency_potential()
        self.check_rust_integration()
        self.check_sdr_memory()
        self.check_wta_router()
        self.check_direct_wiring()
        self.report()

    def check_event_driven_integrity(self) -> None:
        try:
            layer = DynamicLiquidLayer(
                input_size=10, hidden_size=20, decay=0.9, use_rust=False)
            v = layer.v
            thresh = layer.dynamic_thresh
            is_pure_list = isinstance(v, list) and isinstance(thresh, list)
            if is_pure_list:
                self.log("行列演算排除", True, "Numpy依存なし（純粋なPython List）を確認。")
            else:
                self.log("行列演算排除", False, "状態取得時にNumpy配列が検出されました。")
        except Exception as e:
            self.log("行列演算排除", False, f"実行エラー: {e!s}")

    def check_structural_plasticity(self) -> None:
        try:
            num_inputs = 50
            layer = STDPLayer(num_inputs=num_inputs, num_outputs=5)
            initial_connections = sum(len(s) for s in layer.synapses)

            active_pattern = [1 if i < 5 else 0 for i in range(num_inputs)]
            for _ in range(200):
                layer.process_step(active_pattern, reward=1.0)

            final_connections = sum(len(s) for s in layer.synapses)
            if final_connections < initial_connections:
                reduction = (
                    (initial_connections - final_connections) / initial_connections) * 100
                self.log("構造的可塑性", True,
                         f"不要シナプスの刈り込みを確認（{reduction:.1f}% 削減）。")
            else:
                self.log("構造的可塑性", False, "シナプスが刈り込まれていません。")
        except Exception as e:
            self.log("構造的可塑性", False, f"実行エラー: {e!s}")

    def check_homeostatic_stability(self) -> None:
        try:
            layer = DynamicLiquidLayer(
                input_size=10, hidden_size=10, decay=0.9,
                input_scale=30.0, target_rate=0.01, use_rust=False
            )
            thresh_start = list(layer.dynamic_thresh)
            avg_start = sum(thresh_start) / len(thresh_start)

            for _ in range(300):
                layer.refractory = [0.0] * layer.size
                layer.v = [50.0] * layer.size
                layer.forward(active_inputs=list(range(10)),
                              prev_active_hidden=[], feedback_active=[])

            thresh_end = list(layer.dynamic_thresh)
            avg_end = sum(thresh_end) / len(thresh_end)

            if avg_end > avg_start:
                self.log("ホメオスタシス", True,
                         f"動的閾値の上昇を確認（開始: {avg_start:.4f}, 終了: {avg_end:.4f}）。")
            else:
                self.log("ホメオスタシス", False, "閾値が上昇していません。")
        except Exception as e:
            self.log("ホメオスタシス", False, f"実行エラー: {e!s}")

    def check_energy_efficiency_potential(self) -> None:
        try:
            num_inputs = 1000
            layer = DynamicLiquidLayer(
                input_size=num_inputs, hidden_size=100, decay=0.9,
                density=0.01, use_rust=False)
            start_time = time.time()
            for _ in range(100):
                layer.forward(active_inputs=[random.randint(
                    0, 999)], prev_active_hidden=[])
            end_time = time.time()
            self.log("省エネ性能", True,
                     f"スパース・イベント駆動の処理速度を確認（{end_time - start_time:.4f}s）。")
        except Exception as e:
            self.log("省エネ性能", False, f"実行エラー: {e!s}")

    def check_rust_integration(self) -> None:
        try:
            try:
                from sara_engine import sara_rust_core
            except ImportError:
                try:
                    import sara_rust_core
                except ImportError:
                    from sara_engine import sara_rust_core

            engine = sara_rust_core.SpikeEngine()
            weights: list[dict[int, float]] = [{1: 0.5, 2: 1.0}, {}, {1: 0.8}]
            engine.set_weights(weights)

            active_spikes = [0, 2]
            result = engine.propagate(active_spikes, 0.4, 2)

            if isinstance(result, list):
                self.log("Rustコア統合", True,
                         f"Rust拡張モジュール(SpikeEngine)の演算を確認 (結果: {result})。")
            else:
                self.log("Rustコア統合", False, "Rust関数の戻り値が不正です。")
        except ImportError:
            self.log("Rustコア統合", False,
                     "Rust拡張モジュールがインポートできません。maturin build や pip install -e . を確認してください。")
        except AttributeError as e:
            self.log("Rustコア統合", False, f"Rustモジュールの関数呼び出しエラー: {e}")
        except Exception as e:
            self.log("Rustコア統合", False, f"実行エラー: {e!s}")

    def check_sdr_memory(self) -> None:
        try:
            try:
                from sara_engine import sara_rust_core
            except ImportError:
                try:
                    import sara_rust_core
                except ImportError:
                    from sara_engine import sara_rust_core

            memory = sara_rust_core.ScalableSDRMemory(0.1)
            memory.add_memory(1, [1, 2, 3, 4, 5])
            memory.add_memory(2, [6, 7, 8, 9, 10])
            results = memory.search([1, 2, 3], 5)

            if results and results[0][0] == 1:
                self.log("SDR連想メモリ", True, "ScalableSDRMemoryの高速検索機能を確認。")
            else:
                self.log("SDR連想メモリ", False, "検索結果が不正です。")
        except Exception as e:
            self.log("SDR連想メモリ", False, f"実行エラー: {e!s}")

    def check_wta_router(self) -> None:
        try:
            try:
                from sara_engine import sara_rust_core
            except ImportError:
                try:
                    import sara_rust_core
                except ImportError:
                    from sara_engine import sara_rust_core

            router = sara_rust_core.SpikeWTARouter(
                input_dim=10, num_experts=5, top_k=2)
            weights = [{0: 1.0, 1: 0.5, 2: 0.2} for _ in range(10)]
            router.set_weights(weights)
            winners = router.route([0, 1, 2], learning=True)

            if len(winners) <= 2:
                self.log("WTAルーティング", True, "SpikeWTARouterの競合学習とルーティング機能を確認。")
            else:
                self.log("WTAルーティング", False, "WTAの出力サイズが不正です。")
        except Exception as e:
            self.log("WTAルーティング", False, f"実行エラー: {e!s}")

    def check_direct_wiring(self) -> None:
        try:
            try:
                from sara_engine import sara_rust_core
            except ImportError:
                try:
                    import sara_rust_core
                except ImportError:
                    from sara_engine import sara_rust_core

            tokens = [1, 2, 3, 1, 2, 4, 1, 2, 3]
            synapses = sara_rust_core.build_direct_synapses(
                tokens, context_window=2)

            if isinstance(synapses, dict) and len(synapses) > 0:
                self.log("Direct Wiring", True, "コーパスからの直接シナプス結線構築機能を確認。")
            else:
                self.log("Direct Wiring", False, "戻り値が不正、または空です。")
        except Exception as e:
            self.log("Direct Wiring", False, f"実行エラー: {e!s}")

    def report(self) -> None:
        print("\n" + "=" * 50)
        total = len(self.results)
        passed = sum(1 for r in self.results if r[1])
        print(f"診断完了: {passed}/{total} 項目合格")
        if passed == total:
            print("ステータス: 健全（生物学的ポリシーおよびRust統合に完全準拠）")
        else:
            print("ステータス: 要調整（失敗項目を確認してください）")
        print("=" * 50)


if __name__ == "__main__":
    checker = SARAHealthCheck()
    checker.run_checks()
