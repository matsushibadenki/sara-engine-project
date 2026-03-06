# {
#     "//": "ディレクトリパス: tests/test_cognitive_agent.py",
#     "//": "ファイルの日本語タイトル: 認知アーキテクチャ・テスト",
#     "//": "ファイルの目的や内容: CognitiveArchitecture の自律ループ（感覚→無意識→意識→行動→報酬）が正常に機能し、ドーパミンによる価値観の更新が行われるかをテストする。"
# }

from sara_engine.models.cognitive_architecture import CognitiveArchitecture
import random

def test_cognitive_loop():
    print("--- 🧠 認知アーキテクチャのテストを開始 ---")
    # 小規模なエージェントを生成
    agent = CognitiveArchitecture(n_sensory=10, n_liquid=50, n_actions=3)
    
    actions_taken = 0
    for step in range(200): # 200ミリ秒のシミュレーション
        # 20%の確率でランダムな感覚ノイズが入ってくる
        sensory_input = [random.random() < 0.2 for _ in range(10)]
        
        # エージェントの内部時間を進め、行動を問う
        action_id = agent.step_environment(sensory_input)
        
        if action_id != -1:
            actions_taken += 1
            print(f"[Step {step:03d}] 意識に浮上 💡 -> アクション {action_id} を選択")
            
            # 簡易的な環境のフィードバック（アクション1が正解と仮定）
            reward = 1.0 if action_id == 1 else -0.2
            agent.apply_reward(reward)
            
            print(f"          環境からの報酬: {reward:+.1f} | ドーパミン放出による学習を実行 (期待価値 V: {agent.expected_reward:.4f})")
            
    print(f"\n--- ✅ テスト完了：エラーなし。合計 {actions_taken} 回の意思決定が行われました ---")

if __name__ == "__main__":
    test_cognitive_loop()