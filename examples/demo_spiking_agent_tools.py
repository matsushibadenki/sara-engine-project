_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_spiking_agent_tools.py",
    "//": "ファイルの日本語タイトル: SpikingLLMエージェント 汎用ツール連携デモ",
    "//": "ファイルの目的や内容: 抽象化されたToolRegistryを用い、外部から任意のPython関数をSNNのアクションスパイクとして割り当て、モデルに自律実行させる完全統合テスト。"
}

import re
from sara_engine.agent.sara_agent import SaraAgent

# --- 外部から登録するカスタムツールの定義 ---
def my_calculator(context: str) -> str:
    """数式を抽出して計算するツール"""
    target = context.split("<CALC>")[0] if "<CALC>" in context else context
    match = re.search(r'([0-9\+\-\*\/\s\(\)\.]+)(?:は|の計算|はいくつ|の答え|)$', target.strip())
    expression = match.group(1).strip() if match else target.split()[-1]
    try:
        res = eval(expression)
        return str(int(res)) if isinstance(res, float) and res.is_integer() else str(res)
    except:
        return "ERROR"

def my_weather_api(context: str) -> str:
    """ダミーの天気APIツール"""
    if "東京" in context: return "晴れ（25℃）"
    elif "大阪" in context: return "曇り（22℃）"
    return "不明"

def main():
    print("===" * 15)
    print("[INFO] Starting Agentic SARA (SpikingLLM + Dynamic Tools)")
    print("===" * 15)
    
    agent = SaraAgent()
    
    # 🌟 外部からSNNに対してツールを動的にアタッチする
    print("[INFO] Registering external tools to the agent's nervous system...")
    agent.register_tool("<CALC>", my_calculator)
    agent.register_tool("<WEATHER>", my_weather_api)

    # --- 1. SNNへの振る舞いの教示 (Teaching Mode) ---
    print("\n[TRAINING] Teaching the SNN how to route to external tools...")
    training_data = [
        "15 + 28 は <CALC> 43 = 43 です。",
        "100 - 42 は <CALC> 58 = 58 です。",
        "東京 の 天気 は <WEATHER> 晴れ（25℃） = 晴れ です。",
        "大阪 の 天気 は <WEATHER> 曇り（22℃） = 曇り です。"
    ]
    
    # 複数回教示してシナプス結合を強化
    for text in training_data:
        for _ in range(3):
            agent.chat(text, teaching_mode=True)
            
    print("[TRAINING] Done.")

    # --- 2. 推論・自律ツール実行テスト ---
    prompts = [
        "python_expert: 100 - 42 は <CALC>",
        "general: 東京 の 天気 は <WEATHER>"
    ]
    
    for prompt in prompts:
        print(f"\n[INFERENCE] Prompt: {prompt}")
        print("[INFERENCE] Agent takes over...")
        
        # SpikingLLMが内部で推論し、必要なスパイクを発火させる
        response = agent.chat(prompt, teaching_mode=False)
        print("\n[RESULT] Agent Output:")
        print(response)

if __name__ == "__main__":
    main()