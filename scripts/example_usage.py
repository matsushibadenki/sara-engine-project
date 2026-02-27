{
    "//": "ディレクトリパス: scripts/example_usage.py",
    "//": "ファイルの日本語タイトル: SARAライブラリ使用例 (終了コマンド対応版)",
    "//": "ファイルの目的や内容: 構築したsara_engineライブラリをインポートして推論を実行するテストスクリプト。チャットの終了機能を実装。"
}

import time
from sara_engine.inference import SaraInference

def main():
    print("SARAエンジンをロード中...")
    sara = SaraInference(model_path="models/distilled_sara_llm.msgpack")
    
    print("準備完了！終了するには 'quit' または 'exit' と入力してください。")
    while True:
        try:
            user_input = input("You: ")
        except (KeyboardInterrupt, EOFError): 
            break
            
        # 💡 終了コマンドの検知を追加（ここでループを即座に抜ける）
        if user_input.strip().lower() in ["quit", "exit"]:
            print("SARA: さようなら！またお話ししましょう。")
            break
            
        if not user_input.strip(): 
            continue
        
        sara.reset_buffer()
        
        start_time = time.time()
        
        prompt = f"You: {user_input}\nSARA:"
        
        response = sara.generate(
            prompt, 
            max_length=100, 
            top_k=1, 
            temperature=0.0,
            stop_conditions=["\n"]
        )
        
        elapsed_time = time.time() - start_time
        
        if not response:
            print("SARA: （記憶にありません）")
        else:
            # 最後の改行文字を消して綺麗に表示
            clean_response = response.replace('\n', '')
            print(f"SARA: {clean_response}  [⏱️ {elapsed_time:.3f}秒]")

if __name__ == "__main__":
    main()