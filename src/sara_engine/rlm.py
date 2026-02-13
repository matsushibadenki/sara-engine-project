# file_path: src/sara_engine/rlm.py
_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/rlm.py",
    "//": "タイトル: SARA RLM コントローラー (Extractor Mode)",
    "//": "目的: SNNがFINALを選択した際、直前に読んだテキストから答えを自動抽出する機能を追加"
}

import re
from typing import List, Optional
from .sara_gpt_core import SaraGPT

class TextEnvironment:
    """Environment E: ドキュメント管理と検索"""
    def __init__(self, text: str, chunk_size: int = 300):
        self.raw_text = text
        self.chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        self.last_search_hits = [] 
        
    def get_metadata(self) -> str:
        return f"[DOC] {len(self.raw_text)} chars, {len(self.chunks)} chunks"
    
    def read_chunk(self, chunk_index: int) -> str:
        if 0 <= chunk_index < len(self.chunks):
            return self.chunks[chunk_index] # 生のテキストを返すように変更
        return "[ERROR] Index out of range."
    
    def search(self, keyword: str) -> str:
        self.last_search_hits = []
        preview = ""
        
        # システム語句は無視
        if keyword.upper() in ["SEARCH", "READ", "FINAL", "CHUNK", "ACTION", "INFO"]:
            return f"[SEARCH] Ignored system word '{keyword}'."

        found = False
        for i, chunk in enumerate(self.chunks):
            if keyword.lower() in chunk.lower():
                self.last_search_hits.append(i)
                found = True
                if not preview:
                    idx = chunk.lower().find(keyword.lower())
                    start = max(0, idx - 15)
                    end = min(len(chunk), idx + 40)
                    preview = chunk[start:end].replace("\n", " ")
        
        if not found:
            return f"[SEARCH] No matches for '{keyword}'."
        
        hits_str = ", ".join(map(str, self.last_search_hits))
        return f"[SEARCH] Found '{keyword}' in chunks: {hits_str}. Preview: \"...{preview}...\""

class SaraRecursiveAgent:
    """Root Model: SNN + Answer Extractor"""
    def __init__(self, sara_brain: SaraGPT):
        self.brain = sara_brain
        self.max_steps = 10
        self.last_read_content = "" # 直前に読んだ内容を保持
        
    def solve(self, long_context: str, query: str) -> str:
        env = TextEnvironment(long_context)
        
        # --- ターゲットキーワード抽出 ---
        clean_query = query.replace("?", "").replace(".", "").replace(",", "")
        query_words = clean_query.split()
        stopwords = ["what", "is", "the", "does", "how", "to", "a", "an", "of", "in", "it", "this"]
        keywords = [w for w in query_words if w.lower() not in stopwords]
        
        if not keywords: keywords = ["SARA"] 
        target_keyword = keywords[-1]
        
        initial_context = (
            f"SYSTEM: START_RLM. "
            f"QUERY: {query} "
            f"COMMANDS: SEARCH, READ, FINAL."
        )
        
        print(f"\n--- RLM Session Started ---\nQuery: {query}")
        print(f"Target Keyword: '{target_keyword}'")
        
        self.brain.reset_state()
        self.brain.listen(initial_context, online_learning=True)
        
        final_answer = ""
        current_state_hint = "SEARCH"
        self.last_read_content = "" # リセット
        
        for step in range(1, self.max_steps + 1):
            print(f"\n[Step {step}] Thinking... (Hint: {current_state_hint})")
            
            # SNN思考生成
            thought = self.brain.think(
                length=8, 
                vocabulary=["SEARCH", "READ", "FINAL", "Chunk", target_keyword] + keywords, 
                trigger_text="ACTION:"
            )
            
            clean_thought = thought.replace("ACTION:", "").strip()
            print(f"  Raw Output: {clean_thought}")
            
            # --- 解析ロジック ---
            action_type = "NONE"
            action_arg = ""
            tokens = clean_thought.split()
            
            # 1. ヒント優先
            found_hint = False
            for token in tokens:
                if token.upper() == current_state_hint:
                    action_type = current_state_hint
                    found_hint = True
                    try:
                        idx = tokens.index(token) + 1
                        if idx < len(tokens): action_arg = tokens[idx]
                    except: pass
                    break
            
            # 2. 最初に見つかったコマンド
            if not found_hint:
                for token in tokens:
                    if token.upper() in ["SEARCH", "READ", "FINAL"]:
                        action_type = token.upper()
                        try: action_arg = tokens[tokens.index(token) + 1]
                        except: pass
                        break
            
            # 3. 強制遷移
            if action_type == "NONE":
                action_type = current_state_hint
                print(f"  -> AUTO-ASSIST: SNN silent, forcing state {action_type}")

            # --- 実行 ---
            result_text = ""
            
            if action_type == "SEARCH":
                if not action_arg or action_arg.upper() in ["CHUNK", "INFO", "SEARCH", "READ", "FINAL", "ACTION"]:
                    keyword = target_keyword
                    print(f"  -> AUTO-CORRECT: Replaced invalid arg '{action_arg}' with '{keyword}'")
                else:
                    keyword = action_arg
                
                print(f"  -> ACTION: SEARCH '{keyword}'")
                result_text = env.search(keyword)
                
                if "Found" in result_text:
                    current_state_hint = "READ"
                
            elif action_type == "READ":
                chunk_id = -1
                if action_arg and action_arg.isdigit():
                    chunk_id = int(action_arg)
                elif env.last_search_hits:
                    chunk_id = env.last_search_hits[0]
                    print(f"  -> AUTO-ASSIST: Using Chunk {chunk_id} from search results")
                else:
                    chunk_id = 0
                    print(f"  -> AUTO-ASSIST: Fallback to Chunk 0")
                
                if chunk_id >= 0:
                    content = env.read_chunk(chunk_id)
                    self.last_read_content = content # 読んだ内容を保存
                    # 表示用には短縮版を
                    preview = content.replace("\n", " ")[:60]
                    result_text = f"[CHUNK {chunk_id}] {preview}..."
                    current_state_hint = "FINAL"

            elif action_type == "FINAL":
                # --- 自動抽出ロジック (Answer Extractor) ---
                print(f"  -> DECISION: FINAL command detected. Extracting answer from context...")
                
                # 直前に読んだテキストがある場合、そこから答えを探す
                if self.last_read_content:
                    # 文単位に分割
                    sentences = re.split(r'(?<=[.!?])\s+', self.last_read_content)
                    candidates = []
                    
                    # ターゲットキーワードを含む文を探す
                    for s in sentences:
                        if target_keyword.lower() in s.lower():
                            candidates.append(s)
                            
                    if candidates:
                        # 最も関連性が高そうな文（最初のマッチ）を採用
                        final_answer = f"Found in context: \"{candidates[0].strip()}\""
                    else:
                        # キーワードがなければ全体から適当に抜粋
                        final_answer = f"Context snippet: \"{self.last_read_content[:100]}...\""
                else:
                    # 何も読んでいない場合
                    final_answer = "No document read yet."
                
                break
            
            # 観察
            observation = f"RESULT: {result_text}"
            print(f"  -> OBSERVATION: {result_text}")
            self.brain.listen(observation, online_learning=True)
            
        if not final_answer:
            final_answer = "Timeout."
            
        return final_answer