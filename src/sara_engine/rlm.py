# file_path: src/sara_engine/rlm.py (v3 - STATE MACHINE APPROACH)
_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/rlm.py",
    "//": "タイトル: SARA RLM コントローラー (v3 - State Machine)",
    "//": "改善: SNNに頼らず、明示的な状態マシンでロジックを制御"
}

import re
from typing import List, Optional, Tuple
# from .sara_gpt_core import SaraGPT  # 実際の使用時はコメント解除

class TextEnvironment:
    """Environment E: ドキュメント管理と検索"""
    def __init__(self, text: str, chunk_size: int = 300):
        self.raw_text = text
        self.chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        self.last_search_hits: List[int] = []
        
    def get_metadata(self) -> str:
        return f"[DOC] {len(self.raw_text)} chars, {len(self.chunks)} chunks"
    
    def read_chunk(self, chunk_index: int) -> str:
        """Chunkを読み取り、完全にクリーンな内容を返す"""
        if 0 <= chunk_index < len(self.chunks):
            content = self.chunks[chunk_index]
            
            # 方法1: 行単位でクリーニング
            lines = content.split('\n')
            clean_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # ノイズパターンを除外
                if line.startswith(']') or line.startswith('['):
                    continue
                if 'EDACTED' in line or 'REDACTED' in line:
                    continue
                if line.startswith('Log '):
                    continue
                if line.startswith('-'):  # 区切り線
                    continue
                    
                # 有効な行のみ追加
                if len(line) > 3:
                    clean_lines.append(line)
            
            # 方法2: 最初の完全な文（Sectionまたは大文字）を見つける
            result_lines = []
            found_start = False
            
            for line in clean_lines:
                if not found_start:
                    # 完全な文の開始を検出
                    if (line.startswith('Section') or 
                        (line[0].isupper() and len(line) > 10 and 
                         not line.startswith('LOG') and 
                         not line.startswith('CONFIDENTIAL'))):
                        found_start = True
                        result_lines.append(line)
                else:
                    result_lines.append(line)
            
            # 結果が空なら元のクリーン版を返す
            if not result_lines and clean_lines:
                result_lines = clean_lines
            
            return '\n'.join(result_lines) if result_lines else content
            
        return "[ERROR] Index out of range."
    
    def search(self, keyword: str) -> str:
        """キーワードでドキュメントを検索"""
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
    """
    Root Model v3: State Machine Approach
    
    SNNの出力に完全に依存せず、明示的な状態マシンでロジックを制御。
    SNNはヒント生成のみに使用。
    """
    def __init__(self, sara_brain):  # SaraGPT型
        self.brain = sara_brain
        self.max_steps = 15
        self.last_read_content = ""
        self.search_history: List[str] = []
        
    def _extract_keywords(self, query: str) -> Tuple[List[str], str]:
        """クエリからキーワードを抽出"""
        clean_query = query.replace("?", "").replace(".", "").replace(",", "").lower()
        query_words = clean_query.split()
        stopwords = {"what", "is", "the", "does", "how", "to", "a", "an", "of", "in", 
                     "it", "this", "where", "when", "who", "why"}
        keywords = [w for w in query_words if w not in stopwords]
        
        if not keywords:
            keywords = ["information"]
        
        target_keyword = keywords[-1]
        return keywords, target_keyword
        
    def solve(self, long_context: str, query: str) -> str:
        env = TextEnvironment(long_context)
        
        keywords, target_keyword = self._extract_keywords(query)
        
        print(f"\n=== RLM Session (v3 - State Machine) ===")
        print(f"Query: {query}")
        print(f"Keywords: {keywords}")
        print(f"Target: {target_keyword}")
        
        # 初期化
        self.last_read_content = ""
        self.search_history = []
        
        # 状態マシン
        state = "INIT"
        current_keyword_idx = 0
        read_attempts = 0
        max_read_attempts = 3
        
        for step in range(1, self.max_steps + 1):
            print(f"\n[Step {step}] State: {state}")
            
            # 状態マシンのロジック
            if state == "INIT":
                # 最初のキーワードで検索
                keyword = keywords[current_keyword_idx] if current_keyword_idx < len(keywords) else target_keyword
                self.search_history.append(keyword)
                
                result = env.search(keyword)
                print(f"  ACTION: SEARCH '{keyword}'")
                print(f"  Result: {result}")
                
                if "Found" in result:
                    state = "READ"
                    read_attempts = 0
                else:
                    # 次のキーワードを試す
                    current_keyword_idx += 1
                    if current_keyword_idx >= len(keywords):
                        return "ERROR: No matching chunks found."
                    
            elif state == "READ":
                # Chunkを読む
                if not env.last_search_hits:
                    return "ERROR: No chunks to read."
                
                chunk_id = env.last_search_hits[0]
                content = env.read_chunk(chunk_id)
                self.last_read_content = content
                
                preview = content.replace("\n", " ")[:100]
                print(f"  ACTION: READ CHUNK {chunk_id}")
                print(f"  Result: {preview}...")
                
                # 答えを含むか確認
                contains_answer = any(kw.lower() in content.lower() for kw in keywords[:2])
                
                if contains_answer:
                    state = "EXTRACT"
                else:
                    # 別のキーワードで再検索
                    read_attempts += 1
                    if read_attempts >= max_read_attempts:
                        state = "EXTRACT"  # 諦めて抽出を試みる
                    else:
                        current_keyword_idx += 1
                        if current_keyword_idx < len(keywords):
                            state = "INIT"
                        else:
                            state = "EXTRACT"
                            
            elif state == "EXTRACT":
                # 答えを抽出
                print(f"  ACTION: EXTRACT ANSWER")
                
                if self.last_read_content:
                    answer = self._extract_answer_v3(self.last_read_content, target_keyword, keywords)
                    return answer
                else:
                    return "ERROR: No content to extract from."
        
        return "TIMEOUT: Maximum steps exceeded."
    
    def _extract_answer_v3(self, content: str, target_keyword: str, all_keywords: list) -> str:
        """
        Answer Extractor v3: シンプルで確実な抽出
        """
        # 行単位で処理
        lines = content.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            # ノイズパターンを除外
            if not line:
                continue
            if line.startswith("]") or line.startswith("["):
                continue
            if "EDACTED" in line or "REDACTED" in line:
                continue
            if line.startswith("Log "):
                continue
            
            # 有効な行のみ
            if len(line) > 5:
                clean_lines.append(line)
        
        # 文単位に再構成
        full_text = " ".join(clean_lines)
        sentences = re.split(r'(?<=[.!?])\s+', full_text)
        
        # キーワードマッチング
        candidates = []
        for s in sentences:
            s = s.strip()
            if not s or len(s) < 10:
                continue
                
            # ターゲットキーワードを含む
            if target_keyword.lower() in s.lower():
                candidates.append((s, 3))  # 優先度3
            # 他のキーワードを含む
            elif any(kw.lower() in s.lower() for kw in all_keywords):
                candidates.append((s, 2))  # 優先度2
            # Section で始まる
            elif s.startswith("Section"):
                candidates.append((s, 1))  # 優先度1
        
        # 優先度でソート
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if candidates:
            answer = candidates[0][0]
            return f'"{answer}"'
        else:
            return "ERROR: Unable to extract clean answer."