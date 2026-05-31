# ディレクトリパス: scripts/collect_all.py
# ファイルの日本語タイトル: SARA統合コーパス・コレクター（高品質フィルター・禁則処理機能付き）
# ファイルの目的や内容: 異なるソースのクリーニングを一手に引き受け、一貫性のある高品質な学習データを data/processed/ に作成する。不完全な文やノイズを強力に弾く処理を追加。

import os
import re

class CorpusIntegrator:
    def __init__(self, output_path):
        self.output_path = output_path
        self.seen_lines = set()
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    self.seen_lines.add(line.strip())

    def clean_generic(self, text):
        text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    def clean_wikipedia(self, text):
        text = re.sub(r'\{\{.*?\}\}', '', text)
        text = re.sub(r'\[\[(?:ファイル|画像|File|Image):.*?\]\]', '', text)
        text = re.sub(r'\[\[([^|]*?)\|([^|]*?)\]\]', r'\2', text)
        text = re.sub(r'\[\[(.*?)\]\]', r'\1', text)
        return text

    def clean_arxiv(self, text):
        text = re.sub(r'\$.*?\$', '', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        text = re.sub(r'\{.*?\}', '', text)
        return text
        
    def clean_math(self, text):
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    def clean_document(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[▼▶■◆●]', '', text)
        return text.strip()

    def add_source(self, raw_text, source_type="generic"):
        if source_type == "math":
            text = self.clean_math(raw_text)
        elif source_type == "document":
            text = self.clean_document(raw_text)
        else:
            text = self.clean_generic(raw_text)
            if source_type == "wikipedia":
                text = self.clean_wikipedia(text)
            elif source_type == "arxiv":
                text = self.clean_arxiv(text)
        
        # 禁則処理：閉じカッコ類を前の文のセットにする
        text = re.sub(r'([。！？]+)([」）』】〕〉》\]\)]+)', r'\1\2\n', text)
        text = re.sub(r'([。！？]+)([^」）』】〕〉》\]\)\n])', r'\1\n\2', text)
        
        new_lines_count = 0
        with open(self.output_path, "a", encoding="utf-8") as f:
            for line in text.split('\n'):
                line = line.strip()
                
                # 💡 高品質フィルター 1: 文頭のゴミ（句読点、閉じカッコなど）を削除
                line = re.sub(r'^[。、！？\s」）』】〕〉》\]\)]+', '', line)
                
                # 💡 高品質フィルター 2: 有意義な長さの確保（ノイズ弾きのために10文字以上に設定）
                if len(line) < 10:
                    continue
                    
                # 💡 高品質フィルター 3: 不自然な文頭の除外（「を」「に」「が」等の助詞から始まる文は、前後の文脈が切断されたノイズとみなす）
                if re.match(r'^[をにがはでへとのも]', line):
                    continue
                
                # 重複チェックを通過したものだけを書き込む
                if line not in self.seen_lines:
                    f.write(line + "\n")
                    self.seen_lines.add(line)
                    new_lines_count += 1
        
        if new_lines_count > 0:
            print(f"📥 {source_type} から {new_lines_count} 文の高品質な知識を統合しました。")
        else:
            print(f"ℹ️ {source_type} からの入力はすべて重複または品質基準を満たさなかったため追加されませんでした。")