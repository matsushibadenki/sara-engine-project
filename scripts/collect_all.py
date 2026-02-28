# ディレクトリパス: scripts/collect_all.py
# ファイルの日本語タイトル: SARA統合コーパス・コレクター（重複排除機能付き）
# ファイルの目的や内容: 異なるソースのクリーニングを一手に引き受け、一貫性のある学習データを作成する。数式コーパスの統合処理も追加。

import os
import re

class CorpusIntegrator:
    def __init__(self, output_path="data/corpus.txt"):
        self.output_path = output_path
        self.seen_lines = set()
        os.makedirs("data", exist_ok=True)
        
        # 既存のコーパスを読み込み、重複チェック用セットを構築
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    self.seen_lines.add(line.strip())

    def clean_generic(self, text):
        # 共通：URL削除、連続空白整理
        text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    def clean_wikipedia(self, text):
        # Wikipedia：マークアップ削除
        text = re.sub(r'\{\{.*?\}\}', '', text)
        text = re.sub(r'\[\[(?:ファイル|画像|File|Image):.*?\]\]', '', text)
        text = re.sub(r'\[\[([^|]*?)\|([^|]*?)\]\]', r'\2', text)
        text = re.sub(r'\[\[(.*?)\]\]', r'\1', text)
        return text

    def clean_arxiv(self, text):
        # arXiv：LaTeX数式・コマンド削除 (通常のテキスト抽出用)
        text = re.sub(r'\$.*?\$', '', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        text = re.sub(r'\{.*?\}', '', text)
        return text
        
    def clean_math(self, text):
        # 数式コーパス用: LaTeXのバックスラッシュや数式記号を保持するため、過度なクリーニングは行わない
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    def add_source(self, raw_text, source_type="generic"):
        if source_type == "math":
            text = self.clean_math(raw_text)
        else:
            text = self.clean_generic(raw_text)
            
            if source_type == "wikipedia":
                text = self.clean_wikipedia(text)
            elif source_type == "arxiv":
                text = self.clean_arxiv(text)
        
        # 1行1文に分割
        text = text.replace('。', '。\n').replace('！', '！\n').replace('？', '？\n')
        
        new_lines_count = 0
        with open(self.output_path, "a", encoding="utf-8") as f:
            for line in text.split('\n'):
                line = line.strip()
                # 重複していない有意義な長さの行のみ採用
                if len(line) > 5 and line not in self.seen_lines:
                    f.write(line + "\n")
                    self.seen_lines.add(line)
                    new_lines_count += 1
        
        if new_lines_count > 0:
            print(f"📥 {source_type} から {new_lines_count} 文の新しい知識を統合しました。")
        else:
            print(f"ℹ️ {source_type} からの入力はすべて重複または短すぎたため、追加されませんでした。")

if __name__ == "__main__":
    integrator = CorpusIntegrator()
    
    # 既存の青空文庫などのデータ流し込み処理がある場合はここに記述
    
    # 数式コーパスの読み込みと結合
    math_corpus_path = "data/math_corpus.txt"
    if os.path.exists(math_corpus_path):
        print(f"--- 数式コーパス ({math_corpus_path}) を統合中 ---")
        with open(math_corpus_path, "r", encoding="utf-8") as f:
            math_text = f.read()
        integrator.add_source(math_text, source_type="math")
    else:
        print(f"⚠️ {math_corpus_path} が見つかりません。")