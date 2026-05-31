# ディレクトリパス: scripts/data/collect_docs.py
# ファイルの日本語タイトル: 多目的ドキュメントエクストラクター
# ファイルの目的や内容: CSV, HTML, PDF等の多様なデータ形式からテキストを抽出し、SNN学習用の中間コーパス（interim）として保存する。さらに、対話学習用のQAペアを自動生成して保存する。

import os
import sys
import csv
import json
import urllib.request
import urllib.parse
from html.parser import HTMLParser

# srcディレクトリをパスに追加してモジュールをインポートできるようにする
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'src')))
try:
    from sara_engine.utils.corpus import clean_corpus_lines, generate_conversational_pairs
except ImportError:
    print("⚠️ sara_engine.utils.corpus が見つかりません。デフォルトの処理を使用します。")

    def clean_corpus_lines(lines: list[str], merge_wrapped: bool = False, **kwargs) -> list[str]:  # type: ignore[misc]
        return lines

    def generate_conversational_pairs(lines: list[str]) -> list[tuple[str, str]]:
        return []


try:
    import PyPDF2
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


class SimpleHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_data = []
        self.in_script_or_style = False

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'nav', 'header', 'footer'):
            self.in_script_or_style = True

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'nav', 'header', 'footer'):
            self.in_script_or_style = False

    def handle_data(self, data):
        if not self.in_script_or_style:
            text = data.strip()
            if text:
                self.text_data.append(text)


def extract_from_csv(file_path, text_columns=None):
    """CSVからテキストを抽出（特定の列インデックスを指定可能）"""
    extracted_texts = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        for row in reader:
            if text_columns:
                for col_idx in text_columns:
                    if col_idx < len(row) and row[col_idx].strip():
                        extracted_texts.append(row[col_idx].strip())
            else:
                text = " ".join([cell.strip() for cell in row if cell.strip()])
                if text:
                    extracted_texts.append(text)
    return extracted_texts


def extract_from_html(url):
    """URLからWebページの本文を抽出"""
    extracted_texts = []
    try:
        # 日本語URLを処理するために、安全な文字以外をURLエンコードする
        encoded_url = urllib.parse.quote(url, safe=':/?=&%')

        req = urllib.request.Request(
            encoded_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            html_content = response.read().decode('utf-8', errors='ignore')
            parser = SimpleHTMLParser()
            parser.feed(html_content)
            extracted_texts = parser.text_data
    except Exception as e:
        print(f"❌ HTML抽出失敗 ({url}): {e}")
    return extracted_texts


def extract_from_pdf(file_path):
    """PDFファイルからページごとのテキストを抽出"""
    if not HAS_PYPDF:
        print("⚠️ PDFを処理するには 'PyPDF2' ライブラリが必要です。'pip install PyPDF2' を実行してください。")
        return []

    extracted_texts = []
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    # PDF特有の不自然な改行をスペースに置換
                    text = text.replace('\n', ' ')
                    extracted_texts.append(text.strip())
    except Exception as e:
        print(f"❌ PDF抽出失敗 ({file_path}): {e}")
    return extracted_texts


def process_document(source_type, source, output_path, **kwargs):
    print(f"--- {source_type.upper()} からテキストを抽出中: {source} ---")

    if source_type == 'csv':
        texts = extract_from_csv(source, kwargs.get('text_columns'))
    elif source_type == 'html':
        texts = extract_from_html(source)
    elif source_type == 'pdf':
        texts = extract_from_pdf(source)
    else:
        print(f"❌ 未対応のフォーマットです: {source_type}")
        return

    if not texts:
        print("ℹ️ 抽出できるテキストがありませんでした。")
        return

    # 前処理と行の結合（段落化、ノイズ除去、箇条書きの正規化など）
    cleaned_texts = clean_corpus_lines(texts, merge_wrapped=True)

    # 1. 通常の知識コーパスとして保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    valid_count = 0
    with open(output_path, "a", encoding="utf-8") as f:
        for text in cleaned_texts:
            # 短すぎるノイズ行は学習の邪魔になるため除外
            if len(text) > 15:
                f.write(text + "\n")
                valid_count += 1

    print(f"✅ {valid_count} 件のテキストブロックを {output_path} に追加しました。")

    # 2. 会話形式（チャット学習用）のペアを自動生成して保存
    chat_pairs = generate_conversational_pairs(cleaned_texts)
    if chat_pairs:
        # 出力ディレクトリ内の 'chat_data.jsonl' に追記する
        chat_output_path = os.path.join(
            os.path.dirname(output_path), "chat_data.jsonl")
        with open(chat_output_path, "a", encoding="utf-8") as f_chat:
            for prompt, response in chat_pairs:
                json_line = json.dumps(
                    {"prompt": prompt, "response": response}, ensure_ascii=False)
                f_chat.write(json_line + "\n")
        print(f"✅ {len(chat_pairs)} 件の対話ペアを {chat_output_path} に追加しました。")
