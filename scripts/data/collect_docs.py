# ディレクトリパス: scripts/collect_docs.py
# ファイルの日本語タイトル: 多目的ドキュメントエクストラクター
# ファイルの目的や内容: CSV, HTML, PDF等の多様なデータ形式からテキストを抽出し、SNN学習用の中間コーパス（interim）として保存する。日本語URLの自動エンコードに対応。

import os
import csv
import urllib.request
import urllib.parse
from html.parser import HTMLParser

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
        
        req = urllib.request.Request(encoded_url, headers={'User-Agent': 'Mozilla/5.0'})
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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        for text in texts:
            # 短すぎるノイズ行は学習の邪魔になるため除外
            if len(text) > 15:
                f.write(text + "\n")
                
    print(f"✅ {len(texts)} 件のテキストブロックを {output_path} に追加しました。")