# ディレクトリパス: scripts/collect_aozora.py
# ファイルの日本語タイトル: 青空文庫・高精度クリーニングコレクター
# ファイルの目的や内容: ZIP内の本文から、SNN学習の邪魔になる注釈、ルビ、特殊記号を完全に除去する。

import re
import zipfile
import urllib.request
import io
import os

def clean_aozora_text(text):
    """青空文庫のテキストからノイズを極限まで取り除く"""
    # 1. ヘッダーとフッターの削除
    parts = re.split(r'\-{5,}', text)
    if len(parts) >= 3:
        text = parts[2]

    # 2. 青空文庫特有の記号削除
    text = re.sub(r'［＃.*?］', '', text)  # 注釈
    text = re.sub(r'《.*?》', '', text)    # ルビ
    text = re.sub(r'｜', '', text)         # ルビ開始記号
    text = re.sub(r'〔.*?〕', '', text)    # 挿入句
    
    # 3. 不自然な改行を結合し、句点で分割
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'([^。！？\n])\n([^。！？\n])', r'\1\2', text)
    text = text.replace('。', '。\n').replace('！', '！\n').replace('？', '？\n')
    
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if len(line) > 5:
            lines.append(line)
            
    return '\n'.join(lines)

def collect_aozora(author_id, urls, author_name):
    print(f"--- {author_name} の作品を収集・浄化中 ---")
    corpus_path = "data/corpus.txt"
    os.makedirs("data", exist_ok=True)

    collected_count = 0
    with open(corpus_path, "a", encoding="utf-8") as f:
        for url in urls:
            try:
                response = urllib.request.urlopen(url)
                with zipfile.ZipFile(io.BytesIO(response.read())) as z:
                    for filename in z.namelist():
                        if filename.endswith(".txt"):
                            with z.open(filename) as txt_f:
                                content = txt_f.read().decode("shift_jis")
                                cleaned = clean_aozora_text(content)
                                f.write(cleaned + "\n")
                                collected_count += cleaned.count('\n')
                print(f"✅ 成功: {url}")
            except Exception as e:
                print(f"❌ 失敗: {url} ({e})")
    print(f"✨ {author_name} から {collected_count} 文を追加しました。")

if __name__ == "__main__":
    # 夏目漱石の作品リスト
    soseki_urls = [
        "https://www.aozora.gr.jp/cards/000148/files/773_ruby_5968.zip",  # 坊っちゃん
        "https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip",  # 吾輩は猫である
    ]
    collect_aozora("000148", soseki_urls, "夏目漱石")