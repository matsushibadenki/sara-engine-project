# {
#     "//": "ディレクトリパス: scripts/utils/manage_db.py",
#     "//": "ファイルの日本語タイトル: SARAコーパス・データベース・マネージャー",
#     "//": "ファイルの目的や内容: プレーンテキストや対話データ(JSONL)をDBで一元管理し、自己組織化学習用・蒸留学習用それぞれの形式へ柔軟にエクスポートする。"
# }

import sqlite3
import os
import json
import re

class SaraCorpusDB:
    def __init__(self, db_path="data/sara_corpus.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS corpus (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text_type TEXT DEFAULT 'document', -- 'document' or 'chat'
            content TEXT UNIQUE,
            source TEXT,
            lang TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.conn.execute(query)
        self.conn.commit()

    def add_texts(self, texts, text_type="document", source="unknown", lang="ja"):
        query = "INSERT OR IGNORE INTO corpus (text_type, content, source, lang) VALUES (?, ?, ?, ?)"
        data = [(text_type, t.strip(), source, lang) for t in texts if len(t.strip()) > 2]
        cur = self.conn.executemany(query, data)
        self.conn.commit()
        return cur.rowcount

    def import_file(self, file_path, source_name=None):
        """ファイルを解析してDBに登録する"""
        if not os.path.exists(file_path):
            print(f"[エラー] ファイルが見つかりません: {file_path}")
            return 0
            
        if source_name is None:
            source_name = os.path.basename(file_path)
            
        added_count = 0
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                added_count = self.add_texts(lines, text_type="document", source=source_name)
                
        elif file_path.endswith('.jsonl'):
            chats = []
            docs = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        data = json.loads(line)
                        if "prompt" in data and "completion" in data:
                            chats.append(json.dumps(data, ensure_ascii=False))
                        elif "text" in data:
                            docs.append(data["text"])
                    except json.JSONDecodeError:
                        pass
            if chats:
                added_count += self.add_texts(chats, text_type="chat", source=source_name)
            if docs:
                added_count += self.add_texts(docs, text_type="document", source=source_name)
                
        return added_count

    def get_stats(self):
        cur = self.conn.cursor()
        cur.execute("SELECT text_type, COUNT(*) FROM corpus GROUP BY text_type")
        return cur.fetchall()

    def export_for_self_organized(self, out_path="data/processed/corpus.txt"):
        """自己組織化SNN学習用に、すべてのテキストを連続したプレーンテキストとして出力"""
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        count = 0
        with open(out_path, 'w', encoding='utf-8') as f:
            cur = self.conn.execute("SELECT text_type, content FROM corpus")
            for row in cur.fetchall():
                t_type, content = row
                if t_type == "chat":
                    data = json.loads(content)
                    f.write(f"User: {data['prompt']}\nSARA: {data['completion']}\n")
                else:
                    f.write(f"{content}\n")
                count += 1
        return count

    def export_for_distillation(self, out_path="data/raw/chat_data.jsonl"):
        """蒸留学習(BP)用に、プロンプト・コンプリーションのペアを含むJSONLとして出力"""
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        count = 0
        with open(out_path, 'w', encoding='utf-8') as f:
            cur = self.conn.execute("SELECT text_type, content FROM corpus")
            for row in cur.fetchall():
                t_type, content = row
                if t_type == "chat":
                    f.write(f"{content}\n")
                else:
                    text = content.strip()
                    if len(text) < 12:
                        continue
                    head = re.split(r"[、。]", text, maxsplit=1)[0].strip("「」『』 ")
                    if 2 <= len(head) <= 24:
                        pair = {"prompt": f"{head}について教えてください。", "response": text}
                    else:
                        pair = {"prompt": "この内容を説明してください。", "response": text}
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                count += 1
        return count
