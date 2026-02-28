# ディレクトリパス: scripts/manage_db.py
# ファイルの日本語タイトル: SARAコーパス・データベース・マネージャー
# ファイルの目的や内容: SQLiteを使用して大量のコーパスを管理。重複排除と高速な読み出しを実現する。

import sqlite3
import os

class SaraCorpusDB:
    def __init__(self, db_path="data/sara_corpus.db"):
        os.makedirs("data", exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS corpus (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT UNIQUE,
            source TEXT,
            lang TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.conn.execute(query)
        self.conn.commit()

    def add_texts(self, texts, source="unknown", lang="ja"):
        query = "INSERT OR IGNORE INTO corpus (content, source, lang) VALUES (?, ?, ?)"
        data = [(t.strip(), source, lang) for t in texts if len(t.strip()) > 5]
        cur = self.conn.executemany(query, data)
        self.conn.commit()
        return cur.rowcount

    def get_count(self):
        return self.conn.execute("SELECT COUNT(*) FROM corpus").fetchone()[0]

    def fetch_all(self):
        cur = self.conn.execute("SELECT id, content FROM corpus ORDER BY id")
        while True:
            row = cur.fetchone()
            if row is None: break
            yield row