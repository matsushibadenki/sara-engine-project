# {
#     "//": "ディレクトリパス: scripts/utils/manage_db.py",
#     "//": "ファイルの日本語タイトル: SARAコーパス・データベース・マネージャー",
#     "//": "ファイルの目的や内容: プレーンテキストや対話データ(JSONL)をDBで一元管理し、自己組織化学習用・蒸留学習用それぞれの形式へ柔軟にエクスポートする。"
# }

import sqlite3
import os
import json
import re
from typing import List


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
            category TEXT DEFAULT 'general',
            quality_score REAL DEFAULT 1.0,
            source_version TEXT DEFAULT '',
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.conn.execute(query)
        self._ensure_schema()
        self.conn.commit()

    def _ensure_schema(self):
        columns = {
            "category": "TEXT DEFAULT 'general'",
            "quality_score": "REAL DEFAULT 1.0",
            "source_version": "TEXT DEFAULT ''",
            "is_active": "INTEGER DEFAULT 1",
        }
        existing = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(corpus)").fetchall()
            if len(row) > 1
        }
        for column_name, definition in columns.items():
            if column_name not in existing:
                self.conn.execute(
                    f"ALTER TABLE corpus ADD COLUMN {column_name} {definition}"
                )

    def _normalize_category(self, category, text_type):
        if isinstance(category, str) and category.strip():
            return category.strip().lower()
        return "dialogue" if text_type == "chat" else "document"

    def _normalize_quality_score(self, quality_score):
        try:
            score = float(quality_score)
        except (TypeError, ValueError):
            score = 1.0
        return max(0.0, min(score, 1.0))

    def _build_material_filter_clause(
        self,
        category=None,
        min_quality_score=0.0,
        source=None,
        include_inactive=True,
    ):
        clauses: List[str] = []
        params: List[object] = []
        if not include_inactive:
            clauses.append("is_active = 1")
        normalized_category = str(category or "").strip().lower()
        if normalized_category:
            clauses.append("category = ?")
            params.append(normalized_category)
        normalized_source = str(source or "").strip()
        if normalized_source:
            clauses.append("source = ?")
            params.append(normalized_source)
        try:
            min_score = float(min_quality_score)
        except (TypeError, ValueError):
            min_score = 0.0
        clauses.append("quality_score >= ?")
        params.append(max(0.0, min(min_score, 1.0)))
        return " AND ".join(clauses), params

    def add_texts(
        self,
        texts,
        text_type="document",
        source="unknown",
        lang="ja",
        category=None,
        quality_score=1.0,
        source_version="",
        is_active=True,
    ):
        query = (
            "INSERT OR IGNORE INTO corpus "
            "(text_type, content, source, lang, category, quality_score, source_version, is_active) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        )
        normalized_category = self._normalize_category(category, text_type)
        normalized_quality = self._normalize_quality_score(quality_score)
        normalized_version = str(source_version or "").strip()
        active_flag = 1 if bool(is_active) else 0
        data = [
            (
                text_type,
                t.strip(),
                source,
                lang,
                normalized_category,
                normalized_quality,
                normalized_version,
                active_flag,
            )
            for t in texts
            if len(t.strip()) > 2
        ]
        cur = self.conn.executemany(query, data)
        self.conn.commit()
        return cur.rowcount

    def import_file(
        self,
        file_path,
        source_name=None,
        category=None,
        lang="ja",
        source_version="",
        quality_score=1.0,
        is_active=True,
    ):
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
                added_count = self.add_texts(
                    lines,
                    text_type="document",
                    source=source_name,
                    lang=lang,
                    category=category or "document",
                    quality_score=quality_score,
                    source_version=source_version,
                    is_active=is_active,
                )
                
        elif file_path.endswith('.jsonl'):
            chats = []
            docs = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        if "prompt" in data and "completion" in data:
                            chats.append(
                                {
                                    "content": json.dumps(data, ensure_ascii=False),
                                    "lang": data.get("lang", lang),
                                    "category": data.get("category", category or "dialogue"),
                                    "quality_score": data.get("quality_score", quality_score),
                                    "source_version": data.get("source_version", source_version),
                                    "is_active": data.get("is_active", is_active),
                                }
                            )
                        elif "prompt" in data and "response" in data:
                            normalized = {
                                "prompt": data["prompt"],
                                "completion": data["response"],
                            }
                            chats.append(
                                {
                                    "content": json.dumps(normalized, ensure_ascii=False),
                                    "lang": data.get("lang", lang),
                                    "category": data.get("category", category or "dialogue"),
                                    "quality_score": data.get("quality_score", quality_score),
                                    "source_version": data.get("source_version", source_version),
                                    "is_active": data.get("is_active", is_active),
                                }
                            )
                        elif "text" in data:
                            docs.append(
                                {
                                    "content": data["text"],
                                    "lang": data.get("lang", lang),
                                    "category": data.get("category", category or "document"),
                                    "quality_score": data.get("quality_score", quality_score),
                                    "source_version": data.get("source_version", source_version),
                                    "is_active": data.get("is_active", is_active),
                                }
                            )
                    except json.JSONDecodeError:
                        pass
            if chats:
                for item in chats:
                    added_count += self.add_texts(
                        [item["content"]],
                        text_type="chat",
                        source=source_name,
                        lang=item["lang"],
                        category=item["category"],
                        quality_score=item["quality_score"],
                        source_version=item["source_version"],
                        is_active=item["is_active"],
                    )
            if docs:
                for item in docs:
                    added_count += self.add_texts(
                        [item["content"]],
                        text_type="document",
                        source=source_name,
                        lang=item["lang"],
                        category=item["category"],
                        quality_score=item["quality_score"],
                        source_version=item["source_version"],
                        is_active=item["is_active"],
                    )
                
        return added_count

    def get_stats(self):
        cur = self.conn.cursor()
        cur.execute("SELECT text_type, COUNT(*) FROM corpus GROUP BY text_type")
        return cur.fetchall()

    def get_material_summary(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT
                COUNT(*) AS total_count,
                SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) AS active_count,
                SUM(CASE WHEN is_active = 0 THEN 1 ELSE 0 END) AS inactive_count,
                COALESCE(AVG(quality_score), 0.0) AS avg_quality_score
            FROM corpus
            """
        )
        row = cur.fetchone() or (0, 0, 0, 0.0)

        cur.execute(
            """
            SELECT category, COUNT(*)
            FROM corpus
            GROUP BY category
            ORDER BY COUNT(*) DESC, category ASC
            """
        )
        categories = cur.fetchall()

        return {
            "total_count": int(row[0] or 0),
            "active_count": int(row[1] or 0),
            "inactive_count": int(row[2] or 0),
            "avg_quality_score": float(row[3] or 0.0),
            "categories": [(str(category), int(count)) for category, count in categories],
        }

    def get_review_summary(self):
        cur = self.conn.cursor()

        def _group_rows(group_by):
            rows = cur.execute(
                f"""
                SELECT {group_by}, COUNT(*), COALESCE(AVG(quality_score), 0.0)
                FROM corpus
                GROUP BY {group_by}
                ORDER BY COUNT(*) DESC, {group_by} ASC
                """
            ).fetchall()
            return [
                {
                    "key": str(key or ""),
                    "count": int(count or 0),
                    "avg_quality_score": float(avg_quality or 0.0),
                }
                for key, count, avg_quality in rows
            ]

        status_rows = cur.execute(
            """
            SELECT is_active, COUNT(*), COALESCE(AVG(quality_score), 0.0)
            FROM corpus
            GROUP BY is_active
            ORDER BY is_active DESC
            """
        ).fetchall()
        return {
            "by_category": _group_rows("category"),
            "by_source": _group_rows("source"),
            "by_lang": _group_rows("lang"),
            "by_status": [
                {
                    "key": "active" if int(is_active or 0) == 1 else "inactive",
                    "count": int(count or 0),
                    "avg_quality_score": float(avg_quality or 0.0),
                }
                for is_active, count, avg_quality in status_rows
            ],
        }

    def _build_export_query(
        self,
        category=None,
        min_quality_score=0.0,
        source=None,
        show_inactive=False,
    ):
        where_clause, params = self._build_material_filter_clause(
            category=category,
            min_quality_score=min_quality_score,
            source=source,
            include_inactive=show_inactive,
        )
        return (
            f"""
            SELECT text_type, content, category, quality_score, source, source_version, lang, is_active
            FROM corpus
            WHERE {where_clause}
            ORDER BY quality_score DESC, id ASC
            """,
            params,
        )

    def list_materials(
        self,
        category=None,
        min_quality_score=0.0,
        source=None,
        show_inactive=False,
        limit=20,
    ):
        query, params = self._build_export_query(
            category=category,
            min_quality_score=min_quality_score,
            source=source,
            show_inactive=show_inactive,
        )
        try:
            limit_value = max(1, int(limit))
        except (TypeError, ValueError):
            limit_value = 20
        query += "\nLIMIT ?"
        params = list(params) + [limit_value]
        cur = self.conn.execute(query, params)
        rows = cur.fetchall()
        materials = []
        for row in rows:
            text_type, content, row_category, quality_score, source, source_version, lang, is_active = row
            preview = content
            if text_type == "chat":
                try:
                    payload = json.loads(content)
                    preview = f"{payload.get('prompt', '')} -> {payload.get('completion', '')}"
                except (TypeError, ValueError, json.JSONDecodeError):
                    preview = content
            preview = str(preview).strip().replace("\n", " ")
            materials.append(
                {
                    "text_type": text_type,
                    "category": row_category,
                    "quality_score": float(quality_score),
                    "source": source,
                    "source_version": source_version,
                    "lang": lang,
                    "is_active": bool(is_active),
                    "preview": preview[:120],
                }
            )
        return materials

    def summarize_export_plan(
        self,
        category=None,
        min_quality_score=0.0,
        source=None,
        show_inactive=False,
    ):
        query, params = self._build_export_query(
            category=category,
            min_quality_score=min_quality_score,
            source=source,
            show_inactive=show_inactive,
        )
        wrapped_query = (
            f"SELECT text_type, category, COUNT(*), COALESCE(AVG(quality_score), 0.0) "
            f"FROM ({query}) GROUP BY text_type, category ORDER BY COUNT(*) DESC, category ASC"
        )
        cur = self.conn.execute(wrapped_query, params)
        rows = cur.fetchall()
        total = sum(int(row[2] or 0) for row in rows)
        return {
            "total_count": total,
            "items": [
                {
                    "text_type": str(text_type),
                    "category": str(row_category),
                    "count": int(count or 0),
                    "avg_quality_score": float(avg_quality or 0.0),
                }
                for text_type, row_category, count, avg_quality in rows
            ],
        }

    def set_material_active_state(
        self,
        is_active,
        category=None,
        min_quality_score=0.0,
        source=None,
        include_inactive=True,
    ):
        where_clause, params = self._build_material_filter_clause(
            category=category,
            min_quality_score=min_quality_score,
            source=source,
            include_inactive=include_inactive,
        )
        cur = self.conn.execute(
            f"UPDATE corpus SET is_active = ? WHERE {where_clause}",
            [1 if is_active else 0, *params],
        )
        self.conn.commit()
        return int(cur.rowcount or 0)

    def export_for_self_organized(
        self,
        out_path="data/processed/corpus.txt",
        category=None,
        min_quality_score=0.0,
        source=None,
        show_inactive=False,
    ):
        """自己組織化SNN学習用に、すべてのテキストを連続したプレーンテキストとして出力"""
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        count = 0
        with open(out_path, 'w', encoding='utf-8') as f:
            query, params = self._build_export_query(
                category=category,
                min_quality_score=min_quality_score,
                source=source,
                show_inactive=show_inactive,
            )
            cur = self.conn.execute(query, params)
            for row in cur.fetchall():
                t_type, content, _category, _quality_score, _source, _source_version, _lang, _is_active = row
                if t_type == "chat":
                    data = json.loads(content)
                    f.write(f"User: {data['prompt']}\nSARA: {data['completion']}\n")
                else:
                    f.write(f"{content}\n")
                count += 1
        return count

    def export_for_distillation(
        self,
        out_path="data/raw/chat_data.jsonl",
        category=None,
        min_quality_score=0.0,
        source=None,
        show_inactive=False,
    ):
        """蒸留学習(BP)用に、プロンプト・コンプリーションのペアを含むJSONLとして出力"""
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        count = 0
        with open(out_path, 'w', encoding='utf-8') as f:
            query, params = self._build_export_query(
                category=category,
                min_quality_score=min_quality_score,
                source=source,
                show_inactive=show_inactive,
            )
            cur = self.conn.execute(query, params)
            for row in cur.fetchall():
                t_type, content, category, quality_score, source, source_version, lang, _is_active = row
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
                    pair["category"] = category
                    pair["quality_score"] = float(quality_score)
                    pair["source"] = source
                    pair["source_version"] = source_version
                    pair["lang"] = lang
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                count += 1
        return count
