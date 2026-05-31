import importlib.util
import json
import os
import tempfile


def _load_manage_db_module():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "utils", "manage_db.py")
    )
    spec = importlib.util.spec_from_file_location("manage_db_script", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manage_db_imports_metadata_and_skips_inactive_exports():
    module = _load_manage_db_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "corpus.db")
        jsonl_path = os.path.join(tmpdir, "materials.jsonl")
        corpus_path = os.path.join(tmpdir, "corpus.txt")
        chat_path = os.path.join(tmpdir, "chat_data.jsonl")

        with open(jsonl_path, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "text": "Active research note for SNN memory.",
                        "category": "research",
                        "quality_score": 0.9,
                        "source_version": "v1",
                        "lang": "en",
                        "is_active": True,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.write(
                json.dumps(
                    {
                        "text": "Inactive note should not be exported.",
                        "category": "research",
                        "quality_score": 0.1,
                        "source_version": "old",
                        "lang": "en",
                        "is_active": False,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.write(
                json.dumps(
                    {
                        "prompt": "Hello",
                        "response": "Hi there",
                        "category": "dialogue",
                        "quality_score": 0.8,
                        "source_version": "chat-v1",
                        "lang": "en",
                        "is_active": True,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        db = module.SaraCorpusDB(db_path)
        added = db.import_file(jsonl_path)

        assert added == 3

        summary = db.get_material_summary()
        assert summary["total_count"] == 3
        assert summary["active_count"] == 2
        assert summary["inactive_count"] == 1
        assert ("research", 2) in summary["categories"]
        assert ("dialogue", 1) in summary["categories"]

        exported_corpus = db.export_for_self_organized(corpus_path)
        exported_chat = db.export_for_distillation(chat_path)

        assert exported_corpus == 2
        assert exported_chat == 2

        with open(corpus_path, "r", encoding="utf-8") as handle:
            corpus_text = handle.read()
        assert "Active research note for SNN memory." in corpus_text
        assert "Inactive note should not be exported." not in corpus_text

        with open(chat_path, "r", encoding="utf-8") as handle:
            chat_lines = [json.loads(line) for line in handle if line.strip()]
        assert len(chat_lines) == 2
        assert any(item.get("category") == "research" for item in chat_lines)
        assert any(item.get("prompt") == "Hello" or item.get("completion") == "Hi there" for item in chat_lines)


def test_manage_db_exports_only_filtered_category_and_quality():
    module = _load_manage_db_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "corpus.db")
        corpus_path = os.path.join(tmpdir, "corpus.txt")
        chat_path = os.path.join(tmpdir, "chat_data.jsonl")

        db = module.SaraCorpusDB(db_path)
        db.add_texts(
            ["High quality research note."],
            text_type="document",
            category="research",
            quality_score=0.95,
        )
        db.add_texts(
            ["Lower quality research note."],
            text_type="document",
            category="research",
            quality_score=0.4,
        )
        db.add_texts(
            [json.dumps({"prompt": "Hello", "completion": "Hi"}, ensure_ascii=False)],
            text_type="chat",
            category="dialogue",
            quality_score=0.9,
        )

        exported_corpus = db.export_for_self_organized(
            corpus_path,
            category="research",
            min_quality_score=0.8,
        )
        exported_chat = db.export_for_distillation(
            chat_path,
            category="research",
            min_quality_score=0.8,
        )

        assert exported_corpus == 1
        assert exported_chat == 1

        with open(corpus_path, "r", encoding="utf-8") as handle:
            corpus_text = handle.read()
        assert "High quality research note." in corpus_text
        assert "Lower quality research note." not in corpus_text
        assert "Hello" not in corpus_text

        with open(chat_path, "r", encoding="utf-8") as handle:
            chat_lines = [json.loads(line) for line in handle if line.strip()]
        assert len(chat_lines) == 1
        assert chat_lines[0]["category"] == "research"
        assert chat_lines[0]["quality_score"] == 0.95


def test_manage_db_lists_and_summarizes_filtered_materials():
    module = _load_manage_db_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        db = module.SaraCorpusDB(os.path.join(tmpdir, "corpus.db"))
        db.add_texts(
            ["Research note about STDP."],
            text_type="document",
            category="research",
            quality_score=0.9,
            source="paper_notes",
            source_version="v2",
            lang="en",
        )
        db.add_texts(
            [json.dumps({"prompt": "Hi", "completion": "Hello"}, ensure_ascii=False)],
            text_type="chat",
            category="dialogue",
            quality_score=0.95,
            source="chat_seed",
            source_version="v1",
            lang="en",
        )

        materials = db.list_materials(category="research", min_quality_score=0.8, limit=5)
        plan = db.summarize_export_plan(category="research", min_quality_score=0.8)

        assert len(materials) == 1
        assert materials[0]["category"] == "research"
        assert materials[0]["source"] == "paper_notes"
        assert "STDP" in materials[0]["preview"]

        assert plan["total_count"] == 1
        assert plan["items"][0]["text_type"] == "document"
        assert plan["items"][0]["category"] == "research"


def test_manage_db_can_include_inactive_and_filter_by_source():
    module = _load_manage_db_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        db = module.SaraCorpusDB(os.path.join(tmpdir, "corpus.db"))
        db.add_texts(
            ["Active research note."],
            text_type="document",
            category="research",
            quality_score=0.9,
            source="paper_notes",
            is_active=True,
        )
        db.add_texts(
            ["Inactive archived note."],
            text_type="document",
            category="research",
            quality_score=0.85,
            source="paper_notes",
            is_active=False,
        )
        db.add_texts(
            ["Different source note."],
            text_type="document",
            category="research",
            quality_score=0.95,
            source="other_notes",
            is_active=True,
        )

        active_only = db.list_materials(category="research", source="paper_notes", limit=10)
        include_inactive = db.list_materials(
            category="research",
            source="paper_notes",
            show_inactive=True,
            limit=10,
        )
        plan = db.summarize_export_plan(
            category="research",
            source="paper_notes",
            show_inactive=True,
        )

        assert len(active_only) == 1
        assert active_only[0]["preview"] == "Active research note."
        assert len(include_inactive) == 2
        assert {item["is_active"] for item in include_inactive} == {True, False}
        assert plan["total_count"] == 2
        assert plan["items"][0]["category"] == "research"


def test_manage_db_can_activate_and_deactivate_by_filter():
    module = _load_manage_db_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        db = module.SaraCorpusDB(os.path.join(tmpdir, "corpus.db"))
        db.add_texts(
            ["Active paper note."],
            text_type="document",
            category="research",
            quality_score=0.9,
            source="paper_notes",
            is_active=True,
        )
        db.add_texts(
            ["Inactive paper note."],
            text_type="document",
            category="research",
            quality_score=0.85,
            source="paper_notes",
            is_active=False,
        )
        db.add_texts(
            ["Other source note."],
            text_type="document",
            category="research",
            quality_score=0.95,
            source="other_notes",
            is_active=True,
        )

        deactivated = db.set_material_active_state(
            False,
            category="research",
            source="paper_notes",
            min_quality_score=0.8,
        )
        after_deactivate = db.list_materials(
            category="research",
            source="paper_notes",
            show_inactive=True,
            limit=10,
        )
        reactivated = db.set_material_active_state(
            True,
            category="research",
            source="paper_notes",
            min_quality_score=0.8,
        )
        after_reactivate = db.list_materials(
            category="research",
            source="paper_notes",
            show_inactive=True,
            limit=10,
        )

        assert deactivated == 2
        assert all(item["is_active"] is False for item in after_deactivate)
        assert reactivated == 2
        assert all(item["is_active"] is True for item in after_reactivate)


def test_manage_db_builds_review_summary():
    module = _load_manage_db_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        db = module.SaraCorpusDB(os.path.join(tmpdir, "corpus.db"))
        db.add_texts(
            ["Research note A."],
            text_type="document",
            category="research",
            quality_score=0.9,
            source="paper_notes",
            lang="en",
            is_active=True,
        )
        db.add_texts(
            ["Research note B."],
            text_type="document",
            category="research",
            quality_score=0.7,
            source="paper_notes",
            lang="ja",
            is_active=False,
        )
        db.add_texts(
            [json.dumps({"prompt": "Hi", "completion": "Hello"}, ensure_ascii=False)],
            text_type="chat",
            category="dialogue",
            quality_score=0.95,
            source="chat_seed",
            lang="en",
            is_active=True,
        )

        report = db.get_review_summary()

        assert report["by_category"][0]["key"] in {"research", "dialogue"}
        assert any(item["key"] == "paper_notes" and item["count"] == 2 for item in report["by_source"])
        assert any(item["key"] == "en" and item["count"] == 2 for item in report["by_lang"])
        assert any(item["key"] == "inactive" and item["count"] == 1 for item in report["by_status"])
