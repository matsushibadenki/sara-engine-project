import importlib.util
import os


def _load_script():
    module_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "scripts",
        "eval",
        "real_data_external_validity.py",
    )
    spec = importlib.util.spec_from_file_location("real_data_external_validity_script", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_real_data_tasks_prefers_rare_discriminative_terms():
    module = _load_script()
    docs = [
        "common retrieval alpha sparse spikes reduce event cost for memory.",
        "common retrieval beta dense scans spend more event cost for memory.",
        "common retrieval gamma predictive coding stabilizes summaries.",
    ]

    tasks = module.build_real_data_tasks(docs, max_cases=3)

    assert len(tasks) == 3
    assert "alpha" in tasks[0]["query"]
    assert "beta" in tasks[1]["query"]
    assert "gamma" in tasks[2]["query"]


def test_real_data_external_validity_passes_and_reports_ann_cost_advantage(tmp_path):
    module = _load_script()
    corpus = tmp_path / "external_validity_corpus.txt"
    corpus.write_text(
        "\n".join(
            [
                "alpha sparse spikes reduce event cost for retrieval and memory.",
                "beta dense scan baseline spends more work across every document.",
                "gamma predictive coding stabilizes low energy summaries.",
                "delta continual memory keeps useful facts available after updates.",
                "epsilon world model forecasts simple transitions with sparse state.",
                "zeta release gate records accuracy and energy proxy evidence.",
                "eta neuro fluid routing moves attention toward important evidence.",
                "theta nested learning separates fast adaptation from slow memory.",
            ]
        ),
        encoding="utf-8",
    )

    report = module.run_real_data_external_validity(
        corpus_path=str(corpus),
        max_docs=8,
        max_cases=8,
    )

    assert report["passed"] is True
    assert report["metrics"]["real_data_qa_accuracy"] >= 0.80
    assert report["metrics"]["ann_proxy_qa_accuracy"] >= 0.80
    assert report["metrics"]["dense_embedding_ann_proxy_qa_accuracy"] >= 0.80
    assert report["metrics"]["bm25_offline_proxy_qa_accuracy"] >= 0.80
    assert report["metrics"]["ann_cost_advantage_proxy"] >= 2.0
    assert report["metrics"]["dense_embedding_ann_cost_advantage_proxy"] >= 2.0
    assert report["metrics"]["bm25_offline_cost_advantage_proxy"] >= 2.0
    assert report["metrics"]["real_pretrained_embedding_reference_available"] == 0.0
    assert report["metrics"]["real_pretrained_embedding_reference_qa_accuracy"] == 0.0
    assert report["metrics"]["real_pretrained_embedding_reference_cost_advantage_proxy"] == 0.0
    assert report["metrics"]["real_pretrained_embedding_faiss_reference_available"] == 0.0
    assert report["metrics"]["real_pretrained_embedding_faiss_reference_qa_accuracy"] == 0.0
    assert report["metrics"]["real_pretrained_embedding_faiss_reference_cost_advantage_proxy"] == 0.0
    assert report["metrics"]["real_cross_encoder_reference_available"] == 0.0
    assert report["metrics"]["real_cross_encoder_reference_qa_accuracy"] == 0.0
    assert report["metrics"]["real_cross_encoder_reference_cost_advantage_proxy"] == 0.0
    assert report["ann_pretrained_embedding_reference"]["available"] is False
    assert report["ann_pretrained_embedding_reference"]["reason"] == "not_configured"
    assert report["ann_pretrained_embedding_faiss_reference"]["available"] is False
    assert report["ann_pretrained_embedding_faiss_reference"]["reason"] == "not_configured"
    assert report["ann_cross_encoder_reference"]["available"] is False
    assert report["ann_cross_encoder_reference"]["reason"] == "not_configured"
    assert report["bm25_offline_proxy"]["accuracy"] >= 0.80
    assert report["metrics"]["performance_energy_ratio_proxy"] >= 2.0
    assert report["benchmark_context"]["max_docs"] == 8
    assert report["benchmark_context"]["max_cases"] == 8
    assert report["benchmark_context"]["pretrained_embedding_reference_available"] is False
    assert report["benchmark_context"]["pretrained_embedding_reference_reason"] == "not_configured"
    assert report["benchmark_context"]["pretrained_embedding_faiss_reference_available"] is False
    assert report["benchmark_context"]["pretrained_embedding_faiss_reference_reason"] == "not_configured"
    assert report["benchmark_context"]["cross_encoder_reference_available"] is False
    assert report["benchmark_context"]["cross_encoder_reference_reason"] == "not_configured"
    assert len(report["benchmark_context"]["corpus_sha256"]) == 64
    assert report["benchmark_context"]["retriever_strategy"]
    per_task_summary = report["per_task_external_validity_summary"]
    assert per_task_summary["schema"] == "sara-per-task-external-validity-summary-v1"
    assert per_task_summary["case_count"] == 8
    assert per_task_summary["failure_type_counts"]["none"] == 8
    assert per_task_summary["abstention_rate"] == 0.0
    assert per_task_summary["avg_dense_cost_advantage_proxy"] >= 2.0
    first_case = per_task_summary["cases"][0]
    assert set(first_case) == {"case_id", "query", "quality", "cost", "abstention", "failure_type"}
    assert first_case["quality"]["sara_correct"] is True
    assert first_case["cost"]["dense_cost_advantage_proxy"] >= 1.0
    assert first_case["abstention"]["expected_behavior"] == "retrieve"
    assert report["metrics"]["per_task_external_validity_summary_available"] == 1.0
    assert report["metrics"]["per_task_external_validity_case_count"] == 8.0
    assert report["thresholds"]["min_performance_energy_ratio_proxy"] == 2.0
    assert report["check_details"]["performance_energy_ratio_proxy"]["passed"] is True
    assert report["check_details"]["performance_energy_ratio_proxy"]["required_min"] == 2.0
    assert report["check_details"]["dense_embedding_cost_advantage"]["passed"] is True
    assert report["check_details"]["trend.no_regressions"]["required_max"] == 0
    assert report["metrics"]["sara_metabolic_cost_reduction_proxy"] >= 1.0
    assert report["metrics"]["sara_metabolic_early_stop_rate"] > 0.0
    assert report["metrics"]["sara_metabolic_avg_processed_query_tokens"] <= report["metrics"]["sara_metabolic_avg_query_tokens"]
    assert report["metrics"]["negative_control_abstention_integrity"] == 1.0
    assert report["metrics"]["negative_control_ann_overselection_observed"] == 1.0
    assert report["metrics"]["negative_control_cost_advantage_proxy"] >= 2.0
    assert report["check_details"]["negative_control_abstention"]["passed"] is True
    assert report["negative_controls"]["sara_predicted_doc_index"] == -1
    assert report["metrics"]["partial_evidence_abstention_integrity"] == 1.0
    assert report["metrics"]["partial_evidence_ann_overselection_observed"] == 1.0
    assert report["metrics"]["partial_evidence_cost_advantage_proxy"] >= 2.0
    assert report["check_details"]["partial_evidence_abstention"]["passed"] is True
    partial_case = report["negative_controls"]["cases"]["partial_evidence_query"]
    assert partial_case["sara_predicted_doc_index"] == -1
    assert partial_case["sara_retrieval_diagnostics"]["abstained_by_match_ratio"] is True
    assert report["metrics"]["contrastive_control_accuracy"] == 1.0
    assert report["metrics"]["contrastive_control_rare_decider_first_rate"] == 1.0
    assert report["metrics"]["contrastive_control_cost_advantage_proxy"] >= 2.0
    assert report["check_details"]["contrastive_control_accuracy"]["passed"] is True
    contrastive_case = report["contrastive_controls"]["cases"][0]
    assert contrastive_case["sara_correct"] is True
    assert contrastive_case["rare_decider_processed_first"] is True
    assert report["metrics"]["sparse_rag_rerank_bounded_count_observed"] == 1.0
    assert report["metrics"]["sparse_rag_rerank_source_agreement_observed"] >= 0.5
    assert report["metrics"]["sparse_rag_rerank_contradiction_guard_observed"] == 1.0
    assert report["metrics"]["sparse_rag_rerank_freshness_observed"] == 1.0
    assert report["metrics"]["sparse_rag_rerank_citation_grounding_observed"] == 1.0
    assert report["metrics"]["sparse_rag_rerank_source_reliability_observed"] == 1.0
    assert report["metrics"]["sparse_rag_rerank_source_diversity_observed"] == 1.0
    assert report["sparse_rag_rerank"]["observed_only"] is True
    assert report["metrics"]["sparse_diffusion_real_data_partition_integrity"] == 1.0
    assert report["metrics"]["sparse_diffusion_real_data_denoise_accuracy"] == 1.0
    assert report["metrics"]["sparse_diffusion_real_data_event_cost_advantage"] >= 2.0
    assert report["metrics"]["sparse_diffusion_real_data_single_pass_integrity"] == 1.0
    assert report["check_details"]["sparse_diffusion_real_data_denoise_accuracy"]["passed"] is True
    assert report["check_details"]["sparse_diffusion_real_data_event_cost_advantage"]["passed"] is True
    assert report["sparse_diffusion_real_data"]["observed_only"] is True
    assert report["repository_fixture_probe"]["observed_only"] is True
    assert report["repository_fixture_probe"]["passed"] is True
    assert report["repository_fixture_probe"]["case_count"] == 8
    assert report["metrics"]["repository_fixture_retrieval_accuracy"] == 1.0
    assert report["metrics"]["repository_fixture_abstention_integrity"] == 1.0
    assert report["repository_fixture_probe"]["accuracy_by_task_type"]["noisy"] == 1.0
    assert report["repository_fixture_probe"]["accuracy_by_task_type"]["adversarial"] == 1.0
    assert report["repository_fixture_probe"]["accuracy_by_task_type"]["delayed"] == 1.0
    assert report["metrics"]["rag_query_decomposition_bounded_count_observed"] == 1.0
    assert report["metrics"]["rag_query_decomposition_coverage_observed"] == 1.0
    assert report["metrics"]["rag_query_decomposition_nonempty_observed"] == 1.0
    assert report["metrics"]["rag_query_decomposition_subquery_hit_observed"] == 1.0
    assert report["metrics"]["rag_query_decomposition_merged_selection_observed"] == 1.0
    assert report["metrics"]["rag_query_decomposition_merged_citation_grounding_observed"] == 1.0
    assert report["metrics"]["rag_query_decomposition_merged_source_reliability_observed"] == 1.0
    assert report["metrics"]["rag_query_decomposition_merged_source_diversity_observed"] == 1.0
    assert report["rag_query_decomposition"]["observed_only"] is True
    assert all(report["checks"].values())


def test_real_data_external_validity_marks_missing_pretrained_embedding_directory(tmp_path):
    module = _load_script()
    corpus = tmp_path / "external_validity_corpus.txt"
    corpus.write_text(
        "\n".join(
            [
                "alpha sparse spikes reduce event cost for retrieval and memory.",
                "beta dense scan baseline spends more work across every document.",
                "gamma predictive coding stabilizes low energy summaries.",
                "delta continual memory keeps useful facts available after updates.",
            ]
        ),
        encoding="utf-8",
    )

    report = module.run_real_data_external_validity(
        corpus_path=str(corpus),
        max_docs=4,
        max_cases=4,
        pretrained_embedding_model_path=str(tmp_path / "missing-model"),
    )

    assert report["ann_pretrained_embedding_reference"]["available"] is False
    assert report["ann_pretrained_embedding_reference"]["reason"] == "missing_directory"
    assert report["ann_pretrained_embedding_faiss_reference"]["available"] is False
    assert report["ann_pretrained_embedding_faiss_reference"]["reason"] == "missing_directory"
    assert report["ann_cross_encoder_reference"]["available"] is False
    assert report["ann_cross_encoder_reference"]["reason"] == "not_configured"
    assert report["metrics"]["real_pretrained_embedding_reference_available"] == 0.0
    assert report["metrics"]["real_pretrained_embedding_faiss_reference_available"] == 0.0
    assert report["metrics"]["real_cross_encoder_reference_available"] == 0.0


def test_real_data_external_validity_marks_missing_cross_encoder_directory(tmp_path):
    module = _load_script()
    corpus = tmp_path / "external_validity_corpus.txt"
    corpus.write_text(
        "\n".join(
            [
                "alpha sparse spikes reduce event cost for retrieval and memory.",
                "beta dense scan baseline spends more work across every document.",
                "gamma predictive coding stabilizes low energy summaries.",
                "delta continual memory keeps useful facts available after updates.",
            ]
        ),
        encoding="utf-8",
    )

    report = module.run_real_data_external_validity(
        corpus_path=str(corpus),
        max_docs=4,
        max_cases=4,
        cross_encoder_model_path=str(tmp_path / "missing-cross-encoder"),
    )

    assert report["ann_cross_encoder_reference"]["available"] is False
    assert report["ann_cross_encoder_reference"]["reason"] == "missing_directory"
    assert report["metrics"]["real_cross_encoder_reference_available"] == 0.0


def test_metabolic_sparse_retriever_early_stops_on_discriminative_token():
    module = _load_script()
    docs = [
        "alpha sparse spikes reduce event cost for retrieval and memory.",
        "beta dense scan baseline spends more work across every document.",
        "gamma predictive coding stabilizes low energy summaries.",
    ]
    retriever = module.MetabolicSparseEventRetriever(docs)

    predicted_index, event_cost = retriever.search("alpha sparse retrieval memory")

    assert predicted_index == 0
    assert event_cost < module.SparseEventRetriever(docs).search("alpha sparse retrieval memory")[1]
    assert retriever.last_diagnostics["early_stopped"] is True
    assert retriever.last_diagnostics["processed_token_count"] < retriever.last_diagnostics["query_token_count"]


def test_bm25_offline_proxy_matches_rare_keyword_tasks_with_dense_cost():
    module = _load_script()
    docs = [
        "common retrieval alpha sparse spikes reduce event cost for memory.",
        "common retrieval beta dense scans spend more event cost for memory.",
        "common retrieval gamma predictive coding stabilizes summaries.",
    ]
    tasks = module.build_real_data_tasks(docs, max_cases=3)

    bm25 = module._score_retriever(module.BM25OfflineProxyRetriever(docs), tasks, docs)
    sparse = module._score_retriever(module.MetabolicSparseEventRetriever(docs), tasks, docs)

    assert bm25["accuracy"] == 1.0
    assert bm25["avg_event_cost_proxy"] > sparse["avg_event_cost_proxy"]


def test_repository_fixture_probe_covers_noisy_adversarial_and_delayed_cases():
    module = _load_script()

    report = module._score_repository_fixture_cases()

    assert report["passed"] is True
    assert report["metrics"]["repository_fixture_retrieval_accuracy"] == 1.0
    assert report["metrics"]["repository_fixture_abstention_integrity"] == 1.0
    assert report["accuracy_by_task_type"]["noisy"] == 1.0
    assert report["accuracy_by_task_type"]["adversarial"] == 1.0
    assert report["accuracy_by_task_type"]["delayed"] == 1.0
    partial_case = next(case for case in report["cases"] if case["task_type"] == "partial")
    assert partial_case["expected_behavior"] == "abstain"
    assert partial_case["sara_predicted_doc_index"] == -1


def test_per_task_external_validity_summary_separates_failure_types():
    module = _load_script()
    tasks = [
        {"case_id": "ok", "query": "alpha", "expected_doc_index": 0},
        {"case_id": "miss", "query": "beta", "expected_doc_index": 1},
        {"case_id": "abstain", "query": "gamma", "expected_doc_index": 2},
    ]
    sparse_score = {
        "case_results": [
            {
                "case_id": "ok",
                "correct": True,
                "predicted_doc_index": 0,
                "event_cost_proxy": 2,
                "summary_keyword_coverage": 1.0,
            },
            {
                "case_id": "miss",
                "correct": False,
                "predicted_doc_index": 0,
                "event_cost_proxy": 3,
                "summary_keyword_coverage": 0.75,
            },
            {
                "case_id": "abstain",
                "correct": False,
                "predicted_doc_index": -1,
                "event_cost_proxy": 1,
                "summary_keyword_coverage": 0.0,
            },
        ]
    }
    dense_score = {
        "case_results": [
            {"case_id": "ok", "correct": True, "event_cost_proxy": 10},
            {"case_id": "miss", "correct": True, "event_cost_proxy": 12},
            {"case_id": "abstain", "correct": True, "event_cost_proxy": 8},
        ]
    }
    embedding_score = {
        "case_results": [
            {"case_id": "ok", "correct": True, "event_cost_proxy": 64},
            {"case_id": "miss", "correct": True, "event_cost_proxy": 64},
            {"case_id": "abstain", "correct": True, "event_cost_proxy": 64},
        ]
    }

    summary = module.build_per_task_external_validity_summary(
        tasks,
        sparse_score,
        dense_score,
        embedding_score,
    )

    assert summary["failure_type_counts"] == {"abstained": 1, "none": 1, "wrong_document": 1}
    assert summary["abstention_rate"] == 1 / 3
    assert summary["cases"][1]["failure_type"] == "wrong_document"
    assert summary["cases"][2]["abstention"]["sara_abstained"] is True


def test_metabolic_sparse_retriever_abstains_on_partial_evidence():
    module = _load_script()
    docs = [
        "alpha sparse spikes reduce event cost for retrieval and memory.",
        "beta dense scan baseline spends more work across every document.",
        "gamma predictive coding stabilizes low energy summaries.",
    ]
    retriever = module.MetabolicSparseEventRetriever(docs)

    predicted_index, event_cost = retriever.search(
        "retrieval memory sara_absent_probe_token no_matching_memory_event"
    )

    assert predicted_index == -1
    assert event_cost > 0
    assert retriever.last_diagnostics["best_match_ratio"] < retriever.last_diagnostics["min_match_ratio"]
    assert retriever.last_diagnostics["abstained_by_match_ratio"] is True


def test_contrastive_control_uses_rare_decider_before_common_overlap():
    module = _load_script()

    report = module._score_contrastive_controls()

    assert report["accuracy"] == 1.0
    assert report["rare_decider_first_rate"] == 1.0
    assert report["cost_advantage_proxy"] >= 2.0
    assert all(case["sara_correct"] for case in report["cases"])
    assert all(case["rare_decider_processed_first"] for case in report["cases"])


def test_external_validity_trend_detects_energy_ratio_regression():
    module = _load_script()
    context = {
        "corpus_sha256": "same",
        "task_sha256": "same",
        "max_docs": 8,
        "max_cases": 8,
    }
    report = {
        "benchmark_context": context,
        "metrics": {
            "real_data_qa_accuracy": 1.0,
            "real_data_summary_keyword_coverage": 0.80,
            "continual_memory_hit_rate": 1.0,
            "performance_energy_ratio_proxy": 6.0,
            "ann_cost_advantage_proxy": 6.0,
        }
    }
    history = [
        {
            "benchmark_context": context,
            "metrics": {
                "real_data_qa_accuracy": 1.0,
                "real_data_summary_keyword_coverage": 0.80,
                "continual_memory_hit_rate": 1.0,
                "performance_energy_ratio_proxy": 10.0,
                "ann_cost_advantage_proxy": 10.0,
            }
        }
    ]

    trend = module.build_external_validity_trend(report, history, regression_tolerance=0.10)

    assert trend["has_previous"] is True
    assert trend["regression_count"] == 2
    assert {item["metric"] for item in trend["regressions"]} == {
        "performance_energy_ratio_proxy",
        "ann_cost_advantage_proxy",
    }


def test_external_validity_trend_skips_incompatible_context():
    module = _load_script()
    report = {
        "benchmark_context": {
            "corpus_sha256": "new",
            "task_sha256": "new",
            "max_docs": 16,
            "max_cases": 8,
        },
        "metrics": {
            "real_data_qa_accuracy": 1.0,
            "real_data_summary_keyword_coverage": 1.0,
            "continual_memory_hit_rate": 1.0,
            "performance_energy_ratio_proxy": 4.0,
            "ann_cost_advantage_proxy": 4.0,
        },
    }
    history = [
        {
            "benchmark_context": {
                "corpus_sha256": "old",
                "task_sha256": "old",
                "max_docs": 8,
                "max_cases": 8,
            },
            "metrics": {
                "real_data_qa_accuracy": 1.0,
                "real_data_summary_keyword_coverage": 1.0,
                "continual_memory_hit_rate": 1.0,
                "performance_energy_ratio_proxy": 100.0,
                "ann_cost_advantage_proxy": 100.0,
            },
        }
    ]

    trend = module.build_external_validity_trend(report, history, regression_tolerance=0.05)

    assert trend["has_previous"] is True
    assert trend["comparison_active"] is False
    assert trend["regression_count"] == 0
    assert "benchmark_context_changed" in trend["comparison_skipped_reason"]


def test_external_validity_history_roundtrip():
    module = _load_script()
    history_path = os.path.join("workspace", "tests", "external_validity_history_roundtrip.json")
    report = {
        "passed": True,
        "doc_count": 2,
        "task_count": 2,
        "metrics": {"real_data_qa_accuracy": 1.0},
        "benchmark_context": {"corpus_sha256": "abc", "task_sha256": "def"},
        "checks": {"trend.no_regressions": True},
    }

    saved = module.append_external_validity_history(str(history_path), report)
    history = module.load_external_validity_history(saved)

    assert len(history) >= 1
    assert history[-1]["metrics"]["real_data_qa_accuracy"] == 1.0
    assert history[-1]["benchmark_context"]["corpus_sha256"] == "abc"
    assert history[-1]["checks"]["trend.no_regressions"] is True
