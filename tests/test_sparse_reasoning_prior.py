from sara_engine.reasoning.sparse_reasoning_prior import build_sparse_reasoning_prior


def test_reasoning_prior_selects_direction_from_source_backed_evidence():
    result = build_sparse_reasoning_prior(
        {
            "case_id": "upward",
            "target": "load",
            "expected_direction": "up",
            "expected_magnitude": "moderate",
            "evidence": [
                {
                    "source_ref": "fixture://trend",
                    "direction": "up",
                    "magnitude": "moderate",
                    "relevance": 0.9,
                }
            ],
        }
    )

    assert result.predicted_direction == "up"
    assert result.predicted_magnitude == "moderate"
    assert result.logic_to_state_consistent is True
    assert result.abstained is False
    assert result.sparse_prior_signature


def test_reasoning_prior_abstains_when_sudden_shift_lacks_external_event():
    result = build_sparse_reasoning_prior(
        {
            "case_id": "missing-external",
            "sudden_shift": True,
            "expected_abstain": True,
            "evidence": [
                {
                    "source_ref": "fixture://history",
                    "direction": "down",
                    "magnitude": "small",
                    "relevance": 0.8,
                    "external_event": False,
                }
            ],
        }
    )

    assert result.abstained is True
    assert result.abstention_reason == "external_event_missing"
    assert result.selected_route == "request_external_context"
    assert result.logic_to_state_consistent is True


def test_reasoning_prior_rejects_missing_source_and_low_relevance():
    result = build_sparse_reasoning_prior(
        {
            "case_id": "unsupported",
            "expected_abstain": True,
            "evidence": [
                {"direction": "up", "magnitude": "large", "relevance": 1.0},
                {
                    "source_ref": "fixture://weak",
                    "direction": "up",
                    "magnitude": "large",
                    "relevance": 0.2,
                },
            ],
        }
    )

    assert result.abstained is True
    assert result.relevant_evidence_count == 0
    assert result.event_relevance == 0.0
    assert all(not row["relevant"] for row in result.trace)
