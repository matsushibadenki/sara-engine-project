from collections import Counter
from sara_engine.evaluation.sepsis_data import load_traces, split_name

def test_sepsis_split_is_case_complete_and_fixed():
    traces = load_traces(); counts = Counter(split_name(trace) for trace in traces)
    assert counts == {"training": 735, "development": 157, "frozen_test": 158}
    assert len({trace.case_id for trace in traces}) == len(traces) == 1050
