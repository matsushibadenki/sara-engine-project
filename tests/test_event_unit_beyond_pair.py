from sara_engine.evaluation.event_unit_beyond_pair import BeyondPairLearner,generate_beyond_pair,run_beyond_pair

def test_beyond_pair_generator_is_balanced_and_deterministic():
 rows=generate_beyond_pair(seeds=(1,),count_per_family=4,split="training")
 assert rows==generate_beyond_pair(seeds=(1,),count_per_family=4,split="training")
 for family in {r.family for r in rows}:
  assert sum(r.label for r in rows if r.family==family)==2

def test_pair_triplet_and_branch_arms_are_bounded_and_replayable():
 train=generate_beyond_pair(seeds=(2,),count_per_family=8,split="training");dev=generate_beyond_pair(seeds=(2,),count_per_family=4,split="development")
 for arm in BeyondPairLearner.ARMS:
  a=run_beyond_pair(arm,train,dev);b=run_beyond_pair(arm,train,dev)
  assert a["prediction_trace_sha256"]==b["prediction_trace_sha256"]
  assert a["maximum_event_work"]<=256 and a["state_bytes"]<=2097152
