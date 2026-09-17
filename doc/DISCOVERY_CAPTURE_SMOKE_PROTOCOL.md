# Discovery Capture Integration Smoke Protocol v1

This is a development-only instrumentation check, not a new event-unit performance claim or a policy-comparison experiment. It must not read BPI/Sepsis checkpoints, any held-out event-unit family, or existing frozen result files.

- Fixed generator: `generate_episodes` in `event_unit_causal_isolation.py`, namespace `capture-smoke-v1`, seed `92017`, five existing families, four training and two development episodes per family. No held-out episodes are generated.
- Fixed evaluator: `run_v2_development_arm` with the existing `B_compact_event` arm and no intervention or outcome shuffle. Its development accuracy is the captured score; the prediction-trace digest is reported only as a replay check.
- Fixed action: one root intent followed by one child intent. The intent binds the candidate implementation file hash, this protocol file hash, evaluator ID, hypothesis ID, and fixed manual policy ID before the evaluator executes.
- Costs: CPU milliseconds are the ceiling of measured process CPU nanoseconds, input `event_count` is the exact number of generated `UnitEvent` instances across training and development, and `state_bytes` is the evaluator's retained learner-state measurement. Input-event count is not total internal event work and CPU time is not energy.
- Stop: if intent append fails, do not evaluate; if evaluation or outcome append fails, leave the intent pending. Do not synthesize a failure or missing score. Do not export or approve the resulting snapshot automatically.
- Output: a new managed `workspace/` capture JSONL path only. Refuse to reuse an existing capture path. No existing report or model artifact is overwritten.

Passing means that the evaluator observed a persisted pending intent, a matching outcome was appended, replayed capture is intact, and the reported score/cost fields agree with the evaluator and generated inputs. This says nothing about a spike advantage, prospective exploration-policy gain, or production readiness.

日本語: このスモーク実験は「選択を記録してから評価する」計測経路の確認だけです。凍結済み評価の再採点や性能昇格には使いません。

简体中文：此冒烟实验只验证“先记录选择，再执行评估”的采集路径，不用于重评冻结测试或提升性能声明。
