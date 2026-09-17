# Temporal Transfer v1: Preregistered Development Task

The frozen [machine-readable protocol](../data/processed/benchmark_fixtures/temporal_transfer_v1.json) has SHA-256 `10d186195b7c2fff6d441f7d8e9ff254e5a8d71198216cea060d91ca7e9789cd`. Generator/evaluator source SHA-256 is `0020c04d27767efac83cd6a5513d5d8fd551ff5afef080085c46b68638c62152`; the existing event-unit implementation remains frozen at `8fbfcc8efb8f19773c789d374b670d0553ecea6acb56aa492d2f8742dd8d76e0`.

This task directly addresses the [capture-pair overlap finding](DISCOVERY_CAPTURE_PAIR_RESULT.md): despite disjoint seeds, all 250 earlier development examples repeated a training labeled pattern. Here the target rule is the time gap between routes 0 and 1. Training uses nuisance routes 20–23 and absolute time origins 1–4; development uses never-trained nuisance routes 24–27 and origins 20–23. Labels are balanced within each seed. A pre-execution gate requires zero identical `(family, input events, label)` patterns, zero overlapping episode IDs, and zero nuisance-route overlap. The fixed split contains 200 training and 100 development episodes; no held-out set is generated.

The B compact-event and C temporal-state arms, plus C time-shuffle and C state-reset controls, use the same local learner. Training uses the existing bounded update. Development predictions are evaluated without passing true development labels to the learner or modifying learned weights. The preregistered diagnostic thresholds are C−B at least +0.15, a positive C−B difference on all five development seeds, C−time-shuffle at least +0.10, and C−state-reset at least +0.10, with exact prediction-trace replay and bounded state/event work. A failed gate is retained as a negative result without retuning.

Pattern disjointness is a necessary but narrow check: the core pair-gap motif remains shared across splits, by design. Passing would demonstrate limited nuisance/time-origin transfer on this synthetic rule, not new task-family generalization, spike-specific benefit, real-event transfer, energy efficiency, or an exploration-policy gain. No arm has been scored under this protocol. The next step is a separate intent-before-outcome capture runner and a pre-execution identity audit; approval/export remains off.

日本語: 前回の全例パターン重複を避けるため、開発側の妨害routeと絶対時刻を学習側から分離しました。判定閾値は評価前に固定済みです。ただし時間差の核となる規則は共通であり、実データや未知の概念への汎化を示す試験ではありません。現時点で評価結果はありません。

简体中文：为避免上次全部开发样本与训练模式重复，本协议将开发集的干扰路由和绝对时间与训练集分开，并在评估前固定阈值。核心时间间隔规则仍然共享，因此这不是现实数据或新概念的泛化测试。目前尚未运行任何评估臂。
