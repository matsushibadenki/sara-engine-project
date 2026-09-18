# CartPole Terminal-Feedback Development Result V2

[Done] The one authorized V2 development run completed under [preregistration](CARTPOLE_DEVELOPMENT_PROTOCOL_V2.md) SHA-256 `1da4356e7eb88627a1fe62390f2a7bfdb3f06047506693043fcf03b6b67132bd`. The canonical [result](../workspace/evaluation/cartpole_terminal_development_v2.json) is SHA-256 `866c7c35a1bb39d61818fe779b636f44353877a215848f6f02f7412ce82ef63e`; its persistent [one-shot reservation](../workspace/evaluation/cartpole_terminal_development_v2.json.lock) is SHA-256 `ee2aa1e456b92e0edccb9b0b079ce18aa55d99854a3987fb4fb564f53f22a790`. Re-reading the saved result reproduces the registered decision exactly. There are five runs per condition, 128 training and 32 development episodes per run, with no development updates. Neither V1 nor V2 held-out seeds were opened.

| Condition | Train mean steps | Development mean steps | Development 500-step fraction | Peak policy bytes | Agent CPU µs/step | Internal work/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A scalar local | 54.20 | 70.56875 | 0.03125 | 8,553 | 78.993 | 4.235 |
| B compact event | 54.20 | 70.56875 | 0.03125 | 6,662 | 34.985 | 4.235 |
| C stateful spiking | 19.94375 | 16.08125 | 0 | 21,159 | 136.615 | 25.597 |
| C spike bypass | 54.20 | 70.56875 | 0.03125 | 6,665 | 35.495 | 4.235 |
| C state reset each step | 54.20 | 70.56875 | 0.03125 | 21,183 | 140.715 | 44.235 |
| A no learning | 22.26719 | 21.15625 | 0 | 5,361 | 25.579 | 4.000 |

CPU and event-work ratios use **all 800 episodes per condition**, normalized by the actual decision count; peak policy bytes cover training and development. CPU includes Python state-accounting overhead and excludes environment physics. These are not physical-energy measurements, and shorter spiking episodes make per-episode CPU misleading. Compact, spike-bypass, and state-reset conditions have exactly the same action trace and final weights as scalar on every run; the compact backend changes representation/cost, not behavior. Total recorded spikes across all 800 episodes are 25,892 from 61,348 input events (42.2%) for intact C and 183,916 from 183,916 input events (100%) for per-step-reset C. This is consistent with feature suppression in the persistent pathway, but the record does not separate membrane from refractory effects. No-learning has zero terminal updates; all development conditions have zero updates.

| Run | A/B/bypass/reset development mean | C development mean | No-learning development mean |
| --- | ---: | ---: | ---: |
| 0 | 278.938 | 16.062 | 20.125 |
| 1 | 10.750 | 15.500 | 20.031 |
| 2 | 38.344 | 21.094 | 20.969 |
| 3 | 13.188 | 13.312 | 23.406 |
| 4 | 11.625 | 14.438 | 21.250 |

[Done] The frozen conjunctive gate **failed**. C mean `16.08125` missed the required `100`; C−B and C−both spike controls are `−54.4875` rather than the required `+20/+20/+10`; C−no-learning is `−5.075` rather than `+20`. C beat B in only 3/5 runs, short of 4/5. Exact A/B parity, zero development updates, and all resource ceilings passed. The high B aggregate is dominated by run 0; four other run means are at most 38.344. Do not claim stable task mastery, spiking benefit, or energy efficiency. The specific persistent spiking state/refractory path is harmful here, while resetting it or bypassing spikes recovers route-level behavior. This does **not** isolate membrane potential from refractory dynamics or show that SNNs in general are harmful. No threshold, policy, split, or source may be tuned under V2; its held-out partition stays sealed. Any new hypothesis needs a separately preregistered task and fresh development identities.

日本語: 事前登録済みV2の開発ゲートは陰性です。平均ステップ数はcompactの70.56875に対し通常spikingが16.08125で、spike-bypassと毎ステップ状態リセットはcompactと完全に同じ行動・最終重みになりました。ただしcompactの平均も実行0に強く依存し、安定した制御成功とは言えません。CPUは物理的な消費電力ではありません。条件を調整せず、held-out は開きません。

简体中文：预注册的 V2 开发门槛未通过。compact 平均为 70.56875 步，正常脉冲状态仅为 16.08125 步；绕过脉冲或每步重置状态时，动作和最终权重与 compact 完全一致。但 compact 的均值高度依赖运行 0，不能称为稳定掌握控制任务。CPU 时间不是实际能耗。不会调整 V2 条件，也不会打开留出集。
