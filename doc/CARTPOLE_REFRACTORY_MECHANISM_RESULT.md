# Synthetic Refractory Mechanism Result V1

[Done] The one registered development execution completed under [protocol](CARTPOLE_REFRACTORY_MECHANISM_PROTOCOL.md) SHA-256 `52aeb63b2cc18ef99891cb868ff00a78f7fa2e22db688347c9d12a3e74f94b8d`. The canonical [result](../workspace/evaluation/cartpole_refractory_mechanism_v1.json) is SHA-256 `89b1fe5461b92b01da1cd68ebcb08525d42853ef2c8dd706d6833f02d00ea435`; its persistent [one-shot reservation](../workspace/evaluation/cartpole_refractory_mechanism_v1.json.lock) is SHA-256 `c2d460a1c70bd3dba9cf05b63dff9286eeea7e63c2a75b3cbe48a4efa4ff5adb`. Re-reading the canonical result reproduces the registered gate decision exactly. All 30 seed/arm records are present; no held-out partition exists.

| Arm | Development correct per seed (out of 32) | Aggregate accuracy | Training updates per seed |
| --- | --- | ---: | --- |
| Scalar raw count | 16 / 16 / 16 / 16 / 16 | 0.50 | 33 / 30 / 36 / 30 / 42 |
| Compact explicit gap | 32 / 32 / 32 / 32 / 32 | 1.00 | 1 / 1 / 1 / 1 / 1 |
| Stateful refractory spike count | 32 / 32 / 32 / 32 / 32 | 1.00 | 1 / 1 / 1 / 1 / 1 |
| Refractory disabled | 16 / 16 / 16 / 16 / 16 | 0.50 | 33 / 30 / 36 / 30 / 42 |
| Rate-matched, label-blind mask | 16 / 16 / 16 / 16 / 16 | 0.50 | 32 / 31 / 33 / 31 / 41 |
| Timing removed | 16 / 16 / 16 / 16 / 16 | 0.50 | 33 / 30 / 36 / 30 / 42 |

[Done] Every preregistered development gate check passed. Compact and spiking predictions are identical on every development episode, with exactly 16 one-feature episodes per seed in both the spiking and rate-matched-mask conditions. The matched mask retains the same aggregate feature count but loses the target association. Disabling refractory suppression or erasing the gap also loses the signal. Thus the configured refractory gate *can* encode this deliberately identifiable near/far interval; it is not merely a generic drop in event rate on this task. Explicit non-spiking timing encodes it equally well, so there is **no spike-specific advantage** here.

This is a narrow synthetic positive control with the same core gap rule in training and development; absolute time and nuisance routes differ, but the family is not new. It does not repair the negative [CartPole V2 development result](CARTPOLE_DEVELOPMENT_RESULT_V2.md), establish useful control performance, isolate membrane potential from other spiking implementations, measure energy, or authorize CartPole held-out evaluation. No protocol, threshold, or source is retuned after the result.

日本語: 事前登録した合成課題では、不応期spikingと明示的な時間差を使うcompactが各シード32/32、特徴保持量を合わせた非因果マスクと他の対照が16/32でした。不応期がこの特定の時間差を符号化できることは示しましたが、非スパイク方式と同等で、スパイク固有の優位性はありません。CartPole V2の陰性結果やheld-out の判断は変わりません。

简体中文：在预注册的合成任务中，不应期脉冲和显式时间差的 compact 条件均为每个种子 32/32；特征保留率匹配但与标签无关的掩码及其他对照均为 16/32。这说明不应期能够编码这个特定时间间隔，但非脉冲方法同样有效，没有脉冲特有优势。CartPole V2 的阴性结果和留出集决定不变。
