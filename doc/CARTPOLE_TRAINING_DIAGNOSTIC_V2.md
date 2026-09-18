# CartPole V2 Training-Only Mechanism Diagnostic

[Done] This is a post hoc, read-only diagnosis of the frozen negative [V2 result](CARTPOLE_DEVELOPMENT_RESULT_V2.md), **not** a new candidate evaluation or a revision of its gate. The [diagnostic code](../src/sara_engine/evaluation/cartpole_training_diagnostics.py), SHA-256 `f7bb864d90f00dc839e95a5b336ed3ec96e26e591aeb92ce7aa6b913a2fd857c`, accepts only result SHA-256 `866c7c35a1bb39d61818fe779b636f44353877a215848f6f02f7412ce82ef63e`, checks the negative decision and fixed protocol identity, and returns **training records only**. It neither constructs CartPole nor reads held-out data. Its statistics use the four fixed consecutive 32-episode blocks in each 128-episode training run.

| Run | Compact training block means (steps) | Compact episodes >250 | Spiking training block means (steps) | Spiking spikes/input |
| --- | --- | ---: | --- | ---: |
| 0 | 19.47 / 174.16 / 252.91 / 233.88 | 38 | 17.66 / 22.09 / 19.88 / 19.53 | 0.4205 |
| 1 | 33.06 / 21.25 / 24.22 / 21.31 | 0 | 19.56 / 20.28 / 17.22 / 24.06 | 0.4206 |
| 2 | 25.81 / 24.62 / 19.88 / 19.47 | 0 | 19.88 / 20.66 / 21.09 / 19.47 | 0.4201 |
| 3 | 39.06 / 22.97 / 25.88 / 28.31 | 1 | 21.31 / 17.44 / 18.72 / 15.84 | 0.4207 |
| 4 | 23.59 / 33.72 / 18.41 / 22.03 | 0 | 18.38 / 22.53 / 21.44 / 21.84 | 0.4220 |

Compact's large gain is confined to run 0, especially blocks 2–4. Of 640 compact training episodes, 601 have fewer than 250 steps and 39 have more than 250; 38 of those 39 long episodes occur in run 0. All 640 intact-spiking training episodes have fewer than 250 steps. Because the frozen terminal update is `0.05 × (steps/500 − 0.5) × eligibility`, it gives a negative update for every spiking training episode and for 601/640 compact episodes. This is an accounting observation, not proof that the reference value caused the failure; no alternative reward reference was scored.

The configured neuron receives current `1.6` on an active route. At that current, a non-refractory input crosses the spike threshold in one step; after a spike, membrane potential resets to zero, and the next two steps suppress input. An exhaustive synthetic probe of all Boolean input/no-input patterns through length eight matches an exact two-step refractory mask on every pattern, with membrane potential always zero. Thus this particular C pathway is not using subthreshold membrane accumulation for the registered task. Its roughly 42% training spike/input ratio measures heavy route-feature suppression; the per-step-reset control restores one spike per input and exactly restores the compact action and weight traces. This strongly narrows the mechanism to refractory feature masking in this configuration, but it does not establish a general verdict on spiking neurons or show how a different local credit rule would behave.

[Next] If CartPole work resumes, first preregister a new small synthetic mechanism task that independently manipulates refractory duration and route-feature retention under matched exploration, without changing the frozen V2 result. A real-environment follow-up requires fresh development identities and a separate gate. Do not retune the V2 reward reference or use either sealed held-out split to explain this failure.

日本語: 保存済みV2の学習区間だけを事後分析しました。compactの改善はほぼ実行0に集中し、通常spikingは全640学習エピソードが250ステップ未満でした。現在の入力電流ではニューロンは膜電位を蓄積せず、2ステップの不応期による特徴マスクとして振る舞います。この診断を新しい候補評価や一般的なSNNの結論には使いません。held-out は未開封です。

简体中文：仅对已保存的 V2 训练部分进行事后分析。compact 的提升几乎集中在运行 0；正常脉冲臂的全部 640 个训练回合都少于 250 步。在当前输入电流下，神经元不积累膜电位，而表现为持续两步的不应期特征掩码。这不是新的候选评估，也不能推论所有 SNN。留出集仍然封存。
