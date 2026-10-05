# Readout-visible temporal information v1

## English

[Done] Frozen research contract `5dd6daf0…6123` implements a streaming input observer, independent of any neuron output. It retains first ordinal, last ordinal and count for each arriving route, with at most 95 events and 24 records. Validation precedes mutation; snapshots are immutable. It does not buffer the full history or receive an outcome.

The observer exposes three fixed readout channels per route: first-position strength, last-position strength and frequency. Maximum vocabulary/output is 72 channels; diagnostic serialization is at most 101 bytes (not Python resident memory or total model state). Each arrival visits one record; snapshot sorting involves at most 24 records and projection emits at most 72 channels. This is explicit shared temporal memory. Any future arm must pay its observation, storage and projection cost; gains cannot be assigned to spikes merely because a spiking backend is attached.

The prior first-cue-order fixtures distinguish all 102 paired packet/readout signatures under this interface. This repairs the specific information-loss boundary, not the consumed electricity learner. A deliberate pair differing only in middle order has identical first/last/count statistics: the interface is lossy and does not preserve all ordering or elapsed-time information. No local learning, real-data evaluation, spike benefit or energy saving follows from this audit.

Audit output: `workspace/evaluation/temporal_information_contract_v1.json`. [Next] Separately preregister local learning with matched observer/readout contracts and an observer-only baseline plus backend-output ablations. Freeze how backend outputs reach the readout and charge all state/event costs before scoring. Consumed electricity attempts stay closed; 2014 remains sealed. Real confirmation and promotion are [Later].

## 日本語

[Done] backendの発火に依存せず、入力routeの最初・最後の出現位置と回数を保持する共通observerを実装・契約凍結しました。95event、24record、72readout channel、wire最大101byteに制限します。wireサイズはPython常駐メモリやモデル全体のサイズではありません。全backendにobserverの更新・保存・投影コストを計上します。

102組すべてで最初の順序情報がreadoutまで残ります。ただし中間順序が異なっても同じ統計になる例を保持し、完全な時間情報保存とは主張しません。これは既存の電力実験の修正・再実行でも、spikeの性能優位でもありません。

[Next] 共通observer／readout、observer単独baseline、backend出力ablationを固定した局所学習実験を別途事前登録します。実学習・実データ閲覧は未実施、2014年は封印維持です。[Later] 新鮮な実データ確認と採用。

## 简体中文

[Done] 实现并冻结与backend发放无关的共享observer，保留输入route的首次、末次位置及次数。限制为95个event、24个record、72个readout channel、wire最多101byte；wire大小不是Python常驻内存或整个模型大小。各backend均须计入observer的更新、存储和投影成本。

全部102对都将最初顺序信息保留到readout。但保留中间顺序不同却统计相同的反例，不声称完整保留时间信息。这不是修改或重跑电力实验，也不证明spike性能优势。

[Next] 另行预注册局部学习实验，固定共享observer／readout、observer独立baseline及backend输出ablation。未执行学习或读取真实数据，2014年继续封存。[Later] 新鲜真实数据确认与采用。
