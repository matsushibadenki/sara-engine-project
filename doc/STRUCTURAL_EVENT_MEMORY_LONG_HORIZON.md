# Structural Event Memory Long-Horizon Retest

## English

The Phase 18 retest uses 202 observed-only records from two independent domains across frozen horizon indices 0-100. The runner verifies SHA-256 fingerprints for the protocol, manifest, candidate source, and Event Memory cache before evaluation.

All three profiles achieved recall `1.0`, contradiction recovery `1.0`, zero evictions, and event cost `64`. Fixed retention used `max_entries=4`; linear and logarithmic retention used `max_entries=8`. Because the evidence collapsed into only two structural proposals, the larger tiered caches were not stressed. This is a successful boundedness and reproducibility result, but a negative differentiation result. It does not justify production promotion.

Run:

```bash
PYTHONPATH=src python scripts/sara_cli.py eval-structural-event-memory-long-horizon
```

The next experiment must exceed the smallest cache capacity with simultaneously retrievable delayed candidates and hold total state, event, and maintenance budgets equal across profiles.

## 日本語

Phase 18 の再検証では、独立した2ドメインから得た observed-only の202レコードを、凍結した horizon index 0-100 で評価します。実行前に protocol、manifest、candidate source、Event Memory cache の SHA-256 を検証します。

3方式はいずれも recall `1.0`、contradiction recovery `1.0`、eviction 0、event cost `64` でした。fixed は `max_entries=4`、linear と logarithmic は `max_entries=8` です。入力が2件の構造提案へ圧縮されたため、階層 cache の容量差には負荷がかかりませんでした。再現性と boundedness の検証としては成功ですが、方式差を示す結果としては陰性です。本番昇格の根拠にはしません。

次の実験では、同時に検索可能な遅延候補数を最小 cache 容量より多くし、方式間の総 state、event、maintenance budget を等しく固定します。

## 简体中文

Phase 18 复测使用来自两个独立域的 202 条 observed-only 记录，冻结的 horizon index 为 0-100。运行前会验证 protocol、manifest、candidate source 和 Event Memory cache 的 SHA-256。

三种策略均达到 recall `1.0`、contradiction recovery `1.0`、零淘汰和 event cost `64`。fixed 使用 `max_entries=4`，linear 与 logarithmic 使用 `max_entries=8`。由于输入被压缩为两个结构提案，分层缓存没有受到容量压力。因此该结果证明了可复现性与有界性，但未证明策略差异，也不支持生产升级。

下一项实验必须让可同时检索的延迟候选数量超过最小缓存容量，并在各策略之间保持相同的总 state、event 和 maintenance budget。
