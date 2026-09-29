# Structural Event Memory Long-Horizon Retest

## English

The Phase 18 retest uses 202 observed-only records from two independent domains across frozen horizon indices 0-100. The runner verifies SHA-256 fingerprints for the protocol, manifest, candidate source, and Event Memory cache before evaluation.

All three profiles achieved recall `1.0`, contradiction recovery `1.0`, zero evictions, and event cost `64`. Fixed retention used `max_entries=4`; linear and logarithmic retention used `max_entries=8`. Because the evidence collapsed into only two structural proposals, the larger tiered caches were not stressed. This is a successful boundedness and reproducibility result, but a negative differentiation result. It does not justify production promotion.

Run:

```bash
PYTHONPATH=src python scripts/sara_cli.py eval-structural-event-memory-long-horizon
```

The next experiment must exceed the smallest cache capacity with simultaneously retrievable delayed candidates and hold total state, event, and maintenance budgets equal across profiles.

That capacity-pressure experiment is now complete. With nine candidates and four slots per profile, every arm retained `4/9` candidates at event cost `221`. Fixed covered two age bands, linear covered two, and logarithmic covered early, middle, and recent bands (`1/1/2`). This demonstrates temporal redistribution, not greater aggregate capacity. Promotion remains blocked pending evaluator-isolated order and relevance controls.

The evaluator-isolated follow-up used 12 hashed orderings and three hidden relevance profiles. Logarithmic retention kept all age bands but produced delayed-heavy utility `0.4000`, below fixed at `0.4167`, and won only `3/12` orderings. The frozen promotion gate failed. Further hierarchy work on this fixture family is stopped; reopening requires independently observed task relevance and outcome utility.

## 日本語

Phase 18 の再検証では、独立した2ドメインから得た observed-only の202レコードを、凍結した horizon index 0-100 で評価します。実行前に protocol、manifest、candidate source、Event Memory cache の SHA-256 を検証します。

3方式はいずれも recall `1.0`、contradiction recovery `1.0`、eviction 0、event cost `64` でした。fixed は `max_entries=4`、linear と logarithmic は `max_entries=8` です。入力が2件の構造提案へ圧縮されたため、階層 cache の容量差には負荷がかかりませんでした。再現性と boundedness の検証としては成功ですが、方式差を示す結果としては陰性です。本番昇格の根拠にはしません。

次の実験では、同時に検索可能な遅延候補数を最小 cache 容量より多くし、方式間の総 state、event、maintenance budget を等しく固定します。

この capacity-pressure 実験は完了しました。9候補を各方式4枠で評価すると、全方式が event cost `221` で `4/9` を保持しました。fixed と linear は2つの時間帯、logarithmic は early / middle / recent の3帯（`1/1/2`）を保持しました。これは総容量の改善ではなく、時間的な再配分を示します。候補順序と relevance を evaluator 側で分離した反復検証まで昇格は保留します。

evaluator 分離後の追試では、12個のハッシュ固定順序と3種類の非公開 relevance 条件を使いました。logarithmic は全時間帯を保持しましたが、delayed-heavy utility は `0.4000` で fixed の `0.4167` を下回り、勝利は `3/12` でした。凍結した昇格 gate は不合格です。この fixture 系での階層化拡張は停止し、実タスク由来の relevance と outcome utility が得られた場合だけ再開します。

## 简体中文

Phase 18 复测使用来自两个独立域的 202 条 observed-only 记录，冻结的 horizon index 为 0-100。运行前会验证 protocol、manifest、candidate source 和 Event Memory cache 的 SHA-256。

三种策略均达到 recall `1.0`、contradiction recovery `1.0`、零淘汰和 event cost `64`。fixed 使用 `max_entries=4`，linear 与 logarithmic 使用 `max_entries=8`。由于输入被压缩为两个结构提案，分层缓存没有受到容量压力。因此该结果证明了可复现性与有界性，但未证明策略差异，也不支持生产升级。

下一项实验必须让可同时检索的延迟候选数量超过最小缓存容量，并在各策略之间保持相同的总 state、event 和 maintenance budget。

该 capacity-pressure 实验现已完成。九个候选共享每种策略四个槽位时，所有策略都以 event cost `221` 保留 `4/9`。fixed 和 linear 覆盖两个时间带，logarithmic 覆盖 early、middle、recent 三个时间带（`1/1/2`）。这说明的是时间分配差异，而不是总容量提升。在完成候选顺序与 relevance 的 evaluator 隔离复测之前，不进行生产升级。

evaluator 隔离复测使用了 12 个哈希冻结顺序和三种隐藏 relevance 条件。logarithmic 虽覆盖全部时间带，但 delayed-heavy utility 为 `0.4000`，低于 fixed 的 `0.4167`，且仅在 `3/12` 个顺序中胜出。冻结的升级门槛未通过。该 fixture 系列不再扩展层级策略，除非获得独立观测的真实任务 relevance 与 outcome utility。
