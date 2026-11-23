# KSEA development timestamp preflight V1

## Registered scope

Process only six caller-supplied local binary streams for KSEA / WBAN 24233, January through June 2024, in month order. This finite protocol operates on synthetic records for verification. Any real acquisition requires separate registration with an independently fixed scope and budget. July is excluded and August remains sealed.

Read at most 262,086 bounded line calls (182 × 1,440 minutes plus one EOF or failing read per month). Each call is capped at 102 bytes, for an aggregate ceiling of 26,732,772 returned bytes. Interpret only the 29-byte station and timestamp header of each fixed-width layout. Require matching station, local month, valid local calendar minute, consistent UTC hour/minute (PST +8h year-round), and strictly increasing timestamps. Abort on structural or timestamp failure. The payload may be transported in memory by the bounded reader and is opaque. No network, output writer, learner, scoring or model calls.

Count unique indexed minutes, absent minutes and dates with any indexed row. The number of dates with a row is an upper bound on valid observation dates. Fewer than 180 indexed dates excludes this fixed development split under that necessary condition. At or above 180, return only that the split cannot be excluded by this bound. Even a fully indexed day may contain invalid sensor values. Neither status qualifies the independent task.

Verify leap day, four calendar/split boundaries (start, January rollover, last development minute, first excluded month), month and station mismatches, UTC wrap, missing rows, duplicate/reversed timestamps, invalid calendar/time fields, overlength lines, and opaque payloads. Exercise 179 and 180 indexed days. Existing Kitchen ON conditions, including the 70,212 eligible arrivals and original development/test separation, remain specific to their original contract. This separate weather hypothesis preserves all previous negative results and seals. Repeated current states, opposing histories, both classes, support/effect, and matched actual local updates remain prerequisites for a comparison.


## Verification and next gate

🟢 [Done] The preregistered timestamp bound is implemented and verified with synthetic inputs. 15 preflight tests and 10 roadmap tests pass. No real observation files were obtained or read.

🟢 [Done] The [delayed outcome follow-up](R2_ASOS_KSEA_DELAYED_CONTRACT.md) fixes the development acquisition ceiling and the normalized outcome contract. Real coverage and support remain pending.

日本語　KSEA開発期間の日時上限を合成データで検証した。真の観測範囲と両クラスの支持は未確認で、8月のテスト封印を維持する。

简体中文　使用合成数据验证了KSEA开发期的时间戳上界。实际观测覆盖与双类别支持尚待核实，8月测试期继续封存。
