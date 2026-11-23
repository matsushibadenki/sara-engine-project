"""Server-rendered views for the local Autobot dashboard."""

import json
from html import escape

PAGES = {"/": 0, "/queue": 1, "/alerts": 2, "/diagnostics": 3}

_LABELS = {
    "en": (
        ("Overview", "Queue", "Alerts", "Diagnostics"),
        ("Process", "Last metrics update", "Pending", "New samples", "Failed items",
         "Dead letters", "Recent critical alerts", "Window (minutes)", "Control actions",
         "Running", "Stopped", "Unavailable", "Refresh", "Language", "Refresh failed", "No metrics received", "Metrics could not be read"),
    ),
    "ja": (
        ("概要", "キュー", "アラート", "診断"),
        ("プロセス", "メトリクス更新時刻", "処理待ち", "新規サンプル", "失敗件数",
         "デッドレター", "直近の重大アラート", "集計期間（分）", "制御アクション",
         "稼働中", "停止中", "取得不可", "更新", "言語", "更新に失敗しました", "メトリクス未取得", "メトリクスを読み取れません"),
    ),
    "zh-CN": (
        ("概览", "队列", "告警", "诊断"),
        ("进程", "指标更新时间", "待处理", "新增样本", "失败数",
         "死信", "近期严重告警", "统计窗口（分钟）", "控制操作",
         "运行中", "已停止", "不可用", "刷新", "语言", "刷新失败", "尚无指标", "无法读取指标"),
    ),
}
_LANG_NAMES = {"en": "English", "ja": "日本語", "zh-CN": "简体中文"}

_STYLE = """
:root { color-scheme: dark; font-family: system-ui, sans-serif; background: #10151c; color: #eff5fa; }
* { box-sizing: border-box; }
body { margin: 0; }
.shell { max-width: 1040px; margin: auto; padding: 24px max(16px, env(safe-area-inset-right)) 48px max(16px, env(safe-area-inset-left)); }
header, .tools { display: flex; align-items: center; flex-wrap: wrap; gap: 12px; }
header { justify-content: space-between; }
.brand { font-weight: 720; letter-spacing: .035em; }
nav { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 20px; border-bottom: 1px solid #34424f; }
a { color: #c5d1dc; text-decoration: none; display: inline-block; padding: 9px 10px; border-radius: 6px; }
a[aria-current] { background: #233a42; color: #aff1df; }
a:hover, button:hover { background: #30404d; }
a:focus-visible, button:focus-visible { outline: 2px solid #aff1df; outline-offset: 2px; }
.tools { gap: 2px; }
button { font: inherit; cursor: pointer; color: inherit; background: #22313c; border: 1px solid #435767; padding: 8px 12px; border-radius: 7px; }
form { margin-bottom: 14px; }
h1 { font-size: clamp(1.65rem, 4vw, 2.4rem); margin: 26px 0 16px; }
dl { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 210px), 1fr)); gap: 12px; margin: 0; }
dl > div, pre { min-width: 0; border: 1px solid #34424f; border-radius: 10px; background: #18232e; padding: 16px; }
dt { color: #bac8d4; }
dd { margin: 6px 0 0; font-size: 1.2rem; font-weight: 650; overflow-wrap: anywhere; }
.ok { color: #9aebcd; }.warn { color: #ffd085; }.bad { color: #ffaaa5; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; }
#error { color: #ffaaa5; }
"""


def render_page(path, lang, status):
    page = PAGES[path]
    lang = lang if lang in _LABELS else "en"
    names, labels = _LABELS[lang]
    process = status.get("process", {})
    metrics_source = status.get("metrics", {})
    metrics = metrics_source.get("data") or {}
    metrics_error = metrics_source.get("error")
    logs = status.get("logs", {})
    running = process.get("running")

    def card(index, value, tone=""):
        return (f'<div><dt>{labels[index]}</dt><dd class="{tone}">'
                f'{escape(str(labels[11] if value is None else value))}</dd></div>')

    def count(index, value, highlight=False):
        if type(value) is not int or value < 0:
            value = None
        tone = "warn" if highlight and value else ""
        return card(index, value, tone)

    if page == 0:
        process_label = labels[9] if running else labels[10] if running is False else None
        cards = card(0, process_label, "ok" if running else "bad" if running is False else "")
        cards += card(1, metrics.get("ts"))
    elif page == 1:
        actions = metrics.get("control_actions")
        cards = count(2, metrics.get("queue_pending")) + count(3, metrics.get("new_samples"))
        cards += count(4, metrics.get("failed_item_count"), highlight=True)
        if isinstance(actions, list):
            actions = ", ".join(map(str, actions)) if actions else "—"
        cards += card(8, actions)
    elif page == 2:
        dead_letters = logs.get("dead_letter_lines") if logs.get("dead_letters_exists") else None
        critical = logs.get("critical_alerts_recent") if logs.get("alerts_exists") else None
        cards = count(5, dead_letters, highlight=True)
        cards += count(6, critical, highlight=True)
        cards += card(7, logs.get("critical_alert_window_minutes"))

    if page == 3:
        content = f'<pre>{escape(json.dumps(status, ensure_ascii=False, indent=2))}</pre>'
    else:
        notice = ""
        if page in (0, 1) and metrics_error == "not_found":
            notice = f'<p>{labels[15]}</p>'
        elif page in (0, 1) and metrics_error:
            notice = f'<p class="bad">{labels[16]}</p>'
        content = f"{notice}<dl>{cards}</dl>"

    navigation = "".join(
        f'<a href="{route}?lang={lang}" {"aria-current=page" if route == path else ""}>'
        f'{names[index]}</a>' for route, index in PAGES.items()
    )
    languages = "".join(
        f'<a href="{path}?lang={code}" lang="{code}" {"aria-current=true" if code == lang else ""}>'
        f'{name}</a>' for code, name in _LANG_NAMES.items()
    )

    return f"""<!doctype html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{names[page]} · SARA Autobot</title>
<link rel="icon" href="data:,">
<style>{_STYLE}</style>
</head>
<body>
<div class="shell">
<header>
  <span class="brand">SARA Autobot</span>
  <div class="tools" role="group" aria-label="{labels[13]}">{languages}</div>
</header>
<nav>{navigation}</nav>
<main>
  <h1>{names[page]}</h1>
  <form><input type="hidden" name="lang" value="{lang}">
    <button type="submit">{labels[12]}</button></form>
  <p id="error" role="status" hidden>{labels[14]}</p>
  <div id="panel">{content}</div>
</main>
</div>
<script>
const panel = document.querySelector('#panel');
const error = document.querySelector('#error');
async function refresh() {{
  try {{
    const response = await fetch(location.href, {{cache: 'no-store'}});
    if (!response.ok) throw Error('Dashboard request failed');
    const page = new DOMParser().parseFromString(await response.text(), 'text/html');
    const next = page.querySelector('#panel');
    if (!next) throw Error('Dashboard panel missing');
    if (panel.innerHTML !== next.innerHTML) panel.replaceChildren(...next.childNodes);
    error.hidden = true;
  }} catch (failure) {{
    error.hidden = false;
    console.error(failure);
  }} finally {{
    setTimeout(refresh, 3000);
  }}
}}
setTimeout(refresh, 3000);
</script>
</body>
</html>
"""
