from __future__ import annotations

from dataclasses import dataclass
from urllib import parse, robotparser


@dataclass
class CrawlPolicy:
    max_pages_per_cycle: int = 24
    max_links_per_page: int = 100
    allowed_schemes: tuple[str, ...] = ("http", "https")
    blocked_extensions: tuple[str, ...] = (
        ".exe", ".dmg", ".apk", ".iso", ".bin", ".msi", ".pkg"
    )
    user_agent: str = "SARA-Autobot/1.0"
    strict_allowlist_mode: bool = False
    allowed_domains: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self._robots_cache: dict[str, robotparser.RobotFileParser] = {}

    def is_allowed_url(self, url: str) -> bool:
        lowered = url.lower()
        if not lowered.startswith(("http://", "https://")):
            return False
        if lowered.endswith(self.blocked_extensions):
            return False
        if self.strict_allowlist_mode:
            parsed = parse.urlparse(url)
            host = (parsed.netloc or "").lower()
            if host not in set(d.lower() for d in self.allowed_domains):
                return False
        return True

    def is_allowed_by_robots(self, url: str) -> bool:
        parsed = parse.urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False
        origin = f"{parsed.scheme}://{parsed.netloc}"
        parser = self._robots_cache.get(origin)
        if parser is None:
            parser = robotparser.RobotFileParser()
            parser.set_url(parse.urljoin(origin, "/robots.txt"))
            try:
                parser.read()
            except Exception:
                # If robots cannot be fetched, fail open to avoid deadlock.
                self._robots_cache[origin] = parser
                return True
            self._robots_cache[origin] = parser
        try:
            return bool(parser.can_fetch(self.user_agent, url))
        except Exception:
            return True
