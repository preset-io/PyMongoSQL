"""Compute the published version of this fork, normalized as the wheel will be.

Stable builds publish the declared ``__version__`` (a four-part Preset release
such as 0.7.4.1). Pull-request builds publish ``<version>+pr.<change>.<sha>``.
PEP 440 normalizes local segments (lower case; a numeric segment loses leading
zeros), so the filename must be derived from the normalized form or it will not
match the file the build backend writes.
"""

import re
import sys
from pathlib import Path

from packaging.version import Version

INIT = Path(__file__).resolve().parents[1] / "pymongosql" / "__init__.py"


def declared_version(source=None):
    source = INIT.read_text() if source is None else source
    matches = re.findall(r'^__version__: str = "([^"]+)"$', source, re.MULTILINE)
    if len(matches) != 1:
        raise SystemExit("expected exactly one __version__ declaration")
    version = Version(matches[0])
    if str(version) != matches[0] or version.local or len(version.release) != 4:
        raise SystemExit(f"__version__ must be a normalized four-part release, got {matches[0]!r}")
    return matches[0]


def release_version(base, change_id=None, revision=None):
    if change_id is None:
        return str(Version(base))
    if not re.fullmatch(r"[0-9]+", change_id) or not re.fullmatch(r"[0-9a-fA-F]{7,40}", revision or ""):
        raise SystemExit("pull-request builds need a numeric change id and a git revision")
    return str(Version(f"{base}+pr.{change_id}.{revision}"))


if __name__ == "__main__":
    print(release_version(declared_version(), *sys.argv[1:]))
