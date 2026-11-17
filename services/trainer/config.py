"""Generate the ignored container overlay from the host's local DVC remote."""

import configparser
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def generate(source=Path(".dvc/config.local"), target=Path("services/trainer/dvc.local")):
    config = configparser.ConfigParser(interpolation=None)
    if not source.is_file():
        raise SystemExit(f"Missing {source}: configure the host DVC SSH remote first.")
    config.read(source)
    remote = "'remote \"storage\"'"
    if not config.has_section(remote) or not all(
        config.has_option(remote, key) for key in ("url", "user")
    ):
        raise SystemExit(f'{source} must define remote "storage" with url and user.')
    url = urlsplit(config[remote]["url"])
    if url.scheme != "ssh" or not url.hostname:
        raise SystemExit("The storage remote must use an ssh:// URL.")
    authority = "host.docker.internal" + (f":{url.port}" if url.port else "")
    url = urlunsplit((url.scheme, authority, url.path, url.query, url.fragment))
    target.write_text(
        "# Read-only overlay; the host's config.local remains untouched.\n"
        "[core]\n    site_cache_dir = /tmp/dvc-site\n"
        f'[{remote}]\n    url = {url}\n    user = {config[remote]["user"]}\n'
    )
    print(f"Generated {target}")


if __name__ == "__main__":
    generate()
