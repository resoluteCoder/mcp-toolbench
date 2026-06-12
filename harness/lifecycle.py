"""Server lifecycle management: clone repos, start/stop servers, health checks."""

import shutil
import subprocess
import time
from pathlib import Path

import requests

REPOS_DIR = Path(__file__).resolve().parent.parent / "repos"


def setup_server(config: dict) -> subprocess.Popen | None:
    """Clone repo (if configured), run setup, start the server.

    Returns the server process handle, or None if the server is
    externally managed (no 'start' command configured).
    """
    repo_dir = None

    if config.get("repo"):
        repo_dir = REPOS_DIR / config["name"]
        _clone_repo(config["repo"], repo_dir)

    if config.get("setup") and repo_dir:
        subprocess.run(
            config["setup"],
            shell=True,
            cwd=repo_dir,
            check=True,
        )

    if config.get("start"):
        cwd = repo_dir if repo_dir else None
        process = subprocess.Popen(
            config["start"],
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if config.get("endpoint"):
            wait_for_healthy(config["endpoint"])

        return process

    return None


def teardown_server(process: subprocess.Popen | None, config: dict | None = None) -> None:
    """Stop the server process and optionally clean up the cloned repo."""
    if process:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def wait_for_healthy(endpoint: str, timeout: int = 60, interval: float = 2.0) -> None:
    """Poll the endpoint until it responds with a 2xx status."""
    deadline = time.time() + timeout
    last_error = None

    while time.time() < deadline:
        try:
            resp = requests.get(endpoint, timeout=5)
            if resp.ok:
                return
        except requests.ConnectionError as e:
            last_error = e
        time.sleep(interval)

    raise TimeoutError(
        f"Server at {endpoint} not healthy after {timeout}s. "
        f"Last error: {last_error}"
    )


def _clone_repo(repo_url: str, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(dest)],
        check=True,
    )
