"""Local Python Worker integration test. No Cloudflare account or real secrets.

Requires current uv, workers-py and Node.js. From sdks/python:
    uv run --with 'workers-py>=1.17.5' --with 'uv>=0.12.3' python e2e/cloudflare/run.py

Optionally pass --wrangler /absolute/path/to/wrangler/bin/wrangler.js to reuse
an installed CLI; otherwise npm installs the example's Wrangler dependency in
a temporary directory. All generated projects and processes are cleaned up.
"""

import argparse
import copy
import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

SDK = Path(__file__).resolve().parents[2]
VECTORS = json.loads((SDK / "testdata/vectors.json").read_text())


class Fixture(BaseHTTPRequestHandler):
    redirected = 0
    encoded_overrides = 0

    def log_message(self, *args):
        pass

    def do_GET(self):
        url = urlsplit(self.path)
        if url.path == "/must-not-follow/v1/resolve":
            type(self).redirected += 1
        if self.headers.get("authorization") != "Bearer " + VECTORS["token"]:
            self.send_error(401)
            return
        if parse_qs(url.query).get("with") == ["shared:dev&with=other:prod"]:
            type(self).encoded_overrides += 1

        status, body = 200, copy.deepcopy(VECTORS["resolve"])
        if url.path == "/redirect/v1/resolve":
            self.send_response(302)
            self.send_header("location", "/must-not-follow/v1/resolve")
            self.end_headers()
            return
        if url.path == "/unauthorized/v1/resolve":
            status, body = (
                401,
                {"error": {"code": "unauthorized", "message": "revoked"}},
            )
        if url.path == "/tampered/v1/resolve":
            body["layers"][-1]["secrets"][0]["name"] += "_SWAPPED"
        payload = (
            b"not JSON"
            if url.path == "/malformed/v1/resolve"
            else json.dumps(body).encode()
        )
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        if url.path == "/slow-body/v1/resolve":
            time.sleep(0.2)
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass  # The client aborted a timed-out body, as required.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrangler", type=Path)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Fixture)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    process = None
    try:
        with tempfile.TemporaryDirectory(prefix="seekrit-python-worker-") as directory:
            project = Path(directory)
            source = project / "src"
            source.mkdir()
            shutil.copy(Path(__file__).with_name("entry.py"), source / "entry.py")
            (source / "vectors.py").write_text("VECTORS = " + repr(VECTORS))
            # Exercise the example's dependency manifest and the SDK's built wheel.
            example = SDK / "examples/cloudflare-workers"
            pyproject = (
                (example / "pyproject.toml")
                .read_text()
                .replace('path = "../.."', "path = " + json.dumps(str(SDK)))
            )
            (project / "pyproject.toml").write_text(pyproject)
            shutil.copy(example / "package.json", project / "package.json")
            config = json.loads((example / "wrangler.jsonc").read_text())
            config.pop("secrets")
            config["vars"] = {"FIXTURE_URL": f"http://127.0.0.1:{server.server_port}"}
            (project / "wrangler.jsonc").write_text(json.dumps(config))

            env = dict(os.environ, WRANGLER_SEND_METRICS="false")
            env["XDG_CONFIG_HOME"] = str(project / ".config")
            if args.wrangler:
                cli = args.wrangler.resolve()
                modules = project / "node_modules"
                (modules / ".bin").mkdir(parents=True)
                (modules / "wrangler").symlink_to(
                    cli.parent.parent, target_is_directory=True
                )
                (modules / ".bin/wrangler").symlink_to(cli)
            else:
                subprocess.run(
                    ["npm", "install", "--no-audit", "--no-fund"],
                    cwd=project,
                    env=env,
                    check=True,
                )

            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            with (project / "worker.log").open("w+") as log:
                process = subprocess.Popen(
                    ["pywrangler", "dev", "--ip", "127.0.0.1", "--port", str(port)],
                    cwd=project,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                deadline = time.monotonic() + 180
                try:
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            raise RuntimeError(
                                "Pywrangler exited before the Worker was ready"
                            )
                        try:
                            with urllib.request.urlopen(
                                f"http://127.0.0.1:{port}/", timeout=15
                            ) as reply:
                                result = json.load(reply)
                            assert len(result["passed"]) == 9, result
                            assert Fixture.redirected == 0, (
                                "token-bearing redirect was followed"
                            )
                            assert Fixture.encoded_overrides == 1, (
                                "override was not transmitted correctly"
                            )
                            print(
                                "Local Python Worker passed: "
                                + ", ".join(result["passed"])
                            )
                            return
                        except urllib.error.HTTPError as exc:
                            print(exc.read().decode(errors="replace"))
                            raise
                        except (urllib.error.URLError, TimeoutError):
                            time.sleep(0.5)
                    raise RuntimeError("Timed out waiting for the local Python Worker")
                except Exception:
                    log.seek(0)
                    print(log.read())  # Contains only public test fixtures.
                    raise
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=15)
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
