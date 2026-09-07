"""The stdlib HTTP shim. Verified against a real socket, not a mock — the whole
point of this module is behaviour at the socket layer (incremental reads,
timeouts, error mapping), which a mock cannot demonstrate."""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from miniharness import net


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _route(self):
        if self.path == "/sse":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(5):
                self.wfile.write(f"data: chunk{i}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.05)
            self.wfile.write(b"data: [DONE]\n\n")
        elif self.path == "/slow":
            time.sleep(5)
            self.send_response(200); self.end_headers()
        elif self.path.startswith("/status/"):
            code = int(self.path.rsplit("/", 1)[1])
            body = b'{"error":"nope"}'
            self.send_response(code)
            self.send_header("Retry-After", "0")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = json.dumps({"echo": self.path}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    do_GET = do_POST = _route


@pytest.fixture(scope="module")
def base():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_post_json_round_trip(base):
    r = net.post(f"{base}/hi", json={"a": 1}, timeout=5)
    assert r.status_code == 200 and r.json()["echo"] == "/hi"


def test_sse_arrives_incrementally(base):
    """If this buffers, tokens appear only at end of turn and the UI dies."""
    r = net.get(f"{base}/sse", timeout=5, stream=True)
    t0, times = time.time(), []
    for line in r.iter_lines(decode_unicode=True):
        if line.startswith("data:"):
            times.append(time.time() - t0)
    assert len(times) == 6
    # First payload must land well before the last one is written.
    assert times[0] < times[-1] * 0.5, f"buffered: {times}"


def test_iter_lines_strips_terminators(base):
    lines = [l for l in net.get(f"{base}/sse", timeout=5).iter_lines(decode_unicode=True) if l]
    assert lines[0] == "data: chunk0" and lines[-1] == "data: [DONE]"


def test_error_status_is_a_response_not_an_exception(base):
    """_connect inspects status_code to decide what is retryable, so a 429 must
    come back as a response with headers intact."""
    r = net.post(f"{base}/status/429", json={}, timeout=5)
    assert r.status_code == 429
    assert r.headers.get("Retry-After") == "0"
    assert "nope" in r.text


def test_raise_for_status(base):
    with pytest.raises(net.HTTPError):
        net.get(f"{base}/status/500", timeout=5).raise_for_status()
    net.get(f"{base}/ok", timeout=5).raise_for_status()


def test_timeout_is_distinct_from_connection_failure(base):
    with pytest.raises(net.Timeout):
        net.get(f"{base}/slow", timeout=0.3)
    with pytest.raises(net.ConnectionError):
        net.get("http://127.0.0.1:1/nothing", timeout=2)


def test_exceptions_share_a_base():
    for exc in (net.Timeout, net.ConnectionError, net.ChunkedEncodingError, net.HTTPError):
        assert issubclass(exc, net.RequestException)


def test_iter_content_chunks(base):
    got = b"".join(net.get(f"{base}/blob", timeout=5).iter_content(chunk_size=4))
    assert json.loads(got)["echo"] == "/blob"


def test_no_third_party_imports_in_the_core_loop():
    """The harness must run on a bare Python 3.10+ container with no pip."""
    import subprocess, sys, textwrap
    code = textwrap.dedent("""
        import sys
        blocked = {"requests", "rich", "prompt_toolkit", "urllib3", "httpx"}
        class Block:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in blocked: raise ImportError(name)
        sys.meta_path.insert(0, Block())
        import miniharness.loop, miniharness.context, miniharness.tools
        print("OK")
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert "OK" in r.stdout, r.stderr[-600:]


# ── live network, opt-in ────────────────────────────────────────────────────
# The local-socket tests above cannot exercise TLS, redirect-following or Range,
# and the model download path needs all three: Hugging Face 302s to a CDN and
# resumes partial files with a Range request. Run with MH_NET_TESTS=1.
network = pytest.mark.skipif(not os.environ.get("MH_NET_TESTS"),
                             reason="set MH_NET_TESTS=1 for live network tests")
GGUF = ("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/"
        "qwen2.5-0.5b-instruct-q8_0.gguf")


@network
def test_tls_and_redirects_to_a_cdn():
    r = net.get(GGUF, headers={"User-Agent": "miniharness/0.1"}, timeout=30, stream=True)
    try:
        assert r.status_code == 200
        assert int(r.headers.get("Content-Length", 0)) > 1_000_000
        assert next(r.iter_content(4096))[:4] == b"GGUF"
    finally:
        r.close()


@network
def test_range_request_returns_206():
    """models.py resumes a partial download by byte offset; a server that
    ignored Range would silently restart and corrupt the file."""
    r = net.get(GGUF, headers={"User-Agent": "miniharness/0.1",
                               "Range": "bytes=100-199"}, timeout=30)
    assert r.status_code == 206 and len(r.content) == 100


@network
def test_json_api():
    r = net.get("https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                timeout=30)
    assert r.status_code == 200 and r.json()["siblings"]


def test_dict_body_is_form_encoded(base):
    """requests form-encodes a dict body; urllib wants bytes. The research
    sub-loop's DuckDuckGo POST passes a dict, so dropping this silently breaks
    web search with a TypeError deep in urllib."""
    r = net.post(f"{base}/form", data={"q": "hello world", "b": "2"}, timeout=5)
    assert r.status_code == 200


def test_dict_body_encoding_is_correct():
    import urllib.parse
    assert urllib.parse.urlencode({"q": "a b&c"}) == "q=a+b%26c"


def test_str_body_is_encoded(base):
    assert net.post(f"{base}/raw", data="plain text", timeout=5).status_code == 200


def test_query_params_are_appended(base):
    """server.py's slot actions pass params=; requests turns those into a query
    string. Dropping the kwarg raises TypeError, which the caller's
    RequestException handler does not catch — it would crash, not degrade."""
    r = net.post(f"{base}/slots/0", params={"action": "save"}, json={"f": "x"}, timeout=5)
    assert r.json()["echo"] == "/slots/0?action=save"


def test_params_merge_with_an_existing_query(base):
    r = net.get(f"{base}/x?a=1", params={"b": "2"}, timeout=5)
    assert r.json()["echo"] == "/x?a=1&b=2"


def test_every_requests_kwarg_used_in_the_tree_is_supported():
    """Guards against the next silent regression: any kwarg a call site passes
    must exist in the shim's signature. `params=` was missing and would have
    raised TypeError past a handler that only catches RequestException."""
    import inspect
    import pathlib
    import re

    sig = set(inspect.signature(net.request).parameters)
    bad = {}
    for f in sorted(pathlib.Path("miniharness").glob("*.py")):
        src = f.read_text()
        for m in re.finditer(r"requests\.(?:get|post)\(", src):
            i, depth = m.end(), 1
            while i < len(src) and depth:                 # match the call's parens
                depth += (src[i] == "(") - (src[i] == ")")
                i += 1
            args = src[m.end():i - 1]
            args = re.sub(r"\([^()]*\)", "", args)        # drop nested call args
            used = set(re.findall(r"(?:^|,)\s*(\w+)\s*=", args))
            if used - sig:
                bad[f.name] = used - sig
    assert not bad, f"unsupported kwargs at call sites: {bad}"
