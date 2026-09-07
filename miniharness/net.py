"""The subset of ``requests`` this harness actually uses, on the stdlib.

Every HTTP call here is one of three shapes: POST some JSON, GET a file, or
read an SSE stream line by line. ``urllib`` does all three. Carrying a
third-party dependency for it costs more than it saves — a benchmark container
without ``pip`` could not run the harness at all, and the failure surfaced as an
ImportError in the agent loop rather than anything to do with the task.

The API deliberately mirrors ``requests`` for the members in use, so call sites
read the same and stay familiar. It is not a general-purpose client: no
sessions, no cookies, no redirect policy beyond urllib's default.
"""
import json as _json
import socket
import urllib.error
import urllib.parse
import urllib.request


class RequestException(Exception):
    """Base for every failure this module raises."""


class ConnectionError(RequestException):  # noqa: A001 - mirrors requests
    pass


class Timeout(RequestException):
    pass


class ChunkedEncodingError(RequestException):
    """A stream ended mid-body. Raised while iterating, not while connecting."""


class HTTPError(RequestException):
    pass


class Response:
    """File-like HTTP response with the handful of accessors we depend on.

    Bodies are read lazily: ``iter_lines``/``iter_content`` stream, while
    ``text``/``json`` buffer on first access. A response used for streaming must
    not also be asked for ``.text`` — that is true of ``requests`` as well.
    """

    def __init__(self, raw, status_code: int, headers, url: str):
        self.raw = raw
        self.status_code = status_code
        self.headers = headers
        self.url = url
        self._body = None

    # -- context manager so `with get(...) as r:` works ---------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        try:
            self.raw.close()
        except Exception:
            pass

    @property
    def content(self) -> bytes:
        if self._body is None:
            try:
                self._body = self.raw.read()
            except (socket.timeout, TimeoutError) as e:
                raise Timeout(str(e)) from e
            except OSError as e:
                raise ChunkedEncodingError(str(e)) from e
            finally:
                self.close()
        return self._body

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", "replace")

    def json(self):
        return _json.loads(self.text)

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    def raise_for_status(self):
        if not self.ok:
            raise HTTPError(f"{self.status_code} from {self.url}: {self.text[:400]}")

    def iter_content(self, chunk_size: int = 8192):
        try:
            while True:
                chunk = self.raw.read(chunk_size)
                if not chunk:
                    return
                yield chunk
        except (socket.timeout, TimeoutError) as e:
            raise Timeout(str(e)) from e
        except OSError as e:
            raise ChunkedEncodingError(str(e)) from e
        finally:
            self.close()

    def iter_lines(self, decode_unicode: bool = False):
        """Yield lines without their terminator, as ``requests`` does.

        Server-sent events are newline-delimited, and ``readline`` on the raw
        socket already blocks per line, so this stays incremental — tokens
        arrive as the server emits them rather than at end of body.
        """
        try:
            for raw in self.raw:
                line = raw.rstrip(b"\r\n")
                yield line.decode("utf-8", "replace") if decode_unicode else line
        except (socket.timeout, TimeoutError) as e:
            raise Timeout(str(e)) from e
        except OSError as e:
            raise ChunkedEncodingError(str(e)) from e
        finally:
            self.close()


def request(method: str, url: str, *, headers=None, json=None, data=None,
            params=None, timeout=60, stream=False) -> Response:
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    hdrs = dict(headers or {})
    if json is not None:
        body = _json.dumps(json).encode("utf-8")
        hdrs.setdefault("Content-Type", "application/json")
    elif isinstance(data, dict):
        # requests form-encodes a dict body; urllib wants bytes. The research
        # sub-loop's DuckDuckGo POST relies on this.
        body = urllib.parse.urlencode(data).encode("utf-8")
        hdrs.setdefault("Content-Type", "application/x-www-form-urlencoded")
    elif isinstance(data, str):
        body = data.encode("utf-8")
    else:
        body = data
    req = urllib.request.Request(url, data=body, headers=hdrs, method=method.upper())
    try:
        raw = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        # A 4xx/5xx is a response, not an exception — callers inspect
        # status_code and decide whether it is retryable.
        return Response(e, e.code, e.headers, url)
    except (socket.timeout, TimeoutError) as e:
        raise Timeout(f"{url}: {e}") from e
    except urllib.error.URLError as e:
        # urllib wraps a socket timeout in URLError, so the distinction between
        # "unreachable" and "too slow" is only visible through .reason.
        if isinstance(e.reason, (socket.timeout, TimeoutError)):
            raise Timeout(f"{url}: {e.reason}") from e
        raise ConnectionError(f"{url}: {e.reason}") from e
    except OSError as e:
        raise ConnectionError(f"{url}: {e}") from e
    return Response(raw, raw.status, raw.headers, url)


def get(url, **kw) -> Response:
    return request("GET", url, **kw)


def post(url, **kw) -> Response:
    return request("POST", url, **kw)
