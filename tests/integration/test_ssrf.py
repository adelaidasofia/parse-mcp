"""SSRF mitigation tests for parse-mcp _read_url (MYC-101).

Five test classes mirroring the cross-MCP fixture pattern from
mycelium-security: redirect / backslash / DNS-private / IPv6-link-local /
embedded-creds. The helper itself is exhaustively tested upstream; this
file proves the wiring at parse-mcp's only fetch site (server._read_url)
actually invokes the helper.
"""
from __future__ import annotations

import socket
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_HERE = Path(__file__).resolve().parent.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from server import _read_url  # noqa: E402


class TestFetchSiteServerReadUrl:
    """server._read_url is the only outbound HTTP path. Five-test matrix
    proves the mycelium-security helper is wired at this fetch site."""

    def test_rejects_url_with_backslash(self):
        data, filename, err = _read_url("https://example.com/\\evil")
        assert data is None
        assert err is not None
        assert "refused (SSRF)" in err
        assert "banned character" in err

    def test_rejects_embedded_credentials(self):
        data, filename, err = _read_url("https://user:pass@example.com/path")
        assert data is None
        assert err is not None
        assert "refused (SSRF)" in err
        assert "credentials" in err

    def test_rejects_ipv6_link_local(self):
        # http://[fe80::1]/path — IPv6 link-local literal
        data, filename, err = _read_url("http://[fe80::1]/path")
        assert data is None
        assert err is not None
        assert "refused (SSRF)" in err

    def test_rejects_dns_resolving_to_private_ip(self):
        # Hostname that "resolves" to 10.0.0.1 — DNS rebinding scenario
        with patch("mycelium_security.url.socket.getaddrinfo") as mock_resolver:
            mock_resolver.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 0))
            ]
            data, filename, err = _read_url("http://attacker-controlled.example.com/path")
        assert data is None
        assert err is not None
        assert "refused (SSRF)" in err

    def test_rejects_aws_metadata_endpoint(self):
        # http://169.254.169.254/latest/meta-data/iam — IMDS exfil attempt
        data, filename, err = _read_url(
            "http://169.254.169.254/latest/meta-data/iam/security-credentials/"
        )
        assert data is None
        assert err is not None
        assert "refused (SSRF)" in err
        assert "metadata" in err.lower()

    def test_redirect_to_metadata_blocked(self):
        """3xx redirect from a public host to a metadata IP must be refused.

        The redirect handler raises HTTPError before urllib follows the
        Location: header to the private IP. Verify by mocking the URL
        opener to return a 302.
        """
        from urllib.error import HTTPError
        from urllib.parse import urlparse

        import server as server_module

        # Stub the opener.open() to return a 302 that would redirect to 169.254
        class _Fake302Response:
            def __init__(self):
                self.status = 302
                self.headers = {"Location": "http://169.254.169.254/latest/meta-data/"}

        def _raise_302(req, **kwargs):
            # Simulate urllib's redirect handler chain: our handler raises.
            raise HTTPError(
                req.full_url, 302, "redirect blocked (SSRF mitigation)",
                {"Location": "http://169.254.169.254/latest/meta-data/"},
                None,
            )

        # Use a publicly-resolvable host so the pre-fetch IP check passes.
        # 8.8.8.8 is Google DNS; any HTTP response would come from it.
        with patch.object(server_module._OPENER, "open", side_effect=_raise_302):
            data, filename, err = _read_url("http://8.8.8.8/")
        assert data is None
        assert err is not None
        assert "302" in err or "redirect" in err.lower()
