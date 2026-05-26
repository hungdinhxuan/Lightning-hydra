"""Dex / OIDC session helpers for Kubeflow gateway-protected HTTP APIs."""

from __future__ import annotations

import os
import re
from urllib.parse import urlencode, urlsplit

import requests


def get_dex_session_cookies(
    entry_url: str,
    *,
    dex_username: str,
    dex_password: str,
    dex_auth_type: str = "local",
    skip_tls_verify: bool = False,
) -> str:
    """
    Authenticate against Dex via the Kubeflow ingress and return session cookies.

    :param entry_url: Any URL behind the gateway (e.g. MLflow health or KFP API).
    :return: Cookie header value in the form ``key1=value1; key2=value2``.
    """
    if dex_auth_type not in {"ldap", "local"}:
        raise ValueError(
            f"Invalid dex_auth_type '{dex_auth_type}', must be one of: ['ldap', 'local']"
        )

    session = requests.Session()
    verify = not skip_tls_verify

    response = session.get(entry_url, allow_redirects=True, verify=verify)
    if response.status_code == 200:
        pass
    elif response.status_code == 403:
        url_obj = urlsplit(response.url)
        url_obj = url_obj._replace(
            path="/oauth2/start",
            query=urlencode({"rd": url_obj.path}),
        )
        response = session.get(url_obj.geturl(), allow_redirects=True, verify=verify)
    else:
        raise RuntimeError(
            f"HTTP status code '{response.status_code}' for GET against: {entry_url}"
        )

    if len(response.history) == 0:
        return ""

    url_obj = urlsplit(response.url)
    if re.search(r"/auth$", url_obj.path):
        url_obj = url_obj._replace(
            path=re.sub(r"/auth$", f"/auth/{dex_auth_type}", url_obj.path)
        )

    if re.search(r"/auth/.*/login$", url_obj.path):
        dex_login_url = url_obj.geturl()
    else:
        response = session.get(url_obj.geturl(), allow_redirects=True, verify=verify)
        if response.status_code != 200:
            raise RuntimeError(
                f"HTTP status code '{response.status_code}' for GET against: {url_obj.geturl()}"
            )
        dex_login_url = response.url

    response = session.post(
        dex_login_url,
        data={"login": dex_username, "password": dex_password},
        allow_redirects=True,
        verify=verify,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"HTTP status code '{response.status_code}' for POST against: {dex_login_url}"
        )
    if len(response.history) == 0:
        raise RuntimeError(
            "Login credentials are probably invalid - "
            f"No redirect after POST to: {dex_login_url}"
        )

    url_obj = urlsplit(response.url)
    if re.search(r"/approval$", url_obj.path):
        response = session.post(
            url_obj.geturl(),
            data={"approval": "approve"},
            allow_redirects=True,
            verify=verify,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"HTTP status code '{response.status_code}' for POST against: {url_obj.geturl()}"
            )

    return "; ".join(f"{cookie.name}={cookie.value}" for cookie in session.cookies)


def dex_credentials_from_env() -> tuple[str, str, str]:
    """Read Dex credentials from environment variables."""
    username = os.environ.get("DEX_USERNAME", "admin")
    password = os.environ.get("DEX_PASSWORD")
    if not password:
        raise RuntimeError(
            "DEX_PASSWORD is not set. Source ~/.config/mlops/kubeflow-dex.env first."
        )
    auth_type = os.environ.get("DEX_AUTH_TYPE", "local")
    return username, password, auth_type


def gateway_requires_dex_auth(tracking_uri: str) -> bool:
    """Return True when the tracking URI goes through the Kubeflow ingress."""
    host = urlsplit(tracking_uri).hostname or ""
    return host not in {"127.0.0.1", "localhost"}
