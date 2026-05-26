"""MLflow client helpers for Kubeflow gateway (Dex) authentication."""

from __future__ import annotations

from urllib.parse import urljoin

from src.utils.dex_session import (
    dex_credentials_from_env,
    gateway_requires_dex_auth,
    get_dex_session_cookies,
)


def register_dex_auth_for_mlflow(
    tracking_uri: str,
    *,
    dex_username: str | None = None,
    dex_password: str | None = None,
    dex_auth_type: str | None = None,
    skip_tls_verify: bool = False,
) -> None:
    """
    Register Dex session cookies on all MLflow tracking REST calls.

    Required when ``MLFLOW_TRACKING_URI`` points at the Kubeflow ingress
    (for example ``http://192.168.0.240/mlflow``). Not needed for port-forward
    or in-cluster service URLs.
    """
    if not gateway_requires_dex_auth(tracking_uri):
        return

    if dex_username is None or dex_password is None or dex_auth_type is None:
        env_user, env_password, env_auth_type = dex_credentials_from_env()
        dex_username = dex_username or env_user
        dex_password = dex_password or env_password
        dex_auth_type = dex_auth_type or env_auth_type

    entry_url = urljoin(tracking_uri.rstrip("/") + "/", "health")
    cookies = get_dex_session_cookies(
        entry_url,
        dex_username=dex_username,
        dex_password=dex_password,
        dex_auth_type=dex_auth_type,
        skip_tls_verify=skip_tls_verify,
    )

    from mlflow.tracking.request_header.abstract_request_header_provider import (
        RequestHeaderProvider,
    )
    from mlflow.tracking.request_header.registry import (
        _request_header_provider_registry,
    )

    class DexCookieProvider(RequestHeaderProvider):
        def in_context(self) -> bool:
            return True

        def request_headers(self) -> dict[str, str]:
            return {"Cookie": cookies}

    _request_header_provider_registry.register(DexCookieProvider)
