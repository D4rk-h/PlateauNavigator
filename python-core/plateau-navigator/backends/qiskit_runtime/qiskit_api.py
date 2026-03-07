# qiskit_api.py

import requests
import os
import time
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta


class QiskitRuntimeAPI:
    """
    Thin wrapper around the IBM Quantum Runtime REST API.

    Authentication:
        Credentials resolved in priority order:
            1. Constructor arguments (api_key, crn)
            2. Environment variables: IBMQ_API_KEY, IBMQ_CRN
        Bearer tokens are fetched via IBM IAM and refreshed automatically
        before expiry.

    Session management:
        IBM Runtime sessions batch multiple jobs to the same QPU,
        avoiding re-queuing between VQE gradient evaluations.
        Use create_session() / close_session() or the QiskitBackend
        context manager which handles this automatically.
    """

    _IAM_URL = "https://iam.cloud.ibm.com/identity/token"

    def __init__(
        self,
        api_base: str = "https://quantum.cloud.ibm.com/api/v1",
        crn: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.api_base = api_base
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.api_key = api_key or os.environ.get("IBMQ_API_KEY")
        self.crn = crn or os.environ.get("IBMQ_CRN")

        if not self.api_key:
            raise ValueError(
                "IBM API key required. Pass api_key= or set IBMQ_API_KEY env var."
            )
        if not self.crn:
            raise ValueError(
                "IBM CRN required. Pass crn= or set IBMQ_CRN env var."
            )

        self._bearer_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._authenticate()

    def _authenticate(self) -> None:
        self._bearer_token = self._fetch_bearer_token(self.api_key)
        self._token_expiry = datetime.now() + timedelta(minutes=55)

    def _ensure_valid_token(self) -> None:
        if (
            not self._bearer_token
            or not self._token_expiry
            or datetime.now() >= self._token_expiry
        ):
            self._authenticate()

    def _fetch_bearer_token(self, api_key: str) -> str:
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": api_key,
        }
        response = requests.post(self._IAM_URL, headers=headers, data=data)
        response.raise_for_status()
        return response.json()["access_token"]

    def _get_headers(self) -> Dict[str, str]:
        self._ensure_valid_token()
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._bearer_token}",
            "Service-CRN": self.crn,
            "IBM-API-Version": "2025-05-01",
        }

    # ── HTTP helpers ───────────────────────────────────────────────────────

    def _request(
        self,
        method: str,
        url: str,
        retryable_statuses: tuple = (429, 500, 502, 503, 504),
        **kwargs,
    ) -> requests.Response:
        """
        Execute an HTTP request with exponential backoff retry.

        Retries on transient failures (rate limits, server errors).
        Auth errors (401, 403) are not retried — they indicate a
        credential problem that won't resolve itself.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.request(
                    method, url, headers=self._get_headers(), **kwargs
                )
                if resp.status_code == 401:
                    self._authenticate()
                    resp = requests.request(
                        method, url, headers=self._get_headers(), **kwargs
                    )
                if resp.status_code in retryable_statuses:
                    wait = self.retry_delay * (2 ** attempt)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp
            except requests.exceptions.ConnectionError as e:
                last_exc = e
                time.sleep(self.retry_delay * (2 ** attempt))

        raise ConnectionError(
            f"Request to {url} failed after {self.max_retries} attempts. "
            f"Last error: {last_exc}"
        )

    def get_backends(self) -> Dict[str, Any]:
        return self._request("GET", f"{self.api_base}/backends").json()

    def get_backend(self, backend_name: str) -> Dict[str, Any]:
        return self._request("GET", f"{self.api_base}/backends/{backend_name}").json()

    def submit_job(
        self,
        program_id: str,
        backend: str,
        params: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "program_id": program_id,
            "backend": backend,
            "params": params,
        }
        if session_id:
            payload["session_id"] = session_id
        return self._request("POST", f"{self.api_base}/jobs", json=payload).json()

    def get_job(self, job_id: str) -> Dict[str, Any]:
        return self._request("GET", f"{self.api_base}/jobs/{job_id}").json()

    def get_job_results(self, job_id: str) -> Dict[str, Any]:
        return self._request("GET", f"{self.api_base}/jobs/{job_id}/results").json()

    def cancel_job(self, job_id: str) -> None:
        resp = self._request("DELETE", f"{self.api_base}/jobs/{job_id}")
        if resp.status_code not in (200, 204):
            raise RuntimeError(f"Unexpected status {resp.status_code} cancelling job {job_id}")

    def list_jobs(
        self,
        limit: int = 10,
        offset: int = 0,
        backend: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if backend:
            params["backend"] = backend
        return self._request("GET", f"{self.api_base}/jobs", params=params).json()

    def create_session(
        self, backend: str, mode: str = "dedicated", max_ttl: int = 28800
    ) -> Dict[str, Any]:
        """
        Create a Runtime session.

        Args:
            backend: QPU backend name. Session is bound to a specific backend.
            mode:    'dedicated' (exclusive access) or 'batch'.
            max_ttl: Maximum session lifetime in seconds (default 8 hours).
        """
        payload = {"backend": backend, "mode": mode, "max_ttl": max_ttl}
        return self._request("POST", f"{self.api_base}/sessions", json=payload).json()

    def get_session(self, session_id: str) -> Dict[str, Any]:
        return self._request("GET", f"{self.api_base}/sessions/{session_id}").json()

    def close_session(self, session_id: str) -> None:
        self._request("DELETE", f"{self.api_base}/sessions/{session_id}")