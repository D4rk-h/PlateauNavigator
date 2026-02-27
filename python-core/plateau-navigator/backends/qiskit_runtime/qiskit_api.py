import requests
import os
from typing import Optional, Dict, Any
from datetime import datetime, timedelta


class QiskitRuntimeAPI:

    def __init__(self, api_base: str = "https://quantum.cloud.ibm.com/api/v1", crn: Optional[str] = None, api_key: Optional[str] = None):
        self.api_base = api_base
        self.crn = crn or os.environ['CRN']
        self.api_key = api_key or os.environ['APIKEY']
        if not self.api_key:
            raise ValueError("API key must be provided or set in APIKEY environment variable")
        if not self.crn:
            raise ValueError("CRN must be provided or set in CRN environment variable")
        self.bearer_token = None
        self.token_expiration = None
        self._authenticate()

    def _authenticate(self) -> None:
        self.bearer_token = self.request_bearer_token(self.api_key)
        self.token_expiration = datetime.now() + timedelta(hours=1)

    def _ensure_valid_token(self) -> None:
        if not self.bearer_token or (self.token_expiration and datetime.now() >= self.token_expiration):
            self._authenticate()

    def request_bearer_token(self, api_key: Optional[str] = None) -> str:
        api_key = api_key or self.api_key
        iam_url = "https://iam.cloud.ibm.com/identity/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {"grant_type": "urn:ibm:params:oauth:grant-type:apikey","apikey": api_key}
        response = requests.post(iam_url, headers=headers, data=data)
        response.raise_for_status()
        token_data = response.json()
        return token_data["access_token"]

    def _get_headers(self) -> Dict[str, str]:
        self._ensure_valid_token()
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.bearer_token}",
            "Service-CRN": self.crn,
            "IBM-API-Version": "2025-05-01"
        }

    def get_backends(self) -> Dict[str, Any]:
        url = f"{self.api_base}/backends"
        response = requests.get(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json()

    def get_backend(self, backend_name: str) -> Dict[str, Any]:
        url = f"{self.api_base}/backends/{backend_name}"
        response = requests.get(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json()

    def submit_job(self, program_id: str, backend: str, params: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        url = f"{self.api_base}/jobs"
        headers = self._get_headers()
        headers["Content-Type"] = "application/json"
        payload = {
            "program_id": program_id,
            "backend": backend,
            "params": params
        }
        if session_id:
            payload["session_id"] = session_id
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()

    def get_job(self, job_id: str) -> Dict[str, Any]:
        url = f"{self.api_base}/jobs/{job_id}"
        response = requests.get(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json()

    def get_job_results(self, job_id: str) -> Dict[str, Any]:
        url = f"{self.api_base}/jobs/{job_id}/results"
        response = requests.get(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json()

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        url = f"{self.api_base}/jobs/{job_id}"
        response = requests.delete(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json()

    def create_session(self, mode: str = "dedicated", max_ttl: int = 28800) -> Dict[str, Any]:
        url = f"{self.api_base}/sessions"
        headers = self._get_headers()
        headers["Content-Type"] = "application/json"
        payload = {"mode": mode, "max_ttl": max_ttl}
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()

    def get_session(self, session_id: str) -> Dict[str, Any]:
        url = f"{self.api_base}/sessions/{session_id}"
        response = requests.get(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json()

    def close_session(self, session_id: str) -> Dict[str, Any]:
        url = f"{self.api_base}/sessions/{session_id}"
        response = requests.delete(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json()

    def list_jobs(self, limit: int = 10, offset: int = 0, backend: Optional[str] = None) -> Dict[str, Any]:
        url = f"{self.api_base}/jobs"
        params = {"limit": limit, "offset": offset}
        if backend:
            params["backend"] = backend
        response = requests.get(url, headers=self._get_headers(), params=params)
        response.raise_for_status()
        return response.json()