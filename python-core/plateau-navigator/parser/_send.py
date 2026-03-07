from _parse_file import ParseFile
import requests
from requests import Response
from requests.exceptions import ConnectionError, Timeout


class SendCode:
    def __init__(self, parser_instance: ParseFile, port: int = 8080, script_type: str = "QASM", desired_type: str = "QISKIT"):
        self.port = port
        self.script_type = script_type
        self.desired_type = desired_type
        self.script_lines_list = parser_instance._read()
        self._session = requests.Session()

    def _is_server_up(self) -> bool:
        url = f"http://localhost:{self.port}/health"
        try:
            response = self._session.get(url, timeout=2)
            return response.status_code == 200
        except (ConnectionError, Timeout, Exception):
            return False

    def _format(self) -> str:
        return "\n".join(line for line in self.script_lines_list)

    def _send(self) -> dict:
        if not self._is_server_up():
            raise ConnectionError("Error: Server is not running")
        url = f"http://localhost:{self.port}/api/parse"
        content = {
            "script": self._format(),
            "scriptType": self.script_type,
            "desiredType": self.desired_type
        }
        try:
            response = self._session.post(
                url,
                json=content,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except (ConnectionError, Timeout) as e:
            raise ConnectionError(f"Error: Could not reach server — {e}") from e
        
    def _get_code(self, response: dict):
        script_content = response.get("parsedScript", "")
        if not script_content:
            print("No code content found in response.")
            return
        lines = script_content.split("\n")
        for i, line in enumerate(lines, start=1):
            print(f"{i:>3} | {line}")