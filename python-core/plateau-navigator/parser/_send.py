from ._parse_file import ParseFile
from urllib.request import urlopen
from urllib.error import URLError


class SendCode:
    def __init__(self, is_up: bool, parser_instance: ParseFile, port=8080):
        self.is_up = is_up
        self.port = port
        self.script_lines_list = parser_instance._read()

    def _is_server_up(self) -> bool:
        url = "http://localhost:{self.port}/health"
        try:
            with urlopen(url, timeout=2) as response:
                return response.getcode() == 200
        except (URLError, Exception):
            return False

    def _format(self) -> str:
        self.text_to_send = "\n".join(line for line in self.script_lines_list)
        return self.text_to_send

    def _send(self):
        if not self._is_server_up:
            raise URLError("Error: Server is not running")