class SendCode:
    def __init__(self, is_up: bool, port: int, QParser_instance: ParseFile):
        self.is_up = is_up
        self.port = port
        self.script_lines_list = QParser_instance._read()

    def _is_server_up(self):
        pass

    def _format(self):
        self.text_to_send = "\n".join(line for line in self.script_lines_list)
        return self.text_to_send

    def _send(self):
        pass