from pathlib import Path

class ParseFile:
    def __init__(self, extension: str, path_to_script: str):
        if extension not in ["qasm", "py"]:
            raise Exception("Error: Script extension must be ['qasm' or 'py']")
        self.extension = extension
        self.path_to_script = path_to_script
        if not Path(self.path_to_script).exists:
            raise FileExistsError("File path provided do not exists")
        self.plain_code = ""

    def _read(self):
        script_lines = []
        with open(self.path_to_script, 'r') as s:
            for line in s:
                script_lines.append(line)
        return script_lines