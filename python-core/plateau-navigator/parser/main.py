from _parse_file import ParseFile
from _send import SendCode

if __name__ == "__main__":
    script_type = "QASM"
    desired_type = "QISKIT"
    parser = ParseFile(script_type, "test.qasm")
    send_code = SendCode(parser_instance=parser, 
                         port=8080, script_type=parser.extension, desired_type=desired_type)
    response = send_code._send()
    send_code._get_code(response)
