# file adding source: https://python.langchain.com/docs/integrations/tools/bearly
# subprocess source: https://betterprogramming.pub/building-a-custom-langchain-tool-for-generating-executing-code-fa20a3c89cfd
# interpreter tool source: https://github.com/shroominic/codeinterpreter-api/blob/main/src/codeinterpreterapi/session.py#L106

import subprocess
from langchain.tools import StructuredTool

def run_handler(self, code: str) -> str:
    """Run code in container and send the output to the user"""
    default_file_path = "scipt.py"

    with open("script.py", "w") as f:
        f.write(code)

    interpreter = r"path\to\interpreter"
    completed_process = subprocess.run(
        [interpreter, default_file_path], capture_output=True, timeout=10
    )

    print(completed_process, completed_process.stderr)
    succeeded = "Succeeded" if completed_process.returncode == 0 else "Failed"
    stdout = completed_process.stdout
    stderr = completed_process.stderr
    return f"Program {succeeded}\nStdout:{stdout}\nStderr:{stderr}"

class SimpleInterpreter:

    description_template = (
        "Evaluates python code in a sandbox environment. The environment resets on every execution. "
        "You must send the whole script every time and print your outputs. Script should be pure python code that can be evaluated. "
        "It should be in python format NOT markdown. The code should NOT be wrapped in backticks. "
        "All python packages including requests, matplotlib, scipy, numpy, pandas, etc are available."
        "If you have any files outputted write them to 'output/' relative to the execution path. "
        "Output can only be read from the directory, stdout, and stdin. Do not use things like plot.show() as it will not work instead write them out `output/` and a link to the file will be returned. "
        "print() any output and results so you can capture the output. \n\n"

        "The following files available in the evaluation environment:\n"
        "{file_information}"
    )

    file_description_template = "- path: '{file_path}'\n"
    def __init__(self) -> None:
        self.files = []

    def add_file(self, source_path, target_path):
        self.files.append((source_path, target_path))
        
    def get_tool(self):
        file_information = ""
        for source, target in self.files:
            file_information += self.file_description_template.format(target)

        description = self.description_template.format(file_information=file_information)


        tool = StructuredTool(
            name="python",
            description=description,
            func=run_handler,
            args_schema=CodeInput,  # type: ignore
        )

        return tool




