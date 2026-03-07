import subprocess
import sys
from pathlib import Path

def run_api_server(port=8080):
    try:
        current_file = Path(__file__).resolve()
        plateau_navigator_root = current_file.parent.parent.parent.parent
        parent_dir = plateau_navigator_root.parent
        qparser_dir = parent_dir / "QParser/qparser"
        if not qparser_dir.exists():
            raise FileNotFoundError(
                f"Target dir not found: {qparser_dir}\n"
                f"Make sure you have cloned the QParser repository right in the same father dir as plateau-navigator."
            )
        print(f"Target dir found: {qparser_dir}")
        print("Compiling QParser with maven...")
        compile_process = subprocess.run(
            ["mvn", "clean", "package", "-DskipTests"],
            cwd=str(qparser_dir),
            capture_output=True,
            text=True
        )
        if compile_process.returncode != 0:
            print("Error compiling:", file=sys.stderr)
            print(compile_process.stderr, file=sys.stderr)
            raise RuntimeError("Maven compiling failed")
        print("Successfully compiled.")
        target_dir = qparser_dir / "target"
        jar_files = list(target_dir.glob("*.jar"))
        executable_jars = [
            jar for jar in jar_files
            if not jar.name.startswith("original-")
            and not jar.name.endswith("-sources.jar")
            and not jar.name.endswith("-javadoc.jar")
        ]
        if not executable_jars:
            raise FileNotFoundError(f"Not found jar executable at {target_dir}")
        jar_file = None
        for jar in executable_jars:
            if "with-dependencies" in jar.name or "jar-with-dependencies" in jar.name:
                jar_file = jar
                break
        if jar_file is None:
            jar_file = executable_jars[0]
        print(f"Executing jar: {jar_file.name}")
        print(f"Port: {port}")
        execute_process = subprocess.run(
            ["java", "-jar", str(jar_file), str(port)],
            cwd=str(qparser_dir)
        )
        return execute_process.returncode
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_api_server(port)
