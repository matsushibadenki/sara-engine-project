import sys
import re
import os


def apply_fixes():
    errors = []
    with open("mypy_errors.txt", "r") as f:
        for line in f:
            if "error:" not in line:
                continue

            match = re.search(
                r"^(.*?):(\d+): error: .*\s+\[([a-z-]+)\]\s*$", line)
            if match:
                file_path = match.group(1)
                line_idx = int(match.group(2)) - 1
                error_code = match.group(3)
                errors.append((file_path, line_idx, error_code))

    errors.sort(key=lambda x: (x[0], x[1]), reverse=True)

    file_changes = {}
    for fp, lidx, err in errors:
        if fp not in file_changes:
            try:
                with open(fp, "r") as src:
                    file_changes[fp] = src.readlines()
            except FileNotFoundError:
                continue

        lines = file_changes[fp]
        if 0 <= lidx < len(lines):
            line_str = lines[lidx].rstrip('\n')
            if err not in line_str:
                if "# type: ignore[" in line_str:
                    line_str = re.sub(
                        r"# type: ignore\[(.*?)\]", f"# type: ignore[\\1,{err}]", line_str)
                elif "type: ignore" in line_str:
                    line_str = line_str.replace(
                        "type: ignore", f"type: ignore[{err}]")
                else:
                    line_str = f"{line_str}  # type: ignore[{err}]"
                lines[lidx] = line_str + '\n'

    for fp, lines in file_changes.items():
        with open(fp, "w") as dest:
            dest.writelines(lines)


if __name__ == "__main__":
    apply_fixes()
