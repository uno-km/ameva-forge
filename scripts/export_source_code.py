import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

EXCLUDE_DIRS = {'.git', '.venv', 'node_modules', 'dist', 'build', 'coverage', '__pycache__', 'releases', 'tools'}
EXCLUDE_FILES = {'package-lock.json', 'pnpm-lock.yaml', '*.whl', '*.tar.gz', '*.tgz', '*.png', '*.jpg', '*.jpeg', '*.gif', '*.mp4', '*.mp3', '.env*', '*.pem', '*.key', '*.cert', 'API token', '비밀번호', '개인키', '인증서'}
LANG_MAP = {'.py': 'python', '.ts': 'typescript', '.js': 'javascript', '.md': 'markdown', '.json': 'json', '.html': 'html', '.css': 'css', '.wgsl': 'wgsl'}

def cmd(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, encoding='utf-8').stdout.strip()
    except Exception:
        return ""

def generate_tree(path: Path, prefix="") -> str:
    try:
        entries = sorted([p for p in path.iterdir() if p.name not in EXCLUDE_DIRS and not any(p.match(pat) for pat in EXCLUDE_FILES)], key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        return ""
    lines = []
    for i, p in enumerate(entries):
        is_last = (i == len(entries) - 1)
        conn = "ㄴ-- " if is_last else "├── "
        lines.append(f"{prefix}{conn}{p.name}")
        if p.is_dir():
            child_prefix = prefix + ("    " if is_last else "│   ")
            child_tree = generate_tree(p, child_prefix)
            if child_tree:
                lines.append(child_tree)
    return "\n".join(lines)

def main():
    root = Path(__file__).resolve().parent.parent
    now = datetime.now()
    out_dir = root / 'scripts' / 'codes'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{now:%Y%m%d_%H%M%S}_{now.microsecond // 1000:03d}_export.txt"

    branch, commit, porcelain = cmd(["git", "branch", "--show-current"]) or "UNKNOWN", cmd(["git", "rev-parse", "HEAD"]) or "UNKNOWN", cmd(["git", "status", "--porcelain"])
    sep = "=" * 80

    header = (
        f"# 프로젝트 디렉터리\n#root\n{generate_tree(root)}\n\n\n"
        f"# PROJECT_METADATA\nProject: AMEVA-Forge\nSnapshot Date: {now:%Y-%m-%d}\nBranch: {branch}\nCommit: {commit}\n"
        f"Working Tree: {'DIRTY' if porcelain else 'CLEAN'}\nSnapshot State: {'HEAD + working tree snapshot' if porcelain else 'HEAD snapshot'}\n"
        f"Operating System: {sys.platform}\nPython Version: {sys.version.split()[0]}\nNode Version: {cmd(['node', '-v']) or 'Unknown'}\n"
        f"Package Manager: npm / pnpm\nCurrent Stage: Internal Alpha\nTarget Release: 0.1.0 Public Technical Preview\n\n\n"
    )

    with open(out_file, "w", encoding="utf-8") as out:
        out.write(header)
        for root_dir, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted([d for d in dirnames if d not in EXCLUDE_DIRS])
            for f in sorted(filenames):
                p = Path(root_dir) / f
                if any(p.match(pat) for pat in EXCLUDE_FILES) or f.endswith("_export.txt"):
                    continue
                rel_path = p.relative_to(root).as_posix()
                lang = LANG_MAP.get(p.suffix.lower(), "text")
                try:
                    content = p.read_text(encoding="utf-8")
                except Exception as e:
                    content = f"<에러 발생 또는 바이너리 파일: {e}>"
                out.write(
                    f"{sep}\nFILE_BEGIN\n{sep}\nFILE_NAME: {f}\nFILE_PATH: {rel_path}\n"
                    f"FILE_LANGUAGE: {lang}\n{sep}\nFILE_CONTENT_BEGIN\n{sep}\n\n"
                    f"{content}{'' if content.endswith(chr(10)) else chr(10)}\n"
                    f"{sep}\nFILE_CONTENT_END\n{sep}\nFILE_END: {rel_path}\n{sep}\n\n"
                )

    print(f"소스코드 추출이 완료되었습니다: {out_file}")

if __name__ == "__main__":
    main()
