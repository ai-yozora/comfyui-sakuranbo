#!/usr/bin/env python3
"""Keep every commit, tag, and file in this repository to one identity.

Rules:
- Every commit is authored and committed as ALLOWED_NAME <ALLOWED_EMAIL>, and every
  annotated tag is tagged as the same identity.
- Commits and tags are not signed (a signature names its key's owner).
- Commit messages, tag messages, file contents, and file paths contain no email address
  other than ALLOWED_EMAIL, no co-author trailer, and no term in tools/denylist.json.
  The denylist holds SHA-256 digests, so the terms never appear in the repository.

Problems are reported by location only, never by the offending value, because CI logs
of a public repository are public.

The git hooks in .githooks run this before a commit and before a push, and CI runs it on
every push and pull request. Run `git config core.hooksPath .githooks` once per clone.

Usage:
  provenance_check.py pre-commit        the identity git will use, and the staged files
  provenance_check.py commit-msg FILE   a commit message
  provenance_check.py pre-push          what a push sends (reads the hook's stdin)
  provenance_check.py all               every commit, tag, and file in every ref
  provenance_check.py add-term          add the term read from stdin to the denylist
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ALLOWED_NAME = b"ai-yozora"
ALLOWED_EMAIL = b"support@sakuranbo.ai"
ALLOWED = f"{ALLOWED_NAME.decode()} <{ALLOWED_EMAIL.decode()}>"

ROOT = Path(__file__).resolve().parent.parent
DENYLIST = ROOT / "tools" / "denylist.json"

EMAIL = re.compile(rb"[a-z0-9._%+-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}")
CO_AUTHOR = re.compile(rb"^[ \t]*co-authored-by[ \t]*:", re.MULTILINE)
IDENT = re.compile(rb"^(.*) <(.*)> -?\d+ [+-]\d{4}$")
SIGNATURE = re.compile(rb"-----begin [a-z ]*signature-----")
SCISSORS = b"# ------------------------ >8 ------------------------"


def git(*args: str, stdin: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=ROOT, input=stdin, capture_output=True, check=True
    ).stdout


def line_of(data: bytes, offset: int) -> int:
    return data.count(b"\n", 0, offset) + 1


class Checker:
    def __init__(self) -> None:
        raw = json.loads(DENYLIST.read_text(encoding="utf-8"))["sha256"]
        self.denylist = {int(n): {bytes.fromhex(h) for h in hs} for n, hs in raw.items()}
        self.problems: list[str] = []
        self.counts = {"commits": 0, "tags": 0, "files": 0}

    def report(self, problem: str) -> None:
        if problem not in self.problems:
            self.problems.append(problem)

    def identity(self, where: str, role: str, value: bytes) -> None:
        match = IDENT.match(value.strip())
        if not match or match.group(1) != ALLOWED_NAME or match.group(2) != ALLOWED_EMAIL:
            self.report(f"{where}: the {role} is not {ALLOWED}")

    def text(self, where: str, data: bytes, *, message: bool = False) -> None:
        lowered = data.lower()
        for match in EMAIL.finditer(lowered):
            if match.group() != ALLOWED_EMAIL:
                line = line_of(lowered, match.start())
                self.report(f"{where}: line {line} has an email address other than {ALLOWED_EMAIL.decode()}")
        if message:
            match = CO_AUTHOR.search(lowered)
            if match:
                self.report(f"{where}: line {line_of(lowered, match.start())} is a co-author trailer")
        for length, digests in self.denylist.items():
            for start in range(len(lowered) - length + 1):
                if hashlib.sha256(lowered[start : start + length]).digest() in digests:
                    self.report(f"{where}: line {line_of(lowered, start)} has a denied term")

    def headers(self, where: str, header: bytes) -> None:
        roles = (b"author", b"committer", b"tagger")
        for line in header.split(b"\n"):
            key, _, value = line.strip().partition(b" ")
            if key in roles:
                self.identity(where, key.decode(), value)
            elif key.startswith(b"gpgsig"):
                self.report(f"{where}: is signed")

    def commit(self, oid: str) -> None:
        self.counts["commits"] += 1
        header, _, message = git("cat-file", "commit", oid).partition(b"\n\n")
        where = f"commit {oid[:12]}"
        self.headers(where, header)
        self.text(f"{where} message", message, message=True)

    def tag(self, oid: str) -> None:
        self.counts["tags"] += 1
        header, _, message = git("cat-file", "tag", oid).partition(b"\n\n")
        where = f"tag {oid[:12]}"
        self.headers(where, header)
        if SIGNATURE.search(message.lower()):
            self.report(f"{where}: is signed")
        self.text(f"{where} message", message, message=True)

    def blobs(self, entries: list[tuple[str, bytes]]) -> None:
        """Check each (blob oid, path): the path and the contents."""
        if not entries:
            return
        out = git("cat-file", "--batch", stdin=b"".join(oid.encode() + b"\n" for oid, _ in entries))
        pos = 0
        for _, path in entries:
            end = out.index(b"\n", pos)
            size = int(out[pos:end].split()[2])
            content = out[end + 1 : end + 1 + size]
            pos = end + 1 + size + 1
            shown = path.decode("utf-8", "replace")
            self.counts["files"] += 1
            self.text(f"path {shown}", path)
            self.text(f"file {shown}", content)

    def history(self, revs: list[str]) -> None:
        """Check every commit, tree path, and blob reachable from revs."""
        for oid in git("rev-list", *revs).split():
            self.commit(oid.decode())
        listed = [line.split(b" ", 1) for line in git("rev-list", "--objects", *revs).splitlines()]
        named = [(parts[0].decode(), parts[1]) for parts in listed if len(parts) == 2]
        if not named:
            return
        kinds = git("cat-file", "--batch-check=%(objecttype)", stdin=b"".join(o.encode() + b"\n" for o, _ in named))
        entries = []
        for (oid, path), kind in zip(named, kinds.split(), strict=True):
            if kind == b"blob":
                entries.append((oid, path))
            else:
                self.text(f"path {path.decode('utf-8', 'replace')}", path)
        self.blobs(entries)

    def peel_tags(self, oid: str) -> None:
        while git("cat-file", "-t", oid).strip() == b"tag":
            self.tag(oid)
            header = git("cat-file", "tag", oid).partition(b"\n\n")[0]
            oid = header.split(b"\n", 1)[0].split(b" ", 1)[1].decode()

    def finish(self, action: str) -> int:
        if self.problems:
            for problem in self.problems:
                print(f"provenance: {problem}", file=sys.stderr)
            print(f"provenance: {len(self.problems)} problem(s). {action}", file=sys.stderr)
            return 1
        summary = ", ".join(f"{n} {k}" for k, n in self.counts.items() if n)
        print(f"provenance: ok ({summary or 'nothing to check'})")
        return 0


def pre_commit(checker: Checker) -> int:
    for var, role in (("GIT_AUTHOR_IDENT", "author"), ("GIT_COMMITTER_IDENT", "committer")):
        checker.identity("this commit", role, git("var", var))
    entries = []
    for record in git("ls-files", "-s", "-z").split(b"\0"):
        if not record:
            continue
        meta, _, path = record.partition(b"\t")
        mode, oid, _ = meta.split(b" ")
        if mode != b"160000":
            entries.append((oid.decode(), path))
    checker.blobs(entries)
    return checker.finish("The commit was not made.")


def commit_msg(checker: Checker, path: str) -> int:
    message = Path(path).read_bytes().split(SCISSORS, 1)[0]
    kept = b"\n".join(line for line in message.split(b"\n") if not line.startswith(b"#"))
    checker.text("commit message", kept, message=True)
    return checker.finish("The commit was not made.")


def pre_push(checker: Checker) -> int:
    revs = []
    for line in sys.stdin.read().splitlines():
        parts = line.split()
        if len(parts) == 4 and set(parts[1]) != {"0"}:
            checker.peel_tags(parts[1])
            revs.append(parts[1])
    if revs:
        checker.history(revs)
    return checker.finish("Nothing was pushed.")


def check_all(checker: Checker) -> int:
    for line in git("for-each-ref", "--format=%(objecttype) %(objectname)", "refs/tags").splitlines():
        kind, oid = line.decode().split()
        if kind == "tag":
            checker.peel_tags(oid)
    checker.history(["--all"])
    return checker.finish("Fix the history before it is merged or published.")


def add_term() -> int:
    term = sys.stdin.read().strip().lower().encode()
    if not term:
        print("provenance: no term on stdin", file=sys.stderr)
        return 1
    data = json.loads(DENYLIST.read_text(encoding="utf-8"))
    digests = set(data["sha256"].get(str(len(term)), []))
    digests.add(hashlib.sha256(term).hexdigest())
    data["sha256"][str(len(term))] = sorted(digests)
    data["sha256"] = dict(sorted(data["sha256"].items(), key=lambda item: int(item[0])))
    DENYLIST.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"provenance: added a term of {len(term)} characters")
    return 0


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else ""
    if command == "add-term":
        return add_term()
    checker = Checker()
    if command == "pre-commit":
        return pre_commit(checker)
    if command == "commit-msg" and len(argv) == 3:
        return commit_msg(checker, argv[2])
    if command == "pre-push":
        return pre_push(checker)
    if command == "all":
        return check_all(checker)
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
