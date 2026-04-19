"""VENDORED from /home/vegar/Documents/Fractal/evolution/stage11_introspection/patch_parser.py
Source commit: ec2bddd2ea4cdf83693ecd1f9af84d4fadd24687 (Fix Loop 3 protected-file bypass: use PROTECTED_PREFIXES directly)
Original author: Vegar Ratdal (Fractal project)
License: MIT (same ownership)
Modifications: none — file is used verbatim. Already supports allowed_files
               scope (line 119) and auto-skips AST validation for non-.py
               files (line 185), so YAML patches work unchanged.
"""

import ast
import os
import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CodePatch:
    """A single code change."""
    file_path: str        # relative to project root
    old_text: str         # exact text to find
    new_text: str         # replacement text
    n_lines_changed: int  # approximate line count delta

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "old_text_preview": self.old_text[:200],
            "new_text_preview": self.new_text[:200],
            "n_lines_changed": self.n_lines_changed,
        }


@dataclass
class ParseResult:
    """Result of parsing an LLM response."""
    patches: List[CodePatch]
    no_fix: bool
    no_fix_reason: Optional[str]
    parse_error: Optional[str]
    raw_response: str

    def to_dict(self) -> dict:
        return {
            "n_patches": len(self.patches),
            "no_fix": self.no_fix,
            "no_fix_reason": self.no_fix_reason,
            "parse_error": self.parse_error,
            "patches": [p.to_dict() for p in self.patches],
        }


class PatchParser:
    """Parse LLM response into CodePatch objects."""

    PATCH_PATTERN = re.compile(
        r'<<<PATCH\s*\n'
        r'file:\s*(.+?)\s*\n'
        r'---old---\s*\n'
        r'(.*?)\n'
        r'---new---\s*\n'
        r'(.*?)\n'
        r'>>>PATCH',
        re.DOTALL
    )

    def parse(self, response: str) -> ParseResult:
        """Parse LLM response into patches."""
        if "NO_FIX_FOUND" in response:
            reason_match = re.search(
                r'NO_FIX_FOUND:\s*(.+?)(?:\n|$)', response)
            return ParseResult(
                patches=[],
                no_fix=True,
                no_fix_reason=reason_match.group(1).strip()
                    if reason_match else "unknown",
                parse_error=None,
                raw_response=response,
            )

        matches = self.PATCH_PATTERN.findall(response)
        if not matches:
            return ParseResult(
                patches=[],
                no_fix=False,
                no_fix_reason=None,
                parse_error="No valid <<<PATCH blocks found in response",
                raw_response=response,
            )

        patches = []
        for file_path, old_text, new_text in matches:
            file_path = file_path.strip()
            old_text = old_text.strip()
            new_text = new_text.strip()

            n_old = len(old_text.splitlines())
            n_new = len(new_text.splitlines())
            n_changed = abs(n_new - n_old) + min(n_old, n_new)

            patches.append(CodePatch(
                file_path=file_path,
                old_text=old_text,
                new_text=new_text,
                n_lines_changed=n_changed,
            ))

        return ParseResult(
            patches=patches,
            no_fix=False,
            no_fix_reason=None,
            parse_error=None,
            raw_response=response,
        )

    def validate_patches(self, patches: List[CodePatch],
                         project_root: str,
                         max_total_lines: int = 50,
                         protected_files: List[str] = None,
                         allowed_files: List[str] = None,
                         ) -> List[str]:
        """Validate patches before applying.

        Returns list of error strings. Empty list = all valid.
        If allowed_files is set, patches targeting files outside
        the allowed+protected set are rejected.
        """
        errors = []
        total_lines = 0
        protected = protected_files or []
        allowed = allowed_files or []
        known_files = set(allowed) | set(protected)
        context_was_provided = (protected_files is not None
                                or allowed_files is not None)

        for i, patch in enumerate(patches):
            if context_was_provided and not known_files:
                errors.append(
                    f"Patch {i}: {patch.file_path} rejected — no "
                    f"allowed/protected file context available")
                continue
            if known_files and patch.file_path not in known_files:
                errors.append(
                    f"Patch {i}: {patch.file_path} is not in the "
                    f"allowed or protected file set")
                continue
            abs_path = os.path.join(project_root, patch.file_path)
            if not os.path.isfile(abs_path):
                errors.append(
                    f"Patch {i}: file not found: {patch.file_path}")
                continue

            try:
                with open(abs_path) as f:
                    content = f.read()
            except Exception as e:
                errors.append(
                    f"Patch {i}: cannot read {patch.file_path}: {e}")
                continue

            if patch.old_text not in content:
                errors.append(
                    f"Patch {i}: old_text not found in "
                    f"{patch.file_path} "
                    f"(first 80 chars: {patch.old_text[:80]!r})")
                continue

            count = content.count(patch.old_text)
            if count > 1:
                errors.append(
                    f"Patch {i}: old_text appears {count} times in "
                    f"{patch.file_path} — must be unique")
                continue

            new_content = content.replace(
                patch.old_text, patch.new_text, 1)
            if patch.file_path.endswith(".py"):
                try:
                    ast.parse(new_content)
                except SyntaxError as e:
                    errors.append(
                        f"Patch {i}: syntax error after applying to "
                        f"{patch.file_path}: {e}")
                    continue

            total_lines += patch.n_lines_changed

        if total_lines > max_total_lines:
            errors.append(
                f"Total lines changed ({total_lines}) exceeds "
                f"max ({max_total_lines})")

        return errors
