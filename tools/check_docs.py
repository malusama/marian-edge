#!/usr/bin/env python3
"""Check release, deployment, and HTTP-contract facts in project Markdown.

This intentionally uses only the Python standard library.  Run it from any
directory; paths are resolved relative to the repository containing this file.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
README_PATHS = (Path("README.md"), Path("README.zh-CN.md"))
IMMERSIVE_PATH = Path("docs/IMMERSIVE_TRANSLATE.md")
DEPLOYMENT_DOCS = (*README_PATHS, IMMERSIVE_PATH, Path("docs/OPERATIONS.md"))


class Checks:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def text(self, relative_path: Path | str) -> str:
        path = ROOT / relative_path
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            self.errors.append(f"{relative_path}: cannot read UTF-8 text: {error}")
            return ""


def workspace_version(checks: Checks) -> str:
    cargo = checks.text("Cargo.toml")
    section = re.search(
        r"(?ms)^\[workspace\.package\]\s*$\n(?P<body>.*?)(?=^\[|\Z)", cargo
    )
    if section is None:
        checks.errors.append("Cargo.toml: missing [workspace.package] section")
        return ""
    version = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', section["body"])
    if version is None:
        checks.errors.append("Cargo.toml: [workspace.package] has no version")
        return ""
    value = version.group(1)
    checks.require(
        re.fullmatch(r"\d+\.\d+\.\d+", value) is not None,
        f"Cargo.toml: workspace version {value!r} is not X.Y.Z",
    )
    return value


def fenced_code_blocks(markdown: str) -> list[str]:
    return re.findall(r"(?ms)^```[^\n]*\n(.*?)^```\s*$", markdown)


def check_release_references(checks: Checks, version: str) -> None:
    if not version:
        return
    version_re = re.escape(version)
    for relative_path in README_PATHS:
        text = checks.text(relative_path)
        blocks = fenced_code_blocks(text)
        pinned_blocks = [
            block
            for block in blocks
            if "raw.githubusercontent.com/malusama/marian-mlx/v" in block
            and "/scripts/install-macos.sh" in block
        ]
        checks.require(
            any(
                f"/v{version}/scripts/install-macos.sh" in block
                and f"MARIAN_MLX_VERSION=v{version}" in block
                for block in pinned_blocks
            ),
            f"{relative_path}: pinned installer block must use v{version} in both "
            "the download URL and MARIAN_MLX_VERSION",
        )
        installer_versions = set(
            re.findall(
                r"raw\.githubusercontent\.com/malusama/marian-mlx/v"
                r"(\d+\.\d+\.\d+)/scripts/install-macos\.sh",
                text,
            )
        )
        checks.require(
            installer_versions == {version},
            f"{relative_path}: pinned installer URL versions are "
            f"{sorted(installer_versions)!r}; expected only {version}",
        )
        environment_versions = set(
            re.findall(r"MARIAN_MLX_VERSION=v(\d+\.\d+\.\d+)", text)
        )
        checks.require(
            environment_versions == {version},
            f"{relative_path}: MARIAN_MLX_VERSION values are "
            f"{sorted(environment_versions)!r}; expected only {version}",
        )
        image_versions = set(
            re.findall(
                r"ghcr\.io/malusama/marian-mlx:cpu-(\d+\.\d+\.\d+)", text
            )
        )
        checks.require(
            image_versions == {version},
            f"{relative_path}: versioned CPU image tags are "
            f"{sorted(image_versions)!r}; expected only cpu-{version}",
        )

    for relative_path in DEPLOYMENT_DOCS:
        text = checks.text(relative_path)
        references = {
            "pinned installer URL": set(
                re.findall(
                    r"raw\.githubusercontent\.com/malusama/marian-mlx/v"
                    r"(\d+\.\d+\.\d+)/scripts/install-macos\.sh",
                    text,
                )
            ),
            "MARIAN_MLX_VERSION": set(
                re.findall(r"MARIAN_MLX_VERSION=v(\d+\.\d+\.\d+)", text)
            ),
            "versioned CPU image": set(
                re.findall(
                    r"ghcr\.io/malusama/marian-mlx:cpu-(\d+\.\d+\.\d+)",
                    text,
                )
            ),
        }
        for label, found in references.items():
            checks.require(
                not found or found == {version},
                f"{relative_path}: {label} versions are {sorted(found)!r}; "
                f"expected only {version}",
            )

    changelog = checks.text("CHANGELOG.md")
    checks.require(
        re.search(
            rf"(?m)^## \[{version_re}\] - \d{{4}}-\d{{2}}-\d{{2}}\s*$",
            changelog,
        )
        is not None,
        f"CHANGELOG.md: missing dated '## [{version}] - YYYY-MM-DD' heading",
    )
    checks.require(
        re.search(
            rf"(?m)^\[Unreleased\]: https://github\.com/malusama/marian-mlx/"
            rf"compare/v{version_re}\.\.\.HEAD\s*$",
            changelog,
        )
        is not None,
        f"CHANGELOG.md: [Unreleased] link must compare v{version}...HEAD",
    )
    checks.require(
        re.search(
            rf"(?m)^\[{version_re}\]: https://github\.com/malusama/marian-mlx/"
            rf"compare/v\d+\.\d+\.\d+\.\.\.v{version_re}\s*$",
            changelog,
        )
        is not None,
        f"CHANGELOG.md: [{version}] link must compare the prior tag to v{version}",
    )


def tracked_markdown(checks: Checks) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-z", "--", "*.md"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", b"").decode("utf-8", "replace").strip()
        checks.errors.append(
            "cannot enumerate tracked Markdown with git ls-files"
            + (f": {detail}" if detail else f": {error}")
        )
        return []
    return [Path(item.decode("utf-8")) for item in result.stdout.split(b"\0") if item]


def markdown_link_targets(markdown: str) -> list[tuple[int, str]]:
    targets: list[tuple[int, str]] = []
    patterns = (
        re.compile(r"!?\[[^\]\n]*\]\(([^)\n]+)\)"),
        re.compile(r"(?m)^\s*\[[^\]\n]+\]:\s*(\S+)"),
    )
    for pattern in patterns:
        for match in pattern.finditer(markdown):
            raw = match.group(1).strip()
            if raw.startswith("<") and ">" in raw:
                raw = raw[1 : raw.index(">")]
            else:
                raw = raw.split(maxsplit=1)[0]
            line = markdown.count("\n", 0, match.start()) + 1
            targets.append((line, raw))
    return targets


def relative_link_path(target: str) -> str | None:
    if not target or target.startswith(("#", "/", "//")):
        return None
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc:
        return None
    path = unquote(parsed.path).replace("\\ ", " ")
    return path or None


def github_heading_slugs(markdown: str) -> set[str]:
    slugs: set[str] = set()
    occurrences: dict[str, int] = {}
    for heading in re.findall(r"(?m)^#{1,6}\s+(.+?)\s*#*\s*$", markdown):
        plain = re.sub(r"<[^>]+>", "", heading)
        plain = re.sub(r"[`*_~]", "", plain).strip().lower()
        base = re.sub(r"[^\w\- ]", "", plain, flags=re.UNICODE)
        base = re.sub(r"\s+", "-", base)
        occurrence = occurrences.get(base, 0)
        occurrences[base] = occurrence + 1
        slugs.add(base if occurrence == 0 else f"{base}-{occurrence}")
    return slugs


def check_relative_links(checks: Checks) -> int:
    markdown_paths = tracked_markdown(checks)
    link_count = 0
    for relative_document in markdown_paths:
        markdown = checks.text(relative_document)
        for line, target in markdown_link_targets(markdown):
            relative_target = relative_link_path(target)
            if relative_target is None:
                continue
            link_count += 1
            destination = (ROOT / relative_document.parent / relative_target).resolve()
            try:
                destination.relative_to(ROOT.resolve())
            except ValueError:
                checks.errors.append(
                    f"{relative_document}:{line}: relative link escapes the repository: "
                    f"{target}"
                )
                continue
            checks.require(
                destination.exists(),
                f"{relative_document}:{line}: broken relative link {target!r} "
                f"(resolved to {destination})",
            )
            fragment = unquote(urlsplit(target).fragment)
            if destination.is_file() and destination.suffix.lower() == ".md" and fragment:
                try:
                    destination_markdown = destination.read_text(encoding="utf-8")
                except (OSError, UnicodeError) as error:
                    checks.errors.append(
                        f"{relative_document}:{line}: cannot inspect anchor in "
                        f"{destination}: {error}"
                    )
                    continue
                checks.require(
                    fragment in github_heading_slugs(destination_markdown),
                    f"{relative_document}:{line}: missing Markdown anchor "
                    f"#{fragment} in {destination.relative_to(ROOT)}",
                )
    return link_count


def check_fenced_code_blocks(checks: Checks) -> None:
    for relative_document in tracked_markdown(checks):
        opening: tuple[str, int, int] | None = None
        for line_number, line in enumerate(
            checks.text(relative_document).splitlines(), start=1
        ):
            match = re.match(r"^\s*(`{3,}|~{3,})(.*)$", line)
            if match is None:
                continue
            marker = match.group(1)
            if opening is None:
                opening = (marker[0], len(marker), line_number)
                continue
            character, width, _ = opening
            if (
                marker[0] == character
                and len(marker) >= width
                and not match.group(2).strip()
            ):
                opening = None
        if opening is not None:
            _, _, line_number = opening
            checks.errors.append(
                f"{relative_document}:{line_number}: unclosed fenced code block"
            )


def check_deployment_ports(checks: Checks) -> None:
    server = checks.text("crates/marian-server/src/main.rs")
    checks.require(
        re.search(
            r'MARIAN_MLX_BIND"[^\]]*default_value\s*=\s*"127\.0\.0\.1:3000"',
            server,
            re.DOTALL,
        )
        is not None,
        "crates/marian-server/src/main.rs: MARIAN_MLX_BIND must default to "
        "127.0.0.1:3000",
    )

    installer = checks.text("scripts/install-macos.sh")
    checks.require(
        re.search(r"(?m)^\s*PORT=3000\s*$", installer) is not None,
        "scripts/install-macos.sh: missing native installer default PORT=3000",
    )
    checks.require(
        'xml_escape "127.0.0.1:$PORT"' in installer,
        "scripts/install-macos.sh: generated LaunchAgent must bind 127.0.0.1:$PORT",
    )

    dockerfile = checks.text("Dockerfile")
    for expected, description in (
        ("MARIAN_MLX_BIND=0.0.0.0:3000", "container bind"),
        ("EXPOSE 3000", "exposed port"),
        ("http://127.0.0.1:3000/readyz", "healthcheck URL"),
    ):
        checks.require(
            expected in dockerfile,
            f"Dockerfile: {description} must use port 3000 ({expected!r})",
        )

    compose = checks.text("compose.yaml")
    expected_mapping = '"127.0.0.1:${MARIAN_MLX_HOST_PORT:-3000}:3000"'
    checks.require(
        expected_mapping in compose,
        "compose.yaml: ports must contain " + expected_mapping,
    )
    checks.require(
        '"http://127.0.0.1:3000/readyz"' in compose,
        "compose.yaml: container healthcheck must use 127.0.0.1:3000/readyz",
    )
    checks.require(
        '"${MARIAN_MLX_IMAGE:-ghcr.io/malusama/marian-mlx:cpu}"' in compose,
        "compose.yaml: image must be overrideable through MARIAN_MLX_IMAGE",
    )


def check_custom_port_contract(checks: Checks) -> None:
    for relative_path in (*README_PATHS, IMMERSIVE_PATH):
        text = checks.text(relative_path)
        native_blocks = [
            block
            for block in fenced_code_blocks(text)
            if (
                "MARIAN_MLX_PORT=3100" in block
                or (
                    re.search(r"(?m)^\s*PORT=3100\s*$", block) is not None
                    and re.search(
                        r'MARIAN_MLX_PORT=(?:"\$PORT"|\$PORT)(?![A-Za-z0-9_])',
                        block,
                    )
                    is not None
                )
            )
        ]
        checks.require(
            any(
                (
                    (
                        "127.0.0.1:$PORT/readyz" in block
                        and "127.0.0.1:$PORT/info" in block
                    )
                    or (
                        'SERVICE_ORIGIN="http://127.0.0.1:$PORT"' in block
                        and '$SERVICE_ORIGIN/readyz' in block
                        and '$SERVICE_ORIGIN/info' in block
                    )
                )
                and "127.0.0.1:3100/imme" in block
                and "127.0.0.1:3000/imme" not in block
                for block in native_blocks
            ),
            f"{relative_path}: one native 3100 code block must configure the "
            "installer and use that same port for readyz, info, and /imme",
        )

    for relative_path in (*README_PATHS, IMMERSIVE_PATH, Path("docs/OPERATIONS.md")):
        text = checks.text(relative_path)
        compose_blocks = [
            block
            for block in fenced_code_blocks(text)
            if "MARIAN_MLX_HOST_PORT=3100" in block
        ]
        checks.require(
            any(
                "http://127.0.0.1:3100/readyz" in block
                and "http://127.0.0.1:3100/imme" in block
                and "http://127.0.0.1:3000/imme" not in block
                for block in compose_blocks
            ),
            f"{relative_path}: one Compose 3100 code block must use host port "
            "3100 for both readyz and /imme",
        )


def source_routes(checks: Checks) -> set[tuple[str, str]]:
    source = checks.text("crates/marian-server/src/lib.rs")
    routes = {
        (method.upper(), path)
        for path, method in re.findall(
            r'\.route\(\s*"(/[^"]+)"\s*,\s*(get|post|put|patch|delete)\s*\(',
            source,
            re.IGNORECASE | re.DOTALL,
        )
    }
    checks.require(
        bool(routes),
        "crates/marian-server/src/lib.rs: could not extract any Axum routes",
    )
    return routes


def documented_routes(markdown: str) -> set[tuple[str, str]]:
    return {
        (method.upper(), path)
        for method, path in re.findall(
            r"(?m)^\|\s*`(GET|POST|PUT|PATCH|DELETE)\s+(/[^`\s]+)`\s*\|",
            markdown,
            re.IGNORECASE,
        )
    }


def format_routes(routes: set[tuple[str, str]]) -> str:
    return ", ".join(f"{method} {path}" for method, path in sorted(routes)) or "(none)"


def check_route_tables(checks: Checks) -> int:
    routes = source_routes(checks)
    for relative_path in README_PATHS:
        documented = documented_routes(checks.text(relative_path))
        missing = routes - documented
        extra = documented - routes
        checks.require(
            not missing and not extra,
            f"{relative_path}: API table differs from server routes; "
            f"missing [{format_routes(missing)}], extra [{format_routes(extra)}]",
        )
    return len(routes)


def check_immersive_contract(checks: Checks) -> None:
    source = checks.text("crates/marian-server/src/lib.rs")
    for source_fact, message in (
        ("const MAX_TEXT_BYTES: usize = 64 * 1024;", "64 KiB body limit"),
        ("request.text_list.len() > 256", "256-item /imme limit"),
        ("unwrap_or(512).clamp(1, 2_048)", "max_output_tokens contract"),
    ):
        checks.require(
            source_fact in source,
            f"crates/marian-server/src/lib.rs: missing {message} source fact",
        )
    for struct_name, expected_fields in (
        (
            "TranslateRequest",
            {"text", "from", "to", "max_output_tokens"},
        ),
        (
            "ImmersiveRequest",
            {"source_lang", "target_lang", "text_list"},
        ),
    ):
        match = re.search(
            rf"(?s)pub struct {struct_name}\s*\{{(?P<body>.*?)^\}}",
            source,
            re.MULTILINE,
        )
        fields = (
            set(re.findall(r"(?m)^\s*pub\s+([a-z_][a-z0-9_]*)\s*:", match["body"]))
            if match is not None
            else set()
        )
        checks.require(
            fields == expected_fields,
            f"crates/marian-server/src/lib.rs: {struct_name} fields are "
            f"{sorted(fields)!r}; expected {sorted(expected_fields)!r}",
        )
    checks.require(
        'alias = "source_lang"' in source and 'alias = "target_lang"' in source,
        "crates/marian-server/src/lib.rs: /translate language aliases changed",
    )

    for relative_path in (*README_PATHS, IMMERSIVE_PATH):
        document = checks.text(relative_path)
        for field in ("source_lang", "target_lang", "text_list"):
            checks.require(
                f'"{field}"' in document or f"`{field}`" in document,
                f"{relative_path}: request contract must document {field!r}",
            )
        checks.require(
            "64 KiB" in document,
            f"{relative_path}: must document the 64 KiB whole-JSON-body limit",
        )
        checks.require(
            re.search(
                r"(?:text_list.{0,120}\b256\b|\b256\b.{0,120}(?:items|项目))",
                document,
                re.IGNORECASE | re.DOTALL,
            )
            is not None,
            f"{relative_path}: must document the 256-item text_list limit",
        )
        english_scope = re.search(
            r"(?:whole|entire|complete|total|combined|aggregate)\s+JSON\s+"
            r"(?:request\s+)?body",
            document,
            re.IGNORECASE,
        )
        chinese_scope = re.search(
            r"(?:完整|整个|全部).{0,20}JSON.{0,20}(?:请求体|body)", document
        )
        checks.require(
            english_scope is not None or chinese_scope is not None,
            f"{relative_path}: 64 KiB scope must explicitly say the whole/entire "
            "JSON request body",
        )
        checks.require(
            "max_output_tokens" in document and re.search(r"\b512\b", document),
            f"{relative_path}: must document the 512-token output-budget contract",
        )


def check_controller_lifecycle(checks: Checks) -> None:
    expected = {
        "status",
        "logs",
        "restart",
        "stop",
        "start",
        "verify",
        "rollback",
        "update",
        "uninstall",
    }
    controller = checks.text("scripts/marian-mlxctl")
    commands = set(re.findall(r"(?m)^  ([a-z][a-z-]*)\)\s*$", controller))
    checks.require(
        expected <= commands,
        "scripts/marian-mlxctl: expected lifecycle commands are missing: "
        + ", ".join(sorted(expected - commands)),
    )
    for relative_path in (*README_PATHS, Path("docs/OPERATIONS.md")):
        document = checks.text(relative_path)
        for command in expected:
            checks.require(
                f"marian-mlxctl {command}" in document,
                f"{relative_path}: installed-service lifecycle omits "
                f"'marian-mlxctl {command}'",
            )


def check_architecture_contracts(checks: Checks, version: str) -> None:
    cargo = checks.text("Cargo.toml")
    checks.require(
        '"crates/marian-metal"' in cargo
        and f'marian-metal = {{ version = "{version}", path = "crates/marian-metal" }}'
        in cargo,
        "Cargo.toml: marian-metal package and source path must share one name",
    )
    metal_cargo = checks.text("crates/marian-metal/Cargo.toml")
    checks.require(
        'name = "marian-metal"' in metal_cargo,
        "crates/marian-metal/Cargo.toml: package must be named marian-metal",
    )
    for relative_path in (
        Path("CONTRIBUTING.md"),
        Path("docs/ARCHITECTURE.md"),
        Path(".github/workflows/ci.yml"),
        Path(".github/workflows/release-macos.yml"),
    ):
        document = checks.text(relative_path)
        checks.require(
            "-p marian-mlx" not in document and "`marian-mlx`" not in document,
            f"{relative_path}: stale internal marian-mlx Cargo package name",
        )

    manifest = checks.text("crates/marian-model/src/manifest.rs")
    checks.require(
        'MODEL_FORMAT_V1: &str = "marian-edge.transformer-ssru.v1"' in manifest
        and 'LEGACY_MODEL_FORMAT_V1: &str = "marian-mlx.transformer-ssru.v1"'
        in manifest,
        "marian-model: canonical and legacy manifest namespaces changed",
    )
    converter = checks.text("tools/convert_marian.py")
    checks.require(
        'MANIFEST_FORMAT = "marian-edge.transformer-ssru.v1"' in converter
        and 'WEIGHTS_FORMAT = "marian-mlx.transformer-ssru.v1"' in converter,
        "tools/convert_marian.py: manifest namespace and byte-stable weights "
        "metadata must remain separate",
    )
    checks.require(
        '"format": "marian-edge.transformer-ssru.v1"'
        in checks.text("docker/prepare-model.sh"),
        "docker/prepare-model.sh: new manifests must use marian-edge v1",
    )
    checks.require(
        '"marian-edge\\.transformer-ssru\\.v1"'
        in checks.text("scripts/prepare-enzh-model.sh"),
        "scripts/prepare-enzh-model.sh: converted manifest must use canonical format",
    )
    for relative_path in ("scripts/install-macos.sh", "scripts/marian-mlxctl"):
        script = checks.text(relative_path)
        checks.require(
            "(marian-edge|marian-mlx)\\.transformer-ssru\\.v1" in script,
            f"{relative_path}: must accept canonical and historical manifests",
        )

    model_lib = checks.text("crates/marian-model/src/lib.rs")
    cpu_limits = checks.text("crates/marian-cpu/src/limits.rs")
    cpu_engines = (
        checks.text("crates/marian-cpu/src/engine.rs"),
        checks.text("crates/marian-cpu/src/q8_engine.rs"),
    )
    checks.require(
        "CPU_MAXIMUM_" not in model_lib
        and "MAXIMUM_SOURCE_TOKENS" in cpu_limits
        and "MAXIMUM_GENERATION_STEPS" in cpu_limits
        and "MAXIMUM_ENGINE_BATCH" in cpu_limits
        and all("MAXIMUM_ENGINE_BATCH" in engine for engine in cpu_engines)
        and all("const MAXIMUM_BATCH" not in engine for engine in cpu_engines),
        "CPU execution limits must remain private to marian-cpu",
    )

    metal_config = checks.text("crates/marian-metal/src/config.rs")
    checks.require(
        'format!("MARIAN_MLX_{suffix}")' in metal_config
        and 'format!("MARIAN_EDGE_{suffix}")' in metal_config
        and "conflicting settings" in metal_config,
        "Metal tuning must keep MARIAN_MLX canonical, accept MARIAN_EDGE alias, "
        "and reject conflicts",
    )

    checks.require(
        re.search(r'(?m)^panic\s*=\s*"unwind"\s*$', cargo) is not None
        and "backend_panic_stops_readiness_and_future_admission"
        in checks.text("crates/marian-core/src/scheduler.rs"),
        "release panic unwinding requires the backend readiness regression",
    )


def main() -> int:
    checks = Checks()
    version = workspace_version(checks)
    check_release_references(checks, version)
    link_count = check_relative_links(checks)
    check_fenced_code_blocks(checks)
    check_deployment_ports(checks)
    check_custom_port_contract(checks)
    route_count = check_route_tables(checks)
    check_immersive_contract(checks)
    check_controller_lifecycle(checks)
    check_architecture_contracts(checks, version)

    if checks.errors:
        print(f"documentation checks failed ({len(checks.errors)} issue(s)):", file=sys.stderr)
        for error in checks.errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        f"documentation checks passed: version {version}, {route_count} routes, "
        f"{link_count} relative links"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
