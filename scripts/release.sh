#!/usr/bin/env bash
# Releases: one version number in every place that has one, release notes in CHANGELOG.md and a
# git tag named by the version (0.1.0, no "v"). Pushing the tag starts the Release workflow
# (.github/workflows/release.yml), which checks, packages and publishes it.
# Usage:
#   scripts/release.sh VERSION        set VERSION everywhere, add its notes (the commits since the
#                                     previous release tag) to CHANGELOG.md, commit and tag
#   scripts/release.sh check [TAG]    check that every version number agrees, and equals TAG
#   scripts/release.sh notes VERSION  print VERSION's release notes from CHANGELOG.md
#   scripts/release.sh build-id       print the release tag at HEAD, or the commit (-dirty)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TAG_PATTERN='[0-9]*.[0-9]*.[0-9]*'
CHANGELOG=CHANGELOG.md

die() { echo "$*" >&2; exit 1; }

# "file version" for each place that holds the version.
versions() {
    echo "engine/pyproject.toml $(sed -n 's/^version = "\(.*\)"/\1/p' engine/pyproject.toml)"
    echo "engine/src/engine/__init__.py $(sed -n 's/^__version__ = "\(.*\)"/\1/p' engine/src/engine/__init__.py)"
    python3 - <<'PY'
import json
package = json.load(open("desktop/package.json"))
lock = json.load(open("desktop/package-lock.json"))
print("desktop/package.json", package["version"])
print("desktop/package-lock.json", lock["version"])
print("desktop/package-lock.json:packages", lock["packages"][""]["version"])
PY
}

set_version() {
    local version=$1
    sed -i "s/^version = \".*\"/version = \"$version\"/" engine/pyproject.toml
    sed -i "s/^__version__ = \".*\"/__version__ = \"$version\"/" engine/src/engine/__init__.py
    # Written as npm writes them, so only the version lines change.
    python3 - "$version" <<'PY'
import json, sys
version = sys.argv[1]
def rewrite(path, change):
    with open(path) as f:
        data = json.load(f)
    change(data)
    with open(path, "w") as f:
        f.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
rewrite("desktop/package.json", lambda d: d.update(version=version))
def lock(d):
    d["version"] = version
    d["packages"][""]["version"] = version
rewrite("desktop/package-lock.json", lock)
PY
}

check() {
    local tag=${1:-} found
    found=$(versions | awk '{print $2}' | sort -u)
    if [[ $(wc -l <<<"$found") -ne 1 ]]; then
        versions >&2
        die "The version numbers differ: run scripts/release.sh VERSION (make release VERSION=...)."
    fi
    if [[ -n "$tag" && "$found" != "$tag" ]]; then
        die "The tag is $tag, but the version is $found: tag a commit made by make release VERSION=$tag."
    fi
    echo "$found"
}

# The previous release tag before HEAD, if any.
previous_tag() { git describe --tags --abbrev=0 --match "$TAG_PATTERN" HEAD 2>/dev/null || true; }

# The release notes of VERSION: the commits since the previous release tag.
write_notes() {
    local version=$1 previous range
    previous=$(previous_tag)
    range=${previous:+$previous..}HEAD
    echo "## $version ($(date +%F))"
    echo
    if [[ -n "$previous" ]]; then echo "Changes since $previous:"; else echo "First release."; fi
    echo
    git log --no-merges --format='- %s (%h)' "$range"
}

notes() {
    local version=$1
    [[ -f "$CHANGELOG" ]] || die "No $CHANGELOG."
    # From "## VERSION " to the next "## " heading, without the heading and the blank lines
    # around the text.
    awk -v v="$version" '
        /^## / { if (found) exit; if ($2 == v) { found = 1; next } }
        found && /^$/ { blank++; next }
        found { if (text) while (blank-- > 0) print ""; blank = 0; text = 1; print }
        END { if (!found) exit 1 }
    ' "$CHANGELOG" || die "No notes for $version in $CHANGELOG."
}

build_id() {
    local commit tag
    commit=$(git rev-parse --short HEAD 2>/dev/null) || { echo unknown; return; }
    if [[ -n "$(git status --porcelain)" ]]; then echo "$commit-dirty"; return; fi
    tag=$(git describe --tags --exact-match --match "$TAG_PATTERN" HEAD 2>/dev/null || true)
    echo "${tag:-$commit}"
}

release() {
    local version=$1 current previous body
    [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "VERSION must look like 0.1.0, not '$version'."
    [[ -z "$(git status --porcelain)" ]] || die "Commit or stash your changes first."
    git rev-parse -q --verify "refs/tags/$version" >/dev/null && die "Tag $version already exists."
    current=$(check)
    previous=$(previous_tag)
    if [[ -n "$previous" && "$(printf '%s\n' "$previous" "$version" | sort -V | tail -1)" != "$version" \
          || "$previous" == "$version" ]]; then
        die "$version is not newer than the previous release $previous."
    fi
    [[ -n "$(git log -1 --format=%h ${previous:+$previous..}HEAD)" ]] || die "No commits since $previous."

    body=$(write_notes "$version")
    set_version "$version"
    {
        echo "# Changelog"
        echo
        echo "Release notes of tinylm-studio: the commits of each release. \`make release\` adds them."
        echo
        echo "$body"
        if [[ -f "$CHANGELOG" ]]; then echo; sed '1,/^## /{/^## /!d}' "$CHANGELOG"; fi
    } > "$CHANGELOG.new"
    mv "$CHANGELOG.new" "$CHANGELOG"
    check "$version" >/dev/null

    git add engine/pyproject.toml engine/src/engine/__init__.py desktop/package.json \
        desktop/package-lock.json "$CHANGELOG"
    git commit -q -m "Release $version"
    git tag -a "$version" -m "tinylm-studio $version" -m "$(notes "$version")"
    echo "Released $version (was $current): commit $(git rev-parse --short HEAD), tag $version."
    echo "Publish it: git push origin HEAD $version"
}

case "${1:-}" in
    check) check "${2:-}" ;;
    notes) [[ -n "${2:-}" ]] || die "Usage: $0 notes VERSION"; notes "$2" ;;
    build-id) build_id ;;
    ""|-h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//' ;;
    *) release "$1" ;;
esac
