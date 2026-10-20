# Releasing

A release is a git tag named by the version, e.g. `0.1.0` (no `v`). Pushing the tag to GitHub
(`github.com/redacid/tinylm-studio`) runs the checks, builds the `.deb` and `.rpm` packages and
publishes them as a GitHub release. Engine container images (cpu and cuda) are built and pushed for the tag as well.

## Making a release

```bash
make release VERSION=0.2.0       # on a clean main: set the version, write the notes, commit, tag
git push origin main 0.2.0       # the tag starts the Release workflow
```

`make release` (`scripts/release.sh`):

1. refuses to run with uncommitted changes, when the tag exists, when the version is not newer
   than the previous release tag, or when there are no commits since it;
2. sets the version everywhere it is written: `engine/pyproject.toml`,
   `engine/src/engine/__init__.py`, `desktop/package.json` and `desktop/package-lock.json`;
3. adds the release notes to the top of [CHANGELOG.md](../CHANGELOG.md): a `## 0.2.0 (date)`
   section listing the commits since the previous release tag (all commits for the first one);
4. commits this as `Release 0.2.0` and creates the annotated tag `0.2.0` with the same notes.

Nothing is pushed: check the commit (`git show`), edit the notes if you like
(`git commit --amend`, then `git tag -f -a 0.2.0 -m ...`), and push the branch and the tag.
To undo before pushing: `git tag -d 0.2.0 && git reset --hard HEAD~1`.

Other targets:

```bash
make version                     # the version, and whether builds are named by a tag or a commit
make version-check               # every version number agrees (and equals the tag at HEAD)
make release-notes VERSION=0.1.0 # a release's notes from CHANGELOG.md
```

## What builds are named by

The same rule holds on GitHub and on your computer:

| Checkout | Packages (`make deb`, `make rpm`, …) | Engine image (`make image`) |
|---|---|---|
| A release tag (`git checkout 0.2.0`), no uncommitted changes | `tinylm-studio-0.2.0-amd64.deb` | `tinylm-engine:0.2.0-cuda` |
| A branch or any other commit | `tinylm-studio-0.2.0-5a0d901-amd64.deb` | `tinylm-engine:0.2.0-5a0d901-cuda` |
| Uncommitted changes | `tinylm-studio-0.2.0-5a0d901-dirty-amd64.deb` | `tinylm-engine:0.2.0-5a0d901-dirty-cuda` |

The image also gets the moving tag `<variant>` (`tinylm-engine:cuda`). A build at a release tag
fails if the tag is not the version in the code (a tag not made by `make release`). The commit is
still recorded in every build: the package's `build-commit` file, the image's
`org.opencontainers.image.revision` label, and the engine's `GET /health` and log.

## GitHub Actions

Three workflows in `.github/workflows/`:

- **CI** (`ci.yml`): on pushes to `main`, on pull requests and as the first step of a release.
  Engine: version numbers agree, ruff, pytest (CPU PyTorch and llama.cpp), and the desktop app's
  generated engine API types are up to date (`make api-types` changes nothing). Desktop app:
  TypeScript type check and build.
- **Release** (`release.yml`): on a pushed tag like `0.2.0`. Runs CI, checks the tag is the
  version in the code, builds the `.deb` and `.rpm` (`make package PACKAGES="deb rpm"`) and
  creates the GitHub release `tinylm-studio 0.2.0` with the tag's notes from CHANGELOG.md and
  both packages attached.
- **Engine image** (`images.yml`): on a pushed tag like `0.2.0`, both variants; also by hand:
  in GitHub, Actions › Engine image › Run workflow, pick the release tag (or a branch) under *Use
  workflow from* and the variant (`cpu`, `cuda` or both). To build images only by hand, remove
  the `push: tags:` trigger at the top of the file (a comment there says so). It runs `make image-push` and `make base-push` with
  `IMAGE=ghcr.io/<owner>/tinylm-engine`, so the image is named as in the table above. The base
  image (llama.cpp, PyTorch) is pulled from the registry when it exists for the current
  dependencies, otherwise built first; the CUDA base takes hours on a GitHub runner (the job may
  run up to 6 hours) and needs most of its disk, which the workflow frees first.

The images go to the GitHub Container Registry (`ghcr.io`), which the workflow logs in to with its
own token. A new package there is private until you make it public in its settings on GitHub;
private packages count against the account's storage, and the CUDA images are several GB each.
To pull them on RunPod or in a cluster, make the package public or give the puller a token.
