# Release Process

TorchDR uses automated releases via GitHub Actions. Versions are automatically extracted from git tags using `setuptools-scm`.

## Creating a New Release

### 1. Prepare the release

```bash
# Update RELEASES.rst with changelog for the new version
# Commit any final changes
git add RELEASES.rst
git commit -m "Prepare release v0.4.0"
git push
```

### 2. Create and push a version tag

```bash
# Create a new tag (use v prefix for consistency)
git tag v0.4.0

# Push the tag to trigger automatic release
git push origin v0.4.0
```

### 3. Automated workflow

Once the tag is pushed, GitHub Actions will automatically:
- Build the package (both wheel and source distribution)
- Extract version `0.4.0` from the tag `v0.4.0`
- Publish to PyPI via trusted publishing

### 4. Verify the release

- Check the [Actions tab](https://github.com/TorchDR/TorchDR/actions) for workflow status
- Once complete, verify on [PyPI](https://pypi.org/project/torchdr/)

## Version Numbering

TorchDR follows [Semantic Versioning](https://semver.org/):
- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, backward compatible

Supported tag formats (all work the same):
- `v0.4.0` ✅ (recommended)
- `0.4.0` ✅
- `v0.4` ✅
- `0.4` ✅

## Development Versions

Between releases, `setuptools-scm` automatically generates development versions:
- Format: `0.3.dev44+g0d06a62.d20251110`
  - `0.3`: Base version from latest tag
  - `dev44`: 44 commits since that tag
  - `g0d06a62`: Current commit hash
  - `d20251110`: Date

## PyPI Trusted Publishing Setup

The release workflow uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) which is more secure than API tokens.

To configure (one-time setup for new maintainers):
1. Go to [PyPI project settings](https://pypi.org/manage/project/torchdr/settings/publishing/)
2. Add a new "Trusted Publisher":
   - **Owner**: TorchDR
   - **Repository**: TorchDR
   - **Workflow**: release.yml
   - **Environment**: release

## Troubleshooting

### Tag already exists
```bash
# Delete local tag
git tag -d v0.4.0

# Delete remote tag
git push origin :refs/tags/v0.4.0

# Create new tag
git tag v0.4.0
git push origin v0.4.0
```

### Release failed
- Check GitHub Actions logs for errors
- Verify PyPI trusted publishing is configured
- Ensure the tag matches the version pattern

### Wrong version published
- Yank the bad release on PyPI (doesn't delete it)
- Fix the issue and create a new patch release
