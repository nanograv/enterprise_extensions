# enterprise_extensions Release Process

**Status**: Semi-automated (requires manual version update + GitHub release)

## Prerequisites
- Maintainer access to [nanograv/enterprise_extensions](https://github.com/nanograv/enterprise_extensions)
- GitHub environment `pypi` configured with trusted publishing
- `id-token: write` permission enabled

## Release Steps

### 1. Update Version Number
```bash
# Edit enterprise_extensions/__init__.py
# Change: __version__ = "3.0.2"
# To:     __version__ = "3.0.3"
```

### 2. Update Release Notes
```bash
# Edit HISTORY.rst to add new version entry
# Add release date and changes
# Example format:
# 3.0.3 (2024-01-15)
# Fix bug in HyperModel parameter handling.
# Add support for new noise definitions.
```

### 3. Prepare Release
```bash
# Ensure main branch is up to date
git checkout main
git pull origin main

# Run tests locally
make test
make lint

# Check that all CI tests are passing on GitHub
# Visit: https://github.com/nanograv/enterprise_extensions/actions
```

### 4. Create GitHub Release
1. Go to [enterprise_extensions releases](https://github.com/nanograv/enterprise_extensions/releases)
2. Click "Create a new release"
3. Choose a tag version (e.g., `v3.0.3`)
4. Set release title (e.g., `v3.0.3`)
5. Add release notes describing changes
6. Click "Publish release"

### 5. Automated Process
- GitHub Actions will automatically:
  - Run tests across Python 3.9-3.12
  - Build source distribution and wheel
  - Test deployability with `twine check`
  - Upload to PyPI using trusted publishing

## Version Management
- **Manual version management** in `enterprise_extensions/__init__.py`
- Must update version number before each release
- Follow semantic versioning (major.minor.patch)
- Update both `__init__.py` and `HISTORY.rst`

## Troubleshooting
- **Upload fails**: Check GitHub environment permissions and trusted publishing setup
- **Version mismatch**: Ensure version in `__init__.py` matches git tag
- **Tests fail**: Ensure all dependencies are installed locally before release

## Release Checklist
- [ ] Version number updated in `__init__.py`
- [ ] Release notes added to `HISTORY.rst`
- [ ] All tests passing on GitHub Actions
- [ ] GitHub release created
- [ ] PyPI upload successful (check [PyPI page](https://pypi.org/project/enterprise-extensions/))

## Example Release Commands
```bash
# Update version in __init__.py first
# Then tag a new version
git tag v3.0.3
git push origin v3.0.3

# Then create GitHub release via web interface
```

## Trusted Publishing
This package uses GitHub's trusted publishing feature, which means:
- No API tokens or passwords needed
- Uses GitHub's OIDC (OpenID Connect) for authentication
- More secure than traditional API key authentication
- Configured in GitHub repository settings under "Environments"
