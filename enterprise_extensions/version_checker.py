import ast
import sys
import importlib
import importlib.metadata
import inspect
import hashlib
import os
from typing import Set, Dict, List, Any

def find_imports_in_file(path: str) -> Set[str]:
    """
    Parse a .py file and return the set of top-level package names it imports.

    :param path: Path to the Python file to analyze.
    :type path: str
    :returns: Set of top-level imported package names.
    :rtype: Set[str]
    """
    with open(path, 'r') as f:
        tree = ast.parse(f.read(), filename=path)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
    return imports

def _hash_path(path: str, chunk_size: int = 8192) -> str:
    """
    Compute SHA256 of a file or, for a package directory, of each contained file in sorted order.

    :param path: File or directory path to hash.
    :type path: str
    :param chunk_size: Number of bytes to read per chunk.
    :type chunk_size: int
    :returns: Hexadecimal SHA256 digest string.
    :rtype: str
    """
    h = hashlib.sha256()
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for fname in sorted(files):
                full = os.path.join(root, fname)
                with open(full, 'rb') as f:
                    while chunk := f.read(chunk_size):
                        h.update(chunk)
    else:
        with open(path, 'rb') as f:
            while chunk := f.read(chunk_size):
                h.update(chunk)
    return h.hexdigest()

def get_module_report(name: str) -> Dict[str, Any]:
    """
    Given a top-level module name, return a report dictionary with metadata.

    :param name: Top-level module or package name.
    :type name: str
    :returns: A dictionary containing:
        - name (str): module name
        - version (str): detected version or 'unknown'
        - path (Optional[str]): filesystem path of module source
        - sha256 (Optional[str]): SHA256 hash of the source path
    :rtype: Dict[str, Any]
    """
    # 1) version
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        mod = sys.modules.get(name) or importlib.import_module(name)
        version = getattr(mod, '__version__', 'unknown')
    # 2) path & hash
    try:
        mod = sys.modules.get(name) or importlib.import_module(name)
        src = inspect.getfile(mod)
        digest = _hash_path(src)
    except Exception:
        src, digest = None, None

    return {
        'name':    name,
        'version': version,
        'path':    src,
        'sha256':  digest,
    }

def report_imports(path_to_script: str) -> List[Dict[str, Any]]:
    """
    Scan a script for its imports, then return a list of module reports.

    :param path_to_script: Path to the Python script to scan.
    :type path_to_script: str
    :returns: List of module report dictionaries or error info.
    :rtype: List[Dict[str, Any]]
    """
    names = find_imports_in_file(path_to_script)
    reports = []
    for nm in sorted(names):
        try:
            r = get_module_report(nm)
        except Exception as e:
            r = {'name': nm, 'error': str(e)}
        reports.append(r)
    return reports
