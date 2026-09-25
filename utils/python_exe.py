"""Locate a real Python interpreter for launching subprocesses.

`sys.executable` is only the interpreter when Python itself is the host
process. Under uWSGI (PythonAnywhere, most WSGI deployments) it is the uWSGI
binary, so `[sys.executable, 'script.py']` asks uWSGI to load the script as a
config file and fails with "unable to load configuration from script.py".
"""
import functools
import os
import shutil
import sys


def _looks_like_python(path):
    name = os.path.basename(path or '').lower()
    return name.startswith('python') and os.path.isfile(path)


@functools.lru_cache(maxsize=1)
def python_executable():
    """Return a path to the Python interpreter matching the running process."""
    override = os.environ.get('GIS_PYTHON_EXECUTABLE')
    if override and os.path.isfile(override):
        return override
    if _looks_like_python(sys.executable):
        return sys.executable

    # Embedded: find the interpreter of the active virtualenv, then the base
    # install, preferring an exact major.minor match so installed packages line up.
    version = f'{sys.version_info.major}.{sys.version_info.minor}'
    names = [f'python{version}', f'python{sys.version_info.major}', 'python']
    if os.name == 'nt':
        names = ['python.exe']
    for prefix in dict.fromkeys([sys.prefix, sys.exec_prefix,
                                 getattr(sys, 'base_prefix', sys.prefix)]):
        for sub in ('bin', 'Scripts', ''):
            for name in names:
                candidate = os.path.join(prefix, sub, name)
                if _looks_like_python(candidate):
                    return candidate
    for name in (f'python{version}', 'python3', 'python'):
        found = shutil.which(name)
        if found:
            return found
    return sys.executable
