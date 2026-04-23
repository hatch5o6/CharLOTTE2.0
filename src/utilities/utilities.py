import os

def set_env(path=".env"):
    assert os.path.exists(path), f"NO .env FILE FOUND"
    with open(path) as env_file:
        for line in env_file:
            line = line.strip()
            if "#" in line:
                line = line.split("#")[0].strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

def set_vars_in_path(f, env_path=".env"):
    set_env(env_path)
    return os.path.expandvars(f)
