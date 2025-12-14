import os
import sys
import morl_baselines

def check_pip_editable()->bool:
    site_packages = [
        os.path.realpath(p)
        for p in sys.path
        if "site-packages" in p
    ]
    is_editable = True
    module_file = os.path.realpath(morl_baselines.__file__)
    for sp in site_packages:
        if module_file.startswith(sp):
            is_editable = False
            break
    return is_editable
    


if __name__ == "__main__":
    print(f"morl_baselines.__file__ = {morl_baselines.__file__}")
    sys.exit(0 if check_pip_editable() else 1)