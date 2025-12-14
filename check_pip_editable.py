import os
import sys
import morl_baselines

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

print(f"morl_baselines.__file__ = {module_file}")

sys.exit(0 if is_editable else 1)