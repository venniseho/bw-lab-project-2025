#!/bin/bash
# Isolated additions, with the working SAM3 environment as a read-only base.
set -euo pipefail
module load python/3.12.4
module load opencv/4.12.0
base=/scratch/vho/sam3-whole-scene/env
destination=/scratch/vho/sam3-whole-scene/train-env
if [[ ! -e "$destination" ]]; then
"$base/bin/python" -m venv "$destination"
"$base/bin/python" - "$destination" <<'PY'
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import sysconfig
destination = Path(sys.argv[1])
site = subprocess.check_output([str(destination/'bin/python'), '-c',
                               'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip()
Path(site, 'bwlab_working_environment.pth').write_text(
    'import site; site.addsitedir('+repr(sysconfig.get_path('purelib'))+')\n')
packages = ('torch', 'torchvision', 'numpy', 'pillow', 'timm')
pins = {name: version(name) for name in packages}
(destination/'working_stack_constraints.txt').write_text(''.join(f'{name}=={value}\n' for name, value in pins.items()))
(destination/'base_versions.json').write_text(json.dumps(pins, indent=2)+'\n')
print('Base versions:', pins)
PY
fi
[[ -f "$destination/base_versions.json" && -f "$destination/working_stack_constraints.txt" ]] || {
    echo "Existing directory is not a BWLab training environment: $destination" >&2; exit 2;
}
source "$destination/bin/activate"
python -m pip install -c "$destination/working_stack_constraints.txt" \
    hydra-core==1.3.2 omegaconf==2.3.0 submitit tensorboard zstandard scipy torchmetrics fvcore fairscale \
    scikit-image scikit-learn wandb psutil pycocotools pyyaml
python -m pip freeze > "$destination/installed_packages.txt"
python - <<'PY'
from importlib.metadata import version
import json
from pathlib import Path
import sys
pins = json.loads((Path(sys.prefix)/'base_versions.json').read_text())
assert all(version(name) == value for name, value in pins.items()), 'Working stack versions changed'
import torch
from sam3.sam.prompt_encoder import PromptEncoder
from sam3.sam.mask_decoder import MaskDecoder
from sam3.sam.transformer import TwoWayTransformer
print('Native interactive training imports available; torch:', torch.__version__)
PY
echo "Training environment ready: $destination"
