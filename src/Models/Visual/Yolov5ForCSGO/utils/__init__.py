
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
utils/initialization
"""
# print('utils/initialization.py')
from .activations import *
from .augmentations import *
from .autoanchor import *
from .autobatch import *
from .benchmarks import *
from .callbacks import *
from .dataloaders import *
from .downloads import * 
from .general import *
from .loss import *
from .metrics import *
from .plots import *
from .torch_utils import *



def notebook_init(verbose=True):
    # Check system software and hardware
    print('Checking setup...')

    import os
    import shutil

    from utils.general import check_requirements, emojis, is_colab
    from utils.torch_utils import select_device  # imports

    check_requirements(('psutil', 'IPython'))
    import psutil
    from IPython import display  # to display images and clear console output

    if is_colab():
        shutil.rmtree('/content/sample_data', ignore_errors=True)  # remove colab /sample_data directory

    # System info
    if verbose:
        gb = 1 << 30  # bytes to GiB (1024 ** 3)
        ram = psutil.virtual_memory().total
        total, used, free = shutil.disk_usage("/")
        display.clear_output()
        s = f'({os.cpu_count()} CPUs, {ram / gb:.1f} GB RAM, {(total - free) / gb:.1f}/{total / gb:.1f} GB disk)'
    else:
        s = ''

    select_device(newline=False)
    print(emojis(f'Setup complete ✅ {s}'))
    return display
