# Planning from Point Clouds over Continuous Actions for Multi-object Rearrangement

## Table of Contents

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

   * [Installation](#installation)
      + [More Installation Steps (updated 01.05.2025)](#more-installation-steps-updated-01052025)
- [Quick Start](#quick-start)
   * [Download pre-trained weights and all data](#download-pre-trained-weights-and-all-data)
   * [Run demo script](#run-demo-script)
- [Training](#training)
   * [Training the placement suggester](#training-the-placement-suggester)
   * [Training the object suggester](#training-the-object-suggester)
   * [Training the model deviation estimator](#training-the-model-deviation-estimator)
- [Planning](#planning)
   * [Run planning](#run-planning)
   * [Run execution](#run-execution)

<!-- TOC end -->


<!-- TOC --><a name="installation"></a>
## Installation

First, we will need to create the virtual environment with conda and install all the dependencies to use pytorch3D

```bash
conda create -n vtamp python=3.9
conda activate vtamp
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath

# # If you find any problem installing fvcore try using
# pip install -U fvcore

conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```

Then, we can install the package itself:

```bash
pip install -e ".[develop,notebook]"
```

Then we install pre-commit hooks:

```bash
pre-commit install
```

<!-- TOC --><a name="more-installation-steps-updated-01052025"></a>
### More Installation Steps (updated 01.05.2025)

After running the above, you may need some or all of the below steps to fix some dependencies.

**For training a TAXPoseD model:**

1. Install [pytorch3d from source](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md). The command might be:

    ```
    conda install pytorch3d -c pytorch3d
    ```

1. Make sure you have cuda 11.7, and then add this to your `.bashrc`:

    ```bash
    export CUDA_HOME=/usr/local/cuda-11.7
    export PATH=/usr/local/cuda-11.7/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
    ```

1. Make sure pytorch build cuda version matches cuda version (11.7), e.g.

    ```bash
    conda install pytorch=2.0.1 torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
    ```

1. Missing `fsspec`

    ```
    conda install fsspec
    ```

1. Error: `ImportError: cannot import name 'get_full_repo_name' from 'huggingface_hub' (unknown location)`

    ```
    pip install huggingface-hub==0.24.7
    ```

1. Error: `AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'`

    ```
    pip install transformers==4.28.0
    ```

1. Error: `ModuleNotFoundError: No module named 'vnn.models'`

    ```bash
    # Clone into the 3DVTAMP/vnn folder
    git clone https://github.com/FlyingGiraffe/vnn.git
    ```

1. Missing `pytorch-cluster`

    ```
    pip install torch-cluster==1.6.3
    ```

**For running inference (`evaluate_planning.py`):**

1. Other miscellaneous missing packages:

    ```
    pip install pandas==2.2.2
    conda install pytz=2024.1
    pip install segment_anything
    ```

1. The package `graphviz` may already be pip installed but you may need this too:

    ```
    sudo apt-get install graphviz
    ```

1. Make sure this submodule is there for the MDE:

    ```
    git submodule update --init --remote vtamp/submodules/pyg_libs
    cd vtamp/submodules/pyg_libs
    pip install -e .
    ```

1. Downgrade torch_geometric for MDE:

    ```
    pip install torch-geometric==2.2.0
    ```


<!-- TOC --><a name="quick-start"></a>
# Quick Start

<!-- TOC --><a name="download-pre-trained-weights-and-all-data"></a>
## Download pre-trained weights and all data
Download this folder and extract it into `assets/` under the root directory: https://drive.google.com/drive/folders/1tj9ycY1Lqy2z4rvYoq4KafiOYKGMj-PJ?usp=drive_link.

<!-- TOC --><a name="run-demo-script"></a>
## Run demo script
```
python demo.py
```

<!-- TOC --><a name="training"></a>
# Training

<!-- TOC --><a name="training-the-placement-suggester"></a>
## Training the placement suggester

```
python scripts/training/train_suggesters.py
```

<!-- TOC --><a name="training-the-object-suggester"></a>
## Training the object suggester

```
python scripts/training/train_object_suggester.py
```

<!-- TOC --><a name="training-the-model-deviation-estimator"></a>
## Training the model deviation estimator

```
python scripts/training/train_mde.py
```

<!-- TOC --><a name="planning"></a>
# Planning

The first command runs planning; the second executes the resulting plans.

<!-- TOC --><a name="run-planning"></a>
## Run planning
```
python scripts/planning/benchmark_plans.py
```

<!-- TOC --><a name="run-execution"></a>
## Run execution

```
python scripts/planning/benchmark_sim_execution.py
```
