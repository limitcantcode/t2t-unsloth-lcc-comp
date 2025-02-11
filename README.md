# Unsloth-driven T2T Component by LCC

## What is this for?
Uses [unsloth](https://github.com/unslothai/unsloth) to generate responses from text given your configured model. This will run (finetuned) models from unsloth locally. 

## Setup

Please refer to the [unsloth repo](https://github.com/unslothai/unsloth) for the latest installation instructions for your os. Below is a modified verion specific for this project at the time of writing.

**This component has been tested with NVidia GPUs.** I can't ensure this will work with non-CUDA cards.

Please ensure you have [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed. You may also need to edit your path variables to ensure it's tools can be found.

If working in WSL, ensure you are using WSL2. You may need to additionally install the driver's in WSL as well.

Furthermore, create a `.env` file in the root of this project with the following:
```
MODEL=<name of model like models/lora_model...>
IS_4_BIT=<true or false, same as what you used in training>
```
The model needs to be in this projects `models` directory. You can finetune and/or save them separately using unsloth (tutorials and notebooks on their repo).

### Windows
1. Ensure you have CUDA setup properly
2. Create and activate the virtual environment. Replace the `pytorch-cuda` version with the latest version before the version stated on your CUDA (found using command `nvidia-smi`). For example, I'm using `CUDA 12.6`, but the latest `pytorch-cuda` is for `CUDA 12.4`, so use `pytorch-cuda=12.4`
```
conda create -n jaison-comp-t2t-unsloth python=3.12 pytorch-cuda=12.4 pytorch cudatoolkit -c pytorch -c nvidia -y
conda activate jaison-comp-t2t-unsloth
```
3. Install the remaining dependencies **in the same order**. You will need the machine-specific command to install [xformers](https://github.com/facebookresearch/xformers).
```
pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
# install command for xformers like: pip install -U xformers --index-url https://download.pytorch.org/whl/cu126
pip install --no-deps trl peft accelerate bitsandbytes
pip install -I -r requirements.txt
```

### Unix
1. Ensure you have CUDA setup properly
2. Create and activate the virtual environment.
```
python -m venv venv
source venv/bin/activate
```
3. Install the remaining dependencies **in the same order**. Find the command to install the correct pytorch packages for your system from [their website](https://pytorch.org/get-started/locally/). Also find the command to install the correct `xformers` package for your system from [this repo](https://github.com/facebookresearch/xformers).
```
# install command for pytorch like: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
# install command for xformers like: pip install -U xformers --index-url https://download.pytorch.org/whl/cu126
pip install --no-deps trl peft accelerate bitsandbytes
pip install -I -r requirements.txt
```

## Testing
Assuming you are in the right virtual environment and are in the root directory:
```
python ./src/main.py --port=5000
```
If it runs, it should be fine.

## Configuration
There is no additional configuration.

## Related stuff
Project J.A.I.son: https://github.com/limitcantcode/jaison-core

Join the community Discord: https://discord.gg/Z8yyEzHsYM
