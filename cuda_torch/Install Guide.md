
### NVIDIA Toolkit
The [Nvidia toolkit][https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network] has to be installed. The following commands should do the trick;
```bash
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
sudo apt-get install nvidia-cuda-toolkit
```
### Python installation
The challenge is many distros have a system-integral python installation, which makes package management difficult. Furthermore, virtual environments often don't have permission to install complex libraries. A work around is installing a new Python distro.
```bash
sudo apt-get install python3.13
python3.13 -m pip install numpy wurlitzer ninja jupyterlab notebook
python3.13 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python3.13 -m pip install typing-extensions --upgrade
```

### Specific Files
#### C++ Configuration
The [Microsoft C/C++ Extension for VSCode][https://marketplace.visualstudio.com/items/?itemName=ms-vscode.cpptools] is fine. Note that you may have to modify the "Include Path" setting (can be found by the "C/C++: Edit Configurations (UI)" Ctrl+P command) to have;
```
${workspaceFolder}/**
~/.local/lib/python3.13/site-packages/torch/include/
~/.local/lib/python3.13/site-packages/torch/include/torch/csrc/api/include/
```
#### Half Compilation
`cpp_extensions.py` is located at;
`~/.local/lib/python3.13/site-packages/torch/utils`
If `__half2` operations are having a hard time compiling, comment out the `COMMON_NVCC_FLAGS` which restrict `half` operations.
#### Kernel Compilation
Sometimes kernels are compiled to;
`~/.cache/torch_extensions/py313_cu126/`