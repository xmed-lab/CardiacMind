# install according to your GPU configs
# pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install opencv-python-headless
pip install ms-swift -U
pip install qwen_vl_utils==0.0.11 decord -U
# check the version to see if it is compatible with your CUDA version and PyTorch version
pip install flash-attn==v2.8.0.post2 --no-build-isolation
