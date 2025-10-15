# mkdir -p $HOME/.tools
# wget -O $HOME/.tools/trt.tar.gz https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.13.3/tars/TensorRT-10.13.3.9.Linux.x86_64-gnu.cuda-12.9.tar.gz
cd $HOME/.tools
tar -xzf $HOME/.tools/trt.tar.gz
echo 'export TRT_DIR="$HOME/.tools/TensorRT-10.13.3.9"; export PATH="$TRT_DIR/bin:$PATH"; export LD_LIBRARY_PATH="$TRT_DIR/lib:${LD_LIBRARY_PATH}"; [ -d "$TRT_DIR/targets/x86_64-linux-gnu/lib" ] && export LD_LIBRARY_PATH="$TRT_DIR/targets/x86_64-linux-gnu/lib:${LD_LIBRARY_PATH}"' >> ~/.bashrc