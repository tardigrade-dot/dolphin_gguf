
#### 实验性

模型文件下载:
https://huggingface.co/mradermacher/Dolphin-v2-GGUF

(仅包含部分,copy from huggingface for users cannot visit huggingface)
https://modelscope.cn/models/wagyuuuu/Dolphin-v2-GGUF

需要安装llama_cpp: https://github.com/abetlen/llama-cpp-python

export CMAKE_ARGS="-DGGML_METAL=on" #macos
pip install 'llama-cpp-python[server]'


服务器启动:
/Volumes/sw/conda_envs/phi/bin/python3 -m llama_cpp.server \
    --model /Volumes/sw/GGUF/Dolphin-v2.Q4_K_M.gguf \
    --clip_model_path /Volumes/sw/GGUF/Dolphin-v2.mmproj-Q8_0.gguf \
    --chat_format qwen2.5-vl \
    --n_batch 16 \
    --host 0.0.0.0 --port 8000

需要使用到qwen_vl_utils的部分已经拷贝, 不再依赖, 因此也不需要安装pytorch
pip install openai pillow tqdm ipykernel pymupdf numpy opencv-python 
qwen_vl_utils 去掉了该依赖, 以避免安装pytorch
conda install pytorch torchvision torchaudio -c pytorch

使用:
安装:pip install -e .

dolphin-run --input_path /path/to/demo/zh_page_14.png --output_path /path/to/data_output

#### 说明
原始模型地址: https://github.com/bytedance/Dolphin
存在的问题: 不同尺寸的输入文件和不同的服务器启动参数:n_batch输出的ocr第一阶段坐标会有所不同,所以在有些场景下不可用.
目前在macOS中, n_batch 16 以及一个坐标修正参数(1.54)下, 部分尺寸的突破可以运行.