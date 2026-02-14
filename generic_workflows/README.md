# Pixelle-MCP Project Setup Guide

## Overview
Pixelle-MCP is an AI-powered media generation platform that integrates ComfyUI workflows with various AI models for image/video generation, editing, and captioning.

## Prerequisites
- Linux-based system
- Conda environment manager
- Git and Git LFS

## Initial Setup

### Install Basic Tools
```bash
sudo apt-get update && sudo apt-get install git-lfs cbm ffmpeg
```

## ComfyUI Installation

### 1. Environment Setup
```bash
conda activate base
pip install modelscope datasets huggingface_hub "httpx[socks]"

conda activate system
pip install modelscope datasets huggingface_hub "httpx[socks]"
pip install comfy-cli
pip install "questionary<2.1.0"
pip install --upgrade typer
comfy --here install
```

### 2. Launch ComfyUI
```bash
comfy launch -- --listen 0.0.0.0
```

### 3. Verify and Export Port
```bash
sudo netstat -tuln | grep 8188
featurize port export 8188
```

## Install ComfyUI Dependencies

### Custom Nodes Installation
```bash
conda activate system
cd ComfyUI/custom_nodes

# Clone and install various custom nodes
git clone https://github.com/yolain/ComfyUI-Easy-Use
pip install -r ComfyUI-Easy-Use/requirements.txt

git clone https://github.com/omar92/ComfyUI-QualityOfLifeSuit_Omar92
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
pip install -r ComfyUI-VideoHelperSuite/requirements.txt

git clone https://github.com/wanaigc/ComfyUI-Qwen3-TTS
pip install -r ComfyUI-Qwen3-TTS/requirements.txt
pip install "transformers<5"

git clone https://github.com/stavsap/comfyui-ollama
pip install -r comfyui-ollama/requirements.txt

git clone https://github.com/jamesWalker55/comfyui-various
pip install soundfile

git clone https://github.com/svjack/ComfyUI-QwenVL
pip install -r ComfyUI-QwenVL/requirements.txt
pip install "numpy<2"

#### wan video pose transfer
git clone https://github.com/kijai/ComfyUI-WanVideoWrapper
git clone https://github.com/cubiq/ComfyUI_essentials
git clone https://github.com/kijai/ComfyUI-SCAIL-Pose
git clone https://github.com/M1kep/ComfyLiterals
git clone https://github.com/kijai/ComfyUI-WanAnimatePreprocess
git clone https://github.com/kijai/ComfyUI-KJNodes
git clone https://github.com/kael558/ComfyUI-GGUF-FantasyTalking
git clone https://github.com/aining2022/ComfyUI_Swwan
git clone https://github.com/bollerdominik/ComfyUI-load-lora-from-url

conda activate system
pip install -r ComfyUI-WanVideoWrapper/requirements.txt
pip install -r ComfyUI_essentials/requirements.txt
pip install -r ComfyUI-SCAIL-Pose/requirements.txt
#pip install -r ComfyLiterals/requirements.txt
pip install -r ComfyUI-WanAnimatePreprocess/requirements.txt
pip install -r ComfyUI-KJNodes/requirements.txt
pip install -r ComfyUI-GGUF-FantasyTalking/requirements.txt
pip install -r ComfyUI_Swwan/requirements.txt
pip install -r ComfyUI-load-lora-from-url/requirements.txt
pip install "numpy<2"

cd ../../
cp work/sageattention-1.0.6-py3-none-any.whl .
pip install sageattention-1.0.6-py3-none-any.whl 
```

## Ollama Setup

### Install and Configure Ollama
```bash
cp work/ollama-linux-amd64.tar.zst .
sudo tar -xvf ollama-linux-amd64.tar.zst -C /usr
ollama serve
```

### Run Qwen3-VL Model
```bash
ollama run huihui_ai/qwen3-vl-abliterated:8b
# To stop the model
ollama stop huihui_ai/qwen3-vl-abliterated:8b
```

## Pixelle-MCP Installation

### 1. Clone and Setup
```bash
conda activate base
git clone https://github.com/AIDC-AI/Pixelle-MCP.git
cd Pixelle-MCP
pip install uv
uv run pixelle
uv pip install "httpx[socks]"
```

### 2. Choose LLM Provider
When prompted, choose `deepseek` as the language model provider.

### 3. Verify and Export Port
```bash
sudo netstat -tuln | grep 9004
featurize port export 9004
```

## Configuration

### Environment Variables (.env file)
- `COMFYUI_BASE_URL=http://workspace.featurize.cn:26287` - Specify ComfyUI server URL
- `PUBLIC_READ_URL=""` - Use for standalone tool configuration
- `PUBLIC_READ_URL="http://workspace.featurize.cn:41204"` - Use for individual tool usage

### WebUI Access
- Default credentials: `dev/dev` (username/email and password)

## Model File Downloads

### Z-Image Models
```bash
export HF_ENDPOINT=https://hf-mirror.com
hf download Comfy-Org/z_image_turbo --local-dir="Comfy-Org/z_image_turbo"
cp Comfy-Org/z_image_turbo/split_files/vae/* ComfyUI/models/vae
cp Comfy-Org/z_image_turbo/split_files/text_encoders/* ComfyUI/models/text_encoders
cp Comfy-Org/z_image_turbo/split_files/diffusion_models/* ComfyUI/models/diffusion_models
```

### Qwen-Image Models
```bash
wget https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/resolve/main/v23/Qwen-Rapid-AIO-SFW-v23.safetensors
cp Qwen-Rapid-AIO-SFW-v23.safetensors ComfyUI/models/checkpoints
wget https://huggingface.co/svjack/Qwen_Image/resolve/main/bfs_head_v3_qwen_image_edit_2509.safetensors
cp bfs_head_v3_qwen_image_edit_2509.safetensors ComfyUI/models/loras
```

### Wan2.2 Text-to-Video Models
```bash
wget https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
cp umt5_xxl_fp8_e4m3fn_scaled.safetensors ComfyUI/models/text_encoders
wget https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors
cp wan_2.1_vae.safetensors ComfyUI/models/vae

wget https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors
wget https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors
cp wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors ComfyUI/models/diffusion_models
cp wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors ComfyUI/models/diffusion_models
wget https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors
cp wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors ComfyUI/models/loras
wget https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors
cp wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors ComfyUI/models/loras
```

### Wan2.2 Image-to-Video Models
```bash
wget https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
wget https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors
cp wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors ComfyUI/models/diffusion_models
cp wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors ComfyUI/models/diffusion_models
wget https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors
cp wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors ComfyUI/models/loras
wget https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors
cp wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors ComfyUI/models/loras
```

### ACE step 1.5 text to music 
```bash
wget https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/resolve/main/split_files/diffusion_models/acestep_v1.5_turbo.safetensors
cp acestep_v1.5_turbo.safetensors ComfyUI/models/diffusion_models
wget https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/resolve/main/split_files/vae/ace_1.5_vae.safetensors
cp ace_1.5_vae.safetensors ComfyUI/models/vae
wget https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/resolve/main/split_files/text_encoders/qwen_0.6b_ace15.safetensors
cp qwen_0.6b_ace15.safetensors ComfyUI/models/text_encoders
wget https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/resolve/main/split_files/text_encoders/qwen_4b_ace15.safetensors
cp qwen_4b_ace15.safetensors ComfyUI/models/text_encoders
```

### Wan video pose transfer 
```bash
wget https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/SCAIL/Wan21-14B-SCAIL-preview_fp8_e4m3fn_scaled_KJ.safetensors
cp Wan21-14B-SCAIL-preview_fp8_e4m3fn_scaled_KJ.safetensors ComfyUI/models/diffusion_models
wget https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors
cp lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors ComfyUI/models/loras 
wget https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_Uni3C_controlnet_fp16.safetensors
cp Wan21_Uni3C_controlnet_fp16.safetensors ComfyUI/models/controlnet
wget https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors
cp Wan2_1_VAE_bf16.safetensors ComfyUI/models/vae
wget https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-fp8_e4m3fn.safetensors
cp umt5-xxl-enc-fp8_e4m3fn.safetensors ComfyUI/models/text_encoders
wget https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors
cp clip_vision_h.safetensors ComfyUI/models/clip_vision
mkdir -p ComfyUI/models/detection
wget https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/vitpose-l-wholebody.onnx
cp vitpose-l-wholebody.onnx ComfyUI/models/detection
wget https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/resolve/main/process_checkpoint/det/yolov10m.onnx
cp yolov10m.onnx ComfyUI/models/detection
```

### Wan InfiniteTalk single digit man
```bash
wget https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors
cp Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors ComfyUI/models/diffusion_models
wget https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors
cp lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors ComfyUI/models/loras
wget https://huggingface.co/Kijai/wav2vec2_safetensors/resolve/main/wav2vec2-chinese-base_fp16.safetensors
cp wav2vec2-chinese-base_fp16.safetensors ComfyUI/models/audio_encoders
wget https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/model_patches/wan2.1_infiniteTalk_single_fp16.safetensors
cp wan2.1_infiniteTalk_single_fp16.safetensors ComfyUI/models/model_patches
```

## Available Workflows

### Image Generation/Editing
1. **z-image**: `z_image_turbo_text_to_image_api` - Fast text-to-image generation
2. **qwen-image**:
   - `qwen_image_text_to_image_api` - Text-to-image generation
   - `qwen_image_edit_one_pic_api` - Single image editing
   - `qwen_image_edit_two_pic_api` - Dual image editing
   - `qwen_image_edit_two_pic_head_swap` - Face swapping

### Video Generation
1. **Wan2.2 Text-to-Video**: `Wan2_2_text_to_video_api` - Text-to-video generation
2. **Wan2.2 Image-to-Video**: `Wan2_2_image_to_video_api` - Image-to-video generation
3. **Wan2.1 Pose Transfer**: `wan21_video_pose_transfer` - Image & Video-Pose-to-video generation
4. **Wan2.1 InfiniteTalk digit man**: `wan21_single_digital_man_infinite_talker_{num}_segments` - Image & Audio-to-video generation

### Audio and Captioning
1. **Qwen3 TTS**:
   - `Qwen3_TTS_Voice_Design_api` - Text-to-speech voice design.
   - `Qwen3_TTS_Voice_Clone_api` - Voice clone 
2. **QwenVL3 Captioning**:
   - `qwenvl3_image_describe_api` - Image captioning using Ollama
   - `qwenvl3_video_describe_api` - Video captioning
3. **ACE Step 1.5 Text to Music**:
   - `ace_step_1_5_text_to_music` - Text to Music (default Lyrics [Instrumental] - Pure Music)  
 

## Usage Notes
- Custom workflows are located in `./data/custom_workflows`
- Use `ls ./data/custom_workflows` to list available workflows
- Check workflow parameters for each specific workflow to understand input requirements
- View the parameters section for detailed input specifications of each workflow

## Custom Workflow Management
- Place custom workflow files in `./data/custom_workflows` directory
- The system will automatically detect and make them available through the API

## Model Requirements
Ensure sufficient storage space is available for all model files. The complete setup requires downloading multiple large model files for different functionalities.


# MCP Inspector 连接配置指南

以下是连接 `http://localhost:9004/pixelle/mcp` 的完整步骤，包含环境准备与界面配置。

## 1. 环境准备（安装 Node.js）

MCP Inspector 基于 Node.js 运行，需先安装运行环境。

| 步骤 | 操作说明 | 备注 |
| :--- | :--- | :--- |
| **下载安装** | 访问 https://nodejs.org/，下载并安装 **LTS 版本**。 | 推荐使用 MSI 安装包，自动配置环境变量。 |
| **验证安装** | 打开终端，运行 `node -v` 和 `npm -v`。 | 若显示版本号，则安装成功。 |

## 2. 启动与连接配置

在终端执行以下命令启动 Inspector，然后在 Web 界面中配置连接。

| 配置项 | 参数值 | 说明 |
| :--- | :--- | :--- |
| **启动命令** | `npx @modelcontextprotocol/inspector` | 启动后浏览器自动打开 `http://localhost:6274`。 |
| **传输协议** | **Streamable HTTP** | 根据你的日志，必须选择此模式，而非 SSE。 |
| **服务器地址** | `http://localhost:9004/pixelle/mcp` | 填写你提供的完整 URL。 |
| **认证配置** | (可选) | 若服务器需要 Token，在 Authentication 中设置 Header 和 Token。 |

## 3. 连接与调试

点击 **Connect** 按钮连接服务器。连接成功后，即可在 **Tools** 标签页中测试工具（可以看到所有已有工具及协议），或在 **Resources** 标签页中查看资源列表。

