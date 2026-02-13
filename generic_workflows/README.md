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
```

### 2. Install and Launch ComfyUI
```bash
comfy --here install
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

### Audio and Captioning
1. **Qwen3 TTS**:
   - `Qwen3_TTS_Voice_Design_api` - Text-to-speech voice design.
   - `Qwen3_TTS_Voice_Clone_api` - Voice clone 
3. **QwenVL3 Captioning**:
   - `qwenvl3_image_describe_api` - Image captioning using Ollama
   - `qwenvl3_video_describe_api` - Video captioning

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
