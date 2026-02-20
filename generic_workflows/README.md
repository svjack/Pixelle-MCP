# Pixelle-MCP Project Setup Guide

## Overview
Pixelle-MCP is an AI-powered media generation platform that integrates ComfyUI workflows with various AI models for image/video generation, editing, and captioning. (Both Step-By-Step or Plan-Exe-Overall)

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

#git clone https://github.com/omar92/ComfyUI-QualityOfLifeSuit_Omar92
git clone https://github.com/svjack/ComfyUI-QualityOfLifeSuit_Omar92
# output file locate in: /home/featurize/ComfyUI/custom_nodes/ComfyUI-QualityOfLifeSuit_Omar92/output

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

git clone https://github.com/svjack/ComfyUI-HunyuanVideo-Foley
pip install -r ComfyUI-HunyuanVideo-Foley/requirements.txt

git clone https://github.com/rgthree/rgthree-comfy
pip install -r rgthree-comfy/requirements.txt
git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes

git clone https://github.com/city96/ComfyUI-GGUF
pip install -r ComfyUI-GGUF/requirements.txt

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

pip install playwright
playwright install chromium
git clone https://github.com/svjack/ComfyUI-HTMLRenderer
pip install -r ComfyUI-HTMLRenderer/requirements.txt

git clone https://github.com/christian-byrne/audio-separation-nodes-comfyui
pip install -r audio-separation-nodes-comfyui/requirements.txt
git clone https://github.com/1038lab/ComfyUI-QwenASR
pip install -r ComfyUI-QwenASR/requirements.txt

cd ../../
cp work/sageattention-1.0.6-py3-none-any.whl .
pip install sageattention-1.0.6-py3-none-any.whl 
```

## Install Chrome in Linux (For Html Render)
```bash
### install chorme in linux
# wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
cp work/google-chrome-stable_current_amd64.deb .
sudo apt install ./google-chrome-stable_current_amd64.deb

google-chrome --version
# Google Chrome 145.0.7632.75 
which google-chrome
# wget https://storage.googleapis.com/chrome-for-testing-public/145.0.7632.75/linux64/chromedriver-linux64.zip
cp work/chromedriver-linux64.zip . 
unzip chromedriver-linux64.zip
sudo cp chromedriver-linux64/chromedriver /usr/local/bin/
sudo chmod +x /usr/local/bin/chromedriver
chromedriver --version
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
git clone https://github.com/svjack/Pixelle-MCP.git
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

### Wan2.2 SVI Multiple Shots Image-to-Video 
```bash
modelscope download svjack/Smooth_Mix_Wan_2_2_I2V --local_dir="."
cp smoothMixWan22I2VT2V_i2v* ComfyUI/models/diffusion_models
modelscope download svjack/SVI --local_dir="."
cp SVI_v2_PRO_Wan2.2-I2V-A14B_*.safetensors ComfyUI/models/loras
```

### Wan2.2 First-Last Image-to-Video 
```bash
#wget https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/main/HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf
#cp Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf ComfyUI/models/diffusion_models
#wget https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/main/LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf
#cp Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf ComfyUI/models/diffusion_models
wget https://huggingface.co/QuantStack/Wan2.2-Fun-A14B-InP-GGUF/resolve/main/HighNoise/Wan2.2-Fun-A14B-InP_HighNoise-Q4_K_M.gguf
cp Wan2.2-Fun-A14B-InP_HighNoise-Q4_K_M.gguf ComfyUI/models/diffusion_models
wget https://huggingface.co/QuantStack/Wan2.2-Fun-A14B-InP-GGUF/resolve/main/LowNoise/Wan2.2-Fun-A14B-InP_LowNoise-Q4_K_M.gguf
cp Wan2.2-Fun-A14B-InP_LowNoise-Q4_K_M.gguf ComfyUI/models/diffusion_models
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

### Hunyuanvideo Foley add audio to video
```bash
mkdir -p ComfyUI/models/foley
export HF_ENDPOINT=https://hf-mirror.com
hf download phazei/HunyuanVideo-Foley hunyuanvideo_foley.safetensors --local-dir="."
# wget https://huggingface.co/phazei/HunyuanVideo-Foley/resolve/main/hunyuanvideo_foley.safetensors
cp hunyuanvideo_foley.safetensors ComfyUI/models/foley
wget https://huggingface.co/phazei/HunyuanVideo-Foley/resolve/main/synchformer_state_dict_fp16.safetensors
cp synchformer_state_dict_fp16.safetensors ComfyUI/models/foley
wget https://huggingface.co/phazei/HunyuanVideo-Foley/resolve/main/vae_128d_48k_fp16.safetensors
cp vae_128d_48k_fp16.safetensors ComfyUI/models/foley
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
      - 使用 `图片1` 和 `图片2` 的提示词格式指定对应的图片 
   - `qwen_image_edit_two_pic_head_swap` - Face swapping

### Video Generation
1. **Wan2.2 Text-to-Video**: `Wan2_2_text_to_video_api` - Text-to-video generation
2. **Wan2.2 Image-to-Video**: `Wan2_2_image_to_video_api` - Image-to-video generation
3. **Wan2.2 First-Last Image-to-Video**: `wan2_2_first_last_image_to_video_inpaint_api` - FL-Image-to-video generation (default length 81, don't below this value)
4. **Wan2.2 SVI Multiple shots Image-to-Video**: `Wan2_2_SVI_6_Shots_image_to_video_api` - Multiple Shots Image-to-video generation
5. **Wan2.1 Pose Transfer**: `wan21_video_pose_transfer` - Image & Video-Pose-to-video generation
6. **Wan2.1 InfiniteTalk digit man**: `wan21_single_digital_man_infinite_talker_{num}_segments` - Image & Audio-to-video generation

### Audio and Captioning
1. **Qwen3 TTS**:
   - `Qwen3_TTS_Voice_Design_api` - Text-to-speech voice design.
   - `Qwen3_TTS_Voice_Clone_api` - Voice clone
2. **QwenVL3 ASR**:
   - `qwen3_asr_subtitle` - Audio ASR to subtitle
3. **QwenVL3 Captioning**:
   - `qwenvl3_image_describe_api` - Image captioning using Ollama
   - `qwenvl3_video_describe_api` - Video captioning
4. **ACE Step 1.5 Text to Music**:
   - `ace_step_1_5_text_to_music` - Text to Music (default Lyrics [Instrumental] - Pure Music)
5. **Hunyuanvideo Foley add audio to video**:
   - `{add/merge}_audio_to_video_Foley` - add or merge background audio to video


### Html template render to Image\Video
1. **Html template render to PNG**: `html_template_render` - html template (image + title + text fields) render to image, can produce short videos' frame, PPT and so on.
   - 01. render to PNG:
      - prompt:
      ```bash
      完成一个3张流行音乐演唱会的广告任务,使用z_image_turbo生成3张图片，图片中不包含任何文字，并生成统一的流行风格html模板代码
      （字体采用要符合模板说明中的要求，模板中不包含任何按钮，且结构简单），并进行3次渲染。
      ```
      - prompt:
     ```bash
     完成一个3张水族馆的广告任务,使用z_image_turbo生成3张图片，图片中不包含任何文字,并生成统一的流行风格html模板代码
     （采用蓝色的生命风格，字体采用要符合模板说明中的要求，模板中不包含任何按钮，且结构简单），并进行3次渲染。
      ```

   - 02. render to Short video with audio: (video frame)
      - prompt:
      ```bash
      完成一个3张水族馆图片合成的有声广告短视频任务,使用z_image_turbo生成3张图片，图片中不包含任何文字，
      并生成统一的流行风格html模板代码（采用蓝色的生命风格，字体采用要符合模板说明中的要求，模板中不包含任何按钮，且结构简单），
      并进行3次渲染。使用给你的音频作为音频克隆参考音频克隆每个图片对应的解说text,对3个渲染后图片加上各自的解说音频生成解说短视频片段，
      将3个有声视频连接成一个有声短视频
      ```
      - prompt:
     ```bash
     完成一个5张日常漫画图片合成的心灵鸡汤短视频任务,使用z_image_turbo生成5张图片，图片中不包含任何文字，
     但有相同的日系治愈风格，并生成统一的流行风格html模板代码（采用粉色的浪漫风格，字体采用要符合模板说明中的要求，
     模板中不包含任何按钮，且结构简单），并进行5次渲染。使用给你的音频作为音频克隆参考音频克隆每个图片对应的解说text,
     对5个渲染后图片加上各自的解说音频生成解说短视频片段，将5个有声视频连接成一个有声短视频
      ```
     
   - 03. ACE music -> render to Short music video with audio: (video frame):
      - 001
         - step1:plan prompt:
        ```bash
         	如果让你生成一段中文音乐，之后通过多图片静态配图的方式生成以这段中文音乐为背景的有声视频，你如何来做，给出你的计划，
            不执行(大概的逻辑：生成音乐->asr得到音乐内容->根据带时间戳的字幕生成5个对应时间点的应景图片->html渲染5个有文字的图片
         	->使用空音频根据时间戳确定一些空音频的时长，生成对应的无声视频片段(视频片段加总之后长度为音乐长度)-
         	>连接这些无声视频片段->对合并后的无声视频片段，以生成的音乐为背景音乐生成最终的有声视频)
         	注意在整个过程中，挑选空音频的时长要根据asr得到音乐内容的时间戳，并且总长度为一开始生成的中文音乐长度。
        ```
         
         - step2:exec prompt1:
        ```bash
         	按照你的计划生成一个45s的中文歌曲（歌曲的主题为：中文动漫歌曲，以樱花下浪漫的爱情为主题），在生成对应的图片时
         	使用z_image_turbo（要符合当时时间戳歌词内容的动漫风格背景图片）3张，渲染html的模板设计使用粉色樱花背景
           （注意只使用3中要求的字体），
         	并使得图片占较大部分。html的text部分使用一些符合图片内容的浪漫语句。在合并视频时最长边resize到832，
           每段子视频没有间隔或缓冲，
         	所有时间加起来为45s。执行到生成对应的无声视频片段，暂时不进行后续连接和加声。
         	直接执行，不再列出更改后的计划或中间生成的html模板。
        ```
         
         - step2:exec prompt2:
           ```bash
         	继续执行视频连接（resize到832）和将生成音乐作为背景音乐。
           ```
           
      - 002
   
        ```bash
             如果让你生成一段中文音乐，之后通过多图片静态配图的方式生成以这段中文音乐为背景的有声视频，你如何来做，给出你的计划，不执行
          	(大概的逻辑：生成音乐->asr得到音乐内容->根据带时间戳的字幕生成5个对应时间点的应景图片->html渲染5个有文字的图片
          	->使用空音频根据时间戳确定一些空音频的时长，生成对应的无声视频片段(视频片段加总之后长度为音乐长度)-
          	>连接这些无声视频片段->对合并后的无声视频片段，以生成的音乐为背景音乐生成最终的有声视频)
          	注意在整个过程中，挑选空音频的时长要根据asr得到音乐内容的时间戳，并且总长度为一开始生成的中文音乐长度。
          	按照你的计划生成一个90s的中文歌曲（歌曲的主题为：中文动漫歌曲，以星空下思念为主题），在生成对应的图片时
          	使用z_image_turbo（要符合当时时间戳歌词内容的动漫风格背景图片）5张，渲染html的模板设计使用蓝色星空背景
           （注意只使用3中要求的字体），
          	并使得图片占较大部分。html的text部分使用一些符合图片内容的浪漫语句。在合并视频时最长边resize到832，每段子视频没有间隔或缓冲，
          	所有时间加起来为90s。
          	直接执行，不再列出更改后的计划或中间生成的html模板。
        ```

2. **Html dynamic template render to Video**: `html_dynamic_template_render` - html dynamic template (support by js) (image + title + text fields) render to video, can produce short videos.
      (Similar to NetEase Cloud Music's rotating playback disc)
   - 01. render to single rotating playback disc:
      - prompt:
      ```bash
      完成一个歌曲创作与动态模板视频配置任务，要求这个歌曲符合林则徐这个清末老兵的命运，
      使用z_image_turbo生成1张褐色仿古林则徐图片，图片中不包含任何文字，
      设计一个有隆重历史风格的(宽480高832)html动态模板（不进行打印），背景中有散散星火和流星，
      标题和text都有大小反复放缩效果，图片在旋转地同时也进行反复放缩，
      简短地概括出林则徐的一生。字体采用要符合模板说明中的要求，模板中不包含任何按钮，且结构简单。
      进行一次120s的渲染，再使用ace生成对应时长（120s）的描述林则徐生平的中文歌曲加入到视频里面，
      生成最终的歌曲视频。
      背景知识：
      林则徐（1785年8月30日－1850年11月22日），字元抚，又字少穆、石麟，晚号俟村老人、瓶泉居士等，
      福建省侯官县（今福州市）人，祖籍福建福清。他是中国清代后期著名的政治家、思想家和诗人，官至一品，
      曾任湖广总督、陕甘总督和云贵总督，两次受命为钦差大臣。
      林则徐最为后世所铭记的是他在禁烟运动中的杰出贡献。1839年，他作为钦差大臣赴广东查禁鸦片，
      主持了著名的“虎门销烟”，沉重打击了英国等列强的鸦片贸易，维护了国家主权和民族尊严。他主张学习西方先进技术，
      组织编译《四洲志》，被誉为中国近代“开眼看世界的第一人”。
      林则徐一生为官清廉，治水有方，关心民生，其“苟利国家生死以，岂因祸福避趋之”的名句，至今仍是爱国精神的典范。
      ```
   - 02. render to 3 rotating playback discs:
      - prompt:
        ```bash
          完成一个3张流行音乐演唱会的广告任务,使用z_image_turbo生成3张图片，图片中不包含任何文字，
          并生成统一的流行风格(宽480高832)html动态模板代码（采用红蓝动态粒子闪烁效果，标题和text都有大小反复放缩效果,
          字体采用要符合模板说明中的要求，模板中不包含任何按钮，且结构简单，直接使用你生成的模板，不进行打印），
          并进行3次渲染，每次渲染30s。3个视频连接成一个视频(最长边resize到832)，再使用ace生成对应时长的背景音乐加入到视频里面。
          (最长边resize到832)
        ```



## Usage Notes
- Support Manual Step-by-Step Execution && AI-Assisted Planning & Overall Execution
   - Step-by-Step Execution:
   - Planning & Overall Execution: - plan_knowledge.json
      - Give Your Plan 
      - Now, given the specific input parameters you used for various tools, the returned result is an ordered JSON structure.
      - Edit this JSON structure.
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

