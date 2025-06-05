@echo off
echo ============================================
echo This may take a few minutes, please wait...
echo ============================================
REM Create required directories
mkdir weights\icon_detect
mkdir weights\icon_caption

REM Download icon detection files
echo 1/3 Downloading icon detection files...
echo huggingface-cli download microsoft/OmniParser-v2.0 icon_detect/train_args.yaml
huggingface-cli download microsoft/OmniParser-v2.0 icon_detect/train_args.yaml --local-dir weights
echo huggingface-cli download microsoft/OmniParser-v2.0 icon_detect/config.json
huggingface-cli download microsoft/OmniParser-v2.0 icon_detect/model.pt --local-dir weights
echo huggingface-cli download microsoft/OmniParser-v2.0 icon_detect/model.yaml
huggingface-cli download microsoft/OmniParser-v2.0 icon_detect/model.yaml --local-dir weights

REM Download icon caption files
echo 2/3 Downloading icon caption files...
echo huggingface-cli download microsoft/OmniParser-v2.0 icon_caption/train_args.yaml
huggingface-cli download microsoft/OmniParser-v2.0 icon_caption/config.json --local-dir weights
echo huggingface-cli download microsoft/OmniParser-v2.0 icon_caption/model.pt
huggingface-cli download microsoft/OmniParser-v2.0 icon_caption/generation_config.json --local-dir weights
echo huggingface-cli download microsoft/OmniParser-v2.0 icon_caption/model.safetensors
huggingface-cli download microsoft/OmniParser-v2.0 icon_caption/model.safetensors --local-dir weights

REM Rename folder
echo 3/3 Renaming icon caption folder...
move weights\icon_caption weights\icon_caption_florence

echo All tasks completed successfully.
pause