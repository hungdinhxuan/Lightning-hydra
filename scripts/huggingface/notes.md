# 22-Oct-2025
python upload_model.py \
    --model_name xlsr_conformer_tcm \
    --config_path /home/hungdx/code/Lightning-hydra/configs/experiment/cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_more_elevenlabs_july4.yaml \
    --checkpoint_path /home/hungdx/code/Lightning-hydra/pretrained/S_241214_conf-1.pth \
    --repo_name hungdinhxdev/S_241214_conf-1 \
    --private \
    --verbose \
    --commit_message "Upload model" 