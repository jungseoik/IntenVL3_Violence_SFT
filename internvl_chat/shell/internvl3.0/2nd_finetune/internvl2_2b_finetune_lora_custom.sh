# e.g.
# CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 PER_DEVICE_BATCH_SIZE=4 sh shell/internvl2_2b_finetune_lora.sh
# choose samller PER_DEVICE_BATCH_SIZE to reduce GPU Memory
set -x

GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-512}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='../ckpts/fucking_lora'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 8
# batch size per gpu: 4
# gradient accumulation steps: 16
# total batch size: 512
# epoch: 1
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "../ckpts/InternVL2-2B" \
  --conv_style "internlm2-chat" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data/custom_data.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --use_llm_lora 64 \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 3 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 1 \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 16384 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
import os
import json
import pandas as pd
from internvl_inferencer.vl3 import InternVL3Inferencer
from assets.config import VIDEO_BINARY_CLASSIFICATION_PROMPT, VIDEO_CLASSIFICATION_PROMPT, VIDEO_DESCRIPTION, VIDEO_JUDGEMENT

import concurrent.futures

# Experiment settings
MODEL_LIST = [
    "OpenGVLab/InternVL3-2B"
]
TEMPLATES = {
    "binary": VIDEO_BINARY_CLASSIFICATION_PROMPT,
    "classification": VIDEO_CLASSIFICATION_PROMPT,
    "description": VIDEO_DESCRIPTION,
    "judgement": VIDEO_JUDGEMENT,
}
NUM_SEGMENTS_LIST = [8, 12, 20]
VIDEO_FOLDER = "test/"
VIDEO_CATEGORIES_FILE = "assets/video_categories.json"
OUTPUT_CSV = "results.csv"
MAX_WORKERS = 4  # Number of parallel inference processes


def run_combo(task):
    """
    Runs inference for one (model, template, num_segments) combo over all videos.
    """
    model_id = task["model_id"]
    template = task["template"]
    template_name = task["template_name"]
    num_segments = task["num_segments"]
    video_folder = task["video_folder"]
    video_files = task["video_files"]
    video_categories = task["video_categories"]

    # Load model once per combo
    inferencer = InternVL3Inferencer(model_id=model_id)
    rows = []

    for video_name in video_files:
        video_path = os.path.join(video_folder, video_name)
        ground_truth = video_categories[video_name]
        response = inferencer.infer(
            video_path=video_path,
            template=template,
            num_segments=num_segments
        )
        # Parse JSON category
        predicted_category = None
        if isinstance(response, str):
            start, end = response.find('{'), response.rfind('}')
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(response[start:end+1])
                    predicted_category = parsed.get("category")
                except json.JSONDecodeError:
                    pass

        row = {
            "video_name": video_name,
            "ground_truth": ground_truth,
            "model_name": model_id,
            "template_type": template_name,
            "predicted_category": predicted_category,
            "response": response,
            "num_segment": num_segments,
        }
        rows.append(row)
        # Print row detail
        info = " | ".join(f"{k}: {v}" for k, v in row.items())
        print(info)

    return rows


def main():
    # Load ground-truth labels
    with open(VIDEO_CATEGORIES_FILE, 'r', encoding='utf-8') as f:
        video_categories = json.load(f)

    # Prepare list of videos
    video_files = [f for f in os.listdir(VIDEO_FOLDER)
                   if f.endswith('.mp4') and f in video_categories]

    # Build combo tasks
    combos = []
    for model_id in MODEL_LIST:
        for template_name, template in TEMPLATES.items():
            for num_seg in NUM_SEGMENTS_LIST:
                combos.append({
                    "model_id": model_id,
                    "template": template,
                    "template_name": template_name,
                    "num_segments": num_seg,
                    "video_folder": VIDEO_FOLDER,
                    "video_files": video_files,
                    "video_categories": video_categories,
                })

    # Parallel execution per combo
    all_rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_combo = {executor.submit(run_combo, combo): combo for combo in combos}
        for future in concurrent.futures.as_completed(future_to_combo):
            combo_rows = future.result()
            all_rows.extend(combo_rows)

    # Save consolidated results to CSV
    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
