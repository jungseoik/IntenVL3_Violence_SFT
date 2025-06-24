import os
import json
import pandas as pd
from internvl_inferencer.vl3 import InternVL3Inferencer
from assets.config import NUM_SEGMENTS_LIST, VIDEO_CLASSIFICATION_PROMPT, MODEL_LIST, TEMPLATES, VIDEO_FOLDER ,VIDEO_CATEGORIES_FILE,OUTPUT_CSV, MAX_WORKERS
from typing import List, Dict, Any
import concurrent.futures

def run_combo(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Run inference over a set of videos using a specific combination of
    model, template, and number of segments.

    Args:
        task (dict): Dictionary containing the following keys:
            - model_id (str): Model identifier to be used for inference.
            - template (str): Prompt template for video classification.
            - template_name (str): Descriptive name for the template.
            - num_segments (int): Number of segments to split the video into for inference.
            - video_folder (str): Path to the folder containing video files.
            - video_files (List[str]): List of video filenames to process.
            - video_categories (Dict[str, str]): Ground-truth category labels for each video.

    Returns:
        List[Dict[str, Any]]: List of dictionaries, one per video, each containing:
            - video_name (str)
            - ground_truth (str)
            - model_name (str)
            - template_type (str)
            - predicted_category (Optional[str])
            - response (Any): Raw model response
            - num_segment (int)
    """
    model_id = task["model_id"]
    template = task["template"]
    template_name = task["template_name"]
    num_segments = task["num_segments"]
    video_folder = task["video_folder"]
    video_files = task["video_files"]
    video_categories = task["video_categories"]

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

        info = " | ".join(f"{k}: {v}" for k, v in row.items())
        print(info)

    return rows


def eval() -> None:
    """
    Evaluate all model-template-segment combinations over a set of videos.
    Loads video ground-truth labels, builds inference tasks, executes them in parallel,
    collects all results, and writes them to a CSV file.

    Steps:
        1. Load ground truth labels from VIDEO_CATEGORIES_FILE.
        2. Filter video files in VIDEO_FOLDER that have labels.
        3. Generate all combinations of (model, template, segment count).
        4. Run inference in parallel using multiprocessing.
        5. Aggregate all results into a DataFrame and save to OUTPUT_CSV.

    Returns:
        None
    """
    with open(VIDEO_CATEGORIES_FILE, 'r', encoding='utf-8') as f:
        video_categories = json.load(f)

    video_files = [f for f in os.listdir(VIDEO_FOLDER)
                   if f.endswith('.mp4') and f in video_categories]

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

    all_rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_combo = {executor.submit(run_combo, combo): combo for combo in combos}
        for future in concurrent.futures.as_completed(future_to_combo):
            combo_rows = future.result()
            all_rows.extend(combo_rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"Results saved to {OUTPUT_CSV}")
