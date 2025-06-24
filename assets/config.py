VIDEO_CLASSIFICATION_PROMPT = """
Classify the situation into detailed categories. Respond only in JSON format: {"category": "category_name"}

Category examples:
- Normal
- Abuse
- Arrest
- Arson
- Assault
- Road_Accident
- Burglary
- Explosion
- Fighting
- Robbery
- Shooting
- Stealing
- Shoplifting
- Vandalism
- Other_anomaly

Response examples:
{"category": "Normal"}
{"category": "Fighting"}
{"category": "Shoplifting"}"""

VIDEO_BINARY_CLASSIFICATION_PROMPT = """
Classify the situation as normal or abnormal. Respond only in JSON format: {"category": "normal" | "abnormal"}

Examples:
- Person walking in mall → {"category": "normal"}
- People fighting → {"category": "abnormal"}"""

VIDEO_DESCRIPTION = "explain this video"
VIDEO_JUDGEMENT = "Detect and classify the anomaly events in the video."


CATEGORIES = [
    "Normal",
    "Abuse", 
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
    "RoadAccidents"
]


# Experiment settings
MODEL_LIST = [
    "backseollgi/InternVL3-2B_lora_HIVAU_UCF-crime_24_mf"
]
# TEMPLATES = {
#     "binary": VIDEO_BINARY_CLASSIFICATION_PROMPT,
#     "classification": VIDEO_CLASSIFICATION_PROMPT,
#     "description": VIDEO_DESCRIPTION,
#     "judgement": VIDEO_JUDGEMENT,
# }
TEMPLATES = {
    "binary": VIDEO_BINARY_CLASSIFICATION_PROMPT,
    "classification": VIDEO_CLASSIFICATION_PROMPT,
}
NUM_SEGMENTS_LIST = [8, 12, 20, 24]
VIDEO_FOLDER = "test/"
VIDEO_CATEGORIES_FILE = "assets/video_categories.json"
OUTPUT_CSV = "results.csv"
MAX_WORKERS = 3  # Number of parallel inference processes