import json
from pathlib import Path
import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id_hex = parts[0]
        view_index = int(parts[1])
        return frame_id_hex, view_index
    return "00000", 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    img_width, img_height = pil_image.size
    draw = ImageDraw.Draw(pil_image)

    with open(info_path) as f:
        info = json.load(f)

    _, view_index = extract_frame_info(image_path)

    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        return np.array(pil_image)

    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        x1_scaled, y1_scaled = int(x1 * scale_x), int(y1 * scale_y)
        x2_scaled, y2_scaled = int(x2 * scale_x), int(y2 * scale_y)

        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue
        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        color = (255, 0, 0) if track_id == 0 else COLORS.get(class_id, (255, 255, 255))
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    return np.array(pil_image)


def extract_kart_objects(info_path: str, view_index: int) -> list:
    """
    Extract kart objects from the info.json file, including their center points.
    Filters out karts that are out of sight.

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze

    Returns:
        List of kart objects, each a dictionary containing details like instance_id, kart_name, center, and if it's the ego car.
    """
    with open(info_path) as f:
        info = json.load(f)

    kart_objects = []
    if view_index >= len(info["detections"]):
        return kart_objects

    frame_detections = info["detections"][view_index]
    kart_names = info.get("names", {})

    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = map(int, detection)

        if class_id == 1:  # Filter for karts
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            kart_objects.append({
                "instance_id": track_id,
                "kart_name": kart_names.get(str(track_id), f"Kart {track_id}"),
                "center": (center_x, center_y),
                "is_ego_car": track_id == 0
            })
            
    return kart_objects


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """
    with open(info_path) as f:
        info = json.load(f)
    return info.get("track", "unknown track")


def generate_qa_pairs(info_path: str, view_index: int) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze

    Returns:
        List of dictionaries, each containing a question and answer.
    """
    info_path = Path(info_path)
    frame_id_hex, _ = extract_frame_info(f"{info_path.stem.replace('_info', '')}_{view_index:02d}_im.jpg")
    relative_image_path = f"{info_path.parent.name}/{frame_id_hex}_{view_index:02d}_im.jpg"

    karts = extract_kart_objects(info_path, view_index)
    track_name = extract_track_info(info_path)
    
    qa_pairs = []
    ego_car = next((kart for kart in karts if kart["is_ego_car"]), None)

    if not ego_car:
        return []

    # 1. Ego car question
    qa_pairs.append({"question": "What kart is the ego car?", "answer": ego_car["kart_name"], "image_file": relative_image_path})

    # 2. Total karts question
    qa_pairs.append({"question": "How many karts are there in the scenario?", "answer": str(len(karts)), "image_file": relative_image_path})

    # 3. Track information question
    qa_pairs.append({"question": "What track is this?", "answer": track_name, "image_file": relative_image_path})

    # Relative position and counting questions
    other_karts = [kart for kart in karts if not kart["is_ego_car"]]
    left_karts, right_karts, front_karts, behind_karts = 0, 0, 0, 0

    ego_cx, ego_cy = ego_car["center"]

    for kart in other_karts:
        other_cx, other_cy = kart["center"]
        
        # 4. Relative position questions
        # Using x-coordinate for left/right and y-coordinate for front/behind
        # Lower 'y' is considered 'in front' from a typical 3rd person racing view
        is_left = other_cx < ego_cx
        is_front = other_cy < ego_cy

        if is_left:
            left_karts += 1
            qa_pairs.append({"question": f"Is {kart['kart_name']} to the left or right of the ego car?", "answer": "left", "image_file": relative_image_path})
        else:
            right_karts += 1
            qa_pairs.append({"question": f"Is {kart['kart_name']} to the left or right of the ego car?", "answer": "right", "image_file": relative_image_path})
            
        if is_front:
            front_karts += 1
            qa_pairs.append({"question": f"Is {kart['kart_name']} in front of or behind the ego car?", "answer": "front", "image_file": relative_image_path})
        else:
            behind_karts += 1
            qa_pairs.append({"question": f"Is {kart['kart_name']} in front of or behind the ego car?", "answer": "behind", "image_file": relative_image_path})

    # 5. Counting questions
    qa_pairs.append({"question": "How many karts are to the left of the ego car?", "answer": str(left_karts), "image_file": relative_image_path})
    qa_pairs.append({"question": "How many karts are to the right of the ego car?", "answer": str(right_karts), "image_file": relative_image_path})
    qa_pairs.append({"question": "How many karts are in front of the ego car?", "answer": str(front_karts), "image_file": relative_image_path})
    qa_pairs.append({"question": "How many karts are behind the ego car?", "answer": str(behind_karts), "image_file": relative_image_path})
    
    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.
    """
    info_path = Path(info_file)
    frame_id_hex, _ = extract_frame_info(f"{info_path.stem.replace('_info', '')}_{view_index:02d}_im.jpg")
    
    # Find corresponding image file in the same directory as the info file
    image_file = next(info_path.parent.glob(f"{frame_id_hex}_{view_index:02d}_im.jpg"))

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    frame_id_dec = int(frame_id_hex, 16)
    plt.title(f"Frame {frame_id_dec}, View {view_index}")
    plt.show()

    qa_pairs = generate_qa_pairs(info_file, view_index)

    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)

def generate(data_split: str = "train", output_file: str = "generated_qa_pairs.json"):
    """
    Generate a full QA dataset from a data split and save it to a file.

    Args:
        data_split: The data split to process (e.g., 'train', 'valid').
        output_file: The name of the JSON file to save the pairs to.
    """
    print(f"Generating QA pairs for '{data_split}' split...")
    data_dir = Path(__file__).parent.parent / "data" / data_split
    info_files = list(data_dir.glob("*_info.json"))
    
    all_qa_pairs = []

    for info_file in tqdm(info_files, desc="Processing info files"):
        with open(info_file) as f:
            num_views = len(json.load(f)["detections"])
        
        for view_index in range(num_views):
            qa_pairs = generate_qa_pairs(str(info_file), view_index)
            all_qa_pairs.extend(qa_pairs)

    output_path = data_dir / output_file
    with open(output_path, "w") as f:
        json.dump(all_qa_pairs, f, indent=2)

    print(f"Successfully generated {len(all_qa_pairs)} QA pairs.")
    print(f"Dataset saved to: {output_path}")


if __name__ == "__main__":
    fire.Fire({
        "check": check_qa_pairs,
        "generate": generate
    })
