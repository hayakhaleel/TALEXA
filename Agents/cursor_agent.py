import os
import re
import json
import subprocess
from pathlib import Path
import cv2
import ollama
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Prompts.cursor_prompt import PROMPT


class CursorAgent:

    def __init__(
        self,
        model="qwen2.5:7b",
        images_dir="Data/intermediate/slide_images",
        audio_dir="Data/intermediate/slide_audios"
    ):
        self.model = model
        self.images_dir = images_dir
        self.audio_dir = audio_dir

    def query_ollama(self, prompt, image_path):
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a cursor prediction assistant. Respond only with coordinates in (x, y) format."},
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [str(image_path)]
                    }
                ],
            )
            return response["message"]["content"]
        except Exception as e:
            print(f"Error querying Ollama: {e}")
            return ""

    def extract_point(self, text):
        match = re.search(r'\(?\s*([\d.]+)\s*,\s*([\d.]+)\s*\)?', text)
        if match:
            try:
                return float(match.group(1)), float(match.group(2))
            except ValueError:
                return None
        return None

    def get_audio_duration(self, audio_path):
        cmd = ["ffmpeg", "-i", audio_path]
        result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
        for line in result.stderr.splitlines():
            if "Duration" in line:
                duration = line.split("Duration:")[1].split(",")[0].strip()
                h, m, s = map(float, duration.split(":"))
                return h * 3600 + m * 60 + s
        return 0

    def get_sorted_files(self, folder):
        if not os.path.exists(folder):
            print(f"Error: Folder {folder} does not exist.")
            return []
        files = os.listdir(folder)
        files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.mp3', '.wav'))]
        files = sorted(files, key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)
        return [os.path.join(folder, f) for f in files]

    def generate_cursor(self, subtitles_json):
        slide_images = self.get_sorted_files(self.images_dir)
        slide_audios = self.get_sorted_files(self.audio_dir)

        timeline = []
        global_time = 0
        MAX_RETRIES = 3

        slide_keys = sorted(subtitles_json.keys(), key=lambda x: int(re.search(r'\d+', x).group()))

        for idx, slide_key in enumerate(slide_keys):
            if idx >= len(slide_images) or idx >= len(slide_audios):
                print(f"Warning: Missing image or audio for slide index {idx}. Skipping.")
                continue

            slide = subtitles_json[slide_key]
            image_path = slide_images[idx]
            audio_path = slide_audios[idx]

            img = cv2.imread(image_path)
            if img is None:
                print(f"Error: Could not read image {image_path}")
                continue
            h, w = img.shape[:2]
            center_point = (w // 2, h // 2)

            audio_duration = self.get_audio_duration(audio_path)
            items = slide["items"]
            num_items = len(items)

            if num_items == 0:
                global_time += audio_duration
                continue

            duration_per_item = audio_duration / num_items

            for i, item in enumerate(items):
                focus = item["focus"]
                current_prompt = PROMPT.replace("{focus}", focus)
                
                point = None
                print(f"Slide {idx+1} | Item {i+1}: {focus}")

                for attempt in range(MAX_RETRIES):
                    response_text = self.query_ollama(current_prompt, image_path)
                    extracted = self.extract_point(response_text)
                    if extracted is not None:
                        ex, ey = extracted
                        if 0 <= ex <= w and 0 <= ey <= h:
                            point = (ex, ey)
                            break                     
                    print(f"  [Attempt {attempt+1}] Invalid/Out-of-bounds. Retrying...")
                    current_prompt = (
                        f"{PROMPT.replace('{focus}', focus)}\n\n"
                        f"CRITICAL: Your previous answer was invalid. "
                        f"Provide ONLY (x, y) coordinates. The image size is {w}x{h}. "
                        f"Ensure 0 <= x <= {w} and 0 <= y <= {h}."
                    )

                if point is None:
                    print(f"  [Warning] All retries failed. Defaulting to center {center_point}")
                    point = center_point

                start_time = global_time + i * duration_per_item
                end_time = start_time + duration_per_item

                timeline.append({
                    "start": round(start_time, 3),
                    "end": round(end_time, 3),
                    "cursor": [point[0], point[1]],
                    "focus": focus
                })

            global_time += audio_duration

        return timeline

    def run(self, input_json_path, output_json_path):
        print("Starting CursorAgent with Validation Loop...\n")

        if not os.path.exists(input_json_path):
            print(f"Error: Input file {input_json_path} not found.")
            return

        with open(input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        timeline = self.generate_cursor(data)

        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(timeline, f, indent=2, ensure_ascii=False)

        print(f"\nCursor JSON saved at: {output_json_path}")
        print("CursorAgent completed successfully.")


if __name__ == "__main__":
    agent = CursorAgent(
        model="qwen2.5:7b",
        images_dir="Data/intermediate/slide_images",
        audio_dir="Data/intermediate/slide_audios"
    )

    agent.run(
        input_json_path="Data/intermediate/subtitles.json",
        output_json_path="Data/intermediate/cursor.json"
    )
