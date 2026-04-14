import os
import re
import json
import subprocess
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
        self.cache = {}  

    def query_ollama(self, prompt, image_path):
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise cursor localization assistant. Output ONLY coordinates in (x, y) format."
                    },
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
                x = float(match.group(1))
                y = float(match.group(2))
                if 0 <= x <= 1 and 0 <= y <= 1:
                    return x, y
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
        files = [
            f for f in files
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.mp3', '.wav'))
        ]

        files = sorted(
            files,
            key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
        )

        return [os.path.join(folder, f) for f in files]

    def generate_cursor(self, subtitles_json):
        slide_images = self.get_sorted_files(self.images_dir)
        slide_audios = self.get_sorted_files(self.audio_dir)

        timeline = []
        global_time = 0
        MAX_RETRIES = 3

        slide_keys = sorted(
            subtitles_json.keys(),
            key=lambda x: int(re.search(r'\d+', x).group())
        )

        for idx, slide_key in enumerate(slide_keys):
            if idx >= len(slide_images) or idx >= len(slide_audios):
                print(f"Warning: Missing image/audio for slide {idx}")
                continue

            slide = subtitles_json[slide_key]
            image_path = slide_images[idx]
            audio_path = slide_audios[idx]

            img = cv2.imread(image_path)
            if img is None:
                print(f"Error reading image {image_path}")
                continue

            h, w = img.shape[:2]

            audio_duration = self.get_audio_duration(audio_path)
            items = slide["items"]
            num_items = len(items)

            if num_items == 0:
                global_time += audio_duration
                continue

            duration_per_item = audio_duration / num_items

            for i, item in enumerate(items):
                focus = item["focus"]

                print(f"Slide {idx+1} | Item {i+1}: {focus}")

                if focus in self.cache:
                    ex, ey = self.cache[focus]
                else:
                    current_prompt = PROMPT.replace("{focus}", focus)

                    point = None
                    for attempt in range(MAX_RETRIES):
                        response_text = self.query_ollama(current_prompt, image_path)
                        extracted = self.extract_point(response_text)

                        if extracted is not None:
                            ex, ey = extracted
                            point = (ex, ey)
                            break

                        print(f"  Attempt {attempt+1} failed, retrying...")

                        current_prompt = (
                            f"{PROMPT.replace('{focus}', focus)}\n\n"
                            f"CRITICAL:\n"
                            f"- Return ONLY (x, y)\n"
                            f"- Coordinates must be between 0 and 1\n"
                            f"- Do NOT return pixel values\n"
                        )

                    if point is None:
                        print("  Using fallback center")
                        ex, ey = 0.5, 0.5

                    self.cache[focus] = (ex, ey)

                px = int(ex * w)
                py = int(ey * h)

                if timeline:
                    prev_x, prev_y = timeline[-1]["cursor"]
                    alpha = 0.6
                    px = int(alpha * px + (1 - alpha) * prev_x)
                    py = int(alpha * py + (1 - alpha) * prev_y)

                start_time = global_time + i * duration_per_item
                end_time = start_time + duration_per_item

                timeline.append({
                    "start": round(start_time, 3),
                    "end": round(end_time, 3),
                    "cursor": [px, py],
                    "focus": focus
                })

            global_time += audio_duration

        return timeline

    def run(self, input_json_path, output_json_path):
        print("Starting CursorAgent...\n")

        if not os.path.exists(input_json_path):
            print(f"Error: {input_json_path} not found.")
            return

        with open(input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        timeline = self.generate_cursor(data)

        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(timeline, f, indent=2, ensure_ascii=False)

        print(f"\nSaved to: {output_json_path}")
        print("Done ✅")

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