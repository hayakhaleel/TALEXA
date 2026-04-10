import os
import glob
import subprocess
import time
from pathlib import Path

import httpx


class TalkingHeadAgent:
    def __init__(
        self,
        source_image="Data/input/ref_face.png",
        speech_dir="Data/intermediate/speech",
        output_dir="Data/intermediate/talking_head/lecture1",
        heygen_api_key=None,
        talking_photo_id=None,
        audio_asset_id=None,
        audio_url=None,
        avatar_group_name="talking_head_user",
        poll_interval_s=5,
        max_wait_s=1800,
    ):
        self.source_image = os.path.abspath(source_image) if source_image else None
        self.speech_dir = os.path.abspath(speech_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.heygen_api_key = heygen_api_key or os.getenv("HEYGEN_API_KEY")
        self.talking_photo_id = talking_photo_id or os.getenv("HEYGEN_TALKING_PHOTO_ID")
        self.audio_asset_id = audio_asset_id or os.getenv("HEYGEN_AUDIO_ASSET_ID")
        self.audio_url = audio_url or os.getenv("HEYGEN_AUDIO_URL")
        self.avatar_group_name = avatar_group_name
        self.poll_interval_s = poll_interval_s
        self.max_wait_s = max_wait_s

        os.makedirs(self.output_dir, exist_ok=True)

    def get_audio_files(self):
        return sorted(glob.glob(os.path.join(self.speech_dir, "slide_*.wav")))

    def _require_api_settings(self):
        if not self.heygen_api_key:
            raise ValueError("HEYGEN_API_KEY is not set.")
        if not self.talking_photo_id and not os.path.exists(self.source_image):
            raise ValueError(
                "Provide HEYGEN_TALKING_PHOTO_ID or a valid source_image to create one."
            )

    def _convert_audio_to_mp3(self, audio_path, slide_dir):
        if audio_path.lower().endswith(".mp3"):
            return audio_path

        mp3_path = os.path.join(slide_dir, f"{Path(audio_path).stem}.mp3")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            audio_path,
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "2",
            mp3_path,
        ]
        subprocess.run(cmd, check=True)
        return mp3_path

    def _merge_audio_files(self, audio_files, output_path):
        if not audio_files:
            raise ValueError("No audio files to merge.")

        list_path = os.path.join(os.path.dirname(output_path), "concat_list.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for path in audio_files:
                f.write(f"file '{path.replace(\"'\", \"'\\\\''\")}'\n")

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-c:a",
            "pcm_s16le",
            output_path,
        ]
        subprocess.run(cmd, check=True)
        return output_path

    def _upload_audio_asset(self, client, audio_path):
        with open(audio_path, "rb") as f:
            response = client.post(
                "https://upload.heygen.com/v1/asset",
                headers={
                    "X-Api-Key": self.heygen_api_key,
                    "Content-Type": "audio/mpeg",
                },
                content=f.read(),
            )
        response.raise_for_status()
        payload = response.json()

        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        asset_id = (
            data.get("id")
            or data.get("asset_id")
            or data.get("assetId")
            or payload.get("id")
        )
        if not asset_id:
            raise ValueError(f"Unexpected upload response: {payload}")

        return asset_id

    def _upload_image_asset(self, client, image_path):
        ext = os.path.splitext(image_path)[1].lower()
        content_type = "image/png" if ext == ".png" else "image/jpeg"
        with open(image_path, "rb") as f:
            response = client.post(
                "https://upload.heygen.com/v1/asset",
                headers={
                    "X-Api-Key": self.heygen_api_key,
                    "Content-Type": content_type,
                },
                content=f.read(),
            )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        image_key = data.get("image_key") or data.get("key") or payload.get("image_key")
        if not image_key:
            raise ValueError(f"Unexpected image upload response: {payload}")
        return image_key

    def _create_photo_avatar_group(self, client, image_key):
        payload = {"name": self.avatar_group_name, "image_key": image_key}
        response = client.post(
            "https://api.heygen.com/v2/photo_avatar/avatar_group/create",
            headers={
                "X-Api-Key": self.heygen_api_key,
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        group_id = (
            data.get("data", {}).get("group_id")
            or data.get("data", {}).get("id")
            or data.get("group_id")
        )
        if not group_id:
            raise ValueError(f"Unexpected avatar group response: {data}")
        return group_id

    def _get_talking_photo_id_from_group(self, client, group_id):
        response = client.get(
            f"https://api.heygen.com/v2/avatar_group/{group_id}/avatars",
            headers={"X-Api-Key": self.heygen_api_key},
        )
        response.raise_for_status()
        data = response.json()
        avatars = data.get("data") or data.get("avatars") or []
        if not avatars:
            return None
        first = avatars[0]
        return first.get("id") or first.get("avatar_id")

    def _resolve_talking_photo_id(self, client):
        if self.talking_photo_id:
            return self.talking_photo_id

        image_key = self._upload_image_asset(client, self.source_image)
        group_id = self._create_photo_avatar_group(client, image_key)

        deadline = time.time() + self.max_wait_s
        while time.time() < deadline:
            talking_photo_id = self._get_talking_photo_id_from_group(client, group_id)
            if talking_photo_id:
                self.talking_photo_id = talking_photo_id
                return talking_photo_id
            time.sleep(self.poll_interval_s)

        raise TimeoutError("Timed out waiting for talking_photo_id to become available.")

    def _create_video(self, client, audio_asset_id, audio_url, title):
        talking_photo_id = self._resolve_talking_photo_id(client)
        voice = {"type": "audio"}
        if audio_asset_id:
            voice["audio_asset_id"] = audio_asset_id
        elif audio_url:
            voice["audio_url"] = audio_url
        else:
            raise ValueError("Either audio_asset_id or audio_url must be provided.")

        payload = {
            "title": title,
            "video_inputs": [
                {
                    "character": {
                        "type": "talking_photo",
                        "talking_photo_id": talking_photo_id,
                    },
                    "voice": voice,
                }
            ],
        }
        response = client.post(
            "https://api.heygen.com/v2/video/generate",
            headers={
                "X-Api-Key": self.heygen_api_key,
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        video_id = (
            data.get("data", {}).get("video_id")
            or data.get("data", {}).get("id")
            or data.get("video_id")
        )
        if not video_id:
            raise ValueError(f"Unexpected create response: {data}")
        return video_id

    def _poll_video(self, client, video_id):
        deadline = time.time() + self.max_wait_s
        while time.time() < deadline:
            response = client.get(
                "https://api.heygen.com/v1/video_status.get",
                params={"video_id": video_id},
                headers={"X-Api-Key": self.heygen_api_key},
            )
            response.raise_for_status()
            data = response.json().get("data", {})
            status = data.get("status")

            if status == "completed":
                return data
            if status == "failed":
                error = data.get("error") or "Unknown error"
                raise RuntimeError(f"HeyGen video failed: {error}")

            time.sleep(self.poll_interval_s)

        raise TimeoutError(f"Timed out waiting for video {video_id}")

    def run_one(self, audio_path, output_name="lecture1"):
        slide_name = Path(audio_path).stem
        slide_dir = self.output_dir
        os.makedirs(slide_dir, exist_ok=True)
        self._require_api_settings()

        with httpx.Client(timeout=300) as client:
            audio_asset_id = self.audio_asset_id
            audio_url = self.audio_url
            if not audio_asset_id and not audio_url:
                mp3_path = self._convert_audio_to_mp3(audio_path, slide_dir)
                audio_asset_id = self._upload_audio_asset(client, mp3_path)
            video_id = self._create_video(
                client,
                audio_asset_id=audio_asset_id,
                audio_url=audio_url,
                title=slide_name,
            )
            status_data = self._poll_video(client, video_id)
            video_url = status_data.get("video_url")
            if not video_url:
                raise ValueError(f"Missing video_url for {video_id}")

            final_video = os.path.join(self.output_dir, f"{output_name}.mp4")
            with client.stream("GET", video_url) as response:
                response.raise_for_status()
                with open(final_video, "wb") as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)

        print(f"[TalkingHead] Saved: {final_video}")

    def run(self, limit_slides=None):
        if not self.source_image and not self.talking_photo_id:
            raise ValueError("Source image or HEYGEN_TALKING_PHOTO_ID is required.")
        if self.source_image and not os.path.exists(self.source_image):
            raise FileNotFoundError(f"Source image not found: {self.source_image}")

        if not os.path.isdir(self.speech_dir):
            raise FileNotFoundError(f"Speech directory not found: {self.speech_dir}")

        audio_files = self.get_audio_files()

        if not audio_files:
            raise FileNotFoundError(f"No slide wav files found in: {self.speech_dir}")

        if limit_slides is not None:
            audio_files = audio_files[:limit_slides]

        print(f"[TalkingHead] Slides to merge: {len(audio_files)}")

        os.makedirs(self.output_dir, exist_ok=True)
        merged_wav = os.path.join(self.output_dir, "merged.wav")
        self._merge_audio_files(audio_files, merged_wav)

        output_name = os.path.basename(self.output_dir.rstrip(os.sep)) or "lecture1"
        print(f"[TalkingHead] Generating video for: {output_name}")
        self.run_one(merged_wav, output_name=output_name)

        print("[TalkingHead] All done.")


if __name__ == "__main__":
    agent = TalkingHeadAgent(
        source_image="Data/input/ref_face.png",
        speech_dir="Data/intermediate/speech",
        output_dir="Data/intermediate/talking_head/lecture1",
        heygen_api_key=os.getenv("sk_V2_hgu_kGc48dy2AJZ_vWopK7Gxdq0CurG3uryonH2mprn7NC1R"),
        talking_photo_id=os.getenv("HEYGEN_TALKING_PHOTO_ID"),
    )
    agent.run()
