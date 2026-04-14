import os
import re
import json
import math
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
import whisperx
from f5_tts.api import F5TTS


class SpeechAgent:
    def __init__(
        self,
        subtitles_json_path: str = "Data/intermediate/subtitles.json",
        ref_audio_path: str = "Data/input/ref_clean.wav",
        output_dir: str = "Data/output/speech/lecture1",
        manual_ref_text: Optional[str] = None,
        language: str = "en",
        device: Optional[str] = None,
        max_chunk_chars: int = 220,
        silence_between_chunks_ms: int = 180,
    ):
        self.subtitles_json_path = subtitles_json_path
        self.ref_audio_path = ref_audio_path
        self.output_dir = output_dir
        self.manual_ref_text = manual_ref_text
        self.language = language
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_chunk_chars = max_chunk_chars
        self.silence_between_chunks_ms = silence_between_chunks_ms

        os.makedirs(self.output_dir, exist_ok=True)

        self._patch_torchaudio_load()
        self.tts = F5TTS()
        self.ref_text: Optional[str] = None

    def _patch_torchaudio_load(self):
        def safe_torchaudio_load(path, *args, **kwargs):
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim == 1:
                audio = np.expand_dims(audio, axis=0)
            else:
                audio = audio.T
            return torch.from_numpy(audio), sr

        torchaudio.load = safe_torchaudio_load

    def transcribe_reference_with_whisperx(self, audio_path: str) -> str:
        print(f"[WhisperX] Loading model on {self.device}...")
        compute_type = "float16" if self.device == "cuda" else "int8"

        model = whisperx.load_model(
            "large-v2",
            device=self.device,
            compute_type=compute_type,
        )

        result = model.transcribe(audio_path, language=self.language)

        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=self.device,
        )

        aligned = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio_path,
            self.device,
        )

        text = " ".join(seg["text"].strip() for seg in aligned["segments"])
        text = self.clean_ref_text(text)
        print(f"[WhisperX] Reference text: {text}")
        return text

    def load_subtitles(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.subtitles_json_path):
            raise FileNotFoundError(f"subtitles.json not found: {self.subtitles_json_path}")

        with open(self.subtitles_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        slides = []

        if isinstance(data, dict):
            sorted_keys = sorted(
                data.keys(),
                key=lambda x: int("".join(filter(str.isdigit, str(x))) or 0)
            )

            for idx, key in enumerate(sorted_keys, start=1):
                slide = data[key]
                slide_number = slide.get("slide_number", idx)
                image = slide.get("image", "")
                items = slide.get("items", [])

                text_parts = []
                for item in items:
                    sentence = self.clean_slide_text(item.get("sentence", ""))
                    if sentence:
                        text_parts.append(sentence)

                full_text = " ".join(text_parts).strip()
                if full_text:
                    slides.append({
                        "slide_id": slide_number,
                        "image": image,
                        "text": full_text,
                    })

            return slides

        if isinstance(data, list):
            for idx, slide in enumerate(data, start=1):
                slide_id = slide.get("slide_id", idx)
                image = slide.get("image", "")
                text_parts = []

                if isinstance(slide.get("segments"), list):
                    for seg in slide["segments"]:
                        sentence = self.clean_slide_text(seg.get("text", ""))
                        if sentence:
                            text_parts.append(sentence)
                elif "spoken_text" in slide:
                    spoken = self.clean_slide_text(slide.get("spoken_text", ""))
                    if spoken:
                        text_parts.append(spoken)
                elif "text" in slide:
                    spoken = self.clean_slide_text(slide.get("text", ""))
                    if spoken:
                        text_parts.append(spoken)

                full_text = " ".join(text_parts).strip()
                if full_text:
                    slides.append({
                        "slide_id": slide_id,
                        "image": image,
                        "text": full_text,
                    })

            return slides

        raise ValueError("Unsupported subtitles.json format")

    def clean_ref_text(self, text: str) -> str:
        text = "" if text is None else str(text)
        text = text.replace("\n", " ")
        text = text.replace("•", " ")
        text = text.replace("—", " ")
        text = text.replace("–", " ")
        text = text.replace("→", " to ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def clean_slide_text(self, text: str) -> str:
        text = "" if text is None else str(text)

        replacements = {
            "\n": " ",
            "•": " ",
            "—": " ",
            "–": " ",
            "→": " to ",
            "&": " and ",
            "%": " percent",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        text = re.sub(r"[^\S\r\n]+", " ", text)
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        text = re.sub(r"([,.!?;:]){2,}", r"\1", text)
        text = text.replace("First,", "So first,")
        text = text.replace("Second,", "Then,")
        text = text.replace("Third,", "Next,")
        text = text.replace("Finally,", "And finally,")

        return text.strip()

    def simplify_text_for_retry(self, text: str) -> str:
        text = self.clean_slide_text(text)
        text = re.sub(r"\([^)]*\)", "", text)
        text = re.sub(r"\[[^\]]*\]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def split_into_chunks(self, text: str, max_chars: Optional[int] = None) -> List[str]:
        max_chars = max_chars or self.max_chunk_chars
        text = self.clean_slide_text(text)

        if len(text) <= max_chars:
            return [text]

        sentence_parts = re.split(r"(?<=[.!?])\s+", text)
        sentence_parts = [s.strip() for s in sentence_parts if s.strip()]

        chunks = []
        current = ""

        for sentence in sentence_parts:
            if len(sentence) > max_chars:
                subparts = re.split(r"(?<=[,;:])\s+", sentence)
                subparts = [s.strip() for s in subparts if s.strip()]
            else:
                subparts = [sentence]

            for part in subparts:
                if not current:
                    current = part
                elif len(current) + 1 + len(part) <= max_chars:
                    current += " " + part
                else:
                    chunks.append(current.strip())
                    current = part

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def read_audio(self, path: str) -> Tuple[np.ndarray, int]:
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        return audio, sr

    def write_audio(self, path: str, audio: np.ndarray, sr: int):
        sf.write(path, audio, sr)

    def make_silence(self, sr: int, ms: int) -> np.ndarray:
        samples = int(sr * (ms / 1000.0))
        return np.zeros(samples, dtype=np.float32)

    def merge_wavs(self, input_files: List[str], output_file: str):
        merged_audio = []
        target_sr = None

        for i, wav_path in enumerate(input_files):
            audio, sr = self.read_audio(wav_path)
            if target_sr is None:
                target_sr = sr
            elif sr != target_sr:
                raise ValueError(
                    f"Sample rate mismatch while merging: {wav_path} has {sr}, expected {target_sr}"
                )

            merged_audio.append(audio)

            if i < len(input_files) - 1 and self.silence_between_chunks_ms > 0:
                merged_audio.append(self.make_silence(target_sr, self.silence_between_chunks_ms))

        final_audio = np.concatenate(merged_audio, axis=0)
        self.write_audio(output_file, final_audio, target_sr)

    def audio_is_valid(self, wav_path: str, min_duration_sec: float = 0.45) -> bool:
        if not os.path.exists(wav_path):
            return False

        try:
            audio, sr = self.read_audio(wav_path)
            if len(audio) == 0:
                return False

            duration = len(audio) / sr
            rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0

            if duration < min_duration_sec:
                return False
            if math.isnan(rms) or rms < 1e-5:
                return False

            return True
        except Exception:
            return False

    def generate_chunk(self, text_prompt: str, save_path: str, retry_index: int = 0):
        self.tts.infer(
            ref_file=self.ref_audio_path,
            ref_text=self.ref_text,
            gen_text=text_prompt,
            file_wave=save_path,
        )

    def generate_chunk_with_retry(self, chunk: str, save_path: str) -> bool:
        attempts = [
            chunk,
            self.simplify_text_for_retry(chunk),
            self.simplify_text_for_retry(chunk).replace(",", "."),
        ]

        for i, attempt_text in enumerate(attempts, start=1):
            if not attempt_text.strip():
                continue

            print(f"[Speech] Attempt {i}: {attempt_text}")

            try:
                self.generate_chunk(attempt_text, save_path, retry_index=i - 1)
                if self.audio_is_valid(save_path):
                    return True
            except Exception as e:
                print(f"[Speech] Attempt {i} failed: {e}")

        return False

    def prepare_reference(self):
        if not os.path.exists(self.ref_audio_path):
            raise FileNotFoundError(f"Reference audio not found: {self.ref_audio_path}")

        if self.manual_ref_text and self.manual_ref_text.strip():
            self.ref_text = self.clean_ref_text(self.manual_ref_text)
            print(f"[Speech] Using manual reference text: {self.ref_text}")
        else:
            print("[Speech] No manual ref_text provided. Using WhisperX...")
            self.ref_text = self.transcribe_reference_with_whisperx(self.ref_audio_path)

        if not self.ref_text or len(self.ref_text.split()) < 2:
            raise ValueError("Reference text is too short or invalid. Please provide a better ref_audio or manual_ref_text.")

    def tts_per_slide(self, limit_slides: Optional[int] = None):
        self.prepare_reference()
        slides = self.load_subtitles()

        if limit_slides is not None:
            slides = slides[:limit_slides]

        print(f"[Speech] Slides to process: {len(slides)}")

        for slide in slides:
            slide_id = slide["slide_id"]
            slide_text = self.clean_slide_text(slide["text"])

            if not slide_text:
                print(f"[Speech] Skipping slide {slide_id}: empty text")
                continue

            chunks = self.split_into_chunks(slide_text, self.max_chunk_chars)
            print(f"\n[Speech] Processing slide {slide_id}")
            print(f"[Speech] Number of chunks: {len(chunks)}")

            temp_files = []
            all_good = True

            for idx, chunk in enumerate(chunks, start=1):
                temp_path = os.path.join(self.output_dir, f"temp_slide_{slide_id}_{idx}.wav")
                print(f"[Speech] Slide {slide_id} | chunk {idx}/{len(chunks)}")
                print(chunk)

                success = self.generate_chunk_with_retry(chunk, temp_path)
                if not success:
                    print(f"[Speech] Failed chunk on slide {slide_id}: {idx}")
                    all_good = False
                    break

                temp_files.append(temp_path)

            if not all_good:
                print(f"[Speech] Skipping final save for slide {slide_id} because of failed chunk")
                for f in temp_files:
                    if os.path.exists(f):
                        os.remove(f)
                continue

            final_output = os.path.join(self.output_dir, f"slide_{int(slide_id):03d}.wav")

            if len(temp_files) == 1:
                os.replace(temp_files[0], final_output)
            else:
                self.merge_wavs(temp_files, final_output)
                for f in temp_files:
                    if os.path.exists(f):
                        os.remove(f)

            print(f"[Speech] Saved: {final_output}")

        print("\n[Speech] All done.")

    def run(self, limit_slides: Optional[int] = None):
        print("========== TALEXA SPEECH AGENT ==========")
        self.tts_per_slide(limit_slides=limit_slides)
        print("=============== FINISHED ===============")


if __name__ == "__main__":
    agent = SpeechAgent(
        subtitles_json_path="Data/intermediate/subtitles.json",
        ref_audio_path="Data/input/ref_clean.wav",
        output_dir="Data/output/speech/lecture1",
        manual_ref_text=None,
        language="en",
        max_chunk_chars=220,
        silence_between_chunks_ms=180,
    )
    agent.run()

    def clean_text(self, text: str) -> str:
        """
        Safe cleaning: supports Arabic + English + mixed text
        """
        if text is None:
            return ""

        text = str(text)

        text = text.replace("\n", " ")

        text = re.sub(r"[^\u0600-\u06FFA-Za-z0-9\s\.\?!،]", "", text)

        text = re.sub(r"\s+", " ", text).strip()

        return text

    def split_text(self, text: str, max_len: int = 250) -> List[str]:
        """
        Splits text into chunks without breaking sentences
        """
        sentences = re.split(r"[.!?؟]+", text)

        chunks = []
        current = ""

        for s in sentences:
            s = s.strip()
            if not s:
                continue

            if len(current) + len(s) < max_len:
                current += " " + s
            else:
                if current:
                    chunks.append(current.strip())
                current = s

        if current:
            chunks.append(current.strip())

        return chunks

    def load_subtitles(self) -> List[Dict[str, Any]]:
        with open(self.subtitles_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        slides = []

        for key in sorted(data.keys()):
            slide = data[key]

            text = " ".join(
                self.clean_text(i.get("sentence", ""))
                for i in slide.get("items", [])
            )

            slides.append({
                "slide_id": slide.get("slide_number", key),
                "text": text
            })

        return slides

    def generate_chunk(self, chunk: str, out_file: str):
        audio = self.client.text_to_speech.convert(
            voice_id=self.voice_id,
            text=chunk,
            model_id="eleven_multilingual_v2"
        )

        save(audio, out_file)

    def generate_audio(self, text: str, out_path: str):
        text = self.clean_text(text)

        if not text.strip():
            print("[Skipped empty text]")
            return

        chunks = self.split_text(text)

        temp_files = []

        for i, chunk in enumerate(chunks):
            print(f"  → Chunk {i+1}/{len(chunks)}")

            tmp_file = os.path.join(self.output_dir, f"tmp_{i}.mp3")

            self.generate_chunk(chunk, tmp_file)

            temp_files.append(tmp_file)

        combined = AudioSegment.empty()

        for f in temp_files:
            audio = AudioSegment.from_file(f)
            combined += audio
            combined += AudioSegment.silent(duration=150)

        combined.export(out_path, format="wav")

        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)

    async def process_slide(self, slide: Dict[str, Any]):
        slide_id = slide["slide_id"]

        print(f"\n[Slide {slide_id}] Generating audio...")

        out_path = os.path.join(self.output_dir, f"slide_{slide_id}.wav")

        self.generate_audio(slide["text"], out_path)

        print(f"[Saved] {out_path}")

    async def run_async(self, limit_slides: Optional[int] = None):
        slides = self.load_subtitles()

        if limit_slides:
            slides = slides[:limit_slides]

        print(f"[Speech] Processing {len(slides)} slides...")

        for slide in slides:
            await self.process_slide(slide)

        print("[DONE]")

    def run(self, limit_slides: Optional[int] = None):
        asyncio.run(self.run_async(limit_slides))

if __name__ == "__main__":
    agent = SpeechAgent(
        language="arabic", 
        api_key = "sk_750e0572c5fc6d7cc3920d7ab0ee832dc1b209cd8ffe37ef",
        voice_id = "aoEJEWeOt9DoaRRQTNaB",

    )
    agent.run()
