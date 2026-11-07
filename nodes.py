"""ComfyUI nodes that generate images and videos with the Sakuranbo AI API."""

from __future__ import annotations

import base64
import io

import comfy.model_management as model_management
import numpy as np
import torch
from comfy.utils import ProgressBar
from comfy_api.input_impl import VideoFromFile
from PIL import Image

from . import client

CATEGORY = "Sakuranbo AI"
# Shown only while the model catalog cannot be reached, so the node still loads.
FALLBACK_IMAGE_MODEL = "sakuranbo/cherryimage-1.0-pro"
FALLBACK_VIDEO_MODEL = "sakuranbo/cherry-video-1.0-pro"

IMAGE_SYNC_TIMEOUT_SECONDS = 120
IMAGE_JOB_TIMEOUT_SECONDS = 15 * 60
VIDEO_JOB_TIMEOUT_SECONDS = 60 * 60

SEED = (
    "INT",
    {
        "default": 0,
        "min": 0,
        "max": 0xFFFFFFFFFFFFFFFF,
        "control_after_generate": "fixed",
        "tooltip": "Not sent to the API. Change it to run the node again with the same inputs.",
    },
)
API_KEY = (
    "STRING",
    {
        "default": "",
        "tooltip": (
            f"Leave empty to use the {client.API_KEY_ENV} environment variable. A key typed here "
            "is saved in the workflow and in the metadata of saved images."
        ),
    },
)


def _catalog(modality: str, task: str, fallback: str) -> tuple[list[str], list[dict]]:
    """Model IDs and their parameter schemas, from GET /v1/models."""
    models = client.list_models(modality, task)
    ids = [m["id"] for m in models] or [fallback]
    params = [((m.get("parameters") or {}).get("properties") or {}) for m in models]
    return ids, params


def _choices(params: list[dict], name: str, fallback: str) -> tuple[list, dict]:
    """A combo of every value any model accepts for name, defaulting to the first model's default."""
    values: list = []
    for schema in params:
        for value in (schema.get(name) or {}).get("enum") or []:
            if value not in values:
                values.append(value)
    values = values or [fallback]
    default = (params[0].get(name) or {}).get("default") if params else None
    return values, {"default": default if default in values else values[0]}


def _integer(params: list[dict], name: str, minimum: int, maximum: int, default: int) -> tuple[str, dict]:
    """An INT widget spanning every model's range for name."""
    specs = [schema[name] for schema in params if name in schema]
    if specs:
        minimum = int(min(s.get("minimum", minimum) for s in specs))
        maximum = int(max(s.get("maximum", maximum) for s in specs))
        default = int(specs[0].get("default", minimum))
    return ("INT", {"default": min(max(default, minimum), maximum), "min": minimum, "max": maximum})


def _wait(path: str, cancel_path: str, key: str, *, interval: float, timeout: float) -> dict:
    """Wait for a job with a progress bar. On Cancel in ComfyUI or a timeout, cancel the job too."""
    bar = ProgressBar(100)

    def on_progress(progress: int | None) -> None:
        # ComfyUI's progress hook raises InterruptProcessingException after Cancel.
        if isinstance(progress, int):
            bar.update_absolute(progress, 100)

    try:
        return client.wait_for_job(
            path,
            key,
            interval=interval,
            timeout=timeout,
            on_progress=on_progress,
            check=model_management.throw_exception_if_processing_interrupted,
        )
    except (model_management.InterruptProcessingException, client.JobTimeout):
        try:
            client.request("POST", cancel_path, key=key)
        except client.SakuranboError:
            pass
        raise


def _to_image_batch(files: list[bytes]) -> torch.Tensor:
    images = [Image.open(io.BytesIO(data)).convert("RGB") for data in files]
    size = images[0].size
    arrays = [
        np.asarray(image if image.size == size else image.resize(size, Image.LANCZOS), dtype=np.float32) / 255.0
        for image in images
    ]
    return torch.from_numpy(np.stack(arrays))


class SakuranboImageGeneration:
    """Text to image with POST /v1/images/generations."""

    CATEGORY = CATEGORY
    DESCRIPTION = "Generates images from a text prompt with Sakuranbo AI."
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "job_id")

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        ids, params = _catalog("image", "text-to-image", FALLBACK_IMAGE_MODEL)
        return {
            "required": {
                "model": (ids, {"default": ids[0]}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "resolution": _choices(params, "resolution", "2k"),
                "aspect_ratio": _choices(params, "aspect_ratio", "auto"),
                "output_format": _choices(params, "output_format", "jpeg"),
                "n": _integer(params, "n", 1, 1, 1),
                "seed": SEED,
            },
            "optional": {"api_key": API_KEY},
        }

    def generate(self, model, prompt, resolution, aspect_ratio, output_format, n, seed, api_key=""):
        key = client.api_key(api_key)
        if not prompt.strip():
            raise client.SakuranboError("Enter a prompt.")
        body = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "output_format": output_format,
            "response_format": "b64_json",
        }
        status, result = client.request(
            "POST",
            "/v1/images/generations",
            key=key,
            body=body,
            headers={"Idempotency-Key": client.idempotency_key()},
            timeout=IMAGE_SYNC_TIMEOUT_SECONDS,
        )
        if status == 202:
            job_id = result["id"]
            job = _wait(
                f"/v1/jobs/{job_id}",
                f"/v1/jobs/{job_id}/cancel",
                key,
                interval=2,
                timeout=IMAGE_JOB_TIMEOUT_SECONDS,
            )
            files = [client.download(output["url"]) for output in job.get("outputs") or []]
        else:
            job_id = result.get("job_id", "")
            files = [
                base64.b64decode(item["b64_json"]) if item.get("b64_json") else client.download(item["url"])
                for item in result.get("data") or []
            ]
        if not files:
            raise client.SakuranboError("Sakuranbo AI returned no images.")
        return (_to_image_batch(files), job_id)


class SakuranboVideoGeneration:
    """Text to video with POST /v1/videos, waiting for the job to complete."""

    CATEGORY = CATEGORY
    DESCRIPTION = "Generates a video from a text prompt with Sakuranbo AI."
    FUNCTION = "generate"
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "job_id")

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        ids, params = _catalog("video", "text-to-video", FALLBACK_VIDEO_MODEL)
        return {
            "required": {
                "model": (ids, {"default": ids[0]}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "duration": _integer(params, "duration", 5, 5, 5),
                "resolution": _choices(params, "resolution", "1080p"),
                "aspect_ratio": _choices(params, "aspect_ratio", "adaptive"),
                "audio": ("BOOLEAN", {"default": True}),
                "seed": SEED,
            },
            "optional": {"api_key": API_KEY},
        }

    def generate(self, model, prompt, duration, resolution, aspect_ratio, audio, seed, api_key=""):
        key = client.api_key(api_key)
        if not prompt.strip():
            raise client.SakuranboError("Enter a prompt.")
        body = {
            "model": model,
            "prompt": prompt,
            "duration": duration,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "audio": audio,
        }
        _, video = client.request(
            "POST",
            "/v1/videos",
            key=key,
            body=body,
            headers={"Idempotency-Key": client.idempotency_key()},
        )
        video_id = video["id"]
        job = _wait(
            f"/v1/videos/{video_id}",
            f"/v1/videos/{video_id}/cancel",
            key,
            interval=5,
            timeout=VIDEO_JOB_TIMEOUT_SECONDS,
        )
        outputs = job.get("outputs") or []
        if not outputs:
            raise client.SakuranboError("Sakuranbo AI returned no video.")
        return (VideoFromFile(io.BytesIO(client.download(outputs[0]["url"]))), video_id)


NODE_CLASS_MAPPINGS = {
    "SakuranboImageGeneration": SakuranboImageGeneration,
    "SakuranboVideoGeneration": SakuranboVideoGeneration,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SakuranboImageGeneration": "Sakuranbo AI Image",
    "SakuranboVideoGeneration": "Sakuranbo AI Video",
}
