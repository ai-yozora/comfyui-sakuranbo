# Sakuranbo AI nodes for ComfyUI

Generate images and videos in ComfyUI with the [Sakuranbo AI](https://sakuranbo.ai) API.

| Node               | What it does                                                                  | Outputs                     |
| ------------------ | ----------------------------------------------------------------------------- | --------------------------- |
| Sakuranbo AI Image | Generates images from a prompt with `POST /v1/images/generations`             | `images` (IMAGE), `job_id`  |
| Sakuranbo AI Video | Generates a video from a prompt with `POST /v1/videos` and waits for the job | `video` (VIDEO), `job_id`   |

The nodes read the model list from `GET /v1/models` when ComfyUI loads them, so new models appear without an update to the nodes. The resolution and aspect ratio options cover every model. If the selected model does not accept a value, the run stops with an error that lists the accepted values.

## Install

Pick one:

- ComfyUI-Manager: search for "Sakuranbo AI" and install it.
- comfy-cli: `comfy node install comfyui-sakuranbo`
- git: clone this repository into `ComfyUI/custom_nodes`.

  ```bash
  cd ComfyUI/custom_nodes
  git clone https://github.com/ai-yozora/comfyui-sakuranbo.git
  ```

Restart ComfyUI afterwards. The nodes use only packages that ComfyUI already has.

## Set your API key

Create an API key in the [dashboard](https://app.sakuranbo.ai/). See [Authentication](https://sakuranbo.ai/docs/authentication/).

Set the key in the `SAKURANBO_API_KEY` environment variable before you start ComfyUI:

```bash
# macOS and Linux
export SAKURANBO_API_KEY="<your API key>"
python main.py
```

```bat
:: Windows (Command Prompt)
set SAKURANBO_API_KEY=<your API key>
python main.py
```

You can also enter the key in the node's `api_key` field. A key entered there is saved in the workflow and in the metadata of the images that ComfyUI saves, so do not share those files. The environment variable keeps the key out of them.

## Example workflows

The [`example_workflows`](example_workflows) folder has one workflow per node:

- `sakuranbo_text_to_image.json`: Sakuranbo AI Image, then Save Image.
- `sakuranbo_text_to_video.json`: Sakuranbo AI Video, then Save Video.

ComfyUI lists them in its templates browser under `comfyui-sakuranbo`. You can also drag a file onto the canvas.

## How the nodes behave

- Each run is billed to your Sakuranbo AI balance. See [Pricing](https://sakuranbo.ai/pricing/).
- `seed` is not sent to the API. It only decides whether ComfyUI runs the node again. It is `fixed` by default, so changing a node further down the workflow does not generate (and bill) again. Change the seed, or set it to `randomize`, to generate again.
- The image node waits for the images. If the generation takes longer than the API's waiting time, the node polls the job until it completes.
- The video node polls the job every 5 seconds and shows its progress. Cancel in ComfyUI cancels the job, and a cancelled job is not billed.
- Requests carry an `Idempotency-Key`, so a retry after a network error does not bill twice.
- `job_id` is the ID of the job in your account. The dashboard and `GET /v1/jobs/{id}` show it.

The API reference is in the [Images](https://sakuranbo.ai/docs/images/) and [Videos](https://sakuranbo.ai/docs/videos/) docs.

## Support

Open an issue in this repository, or write to support@sakuranbo.ai.

This repository accepts commits from its maintainer account only, and CI rejects any other commit. Please report bugs and requests as issues rather than pull requests.

## License

[MIT](LICENSE)
