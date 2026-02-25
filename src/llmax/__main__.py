"""Main file to test image generation and editing features."""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from llmax.clients import MultiAIClient
from llmax.models import Deployment, Model
from llmax.utils import logger

load_dotenv()

OUTPUT_DIR = Path("image_test_output")

# ---------------------------------------------------------------------------
# Deployments
# ---------------------------------------------------------------------------

def build_client() -> MultiAIClient:
    """Build a MultiAIClient with image-capable deployments.

    Reads from environment variables. For each model, the following variables
    are supported (GPT_IMAGE_1 shown as example):

        LLMAX_GPT_IMAGE_1_API_KEY        required
        LLMAX_GPT_IMAGE_1_PROVIDER       "openai" (default) | "azure"
        LLMAX_GPT_IMAGE_1_ENDPOINT       required when provider is "azure"
        LLMAX_GPT_IMAGE_1_DEPLOYMENT     deployment name (defaults to model name)
        LLMAX_GPT_IMAGE_1_API_VERSION    Azure API version (defaults to 2025-04-01-preview)

        LLMAX_GEMINI_IMAGE_API_KEY       required for Gemini tests
    """
    gpt_provider = os.getenv("LLMAX_GPT_IMAGE_1_PROVIDER", "openai")
    deployments: dict[Model, Deployment] = {
        "gpt-image-1": Deployment(
            model="gpt-image-1",
            provider=gpt_provider,  # type: ignore[arg-type]
            deployment_name=os.getenv("LLMAX_GPT_IMAGE_1_DEPLOYMENT", "gpt-image-1"),
            api_key=os.getenv("LLMAX_GPT_IMAGE_1_API_KEY", ""),
            endpoint=os.getenv("LLMAX_GPT_IMAGE_1_ENDPOINT", ""),
            api_version=os.getenv("LLMAX_GPT_IMAGE_1_API_VERSION", "2025-04-01-preview"),
        ),
        "gemini-3-pro-image-preview": Deployment(
            model="gemini-3-pro-image-preview",
            provider="gemini",
            deployment_name="gemini-3-pro-image-preview",
            api_key=os.getenv("LLMAX_GEMINI_IMAGE_API_KEY", ""),
        ),
    }
    return MultiAIClient(deployments=deployments, fallback_models={})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save(name: str, data: bytes) -> Path:
    """Save image bytes to the output directory and return the path."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / name
    path.write_bytes(data)
    logger.info(f"  → saved {path} ({len(data):,} bytes)")
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_text_to_image(client: MultiAIClient) -> bytes:
    """Test text_to_image with the latest parameters."""
    logger.info("=== text_to_image ===")

    image_bytes = await client.text_to_image(
        model="gpt-image-1",
        prompt="A serene Japanese garden at sunrise, watercolor style",
        quality="medium",
        aspect_ratio="16:9",
        background="opaque",
        output_format="png",
    )
    save("text_to_image.png", image_bytes)
    return image_bytes


async def test_text_to_image_gemini(client: MultiAIClient) -> None:
    """Test text_to_image with a Gemini model."""
    logger.info("=== text_to_image (Gemini) ===")

    image_bytes = await client.text_to_image(
        model="gemini-3-pro-image-preview",
        prompt="A futuristic city skyline at night, neon lights",
        quality="high",
        aspect_ratio="16:9",
    )
    save("text_to_image_gemini.png", image_bytes)


async def test_stream_text_to_image(client: MultiAIClient) -> None:
    """Test stream_text_to_image, saving each progressive frame."""
    logger.info("=== stream_text_to_image ===")

    frame_index = 0
    async for frame_bytes in client.stream_text_to_image(
        model="gpt-image-1",
        prompt="A majestic mountain range covered in snow, photorealistic",
        quality="medium",
        aspect_ratio="16:9",
        output_format="png",
        partial_images=2,
    ):
        save(f"stream_text_to_image_frame_{frame_index:02d}.png", frame_bytes)
        frame_index += 1

    logger.info(f"  → {frame_index} frame(s) received")


async def test_edit_image(client: MultiAIClient, base_image: bytes) -> bytes:
    """Test edit_image with a single image."""
    logger.info("=== edit_image (single) ===")

    image_input = ("base.png", base_image, "image/png")
    edited_bytes = await client.edit_image(
        model="gpt-image-1",
        prompt="Add a small red torii gate in the foreground",
        image=image_input,
        quality="medium",
        background="opaque",
        output_format="png",
    )
    save("edit_image_single.png", edited_bytes)
    return edited_bytes


async def test_edit_image_multi(client: MultiAIClient, base_image: bytes) -> None:
    """Test edit_image with multiple reference images."""
    logger.info("=== edit_image (multi-image) ===")

    # Generate a second reference image to use alongside the base one
    second_image_bytes = await client.text_to_image(
        model="gpt-image-1",
        prompt="A traditional Japanese paper lantern festival at night, watercolor style",
        quality="low",
        aspect_ratio="1:1",
        output_format="png",
    )
    save("edit_image_multi_ref2.png", second_image_bytes)

    edited_bytes = await client.edit_image(
        model="gpt-image-1",
        prompt="Blend these two scenes into a single cohesive artistic composition",
        image=[
            ("scene1.png", base_image, "image/png"),
            ("scene2.png", second_image_bytes, "image/png"),
        ],
        quality="medium",
        output_format="png",
    )
    save("edit_image_multi.png", edited_bytes)


async def test_edit_image_gemini(client: MultiAIClient, base_image: bytes) -> None:
    """Test edit_image with a Gemini model."""
    logger.info("=== edit_image (Gemini) ===")

    image_input = ("base.png", base_image, "image/png")
    edited_bytes = await client.edit_image(
        model="gemini-3-pro-image-preview",
        prompt="Make the sky more dramatic with stormy clouds",
        image=image_input,
    )
    save("edit_image_gemini.png", edited_bytes)


async def test_stream_edit_image(client: MultiAIClient, base_image: bytes) -> None:
    """Test stream_edit_image, saving each progressive frame."""
    logger.info("=== stream_edit_image ===")

    image_input = ("base.png", base_image, "image/png")
    frame_index = 0
    async for frame_bytes in client.stream_edit_image(
        model="gpt-image-1",
        prompt="Transform the scene into a snowy winter landscape",
        image=image_input,
        quality="medium",
        output_format="png",
        partial_images=2,
    ):
        save(f"stream_edit_image_frame_{frame_index:02d}.png", frame_bytes)
        frame_index += 1

    logger.info(f"  → {frame_index} frame(s) received")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    """Run all image tests sequentially."""
    client = build_client()

    # Generate a base image used by the edit tests
    base_image = await test_text_to_image(client)

    await test_stream_text_to_image(client)
    await test_edit_image(client, base_image)
    await test_edit_image_multi(client, base_image)
    await test_stream_edit_image(client, base_image)

    # Gemini tests — comment out if no Gemini key is available
    await test_text_to_image_gemini(client)
    await test_edit_image_gemini(client, base_image)

    logger.info(f"Total cost: ${client.total_usage:.4f}")
    logger.info(f"All images saved to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
