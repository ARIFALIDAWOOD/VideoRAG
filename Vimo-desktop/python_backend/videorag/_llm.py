# type: ignore
import numpy as np
import base64
import asyncio
from io import BytesIO
from PIL import Image
from openai import AsyncOpenAI, AsyncAzureOpenAI, APIConnectionError, RateLimitError
from dataclasses import asdict, dataclass, field

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import os

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs, logger
from .base import BaseKVStorage
from ._utils import EmbeddingFunc

global_openai_async_client = None

# LLaVA model cache
_llava_model = None
_llava_processor = None
_llava_device = None

def get_openai_async_client_instance(global_config):
    global global_openai_async_client
    if global_openai_async_client is None:
        global_openai_async_client = AsyncOpenAI(
            api_key=global_config["openai_api_key"],
            base_url=global_config["openai_base_url"],
        )
    return global_openai_async_client


def get_llava_model(model_id="llava-hf/llava-1.5-7b-hf", load_in_4bit=True):
    """
    Get or create a cached LLaVA model instance.

    Args:
        model_id: HuggingFace model ID for LLaVA
        load_in_4bit: Whether to use 4-bit quantization (reduces VRAM from ~14GB to ~4GB)
    """
    global _llava_model, _llava_processor, _llava_device

    if _llava_model is None:
        import torch
        from transformers import LlavaForConditionalGeneration, AutoProcessor

        logger.info(f"Loading LLaVA model '{model_id}' (4-bit: {load_in_4bit})...")

        # Determine device
        _llava_device = "cuda" if torch.cuda.is_available() else "cpu"

        if load_in_4bit and _llava_device == "cuda":
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            _llava_model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            _llava_model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto" if _llava_device == "cuda" else None,
                torch_dtype=torch.float16 if _llava_device == "cuda" else torch.float32
            )
            if _llava_device == "cpu":
                _llava_model = _llava_model.to(_llava_device)

        # Use slow tokenizer to avoid Rust tokenizer parsing issues on Windows
        # The fast tokenizer can fail with "untagged enum ModelWrapper" errors
        _llava_processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
        logger.info(f"LLaVA model loaded successfully on {_llava_device}")

    return _llava_model, _llava_processor, _llava_device


def decode_base64_image(base64_string):
    """Decode a base64 image string to PIL Image."""
    # Handle data URI format: data:image/jpeg;base64,xxxxx
    if base64_string.startswith("data:"):
        base64_string = base64_string.split(",", 1)[1]

    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data)).convert("RGB")


# Setup LLM Configuration.
@dataclass
class LLMConfig:
    # To be set
    embedding_func_raw: callable
    embedding_model_name: str
    embedding_dim: int
    embedding_max_token_size: int
    embedding_batch_num: int
    embedding_func_max_async: int
    query_better_than_threshold: float

    best_model_func_raw: callable
    best_model_name: str
    best_model_max_token_size: int
    best_model_max_async: int

    cheap_model_func_raw: callable
    cheap_model_name: str
    cheap_model_max_token_size: int
    cheap_model_max_async: int

    # Caption model configuration
    caption_model_func_raw: callable
    caption_model_name: str
    caption_model_max_async: int

    # Assigned in post init
    embedding_func: EmbeddingFunc = None
    best_model_func: callable = None
    cheap_model_func: callable = None
    caption_model_func: callable = None


    def __post_init__(self):
        embedding_wrapper = wrap_embedding_func_with_attrs(
            embedding_dim = self.embedding_dim,
            max_token_size = self.embedding_max_token_size,
            model_name = self.embedding_model_name)
        self.embedding_func = embedding_wrapper(self.embedding_func_raw)

        self.best_model_func = lambda prompt, *args, **kwargs: self.best_model_func_raw(
            self.best_model_name, prompt, *args, **kwargs
        )

        self.cheap_model_func = lambda prompt, *args, **kwargs: self.cheap_model_func_raw(
            self.cheap_model_name, prompt, *args, **kwargs
        )

        self.caption_model_func = lambda content_list, *args, **kwargs: self.caption_model_func_raw(
            self.caption_model_name, content_list, *args, **kwargs
        )

##### OpenAI Configuration
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = get_openai_async_client_instance(kwargs["global_config"])
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    use_cache = kwargs.pop("use_cache", True)

    # Remove global_config from kwargs as it's not needed for OpenAI API call
    kwargs.pop("global_config", None)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None and use_cache:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        # NOTE: I update here to avoid the if_cache_return["return"] is None
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    if hashing_kv is not None and use_cache:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content

async def gpt_complete(
        model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_embedding(model_name: str, texts: list[str], **kwargs) -> np.ndarray:
    openai_async_client = get_openai_async_client_instance(kwargs["global_config"])
    # Remove global_config from kwargs as it's not needed for OpenAI API call
    kwargs.pop("global_config", None)

    response = await openai_async_client.embeddings.create(
        model=model_name, input=texts, encoding_format="float", **kwargs
    )
    return np.array([dp.embedding for dp in response.data])


async def llava_caption_complete(
    model_name, content_list, **kwargs
) -> str:
    """
    Local LLaVA vision model completion for video caption.

    Args:
        model_name: Not used for local model, kept for API compatibility
        content_list: list of {"type": "image_url", "image_url": {"url": "..."}} and {"type": "text", "text": "..."}
        **kwargs: Must contain global_config with llava_use_4bit and llava_model_id settings
    """
    import torch

    global_config = kwargs.get("global_config", {})
    load_in_4bit = global_config.get("llava_use_4bit", True)
    model_id = global_config.get("llava_model_id", "llava-hf/llava-1.5-7b-hf")

    # Get model (cached)
    model, processor, device = get_llava_model(model_id, load_in_4bit)

    # Parse content_list to extract images and text
    images = []
    text_parts = []

    for item in content_list:
        if item.get("type") == "image_url":
            url = item.get("image_url", {}).get("url", "")
            if url:
                try:
                    img = decode_base64_image(url)
                    images.append(img)
                except Exception as e:
                    logger.warning(f"Failed to decode image: {e}")
        elif item.get("type") == "text":
            text_parts.append(item.get("text", ""))

    # Build prompt
    user_text = " ".join(text_parts) if text_parts else "Describe what you see in this image."

    # LLaVA prompt format
    if images:
        # Add <image> tokens for each image
        image_tokens = "<image>" * len(images)
        prompt = f"USER: {image_tokens}\n{user_text}\nASSISTANT:"
    else:
        prompt = f"USER: {user_text}\nASSISTANT:"

    # Run inference in executor to avoid blocking
    loop = asyncio.get_event_loop()

    def run_inference():
        with torch.no_grad():
            if images:
                inputs = processor(text=prompt, images=images, return_tensors="pt")
            else:
                inputs = processor(text=prompt, return_tensors="pt")

            # Move to device
            inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            # Generate
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=None,
                top_p=None
            )

            # Decode output, skip input tokens
            generated_text = processor.decode(output_ids[0], skip_special_tokens=True)

            # Extract only the assistant's response
            if "ASSISTANT:" in generated_text:
                response = generated_text.split("ASSISTANT:")[-1].strip()
            else:
                response = generated_text

            return response

    result = await loop.run_in_executor(None, run_inference)
    return result


openai_4o_mini_config = LLMConfig(
    embedding_func_raw = openai_embedding,
    embedding_model_name = "text-embedding-3-small",
    embedding_dim = 1536,
    embedding_max_token_size  = 8192,
    embedding_batch_num = 32,
    embedding_func_max_async = 16,
    query_better_than_threshold = 0.2,

    # LLM
    best_model_func_raw = gpt_complete,
    best_model_name = "gpt-4o-mini",
    best_model_max_token_size = 32768,
    best_model_max_async = 16,

    cheap_model_func_raw = gpt_complete,
    cheap_model_name = "gpt-4o-mini",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16,

    # Caption model - now using local LLaVA
    caption_model_func_raw = llava_caption_complete,
    caption_model_name = "llava-hf/llava-1.5-7b-hf",
    caption_model_max_async = 1  # Local model, limit concurrency
)
