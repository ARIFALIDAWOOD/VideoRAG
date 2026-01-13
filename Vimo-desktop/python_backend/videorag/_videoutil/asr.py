import os
import asyncio
from faster_whisper import WhisperModel
from .._utils import logger

# Global model cache to avoid reloading
_whisper_model = None
_whisper_model_size = None

def get_whisper_model(model_size="base", device="auto"):
    """
    Get or create a cached Whisper model instance.

    Args:
        model_size: Model size - tiny, base, small, medium, large-v2, large-v3
        device: Device to use - 'auto', 'cuda', 'cpu'
    """
    global _whisper_model, _whisper_model_size

    # Reload if model size changed
    if _whisper_model is None or _whisper_model_size != model_size:
        logger.info(f"Loading Whisper model '{model_size}' on device '{device}'...")

        # Determine compute type based on device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        compute_type = "float16" if device == "cuda" else "int8"

        _whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        _whisper_model_size = model_size
        logger.info(f"Whisper model loaded successfully on {device}")

    return _whisper_model


async def process_single_segment(semaphore, index, segment_name, audio_file, model_size):
    """
    Process a single audio segment with local Whisper ASR
    """
    async with semaphore:
        try:
            logger.info(f"Processing segment {segment_name} with Whisper model '{model_size}'")

            # Get the model (cached)
            model = get_whisper_model(model_size)

            # Run transcription in executor to avoid blocking
            loop = asyncio.get_event_loop()

            def transcribe():
                segments, info = model.transcribe(
                    audio_file,
                    language=None,  # Auto-detect language
                    vad_filter=True,  # Filter out non-speech
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                return " ".join([seg.text.strip() for seg in segments])

            result = await loop.run_in_executor(None, transcribe)

            if result:
                return index, result
            else:
                logger.warning(f"No transcription result for segment {segment_name}")
                return index, ""

        except Exception as e:
            logger.error(f"ASR failed for segment {segment_name}: {str(e)}")
            raise e


async def speech_to_text_online(video_name, working_dir, segment_index2name, audio_output_format, global_config, max_concurrent=5):
    """
    Local ASR using Whisper with async concurrent processing
    """
    # Get model size from config (default: base)
    model_size = global_config.get('whisper_model_size', 'base')

    cache_path = os.path.join(working_dir, '_cache', video_name)

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create tasks for all segments
    tasks = []
    for index in segment_index2name:
        segment_name = segment_index2name[index]
        audio_file = os.path.join(cache_path, f"{segment_name}.{audio_output_format}")

        task = process_single_segment(
            semaphore, index, segment_name, audio_file, model_size
        )
        tasks.append(task)

    # Execute all tasks concurrently with real-time progress
    total_tasks = len(tasks)
    logger.info(f"Starting ASR for {total_tasks} audio segments (max {max_concurrent} concurrent)...")

    transcripts = {}
    completed = 0

    # Use asyncio.as_completed for real-time progress updates
    for completed_task in asyncio.as_completed(tasks):
        try:
            result = await completed_task
            if isinstance(result, tuple) and len(result) == 2:
                index, text = result
                transcripts[index] = text
                completed += 1
                logger.info(f"Completed {completed}/{total_tasks} segments (Progress: {completed/total_tasks*100:.1f}%)")
            else:
                completed += 1
                logger.info(f"Unexpected result format for segment {completed}")

        except Exception as e:
            completed += 1
            logger.error(f"Task failed: {e}")
            logger.info(f"Failed {completed}/{total_tasks} segments (Progress: {completed/total_tasks*100:.1f}%)")

    logger.info(f"ASR processing completed! Processed {len(transcripts)} segments successfully.")

    return transcripts


async def speech_to_text_async(video_name, working_dir, segment_index2name, audio_output_format, global_config):
    """
    Async speech-to-text function using local Whisper ASR

    Args:
        video_name: Name of the video
        working_dir: Working directory
        segment_index2name: Mapping of segment indices to names
        audio_output_format: Audio file format
        global_config: Global configuration dictionary containing settings
    """
    return await speech_to_text_online(
        video_name, working_dir, segment_index2name, audio_output_format, global_config
    )


def speech_to_text(video_name, working_dir, segment_index2name, audio_output_format, global_config):
    """
    Synchronous wrapper for async speech-to-text function

    Args:
        video_name: Name of the video
        working_dir: Working directory
        segment_index2name: Mapping of segment indices to names
        audio_output_format: Audio file format
        global_config: Global configuration dictionary containing settings
    """
    # Run the async function in an event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        speech_to_text_async(video_name, working_dir, segment_index2name, audio_output_format, global_config)
    )
