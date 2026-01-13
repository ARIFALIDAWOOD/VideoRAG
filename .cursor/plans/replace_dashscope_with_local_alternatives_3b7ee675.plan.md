---
name: Replace Dashscope with Local Alternatives
overview: "Replace Dashscope API (ASR and captioning) with cost-free local alternatives: local Whisper for ASR and local LLaVA for video captioning. This eliminates API costs while maintaining functionality."
todos:
  - id: replace_asr
    content: Replace Dashscope ASR with local Whisper in asr.py
    status: pending
  - id: replace_captioning
    content: Replace Dashscope captioning with local LLaVA in _llm.py
    status: pending
  - id: update_videorag_config
    content: Remove Dashscope parameters from VideoRAG class in videorag.py
    status: pending
  - id: update_api_handler
    content: Remove Dashscope config from videorag_api.py worker processes
    status: pending
  - id: update_frontend_config
    content: Remove DashScope UI components from React frontend files
    status: pending
  - id: update_dependencies
    content: Update Python dependencies (remove dashscope, add whisper/llava)
    status: pending
  - id: update_documentation
    content: Update README to reflect local model usage
    status: pending
---

# Replace Dashscope with Local Alternatives

## Overview

Replace Dashscope API dependencies with cost-free local alternatives:

- **ASR**: Replace Dashscope ASR with local Whisper (using `faster-whisper` or `whisper`)
- **Captioning**: Replace Dashscope vision model with local LLaVA model

## Implementation Plan

### 1. Replace ASR Implementation

**File**: `Vimo-desktop/python_backend/videorag/_videoutil/asr.py`

- Remove `dashscope` import and `Recognition` usage
- Implement local Whisper using `faster-whisper` (faster) or `openai-whisper` (more compatible)
- Update `speech_to_text_online()` to use local Whisper model
- Maintain async/concurrent processing structure
- Add model loading and caching logic

### 2. Replace Captioning Implementation  

**File**: `Vimo-desktop/python_backend/videorag/_llm.py`

- Remove `dashscope_caption_complete()` function
- Implement `llava_caption_complete()` using local LLaVA model
- Use `transformers` library with LLaVA model (e.g., `llava-hf/llava-1.5-7b-hf`)
- Maintain same function signature for compatibility
- Handle base64 image decoding and model inference

### 3. Update VideoRAG Configuration

**File**: `Vimo-desktop/python_backend/videorag/videorag.py`

- Remove `ali_dashscope_api_key` and `ali_dashscope_base_url` parameters
- Remove assertions for Dashscope API keys
- Update `__post_init__` to remove Dashscope requirements
- Add optional parameters for Whisper model size and LLaVA model path

### 4. Update API Handler

**File**: `Vimo-desktop/python_backend/videorag_api.py`

- Remove `ali_dashscope_api_key` and `ali_dashscope_base_url` from `global_config`
- Update `index_worker_process()` and `query_worker_process()` to remove Dashscope config
- Update LLMConfig initialization to use `llava_caption_complete` instead of `dashscope_caption_complete`

### 5. Update Frontend Configuration

**Files**:

- `Vimo-desktop/src/renderer/src/components/VideoRAGConfig.tsx`
- `Vimo-desktop/src/renderer/src/components/InitializationWizard.tsx`
- `Vimo-desktop/src/renderer/src/pages/settings/index.tsx`
- `Vimo-desktop/src/main/handlers/videorag-handlers.ts`
- `Vimo-desktop/src/main/handlers/settings.ts`

- Remove DashScope API key input fields
- Remove DashScope base URL configuration
- Remove DashScope validation checks
- Update configuration interfaces/types

### 6. Update Dependencies

**Files**: Check for `requirements.txt`, `package.json`, or similar

- Remove `dashscope` from Python dependencies
- Add `faster-whisper` or `openai-whisper` for ASR
- Add `transformers`, `torch`, `torchvision`, `pillow` for LLaVA
- Add `accelerate` for model optimization

### 7. Update Documentation

**File**: `README.md` (if it mentions Dashscope)

- Update installation instructions
- Remove Dashscope API key requirements
- Add instructions for downloading Whisper/LLaVA models

## Technical Considerations

### ASR (Whisper)

- Use `faster-whisper` (CTranslate2 backend) for better performance
- Model options: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`
- Default to `base` or `small` for balance of speed/accuracy
- Support GPU acceleration if available

### Captioning (LLaVA)

- Use `llava-hf/llava-1.5-7b-hf` (7B model) or `llava-hf/llava-1.5-13b-hf` (13B model)
- 7B model recommended for most GPUs (requires ~14GB VRAM)
- Support CPU fallback with quantization
- Cache model in memory after first load

### Performance Impact

- Local models require GPU for reasonable performance
- First-time model loading will be slower
- Processing speed may be slower than API calls but eliminates costs
- Consider adding progress indicators for model loading

## Testing Checklist

- [ ] ASR works with local Whisper
- [ ] Captioning works with local LLaVA
- [ ] Configuration UI no longer shows DashScope fields
- [ ] Video indexing completes successfully
- [ ] Query processing works correctly
- [ ] Models load and cache properly
- [ ] Error handling for missing models/GPU