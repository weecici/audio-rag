"""Shared GPU lock for serialising CUDA access across services.

Both the reranker (CrossEncoder) and speech-to-text (faster-whisper) models
need exclusive GPU access when running on a 4 GB VRAM device.  This module
provides a single ``threading.Lock`` that both services acquire before
moving their model onto the GPU.
"""

import threading

gpu_lock = threading.Lock()
