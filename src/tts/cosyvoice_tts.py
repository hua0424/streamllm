# src/tts/cosyvoice_tts.py
"""
CosyVoice2 的 StreamingTTS 适配器（实验机用，D-010）。

⚠️ **未在真机验证**：本机（Blackwell）无法运行 CosyVoice2 官方 pin 的 torch 2.3.1+cu121，
本文件在验证机只做了编译检查。实验机接入步骤见 docs/handoff.md §四.3：
  1) 按官方 requirements 在**独立环境**装 CosyVoice（勿污染主 uv 环境——torch 版本冲突）；
     若独立环境，主进程经进程边界调用（如简易 HTTP/socket 服务）或在该环境内跑全套实验
  2) 下载 pretrained_models/CosyVoice2-0.5B + 一段参考音频
  3) 跑 experiments/scripts/benchmark_cosyvoice.py 得真实 TimingProfile，
     替换 streaming_tts.py 与 run_exp1_latency.py 的占位值
  4) 编排层换 tts=CosyVoiceStreamingTTS(...) 重跑 E1（实测 mouth-to-ear）

设计对齐（paper2_context.md §3.2）：CosyVoice2 是输出端流式——输入完整句子、
输出 chunked audio，与 stream2sentence 的句子级输出正好匹配 StreamingTTS 接口。
"""

import time
from typing import Generator, Optional

import numpy as np

from src.tts.streaming_tts import AudioChunk, StreamingTTS
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CosyVoiceStreamingTTS(StreamingTTS):
    """
    包装 cosyvoice.cli.cosyvoice.CosyVoice2 的 zero-shot 流式合成。
    延迟导入：仅实例化时才 import cosyvoice（主环境无此包时本模块仍可被安全 import）。
    """

    def __init__(self, model_dir: str = "pretrained_models/CosyVoice2-0.5B",
                 ref_text: str = "Hello, this is a reference voice sample.",
                 ref_audio_path: str = "ref_audio.wav"):
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2   # 延迟导入（实验机环境才有）
        except ImportError as e:
            raise RuntimeError(
                "cosyvoice 未安装。请在实验机按官方 requirements 于独立环境安装 "
                "(https://github.com/FunAudioLLM/CosyVoice)，见 docs/handoff.md §四.3"
            ) from e
        self._cosy = CosyVoice2(model_dir)
        self._ref_text = ref_text
        self._ref_audio = ref_audio_path
        self._sample_rate = getattr(self._cosy, "sample_rate", 24000)  # CosyVoice2 输出 24kHz
        self._measured_first_chunk_ms: Optional[float] = None
        logger.info(f"CosyVoice2 ready (sr={self._sample_rate})")

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def first_chunk_latency_ms(self) -> float:
        # 实测值（benchmark 后由编排层建模使用）；未测时用公开指标占位
        return self._measured_first_chunk_ms if self._measured_first_chunk_ms else 45.0

    def synthesize(self, text: str) -> Generator[AudioChunk, None, None]:
        t0 = time.perf_counter()
        first = True
        for out in self._cosy.inference_zero_shot(text, self._ref_text, self._ref_audio,
                                                  stream=True):
            wav = out["tts_speech"]                       # torch.Tensor [1, n]
            samples = wav.squeeze(0).cpu().numpy().astype(np.float32)
            if first:
                self._measured_first_chunk_ms = (time.perf_counter() - t0) * 1000
                first = False
            yield AudioChunk(n_samples=samples.shape[-1], samples=samples,
                             sample_rate=self._sample_rate)
