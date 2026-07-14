# Cover Letter

Dear Editors-in-Chief of *Universal Access in the Information Society*,

We are pleased to submit our manuscript, **"Latency Optimization of Cascaded Voice Dialogue Systems with a Pipeline-Parallel Streaming Architecture,"** for consideration as an original research article in *Universal Access in the Information Society*.

Voice dialogue is one of the most inclusive interaction modalities in the information society: it serves users who cannot rely on screens or keyboards, including people with visual or motor impairments, older adults, and users in hands-busy or eyes-busy situations. For these users, responsiveness is not a luxury but a precondition of usable access — a system that falls silent for many seconds after the user stops speaking is, in practice, inaccessible. Our work addresses exactly this barrier. In conventional cascaded voice systems (ASR–LLM–TTS), the waiting time after the user finishes speaking grows linearly with utterance length, penalizing precisely those users who express complex needs in longer utterances.

The manuscript makes three contributions:

1. **A streaming ASR context-management mechanism** based on Whisper and Silero VAD, using an adaptive sliding window with prefix–suffix context to emit stable transcript fragments during speech, without sacrificing long-sentence accuracy.
2. **Incremental key–value-cache prefilling for the LLM**, which moves most of the prompt-prefill computation into the user's speaking time, so that the post-utterance wait no longer grows with utterance length.
3. **A bilingual long-speech benchmark** (1,132 progressively lengthening utterances synthesized from MultiWOZ and CrossWOZ) with a full-pipeline evaluation: mean time to first token stays around 1.1 s in long-utterance groups — a 34.6%–83.9% reduction over a non-streaming baseline, and an average 5.67 s reduction in the longest group — while transcription error rates remain acceptable.

Because the approach is architectural rather than model-specific, it can be applied to existing high-quality ASR models and text LLMs without retraining, which we believe makes it of practical value to researchers and practitioners building accessible, responsive voice interfaces.

We confirm that this manuscript is original, has not been published previously, and is not under consideration by any other journal. All authors have approved the submission and agree to its publication. We have no competing interests to declare. A large language model was used to assist with translating the manuscript from Chinese and with language editing; the authors reviewed and verified all content and take full responsibility for it, as disclosed in the Acknowledgements.

Thank you for your consideration. We look forward to the reviewers' feedback.

Sincerely,

Zhengyou Liang (corresponding author)
On behalf of the authors: Haihua Mo, Zhengyou Liang
School of Computer, Electronics and Information, Guangxi University
Nanning 530004, Guangxi, China
E-mail: zhyliang@gxu.edu.cn
