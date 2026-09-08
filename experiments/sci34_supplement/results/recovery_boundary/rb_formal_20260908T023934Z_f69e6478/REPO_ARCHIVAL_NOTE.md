# 仓库归档说明（GPU 侧，2026-09-08）

原始封存现场在源码树外 `/root/autodl-tmp/recovery_boundary_runs/rb_formal_20260908T023934Z_f69e6478/`
（含 320 个 FP32 logits NPY sidecar，约 190MB）及其 `rb_formal_20260908T023934Z_f69e6478.tar.gz`
（78MB，sha256 见 `rb_formal_20260908T023934Z_f69e6478.tar.gz.sha256`）。仓库内副本**排除全部 `session_*/*.npy`**；其余工件（manifest/source 快照/
records.jsonl/validation/analysis/boundary/legacy guard/seal.json/session 日志与环境）
逐字节完整。**完整原始 tar（含全部 320 个 NPY）已于 2026-09-08 经作者要求补传仓库**，
位于 `../rb_formal_20260908T023934Z_f69e6478.tar.gz`（78MB，sha256 同目录 `.sha256`）。

- 被排除 NPY 的 SHA-256 逐文件记录于 `seal.json` 的 files 清单与各 record 的
  `logits_sha256`，可通过树外 tarball 完整复核。
- NPY 仍不在解包目录内；在本仓库解包副本上直接运行 `campaign validate/verify` 会因 seal inventory 缺 NPY
  而失败，属预期；权威验证以树外原始目录/tarball 为准（GPU 侧已复核：
  validate ok=true、verify ok=true、tar sha256 OK）。
- 两个 pilot 目录（含 20260908T021245Z 阻断现场）同样归档；阻断原因见
  `../GPU_RUN_NOTES.md` 对应小节与各目录内 `FAILED.json`。

SOURCE_COMMIT：`bfa7d7a37363d327cfa96b5448e151663ff578f8`（paper2）。
