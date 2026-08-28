# Tests

The test suite is split into fast CI tests and real-checkpoint smoke tests.

```text
tests/
├── unit/         # No model stack/download; runs on every push and pull request
├── integration/  # CPU runtime-import check; runs on pull requests and main
└── smoke/        # Real Hugging Face/local checkpoint validation; run before a release
```

Run the fast CI suite locally:

```bash
python -m pytest tests/unit
```

With the CPU runtime dependencies installed, also run:

```bash
python -m pytest tests/integration
```

Run a released checkpoint smoke test:

```bash
python tests/smoke/test_checkpoint_loading.py \
  --model hxxiang/opticaldna-hg38-2048 \
  --device cuda \
  --image assets/640x640.png
```

Use the same command with `hxxiang/opticaldna-rice-2048` for the rice model.

Add `--run-generation` if you also want to test decoder inference. The smoke test will then check both:

- feature extraction **without** a decoder prompt, and
- generation using the default short T1 prompt (`Free OCR.`), plus the same prompt passed explicitly through `PromptGenerator`.
