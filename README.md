# commit-classifier

A Rust library for classifying git commits with large language models and
embeddings. It provides the LLM provider abstraction, the embeddings model
wrapper and the vector store used by
[autosel](https://github.com/sashalevin/autosel), which decides whether Linux
kernel commits should be backported to stable trees.

The code was originally part of autosel and was split out so the classification
machinery could be reused independently, so autosel's history contains earlier
copies of these files.

## Providers

| Provider | Selector | Credentials |
|---|---|---|
| Anthropic Claude | `claude` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai` | `OPENAI_API_KEY` |
| NVIDIA | `nvidia` | `NVIDIA_API_KEY` |
| Kiro CLI | `kiro` | `KIRO_API_KEY`, or an authenticated `kiro-cli` session |
| Ollama | `ollama` | none, talks to a local server |
| Exec | `exec` | none, runs a local binary as the model |

## Embeddings

`HuggingFaceEmbeddings` wraps [candle_embed](https://github.com/ShelbyJenkins/candle_embed)
and defaults to `BAAI/bge-large-en-v1.5` (1024 dimensions). The model is
downloaded from HuggingFace on first use.

## Building

```
cargo build
cargo test
```

The test suite is 66 unit tests. None of them need network access or API keys;
credentials are only used for real classification runs.

Optional CUDA support for embeddings:

```
cargo build --features cuda
```

## License

SPDX-License-Identifier: GPL-2.0-only
(c) 2025, Sasha Levin <sashal@kernel.org>

## Contributing

Open a pull request against `main`. Sign your work with `git commit -s`
(Developer Certificate of Origin). Commit messages use imperative subjects,
optionally prefixed with the subsystem (`llm/kiro: ...`); explain *why* in the
body, wrapped at 72 columns.
