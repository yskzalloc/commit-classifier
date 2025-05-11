// SPDX-License-Identifier: GPL-2.0
// (c) 2025, Sasha Levin <sashal@kernel.org>

//! Library for classifying commits using LLMs and embeddings
//! Provides functionality for generating embeddings, interacting with various LLM providers,
//! and performing similarity search to aid in classification decisions.

pub mod embeddings;
pub mod vectorstore;
pub mod models;
pub mod llm;
pub mod http;

// Re-export common types for easy use
pub use embeddings::{HuggingFaceEmbeddings, Embedding};
pub use vectorstore::{ChromaStore, Document, ScoredDocument, Metadata};
pub use llm::{Llm, LlmError, Message, UniversalLlmRequest, create_llm, LlmFactory};
pub use models::{CommitFeatures, ProviderResult, PredictionResult};
