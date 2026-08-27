// SPDX-License-Identifier: GPL-2.0
// (c) 2025, Sasha Levin <sashal@kernel.org>

use candle_embed::{BasedBertEmbedder, CandleEmbedBuilder};
use log::{info, error};

pub type Embedding = Vec<f32>;

pub struct HuggingFaceEmbeddings {
    model_name: String,
    model: Option<BasedBertEmbedder>,
    embedding_size: usize,
    is_initialized: bool,
}

impl HuggingFaceEmbeddings {
    pub fn new() -> Self {
        let model_name = "BAAI/bge-large-en-v1.5".to_string();

        Self {
            model_name,
            model: None,
            embedding_size: 1024,
            is_initialized: false,
        }
    }

    pub fn initialize(&mut self) -> Result<(), String> {
        if self.is_initialized {
            info!("Embeddings model already initialized, skipping");
            return Ok(());
        }

        info!("Initializing embeddings model: {}", self.model_name);

        let mut builder = CandleEmbedBuilder::new()
            .custom_embedding_model(&self.model_name);

        #[cfg(feature = "cuda")]
        {
            info!("Attempting to use CUDA for embeddings");
            builder = builder.with_device_any_cuda();
        }

        #[cfg(not(feature = "cuda"))]
        {
            info!("CUDA support not enabled, using CPU for embeddings");
            builder = builder.with_device_cpu();
        }

        match builder.build() {
            Ok(embedder) => {
                info!("Successfully built embedder, loading tokenizer");
                if let Err(e) = embedder.load_tokenizer() {
                    error!("Failed to load tokenizer: {}", e);
                    return Err(format!("Failed to load tokenizer: {}", e));
                }
                info!("Successfully loaded tokenizer, loading model");

                if let Err(e) = embedder.load_model() {
                    error!("Failed to load embeddings model: {}", e);
                    return Err(format!("Failed to load embeddings model: {}", e));
                }
                info!("Successfully loaded model with dimensions: {}", embedder.model_dimensions);

                self.embedding_size = embedder.model_dimensions;
                self.model = Some(embedder);
                self.is_initialized = true;

                match self.embed_query("test") {
                    Ok(_) => {
                        info!("Verified embeddings model is operational");
                        Ok(())
                    },
                    Err(e) => {
                        error!("Embeddings model verification failed: {}", e);
                        self.is_initialized = false;
                        self.model = None;
                        Err(format!("Embeddings model verification failed: {}", e))
                    }
                }
            },
            Err(e) => {
                error!("Failed to build embeddings model: {}", e);
                Err(format!("Failed to build embeddings model: {}", e))
            },
        }
    }

    pub fn embed_texts(&self, texts: &[String]) -> Result<Vec<Embedding>, String> {
        let model = match &self.model {
            Some(m) => m,
            None => return Err("Embeddings model not initialized".to_string()),
        };

        let safe_texts: Vec<String> = texts
            .iter()
            .map(|text| {
                if text.trim().is_empty() {
                    " ".to_string()
                } else {
                    text.clone()
                }
            })
            .collect();

        let text_refs: Vec<&str> = safe_texts.iter().map(|s| s.as_str()).collect();

        match model.embed_batch(&text_refs) {
            Ok(embeddings) => Ok(embeddings),
            Err(e) => Err(format!("Failed to generate embeddings: {}", e)),
        }
    }

    pub fn embed_query(&self, text: &str) -> Result<Embedding, String> {
        if !self.is_initialized {
            return Err("Embeddings model not initialized".to_string());
        }

        let model = match &self.model {
            Some(m) => m,
            None => return Err("Embeddings model is None despite initialized flag".to_string()),
        };

        let safe_text = if text.trim().is_empty() { " " } else { text };

        match model.embed_one(safe_text) {
            Ok(embedding) => {
                if embedding.is_empty() {
                    Err("Embedding generated is empty".to_string())
                } else {
                    Ok(embedding)
                }
            },
            Err(e) => {
                error!("Failed to generate embedding for query: {}", e);
                Err(format!("Failed to generate embedding: {}", e))
            }
        }
    }

    pub fn is_initialized(&self) -> bool {
        self.is_initialized
    }
}

impl Default for HuggingFaceEmbeddings {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for HuggingFaceEmbeddings {
    fn clone(&self) -> Self {
        info!("Cloning embeddings model, is_initialized={}", self.is_initialized);

        let mut new_instance = Self {
            model_name: self.model_name.clone(),
            model: None,
            embedding_size: self.embedding_size,
            is_initialized: false,
        };

        if self.is_initialized {
            info!("Original embeddings was initialized, initializing clone");
            if let Err(e) = new_instance.initialize() {
                error!("Failed to initialize cloned embeddings model: {}", e);
            }
        }

        new_instance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let embeddings = HuggingFaceEmbeddings::new();
        assert_eq!(embeddings.model_name, "BAAI/bge-large-en-v1.5");
        assert_eq!(embeddings.embedding_size, 1024);
        assert!(!embeddings.is_initialized);
        assert!(embeddings.model.is_none());
    }

    #[test]
    fn test_is_initialized() {
        let embeddings = HuggingFaceEmbeddings::new();
        assert!(!embeddings.is_initialized());
    }

    #[test]
    fn test_embed_texts_uninitialized() {
        let embeddings = HuggingFaceEmbeddings::new();
        let texts = vec!["Sample text".to_string()];
        let result = embeddings.embed_texts(&texts);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Embeddings model not initialized");
    }

    #[test]
    fn test_embed_query_uninitialized() {
        let embeddings = HuggingFaceEmbeddings::new();
        let result = embeddings.embed_query("Sample query");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Embeddings model not initialized");
    }
}
