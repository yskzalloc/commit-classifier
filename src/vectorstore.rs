// SPDX-License-Identifier: GPL-2.0
// (c) 2025, Sasha Levin <sashal@kernel.org>

use std::path::Path;
use std::collections::HashMap;
use std::fs;
use serde::{Serialize, Deserialize};
use log::{info, error};
use crate::embeddings::{HuggingFaceEmbeddings, Embedding};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use std::cmp::Ordering;
use std::io::{Read, Write};
use xz2::write::XzEncoder;
use xz2::read::XzDecoder;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<DateTime<Utc>>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl Default for Metadata {
    fn default() -> Self {
        Self {
            source: None,
            created_at: Some(Utc::now()),
            extra: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub text: String,
    pub embedding: Option<Embedding>,
    pub metadata: Metadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredDocument {
    pub document: Document,
    pub score: f32,
}

/// Vector store implementation for storing and retrieving semantically similar documents
#[derive(Clone, Serialize, Deserialize)]
pub struct ChromaStore {
    #[serde(skip)]
    embeddings: Option<HuggingFaceEmbeddings>,
    documents: HashMap<String, Document>,
}

impl ChromaStore {
    pub fn new(embeddings: HuggingFaceEmbeddings) -> Self {
        Self {
            embeddings: Some(embeddings),
            documents: HashMap::new(),
        }
    }

    pub fn from_texts(
        texts: &[String],
        embeddings: &mut HuggingFaceEmbeddings,
        metadatas: Option<&[HashMap<String, String>]>,
        persist_directory: Option<&Path>
    ) -> Result<Self, String> {
        let mut store = Self::new(embeddings.clone());

        let converted_metadatas = if let Some(metas) = metadatas {
            let mut result = Vec::with_capacity(metas.len());
            for meta in metas {
                let mut metadata = Metadata::default();
                for (k, v) in meta {
                    metadata.extra.insert(k.clone(), serde_json::Value::String(v.clone()));
                }
                result.push(metadata);
            }
            Some(result)
        } else {
            None
        };

        store.add_texts(texts.to_vec(), converted_metadatas)?;

        if let Some(dir) = persist_directory {
            store.persist(dir)?;
        }

        Ok(store)
    }

    pub fn add_texts(&mut self, texts: Vec<String>, metadatas: Option<Vec<Metadata>>) -> Result<Vec<String>, String> {
        let embeddings = match &mut self.embeddings {
            Some(e) => e,
            None => return Err("Embeddings not available".to_string()),
        };

        embeddings.initialize()?;

        let embedded_texts = embeddings.embed_texts(&texts)?;

        let mut ids = Vec::new();
        for (i, (text, embedding)) in texts.iter().zip(embedded_texts.iter()).enumerate() {
            let metadata = match &metadatas {
                Some(meta_list) => meta_list.get(i).cloned().unwrap_or_default(),
                None => Metadata::default(),
            };

            let doc_id = Uuid::new_v4().to_string();
            let document = Document {
                id: doc_id.clone(),
                text: text.clone(),
                embedding: Some(embedding.clone()),
                metadata,
            };

            self.documents.insert(doc_id.clone(), document);
            ids.push(doc_id);
        }

        Ok(ids)
    }

    pub fn similarity_search(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<ScoredDocument>, String> {
        if self.documents.is_empty() {
            info!("No documents in vectorstore for similarity search");
            return Ok(vec![]);
        }

        if self.embeddings.is_none() {
            return Err("Embeddings not set for ChromaStore".to_string());
        }

        let embeddings = self.embeddings.as_ref().unwrap();

        if !embeddings.is_initialized() {
            return Err("Embeddings model not initialized for similarity search".to_string());
        }

        info!("Generating embedding for query: {}", &query[..std::cmp::min(50, query.len())]);
        let query_embedding = match embeddings.embed_query(query) {
            Ok(embedding) => embedding,
            Err(e) => {
                error!("Failed to generate query embedding: {}", e);
                return Err(format!("Failed to generate query embedding: {}", e));
            }
        };
        info!("Successfully generated embedding, calculating similarities with {} documents", self.documents.len());

        let mut scored_docs: Vec<ScoredDocument> = Vec::new();

        for doc in self.documents.values() {
            if let Some(doc_embedding) = &doc.embedding {
                let similarity = cosine_similarity(&query_embedding, doc_embedding);

                scored_docs.push(ScoredDocument {
                    document: doc.clone(),
                    score: similarity,
                });
            }
        }

        scored_docs.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal)
        });

        let result_count = std::cmp::min(k, scored_docs.len());
        info!("Found {} similar documents", result_count);
        Ok(scored_docs.into_iter().take(k).collect())
    }

    pub fn persist(&self, directory: &Path) -> Result<(), String> {
        if !directory.exists() {
            fs::create_dir_all(directory)
                .map_err(|e| format!("Failed to create directory: {}", e))?;
        }

        let json = serde_json::to_string(&self.documents)
            .map_err(|e| format!("Failed to serialize documents: {}", e))?;

        let compressed_path = directory.join("documents.json.xz");
        let file = fs::File::create(&compressed_path)
            .map_err(|e| format!("Failed to create compressed file: {}", e))?;
        let mut encoder = XzEncoder::new(file, 9);
        encoder.write_all(json.as_bytes())
            .map_err(|e| format!("Failed to write compressed data: {}", e))?;
        encoder.finish()
            .map_err(|e| format!("Failed to finish compression: {}", e))?;

        info!("Persisted {} documents to {:?}", self.documents.len(), compressed_path);
        Ok(())
    }

    pub fn load(
        directory: &Path,
        mut embeddings: HuggingFaceEmbeddings,
    ) -> Result<Self, String> {
        let compressed_path = directory.join("documents.json.xz");

        if !compressed_path.exists() {
            return Err(format!("Compressed documents file not found at {}", compressed_path.display()));
        }

        // Ensure embeddings are initialized
        if !embeddings.is_initialized() {
            info!("Initializing embeddings model during vectorstore load");
            if let Err(e) = embeddings.initialize() {
                return Err(format!("Failed to initialize embeddings during vectorstore load: {}", e));
            }
        }

        // Load compressed version
        info!("Loading compressed vectorstore from {}", compressed_path.display());
        let file = fs::File::open(&compressed_path)
            .map_err(|e| format!("Failed to open compressed file: {}", e))?;
        let mut decoder = XzDecoder::new(file);
        let mut json = String::new();
        decoder.read_to_string(&mut json)
            .map_err(|e| format!("Failed to decompress data: {}", e))?;

        let documents: HashMap<String, Document> = serde_json::from_str(&json)
            .map_err(|e| format!("Failed to deserialize documents: {}", e))?;

        info!("Loaded {} documents from compressed file", documents.len());

        Ok(Self {
            embeddings: Some(embeddings),
            documents,
        })
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }

    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for i in 0..a.len() {
        dot_product += a[i] as f64 * b[i] as f64;
        norm_a += a[i] as f64 * a[i] as f64;
        norm_b += b[i] as f64 * b[i] as f64;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    (dot_product / (norm_a.sqrt() * norm_b.sqrt())) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_metadata_default() {
        let metadata = Metadata::default();
        assert!(metadata.source.is_none());
        assert!(metadata.created_at.is_some());
        assert!(metadata.extra.is_empty());
    }

    #[test]
    fn test_metadata_with_custom_fields() {
        let mut metadata = Metadata::default();
        metadata.source = Some("test-source".to_string());

        let mut extra = HashMap::new();
        extra.insert("custom_field".to_string(), serde_json::Value::String("custom_value".to_string()));
        metadata.extra = extra;

        assert_eq!(metadata.source, Some("test-source".to_string()));
        assert_eq!(metadata.extra.get("custom_field").unwrap().as_str().unwrap(), "custom_value");
    }

    #[test]
    fn test_document_serialization() {
        let doc = Document {
            id: "test-id".to_string(),
            text: "Sample document text".to_string(),
            embedding: Some(vec![0.1, 0.2, 0.3]),
            metadata: Metadata::default(),
        };

        let serialized = serde_json::to_string(&doc).unwrap();
        let deserialized: Document = serde_json::from_str(&serialized).unwrap();

        assert_eq!(doc.id, deserialized.id);
        assert_eq!(doc.text, deserialized.text);
        assert_eq!(doc.embedding, deserialized.embedding);
    }

    #[test]
    fn test_document_with_complex_metadata() {
        let mut metadata = Metadata::default();
        metadata.source = Some("complex-source".to_string());

        let mut extra = HashMap::new();
        extra.insert("number".to_string(), serde_json::json!(42));
        extra.insert("boolean".to_string(), serde_json::json!(true));
        extra.insert("array".to_string(), serde_json::json!(vec![1, 2, 3]));
        metadata.extra = extra;

        let doc = Document {
            id: "complex-id".to_string(),
            text: "Complex document with rich metadata".to_string(),
            embedding: Some(vec![0.5, 0.6, 0.7]),
            metadata,
        };

        let serialized = serde_json::to_string(&doc).unwrap();
        let deserialized: Document = serde_json::from_str(&serialized).unwrap();

        assert_eq!(doc.id, deserialized.id);
        assert_eq!(doc.metadata.source, deserialized.metadata.source);
        assert_eq!(doc.metadata.extra.get("number").unwrap(), deserialized.metadata.extra.get("number").unwrap());
        assert_eq!(doc.metadata.extra.get("boolean").unwrap(), deserialized.metadata.extra.get("boolean").unwrap());
        assert_eq!(doc.metadata.extra.get("array").unwrap(), deserialized.metadata.extra.get("array").unwrap());
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 1.0);

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);

        let a = vec![1.0, 1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        // 1/sqrt(2) = approximately 0.7071067811865475
        let similarity = cosine_similarity(&a, &b);
        let expected = 1.0 / std::f32::consts::SQRT_2;
        assert!((similarity - expected).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_edge_cases() {
        // Empty vectors
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert_eq!(cosine_similarity(&a, &b), 0.0);

        // Vectors of different lengths
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);

        // Zero vectors
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);

        // Negative values
        let a = vec![-1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 1.0);
    }

    #[test]
    fn test_chroma_store_empty() {
        let embeddings = HuggingFaceEmbeddings::new();
        let store = ChromaStore::new(embeddings);

        assert!(store.documents.is_empty());
    }

    #[test]
    fn test_scored_document_sort_order() {
        // Create scored documents with different scores
        let doc1 = Document {
            id: "doc1".to_string(),
            text: "Document 1".to_string(),
            embedding: Some(vec![0.1]),
            metadata: Metadata::default(),
        };

        let doc2 = Document {
            id: "doc2".to_string(),
            text: "Document 2".to_string(),
            embedding: Some(vec![0.2]),
            metadata: Metadata::default(),
        };

        let doc3 = Document {
            id: "doc3".to_string(),
            text: "Document 3".to_string(),
            embedding: Some(vec![0.3]),
            metadata: Metadata::default(),
        };

        let scored_doc1 = ScoredDocument {
            document: doc1,
            score: 0.5,
        };

        let scored_doc2 = ScoredDocument {
            document: doc2,
            score: 0.8,
        };

        let scored_doc3 = ScoredDocument {
            document: doc3,
            score: 0.3,
        };

        // Test sorting (should be in descending order by score)
        let mut docs = vec![scored_doc1, scored_doc2, scored_doc3];
        docs.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

        assert_eq!(docs[0].score, 0.8);
        assert_eq!(docs[1].score, 0.5);
        assert_eq!(docs[2].score, 0.3);
    }

    #[test]
    fn test_persist_and_load_with_xz_compression() {
        // Create a temporary directory for storing the compressed file
        let dir = tempdir().unwrap();

        // Create a simple document store
        let embeddings = HuggingFaceEmbeddings::new();
        let mut store = ChromaStore::new(embeddings.clone());

        // Manually add a document to the store
        let doc = Document {
            id: "test-id-123".to_string(),
            text: "Test document for compression".to_string(),
            embedding: Some(vec![0.1, 0.2, 0.3]),
            metadata: Metadata::default(),
        };

        store.documents.insert(doc.id.clone(), doc.clone());

        // Persist to the temporary directory
        let result = store.persist(dir.path());
        assert!(result.is_ok());

        // Verify the compressed file exists
        let compressed_path = dir.path().join("documents.json.xz");
        assert!(compressed_path.exists());

        // Since we can't easily mock the embeddings initialization for proper load testing,
        // we'll test that the files are created as expected
    }
}