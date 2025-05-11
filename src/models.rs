// SPDX-License-Identifier: GPL-2.0
// (c) 2025, Sasha Levin <sashal@kernel.org>

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::path::PathBuf;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitFeatures {
    pub sha: String,
    pub message: String,
    pub diff: String,
    pub author: String,
    pub date: DateTime<Utc>,
    pub files_changed: Vec<String>,
    pub was_selected: Option<bool>,
    pub upstream_sha: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderResult {
    pub should_select: bool,
    pub explanation: String,
    pub error: Option<String>,
    pub raw_response: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    pub sha: String,
    pub subject: String,
    pub should_select: bool,
    pub vote_ratio: String,
    pub provider_results: HashMap<String, ProviderResult>,
}

// Helper struct for model serialization/deserialization
#[derive(Serialize, Deserialize)]
pub struct ModelState {
    pub persist_directory: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commit_features_serialization() {
        let commit = CommitFeatures {
            sha: "abc123".to_string(),
            message: "Test commit".to_string(),
            diff: "diff --git a/file b/file".to_string(),
            author: "Test Author <test@example.com>".to_string(),
            date: Utc::now(),
            files_changed: vec!["file1".to_string(), "file2".to_string()],
            was_selected: Some(true),
            upstream_sha: Some("def456".to_string()),
        };

        let serialized = serde_json::to_string(&commit).unwrap();
        let deserialized: CommitFeatures = serde_json::from_str(&serialized).unwrap();

        assert_eq!(commit.sha, deserialized.sha);
        assert_eq!(commit.message, deserialized.message);
        assert_eq!(commit.diff, deserialized.diff);
        assert_eq!(commit.author, deserialized.author);
        assert_eq!(commit.files_changed, deserialized.files_changed);
        assert_eq!(commit.was_selected, deserialized.was_selected);
        assert_eq!(commit.upstream_sha, deserialized.upstream_sha);
    }

    #[test]
    fn test_provider_result() {
        let result = ProviderResult {
            should_select: true,
            explanation: "This is a bug fix".to_string(),
            error: None,
            raw_response: None,
        };

        assert!(result.should_select);
        assert_eq!(result.explanation, "This is a bug fix");
        assert_eq!(result.error, None);
        assert_eq!(result.raw_response, None);
    }

    #[test]
    fn test_prediction_result() {
        let mut provider_results = HashMap::new();
        provider_results.insert(
            "provider1".to_string(),
            ProviderResult {
                should_select: true,
                explanation: "This is a bug fix".to_string(),
                error: None,
                raw_response: None,
            }
        );

        let prediction = PredictionResult {
            sha: "abc123".to_string(),
            subject: "Test commit".to_string(),
            should_select: true,
            vote_ratio: "1/1".to_string(),
            provider_results,
        };

        assert_eq!(prediction.sha, "abc123");
        assert_eq!(prediction.subject, "Test commit");
        assert!(prediction.should_select);
        assert_eq!(prediction.vote_ratio, "1/1");
        assert_eq!(prediction.provider_results.len(), 1);
        assert!(prediction.provider_results.contains_key("provider1"));
    }
}
