// SPDX-License-Identifier: GPL-2.0
// (c) 2025, Sasha Levin <sashal@kernel.org>

use crate::llm::LlmError;

/// Common trait for model types that can be used by LLM providers
pub trait ModelType: Clone + std::fmt::Debug {
    /// Convert a string to a model type
    fn from_str(s: &str) -> Result<Self, LlmError> where Self: Sized;

    /// Get the canonical model name for API requests
    fn model_name(&self) -> String;

    /// Get the default model for this provider
    fn default_model() -> Self where Self: Sized;
}

/// Helper for implementing common model functionality
pub struct ModelHelper;

impl ModelHelper {
    /// Helper function to create a standard model type error
    pub fn invalid_model_error(model_name: &str, provider: &str) -> LlmError {
        LlmError::Config(format!("Unknown {} model type: {}", provider, model_name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Example implementation for testing
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum TestModel {
        ModelA,
        ModelB,
    }

    impl ModelType for TestModel {
        fn from_str(s: &str) -> Result<Self, LlmError> {
            match s.to_lowercase().as_str() {
                "model_a" => Ok(TestModel::ModelA),
                "model_b" => Ok(TestModel::ModelB),
                _ => Err(ModelHelper::invalid_model_error(s, "Test"))
            }
        }

        fn model_name(&self) -> String {
            match self {
                TestModel::ModelA => "test-model-a".to_string(),
                TestModel::ModelB => "test-model-b".to_string(),
            }
        }

        fn default_model() -> Self {
            TestModel::ModelA
        }
    }

    #[test]
    fn test_model_from_str() {
        assert_eq!(TestModel::from_str("model_a").unwrap(), TestModel::ModelA);
        assert_eq!(TestModel::from_str("model_b").unwrap(), TestModel::ModelB);

        // Test case insensitivity
        assert_eq!(TestModel::from_str("MODEL_A").unwrap(), TestModel::ModelA);

        // Test invalid model
        let result = TestModel::from_str("invalid_model");
        assert!(result.is_err());
        match result {
            Err(LlmError::Config(msg)) => {
                assert!(msg.contains("Unknown Test model type"));
            },
            _ => panic!("Expected Config error"),
        }
    }

    #[test]
    fn test_model_name() {
        assert_eq!(TestModel::ModelA.model_name(), "test-model-a");
        assert_eq!(TestModel::ModelB.model_name(), "test-model-b");
    }

    #[test]
    fn test_default_model() {
        assert_eq!(TestModel::default_model(), TestModel::ModelA);
    }
}