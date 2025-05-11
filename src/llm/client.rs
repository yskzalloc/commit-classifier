// SPDX-License-Identifier: GPL-2.0
// (c) 2025, Sasha Levin <sashal@kernel.org>

use std::env;
use async_trait::async_trait;
use crate::http::{RetryConfig, AuthType, create_llm_client};
use crate::llm::{Llm, LlmError, Message, UniversalLlmRequest};

/// Base helper for accessing config values from BaseLlmConfig
pub trait ConfigHelper {
    /// Get reference to BaseLlmConfig
    fn base_config(&self) -> &crate::llm::provider::BaseLlmConfig;

    /// Get temperature value
    fn get_temperature(&self) -> f32 {
        self.base_config().temperature
    }

    /// Get max tokens value
    fn get_max_tokens(&self) -> usize {
        self.base_config().max_tokens
    }


    /// Get max retries value
    fn get_max_retries(&self) -> usize {
        self.base_config().max_retries
    }

    /// Get initial backoff value
    fn get_initial_backoff_ms(&self) -> u64 {
        self.base_config().initial_backoff_ms
    }
}

/// Base trait for LLM client implementations
#[async_trait]
pub trait LlmClient: Llm {
    /// Name of the environment variable holding the API key
    fn api_key_env_var(&self) -> &str;

    /// Creates authentication configuration for the client
    fn create_auth(&self, api_key: String) -> AuthType;

    /// Returns custom headers required for this LLM provider
    fn custom_headers(&self) -> Vec<(String, String)>;

    /// Returns the base API URL for the LLM provider
    fn api_url(&self) -> &str;

    /// Returns the model name to use for this provider
    fn model_name(&self) -> String;

    /// Returns the temperature to use for this provider
    fn temperature(&self) -> f32;

    /// Returns the max tokens to use for this provider
    fn max_tokens(&self) -> usize;

    /// Returns max retries for API calls
    fn max_retries(&self) -> usize;

    /// Returns initial backoff in ms for retries
    fn initial_backoff_ms(&self) -> u64;

    /// Creates a client with proper authentication and headers
    fn initialize_client(&self) -> Result<reqwest::Client, LlmError> {
        let api_key = env::var(self.api_key_env_var())
            .map_err(|_| LlmError::Config(format!("{} environment variable not set", self.api_key_env_var())))?;

        create_llm_client(
            self.create_auth(api_key),
            self.custom_headers()
        ).map_err(|e| e.into())
    }

    /// Helper to get an LLM client with initialized credentials
    fn get_http_client(&self) -> Result<reqwest::Client, LlmError> {
        self.initialize_client()
    }

    /// Helper to create a universal request with proper parameters
    fn create_request(&self, prompt: &str) -> UniversalLlmRequest {
        UniversalLlmRequest::new(
            self.model_name(),
            vec![Message::user(prompt.to_string())],
            self.temperature(),
            self.max_tokens()
        )
    }

    /// Helper to create a retry configuration
    fn create_retry_config(&self) -> RetryConfig {
        RetryConfig::new(
            self.max_retries(),
            self.initial_backoff_ms()
        )
    }

}
