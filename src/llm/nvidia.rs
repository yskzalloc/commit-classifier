// SPDX-License-Identifier: GPL-2.0
// (c) 2025, Sasha Levin <sashal@kernel.org>

use reqwest::header;
use async_trait::async_trait;
use serde_json::Value;

use crate::http::{self, AuthType};
use crate::llm::{Llm, LlmError, ResponseExtractor, NvidiaResponse, ModelType};
use crate::llm::provider::{LlmConfig, BaseLlmConfig};
use crate::llm::client::{LlmClient, ConfigHelper};

/// NVIDIA model types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvidiaModelType {
    Llama31Ultra,
    Llama3170B,
    Mixtral,
    Qwen110B,
    DeepSeek,
}

impl crate::llm::ModelType for NvidiaModelType {
    fn from_str(s: &str) -> Result<Self, LlmError> {
        match s.to_lowercase().as_str() {
            "llama-3.1-ultra" => Ok(NvidiaModelType::Llama31Ultra),
            "llama-3.1-70b" => Ok(NvidiaModelType::Llama3170B),
            "mixtral" => Ok(NvidiaModelType::Mixtral),
            "qwen-110b" => Ok(NvidiaModelType::Qwen110B),
            "deepseek" => Ok(NvidiaModelType::DeepSeek),
            _ => Err(crate::llm::ModelHelper::invalid_model_error(s, "NVIDIA")),
        }
    }

    fn model_name(&self) -> String {
        match self {
            NvidiaModelType::Llama31Ultra => "meta-llama/Llama-3.1-405b-instruct".to_string(),
            NvidiaModelType::Llama3170B => "meta-llama/Llama-3.1-70b-instruct".to_string(),
            NvidiaModelType::Mixtral => "mistralai/mixtral-8x22b-instruct-v0.1".to_string(),
            NvidiaModelType::Qwen110B => "qwen/qwen1.5-110b-chat".to_string(),
            NvidiaModelType::DeepSeek => "deepseek-ai/deepseek-r1".to_string(),
        }
    }

    fn default_model() -> Self {
        NvidiaModelType::DeepSeek
    }
}

/// NVIDIA configuration
#[derive(Debug, Clone)]
pub struct NvidiaConfig {
    pub model_type: NvidiaModelType,
    pub top_p: f32,
    pub base: BaseLlmConfig,
}

impl Default for NvidiaConfig {
    fn default() -> Self {
        Self {
            model_type: NvidiaModelType::default_model(),
            top_p: 0.7,
            base: BaseLlmConfig::default(),
        }
    }
}

impl LlmConfig for NvidiaConfig {
    fn from_json(json: Option<Value>) -> Result<Self, LlmError> {
        let json = json.unwrap_or(Value::Object(serde_json::Map::new()));

        let model_type = if let Some(mt) = json.get("model_type").and_then(|v| v.as_str()) {
            NvidiaModelType::from_str(mt)?
        } else {
            NvidiaModelType::default_model()
        };

        let top_p = json.get("top_p")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or_else(|| Self::default().top_p);

        let base = BaseLlmConfig::from_json(&json);

        Ok(Self {
            model_type,
            top_p,
            base,
        })
    }
}

/// NVIDIA implementation
pub struct NvidiaLlm {
    config: NvidiaConfig,
    client: reqwest::Client,
}

impl NvidiaLlm {
    pub fn new(config: NvidiaConfig) -> Result<Self, LlmError> {
        let temp = Self {
            config: config.clone(),
            client: reqwest::Client::new(),
        };

        // Initialize the client with proper credentials
        let client = temp.get_http_client()?;

        // Return the fully initialized instance
        Ok(Self {
            config,
            client,
        })
    }
}

// Implementation of the ConfigHelper trait
impl crate::llm::client::ConfigHelper for NvidiaLlm {
    fn base_config(&self) -> &crate::llm::provider::BaseLlmConfig {
        &self.config.base
    }
}


#[async_trait]
impl LlmClient for NvidiaLlm {
    fn api_key_env_var(&self) -> &str {
        "NVIDIA_API_KEY"
    }

    fn create_auth(&self, api_key: String) -> AuthType {
        AuthType::BearerToken(api_key)
    }

    fn custom_headers(&self) -> Vec<(String, String)> {
        vec![
            (header::ACCEPT.to_string(), "application/json".to_string())
        ]
    }

    fn api_url(&self) -> &str {
        "https://integrate.api.nvidia.com/v1/chat/completions"
    }

    fn model_name(&self) -> String {
        self.config.model_type.model_name()
    }

    // Use the helper trait methods for these common properties
    fn temperature(&self) -> f32 {
        self.get_temperature()
    }

    fn max_tokens(&self) -> usize {
        self.get_max_tokens()
    }


    fn max_retries(&self) -> usize {
        self.get_max_retries()
    }

    fn initial_backoff_ms(&self) -> u64 {
        self.get_initial_backoff_ms()
    }
}

#[async_trait]
impl Llm for NvidiaLlm {
    async fn invoke(&self, prompt: &str) -> Result<String, LlmError> {
        // Create the request using the base implementation
        let mut request = self.create_request(prompt);

        // Add Nvidia-specific parameters
        request = request
            .with_top_p(self.config.top_p)
            .with_stream(false);

        // Create enhanced retry config for Nvidia
        let retry_config = self.create_retry_config().with_server_errors();

        // Send the request with custom retry config
        let response: NvidiaResponse = http::post_json(
            &self.client,
            self.api_url(),
            &request,
            Some(retry_config)
        ).await?;

        // Extract the text using the response extractor trait
        response.extract_text()
    }
}
