// SPDX-License-Identifier: GPL-2.0
// (c) 2025, Sasha Levin <sashal@kernel.org>

use async_trait::async_trait;
use serde_json::Value;
use ollama_rs::{
    Ollama,
    generation::{
        completion::{
            request::GenerationRequest,
        },
        options::GenerationOptions,
    },
};
use regex::Regex;
use url::Url;

use crate::llm::{Llm, LlmError};
use crate::llm::provider::{LlmConfig, BaseLlmConfig};
use crate::llm::client::ConfigHelper;

/// Ollama configuration
#[derive(Debug, Clone)]
pub struct OllamaConfig {
    pub model: String,
    pub base: BaseLlmConfig,
    pub api_host: String,
    pub custom_model: Option<String>,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            model: "llama3".to_string(),
            base: BaseLlmConfig::default(),
            api_host: "http://localhost:11434".to_string(),
            custom_model: None,
        }
    }
}

impl LlmConfig for OllamaConfig {
    fn from_json(json: Option<Value>) -> Result<Self, LlmError> {
        let json = json.unwrap_or(Value::Object(serde_json::Map::new()));

        let model = json.get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("llama3")
            .to_string();

        let base = BaseLlmConfig::from_json(&json);

        let api_host = json.get("api_host")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "http://localhost:11434".to_string());

        let custom_model = json.get("custom_model")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        Ok(Self {
            model,
            base,
            api_host,
            custom_model,
        })
    }
}

// Helper function to remove <think>...</think> blocks from the text
pub fn remove_think_blocks(text: &str) -> String {
    let re = Regex::new(r"(?s)<think>.*?</think>").unwrap_or_else(|_| {
        // If regex creation fails, use a simpler approach as fallback
        Regex::new(r"<think>.*</think>").unwrap()
    });

    re.replace_all(text, "").trim().to_string()
}

/// Ollama implementation
pub struct OllamaLlm {
    config: OllamaConfig,
    client: Ollama,
}

impl OllamaLlm {
    pub fn new(config: OllamaConfig) -> Result<Self, LlmError> {
        // Parse the API host URL to extract host and port
        let api_url = Url::parse(&config.api_host)
            .map_err(|e| LlmError::Config(format!("Invalid API host URL: {}", e)))?;

        let scheme = api_url.scheme();
        let host = api_url.host_str().ok_or_else(||
            LlmError::Config("Missing host in API URL".to_string()))?;

        let port = api_url.port().unwrap_or(11434);

        // Construct full host with scheme
        let full_host = format!("{}://{}", scheme, host);

        // Log extracted values for debugging
        log::debug!("Creating Ollama client with host: '{}', port: {}", full_host, port);

        // Create Ollama client
        let client = Ollama::new(full_host, port);

        Ok(Self {
            config,
            client,
        })
    }

    fn get_effective_model(&self) -> String {
        match &self.config.custom_model {
            Some(model) => model.clone(),
            None => self.config.model.clone(),
        }
    }
}

// Implementation of the ConfigHelper trait
impl ConfigHelper for OllamaLlm {
    fn base_config(&self) -> &BaseLlmConfig {
        &self.config.base
    }
}

#[async_trait]
impl Llm for OllamaLlm {
    async fn invoke(&self, prompt: &str) -> Result<String, LlmError> {
        let model = self.get_effective_model();
        let temperature = self.config.base.temperature;
        let max_tokens = self.config.base.max_tokens;

        if self.config.base.debug_logging {
            log::debug!("Sending prompt to Ollama (model: {}, length: {} chars)",
                model, prompt.len());
        }

        // Create options for the generation request
        let options = GenerationOptions::default()
            .temperature(temperature)
            .num_predict(max_tokens as i32);

        // Create generation request using the ollama-rs crate
        let generation_request = GenerationRequest::new(model, prompt.to_string())
            .options(options);

        // Execute the request
        let response = self.client
            .generate(generation_request)
            .await
            .map_err(|e| LlmError::Api(format!("Ollama API error: {}", e)))?;

        if response.response.is_empty() {
            return Err(LlmError::Api(String::from("Empty response in Ollama API response")));
        }

        // Remove the <think>...</think> blocks that appear in models like qwen3
        let response_text = remove_think_blocks(&response.response);

        Ok(response_text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_ollama_config_from_json() {
        // Test default config
        let config = OllamaConfig::from_json(None).unwrap();
        assert_eq!(config.model, "llama3");
        assert_eq!(config.api_host, "http://localhost:11434");
        assert_eq!(config.custom_model, None);

        // Test custom config
        let json = json!({
            "model_type": "mistral",
            "api_host": "http://my-ollama-server:11434",
            "custom_model": "my-custom-model",
            "temperature": 0.7,
            "max_tokens": 2000
        });

        let config = OllamaConfig::from_json(Some(json)).unwrap();
        assert_eq!(config.model, "mistral");
        assert_eq!(config.api_host, "http://my-ollama-server:11434");
        assert_eq!(config.custom_model, Some("my-custom-model".to_string()));
        assert_eq!(config.base.temperature, 0.7);
        assert_eq!(config.base.max_tokens, 2000);
    }

    #[test]
    fn test_remove_think_blocks() {
        // Test with standard think blocks
        let text = "<think>Some thinking content</think>\nActual response";
        assert_eq!(remove_think_blocks(text), "Actual response");

        // Test with multiline think blocks
        let text = "<think>\nLine 1\nLine 2\n</think>\nActual response";
        assert_eq!(remove_think_blocks(text), "Actual response");

        // Test with multiple think blocks
        let text = "<think>First thinking</think>\nSome response\n<think>More thinking</think>\nMore response";
        assert_eq!(remove_think_blocks(text).replace("\n\n", "\n"), "Some response\nMore response");

        // Test with no think blocks
        let text = "Just a normal response";
        assert_eq!(remove_think_blocks(text), "Just a normal response");
    }
}
