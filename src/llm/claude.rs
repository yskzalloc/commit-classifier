// SPDX-License-Identifier: GPL-2.0
// (c) 2025, Sasha Levin <sashal@kernel.org>

use async_trait::async_trait;
use serde_json::Value;
use std::env;
use anthropic_sdk::Client;

use crate::llm::{Llm, LlmError, ModelType};
use crate::llm::provider::{LlmConfig, BaseLlmConfig};
use crate::llm::claude_model::ClaudeModelType;

/// Claude configuration
#[derive(Debug, Clone)]
pub struct ClaudeConfig {
    pub model_type: ClaudeModelType,
    pub base: BaseLlmConfig,
}

impl Default for ClaudeConfig {
    fn default() -> Self {
        Self {
            model_type: ClaudeModelType::default_model(),
            base: BaseLlmConfig::default(),
        }
    }
}

impl LlmConfig for ClaudeConfig {
    fn from_json(json: Option<Value>) -> Result<Self, LlmError> {
        let json = json.unwrap_or(Value::Object(serde_json::Map::new()));

        let model_type = if let Some(mt) = json.get("model_type").and_then(|v| v.as_str()) {
            ClaudeModelType::from_str(mt)?
        } else if let Some(model) = json.get("model").and_then(|v| v.as_str()) {
            // Handle legacy "model" field for backward compatibility
            match model {
                "claude-3-7-sonnet-20250219" => ClaudeModelType::Sonnet,
                "claude-3-7-opus-20250501" => ClaudeModelType::Opus,
                "claude-3-haiku-20240307" => ClaudeModelType::Haiku,
                _ => ClaudeModelType::default_model(),
            }
        } else {
            ClaudeModelType::default_model()
        };

        let base = BaseLlmConfig::from_json(&json);

        Ok(Self {
            model_type,
            base,
        })
    }
}

/// Claude implementation using the Anthropic SDK
pub struct ClaudeLlm {
    config: ClaudeConfig,
    api_key: String,
}

impl ClaudeLlm {
    pub fn new(config: ClaudeConfig) -> Result<Self, LlmError> {
        let api_key = env::var("ANTHROPIC_API_KEY")
            .map_err(|_| LlmError::Config("ANTHROPIC_API_KEY environment variable not set".to_string()))?;
        Ok(Self {
            config,
            api_key,
        })
    }
}

#[async_trait]
impl Llm for ClaudeLlm {
    async fn invoke(&self, prompt: &str) -> Result<String, LlmError> {
        // Debug logging can be added here if needed
        // Create a new client for this request
        let client = Client::new()
            .auth(&self.api_key)
            .model(&self.config.model_type.model_name())
            .temperature(self.config.base.temperature)
            .max_tokens(self.config.base.max_tokens as i32)
            .system("")
            .messages(&serde_json::json!([{
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }]));
        // Get a response using the Request's execute method
        let mut result = String::new();
        let request = client.build()
            .map_err(|e| LlmError::Config(format!("Failed to build Claude request: {}", e)))?;
        // Use the execute method with a callback that captures the response
        request.execute(|chunk| {
            result = chunk;
            async {}
        })
        .await
        .map_err(|e| LlmError::Api(format!("Claude API error: {}", e)))?;
        Ok(result)
    }
}