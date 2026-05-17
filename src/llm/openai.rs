// SPDX-License-Identifier: GPL-2.0
// (c) 2025, Sasha Levin <sashal@kernel.org>

use async_trait::async_trait;
use serde_json::Value;
use async_openai::{
    types::chat::{
        ChatCompletionRequestMessage,
        ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs,
    },
    config::OpenAIConfig as AsyncOpenAIConfig,
    Client,
};

use crate::llm::{Llm, LlmError, ModelType};
use crate::llm::provider::{LlmConfig, BaseLlmConfig};

/// OpenAI model types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAIModelType {
    GPT4Turbo,
    GPT4O,
}

impl crate::llm::ModelType for OpenAIModelType {
    fn from_str(s: &str) -> Result<Self, LlmError> {
        match s.to_lowercase().as_str() {
            "gpt4" | "gpt4-turbo" => Ok(OpenAIModelType::GPT4Turbo),
            "gpt4o" => Ok(OpenAIModelType::GPT4O),
            _ => Err(crate::llm::ModelHelper::invalid_model_error(s, "OpenAI")),
        }
    }

    fn model_name(&self) -> String {
        match self {
            OpenAIModelType::GPT4Turbo => "gpt-4-turbo-2024-04-09".to_string(),
            OpenAIModelType::GPT4O => "gpt-4o-2024-05-13".to_string(),
        }
    }

    fn default_model() -> Self {
        OpenAIModelType::GPT4O
    }
}

/// OpenAI configuration
#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    pub model_type: OpenAIModelType,
    pub base: BaseLlmConfig,
}

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            model_type: OpenAIModelType::default_model(),
            base: BaseLlmConfig::default(),
        }
    }
}

impl LlmConfig for OpenAIConfig {
    fn from_json(json: Option<Value>) -> Result<Self, LlmError> {
        let json = json.unwrap_or(Value::Object(serde_json::Map::new()));

        let model_type = if let Some(mt) = json.get("model_type").and_then(|v| v.as_str()) {
            OpenAIModelType::from_str(mt)?
        } else {
            OpenAIModelType::default_model()
        };

        let base = BaseLlmConfig::from_json(&json);

        Ok(Self {
            model_type,
            base,
        })
    }
}

/// OpenAI implementation using async-openai crate
pub struct OpenAILlm {
    config: OpenAIConfig,
    client: Client<AsyncOpenAIConfig>,
}

impl OpenAILlm {
    pub fn new(config: OpenAIConfig) -> Result<Self, LlmError> {
        // Create the async-openai client - it automatically reads OPENAI_API_KEY
        // from the environment, so no need to handle that manually
        let client = Client::new();
        Ok(Self {
            config,
            client,
        })
    }
}

// Implementation of the ConfigHelper trait
impl crate::llm::client::ConfigHelper for OpenAILlm {
    fn base_config(&self) -> &crate::llm::provider::BaseLlmConfig {
        &self.config.base
    }
}

#[async_trait]
impl Llm for OpenAILlm {
    async fn invoke(&self, prompt: &str) -> Result<String, LlmError> {
        // Convert the input messages to async-openai format
        let user_message = ChatCompletionRequestUserMessageArgs::default()
            .content(prompt.to_string())
            .build()
            .map_err(|e| LlmError::Config(format!("Failed to build message: {}", e)))?;

        // Convert user message to ChatCompletionRequestMessage
        let message: ChatCompletionRequestMessage = user_message.into();

        // Build the completion request
        let request = CreateChatCompletionRequestArgs::default()
            .model(self.config.model_type.model_name())
            .messages(vec![message])
            .temperature(self.config.base.temperature)
            .max_completion_tokens(self.config.base.max_tokens as u32)
            .build()
            .map_err(|e| LlmError::Config(format!("Failed to build request: {}", e)))?;

        // Log debug information if enabled
        if self.config.base.debug_logging {
            log::debug!("Sending prompt to OpenAI (model: {}, length: {} chars)",
                self.config.model_type.model_name(), prompt.len());
        }

        // Send the request and get the response
        let response = self.client.chat()
            .create(request)
            .await
            .map_err(|e| LlmError::Api(format!("OpenAI API error: {}", e)))?;

        // Extract the response text
        if let Some(choice) = response.choices.first() {
            if let Some(content) = &choice.message.content {
                return Ok(content.clone());
            }
        }

        // Return an error if no response was received
        Err(LlmError::Api("No response content from OpenAI API".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_openai_model_type_from_str() {
        assert_eq!(OpenAIModelType::from_str("gpt4").unwrap(), OpenAIModelType::GPT4Turbo);
        assert_eq!(OpenAIModelType::from_str("gpt4-turbo").unwrap(), OpenAIModelType::GPT4Turbo);
        assert_eq!(OpenAIModelType::from_str("gpt4o").unwrap(), OpenAIModelType::GPT4O);
        // Test case insensitivity
        assert_eq!(OpenAIModelType::from_str("GPT4").unwrap(), OpenAIModelType::GPT4Turbo);
        assert_eq!(OpenAIModelType::from_str("GPT4O").unwrap(), OpenAIModelType::GPT4O);
        // Test invalid model
        assert!(OpenAIModelType::from_str("invalid_model").is_err());
    }
    #[test]
    fn test_openai_model_name() {
        assert_eq!(OpenAIModelType::GPT4Turbo.model_name(), "gpt-4-turbo-2024-04-09");
        assert_eq!(OpenAIModelType::GPT4O.model_name(), "gpt-4o-2024-05-13");
    }
    #[test]
    fn test_openai_config_from_json() {
        // Test default config
        let config = OpenAIConfig::from_json(None).unwrap();
        assert_eq!(config.model_type, OpenAIModelType::default_model());
        // Test custom config
        let json = json!({
            "model_type": "gpt4-turbo",
            "temperature": 0.7,
            "max_tokens": 2000
        });
        let config = OpenAIConfig::from_json(Some(json)).unwrap();
        assert_eq!(config.model_type, OpenAIModelType::GPT4Turbo);
        assert_eq!(config.base.temperature, 0.7);
        assert_eq!(config.base.max_tokens, 2000);
    }
}
