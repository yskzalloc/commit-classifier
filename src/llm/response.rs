// SPDX-License-Identifier: GPL-2.0
// (c) 2025, Sasha Levin <sashal@kernel.org>

use serde::Deserialize;

/// Common response formats for LLM providers
/// These representations cover the standard response formats
/// from the major providers and can be extended for others
/// Universal response structure that can be used by all providers
#[derive(Deserialize, Debug)]
pub struct UniversalResponse<T> {
    #[serde(flatten)]
    pub inner: T,
}

/// Provider response formats
/// Claude-style response with content array format
#[derive(Deserialize, Debug)]
pub struct ContentResponse {
    pub content: Vec<ContentItem>,
}

#[derive(Deserialize, Debug)]
pub struct ContentItem {
    pub text: String,
}

/// OpenAI/Nvidia-style response with choices array format
#[derive(Deserialize, Debug)]
pub struct ChoicesResponse {
    pub choices: Vec<Choice>,
}

#[derive(Deserialize, Debug)]
pub struct Choice {
    pub message: ChoiceMessage,
}

#[derive(Deserialize, Debug)]
pub struct ChoiceMessage {
    pub content: String,
}

/// Parser implementations for extracting text from different response formats
pub trait ResponseExtractor {
    /// Extract text from a response
    fn extract_text(&self) -> Result<String, crate::llm::LlmError>;
}

impl ResponseExtractor for ContentResponse {
    fn extract_text(&self) -> Result<String, crate::llm::LlmError> {
        if self.content.is_empty() {
            return Err(crate::llm::LlmError::Api(String::from("Empty content array in API response")));
        }
        Ok(self.content[0].text.clone())
    }
}

impl ResponseExtractor for ChoicesResponse {
    fn extract_text(&self) -> Result<String, crate::llm::LlmError> {
        if self.choices.is_empty() {
            return Err(crate::llm::LlmError::Api(String::from("Empty choices array in API response")));
        }
        Ok(self.choices[0].message.content.clone())
    }
}

/// Specialized response types with convenience methods
pub type NvidiaResponse = UniversalResponse<ChoicesResponse>;

// Create a generic implementation for all UniversalResponse types
impl<T: ResponseExtractor> ResponseExtractor for UniversalResponse<T> {
    fn extract_text(&self) -> Result<String, crate::llm::LlmError> {
        self.inner.extract_text()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_claude_response_extraction() {
        let response_json = json!({
            "content": [
                {"text": "Response from Claude"}
            ]
        });

        let response: UniversalResponse<ContentResponse> = serde_json::from_value(response_json).unwrap();
        let text = response.extract_text().unwrap();
        assert_eq!(text, "Response from Claude");
    }

    #[test]
    fn test_openai_response_extraction() {
        let response_json = json!({
            "choices": [
                {"message": {"content": "Response from OpenAI"}}
            ]
        });

        let response: UniversalResponse<ChoicesResponse> = serde_json::from_value(response_json).unwrap();
        let text = response.extract_text().unwrap();
        assert_eq!(text, "Response from OpenAI");
    }

    #[test]
    fn test_nvidia_response_extraction() {
        let response_json = json!({
            "choices": [
                {"message": {"content": "Response from NVIDIA"}}
            ]
        });

        let response: NvidiaResponse = serde_json::from_value(response_json).unwrap();
        let text = response.extract_text().unwrap();
        assert_eq!(text, "Response from NVIDIA");
    }

    #[test]
    fn test_ollama_think_block_removal() {
        use crate::llm::ollama::remove_think_blocks;

        let text = "<think>This should be removed</think>\nResponse from Ollama";
        let processed_text = remove_think_blocks(text);
        assert_eq!(processed_text, "Response from Ollama");
    }

    #[test]
    fn test_empty_response_handling() {
        let empty_content_json = json!({"content": []});
        let empty_content: ContentResponse = serde_json::from_value(empty_content_json).unwrap();
        let result = empty_content.extract_text();
        assert!(result.is_err());

        let empty_choices_json = json!({"choices": []});
        let empty_choices: ChoicesResponse = serde_json::from_value(empty_choices_json).unwrap();
        let result = empty_choices.extract_text();
        assert!(result.is_err());
    }
}