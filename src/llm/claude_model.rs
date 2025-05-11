// SPDX-License-Identifier: GPL-2.0
// (c) 2025, Sasha Levin <sashal@kernel.org>

use crate::llm::{LlmError, ModelType, ModelHelper};

/// Claude model types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaudeModelType {
    Sonnet,
    Opus,
    Haiku,
}

impl ModelType for ClaudeModelType {
    fn from_str(s: &str) -> Result<Self, LlmError> {
        match s.to_lowercase().as_str() {
            "sonnet" | "claude-3-sonnet" => Ok(ClaudeModelType::Sonnet),
            "opus" | "claude-3-opus" => Ok(ClaudeModelType::Opus),
            "haiku" | "claude-3-haiku" => Ok(ClaudeModelType::Haiku),
            _ => Err(ModelHelper::invalid_model_error(s, "Claude")),
        }
    }

    fn model_name(&self) -> String {
        match self {
            ClaudeModelType::Sonnet => "claude-3-7-sonnet-20250219".to_string(),
            ClaudeModelType::Opus => "claude-3-7-opus-20250501".to_string(),
            ClaudeModelType::Haiku => "claude-3-haiku-20240307".to_string(),
        }
    }

    fn default_model() -> Self {
        ClaudeModelType::Sonnet
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_claude_model_from_str() {
        assert_eq!(ClaudeModelType::from_str("sonnet").unwrap(), ClaudeModelType::Sonnet);
        assert_eq!(ClaudeModelType::from_str("claude-3-sonnet").unwrap(), ClaudeModelType::Sonnet);
        assert_eq!(ClaudeModelType::from_str("opus").unwrap(), ClaudeModelType::Opus);

        // Test case insensitivity
        assert_eq!(ClaudeModelType::from_str("HAIKU").unwrap(), ClaudeModelType::Haiku);

        // Test invalid model
        let result = ClaudeModelType::from_str("invalid_model");
        assert!(result.is_err());
    }

    #[test]
    fn test_claude_model_name() {
        assert!(ClaudeModelType::Sonnet.model_name().contains("claude-3-7-sonnet"));
        assert!(ClaudeModelType::Opus.model_name().contains("claude-3-7-opus"));
        assert!(ClaudeModelType::Haiku.model_name().contains("claude-3-haiku"));
    }
}
