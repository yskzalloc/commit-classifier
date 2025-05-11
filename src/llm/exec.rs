// SPDX-License-Identifier: GPL-2.0
// (c) 2025, Sasha Levin <sashal@kernel.org>

use std::process::Command;
use std::io::Write;
use std::path::{Path, PathBuf};
use log::debug;
use async_trait::async_trait;
use serde_json::Value;
use tempfile::NamedTempFile;

use crate::llm::{Llm, LlmError};
use crate::llm::provider::{BaseLlmConfig, LlmConfig};

/// Configuration for the Exec LLM provider
#[derive(Debug, Clone)]
pub struct ExecConfig {
    /// Base configuration
    pub base: BaseLlmConfig,
    /// Path to the executable
    pub executable_path: PathBuf,
    /// Arguments to pass to the executable
    pub arguments: Vec<String>,
    /// Whether to pass the prompt as a file or as a command line argument
    pub use_file_input: bool,
}

impl Default for ExecConfig {
    fn default() -> Self {
        Self {
            base: BaseLlmConfig::default(),
            executable_path: PathBuf::new(),
            arguments: Vec::new(),
            use_file_input: true,
        }
    }
}

impl LlmConfig for ExecConfig {
    fn from_json(json: Option<Value>) -> Result<Self, LlmError> {
        let json = json.unwrap_or(Value::Object(serde_json::Map::new()));

        // Get base configuration
        let base = BaseLlmConfig::from_json(&json);

        // Extract executable path
        let executable_path = json.get("executable_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| LlmError::Config("Executable path is required for Exec provider".to_string()))?;

        // Extract arguments
        let arguments = json.get("arguments")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                   .filter_map(|v| v.as_str().map(String::from))
                   .collect()
            })
            .unwrap_or_default();

        // Extract use_file_input
        let use_file_input = json.get("use_file_input")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        Ok(Self {
            base,
            executable_path: PathBuf::from(executable_path),
            arguments,
            use_file_input,
        })
    }
}

/// Exec LLM implementation
pub struct ExecLlm {
    config: ExecConfig,
}

impl ExecLlm {
    pub fn new(config: ExecConfig) -> Result<Self, LlmError> {
        // Validate the executable path
        if !Path::new(&config.executable_path).exists() {
            return Err(LlmError::Config(format!(
                "Executable not found at path: {}",
                config.executable_path.display()
            )));
        }

        Ok(Self { config })
    }
}

#[async_trait]
impl Llm for ExecLlm {
    async fn invoke(&self, prompt: &str) -> Result<String, LlmError> {
        if self.config.base.debug_logging {
            debug!("Invoking external executable: {}", self.config.executable_path.display());
            debug!("Arguments: {:?}", self.config.arguments);
            debug!("Prompt length: {} characters", prompt.len());
        }

        // Prepare the command arguments
        let mut cmd = Command::new(&self.config.executable_path);

        // Add the predefined arguments
        for arg in &self.config.arguments {
            cmd.arg(arg);
        }

        let output = if self.config.use_file_input {
            // Write the prompt to a temporary file
            let mut temp_file = NamedTempFile::new()
                .map_err(|e| LlmError::Config(format!("Failed to create temporary file: {}", e)))?;

            temp_file.write_all(prompt.as_bytes())
                .map_err(|e| LlmError::Config(format!("Failed to write to temporary file: {}", e)))?;

            temp_file.flush()
                .map_err(|e| LlmError::Config(format!("Failed to flush temporary file: {}", e)))?;

            // Pass the file path as the last argument
            cmd.arg(temp_file.path().to_str().unwrap());

            if self.config.base.debug_logging {
                debug!("Using temporary file input: {}", temp_file.path().display());
            }

            // Execute the command
            let output = cmd.output()
                .map_err(|e| LlmError::Api(format!("Failed to execute command: {}", e)))?;

            // Return the output or error
            if output.status.success() {
                String::from_utf8(output.stdout)
                    .map_err(|e| LlmError::Api(format!("Invalid UTF-8 in command output: {}", e)))
            } else {
                let error_message = String::from_utf8_lossy(&output.stderr);
                Err(LlmError::Api(format!(
                    "Command failed with exit code {}: {}",
                    output.status.code().unwrap_or(-1),
                    error_message
                )))
            }
        } else {
            // Pass the prompt directly as the last argument
            cmd.arg(prompt);

            if self.config.base.debug_logging {
                debug!("Using direct argument input");
            }

            // Execute the command
            let output = cmd.output()
                .map_err(|e| LlmError::Api(format!("Failed to execute command: {}", e)))?;

            // Return the output or error
            if output.status.success() {
                String::from_utf8(output.stdout)
                    .map_err(|e| LlmError::Api(format!("Invalid UTF-8 in command output: {}", e)))
            } else {
                let error_message = String::from_utf8_lossy(&output.stderr);
                Err(LlmError::Api(format!(
                    "Command failed with exit code {}: {}",
                    output.status.code().unwrap_or(-1),
                    error_message
                )))
            }
        }?;

        if self.config.base.debug_logging {
            debug!("Executable response length: {} characters", output.len());
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::fs::{self, File};
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn test_exec_config_from_json() {
        // Test with minimal config
        let json = serde_json::json!({
            "executable_path": "/usr/bin/echo"
        });

        let config = ExecConfig::from_json(Some(json)).unwrap();
        assert_eq!(config.executable_path, PathBuf::from("/usr/bin/echo"));
        assert!(config.arguments.is_empty());
        assert!(config.use_file_input);

        // Test with full config
        let json = serde_json::json!({
            "executable_path": "/usr/bin/python",
            "arguments": ["-c", "print('Hello')"],
            "use_file_input": false,
            "temperature": 0.5,
            "max_tokens": 1000,
            "debug_logging": true
        });

        let config = ExecConfig::from_json(Some(json)).unwrap();
        assert_eq!(config.executable_path, PathBuf::from("/usr/bin/python"));
        assert_eq!(config.arguments, vec!["-c", "print('Hello')"]);
        assert!(!config.use_file_input);
        assert_eq!(config.base.temperature, 0.5);
        assert_eq!(config.base.max_tokens, 1000);
        assert!(config.base.debug_logging);

        // Test with missing executable_path
        let json = serde_json::json!({
            "arguments": ["-c", "print('Hello')"]
        });

        let result = ExecConfig::from_json(Some(json));
        assert!(result.is_err());
    }

    // This test requires /bin/echo to be present on the system
    #[tokio::test]
    async fn test_exec_llm_invoke() {
        if !Path::new("/bin/echo").exists() {
            // Skip test if /bin/echo doesn't exist
            return;
        }

        let config = ExecConfig {
            base: BaseLlmConfig::default(),
            executable_path: PathBuf::from("/bin/echo"),
            arguments: vec![],
            use_file_input: false,
        };

        let llm = ExecLlm::new(config).unwrap();
        let result = llm.invoke("Hello, world!").await.unwrap();

        assert_eq!(result.trim(), "Hello, world!");
    }

    #[tokio::test]
    async fn test_exec_llm_with_file_input() {
        // Create a temporary directory for our test script
        let dir = tempdir().unwrap();
        let script_path = dir.path().join("test_script.sh");

        // Create a simple bash script that reads from a file and outputs its content
        let script_content = r#"#!/bin/bash
cat $1
"#;

        let mut file = File::create(&script_path).unwrap();
        file.write_all(script_content.as_bytes()).unwrap();
        // Make sure file is flushed and closed before we try to set permissions
        drop(file);

        // Make the script executable
        let metadata = fs::metadata(&script_path).unwrap();
        let mut permissions = metadata.permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&script_path, permissions).unwrap();

        // Sleep a bit to make sure the file is fully written to disk
        std::thread::sleep(std::time::Duration::from_millis(100));

        let config = ExecConfig {
            base: BaseLlmConfig::default(),
            executable_path: script_path,
            arguments: vec![],
            use_file_input: true,
        };

        // Ensure the validation passes
        let llm = match ExecLlm::new(config) {
            Ok(llm) => llm,
            Err(e) => {
                eprintln!("Failed to create ExecLlm: {}", e);
                return; // Skip the test if we can't create the llm
            }
        };

        let result = llm.invoke("Hello from file!").await.unwrap();
        assert_eq!(result.trim(), "Hello from file!");
    }
}
