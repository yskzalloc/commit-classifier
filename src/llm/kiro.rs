// SPDX-License-Identifier: GPL-2.0
// (c) 2026, Yunseong Kim <yunseong.kim@est.tech>

//! Kiro CLI provider using the Agent Completion Protocol (ACP).
//! Spawns `kiro-cli acp` and speaks JSON-RPC 2.0 over stdin/stdout.

use async_trait::async_trait;
use log::debug;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;

use crate::llm::provider::{BaseLlmConfig, LlmConfig};
use crate::llm::{Llm, LlmError};

/// Upper bound for a single kiro-cli ACP exchange before we give up
/// and either retry or surface a timeout error to the caller.
const INVOKE_TIMEOUT: Duration = Duration::from_secs(300);

/// Kiro CLI configuration
#[derive(Debug, Clone)]
pub struct KiroConfig {
    pub base: BaseLlmConfig,
    pub binary: PathBuf,
}

impl Default for KiroConfig {
    fn default() -> Self {
        Self {
            base: BaseLlmConfig::default(),
            binary: PathBuf::from("kiro-cli"),
        }
    }
}

impl LlmConfig for KiroConfig {
    fn from_json(json: Option<Value>) -> Result<Self, LlmError> {
        let json = json.unwrap_or(Value::Object(serde_json::Map::new()));
        let base = BaseLlmConfig::from_json(&json);

        let binary = json
            .get("binary")
            .or_else(|| json.get("executable_path"))
            .and_then(|v| v.as_str())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("kiro-cli"));

        Ok(Self { base, binary })
    }
}

/// Kiro LLM implementation using ACP (Agent Completion Protocol)
pub struct KiroLlm {
    config: KiroConfig,
}

impl KiroLlm {
    pub fn new(config: KiroConfig) -> Result<Self, LlmError> {
        if std::env::var("KIRO_API_KEY").is_ok() {
            debug!("kiro: using KIRO_API_KEY for authentication");
        }
        Ok(Self { config })
    }
}

#[async_trait]
impl Llm for KiroLlm {
    async fn invoke(&self, prompt: &str) -> Result<String, LlmError> {
        let max_retries = self.config.base.max_retries;
        let mut delay_ms = self.config.base.initial_backoff_ms;

        for attempt in 0..=max_retries {
            let outcome = tokio::time::timeout(INVOKE_TIMEOUT, self.invoke_once(prompt)).await;
            match outcome {
                Ok(Ok(s)) => return Ok(s),
                // Config errors (missing binary, bad path) won't fix themselves on retry.
                Ok(Err(LlmError::Config(e))) => return Err(LlmError::Config(e)),
                Ok(Err(e)) if attempt < max_retries => {
                    debug!("kiro: attempt {attempt} failed: {e}; retrying in {delay_ms}ms");
                }
                Ok(Err(e)) => return Err(e),
                Err(_) if attempt < max_retries => {
                    debug!(
                        "kiro: attempt {attempt} timed out after {}s; retrying in {delay_ms}ms",
                        INVOKE_TIMEOUT.as_secs()
                    );
                }
                Err(_) => {
                    return Err(LlmError::Api(format!(
                        "kiro-cli timed out after {}s ({} attempts)",
                        INVOKE_TIMEOUT.as_secs(),
                        max_retries + 1
                    )));
                }
            }
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            delay_ms = delay_ms.saturating_mul(2);
        }
        unreachable!("retry loop exits via return")
    }
}

impl KiroLlm {
    async fn invoke_once(&self, prompt: &str) -> Result<String, LlmError> {
        let mut child = Command::new(&self.config.binary)
            .args(["acp", "--trust-tools="])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| {
                LlmError::Config(format!(
                    "Failed to spawn kiro-cli acp: {e}. Is kiro-cli installed?"
                ))
            })?;

        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| LlmError::Config("kiro-cli stdin unavailable".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| LlmError::Config("kiro-cli stdout unavailable".into()))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| LlmError::Config("kiro-cli stderr unavailable".into()))?;
        let mut lines = BufReader::new(stdout).lines();

        // Surface kiro-cli's own diagnostics so auth/version failures aren't
        // hidden behind opaque "kiro-cli exited unexpectedly" messages.
        tokio::spawn(async move {
            let mut stderr_lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = stderr_lines.next_line().await {
                debug!("kiro-cli stderr: {line}");
            }
        });

        // 1. initialize
        send_rpc(
            &mut stdin,
            0,
            "initialize",
            json!({
                "protocolVersion": 1,
                "clientCapabilities": {},
                "clientInfo": {"name": "commit-classifier", "version": env!("CARGO_PKG_VERSION")},
            }),
        )
        .await?;
        let init_result = read_response(&mut lines, 0, None).await?;
        let server_version = init_result.get("protocolVersion").and_then(Value::as_u64);
        if server_version != Some(1) {
            return Err(LlmError::Api(format!(
                "kiro-cli ACP protocolVersion mismatch: expected 1, got {server_version:?}"
            )));
        }

        // 2. session/new
        send_rpc(
            &mut stdin,
            1,
            "session/new",
            json!({
                "cwd": ".",
                "mcpServers": [],
            }),
        )
        .await?;
        let session_result = read_response(&mut lines, 1, None).await?;
        let session_id = session_result
            .get("sessionId")
            .and_then(Value::as_str)
            .ok_or_else(|| LlmError::Api("session/new missing sessionId".into()))?
            .to_string();

        // 3. session/prompt
        send_rpc(
            &mut stdin,
            2,
            "session/prompt",
            json!({
                "sessionId": session_id,
                "prompt": [{"type": "text", "text": prompt}],
            }),
        )
        .await?;

        let mut chunks = Vec::new();
        read_response(&mut lines, 2, Some(&mut chunks)).await?;

        drop(stdin);
        let _ = child.kill().await;

        let result = chunks.join("");
        debug!("kiro-cli acp response length: {} chars", result.len());
        Ok(result)
    }
}

async fn send_rpc(
    stdin: &mut tokio::process::ChildStdin,
    id: u64,
    method: &str,
    params: Value,
) -> Result<(), LlmError> {
    let msg = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
    let mut line = serde_json::to_string(&msg)
        .map_err(|e| LlmError::Api(format!("JSON serialize error: {e}")))?;
    line.push('\n');
    stdin
        .write_all(line.as_bytes())
        .await
        .map_err(|e| LlmError::Api(format!("Failed to write to kiro-cli: {e}")))?;
    stdin
        .flush()
        .await
        .map_err(|e| LlmError::Api(format!("Failed to flush kiro-cli stdin: {e}")))?;
    Ok(())
}

async fn read_response(
    lines: &mut tokio::io::Lines<BufReader<tokio::process::ChildStdout>>,
    target_id: u64,
    mut chunks: Option<&mut Vec<String>>,
) -> Result<Value, LlmError> {
    loop {
        let line = lines
            .next_line()
            .await
            .map_err(|e| LlmError::Api(format!("kiro-cli stdout read error: {e}")))?
            .ok_or_else(|| LlmError::Api("kiro-cli exited unexpectedly".into()))?;

        let msg: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => {
                debug!("kiro-cli: skipping non-JSON line: {line}");
                continue;
            }
        };

        if msg.get("id").and_then(Value::as_u64) == Some(target_id) {
            if let Some(error) = msg.get("error") {
                let message = error
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown ACP error");
                return Err(LlmError::Api(format!("kiro-cli ACP error: {message}")));
            }
            return Ok(msg.get("result").cloned().unwrap_or(Value::Null));
        }

        if let (Some(chunks), Some(text)) = (chunks.as_deref_mut(), extract_text_chunk(&msg)) {
            chunks.push(text);
        }
    }
}

fn extract_text_chunk(msg: &Value) -> Option<String> {
    if msg.get("method")?.as_str()? != "session/update" {
        return None;
    }
    let update = msg.get("params")?.get("update")?;
    let update_type = update
        .get("sessionUpdate")
        .or_else(|| update.get("type"))?
        .as_str()?;
    if !matches!(update_type, "AgentMessageChunk" | "agent_message_chunk") {
        return None;
    }
    extract_content(update.get("content")?)
}

fn extract_content(content: &Value) -> Option<String> {
    match content {
        Value::String(s) => Some(s.clone()),
        Value::Object(map) => map.get("text").and_then(Value::as_str).map(String::from),
        Value::Array(items) => {
            let text: String = items.iter().filter_map(extract_content).collect();
            (!text.is_empty()).then_some(text)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kiro_config_defaults() {
        let config = KiroConfig::from_json(None).unwrap();
        assert_eq!(config.binary, PathBuf::from("kiro-cli"));
        assert_eq!(config.base.temperature, 0.0);
        assert_eq!(config.base.max_tokens, 4000);
    }

    #[test]
    fn test_kiro_config_from_json() {
        let json = serde_json::json!({
            "binary": "/usr/local/bin/kiro-cli",
            "temperature": 0.5,
            "max_tokens": 8000,
            "debug_logging": true
        });
        let config = KiroConfig::from_json(Some(json)).unwrap();
        assert_eq!(config.binary, PathBuf::from("/usr/local/bin/kiro-cli"));
        assert_eq!(config.base.temperature, 0.5);
        assert_eq!(config.base.max_tokens, 8000);
        assert!(config.base.debug_logging);
    }

    #[test]
    fn test_kiro_config_executable_path_alias() {
        let json = serde_json::json!({"executable_path": "/opt/kiro-cli"});
        let config = KiroConfig::from_json(Some(json)).unwrap();
        assert_eq!(config.binary, PathBuf::from("/opt/kiro-cli"));
    }

    #[test]
    fn test_extract_text_chunk_snake_case() {
        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "s1",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "hello"}
                }
            }
        });
        assert_eq!(extract_text_chunk(&msg).as_deref(), Some("hello"));
    }

    #[test]
    fn test_extract_text_chunk_camel_case() {
        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "s1",
                "update": {
                    "sessionUpdate": "AgentMessageChunk",
                    "content": [
                        {"type": "text", "text": "hel"},
                        {"type": "text", "text": "lo"}
                    ]
                }
            }
        });
        assert_eq!(extract_text_chunk(&msg).as_deref(), Some("hello"));
    }

    #[test]
    fn test_extract_text_chunk_ignores_non_text() {
        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "s1",
                "update": {"sessionUpdate": "tool_call", "content": {"text": "ignored"}}
            }
        });
        assert!(extract_text_chunk(&msg).is_none());
    }
}
