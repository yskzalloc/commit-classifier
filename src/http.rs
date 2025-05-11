// SPDX-License-Identifier: GPL-2.0
// (c) 2025, Sasha Levin <sashal@kernel.org>

use std::error::Error;
use std::fmt;
use std::time::Duration;
use reqwest::{self, header::{HeaderMap, HeaderValue, HeaderName, self}, StatusCode};
use serde::de::DeserializeOwned;
use log::info;
use tokio::time::sleep;

/// Common HTTP error type
#[derive(Debug)]
pub enum HttpError {
    Request(reqwest::Error),
    Api { status: u16, message: String },
    Parsing(String),
    InvalidHeader(String),
}

impl fmt::Display for HttpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HttpError::Request(err) => write!(f, "HTTP request error: {}", err),
            HttpError::Api { status, message } => write!(f, "API error ({}): {}", status, message),
            HttpError::Parsing(msg) => write!(f, "Response parsing error: {}", msg),
            HttpError::InvalidHeader(msg) => write!(f, "Invalid header: {}", msg),
        }
    }
}

impl Error for HttpError {}

impl From<reqwest::Error> for HttpError {
    fn from(err: reqwest::Error) -> Self {
        HttpError::Request(err)
    }
}

/// HTTP client builder with common configuration
pub struct ClientBuilder {
    headers: HeaderMap,
}

impl Default for ClientBuilder {
    fn default() -> Self {
        Self {
            headers: HeaderMap::new(),
        }
    }
}

impl ClientBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a header with string value
    pub fn with_header(mut self, name: &str, value: &str) -> Result<Self, HttpError> {
        let header_name = HeaderName::from_bytes(name.as_bytes())
            .map_err(|_| HttpError::InvalidHeader(format!("Invalid header name: {}", name)))?;

        let header_value = HeaderValue::from_str(value)
            .map_err(|_| HttpError::InvalidHeader(format!("Invalid header value for {}: {}", name, value)))?;

        self.headers.insert(header_name, header_value);
        Ok(self)
    }

    /// Add content-type JSON header
    pub fn with_json_content_type(mut self) -> Self {
        let value = HeaderValue::from_static("application/json");
        self.headers.insert(header::CONTENT_TYPE, value);
        self
    }

    /// Add authorization bearer token header
    pub fn with_bearer_auth(self, token: &str) -> Result<Self, HttpError> {
        self.with_header("Authorization", &format!("Bearer {}", token))
    }

    /// Build the reqwest HTTP client
    pub fn build(self) -> Result<reqwest::Client, HttpError> {
        let builder = reqwest::Client::builder()
            .default_headers(self.headers);

        builder.build().map_err(HttpError::from)
    }
}

/// Authentication type supported for LLM providers
pub enum AuthType {
    BearerToken(String),
}

/// Create a client configured for an LLM API
pub fn create_llm_client(auth: AuthType, additional_headers: Vec<(String, String)>) -> Result<reqwest::Client, HttpError> {
    let mut builder = ClientBuilder::new().with_json_content_type();

    // Configure authentication
    builder = match auth {
        AuthType::BearerToken(token) => builder.with_bearer_auth(&token)?,
    };

    // Add any additional headers
    for (name, value) in additional_headers {
        builder = builder.with_header(&name, &value)?;
    }

    builder.build()
}

/// Configuration for retrying API requests
pub struct RetryConfig {
    pub max_retries: usize,
    pub initial_backoff_ms: u64,
    pub status_codes: Option<Vec<StatusCode>>,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff_ms: 1000,
            status_codes: Some(vec![StatusCode::TOO_MANY_REQUESTS]),
        }
    }
}

impl RetryConfig {
    /// Create a new retry configuration with custom parameters
    pub fn new(max_retries: usize, initial_backoff_ms: u64) -> Self {
        Self {
            max_retries,
            initial_backoff_ms,
            status_codes: Some(vec![StatusCode::TOO_MANY_REQUESTS]),
        }
    }

    /// Retry on all server errors (5xx)
    pub fn with_server_errors(mut self) -> Self {
        let mut codes = self.status_codes.unwrap_or_default();
        codes.push(StatusCode::INTERNAL_SERVER_ERROR);
        codes.push(StatusCode::BAD_GATEWAY);
        codes.push(StatusCode::SERVICE_UNAVAILABLE);
        codes.push(StatusCode::GATEWAY_TIMEOUT);
        self.status_codes = Some(codes);
        self
    }

    /// Determine if a status code should trigger a retry
    fn should_retry(&self, status: StatusCode) -> bool {
        if let Some(codes) = &self.status_codes {
            codes.contains(&status)
        } else {
            false
        }
    }
}

/// Helper function for making API requests with optional retry capability
pub async fn post_json<T, R>(
    client: &reqwest::Client,
    url: &str,
    body: &T,
    retry_config: Option<RetryConfig>,
) -> Result<R, HttpError>
where
    T: serde::Serialize,
    R: DeserializeOwned,
{
    // If no retry config provided, make a single request
    if retry_config.is_none() {
        let response = client.post(url)
            .json(body)
            .send()
            .await?;

        return process_json_response(response).await;
    }

    // With retry config, handle retries
    let config = retry_config.unwrap();
    let mut backoff_ms = config.initial_backoff_ms;

    for attempt in 0..=config.max_retries {
        let response = client.post(url)
            .json(body)
            .send()
            .await?;

        let status = response.status();

        // If not a retriable status or final attempt, process normally
        if !config.should_retry(status) || attempt == config.max_retries {
            return process_json_response(response).await;
        }

        // Handle rate limit with exponential backoff
        let retry_after = response.headers()
            .get(header::RETRY_AFTER)
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(backoff_ms);

        let wait_time = retry_after.max(backoff_ms);

        // Get error message from response for logging
        let error_text = match response.text().await {
            Ok(text) => text,
            Err(_) => String::from("Unknown error"),
        };

        info!("Request failed with status {}: {}. Retrying in {}ms (attempt {}/{})",
            status, error_text, wait_time, attempt + 1, config.max_retries);

        // Wait before retrying
        sleep(Duration::from_millis(wait_time)).await;

        // Exponential backoff for next attempt
        backoff_ms = backoff_ms.saturating_mul(2);
    }

    Err(HttpError::Api {
        status: StatusCode::TOO_MANY_REQUESTS.as_u16(),
        message: format!("Maximum retries ({}) exceeded for {}", config.max_retries, url),
    })
}

/// Process a JSON response handling error status codes
pub async fn process_json_response<R: DeserializeOwned>(response: reqwest::Response) -> Result<R, HttpError> {
    let status = response.status();

    if !status.is_success() {
        let error_text = response.text().await?;
        return Err(HttpError::Api {
            status: status.as_u16(),
            message: error_text
        });
    }

    let response_text = response.text().await?;

    // Try to parse the response as the expected type
    serde_json::from_str::<R>(&response_text)
        .map_err(|e| HttpError::Parsing(format!("Failed to parse response: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_http_error_display() {
        // Create an API error
        let api_err = HttpError::Api { status: 404, message: "Not Found".to_string() };
        // Create a parsing error
        let parsing_err = HttpError::Parsing("Invalid JSON".to_string());
        // Create a header error
        let header_err = HttpError::InvalidHeader("Bad header".to_string());

        // Test display formatting for each error type
        assert!(format!("{}", api_err).contains("API error (404): Not Found"));
        assert!(format!("{}", parsing_err).contains("Response parsing error: Invalid JSON"));
        assert!(format!("{}", header_err).contains("Invalid header: Bad header"));
    }

    #[test]
    fn test_client_builder_default() {
        let builder = ClientBuilder::default();
        assert!(builder.headers.is_empty());
    }

    #[test]
    fn test_client_builder_new() {
        let builder = ClientBuilder::new();
        assert!(builder.headers.is_empty());
    }

    #[test]
    fn test_client_builder_with_json_content_type() {
        let builder = ClientBuilder::new().with_json_content_type();
        assert_eq!(
            builder.headers.get(header::CONTENT_TYPE).unwrap(),
            "application/json"
        );
    }

    #[test]
    fn test_client_builder_with_valid_header() {
        let result = ClientBuilder::new().with_header("X-Test", "value");
        assert!(result.is_ok());
        let builder = result.unwrap();
        assert_eq!(
            builder.headers.get("X-Test").unwrap().to_str().unwrap(),
            "value"
        );
    }

    #[test]
    fn test_client_builder_with_invalid_header_name() {
        let result = ClientBuilder::new().with_header("Invalid Header", "value");
        assert!(result.is_err());
        match result.err().unwrap() {
            HttpError::InvalidHeader(msg) => {
                assert!(msg.contains("Invalid header name"));
            },
            _ => panic!("Expected InvalidHeader error"),
        }
    }

    #[test]
    fn test_client_builder_with_bearer_auth() {
        let token = "test_token";
        let result = ClientBuilder::new().with_bearer_auth(token);
        assert!(result.is_ok());
        let builder = result.unwrap();
        assert_eq!(
            builder.headers.get("Authorization").unwrap().to_str().unwrap(),
            format!("Bearer {}", token)
        );
    }

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.initial_backoff_ms, 1000);

        // Verify default status codes
        assert!(config.status_codes.is_some());
        let codes = config.status_codes.unwrap();
        assert_eq!(codes.len(), 1);
        assert!(codes.contains(&StatusCode::TOO_MANY_REQUESTS));
    }

    #[test]
    fn test_retry_config_new() {
        let config = RetryConfig::new(5, 2000);
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.initial_backoff_ms, 2000);

        // Verify status codes
        assert!(config.status_codes.is_some());
        let codes = config.status_codes.unwrap();
        assert_eq!(codes.len(), 1);
        assert!(codes.contains(&StatusCode::TOO_MANY_REQUESTS));
    }

    #[test]
    fn test_retry_config_with_server_errors() {
        let config = RetryConfig::default().with_server_errors();

        // Verify all server error codes are included
        assert!(config.status_codes.is_some());
        let codes = config.status_codes.unwrap();
        assert_eq!(codes.len(), 5); // TOO_MANY_REQUESTS + 4 server errors

        assert!(codes.contains(&StatusCode::TOO_MANY_REQUESTS));
        assert!(codes.contains(&StatusCode::INTERNAL_SERVER_ERROR));
        assert!(codes.contains(&StatusCode::BAD_GATEWAY));
        assert!(codes.contains(&StatusCode::SERVICE_UNAVAILABLE));
        assert!(codes.contains(&StatusCode::GATEWAY_TIMEOUT));
    }

    #[test]
    fn test_retry_config_should_retry() {
        let config = RetryConfig::default();

        // Should retry on TOO_MANY_REQUESTS
        assert!(config.should_retry(StatusCode::TOO_MANY_REQUESTS));

        // Should not retry on other status codes
        assert!(!config.should_retry(StatusCode::OK));
        assert!(!config.should_retry(StatusCode::BAD_REQUEST));
        assert!(!config.should_retry(StatusCode::INTERNAL_SERVER_ERROR));

        // With server errors added
        let config_with_server_errors = config.with_server_errors();

        // Should retry on server errors
        assert!(config_with_server_errors.should_retry(StatusCode::TOO_MANY_REQUESTS));
        assert!(config_with_server_errors.should_retry(StatusCode::INTERNAL_SERVER_ERROR));
        assert!(config_with_server_errors.should_retry(StatusCode::BAD_GATEWAY));
        assert!(config_with_server_errors.should_retry(StatusCode::SERVICE_UNAVAILABLE));
        assert!(config_with_server_errors.should_retry(StatusCode::GATEWAY_TIMEOUT));

        // Should still not retry on other status codes
        assert!(!config_with_server_errors.should_retry(StatusCode::OK));
        assert!(!config_with_server_errors.should_retry(StatusCode::BAD_REQUEST));
    }

    #[test]
    fn test_create_llm_client_with_bearer_token() {
        let token = "test_bearer_token";
        let auth = AuthType::BearerToken(token.to_string());
        let result = create_llm_client(auth, vec![]);

        assert!(result.is_ok());
        // We can verify the client was created, but can't inspect headers directly
        // in unit tests without more complex mocking
    }

    #[test]
    fn test_create_llm_client_with_api_key() {
        // Use the BearerToken instead since ApiKey variant was removed
        let auth = AuthType::BearerToken("test_api_key".to_string());
        // Add the header directly in the additional_headers
        let result = create_llm_client(auth, vec![
            ("X-API-Key".to_string(), "test_api_key".to_string())
        ]);

        assert!(result.is_ok());
        // We can verify the client was created successfully
    }

    #[test]
    fn test_create_llm_client_with_additional_headers() {
        let token = "test_bearer_token";
        let auth = AuthType::BearerToken(token.to_string());
        let additional_headers = vec![
            ("X-Custom-Header".to_string(), "custom_value".to_string()),
            ("X-Another-Header".to_string(), "another_value".to_string())
        ];

        let result = create_llm_client(auth, additional_headers);

        assert!(result.is_ok());
        // We can verify the client was created successfully
    }

    #[test]
    fn test_create_llm_client_with_invalid_header() {
        let token = "test_bearer_token";
        let auth = AuthType::BearerToken(token.to_string());
        let additional_headers = vec![
            ("Invalid Header Name".to_string(), "value".to_string())
        ];

        let result = create_llm_client(auth, additional_headers);

        assert!(result.is_err());
        match result.err().unwrap() {
            HttpError::InvalidHeader(msg) => {
                assert!(msg.contains("Invalid header name"));
            },
            _ => panic!("Expected InvalidHeader error"),
        }
    }

    #[test]
    fn test_retry_config_validation() {
        // Test more edge cases for retry config
        let empty_config = RetryConfig {
            max_retries: 0,
            initial_backoff_ms: 0,
            status_codes: None,
        };

        // Should not retry anything with empty status codes
        assert!(!empty_config.should_retry(StatusCode::TOO_MANY_REQUESTS));
        assert!(!empty_config.should_retry(StatusCode::INTERNAL_SERVER_ERROR));

        // Test with custom status codes
        let custom_config = RetryConfig {
            max_retries: 1,
            initial_backoff_ms: 100,
            status_codes: Some(vec![StatusCode::FORBIDDEN, StatusCode::NOT_FOUND]),
        };

        assert!(custom_config.should_retry(StatusCode::FORBIDDEN));
        assert!(custom_config.should_retry(StatusCode::NOT_FOUND));
        assert!(!custom_config.should_retry(StatusCode::TOO_MANY_REQUESTS));
    }

    // Since creating reqwest::Error is difficult in tests, we'll focus on
    // the other error variants that we can test fully

    #[test]
    fn test_http_error_api_variants() {
        // Test API error with various status codes
        let api_err_404 = HttpError::Api { status: 404, message: "Not found".to_string() };
        let err_string_404 = format!("{}", api_err_404);
        assert!(err_string_404.contains("API error (404): Not found"));

        let api_err_429 = HttpError::Api { status: 429, message: "Too many requests".to_string() };
        let err_string_429 = format!("{}", api_err_429);
        assert!(err_string_429.contains("API error (429): Too many requests"));

        let api_err_500 = HttpError::Api { status: 500, message: "Internal server error".to_string() };
        let err_string_500 = format!("{}", api_err_500);
        assert!(err_string_500.contains("API error (500): Internal server error"));
    }

    #[test]
    fn test_client_builder_chaining() {
        // Test method chaining patterns
        let builder_result = ClientBuilder::new()
            .with_json_content_type()
            .with_header("X-Test", "value");

        assert!(builder_result.is_ok());

        // Test invalid header during chaining
        let builder_error = ClientBuilder::new()
            .with_json_content_type()
            .with_header("Invalid Header", "value");

        assert!(builder_error.is_err());
    }

    #[test]
    fn test_post_json_parameter_validation() {
        // Test that retry config parameters are validated correctly
        let config = RetryConfig::new(5, 2000);
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.initial_backoff_ms, 2000);

        // Test with server errors
        let config_with_server_errors = config.with_server_errors();
        assert!(config_with_server_errors.status_codes.is_some());
        let codes = config_with_server_errors.status_codes.unwrap();
        assert!(codes.contains(&StatusCode::INTERNAL_SERVER_ERROR));
    }
}
