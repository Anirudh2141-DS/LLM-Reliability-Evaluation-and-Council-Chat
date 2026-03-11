mod handlers;
mod state;

use std::{fs, path::Path};

use axum::{
    routing::{get, post},
    Router,
};
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use rlrgf_models::SystemConfig;
use state::AppState;

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting RLRGF API Server");

    // Load configuration from file (or fallback to defaults if unavailable).
    let config = load_system_config();
    let bind_addr = build_bind_address(&config);

    // Build application state
    let state = AppState::new(config)
        .await
        .expect("Failed to initialize application state");

    // Build router
    let app = Router::new()
        .route("/health", get(handlers::health_check))
        .route("/api/v1/ingest", post(handlers::ingest_document))
        .route("/api/v1/retrieve", post(handlers::retrieve))
        .route("/api/v1/documents", get(handlers::list_documents))
        .route(
            "/api/v1/documents/:doc_id/chunks",
            get(handlers::get_document_chunks),
        )
        .route("/api/v1/stats", get(handlers::system_stats))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    // Start server
    tracing::info!("Listening on {}", bind_addr);
    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .expect("failed to bind TCP listener");
    axum::serve(listener, app).await.unwrap();
}

fn load_system_config() -> SystemConfig {
    let config_path = std::env::var("RLRGF_CONFIG_PATH")
        .unwrap_or_else(|_| "config/default.json".to_string());
    load_system_config_from_path(Path::new(&config_path))
}

fn load_system_config_from_path(path: &Path) -> SystemConfig {
    match fs::read_to_string(path) {
        Ok(raw) => match serde_json::from_str::<SystemConfig>(&raw) {
            Ok(config) => config,
            Err(error) => {
                tracing::warn!(
                    path = %path.display(),
                    %error,
                    "Invalid config file; falling back to SystemConfig::default()"
                );
                SystemConfig::default()
            }
        },
        Err(error) => {
            tracing::warn!(
                path = %path.display(),
                %error,
                "Config file not found/readable; falling back to SystemConfig::default()"
            );
            SystemConfig::default()
        }
    }
}

fn build_bind_address(config: &SystemConfig) -> String {
    format!("{}:{}", config.server.host, config.server.port)
}

#[cfg(test)]
mod tests {
    use super::{build_bind_address, load_system_config_from_path};
    use rlrgf_models::SystemConfig;
    use std::fs;
    use uuid::Uuid;

    #[test]
    fn bind_address_uses_configured_host_and_port() {
        let mut config = SystemConfig::default();
        config.server.host = "127.0.0.1".into();
        config.server.port = 9090;
        assert_eq!(build_bind_address(&config), "127.0.0.1:9090");
    }

    #[test]
    fn load_system_config_reads_host_and_port_from_file() {
        let mut config = SystemConfig::default();
        config.server.host = "127.0.0.1".into();
        config.server.port = 3456;
        let raw = serde_json::to_string(&config).expect("config should serialize");

        let mut path = std::env::temp_dir();
        path.push(format!("rlrgf-config-{}.json", Uuid::new_v4()));
        fs::write(&path, raw).expect("temp config should be written");

        let loaded = load_system_config_from_path(path.as_path());
        let _ = fs::remove_file(path);

        assert_eq!(loaded.server.host, "127.0.0.1");
        assert_eq!(loaded.server.port, 3456);
    }
}
