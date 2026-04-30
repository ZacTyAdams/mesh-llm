//! Peer consultation — ask another model in the mesh for help.
//!
//! This is the core mechanism behind the virtual LLM engine. When a hook
//! fires and decides to consult another model, it calls into this module
//! to find a suitable peer and send it a request over the mesh's QUIC
//! transport.
//!
//! Three consultation patterns:
//!
//! - **Caption** — send an image to a vision-capable peer, get a text description
//! - **Summarize** — send conversation history, get a condensed summary
//! - **Second opinion** — send the same question to a different model, get its answer

use crate::mesh;
use anyhow::Result;
use iroh::EndpointId;
use serde_json::Value;
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ConsultationPerformanceHint {
    avg_ttft_ms: Option<u32>,
    avg_tokens_per_second_milli: Option<u32>,
    rtt_ms: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsultationRequestClass {
    Interactive,
    Throughput,
}

fn compare_optional_ascending(left: Option<u32>, right: Option<u32>) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => left.cmp(&right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn compare_optional_descending(left: Option<u32>, right: Option<u32>) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => right.cmp(&left),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn compare_consultation_performance(
    left: ConsultationPerformanceHint,
    right: ConsultationPerformanceHint,
    request_class: ConsultationRequestClass,
) -> Ordering {
    match request_class {
        ConsultationRequestClass::Interactive => {
            compare_optional_ascending(left.avg_ttft_ms, right.avg_ttft_ms)
                .then_with(|| {
                    compare_optional_descending(
                        left.avg_tokens_per_second_milli,
                        right.avg_tokens_per_second_milli,
                    )
                })
                .then_with(|| compare_optional_ascending(left.rtt_ms, right.rtt_ms))
        }
        ConsultationRequestClass::Throughput => compare_optional_descending(
            left.avg_tokens_per_second_milli,
            right.avg_tokens_per_second_milli,
        )
        .then_with(|| compare_optional_ascending(left.avg_ttft_ms, right.avg_ttft_ms))
        .then_with(|| compare_optional_ascending(left.rtt_ms, right.rtt_ms)),
    }
}

fn model_performance_hint(peer: &mesh::PeerInfo, model_name: &str) -> ConsultationPerformanceHint {
    ConsultationPerformanceHint {
        avg_ttft_ms: peer.advertised_avg_ttft_ms(model_name),
        avg_tokens_per_second_milli: peer.advertised_avg_tokens_per_second_milli(model_name),
        rtt_ms: peer.rtt_ms,
    }
}

fn select_capability_peer_from_peers<F>(
    peers: &[mesh::PeerInfo],
    exclude_model: &str,
    request_class: ConsultationRequestClass,
    supports: F,
) -> Option<(EndpointId, String)>
where
    F: Fn(&mesh::ServedModelDescriptor) -> bool,
{
    peers
        .iter()
        .filter_map(|peer| {
            peer.served_model_descriptors
                .iter()
                .filter(|descriptor| {
                    supports(descriptor)
                        && descriptor.identity.model_name != exclude_model
                        && !descriptor.identity.model_name.is_empty()
                })
                .map(|descriptor| {
                    let model_name = descriptor.identity.model_name.clone();
                    (
                        peer.id,
                        model_name.clone(),
                        model_performance_hint(peer, &model_name),
                    )
                })
                .min_by(|(_, left_model, left_perf), (_, right_model, right_perf)| {
                    compare_consultation_performance(*left_perf, *right_perf, request_class)
                        .then_with(|| left_model.cmp(right_model))
                })
        })
        .min_by(
            |(left_id, left_model, left_perf), (right_id, right_model, right_perf)| {
                compare_consultation_performance(*left_perf, *right_perf, request_class)
                    .then_with(|| left_model.cmp(right_model))
                    .then_with(|| left_id.as_bytes().cmp(right_id.as_bytes()))
            },
        )
        .map(|(peer_id, model_name, _)| (peer_id, model_name))
}

fn find_different_model_peers_from_peers(
    peers: &[mesh::PeerInfo],
    current_model: &str,
    n: usize,
    request_class: ConsultationRequestClass,
) -> Vec<(EndpointId, String)> {
    use crate::models::CapabilityLevel;

    let mut candidates: Vec<_> = peers
        .iter()
        .flat_map(|peer| {
            peer.served_model_descriptors
                .iter()
                .filter(|descriptor| {
                    descriptor.identity.model_name != current_model
                        && !descriptor.identity.model_name.is_empty()
                })
                .map(|descriptor| {
                    let model_name = descriptor.identity.model_name.clone();
                    (
                        peer.id,
                        model_name.clone(),
                        descriptor.capabilities.reasoning != CapabilityLevel::None,
                        model_performance_hint(peer, &model_name),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect();

    candidates.sort_by(
        |(left_id, left_model, left_reasoning, left_perf),
         (right_id, right_model, right_reasoning, right_perf)| {
            right_reasoning
                .cmp(left_reasoning)
                .then_with(|| {
                    compare_consultation_performance(*left_perf, *right_perf, request_class)
                })
                .then_with(|| left_model.cmp(right_model))
                .then_with(|| left_id.as_bytes().cmp(right_id.as_bytes()))
        },
    );

    let mut seen_models = std::collections::HashSet::new();
    candidates.retain(|(_, model, _, _)| seen_models.insert(model.clone()));
    candidates.truncate(n);
    candidates
        .into_iter()
        .map(|(id, model, _, _)| (id, model))
        .collect()
}

// ---------------------------------------------------------------------------
// Peer discovery
// ---------------------------------------------------------------------------

/// Find a peer that can handle vision (images).
/// Returns None if no vision-capable peer exists in the mesh.
pub async fn find_vision_peer(
    node: &mesh::Node,
    exclude_model: &str,
    request_class: ConsultationRequestClass,
) -> Option<(EndpointId, String)> {
    let peers = node.peers().await;
    select_capability_peer_from_peers(&peers, exclude_model, request_class, |descriptor| {
        descriptor.capabilities.supports_vision_runtime()
    })
}

/// Find a peer that can handle audio.
/// Returns None if no audio-capable peer exists in the mesh.
pub async fn find_audio_peer(
    node: &mesh::Node,
    exclude_model: &str,
    request_class: ConsultationRequestClass,
) -> Option<(EndpointId, String)> {
    let peers = node.peers().await;
    select_capability_peer_from_peers(&peers, exclude_model, request_class, |descriptor| {
        descriptor.capabilities.supports_audio_runtime()
    })
}

/// Find up to `n` peers serving a *different* model from the current one,
/// ranked by score (best first).
///
/// Picks peers running a different model for diversity. Prefers reasoning-capable
/// models, then lower RTT. Deduplicates by model name — two nodes running the
/// same model don't give diversity, just redundancy.
pub async fn find_different_model_peers(
    node: &mesh::Node,
    current_model: &str,
    n: usize,
    request_class: ConsultationRequestClass,
) -> Vec<(EndpointId, String)> {
    let peers = node.peers().await;
    find_different_model_peers_from_peers(&peers, current_model, n, request_class)
}

// ---------------------------------------------------------------------------
// Consultation requests
// ---------------------------------------------------------------------------

/// Consultation timeout — 20s for all hooks. Triggers are rare enough that
/// a pause is acceptable, and mesh peers often need 6-10s to respond.
pub const TIMEOUT_CONSULTATION: std::time::Duration = std::time::Duration::from_secs(20);

/// Send a chat completion request to a peer over the mesh QUIC tunnel.
/// Returns the assistant message content, or an error.
pub async fn chat_completion(
    node: &mesh::Node,
    peer_id: EndpointId,
    model: &str,
    messages: Vec<Value>,
    max_tokens: u32,
    timeout: std::time::Duration,
) -> Result<String> {
    match tokio::time::timeout(
        timeout,
        chat_completion_inner(node, peer_id, model, messages, max_tokens),
    )
    .await
    {
        Ok(result) => result,
        Err(_) => anyhow::bail!("consultation timed out after {}s", timeout.as_secs()),
    }
}

async fn chat_completion_inner(
    node: &mesh::Node,
    peer_id: EndpointId,
    model: &str,
    messages: Vec<Value>,
    max_tokens: u32,
) -> Result<String> {
    let request_body = serde_json::json!({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": false,
        // Disable hooks on the peer — prevent recursive consultation loops.
        // Without this, the peer could consult another peer about our request,
        // which could consult another, etc.
        "mesh_hooks": false,
    });
    let body_bytes = serde_json::to_vec(&request_body)?;

    // Build a minimal HTTP request
    let http_request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         \r\n",
        body_bytes.len()
    );

    let mut raw = http_request.into_bytes();
    raw.extend_from_slice(&body_bytes);

    // Open QUIC tunnel to peer and send request
    let (mut send, mut recv) = node.open_http_tunnel(peer_id).await?;
    send.write_all(&raw).await?;
    send.finish()?;

    // Read the full HTTP response
    let response_bytes = recv.read_to_end(64 * 1024).await?;
    let response_str = String::from_utf8_lossy(&response_bytes);

    // Parse HTTP status line
    let header_end = response_str
        .find("\r\n\r\n")
        .ok_or_else(|| anyhow::anyhow!("malformed HTTP response: no header terminator"))?;
    let headers = &response_str[..header_end];
    let status_line = headers.lines().next().unwrap_or("");
    let status_code: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if status_code != 200 {
        anyhow::bail!(
            "peer returned HTTP {status_code}: {}",
            &response_str[..response_str.len().min(200)]
        );
    }

    let body = &response_str[header_end + 4..];
    let parsed: Value = serde_json::from_str(body).map_err(|e| {
        anyhow::anyhow!(
            "failed to parse peer response body: {e}\nraw: {}",
            &body[..body.len().min(200)]
        )
    })?;

    // Extract the assistant message content
    let content = parsed["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();

    if content.is_empty() {
        anyhow::bail!("peer returned empty response");
    }

    Ok(content)
}

// ---------------------------------------------------------------------------
// High-level consultation patterns
// ---------------------------------------------------------------------------

/// Ask a vision peer to caption an image.
/// `image_url` should be the full data URL (data:image/png;base64,...).
pub async fn caption_image(
    node: &mesh::Node,
    peer_id: EndpointId,
    model: &str,
    image_url: &str,
    user_text: &str,
) -> Result<String> {
    let prompt = if user_text.is_empty() {
        "Describe this image concisely in one paragraph.".to_string()
    } else {
        format!("The user asked: \"{user_text}\"\n\nDescribe this image concisely, focusing on details relevant to the user's question.")
    };

    let messages = vec![serde_json::json!({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    })];

    chat_completion(node, peer_id, model, messages, 256, TIMEOUT_CONSULTATION).await
}

/// Ask a peer for a second opinion on the user's question.
///
/// Sends only the last user message (not the full conversation) and asks
/// for a short, direct answer. The result is injected into the uncertain
/// model's KV cache as context — it should be concise (a fact, a key point,
/// a starting direction), not a full essay.
pub async fn second_opinion(
    node: &mesh::Node,
    peer_id: EndpointId,
    model: &str,
    messages: &[Value],
    timeout: std::time::Duration,
) -> Result<String> {
    // Extract just the last user message text
    let last_user_text = messages
        .iter()
        .rev()
        .find(|m| m["role"].as_str() == Some("user"))
        .and_then(|m| {
            // Handle both string content and multimodal array content
            if let Some(s) = m["content"].as_str() {
                Some(s.to_string())
            } else if let Some(parts) = m["content"].as_array() {
                parts
                    .iter()
                    .find(|p| p["type"].as_str() == Some("text"))
                    .and_then(|p| p["text"].as_str())
                    .map(|s| s.to_string())
            } else {
                None
            }
        })
        .unwrap_or_default();

    if last_user_text.is_empty() {
        anyhow::bail!("no user message found for second opinion");
    }

    // Truncate very long user messages — we want a fast answer
    let user_text = if last_user_text.len() > 2000 {
        let end = last_user_text
            .char_indices()
            .take_while(|(i, _)| *i < 2000)
            .last()
            .map_or(0, |(i, c)| i + c.len_utf8());
        format!("{}...", &last_user_text[..end])
    } else {
        last_user_text
    };

    let ask_messages = vec![serde_json::json!({
        "role": "user",
        "content": format!(
            "Answer this briefly and directly in 2-3 sentences:\n\n{user_text}"
        )
    })];

    chat_completion(node, peer_id, model, ask_messages, 192, timeout).await
}

/// Fan out a second-opinion request to up to 2 peers, return the first
/// response. If only one peer is available, falls back to a single call.
pub async fn race_second_opinion(
    node: &mesh::Node,
    peers: &[(EndpointId, String)],
    messages: &[Value],
    timeout: std::time::Duration,
) -> Option<(String, EndpointId, String)> {
    if peers.is_empty() {
        return None;
    }

    if peers.len() == 1 {
        let (id, model) = &peers[0];
        return match second_opinion(node, *id, model, messages, timeout).await {
            Ok(text) => Some((text, *id, model.clone())),
            Err(e) => {
                tracing::warn!(
                    "virtual: second opinion from {} failed: {e}",
                    id.fmt_short()
                );
                None
            }
        };
    }

    // Race two peers — fire both via JoinSet, take first Ok, abort the rest.
    let mut set = tokio::task::JoinSet::new();

    for (id, model) in peers.iter().skip(1).take(1) {
        let node = node.clone();
        let msgs = messages.to_vec();
        let id = *id;
        let model = model.clone();
        let t = timeout;
        set.spawn(async move {
            second_opinion(&node, id, &model, &msgs, t)
                .await
                .map(|text| (text, id, model))
        });
    }
    // Spawn the best peer last so it appears in the set too
    {
        let node = node.clone();
        let msgs = messages.to_vec();
        let id = peers[0].0;
        let model = peers[0].1.clone();
        let t = timeout;
        set.spawn(async move {
            second_opinion(&node, id, &model, &msgs, t)
                .await
                .map(|text| (text, id, model))
        });
    }

    while let Some(result) = set.join_next().await {
        if let Ok(Ok((text, id, model))) = result {
            tracing::info!("virtual: peer {} ({model}) won the race", id.fmt_short());
            set.abort_all();
            return Some((text, id, model));
        }
    }

    tracing::warn!("virtual: all peers failed");
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::OwnershipSummary;
    use crate::mesh::{
        ModelRuntimeDescriptor, NodeRole, PeerInfo, ServedModelDescriptor, ServedModelIdentity,
    };
    use crate::models::{CapabilityLevel, ModelCapabilities};
    use iroh::{EndpointAddr, SecretKey};
    use std::collections::HashMap;

    fn test_endpoint_id(seed: u8) -> EndpointId {
        EndpointId::from(SecretKey::from_bytes(&[seed; 32]).public())
    }

    fn test_peer(
        seed: u8,
        rtt_ms: Option<u32>,
        descriptors: Vec<ServedModelDescriptor>,
        runtimes: Vec<ModelRuntimeDescriptor>,
    ) -> PeerInfo {
        let id = test_endpoint_id(seed);
        PeerInfo {
            id,
            addr: EndpointAddr {
                id,
                addrs: Default::default(),
            },
            tunnel_port: None,
            role: NodeRole::Host { http_port: 9337 },
            first_joined_mesh_ts: None,
            models: vec![],
            vram_bytes: 0,
            rtt_ms,
            model_source: None,
            serving_models: descriptors
                .iter()
                .map(|descriptor| descriptor.identity.model_name.clone())
                .collect(),
            hosted_models: vec![],
            hosted_models_known: false,
            available_models: vec![],
            requested_models: vec![],
            last_seen: std::time::Instant::now(),
            last_mentioned: std::time::Instant::now(),
            moe_recovered_at: None,
            version: None,
            gpu_name: None,
            hostname: None,
            is_soc: None,
            gpu_vram: None,
            gpu_reserved_bytes: None,
            gpu_mem_bandwidth_gbps: None,
            gpu_compute_tflops_fp32: None,
            gpu_compute_tflops_fp16: None,
            available_model_metadata: vec![],
            experts_summary: None,
            available_model_sizes: HashMap::new(),
            served_model_descriptors: descriptors,
            served_model_runtime: runtimes,
            owner_attestation: None,
            owner_summary: OwnershipSummary::default(),
        }
    }

    fn descriptor(
        model_name: &str,
        vision: bool,
        audio: bool,
        reasoning: CapabilityLevel,
    ) -> ServedModelDescriptor {
        ServedModelDescriptor {
            identity: ServedModelIdentity {
                model_name: model_name.to_string(),
                ..Default::default()
            },
            capabilities: ModelCapabilities {
                vision: if vision {
                    CapabilityLevel::Supported
                } else {
                    CapabilityLevel::None
                },
                audio: if audio {
                    CapabilityLevel::Supported
                } else {
                    CapabilityLevel::None
                },
                reasoning,
                ..Default::default()
            },
            topology: None,
        }
    }

    fn runtime(
        model_name: &str,
        tps_milli: Option<u32>,
        ttft_ms: Option<u32>,
    ) -> ModelRuntimeDescriptor {
        ModelRuntimeDescriptor {
            model_name: model_name.to_string(),
            identity_hash: None,
            context_length: Some(8192),
            ready: true,
            avg_tokens_per_second_milli: tps_milli,
            avg_ttft_ms: ttft_ms,
        }
    }

    #[test]
    fn select_capability_peer_prefers_lower_ttft_over_rtt() {
        let fast_rtt_slow_model = test_peer(
            1,
            Some(20),
            vec![descriptor(
                "vision-slow",
                true,
                false,
                CapabilityLevel::None,
            )],
            vec![runtime("vision-slow", Some(18_000), Some(900))],
        );
        let slower_rtt_fast_model = test_peer(
            2,
            Some(80),
            vec![descriptor(
                "vision-fast",
                true,
                false,
                CapabilityLevel::None,
            )],
            vec![runtime("vision-fast", Some(12_000), Some(200))],
        );

        let selected = select_capability_peer_from_peers(
            &[fast_rtt_slow_model, slower_rtt_fast_model],
            "excluded",
            ConsultationRequestClass::Interactive,
            |candidate| candidate.capabilities.supports_vision_runtime(),
        );

        assert_eq!(
            selected.map(|(_, model)| model),
            Some("vision-fast".to_string())
        );
    }

    #[test]
    fn select_capability_peer_falls_back_to_rtt_without_perf_data() {
        let first = test_peer(
            1,
            Some(25),
            vec![descriptor("audio-a", false, true, CapabilityLevel::None)],
            vec![runtime("audio-a", None, None)],
        );
        let second = test_peer(
            2,
            Some(90),
            vec![descriptor("audio-b", false, true, CapabilityLevel::None)],
            vec![runtime("audio-b", None, None)],
        );

        let selected = select_capability_peer_from_peers(
            &[first, second],
            "excluded",
            ConsultationRequestClass::Interactive,
            |candidate| candidate.capabilities.supports_audio_runtime(),
        );

        assert_eq!(
            selected.map(|(_, model)| model),
            Some("audio-a".to_string())
        );
    }

    #[test]
    fn find_different_model_peers_prefers_reasoning_then_perf() {
        let non_reasoning = test_peer(
            1,
            Some(10),
            vec![descriptor("fast-chat", false, false, CapabilityLevel::None)],
            vec![runtime("fast-chat", Some(25_000), Some(150))],
        );
        let reasoning_slow_rtt_fast_ttft = test_peer(
            2,
            Some(120),
            vec![descriptor(
                "reasoner-a",
                false,
                false,
                CapabilityLevel::Supported,
            )],
            vec![runtime("reasoner-a", Some(10_000), Some(220))],
        );
        let reasoning_fast_rtt_slow_ttft = test_peer(
            3,
            Some(20),
            vec![descriptor(
                "reasoner-b",
                false,
                false,
                CapabilityLevel::Supported,
            )],
            vec![runtime("reasoner-b", Some(18_000), Some(800))],
        );

        let selected = find_different_model_peers_from_peers(
            &[
                non_reasoning,
                reasoning_fast_rtt_slow_ttft,
                reasoning_slow_rtt_fast_ttft,
            ],
            "current-model",
            3,
            ConsultationRequestClass::Interactive,
        );

        assert_eq!(
            selected,
            vec![
                (test_endpoint_id(2), "reasoner-a".to_string()),
                (test_endpoint_id(3), "reasoner-b".to_string()),
                (test_endpoint_id(1), "fast-chat".to_string()),
            ]
        );
    }

    #[test]
    fn find_different_model_peers_deduplicates_by_model_name() {
        let better = test_peer(
            1,
            Some(60),
            vec![descriptor(
                "shared-model",
                false,
                false,
                CapabilityLevel::Supported,
            )],
            vec![runtime("shared-model", Some(18_000), Some(180))],
        );
        let worse = test_peer(
            2,
            Some(20),
            vec![descriptor(
                "shared-model",
                false,
                false,
                CapabilityLevel::Supported,
            )],
            vec![runtime("shared-model", Some(5_000), Some(900))],
        );

        let selected = find_different_model_peers_from_peers(
            &[worse, better],
            "current-model",
            2,
            ConsultationRequestClass::Interactive,
        );

        assert_eq!(
            selected,
            vec![(test_endpoint_id(1), "shared-model".to_string())]
        );
    }

    #[test]
    fn select_capability_peer_prefers_higher_tps_for_throughput_class() {
        let fast_ttft = test_peer(
            1,
            Some(20),
            vec![descriptor(
                "vision-fast-start",
                true,
                false,
                CapabilityLevel::None,
            )],
            vec![runtime("vision-fast-start", Some(9_000), Some(150))],
        );
        let high_tps = test_peer(
            2,
            Some(40),
            vec![descriptor(
                "vision-high-tps",
                true,
                false,
                CapabilityLevel::None,
            )],
            vec![runtime("vision-high-tps", Some(24_000), Some(400))],
        );

        let selected = select_capability_peer_from_peers(
            &[fast_ttft, high_tps],
            "excluded",
            ConsultationRequestClass::Throughput,
            |candidate| candidate.capabilities.supports_vision_runtime(),
        );

        assert_eq!(
            selected.map(|(_, model)| model),
            Some("vision-high-tps".to_string())
        );
    }
}
