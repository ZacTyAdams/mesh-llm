#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$ROOT_DIR/target/release/mesh-llm"

# Optional overrides:
#   QUICKTEST_JOIN_TOKEN      Explicit join token (highest priority)
#   QUICKTEST_TOKEN_FILE      Cached token path (default: ~/.mesh-llm/quicktest-worker.token)
#   QUICKTEST_FORCE_DISCOVERY 1/0, default 1 (ignore token cache and always test LAN discovery)
#   QUICKTEST_DISCOVER_WAIT   LAN discover window in seconds (default: 6)
#   QUICKTEST_RESTART_DELAY   Delay between restarts in seconds (default: 3)
#   QUICKTEST_OFFLINE         1/0, default 1
#   QUICKTEST_HEADLESS        1/0, default 1
#   QUICKTEST_LISTEN_ALL      1/0, default 1
#   QUICKTEST_LOG_FORMAT      pretty/json, default pretty
#   QUICKTEST_MODEL           Model ref to serve
#   QUICKTEST_INTERACTIVE     1/0, default 0 (enable TUI/interactive terminal mode)
TOKEN_FILE="${QUICKTEST_TOKEN_FILE:-$HOME/.mesh-llm/quicktest-worker.token}"
FORCE_DISCOVERY="${QUICKTEST_FORCE_DISCOVERY:-1}"
DISCOVER_WAIT="${QUICKTEST_DISCOVER_WAIT:-6}"
RESTART_DELAY="${QUICKTEST_RESTART_DELAY:-3}"
OFFLINE="${QUICKTEST_OFFLINE:-1}"
HEADLESS="${QUICKTEST_HEADLESS:-1}"
LISTEN_ALL="${QUICKTEST_LISTEN_ALL:-1}"
LOG_FORMAT="${QUICKTEST_LOG_FORMAT:-pretty}"
MODEL_REF="${QUICKTEST_MODEL:-unsloth/gemma-4-31B-it-GGUF:UD-IQ2_XXS}"
INTERACTIVE="${QUICKTEST_INTERACTIVE:-0}"

stop_requested=0
runtime_pid=""
DISCOVERY_LAST_MATCH=""

on_signal() {
	stop_requested=1
	if [[ -n "$runtime_pid" ]]; then
		kill -TERM "$runtime_pid" 2>/dev/null || true
	fi
}

trap on_signal INT TERM

if [[ ! -x "$BIN" ]]; then
	echo "mesh-llm binary not found at: $BIN" >&2
	echo "Build first with: just build" >&2
	exit 1
fi

mkdir -p "$(dirname "$TOKEN_FILE")"

discover_lan_token() {
	local out token
	out="$($BIN discover --lan --auto --wait-secs "$DISCOVER_WAIT" 2>&1 || true)"
	DISCOVERY_LAST_MATCH="$(printf '%s\n' "$out" | sed -n 's/^Selected LAN match:[[:space:]]*//p' | tail -n1)"

	token="$(printf '%s\n' "$out" | awk '/^eyJ/{t=$0} END{print t}')"

	if [[ -z "$token" ]]; then
		token="$(printf '%s\n' "$out" | sed -n 's/^[[:space:]]*token:[[:space:]]*//p' | head -n1)"
	fi
	if [[ -z "$token" ]]; then
		token="$(printf '%s\n' "$out" | awk '{for (i = 1; i <= NF; i++) { if ($i ~ /^eyJ/) { print $i; exit } }}')"
	fi

	# Ignore shortened preview tokens from non-auto discovery output.
	if [[ "$token" == *"..."* ]]; then
		token=""
	fi

	printf '%s' "$token"
}

token_fingerprint() {
	local token="$1"
	local len="${#token}"
	if (( len <= 18 )); then
		printf '%s' "$token"
		return
	fi
	printf '%s...%s' "${token:0:8}" "${token:len-6:6}"
}

print_discovery_check() {
	local source="$1"
	local token="$2"
	local mesh="$3"
	local fp
	fp="$(token_fingerprint "$token")"
	if [[ -n "$mesh" ]]; then
		echo "[quicktest-gemma] discovery check: source=$source mesh=\"$mesh\" token_fp=$fp"
	else
		echo "[quicktest-gemma] discovery check: source=$source mesh=unknown token_fp=$fp"
	fi
}

load_token() {
	if [[ -n "${QUICKTEST_JOIN_TOKEN:-}" ]]; then
		printf '%s' "$QUICKTEST_JOIN_TOKEN"
		return
	fi

	if [[ "$FORCE_DISCOVERY" == "1" ]]; then
		printf ''
		return
	fi

	if [[ -s "$TOKEN_FILE" ]]; then
		tr -d '[:space:]' < "$TOKEN_FILE"
		return
	fi

	printf ''
}

save_token() {
	local token="$1"
	printf '%s\n' "$token" > "$TOKEN_FILE"
}

run_worker() {
	local token="$1"
	local args=()

	args+=(serve --model "$MODEL_REF" --join "$token" --log-format "$LOG_FORMAT")
	if [[ "$OFFLINE" == "1" ]]; then
		args+=(--offline)
	fi
	if [[ "$HEADLESS" == "1" ]]; then
		args+=(--headless)
	fi
	if [[ "$LISTEN_ALL" == "1" ]]; then
		args+=(--listen-all)
	fi

	echo "[quicktest-gemma] starting worker model: $MODEL_REF"
	echo "[quicktest-gemma] token source: ${TOKEN_FILE}"
	if [[ "$INTERACTIVE" == "1" && -r /dev/tty ]]; then
		"$BIN" "${args[@]}" </dev/tty
	else
		"$BIN" "${args[@]}" </dev/null
	fi
	return $?
}

echo "[quicktest-gemma] binary: $BIN"
echo "[quicktest-gemma] model: $MODEL_REF"
echo "[quicktest-gemma] offline: $OFFLINE, headless: $HEADLESS, listen-all: $LISTEN_ALL"
echo "[quicktest-gemma] force discovery: $FORCE_DISCOVERY"
echo "[quicktest-gemma] token file: $TOKEN_FILE"
echo "[quicktest-gemma] interactive: $INTERACTIVE"

while [[ "$stop_requested" -eq 0 ]]; do
	token="$(load_token)"
	token_source="cached"
	mesh_label=""

	if [[ -n "${QUICKTEST_JOIN_TOKEN:-}" ]]; then
		token_source="env"
	elif [[ "$FORCE_DISCOVERY" == "1" ]]; then
		token_source="lan-discovery"
	elif [[ -z "$token" ]]; then
		token_source="lan-discovery"
	fi

	if [[ -z "$token" ]]; then
		echo "[quicktest-gemma] no cached token, discovering LAN mesh..."
		token="$(discover_lan_token)"
		mesh_label="$DISCOVERY_LAST_MATCH"
		if [[ -n "$token" ]]; then
			save_token "$token"
			echo "[quicktest-gemma] discovered and cached token"
		else
			echo "[quicktest-gemma] no LAN token found, retrying in ${RESTART_DELAY}s"
			sleep "$RESTART_DELAY"
			continue
		fi
	fi

	print_discovery_check "$token_source" "$token" "$mesh_label"

	set +e
	run_worker "$token"
	exit_code=$?
	set -e

	if [[ "$stop_requested" -eq 1 ]]; then
		break
	fi

	echo "[quicktest-gemma] mesh-llm exited with code $exit_code"

	# If a join token is stale/unreachable, force rediscovery on next loop.
	if [[ -z "${QUICKTEST_JOIN_TOKEN:-}" ]]; then
		: > "$TOKEN_FILE"
		echo "[quicktest-gemma] cleared cached token to force LAN rediscovery"
	fi

	echo "[quicktest-gemma] restarting in ${RESTART_DELAY}s"
	sleep "$RESTART_DELAY"
done

echo "[quicktest-gemma] shutdown complete"