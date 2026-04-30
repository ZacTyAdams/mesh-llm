#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$ROOT_DIR/target/release/mesh-llm"

# Optional overrides:
#   QUICKTEST_JOIN_TOKEN      Explicit join token (highest priority)
#   QUICKTEST_TOKEN_FILE      Cached token path (default: ~/.mesh-llm/quicktest-client.token)
#   QUICKTEST_FORCE_DISCOVERY 1/0, default 1 (ignore token cache and always test LAN discovery)
#   QUICKTEST_DISCOVER_WAIT   LAN discover window in seconds (default: 6)
#   QUICKTEST_RESTART_DELAY   Delay between restarts in seconds (default: 3)
#   QUICKTEST_BOOTSTRAP_IF_NONE 1/0, default 1 (start private mesh when no token is found)
#   QUICKTEST_OFFLINE         1/0, default 1
#   QUICKTEST_HEADLESS        1/0, default 0
#   QUICKTEST_LOG_FORMAT      pretty/json, default pretty
TOKEN_FILE="${QUICKTEST_TOKEN_FILE:-$HOME/.mesh-llm/quicktest-client.token}"
FORCE_DISCOVERY="${QUICKTEST_FORCE_DISCOVERY:-1}"
DISCOVER_WAIT="${QUICKTEST_DISCOVER_WAIT:-6}"
RESTART_DELAY="${QUICKTEST_RESTART_DELAY:-3}"
BOOTSTRAP_IF_NONE="${QUICKTEST_BOOTSTRAP_IF_NONE:-1}"
OFFLINE="${QUICKTEST_OFFLINE:-1}"
HEADLESS="${QUICKTEST_HEADLESS:-0}"
LOG_FORMAT="${QUICKTEST_LOG_FORMAT:-pretty}"

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
		echo "[quicktest-client] discovery check: source=$source mesh=\"$mesh\" token_fp=$fp"
	else
		echo "[quicktest-client] discovery check: source=$source mesh=unknown token_fp=$fp"
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

run_client() {
	local token="$1"
	local args=()

	args+=(serve --client --listen-all --log-format "$LOG_FORMAT")
	if [[ "$OFFLINE" == "1" ]]; then
		args+=(--offline)
	fi
	if [[ "$HEADLESS" == "1" ]]; then
		args+=(--headless)
	fi
	if [[ -n "$token" ]]; then
		args+=(--join "$token")
	fi

	echo "[quicktest-client] starting gateway client"
	if [[ -n "$token" ]]; then
		echo "[quicktest-client] token source: ${TOKEN_FILE}"
	else
		echo "[quicktest-client] no token found, bootstrapping a private mesh"
	fi
	if [[ -r /dev/tty ]]; then
		"$BIN" "${args[@]}" </dev/tty &
	else
		"$BIN" "${args[@]}" &
	fi
	runtime_pid=$!
	wait "$runtime_pid"
	local code=$?
	runtime_pid=""
	return "$code"
}

echo "[quicktest-client] binary: $BIN"
echo "[quicktest-client] offline: $OFFLINE, headless: $HEADLESS, log-format: $LOG_FORMAT"
echo "[quicktest-client] force discovery: $FORCE_DISCOVERY"
echo "[quicktest-client] token file: $TOKEN_FILE"

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
		echo "[quicktest-client] no cached token, discovering LAN mesh..."
		token="$(discover_lan_token)"
		mesh_label="$DISCOVERY_LAST_MATCH"
		if [[ -n "$token" ]]; then
			save_token "$token"
			echo "[quicktest-client] discovered and cached token"
		elif [[ "$BOOTSTRAP_IF_NONE" != "1" ]]; then
			echo "[quicktest-client] no LAN token found, retrying in ${RESTART_DELAY}s"
			sleep "$RESTART_DELAY"
			continue
		else
			echo "[quicktest-client] no LAN token found, bootstrapping local private mesh"
		fi
	fi

	if [[ -n "$token" ]]; then
		print_discovery_check "$token_source" "$token" "$mesh_label"
	fi

	set +e
	run_client "$token"
	exit_code=$?
	set -e

	if [[ "$stop_requested" -eq 1 ]]; then
		break
	fi

	echo "[quicktest-client] mesh-llm exited with code $exit_code"

	# If a join token is stale/unreachable, force rediscovery on next loop.
	if [[ -z "${QUICKTEST_JOIN_TOKEN:-}" ]]; then
		: > "$TOKEN_FILE"
		echo "[quicktest-client] cleared cached token to force LAN rediscovery"
	fi

	echo "[quicktest-client] restarting in ${RESTART_DELAY}s"
	sleep "$RESTART_DELAY"
done

echo "[quicktest-client] shutdown complete"
