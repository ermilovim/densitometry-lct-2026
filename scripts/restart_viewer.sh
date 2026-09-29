#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

log() {
  printf '[densitometry] %s\n' "$*"
}

fail() {
  printf '[densitometry] ERROR: %s\n' "$*" >&2
  exit 1
}

[[ "$(uname -s)" == "Linux" ]] || fail "Deployment is supported on Linux only."
[[ "$(uname -m)" == "x86_64" ]] || fail "Deployment requires an x86_64 host."

for command_name in docker curl; do
  command -v "$command_name" >/dev/null 2>&1 || fail "Command '$command_name' is not installed. See DEPLOYMENT.md."
done

docker info >/dev/null 2>&1 || fail "Docker daemon is unavailable. Start Docker and make sure the current user can access it."

image_name="${IMAGE_NAME:-densitometry-qc:latest}"
container_name="${CONTAINER_NAME:-densitometry-qc}"
viewer_port="${VIEWER_PORT:-8765}"
viewer_bind="${VIEWER_BIND:-127.0.0.1}"
gpu_mode="${GPU_MODE:-auto}"
jobs_dir="${VIEWER_JOBS_DIR:-$repo_root/artifacts/viewer_jobs}"
checkpoint_dir="${DXA3D_MODEL_DIR:-$repo_root/artifacts/dxa3d}"
checkpoint="$checkpoint_dir/best_model.pt"
checkpoint_url="https://www.dropbox.com/scl/fi/be4dg1xccgl1fo9wn74i8/epoch-996-loss_valid-points-best_loss-0.0168.pt?rlkey=ytnrrctofyebqtkj5p4554px1&dl=1"
checkpoint_sha256="9e86fb66b73ac27bfa60aa4695bb037d06acab8882e192f793d577711120e2c3"

[[ "$viewer_port" =~ ^[0-9]+$ ]] && ((viewer_port >= 1 && viewer_port <= 65535)) \
  || fail "VIEWER_PORT must be an integer from 1 to 65535."
case "$gpu_mode" in
  auto | require | off) ;;
  *) fail "GPU_MODE must be auto, require, or off." ;;
esac

checksum() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    fail "Neither sha256sum nor shasum is installed."
  fi
}

mkdir -p "$jobs_dir"
mkdir -p "$checkpoint_dir"
if [[ -f "$checkpoint" && "$(checksum "$checkpoint")" != "$checkpoint_sha256" ]]; then
  log "Existing DXA-to-3D checkpoint has an invalid checksum; downloading it again."
  mv "$checkpoint" "$checkpoint.invalid.$(date +%s)"
fi
if [[ ! -f "$checkpoint" ]]; then
  log "Downloading the official DXA-to-3D checkpoint..."
  rm -f "$checkpoint.part"
  curl --fail --location --show-error "$checkpoint_url" --output "$checkpoint.part"
  [[ "$(checksum "$checkpoint.part")" == "$checkpoint_sha256" ]] \
    || fail "Downloaded DXA-to-3D checkpoint has an invalid checksum."
  mv "$checkpoint.part" "$checkpoint"
fi

available_kb="$(df -Pk "$repo_root" | awk 'NR == 2 {print $4}')"
if [[ "$available_kb" =~ ^[0-9]+$ ]] && ((available_kb < 8 * 1024 * 1024)); then
  fail "Less than 8 GiB of free disk space is available for the Docker build."
fi

log "Building Docker image $image_name..."
docker build -t "$image_name" .

gpu_args=()
if [[ "$gpu_mode" != "off" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 \
    && nvidia-smi >/dev/null 2>&1 \
    && docker run --rm --gpus all "$image_name" \
      .venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))' >/dev/null 2>&1; then
    gpu_args=(--gpus all)
    log "NVIDIA GPU is available to Docker."
  elif [[ "$gpu_mode" == "require" ]]; then
    fail "GPU_MODE=require, but Docker cannot access an NVIDIA GPU. Check the driver and NVIDIA Container Toolkit."
  else
    log "WARNING: NVIDIA GPU is unavailable to Docker; starting in slower CPU mode."
  fi
fi

backup_container=""
if docker container inspect "$container_name" >/dev/null 2>&1; then
  backup_container="${container_name}-rollback-$$"
  log "Stopping the previous container..."
  docker rename "$container_name" "$backup_container"
  docker stop "$backup_container" >/dev/null
fi

restore_previous() {
  docker rm -f "$container_name" >/dev/null 2>&1 || true
  if [[ -n "$backup_container" ]]; then
    log "Restoring the previous container after a failed deployment..."
    docker rename "$backup_container" "$container_name" >/dev/null 2>&1 || true
    docker start "$container_name" >/dev/null 2>&1 || true
  fi
}

log "Starting container $container_name on $viewer_bind:$viewer_port..."
if ! docker run -d \
  --name "$container_name" \
  --restart unless-stopped \
  "${gpu_args[@]}" \
  -p "$viewer_bind:$viewer_port:8765" \
  -v "$jobs_dir:/app/artifacts/viewer_jobs" \
  -v "$checkpoint:/models/dxa3d.pt:ro" \
  -e DXA3D_CHECKPOINT=/models/dxa3d.pt \
  "$image_name" >/dev/null; then
  restore_previous
  fail "Container failed to start. The port may already be occupied."
fi

healthy=false
for _ in $(seq 1 60); do
  if docker exec "$container_name" .venv/bin/python -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/health', timeout=2).read()" \
    >/dev/null 2>&1; then
    healthy=true
    break
  fi
  sleep 1
done

if [[ "$healthy" != "true" ]]; then
  docker logs --tail 100 "$container_name" >&2 || true
  restore_previous
  fail "Health check failed. The previous container was restored when it existed."
fi

if [[ -n "$backup_container" ]]; then
  docker rm "$backup_container" >/dev/null
fi

docker ps --filter "name=^/${container_name}$"
log "Deployment completed. Health check: OK."
if [[ "$viewer_bind" == "0.0.0.0" ]]; then
  server_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  log "Open http://${server_ip:-<server-ip>}:$viewer_port"
else
  log "Open http://$viewer_bind:$viewer_port"
fi
