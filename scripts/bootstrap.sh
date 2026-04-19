#!/usr/bin/env bash
set -euo pipefail

echo "[bootstrap] starting"

PROJECT_DIR="${PROJECT_DIR:-/workspace/project}"
R2_ENV_FILE="${R2_ENV_FILE:-/workspace/r2.env}"
R2_BUCKET="${R2_BUCKET:-parameter-golf-train}"
RCLONE_REMOTE="${RCLONE_REMOTE:-r2}"
BOOTSTRAP_PREFIX="${BOOTSTRAP_PREFIX:-bootstrap}"
DATASET_NAME="${DATASET_NAME:-fineweb10B_sp1024}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_DIR}/data/datasets/${DATASET_NAME}}"
TOKENIZERS_DIR="${TOKENIZERS_DIR:-${PROJECT_DIR}/data/tokenizers}"
TOKENIZER_FILE="${TOKENIZER_FILE:-fineweb_1024_bpe.model}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${TOKENIZERS_DIR}/${TOKENIZER_FILE}}"
EXPECTED_DATASET_SHARDS="${EXPECTED_DATASET_SHARDS:-81}"
ENV_FILE="${ENV_FILE:-${PROJECT_DIR}/.env}"
PROJECT_ENV_KEYS="${PROJECT_ENV_KEYS:-TELEGRAM_BOT_TOKEN TELEGRAM_USER_ID}"
ENV_AUTOSYNC_INTERVAL="${ENV_AUTOSYNC_INTERVAL:-300}"
RCLONE_TRANSFERS="${RCLONE_TRANSFERS:-8}"
RCLONE_CHECKERS="${RCLONE_CHECKERS:-16}"
GIT_PUSH_REMOTE="${GIT_PUSH_REMOTE:-git@github.com:yuvaraj-97/parameter-golf-stf.git}"
GIT_SSH_KEY_FILE="${GIT_SSH_KEY_FILE:-/root/.ssh/runpod_parameter_golf}"

load_env_file() {
  local path="$1"

  if [ -f "$path" ]; then
    echo "[bootstrap] loading env from ${path}"
    set -a
    # shellcheck disable=SC1090
    . "$path"
    set +a
    return 0
  fi

  return 1
}

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "error: ${name} is not set in the environment or env file" >&2
    exit 1
  fi
}

load_env_file "${R2_ENV_FILE}" || true

require_env R2_ACCOUNT_ID
require_env R2_ACCESS_KEY_ID
require_env R2_SECRET_ACCESS_KEY

r2_path() {
  printf '%s:%s/%s' "${RCLONE_REMOTE}" "${R2_BUCKET}" "$1"
}

r2_dir_exists() {
  rclone lsf "$(r2_path "$1")" >/dev/null 2>&1
}

r2_file_exists() {
  rclone lsf "$(r2_path "$(dirname "$1")")" 2>/dev/null | grep -Fxq "$(basename "$1")"
}

restore_optional_dir() {
  local source="$1"
  local target="$2"

  if r2_dir_exists "$source"; then
    echo "[bootstrap] restoring ${R2_BUCKET}/${source} -> ${target}"
    mkdir -p "$target"
    rclone copy "$(r2_path "$source")" "$target" --progress --stats 10s || true
  else
    echo "[bootstrap] skipping missing ${R2_BUCKET}/${source}"
  fi
}

setup_git_credentials() {
  mkdir -p /root/.ssh
  local ssh_key="${GIT_SSH_KEY_FILE}"

  if [ ! -f "$ssh_key" ] && [ -f /root/.ssh/id_ed25519 ]; then
    ssh_key=/root/.ssh/id_ed25519
  fi

  if [ -f "$ssh_key" ] && command -v ssh-keygen >/dev/null 2>&1; then
    chmod 600 "$ssh_key"
    ssh-keygen -y -f "$ssh_key" >"${ssh_key}.pub" 2>/dev/null || true
    chmod 644 "${ssh_key}.pub" 2>/dev/null || true
  fi

  if command -v ssh-keyscan >/dev/null 2>&1; then
    ssh-keyscan -H github.com >>/root/.ssh/known_hosts 2>/dev/null || true
    chmod 644 /root/.ssh/known_hosts 2>/dev/null || true
  fi

  cat >/root/.ssh/config <<EOF
Host github.com
  HostName github.com
  User git
  IdentityFile ${ssh_key}
  IdentitiesOnly yes
  StrictHostKeyChecking yes
  BatchMode yes
EOF
  chmod 600 /root/.ssh/config

  if [ -n "${GIT_USER_NAME:-}" ]; then
    git config --global user.name "${GIT_USER_NAME}"
  fi

  if [ -n "${GIT_USER_EMAIL:-}" ]; then
    git config --global user.email "${GIT_USER_EMAIL}"
  fi

  if [ -d "${PROJECT_DIR}/.git" ] && [ -n "${GIT_PUSH_REMOTE}" ]; then
    git -C "${PROJECT_DIR}" remote set-url --push origin "${GIT_PUSH_REMOTE}" || true
  fi

  if [ -s "$ssh_key" ]; then
    echo "[bootstrap] git SSH key ready for push: ${ssh_key}"
  else
    echo "[bootstrap] warning: no git SSH key found at ${ssh_key}; save.sh may not be able to push"
  fi
}

configure_git_fetch() {
  if [ -d "${PROJECT_DIR}/.git" ]; then
    echo "[bootstrap] configuring git fetch refs"
    git -C "${PROJECT_DIR}" config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"
    git -C "${PROJECT_DIR}" fetch origin
  fi
}

write_project_env() {
  local key

  echo "[bootstrap] writing ${ENV_FILE}"
  mkdir -p "$(dirname "${ENV_FILE}")"
  : >"${ENV_FILE}"

  for key in ${PROJECT_ENV_KEYS}; do
    if [ -z "${!key:-}" ]; then
      echo "[bootstrap] warning: ${key} is not set; writing empty value to .env"
    fi
    printf '%s=%s\n' "$key" "${!key:-}" >>"${ENV_FILE}"
  done

  chmod 600 "${ENV_FILE}" 2>/dev/null || true
}

# -----------------------------
# RCLONE CONFIG
# -----------------------------

mkdir -p /root/.config/rclone

cat > /root/.config/rclone/rclone.conf <<EOF
[${RCLONE_REMOTE}]
type = s3
provider = Cloudflare
access_key_id = ${R2_ACCESS_KEY_ID}
secret_access_key = ${R2_SECRET_ACCESS_KEY}
endpoint = https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com
acl = private
EOF
chmod 600 /root/.config/rclone/rclone.conf

# -----------------------------
# RESTORE CONFIG
# -----------------------------

echo "[bootstrap] restoring configs"

mkdir -p /root/.ssh
restore_optional_dir "${BOOTSTRAP_PREFIX}/git" /root
restore_optional_dir "${BOOTSTRAP_PREFIX}/ssh" /root/.ssh
restore_optional_dir "${BOOTSTRAP_PREFIX}/config" /root/.config
restore_optional_dir "${BOOTSTRAP_PREFIX}/home" /root
restore_optional_dir "${BOOTSTRAP_PREFIX}/root" /root
setup_git_credentials
configure_git_fetch
write_project_env

chmod 700 /root/.ssh || true
chmod 600 /root/.ssh/* 2>/dev/null || true
chmod 600 /root/.gitconfig 2>/dev/null || true
chmod 600 /root/.git-credentials 2>/dev/null || true

# -----------------------------
# DATASET
# -----------------------------

echo "[bootstrap] restoring dataset to ${DATASET_DIR}"

mkdir -p "${DATASET_DIR}"
rclone sync "$(r2_path "datasets/${DATASET_NAME}")" "${DATASET_DIR}" \
  --progress \
  --stats 10s \
  --transfers "${RCLONE_TRANSFERS}" \
  --checkers "${RCLONE_CHECKERS}"

dataset_shards="$(find "${DATASET_DIR}" -type f -name '*.bin' | wc -l | tr -d ' ')"
echo "[bootstrap] dataset shards present: ${dataset_shards}/${EXPECTED_DATASET_SHARDS}"

if [ "${EXPECTED_DATASET_SHARDS}" != "0" ] && [ "${dataset_shards}" -lt "${EXPECTED_DATASET_SHARDS}" ]; then
  echo "error: dataset restore is incomplete" >&2
  exit 1
fi

echo "[bootstrap] restoring tokenizers to ${TOKENIZERS_DIR}"

mkdir -p "${TOKENIZERS_DIR}"
if r2_dir_exists "tokenizers"; then
  rclone sync "$(r2_path "tokenizers")" "${TOKENIZERS_DIR}" \
    --progress \
    --stats 10s \
    --transfers "${RCLONE_TRANSFERS}" \
    --checkers "${RCLONE_CHECKERS}"
else
  echo "[bootstrap] skipping missing ${R2_BUCKET}/tokenizers"
fi

if [ ! -f "${TOKENIZER_PATH}" ]; then
  echo "error: tokenizer restore is incomplete; missing ${TOKENIZER_PATH}" >&2
  echo "hint: run python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1 or upload tokenizers/ to R2" >&2
  exit 1
fi

echo "[bootstrap] tokenizer present: ${TOKENIZER_PATH}"

# -----------------------------
# ENV AUTOSYNC
# -----------------------------

echo "[bootstrap] starting .env autosync"

mkdir -p /workspace/logs
cat >/usr/local/bin/pg-env-autosync.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail

while true; do
  if [ -f "${ENV_FILE}" ]; then
    rclone copyto "${ENV_FILE}" "$(r2_path "${BOOTSTRAP_PREFIX}/project/.env")" || true
    chmod 600 "${ENV_FILE}" 2>/dev/null || true
  fi
  sleep "${ENV_AUTOSYNC_INTERVAL}"
done
EOF

chmod +x /usr/local/bin/pg-env-autosync.sh
nohup /usr/local/bin/pg-env-autosync.sh >/workspace/logs/env-autosync.log 2>&1 &

echo "[bootstrap] done"
