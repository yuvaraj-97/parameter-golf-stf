#!/usr/bin/env bash
set -euo pipefail

echo "[bootstrap] starting"

export R2_BUCKET="${R2_BUCKET:-parameter-golf-train}"

# -----------------------------

# RCLONE CONFIG

# -----------------------------

mkdir -p /root/.config/rclone

cat > /root/.config/rclone/rclone.conf <<EOF
[r2]
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
rclone sync r2:${R2_BUCKET}/bootstrap/ssh /root/.ssh || true
rm -rf /root/.gitconfig
rclone copyto r2:${R2_BUCKET}/bootstrap/git/.gitconfig /root/.gitconfig || true

chmod 700 /root/.ssh || true
chmod 600 /root/.ssh/* 2>/dev/null || true

# -----------------------------

# CLONE WITH SPARSE CHECKOUT

# -----------------------------

echo "[bootstrap] cloning repo"

rm -rf /workspace/project

git clone --filter=blob:none --no-checkout [git@github.com](mailto:git@github.com):yuvaraj-97/parameter-golf-stf.git /workspace/project

git -C /workspace/project sparse-checkout init --no-cone

cat > /workspace/project/.git/info/sparse-checkout <<'EOF'
/*
!/*.pt
!/*.ptz
!/telemetry.csv
!/train.log
EOF

git -C /workspace/project checkout HEAD

# -----------------------------

# RESTORE PROJECT FILES

# -----------------------------

echo "[bootstrap] restoring env + scripts"

rclone copyto r2:${R2_BUCKET}/bootstrap/project/.env /workspace/project/.env || true
rclone copyto r2:${R2_BUCKET}/bootstrap/scripts/post_launch_check.sh /workspace/post_launch_check.sh || true

chmod 600 /workspace/project/.env 2>/dev/null || true
chmod +x /workspace/post_launch_check.sh 2>/dev/null || true

# -----------------------------

# DATASET

# -----------------------------

echo "[bootstrap] restoring dataset"

mkdir -p /workspace/project/data/datasets/fineweb10B_sp1024

rclone sync 
r2:${R2_BUCKET}/datasets/fineweb10B_sp1024 
/workspace/project/data/datasets/fineweb10B_sp1024 
--transfers 8 --checkers 16 || true

# -----------------------------

# HF CACHE

# -----------------------------

mkdir -p /workspace/hf/hub

export HF_HOME=/workspace/hf
export HUGGINGFACE_HUB_CACHE=/workspace/hf/hub

# -----------------------------

# AUTOSYNC

# -----------------------------

echo "[bootstrap] starting autosync"

cat >/usr/local/bin/pg-autosync.sh <<EOF
#!/usr/bin/env bash
while true; do
rclone sync /workspace/project/data/datasets/fineweb10B_sp1024 r2:${R2_BUCKET}/datasets/fineweb10B_sp1024 || true
sleep 300
done
EOF

chmod +x /usr/local/bin/pg-autosync.sh
nohup /usr/local/bin/pg-autosync.sh >/workspace/logs/autosync.log 2>&1 &

echo "[bootstrap] done"
