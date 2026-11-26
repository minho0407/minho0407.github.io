#!/usr/bin/env bash
# 우분투 서버(22.04/24.04)에서 한 번 실행:  bash install.sh
set -euo pipefail
REPO="https://github.com/minho0407/minho0407.github.io.git"
DIR="$HOME/daytrade"
APP="$DIR/daytrade-ai"

sudo apt-get update -y
sudo apt-get install -y python3-venv git
[ -d "$DIR/.git" ] || git clone --depth 1 "$REPO" "$DIR"
cd "$APP"
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements-realtime.txt

if [ ! -f .env ]; then
  cp deploy/env.example .env
  chmod 600 .env
fi
sed "s#__DIR__#$APP#g; s#__USER__#$USER#g" deploy/daytrade-realtime.service \
  | sudo tee /etc/systemd/system/daytrade-realtime.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable daytrade-realtime

cat <<MSG

설치 완료. 다음 순서로 진행하세요.
  1) 키 입력:      nano $APP/.env
  2) 연결 점검:    cd $APP && .venv/bin/python realtime.py --check
  3) 감시 시작:    sudo systemctl start daytrade-realtime
  4) 로그 보기:    journalctl -u daytrade-realtime -f
  코드 업데이트:   cd $DIR && git pull && sudo systemctl restart daytrade-realtime
MSG
