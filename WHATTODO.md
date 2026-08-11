# Home PC setup

Use this to get the same research engine, dashboard, and strategy progress on
another PC.

## 1. Get the code

```bash
git clone --recurse-submodules git@github.com:KristijonasSim/trading_engine.git
cd trading_engine
python3 -m venv .venv
.venv/bin/pip install numpy pandas scipy anthropic
```

Also clone `trading-bots` beside this project and make sure its market-data
folder exists.

## 2. Copy the engine state

GitHub contains the code only. Copy this folder from the current PC to the home
PC if you want the same queue, tested results, Pine sources, translations, and
trial history:

```text
/home/kris/trading_engine/state/
```

Copy it into the same path on the home PC:

```text
/home/kris/trading_engine/state/
```

Do not merge state folders from two PCs. Pick one current state folder as the
source of truth, otherwise trial counts can become wrong.

## 3. Install Claude and log in

The implementation worker needs the `claude` command:

```bash
claude
```

Complete the login in the browser.

## 4. Start dashboard and workers

```bash
cp deploy/*.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now engine-dashboard.service
systemctl --user enable --now engine-worker.service
systemctl --user enable --now engine-implementation-worker.service
```

Open the dashboard:

```text
http://127.0.0.1:8777/index.html
```

## 5. Check everything is running

```bash
systemctl --user status engine-dashboard.service
systemctl --user status engine-worker.service
systemctl --user status engine-implementation-worker.service
```

## Current focus

New TradingView harvesting is paused. The workers are currently focused on
testing the existing queue and clearing strategies that need implementation.
