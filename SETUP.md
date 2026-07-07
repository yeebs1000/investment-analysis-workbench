# Setup Guide — Investment Analysis Workbench

This guide assumes **no technical background**. Follow it top to bottom and you'll
have the workbench running on your own computer, reading your own brokerage account.

> **What this app is:** a private dashboard that reads your brokerage holdings and
> market data and turns them into plain-English analysis. It runs entirely on *your*
> computer. It is **read-only** — it can never place, change, or cancel a trade.
>
> **What it is not:** financial advice. It's a decision-support tool. Every decision
> and every trade is yours to make and execute yourself.

> **Windows shortcut:** after installing Python and Node (steps 1–2) and getting the
> code (step 3), double-click **`setup.bat`** to do the rest of the install
> automatically, and **`start.bat`** to launch the gateways + app every time after.
> The manual steps below remain for reference and for macOS/Linux.

---

## 0. What you'll need

**Accounts / software (free):**
- A **Moomoo** brokerage account (this is the main data source). *Or* Interactive
  Brokers / Tiger — see the optional sections at the end.
- About **30 minutes** the first time.

**You'll install three things** (all free, walked through below):
1. **Python** — runs the analysis engine.
2. **Node.js** — runs the dashboard you look at.
3. **Moomoo OpenD** — the small program that securely connects the app to your
   Moomoo account.

**Optional API keys** (all have a free tier; the app works without them):
- **Finnhub** — company fundamentals, analyst ratings, earnings, insider data.
- **FRED** — macro "market weather" (interest rates, credit, volatility).
- **Google Gemini** and/or **Anthropic Claude** — the AI that writes the plain-English briefs.

---

## 1. Install Python

1. Go to **https://www.python.org/downloads/** and download **Python 3.11** (3.11.x).
2. Run the installer. **On the first screen, tick the box "Add python.exe to PATH"** —
   this matters. Then click **Install Now**.
3. To confirm it worked: open a terminal and type `python --version`.
   - **Windows:** press the Start button, type `powershell`, hit Enter, then type the command.
   - You should see something like `Python 3.11.9`.

## 2. Install Node.js

1. Go to **https://nodejs.org/** and download the **LTS** version.
2. Run the installer and accept all the defaults.
3. Confirm: in a terminal type `node --version` — you should see something like `v20.x`.

## 3. Get the code

If you were given a link to the project on GitHub:
1. Install **Git** from **https://git-scm.com/downloads** (accept the defaults), then in a terminal:
   ```
   git clone <the-project-URL>
   cd "Technical Optimiser"
   ```
If you were given a **ZIP file** instead: unzip it somewhere easy to find (e.g. your
Desktop), and open a terminal in that folder.

> **Tip (Windows):** to open a terminal *inside* a folder, open the folder in File
> Explorer, click the address bar, type `powershell`, and press Enter.

## 4. Set up Moomoo OpenD (the data connection)

**OpenD** is a small official Moomoo program that lets the app read your account. It
must be **running and logged in** whenever you use the workbench.

1. Download **OpenD** from Moomoo's developer site:
   **https://www.moomoo.com/download/OpenAPI** (choose your operating system).
2. Install and open it. Log in with your **Moomoo account** username and password.
3. Leave it running. By default it listens on `127.0.0.1` port `11111` — the app
   already expects that, so you don't need to change anything.

> You only need to interact with OpenD once per session: open it, log in, minimize it.

## 5. Set up the backend (the analysis engine)

In a terminal, from the project folder:

```powershell
cd backend

# Create an isolated Python environment (keeps this project's packages separate)
python -m venv .venv

# "Activate" it:
#   Windows PowerShell:
.\.venv\Scripts\Activate.ps1
#   Mac/Linux:
#   source .venv/bin/activate

# Install everything the backend needs (takes a few minutes the first time)
pip install -r requirements.txt
```

> **Windows note:** if activating gives a "running scripts is disabled" error, run this
> once, then try activating again:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## 6. Add your settings and keys

1. In the `backend` folder there's a file called **`.env.example`**. Make a copy of it
   named **`.env`** (just `.env`, no `.example`).
   - Windows terminal: `copy .env.example .env`
   - Mac/Linux: `cp .env.example .env`
2. Open `.env` in any text editor (Notepad is fine). It's full of settings with
   comments explaining each one.
3. **Minimum to get started:** if your Moomoo account is not a US account, set
   `SECURITY_FIRM` to match your region (e.g. `FUTUSG` for Singapore, `FUTUSECURITIES`
   for Hong Kong). Everything else can stay as-is.
4. **Optional free keys** — paste them in if you have them (each line in `.env` says
   where to get the key):
   - `FINNHUB_API_KEY=` — from https://finnhub.io/register
   - `FRED_API_KEY=` — from https://fred.stlouisfed.org/docs/api/api_key.html
   - `GEMINI_API_KEY=` — from https://aistudio.google.com/apikey
   - `ANTHROPIC_API_KEY=` — from https://console.anthropic.com/ (optional)

> Your `.env` file stays on your computer and is never shared or uploaded — it's
> deliberately excluded from the project's version control.

## 7. Set up the frontend (the dashboard)

Open a **second** terminal in the project folder:

```powershell
cd frontend
npm install          # downloads the dashboard's building blocks (first time only)
```

## 8. Run it

You need **two terminals** running at the same time (plus OpenD in the background).

**Terminal 1 — backend:**
```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```
Wait until it says something like `Application startup complete`.

**Terminal 2 — frontend:**
```powershell
cd frontend
npm run dev
```

Now open your web browser to **http://localhost:5173** — that's the workbench.

> **Every time after this**, you just: (1) open OpenD and log in, (2) run the two
> commands above in two terminals, (3) open the browser.

---

## Using the workbench

- **Portfolio** — your holdings, each with a BUY / ACCUMULATE / HOLD / REDUCE / SELL
  badge. Click any row for the full breakdown. A banner at the top tracks how you're
  doing **vs the S&P 500** over time.
- **Optimiser** — a suggested rebalance plan. Pick a "Max per name" cap to protect
  deliberate long-term core positions.
- **Watchlists** — ranks every symbol in any Moomoo watchlist by signal strength.
- **Symbol lookup** — analyze any ticker (`US.AAPL`, `HK.00700`, `SG.D05`): price chart,
  signals, an **options strategist**, and an **Ask** box where you can ask questions in
  plain English ("I have 900 shares, what should I do for income?").
- **Explain with:** switch (top of the page) — turns on AI-written briefs. "Deterministic"
  is free and needs no key; "Gemini"/"Claude" light up once you've added their key.

---

## Optional extras

### View it on your phone (same home Wi-Fi)
The dashboard already works on a phone browser. With the app running on your computer,
find your computer's local IP address and open `http://THAT-IP:5173` on your phone
(both devices on the same Wi-Fi). Your computer must stay on, with OpenD running.
- Find your IP — Windows: run `ipconfig` and look for "IPv4 Address" (e.g. `192.168.1.42`).
- For access away from home, install **Tailscale** (free) on both devices — do **not**
  expose the app directly to the internet; it's connected to your real accounts.

> ℹ️ Every subsystem (charts/scores, options strategist, watchlists) has an
> automatic Moomoo → IBKR → Tiger fallback chain, so any single broker works
> standalone. The Tiger fallback is code-complete but not live-tested against
> a real Tiger account — see the [Broker compatibility table](README.md#%EF%B8%8F-broker-compatibility--read-before-picking-your-setup)
> in the README for the exact breakdown, especially if something looks off on
> a Tiger-only setup.

### Add Interactive Brokers (IBKR)
In `.env` set `IBKR_ENABLED=true` and make sure **IB Gateway** or **TWS** is running
with the API enabled (Configure → Settings → API → *Enable ActiveX and Socket Clients*).
Your IBKR holdings get merged into the same combined portfolio, and IBKR also serves
as a full market-data fallback (charts/scores, and the options strategist with live
Greeks) when Moomoo is unavailable. The options chain needs your IBKR account to have
an options market-data subscription (live or delayed) — without one it still computes
Greeks/IV but can't show live bid/ask/last, the same honest-degrade pattern as the
Level-2 depth chip.

### Add Tiger Brokers
1. `pip install tigeropen` (inside the activated backend environment).
2. In your Tiger app, open the **Open API** page and note your **Tiger ID** and
   **account number**, and download your **RSA private key** file.
3. In `.env` set: `TIGER_ENABLED=true`, `TIGER_ID=...`, `TIGER_ACCOUNT=...`,
   `TIGER_PRIVATE_KEY_PATH=` (the full path to your key file).
4. Tiger now also serves as a market-data fallback (bars, options) when Moomoo/IBKR
   are unavailable, so a Tiger-only setup gets real technical scores/charts. This
   path is new and has not been live-tested against a real Tiger account — if
   numbers look wrong (especially options IV), that's the first thing to check.

### Train the ML signal (advanced, optional)
An optional machine-learning signal can be trained offline. It only activates if it
passes strict purged-walk-forward validation, and never changes the core scoring
unless a human approves it.
```powershell
cd backend
.\.venv\Scripts\python.exe -m app.ml.train --tf day --horizons 10,20 --folds 6
```

**Getting a sharper edge.** The model is only as good as (a) what it learns from and
(b) what target it predicts. Two levers, both judged by the walk-forward itself — not
asserted:
- **Broader universe = the biggest real gain.** Training pulls its universe from *your
  own* positions + watchlists, so it only ever validates names you already chose
  (survivorship bias — the report says so). Add more symbols to a watchlist before
  training to widen it toward "what to buy next," not just "was my judgment supported."
- **Try the vol-adjusted target.** `--label-mode vol_adjusted` ranks names by
  risk-adjusted forward return instead of raw return, so the signal rewards genuine
  relative strength over raw amplitude. Run both and compare the OOS AUC/IC in the two
  reports — keep whichever actually validates sharper:
  ```powershell
  .\.venv\Scripts\python.exe -m app.ml.train --tf day --horizons 10,20 --label-mode median
  .\.venv\Scripts\python.exe -m app.ml.train --tf day --horizons 10,20 --label-mode vol_adjusted
  ```
The feature set also gained 3- and 6-month momentum (the classic momentum band), so a
retrain picks those up automatically. A model trained before this update keeps working
until you retrain (inference uses each model's own saved feature list).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Dashboard loads but shows errors / no data | Is **OpenD** open and logged in? That's the #1 cause. |
| `python` not recognized | Reinstall Python and tick **"Add to PATH"** (Step 1). |
| "running scripts is disabled" on Windows | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then retry. |
| Port 8010 or 5173 "already in use" | Another copy is running — close the other terminal, or restart your computer. If it's a *different* project (not this one) squatting the port, close that instead. |
| Fundamentals/macro/AI panels are blank | Those need their (free) API keys in `.env` — see Step 6. They're optional. |
| Non-US Moomoo account shows nothing | Set `SECURITY_FIRM` in `.env` to your region (e.g. `FUTUSG`). |

---

## A note on safety

- The app is **read-only** by design and cannot trade. It only *reads* your account.
- Your API keys and broker private key live in `backend/.env` (and your key file),
  which stay on your machine and are never committed or uploaded.
- Never expose the app to the public internet. It's built to run locally, on your Wi-Fi,
  or over a private VPN like Tailscale — nothing more.
