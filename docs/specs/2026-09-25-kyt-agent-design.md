# KYT Agent 設計文件：鏈上地址風險調查（簡易版）

- 日期：2026-09-25
- 狀態：設計已確認，待實作

## 1. 目標

輸入一個 Ethereum 主網地址，agent 自主決定往上下游追幾層、查哪些交易，產出附證據的風險報告，交由合規人員在 CLI 核准、駁回或要求補查。

重點：tool calling、自主探索的步數與成本控制、HITL 迴圈、eval（召回率、誤報率）。

### 範圍

- 包含：ETH 主網、Etherscan V2 API、本地標籤庫、CLI 審核介面、可恢復的案件、audit log、錄製重播式 eval
- 不包含：多鏈與跨鏈追蹤、Web 介面、付費標籤 API、任何寫入鏈上的操作

### 安全原則

- 主網只讀，專案內不存在任何私鑰或簽名邏輯
- 硬性規則（黑名單比對、風險下限、預算上限）由程式判斷，不交給 LLM
- 所有判斷留下可稽核紀錄

## 2. 架構

採「手寫 ReAct 迴圈 + 外層流程圖」，以 LangGraph 實作：

```
screen ──(命中制裁)──────────────────────────→ report
   │                                             ↑
   └─(未命中)→ agent ⇄ budget_guard ⇄ tools ─────┘
                 ↑                                 ↓
                 └──── 要求補查（附審核意見）──── review(interrupt) ─→ END（核准 / 駁回）
```

- agent 節點與 tools 節點自行實作，不使用 prebuilt agent
- 每次工具呼叫前經過 budget_guard；超出預算時不執行工具，改為要求 LLM 依現有證據結論並進入 report
- 預算只限制調查階段，report 節點永遠可以執行
- 目標命中制裁時同樣由 report 節點產出報告，證據僅為規則結果

## 3. 專案結構

```
kyt_agent/
├── pyproject.toml / README.md / .env.example
├── data/
│   ├── labels/            # 本地標籤庫 CSV（進版控）
│   └── snapshots/         # 錄製的 Etherscan 回應（進版控，供 eval 重播）
├── var/                   # checkpoint.sqlite、audit/、cases/（不進版控）
├── eval/dataset.jsonl
├── src/kyt_agent/
│   ├── __main__.py / cli.py
│   ├── config.py
│   ├── chain/
│   │   ├── client.py      # ChainClient Protocol 與資料模型
│   │   ├── etherscan.py   # Etherscan V2 實作（httpx）
│   │   └── snapshot.py    # 錄製 / 重播包裝
│   ├── labels.py
│   ├── rules.py
│   ├── graph/
│   │   ├── state.py
│   │   ├── nodes.py
│   │   ├── tools.py
│   │   ├── prompts.py
│   │   └── build.py
│   ├── report.py
│   ├── audit.py
│   └── evaluation/
└── tests/
```

相依方向：`cli` → `graph` → `chain` / `labels` / `rules` / `report` / `audit` → `config`。只有 `graph` 與 `evaluation` 接觸 LLM。

## 4. 元件

### 4.1 config

`pydantic-settings` 讀取 `.env`：

| 設定 | 預設值 | 說明 |
| ---- | ------ | ---- |
| `LLM_MODEL` | `google_genai:gemini-3.8-flash` | 傳給 `init_chat_model` |
| `ETHERSCAN_API_KEY` | 無 | `live` / `record` 模式必填 |
| `CHAIN_MODE` | `live` | `live` / `record` / `replay` |
| `MAX_DEPTH` | 3 | 相對目標地址的最大追蹤層數 |
| `MAX_TOOL_CALLS` | 25 | 每輪調查的工具呼叫上限 |
| `MAX_ADDRESSES` | 20 | 可展開（查交易對手）的地址數上限 |
| `MAX_TOKENS` | 200000 | 單一案件 LLM token 上限 |
| `TOP_COUNTERPARTIES` | 10 | `get_counterparties` 回傳的對手數 |
| `SUPPLEMENT_TOOL_CALLS` | 10 | 每次補查追加的工具呼叫額度 |
| `MAX_REVIEW_ROUNDS` | 3 | 補查輪數上限 |

### 4.2 chain

- `ChainClient` Protocol：`get_normal_transactions(address)`、`get_token_transfers(address)`、`get_transaction(tx_hash)`、`get_contract_info(address)`，回傳 pydantic 模型
- `EtherscanClient`：Etherscan V2（`chainid=1`），處理 rate limit 重試與錯誤
- `SnapshotClient`：包裝任一 client
  - `record`：委派給內部 client，並以「方法名稱 + 參數」為 key 將回應寫入 `data/snapshots/`
  - `replay`：只讀快照；缺少時拋出 `SnapshotMissError`，不回退到網路

### 4.3 labels

- CSV 欄位：`address, name, category, source`
- `category`：`sanctioned`、`mixer`、`exchange`、`defi`、`bridge`、`hack`
- `kyt labels sync-ofac`：從公開的 OFAC SDN 解析清單（0xB10C/ofac-sanctioned-digital-currency-addresses）匯入 ETH 地址至 `data/labels/ofac.csv`
- 其他標籤手動維護於 `data/labels/manual.csv`，每筆必須有可查證的 `source`

### 4.4 rules

- `screen(address)`：目標地址命中 `sanctioned` 時回傳 SEVERE 的 rule hit
- `risk_floor(case_graph)`：依證據計算風險下限
  - 目標地址本身為 `sanctioned` → SEVERE
  - 目標地址一層內有 `sanctioned` 對手 → 至少 HIGH
  - 目標地址一層內有 `mixer` 或 `hack` 對手 → 至少 MEDIUM
  - 其餘 → LOW

### 4.5 graph

**CaseState**

| 欄位 | 說明 |
| ---- | ---- |
| `case_id`、`target` | 案件識別與目標地址 |
| `messages` | agent 對話紀錄 |
| `nodes` | 已發現地址：深度、父地址、標籤、是否已展開 |
| `evidence` | 工具實際取得的證據，ID 為 tx hash 或 `label:<address>` |
| `budget` | 已用工具呼叫、已展開地址、token 數與當輪上限 |
| `rule_hits` | 規則層結果 |
| `report` | 最新報告與版本號 |
| `review_round`、`reviews` | 審核輪數與歷次決策 |

**工具**

| 工具 | 行為 |
| ---- | ---- |
| `get_counterparties(address, direction)` | 彙整一般交易與 token 轉帳，回傳依金額排序的前 N 名對手：筆數、總額、時間區間、範例 tx hash、已知標籤 |
| `lookup_address(address)` | 標籤庫分類、是否為合約、合約名稱 |
| `get_transaction(tx_hash)` | 單筆交易細節 |

- 工具只接受已出現在 `nodes` 中的地址，以及已出現在 `evidence` 中的 tx hash
- 新發現的對手以「父深度 + 1」登記；深度超過 `MAX_DEPTH` 的地址不可展開
- 快照缺漏時工具回傳「資料不可用」而非拋出例外

**prompts**：system prompt 說明調查目標、深追與停止的指引（混幣器、跨鏈橋深追；交易所熱錢包停止）、證據引用規則。

### 4.6 report

`RiskReport`（structured output）：

- `risk_level`：LOW / MEDIUM / HIGH / SEVERE
- `summary`
- `findings[]`：`claim` + `evidence`（至少一個 evidence ID）
- `fund_paths[]`：地址序列，每個節點附標籤
- `recommendation`
- `limitations`

產出後的程式檢查：

1. 證據驗證：所有 evidence ID 必須存在於 state；不符則請 LLM 重寫一次，仍不符則將該 finding 標為未驗證
2. 風險下限：`final_level = max(LLM 判定, risk_floor)`

### 4.7 review（HITL）

- `interrupt()` 將報告交給 CLI，以 rich 呈現風險等級、findings、資金路徑樹
- 決策：核准（意見選填）、駁回（理由必填）、要求補查（意見必填）
- 補查：意見以 HumanMessage 加入對話，工具呼叫額度增加 `SUPPLEMENT_TOOL_CALLS`，回到 agent
- 達 `MAX_REVIEW_ROUNDS` 後只能核准或駁回
- 結案時輸出 `var/cases/<case_id>/report.json` 與 `report.md`
- 使用 SQLite checkpointer，`kyt resume <case_id>` 可接續中斷的審核

### 4.8 audit

`var/audit/<case_id>.jsonl`，只追加。事件：`case_opened`、`rule_screen`、`tool_call`（參數、結果摘要、結果 sha256）、`llm_call`（模型、token）、`budget_exhausted`、`report_generated`（版本）、`evidence_check`、`human_decision`（審核人為 OS 使用者、決策、意見）、`case_closed`。

LangSmith 為選配，以 `LANGSMITH_TRACING=true` 啟用。

## 5. CLI

| 指令 | 說明 |
| ---- | ---- |
| `kyt investigate <address>` | 開新案件並進入審核 |
| `kyt resume <case_id>` | 接續中斷的案件 |
| `kyt labels sync-ofac` | 更新 OFAC 制裁地址 |
| `kyt eval [--record] [--model <provider:model>]` | 錄製快照或執行 eval |

## 6. Eval

- 資料集 `eval/dataset.jsonl`：約 20 筆，陽性與陰性各半，每筆含 `address`、`expected`（`risky` / `clean`）、`category`、`source`
  - 陽性：距制裁地址、混幣器或已知駭客事件地址 1～2 層的地址；不含本身即被制裁的地址
  - 陰性：只與交易所、主流 DeFi 互動的一般地址
  - 地址須從公開資料查證後加入
- `--record`：以確定性爬蟲錄製每個 case 周圍 2 層、每個地址前 `TOP_COUNTERPARTIES` 名對手的資料
- 執行時強制 `CHAIN_MODE=replay`，review 自動核准
- 判定：`risk_level >= HIGH` 視為標記
- 輸出：召回率、誤報率、每 case 的工具呼叫數、token 數、估算成本、快照 miss 次數，並與純規則 baseline（screen + risk_floor，不經 LLM）對照
- 結果寫入 `var/eval/<timestamp>-<model>.json`

## 7. 錯誤處理

| 情境 | 處理 |
| ---- | ---- |
| Etherscan rate limit / 暫時錯誤 | client 內指數退避重試，仍失敗則工具回傳錯誤訊息給 agent |
| 快照缺漏 | 工具回傳「資料不可用」，計入 miss |
| LLM 呼叫非法地址或超出深度 | 工具拒絕並說明原因，計入工具呼叫額度 |
| 預算用盡 | 不執行工具，進入 report，`limitations` 註明 |
| 報告 structured output 解析失敗 | 重試一次，仍失敗則案件標為錯誤並寫入 audit |

## 8. 測試

- 單元測試（無 LLM）：labels、rules、Etherscan client（httpx `MockTransport`）、snapshot 錄製與重播、budget_guard、證據驗證、風險下限
- 流程測試：以可輸出 tool call 的 fake chat model 走完整張 graph，含 interrupt 後的補查迴圈
- `@pytest.mark.integration`：真實 LLM 搭配重播快照跑 1～2 個 case，預設不執行
