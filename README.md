# KYT Agent：鏈上地址風險調查

輸入一個 Ethereum 地址，agent 自主決定往上下游追幾層、查哪些交易，產出附證據的風險報告，交由合規人員在 CLI 核准、駁回或要求補查。

## 範圍與限制

| 項目 | 說明 |
| ---- | ---- |
| 支援 | Ethereum 主網、Etherscan V2 資料、本地標籤庫（OFAC SDN 與手動查證的地址）、CLI 人工審核 |
| 不支援 | 多鏈與跨鏈追蹤、即時交易監控、商業標籤資料 |
| 資料範圍 | 每個端點（一般交易、internal、代幣轉帳）各取最近 `TX_PAGE_SIZE`（預設 100）筆 |
| 定位 | 輔助調查並產出附證據的報告，最終判斷由合規人員決定，不取代商業 KYT 服務 |

## 流程

```
screen ──(命中制裁)──────────────────────────→ report
   └─(未命中)→ agent ⇄ budget_guard ⇄ tools ─────┘
                 ↑                                 ↓
                 └──── 要求補查（附審核意見）──── review(interrupt) ─→ 結案
```

| 階段 | 負責 | 內容 |
| ---- | ---- | ---- |
| screen | 程式 | 目標地址比對標籤庫，命中制裁直接定為 SEVERE |
| agent ⇄ tools | LLM 決策、程式把關 | LLM 決定往哪查；程式限制層數、工具呼叫次數、展開地址數與 token，只允許查詢調查中出現過的地址 |
| report | LLM 撰寫、程式檢查 | 每項發現須引用實際查到的證據；風險等級不得低於規則下限 |
| review | 人 | 核准、駁回或要求補查（附意見，追加工具額度，最多 3 輪） |

規則下限：目標本身被制裁為 SEVERE；目標本身或第 1 層交易對手帶有制裁、混幣器或駭客（`hack`）標籤為 HIGH（直接接觸即須人工審查，粉塵亦同）。目前標籤庫沒有 `hack` 標籤，駭客地址需自行查證後加入 `manual.csv` 才會生效。

代幣旗標：交易對手彙整時，程式依 `data/tokens.csv` 為每筆轉帳打上旗標。ERC-20 的 symbol 為 `ETH`，或與已知代幣同 symbol 但合約不同，標記為偽冒代幣，金額不計入總額；金額低於門檻者標記為粉塵。旗標不影響規則下限，直接接觸制裁、混幣器、駭客地址即使為粉塵或偽冒代幣仍為 HIGH。

間接曝險：查詢資金來源時，工具會附上該地址最近轉入（不含偽冒代幣與粉塵）中來自風險標籤地址的比例；中間地址過半資金來自混幣器、駭客或制裁地址時，報告判定為 HIGH。

成本控制：
- 工具輸出以 `T1`、`T2`… 代替交易 hash，存檔報告與 audit log 會換回真實 hash
- 連續 `WRAP_UP_HINT_AFTER`（預設 6）次查詢未發現風險時，提示 agent 評估是否結案
- 單次 LLM 呼叫逾時 `LLM_TIMEOUT`（預設 120 秒），避免網路卡住時整個案件停住

- 主網只讀，不涉及任何私鑰
- 每個案件以 SQLite checkpointer 保存，審核中斷後可用 `kyt resume` 接續
- 每個案件寫入只追加的 audit log：規則判斷、工具呼叫（含結果 sha256）、LLM token、報告版本、人工決策

## 安裝

```bash
uv sync
cp .env.example .env   # 填入 ETHERSCAN_API_KEY 與 LLM 的 API key
uv run kyt labels sync-ofac
```

`LLM_MODEL` 預設為 `google_genai:gemini-3.8-flash`，可改為任何 `init_chat_model` 支援的 `<provider>:<model>`。

## 使用

```bash
uv run kyt investigate 0x...                    # 開新案件
uv run kyt resume <case_id>                     # 接續中斷的案件
uv run kyt eval --record                        # 錄製 eval 快照（資料集變動時才需要）
uv run kyt eval                                 # 用快照重播跑 eval
uv run kyt eval --fill-missing                  # 重播，缺漏的快照即時補錄（需要 ETHERSCAN_API_KEY）
uv run kyt eval --model openai:gpt-6-luna       # 換模型比較
```

產出：

| 路徑 | 內容 |
| ---- | ---- |
| `var/cases/<case_id>/report.md` | 結案報告與審核紀錄（eval 不寫入） |
| `var/audit/<case_id>.jsonl` | audit log |
| `var/eval/*.json` | eval 結果 |
| `var/checkpoints.sqlite` | 案件狀態 |

## 資料

- `data/labels/ofac.csv`：OFAC SDN 的 ETH 地址（`kyt labels sync-ofac` 更新）
- `data/labels/manual.csv`：手動查證的混幣器、交易所、DeFi 地址，每筆附來源
- `data/labels/relayers.csv`：Tornado Cash relayer（123 個），取自 Tornado.Cash Relayer Registry 合約的 `RelayerRegistered` 事件；經由 relayer 間接接觸混幣器，除非有其他風險跡象，最高判定為 MEDIUM
- `data/tokens.csv`：經查證的主流代幣（USDT、USDC、DAI、WETH），供偽冒代幣與粉塵判斷
- `eval/dataset.jsonl`：28 筆 eval 地址（15 陽性、13 陰性），皆從鏈上實際查得並註明來源
  - v1（2026-09-28）：混幣器存款人 5、制裁直接往來 3、制裁粉塵 2、交易所用戶 10
  - v1.1（2026-09-29）：間接曝險 5、困難陰性 2（中間地址風險轉入低於 10%）、relayer 下游 1（中間地址疑似 Tornado Cash relayer，依審查政策列為陰性）
- `data/snapshots/`：eval 用的 Etherscan 快照

## Eval 結果：v1.1（2026-09-29，Gemini 3.8 Flash）

| 指標 | v1.1 Agent | v1.1 純規則 | v1 Agent |
| ---- | ---------- | ----------- | -------- |
| 召回率 | 100%（15/15） | 67%（10/15） | 100%（10/10） |
| 誤報率 | 8%（1/13） | 0% | 0%（0/10） |
| 平均工具呼叫 | 7.6 | - | 14.6 |
| 平均 token | 22,707 | - | 111,789 |
| 交易所用戶平均 token | 19,359 | - | 186,822 |
| 總成本 | 約 $0.74（28 筆） | - | 約 $2.03（20 筆） |

- 間接曝險 5 筆，純規則全部漏掉，agent 全部攔下
- 唯一誤報為 relayer 下游案例：中間地址 66% 轉入來自 Tornado Cash，agent 依「過半」規則判定 HIGH；目前的規則與工具無法區分 relayer 手續費收入與洗錢中間人
- 偽冒「ETH」代幣案例的報告正確指出地址投毒、無實質資金轉移，並仍判定 HIGH
- 無錯誤；`--fill-missing` 補錄 11 個快照檔；補錄模式即時查詢缺漏資料，故缺漏為 0 是設計使然

## Eval 結果：v1（2026-09-28，Gemini 3.8 Flash）

| 指標 | Agent | 純規則 |
| ---- | ----- | ------ |
| 召回率 | 100% | 100%（混幣器下限調為 HIGH 後） |
| 誤報率 | 0% | 0% |
| 平均工具呼叫 | 14.6 | - |
| 20 筆總成本 | 約 $2.03 | - |

- 純規則召回率 100% 是在混幣器下限調為 HIGH 後，以同一批快照重新計算的結果；`var/eval/` 中保存的 JSON 為調整前執行，記錄為 50%
- 本次執行共 31 次快照缺漏，約占工具呼叫的 10%，已於 v1.1 以 `--fill-missing` 處理

## 已知限制

- 只辨識已在 Relayer Registry 登記的 Tornado Cash relayer，未登記的服務營運者下游仍可能被判為間接曝險
- 困難陰性與粉塵案例各只有 2 筆，相關指標僅供參考
- 轉入比例以每個端點最近 `TX_PAGE_SIZE`（預設 100）筆的轉帳計算，不代表完整歷史
- 標籤庫沒有 `hack` 與 `bridge` 標籤，相關規則需自行加入查證過的地址才會生效
- 部分網路環境以 IPv6 連線 Google API 會卡住，可改用 IPv4 或其他網路

## 開發

```bash
uv run pytest                    # 單元與流程測試（fake model，不打外部 API）
uv run pytest -m integration     # 串接真實 LLM
uv run ruff format src tests && uv run ruff check src tests && uv run mypy src
```

設計文件：[v1](docs/specs/2026-09-25-kyt-agent-design.md)、[v1.1](docs/specs/2026-09-29-kyt-agent-v1.1-design.md)
