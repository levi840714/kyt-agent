# KYT Agent：鏈上地址風險調查（簡易版）

輸入一個 Ethereum 地址，agent 自主決定往上下游追幾層、查哪些交易，產出附證據的風險報告，交由合規人員在 CLI 核准、駁回或要求補查。

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

規則下限：目標本身被制裁為 SEVERE；第 1 層交易對手有制裁、混幣器或駭客地址為 HIGH（直接接觸即須人工審查，粉塵亦同）。

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
uv run kyt eval --model openai:gpt-6-luna       # 換模型比較
```

產出：

| 路徑 | 內容 |
| ---- | ---- |
| `var/cases/<case_id>/report.md` | 結案報告與審核紀錄 |
| `var/audit/<case_id>.jsonl` | audit log |
| `var/eval/*.json` | eval 結果 |
| `var/checkpoints.sqlite` | 案件狀態 |

## 資料

- `data/labels/ofac.csv`：OFAC SDN 的 ETH 地址（`kyt labels sync-ofac` 更新）
- `data/labels/manual.csv`：手動查證的混幣器、交易所、DeFi 地址，每筆附來源
- `eval/dataset.jsonl`：20 筆 eval 地址（10 陽性、10 陰性），皆為 2026-09-28 從鏈上實際查得
- `data/snapshots/`：eval 用的 Etherscan 快照

## Eval 結果（2026-09-28，Gemini 3.8 Flash）

| 指標 | Agent | 純規則 |
| ---- | ----- | ------ |
| 召回率 | 100% | 100%（混幣器下限調為 HIGH 後） |
| 誤報率 | 0% | 0% |
| 平均工具呼叫 | 14.6 | - |
| 20 筆總成本 | 約 $2.03 | - |

已知限制：

- 資料集的陽性案例風險都在第 1 層，尚未測到 agent 追蹤間接曝險的能力
- 代幣轉帳只記錄 symbol，假冒的「ETH」代幣會被誤認為真 ETH
- 模型可能引用自身過時知識（例如稱 Tornado Cash 仍受制裁）
- 乾淨地址的 token 用量約為風險地址的 5 倍

## 開發

```bash
uv run pytest                    # 單元與流程測試（fake model，不打外部 API）
uv run pytest -m integration     # 串接真實 LLM
uv run ruff format src tests && uv run ruff check src tests && uv run mypy src
```

設計文件：[docs/specs/2026-09-25-kyt-agent-design.md](docs/specs/2026-09-25-kyt-agent-design.md)
