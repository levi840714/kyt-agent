# AGENTS.md

鏈上地址風險調查 Agent（KYT）。以 Python、LangChain、LangGraph 實作，形式為 CLI。
設計文件在 `docs/specs/`，實作計畫在 `docs/plans/`。

## 目錄結構

```
kyt_agent/
├── pyproject.toml / uv.lock / README.md / .env.example
├── data/
│   ├── labels/            # 地址標籤庫（OFAC 與手動查證）
│   ├── snapshots/         # eval 用的 Etherscan 快照
│   └── tokens.csv         # 已知代幣，供偽冒代幣與粉塵判斷
├── eval/dataset.jsonl     # eval 資料集
├── src/kyt_agent/
│   ├── cli.py / render.py # Typer 指令與 rich 顯示
│   ├── config.py          # Settings（pydantic-settings）
│   ├── models.py          # 共用領域模型
│   ├── labels.py / tokens.py / rules.py / counterparties.py / report.py / audit.py
│   ├── chain/             # ChainClient Protocol、Etherscan、快照
│   ├── graph/             # LangGraph：state、nodes、tools、prompts、組裝
│   └── evaluation/        # 資料集、爬蟲、指標、runner
├── tests/
└── var/                   # 執行產物（不進版控）
```

## 工具鏈

- **Python**：3.12 以上
- **套件管理**：`uv`（`uv add`、`uv run`），不直接使用 `pip install`
- **Lint / Format**：`ruff`，line-length 100；`graph/prompts.py` 豁免 E501，prompt 文字保持原樣
- **型別檢查**：`mypy src`，公開函式必須有型別標註
- **測試**：`pytest`；單元與流程測試使用 fake model 與 `StubChain`，不打外部 API；串接真實 LLM 的測試標記 `@pytest.mark.integration`，預設不執行
- **設定**：`pydantic-settings` 讀取 `.env`，程式中不得寫死任何 key

```bash
uv sync
uv run pytest
uv run ruff format src tests && uv run ruff check src tests && uv run mypy src
```

## 程式碼規範

- Clean code：單一職責、命名清楚、函式短小、避免過度抽象
- State 與資料模型使用 `TypedDict` / `pydantic.BaseModel`
- 依賴方向單一：`cli` → `graph` → `chain` / `labels` / `rules` / `report` / `audit` → `config`
- 不留死碼、未使用的 import、註解掉的程式碼

### 註解規則

- 只寫必要的註解，說明「為什麼」而非重述程式碼
- 每段註解不超過兩行
- docstring 同樣精簡，只在公開 API 或行為不直觀時撰寫

## LangChain / LangGraph 慣例

- State、Node、組裝分開放置，`build_graph()` 回傳 compiled graph
- 硬性規則（黑名單比對、風險下限、預算上限、證據驗證）由程式執行，不交給 LLM
- Prompt 集中於 `graph/prompts.py`
- 模型透過 `init_chat_model("<provider>:<model>")` 由設定注入，不寫死在 node 中
- 案件以 SQLite checkpointer 保存；node 在 `interrupt()` 之前不得產生副作用，避免 resume 時重複執行

## LLM 模型設定

- `LLM_MODEL` 預設 `google_genai:gemini-3.8-flash`，可切換為任何 `init_chat_model` 支援的模型
- API key 由各 provider SDK 從環境變數讀取，CLI 進入點以 `load_dotenv()` 載入 `.env`
- eval 的成本估算表在 `evaluation/metrics.py` 的 `PRICING`，價格變動時一併更新

## 安全

- `.env` 不進版控，只提交 `.env.example`
- 不在 log、錯誤訊息或工具輸出中出現 API key；Etherscan 錯誤只保留狀態碼或例外類型
- 主網只讀，專案內不存在任何私鑰或簽名邏輯
