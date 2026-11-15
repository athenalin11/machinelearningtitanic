# Titanic – AI6102 team project

腳本皆從專案根目錄執行,例如 `python3 logistic_regression/run_B_experiments.py`。

## 資料夾與檔案

### `data/`
- `raw/`:Kaggle 原始檔(`train.csv`、`test.csv`、`gender_submission.csv`)
- `interim/`:第一版處理結果(`train_processed.csv`、`test_processed.csv`)
- `processed/`:A / B / C 三版特徵資料集,差異與優缺點見 `processed/README.md`

### `features/`
- `improve_experiments.py`:早期特徵實驗(姓氏 / 票號群組、交互項、目標群組存活率),也是 v2 最終提交的特徵來源
- `final_v3.py`:v3 特徵與模型
- `features_v4.py`:B 特徵(`B4`)、C 特徵(`make4`,fold 內重算)、集成(`ens`)
- `rule_model.py`:規則式模型(性別 + 群組),CV 0.847
- `export_datasets.py`:輸出 `data/processed/` 的 A/B/C

### `logistic_regression/`
- `titanic_pipeline.py`:EDA、前處理、LR、C 掃描、L1 vs L2、指標、消融、重複 CV
- `run_B_experiments.py`:在 B 特徵上重跑模型比較、正則化掃描、消融
- `ablation_overlap_check.py`:檢查家庭旗標與 TicketSize 的重疊
- `results/`:`results_B.csv`、`ablation_B.csv`

### `eda/`
- `make_eda_figure.py`:產生報告用的 EDA 圖

### `experiments/`
各版本提交檔的產生腳本與其他嘗試(`final_v2/v4/v6.py`、`make_v3/v5_submission.py`、`check_features.py`、`family_age_experiments.py`)。

### `submissions/`
- `final/submission_v2.csv`:最終提交,public 0.79665
- `history/`:其他版本;分數整理在 `submissions/README.md`

### `report/`
- `report.tex` / `report.pdf`:報告;`build.sh` 編譯
- `figures/`、`tables/`:報告用的圖與表(`tables/` 由實驗腳本輸出)

### `team/`
- `TEAM_TASKS.md`:分工
- `team_experiment_template.py`:隊友實驗用範本(沿用相同 CV 與資料)
- `results/`:隊友結果存放處
