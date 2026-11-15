# 分工與使用說明(AI6102 Team Project:Titanic)

## 分工
| 負責人 | 內容 | 在報告中填寫的位置 |
|---|---|---|
| LIN, YUN-FAN(G2608301F) | EDA、資料準備、特徵工程、Logistic Regression、正則化、feature ablation、cross-validation 設計、版本迭代與 leaderboard | 已完成:`report.tex` 的 3.x 與 4.1 到 4.5 |
| Member 2 | **kNN**、**Naive Bayes** | 4.6 的前兩個小節、Table 的對應列、4.1 的比較段落 |
| Member 3 | **Decision Tree**、**SVM**(線性與 RBF) | 4.6 的後兩個小節、Table 的對應列、4.1 的比較段落 |

共同:在 `report.tex` 填入姓名與 Matric、Kaggle leaderboard 截圖,並一起審稿(搜尋 `\todo` 可找到所有待填處)。

## 實驗規則(請務必一致,結果才能放在同一張表)
1. **資料**:使用 `datasets/B_v4_FamilyFlags_AgeTitlePclassParch_*`(不含標籤資訊,可直接做 CV)。各版本差異見 `datasets/README.md`。
2. **評估**:重複分層 5-fold(5 次重複,random_state=42),報告 accuracy(平均 ± 標準差)、F1、AUC。
3. **調參**:一律 nested CV(外層評估、內層 5-fold 選超參數),不要用 leaderboard 調參。
4. **標準化**:kNN、SVM 需要,放在 Pipeline 內,不要先對全部資料標準化。
5. **執行**:複製 `team_experiment_template.py`,把 `MODEL` 換成自己的模型即可,會輸出與 `results_B.csv` 相同格式的一列。
6. 請同時記錄:train 與 CV 的差距(過擬合)、超參數曲線(k、depth、C)。報告中每個方法需要:方法簡述、為什麼適合或不適合這份資料、調參過程、結果。

## 不要重複做的事(已在 `report.tex` 驗證過)
Deck、Pclass 序數化、Infant / AgeEstimated 等小旗標、特徵選擇(filter / RFE / L1)皆沒有明顯效果。

## 重新產生報告
`bash build_report.sh`(需要 pdflatex)。
