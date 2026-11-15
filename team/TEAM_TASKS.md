# 分工

- Member 1 (LIN, YUN-FAN): EDA、特徵工程、Logistic Regression、消融、整體報告
- Member 2: kNN、Naive Bayes
- Member 3: Decision Tree、SVM

## 共同設定
- 資料: data/processed/ 的 B(不含標籤資訊),說明見該資料夾 README;C 可作對照
- 評估: Repeated Stratified 5-fold CV(同一組 fold),指標 Accuracy / AUC
- 範本: team_experiment_template.py,結果放 team/results/
- 報告: 在 report/report.tex 搜尋 \\todo 補上各自的方法、調參、結果與姓名/Matric
