# 組員用資料集說明(Titanic,AI6102 Team Project)

所有檔案都在 `datasets/`。每個 train 檔都有:
- `PassengerId`、`Survived`(標籤)
- `fold`:0 到 4,**分層 5-fold,seed=42**。請大家一律用這個欄位切分,這樣才能公平比較分數。
- 其餘為特徵。test 檔沒有 `Survived` 和 `fold`。

三個版本都沒有缺失值,也都**沒有標準化**(線性模型、kNN、SVM 請自行在 pipeline 內標準化)。

| 檔案 | 特徵數 | 用到標籤嗎 | 可直接做 CV 嗎 | 建議用途 |
|---|---|---|---|---|
| `A_v1_TitleAge_FamilySize_*` | 21 | 否 | **可以** | 基準線、課堂方法比較 |
| `B_v4_FamilyFlags_AgeTitlePclassParch_*` | 23 | 否 | **可以** | **推薦給大家做模型實驗** |
| `C_v4_B_plus_GroupSurvival_RecomputeInFold_*` | 35 | **是** | **不可以直接用** | 只用來看最終提交,CV 需自行重算(見下) |

## 各版本差異

### A:v1 基本版
- Age:用各 Title 的中位數補值,另有 `AgeMissing`。
- 特徵:`Sex`、`Title`(one-hot)、`FamilySize`、`IsAlone`、`SibSp`、`Parch`、`LogFare`、`HasCabin`、`Embarked`(one-hot)、`Pclass`(one-hot)、`IsChild`。
- 缺點:`FamilySize`、`SibSp`、`Parch` 彼此共線,個別係數不好解釋。

### B:v4 簡化版(label-free)
相對於 A 的調整:
1. **Age 補值**:用 `Title × Pclass × 是否有 Parch` 的中位數(補值誤差 MAE 8.24,對照全體中位數 11.16、舊方法 8.61)。
2. **家庭特徵簡化**:`SibSp`、`Parch`、`FamilySize`、`IsAlone` 換成
   - `HasFamily`:`SibSp + Parch > 0`
   - `HasChildren`:成年人(Age ≥ 18)且 `Parch > 0`
   - `WithParents`:未成年且 `Parch > 0`
   - `LargeFamily`:家庭人數 ≥ 5
3. 新增 `TicketSize`(同票號人數)與 `LogFarePP`(票價除以同票號人數後取 log),這兩項只用計數,不用標籤。
4. `Fare` 用 `log1p`(偏態 4.79 → 0.39)。

### C:v4 + 群組存活率(用到標籤)
在 B 的基礎上加 12 欄,由「同票號 / 同姓氏」的其他乘客的存活結果算出:
- `Ticket_rate`、`Surname_rate`:該組其他人的存活率(平滑,先驗權重 2)。
- `*_n`:該組已知標籤的其他人數。
- `*_WomanChild_r`、`*_AdultMale_r`:只看同組的女性與小孩 / 成年男性的存活率。

這是目前**效果最好的特徵**(CV 約 +1 個百分點,leaderboard 從約 0.77 提升到約 0.79),但有兩個陷阱:

1. **不能直接在 `fold` 上做 CV**。train 列的值是用「全部訓練資料的標籤(排除自己)」算的,驗證 fold 中其他人的標籤已經影響了特徵,CV 分數會偏樂觀。正確做法是在每個 fold 內只用該 fold 的訓練資料重算,可直接呼叫:
   ```python
   from features_v4 import make4
   A, B = make4(train_index, valid_index)   # 以 fold 內的訓練資料計算群組特徵
   ```
2. **Leave-one-out 編碼可能讓樹模型學到「洩漏」**:同組中,存活者的 LOO 比例會比死亡者略低,過度彈性的模型(深樹、GB)可能因此學到不會推廣的規律。請使用淺層模型或較強的正則化。

## 哪一個在方法論上最合理?
- **做模型與特徵實驗:用 B**。沒有標籤洩漏,可以直接用 `fold` 做 CV,分數可信。
- **想用群組特徵:用 C 的做法,但必須在每個 fold 內重算**(`make4`)。這是目前 CV 最高的設計,也是方法論上正確的版本。
- **最終繳交**:v2(leaderboard 0.79665,CV 0.840)與其他版本的差異(最多 7 人)小於測試集的抽樣誤差(約 ±8 人),統計上無法區分。

## 已測試、沒有明顯效果的方向(請不要重複做)
Deck、人均票價以外的票價處理、Infant / AgeEstimated / ChildNoParent / WifeAboard 旗標、Pclass 序數化、特徵選擇(filter / RFE / L1)、不同模型(SVM、kNN、決策樹、Naive Bayes 等)的差異都在誤差範圍內。

## 可以接著做的方向
- 不同模型在 B 上的比較(課堂方法),以及 B 與 C(fold 內重算)的差異。
- 超參數調整請用 nested CV,不要直接看 leaderboard。
- 結果請記錄:模型、資料版本、CV 平均與標準差。
