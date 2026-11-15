# 處理後資料集:A / B / C 的差異與設計理由

train 檔含 `PassengerId`、`Survived`、`fold`(分層 5-fold,seed 42),test 檔沒有後兩者。三版都沒有缺失值,未標準化。

| | A | B | C |
|---|---|---|---|
| 檔名前綴 | `A_v1_TitleAge_FamilySize` | `B_v4_FamilyFlags_AgeTitlePclassParch` | `C_v4_B_plus_GroupSurvival_RecomputeInFold` |
| 特徵數 | 21 | 23 | 35 |
| 用標籤算特徵 | 否 | 否 | 是 |
| 角色 | 第一版基準 | 簡化後的 label-free 版本 | B 加上群組存活資訊 |

## A:第一版基準

**內容**:`Sex`、`Age`、`AgeMissing`、`IsChild`、`FamilySize`、`IsAlone`、`SibSp`、`Parch`、`LogFare`、`HasCabin`,加 Title / Embarked / Pclass 的 one-hot。

**為什麼這樣處理**
- Age 有約 20% 缺值,而 Title(Master / Miss / Mr / Mrs)與年齡高度相關,所以用各 Title 的中位數補值;`AgeMissing` 保留「缺值本身」的資訊(缺 Age 的人多為三等艙)。
- Fare 右偏很重(偏度約 4.8),取 log1p 後約 0.4,讓線性模型不被少數高票價主導。
- Cabin 缺值約 77%,只留 `HasCabin`,因為有艙號大多代表一、二等艙。
- Title 取自姓名,稀有稱謂(Dr、Rev、Col…)合併成 `Rare`,避免類別太稀疏。

**優缺點**:簡單、可解釋;但 `FamilySize = SibSp + Parch + 1`,四個家庭欄位互相共線,線性模型的係數不穩定,也看不出「同行的是小孩還是父母」。

## B:簡化的 label-free 版本

**相對 A 的改動與理由**
- **Age 補值**:改用 `Title × Pclass × 是否有 Parch` 的中位數。Pclass 與 Parch 都影響年齡(Miss 帶 Parch 多半是小孩、一等艙平均較老),分組更細,補值更貼近實際。
- **家庭欄位**:`FamilySize / IsAlone / SibSp / Parch` 換成 `HasFamily`、`HasChildren`(成年且 Parch>0)、`WithParents`(未成年且 Parch>0)、`LargeFamily`(≥5 人)。Parch 對成人與小孩的意義相反(成人帶小孩 vs 小孩跟父母),拆開後能表達這個差別;`LargeFamily` 對應「大家庭存活率明顯偏低」的非線性關係,這是線性模型靠原本欄位學不到的。消融中這組欄位與 `TicketSize` 功能有重疊,兩者同時移除才會明顯下降。
- **新增 `TicketSize` 與 `LogFarePP`**:同票號代表一起旅行,`Fare` 在 Titanic 是整張票的價格,除以人數才是每人票價。兩者只用計數,不用標籤。

**優缺點**:共線少、好解釋、沒有標籤資訊,CV 與 A 相當(約 0.83),代表在線性模型下,A 到 B 的重構主要是換成更好解讀的表示,而不是更高的準確率。`TicketSize` 的計數包含 test 乘客(只用票號,不涉及標籤)。

## C:B + 同組存活率

**內容**:在 B 之上新增 12 欄,由同票號 / 同姓氏的**其他**乘客的存活結果算出:`Ticket_rate`、`Surname_rate`(平滑存活率)、`*_n`(同組已知標籤人數)、`*_WomanChild_r`、`*_AdultMale_r`(分別只看女性與小孩 / 成年男性)。

**為什麼這樣處理**
- 單看性別與艙等的模型到瓶頸,剩下的錯誤多集中在「同組一起存活或一起罹難」的家庭與同行群組(例如整家三等艙女性全數罹難),這類資訊只能從同組其他人的結果得到。
- 平滑 `(s + 2·ȳ)/(n + 2)` 讓小群組往整體均值收縮,避免 n=1 時比率只有 0 或 1。
- 拆成 WomanChild / AdultMale,是因為「女性與小孩優先」的規則在群組內才成立:同組裡女性小孩的結果對女性小孩有預測力,對成年男性沒有。
- 每一列的值排除自己的標籤(leave-one-out),避免特徵直接洩漏該列答案。

**優缺點**
- 目前唯一穩定有效的新資訊,CV 約 +1 個百分點,leaderboard 約 0.77 → 0.79。
- 使用了標籤:train 的值是用全部訓練標籤(排除自己)算的,若直接拿 `fold` 做 CV,驗證 fold 其他人的標籤會滲入特徵,分數偏樂觀;`features/features_v4.py` 的 `make4` 是在每個 fold 內重新計算,檔案內的值則是全訓練集版本。
- Leave-one-out 編碼對彈性模型(深樹、梯度提升)可能產生假訊號:同組存活者的值會略低於死亡者。
- 小群組估計雜訊大;test 內的同組乘客沒有標籤可用,所以 test 的特徵資訊比 train 少,這也是 CV 與 leaderboard 有落差的原因之一。
