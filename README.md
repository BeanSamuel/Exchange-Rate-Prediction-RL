# 2023 Artificial Intelligence Hw2 -- 多國外幣匯率預測(RL)

## 任務
利用強化學習 (Reinforcement Learning) 進行外匯投資決策

## Environment 規則
- 有 8 種外幣+台幣一共 9 種
- 某個時間點 𝑖 只能同時擁有一種幣
- 初始本金 = 1 元台幣
- **Observation**: 第 𝑖 − 10 日 ~ 第 𝑖 日的外匯資料 (𝑖 ≥ 10)
- **Action**: 買入其中一種外幣 or 買入台幣總共 9 種 actions
- 買入新貨幣時會先賣出原貨幣再買入新貨幣(都是以現鈔來看)
- 買入貨幣會以"現鈔賣出"資料，賣出貨幣會以"現鈔買入"資料做計算，中間價差即為銀行手續費。(因為現鈔賣出、現鈔買入皆為銀行面相)
- 若買入的幣別和持有的幣別相同，則不做任何動作
- **Position**: 目前持有的幣別，一共 9 種
- **Total Reward**: 持有台幣時為 0，持有其他外幣，則計算此外幣匯率第 𝑖 日和 𝑖 - 1 日的之間的漲跌幅，每天累加後的總合
- **Total Profit**: 資金成長比 = 最終本金

## 環境
Python版本: Python 3.10.9
```cmd!
pip install -r requirements.txt
```

## 模型訓練
在train.ipynb中運行以下的指令
``` cmd!
!python run.py
```

## 模型預測
在train.ipynb中運行以下的指令
```cmd!
!python run.py test=True test_data='./test.csv'py
```

