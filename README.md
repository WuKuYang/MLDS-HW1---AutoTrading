# 111-上-機器學習與資料科學-作業一 
## MLDS HW1 AutoTrading

學生 : 楊武庫 (WuKu.Yang)  
學號 : P77111126  

### HW1 作業設計概念

本次作業一以LSTM作為主要模型    

輸入為4輸入 ( 開盤價、最高價、最低價、收盤價 )(共5天)，輸出為隔日收盤價   

![image](https://user-images.githubusercontent.com/21212753/195516753-2f26764a-b2c9-4a32-bfdb-028d1b236a86.png)

選用LSTM 主要也是因為此模型架構有時間序列的概念，以SlidingWindow方式取出前五日股價資訊進行訓練    

故訓練後該模型將擁有以前五日股價來預測隔日收盤價的功能    

本次作業策略主要只做多方  

* 股價差異 = 預期股價 - 現在股價
* if (股價差異 > 0) 買入一張 else 出清持股，直到最後一個交易日(最後一天不交易)

## 執行結果 (績效)
![image](https://user-images.githubusercontent.com/21212753/195518397-a103137e-bc67-427e-bc57-6831d194b105.png)

## 訓練過程
![image](https://user-images.githubusercontent.com/21212753/195518538-b988b63d-6de0-4a55-9706-7b915cb58823.png)

## 使用LSTM模型進行預測並產出結果
![image](https://user-images.githubusercontent.com/21212753/195518983-6db74a6f-c28d-4b54-b1de-c8216421d88b.png)

# 其他
### 如何進行訓練與預測 ?
  * 進入資料夾後並執行以下指令   
  ![image](https://user-images.githubusercontent.com/21212753/195519522-8a070ce1-d21a-4f19-98b5-4059de579ad6.png)    

`(HW1_AutoTrading) C:\>cd MLDS-HW1---AutoTrading`

`(HW1_AutoTrading) C:\MLDS-HW1---AutoTrading>python trader.py --training training_data.csv --testing testing_data.csv --output output.csv`  

  ![image](https://user-images.githubusercontent.com/21212753/195520012-c704983e-8481-48e6-9e64-bb3b10412ed2.png)  

`等待訓練完成後，會於程式碼所在位置儲存一份權重檔，Predict行為目前是放置於訓練之後，故訓練完成後會存檔，存檔後再進行讀檔後預測(最終輸出action行為csv檔案)`    
![image](https://user-images.githubusercontent.com/21212753/195520604-d35efe63-ac5e-443c-b1dd-baee97faaceb.png)

![image](https://user-images.githubusercontent.com/21212753/195520427-2a6cc5da-c76e-46f4-812f-5e6d288b4689.png)

  
* 如有需要可依 Step01~Step03修改為 訓練、讀取、預測來彈性使用    
![image](https://user-images.githubusercontent.com/21212753/195521358-36a414ab-d90b-41e2-b49a-81186bb965f1.png)

  
