時系列解析とARIMAモデル

mizuho <- getStockFromKdb(8411, 2016:2017)

library(quantmod)
library(xts)

# xts形式に変換(1列目は証券コードなので削除)
mizuho_xts <- as.xts(read.zoo(mizuho[,-1]))

# ローソク足のグラフを描く
chartSeries(
  mizuho_xts, 
  type="candlesticks"
)



# 対数差分系列にする
log_diff <- diff(log(mizuho_xts$Close))[-1]

# 訓練データとテストデータに分ける
train <- log_diff["::2017-06-30"]

test <- log_diff["2017-07-01::"]

# ARIMAモデルによる予測
# -----------------------------------------------------------------------

# install.packages("forecast")
library(forecast)

# ARIMAモデルの推定
model_arima <- auto.arima(
  train,
  ic="aic",
  stepwise=F,
  approximation=F,
  max.p=10, 
  max.q=10,
  max.order=20,
  parallel=T,
  num.cores=4
)


# train：訓練データ
# ic="aic"：モデル選択の規準としてAICを使う
# ARIMA(p,d,q)のpやd、qといった次数を決めることをモデル選択と呼びます
# AICはよく使われるモデル選択の規準です。予測精度が高くなるようにモデルを選んでくれます
# AIC以外にもBIC(bicと指定)やAICC(aicc)が使えます
# stepwise=F:計算をケチらず、すべての「p,q」を網羅的に調べろという指定
# approximation=F:計算の簡略化を行わないという指定
# max.p=10:ARモデルの最大次数
# max.q=10:MAモデルの最大次数
# max.order=20:p+dの最大値の指定
# max.p・max.q・max.orderの3つを合わせてモデル選択の範囲を指定しています
# これによりARMA(0,0)~ARMA(1,0)~ARMA(10,0)~ARMA(0,10)~ARMA(10,10)までをすべての次数で総当たり的に計算してくれます
# 和分過程か否かは、内部で単位根検定という手法を使って勝手に判断してくれるので、設定は不要です（だからdは指定しなくてOK）
# 今回は当たりませんが、季節変動を取り込んだモデルだった場合は、『max.order』で、ARMAの次数＋季節変動の次数の合計値を指定します（季節変動は、今回は関係ありませんので0次となっています。なので、今回の例では無視できました。季節変動が入ったモデルの場合は注意してください）
# parallel=T:計算を並列化して早くするという指定
# num.cores=4:並列化の際のコア数
# 4にしてあるのは管理人のPCに合わせてあります。お手持ちのPCのコア数を調べて数値を変えてください


model_arima

f_arima <- forecast(model_arima, h=9)

# ナイーブな予測も併せて作成
f_rw <- rwf(train)
f_mean <- meanf(train)


accuracy(f_arima, test[1:9])
accuracy(f_rw, test[1:9])
accuracy(f_mean, test[1:9])



# データを入れると、1期先の予測をしてくれる関数
calcForecast <- function(data){
  model <- Arima(data, order=c(1,0,0))
  return(forecast(model,h=1)$mean)
}
 # 訓練データの長さ
  length(train)
  
  
  f_arima_2 <- rollapply(log_diff, 367, calcForecast)
  
  # 1期先の予測値なので、実際の日付とずれている。
  # lagを使って実際の日付に合わせる
  f_arima_2 <- lag(f_arima_2)
  
  # NAを消す
  f_arima_2<- f_arima_2[!is.na(f_arima_2)]
  
  
  f_mean_2 <- rollapply(log_diff, 367, mean)
  # 1期先の予測値なので、実際の日付とずれている。
  # lagを使って実際の日付に合わせる
  f_mean_2 <- lag(f_mean_2)
  # NAを消す
  f_mean_2 <- f_mean_2[!is.na(f_mean_2)]
  f_mean_2
  
  # 1期前予測
  f_rw_2 <- lag(log_diff["2017-06-30::"])
  f_rw_2<- f_rw_2[!is.na(f_rw_2)]
  
  
  
   accuracy(as.ts(f_arima_2), test)

   accuracy(as.ts(f_mean_2), test)

   accuracy(as.ts(f_rw_2), test)
   
   
   plot(log_diff["2017-06::"], main="みずほHG終値の対数差分系列")
   lines(f_arima_2, col=2, lwd=2)
   lines(f_mean_2, col=4, lwd=2)
   lines(f_rw_2, col=5, lwd=2)
   