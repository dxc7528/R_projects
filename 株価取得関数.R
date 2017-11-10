
# 株価取得関数 
# 2017年7月14日




# -----------------------------------------------------------------------
# Webから株式データを取得する関数
# -----------------------------------------------------------------------

getStockFromKdb <- function(code, year){
  # 『株価データサイト k-db.com』（http://k-db.com/）様からデータを取得する
  # 
  # Args:
  #   code: 証券コード
  #   year: 取得対象となる年
  # 
  # Returns: 
  #   stock: 株価のデータ（日足）
  
  stocks <- NULL
  
  # 指定された証券コード、年をまとめて取得する
  for(i in 1:length(code)){
    for(j in 1:length(year)){
      
      # 『k-db.com』のURLを指定してデータを取得
      url <- paste("http://k-db.com/stocks/", code[i], "-T/1d/", year[j], "?download=csv",sep="")
      stock <- read.csv(url, header = T)
      
      # 列名称を変更
      colnames(stock) <- c("Date", "Open", "High", "Low", "Close", "Volume", "Volume_money")
      
      # 証券コードを追加
      tickerSymbol <- data.frame(tickerSymbol = rep(code[i], nrow(stock)))
      stock <- cbind(tickerSymbol, stock)
      
      # 取得されたデータを保存
      if(is.null(stocks)){
        stocks <- stock
      } else{
        stocks <- rbind(stocks, stock)
      } 
      
      # 次のデータを取得する前に少し待つ
      if(i < length(code) || j < length(year)){
        Sys.sleep(0.5)
      }
      
    }
  }
  
  return(stocks)
}