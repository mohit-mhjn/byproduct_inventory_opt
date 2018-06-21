library(dplyr)
library(plyr)
library(reshape)
library(reshape2)
library(sqldf)
library(RODBC)

setwd("C:/Users/Krishna/Desktop/DPP SALSE FORECAST")


dpp_sales <- read.csv("./Output/Forecast/Final/DPP SALES FORECAST OUTPUT.csv", header=TRUE)

conn1 = odbcConnectExcel2007("DPP SALES DASHBOARD.xlsx") # open a connection to the Excel file
sqlTables(conn1)$TABLE_NAME # show all sheets
limit = sqlFetch(conn1, "Size")
close(conn1)

head(limit)

limit$Month = as.character(limit$Month)

dpp_sales$AVG.Parts.Weight = limit$Upper*dpp_sales$Part.Weight.UL+ limit$Lower*dpp_sales$Part.Weight.LL
dpp_sales$count_of_parts = floor(dpp_sales$Forecast/dpp_sales$AVG.Parts.Weight)
dpp_sales$count_of_parts[which(is.na(dpp_sales$count_of_parts))] <- 0.000001
dpp_sales$NOB = floor(dpp_sales$count_of_parts/dpp_sales$Parts.per.bird)
dpp_sales$NOB[which(is.na(dpp_sales$NOB))]<-0.000001

dpp_sales$count_of_parts[which(is.infinite(dpp_sales$count_of_parts))] <- 0.000001
dpp_sales$NOB[which(is.infinite(dpp_sales$NOB))] <- 0.0000001

sum(dpp_sales$NOB)
sum(dpp_sales$count_of_parts)

write.csv(dpp_sales, file = "DPP SALES FORECAST OUTPUT.csv",row.names = FALSE)

data_new = subset(dpp_sales, dpp_sales$Year == limit$Year & dpp_sales$Month == limit$Month & dpp_sales$Category_New %in% c("MAR","PP"))


### transform the data

d1 <- dcast(data_new, Week.Date+Day+Bird.Size ~ Product.Group, value.var = "NOB",fun.aggregate = sum)
d1$`NA` = NULL


d1$`3JW` = d1$`3 Joint Wings JC`+ d1$`3 Joint Wings MC`
d1$Thigh = d1$`Thigh MC`+d1$`Boneless Thigh `
d1$TotalQleg = d1$Qleg+d1$`Boneless Leg`
d1$Breast_Mod = ifelse( d1$Breast >= d1$Fillet , d1$Breast+d1$SWBH,d1$Fillet + d1$SWBH)


d1$Thigh_Mod = ifelse( d1$Thigh >= d1$Drumstick , d1$Thigh + d1$TotalQleg , d1$Thigh)
d1$Drumstick_Mod = ifelse( d1$Thigh < d1$Drumstick , d1$Drumstick + d1$TotalQleg , d1$Drumstick)
d1$Rib_Mod = ifelse( d1$Rib >= d1$Keel , d1$Rib + d1$Breast_Mod , d1$Rib)
d1$Keel_Mod = ifelse( d1$Rib < d1$Keel , d1$Keel + d1$Breast_Mod , d1$Keel)

d1$Max1 = apply(d1[,c("3JW","Thigh_Mod","Drumstick_Mod","Rib_Mod","Keel_Mod")],1,max)

if(length(colnames(d1)[which(names(d1) == "M2WHFG")])<=0){
  
  d1$NOB_Mod = rowSums(d1[,c("Max1","WB CUT","L1BP","m1BP","M1BP","M2BP","M2SP","M1SP","L1WHFG","NC","S2BP","Others")],na.rm = TRUE)
  
} else{
  d1$NOB_Mod = rowSums(d1[,c("Max1","WB CUT","L1BP","m1BP","M1BP","M2BP","M2SP","M1SP","L1WHFG","M2WHFG","NC","S2BP","Others")],na.rm = TRUE)
  
}


#Without Bird Size


d2 <- dcast(data_new, Week.Date+Day ~ Product.Group, value.var = "NOB",fun.aggregate = sum)
d2$`NA` = NULL


d2$`3JW` = d2$`3 Joint Wings JC`+ d2$`3 Joint Wings MC`
d2$Thigh = d2$`Thigh MC`+d2$`Boneless Thigh `
d2$TotalQleg = d2$Qleg+d2$`Boneless Leg`
d2$Breast_Mod = ifelse( d2$Breast >= d2$Fillet , d2$Breast+d2$SWBH,d2$Fillet + d2$SWBH)


d2$Thigh_Mod = ifelse( d2$Thigh >= d2$Drumstick , d2$Thigh + d2$TotalQleg , d2$Thigh)
d2$Drumstick_Mod = ifelse( d2$Thigh < d2$Drumstick , d2$Drumstick + d2$TotalQleg , d2$Drumstick)
d2$Rib_Mod = ifelse( d2$Rib >= d2$Keel , d2$Rib + d2$Breast_Mod , d2$Rib)
d2$Keel_Mod = ifelse( d2$Rib < d2$Keel , d2$Keel + d2$Breast_Mod , d2$Keel)

d2$Max1 = apply(d2[,c("3JW","Thigh_Mod","Drumstick_Mod","Rib_Mod","Keel_Mod")],1,max)

if(length(colnames(d2)[which(names(d2) == "M2WHFG")])<=0){
  
  d2$NOB_Mod = rowSums(d2[,c("Max1","WB CUT","L1BP","m1BP","M1BP","M2BP","M2SP","M1SP","L1WHFG","NC","S2BP","Others")],na.rm = TRUE)
  
} else{
  d2$NOB_Mod = rowSums(d2[,c("Max1","WB CUT","L1BP","m1BP","M1BP","M2BP","M2SP","M1SP","L1WHFG","M2WHFG","NC","S2BP","Others")],na.rm = TRUE)
  
}



d1 = ddply(d1,c("Week.Date","Day"), transform, TotalUnits = sum(NOB_Mod))
d1$Perdis = d1$NOB_Mod/d1$TotalUnits
d4 <- merge(d1,d2[,c("Week.Date","Day","NOB_Mod")],by= c("Week.Date","Day"))
d4$Final_Units = d4$Perdis*d4$NOB_Mod.y
d4$Final_Units = round(d4$Final_Units,0)
write.csv(d4,"d4.csv",row.names = FALSE)

