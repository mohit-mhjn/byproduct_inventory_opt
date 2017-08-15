setwd("D:/Model_V1/")

library(PythonInR)
library(readxl)

pyConnect()
pyIsConnected()


#system("pyomo solve --solver=cbc 6_inventory_min_concerete.py")
#pyExecfile('output_generation.py')

data <- read_excel("Template.xlsm",sheet = "Input Demand")

size = as.character(unique(data$bird_size))

for(i in 1:length(size)){
  
  size_data = subset(data, data$bird_size==size[i])
  
  size_data$bird_size = NULL
  
  write.csv(size_data, file = "Input.csv",row.names = FALSE)
  
  system("pyomo solve --solver=cbc 6_inventory_min_concerete.py")
  pyExecfile('output_generation.py')
  
  bird_count = read.csv("Bird_count.csv")
  colnames(bird_count) = c("parts",names(bird_count[,2:ncol(bird_count)]))
  inventory = read.csv("Inventory.csv")
  colnames(inventory) = c("parts",names(inventory[,2:ncol(inventory)]))
  bird_count$bird_size = rep(size[i],nrow(bird_count))
  
  inventory$bird_size = rep(size[i],nrow(inventory))
  
  if(i==1){
    write.table(bird_count,file = "birds_requirement.csv",row.names = FALSE,col.names = TRUE,sep = ",",append = TRUE)
    write.table(inventory,file = "imbalanced_inv.csv",row.names = FALSE,col.names = TRUE,sep = ",",append = TRUE)
    
  }
  
  
  if(i>1){
    write.table(bird_count,file = "birds_requirement.csv",row.names = FALSE,col.names = FALSE,sep = ",",append = TRUE)
    write.table(inventory,file = "imbalanced_inv.csv",row.names = FALSE,col.names = FALSE,sep = ",",append = TRUE)
    
  }
  
  rm(size_data,bird_count,inventory)
  
}


