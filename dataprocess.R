dat = read.table("D:\\PythonLearning\\SRT_SalesPrediction\\graph&data\\pred.txt",encoding = 'utf-8',sep = '\t')
mat = as.matrix(dat)
seq = as.vector(mat)
seq[1:90]