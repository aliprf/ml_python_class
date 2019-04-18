from collections import defaultdict


class TrainingDataSet(object):
    labelsDataSet = []
    imagesDataSet = []
    labelIndexes = defaultdict(list)  # dic: dsi -> []  for each data set
    meanVectorByLabel = defaultdict(list)  # dic: dsi -> meanVector  for each data set
    covarianceMatrixByLabel = defaultdict(list)  # dic: dsi -> CoVmatrix  for each data set
    W_matrix_ByLabel = defaultdict(list)  # dic: dsi -> -1/2 CoInverse  for each data set
    w_matrix_ByLabel = defaultdict(list)  # dic: dsi -> -CoInverse*mean  for each data set
    w_zero_ByLabel = defaultdict(list)  # dic: dsi -> -1/2mean_inverse*CoInverse*mean - 1/2ln(det_Cov) + ln(prior)


class TestSetData(object):
    labelsDataSet = []
    imagesDataSet = []