import numpy as np
from data_generator import DataGenerator
from clue_nas import CreateCLUENAS,CompileCLUENAS,CLUENASPredictCellPths
from model_operation import Training
from json_io import JSON2Dict,Dict2JSON
from nas_prcss import SamplingCellPths,SamplingBatches,RankingCellPths,MaxAccByEstimate,CostTime,MaxAccByEstimate
import random
import math

def TrainEval(predict_space="nas201",budget=200):
    nas101_cell_dir="data/nasbench101_cifar10"
    nas201_cell_dir="data/nasbench201_cifar10"
    nas301_cell_dir="data/nasbench301_cifar10"
    batch_size=16
    m=5
    iters=(lambda budget: max(1,round(budget/batch_size)))
    embds_js_path="embds_5.json"
    rank_embds=np.array(JSON2Dict(embds_js_path)["rank_embds"])
    context_embds=np.array(JSON2Dict(embds_js_path)["context_embds"])
    acc_anchors=[[1-math.pow(0.5,i),1.0] for i in range(m)]
    all_cell_pths=[]

    ns101_cell_pths_list=SamplingBatches("exps_setting/exp_setting_ns101_"+str(budget)+".json",nas101_cell_dir,budget,5,False)
    ns201_cell_pths_list=SamplingBatches("exps_setting/exp_setting_ns201_"+str(budget)+".json",nas201_cell_dir,budget,5,False)
    ns301_cell_pths_list=SamplingBatches("exps_setting/exp_setting_ns301_"+str(budget)+".json",nas301_cell_dir,budget,5,False)

    ns101_cell_pths=ns101_cell_pths_list[0]
    ns201_cell_pths=ns201_cell_pths_list[0]
    ns301_cell_pths=ns301_cell_pths_list[0]
    all_cell_pths=ns101_cell_pths+ns201_cell_pths+ns301_cell_pths
    random.shuffle(all_cell_pths)
    data_gen=DataGenerator(all_cell_pths,acc_anchors,embds_js_path)
    dg=data_gen.Gen()
    
    model=CreateCLUENAS(acc_anchors)
    model=CompileCLUENAS(model,0.01)
    Training(model,dg,epochs=100,step_per_epoch=iters(len(all_cell_pths)))
    model=CompileCLUENAS(model,0.001)
    Training(model,dg,epochs=100,step_per_epoch=iters(len(all_cell_pths)))
    if(predict_space=="nas101"):
        acc_key="test_accuracy_108"
        cell_dir=nas101_cell_dir
    elif(predict_space=="nas201"):
        acc_key="test_accuracy_200"
        time_key="train_time_200"
        cell_dir=nas201_cell_dir
    elif(predict_space=="nas301"):
        acc_key="test_accuracy"
        cell_dir=nas301_cell_dir
    maxaccs_arr=[]
    costtimes_arr=[]
    for i in range(10):
        cell_pths=SamplingCellPths(cell_dir,k=1000)
        CLUENASPredictCellPths(model,rank_embds,context_embds,cell_pths)
        cell_pths=RankingCellPths(cell_pths)
        max_accs=[]
        cost_times=[]
        for k in range(20):
            max_acc=MaxAccByEstimate(cell_pths[:(k+1)*10],acc_key="test_accuracy_200",est_key="test_accuracy_12")
            cost_time=CostTime(cell_pths[:(k+1)*10],time_key="train_time_12")
            max_accs.append(max_acc)
            cost_times.append(cost_time)
        maxaccs_arr.append(max_accs)
        costtimes_arr.append(cost_times)
    maxaccs_arr=np.array(maxaccs_arr)
    maxaccs_arr=np.median(maxaccs_arr,axis=0)
    costtimes_arr=np.array(costtimes_arr)
    costtimes_arr=np.median(costtimes_arr,axis=0)
    print(maxaccs_arr)
    print(costtimes_arr)
    return maxaccs_arr.tolist(),costtimes_arr.tolist()


from json_io import Dict2JSON

result={}
result["nas201"]=TrainEval("nas201",budget=5)
Dict2JSON(result,"rnd_results.json")

print(result)