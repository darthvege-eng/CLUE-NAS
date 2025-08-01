import numpy as np
import math
from data_generator import DataGenerator
from clue_nas import CreateCLUENAS,CompileCLUENAS,CLUENASPredictCellPths
from model_operation import Training
from json_io import JSON2Dict,Dict2JSON
from eval_cell import CellPths2Psp
from nas_prcss import SamplingBatches

def TrainEval(budget=100,benchmarks=["nas101","nas201","nas301"]):
    nas101_cell_dir="data/nasbench101_cifar10"
    nas201_cell_dir="data/nasbench201_cifar10"
    nas301_cell_dir="data/nasbench301_cifar10"
    batch_size=16
    iters=(lambda budget: max(1,round(budget/batch_size)))
    repeat=5
    m=5
    # acc_anchors=[[0,1.0] for i in range(m)]
    acc_anchors=[[1-math.pow(0.5,i),1.0] for i in range(m)]
    embds_js_path="embds_5.json"
    rank_embds=JSON2Dict(embds_js_path)["rank_embds"]
    context_embds=JSON2Dict(embds_js_path)["context_embds"]


    ns101_cell_pths_list=SamplingBatches("exps_setting/exp_setting_ns101_"+str(budget)+".json",nas101_cell_dir,budget,repeat,False)
    ns201_cell_pths_list=SamplingBatches("exps_setting/exp_setting_ns201_"+str(budget)+".json",nas201_cell_dir,budget,repeat,False)
    ns301_cell_pths_list=SamplingBatches("exps_setting/exp_setting_ns301_"+str(budget)+".json",nas301_cell_dir,budget,repeat,False)
    eval_ns101_cell_pths=SamplingBatches("exps_setting/eval_setting_ns101.json",nas101_cell_dir,1000,1,True)[0]
    eval_ns201_cell_pths=SamplingBatches("exps_setting/eval_setting_ns201.json",nas201_cell_dir,1000,1,False)[0]
    eval_ns301_cell_pths=SamplingBatches("exps_setting/eval_setting_ns301.json",nas301_cell_dir,1000,1,False)[0]

    ns101_psps=[]
    ns201_psps=[]
    ns301_psps=[]
    for i,ns301_cell_pths in enumerate(ns301_cell_pths_list):
        ns101_cell_pths=ns101_cell_pths_list[i]
        ns201_cell_pths=ns201_cell_pths_list[i]
        train_cell_pths=[]
        if("nas101" in benchmarks):
            train_cell_pths+=ns101_cell_pths
        if("nas201" in benchmarks):
            train_cell_pths+=ns201_cell_pths
        if("nas301" in benchmarks):
            train_cell_pths+=ns301_cell_pths
        data_gen=DataGenerator(train_cell_pths,acc_anchors,embds_js_path)
        dg=data_gen.Gen()
        
        model=CreateCLUENAS(acc_anchors)
        model=CompileCLUENAS(model,0.01)
        Training(model,dg,epochs=100,step_per_epoch=iters(len(train_cell_pths)))
        model=CompileCLUENAS(model,0.001)
        Training(model,dg,epochs=100,step_per_epoch=iters(len(train_cell_pths)))
        
        CLUENASPredictCellPths(model,rank_embds,context_embds,eval_ns101_cell_pths)
        ns101psp=CellPths2Psp(eval_ns101_cell_pths,"test_accuracy_108","pred_accuracy")
        CLUENASPredictCellPths(model,rank_embds,context_embds,eval_ns201_cell_pths)
        ns201psp=CellPths2Psp(eval_ns201_cell_pths,"test_accuracy_200","pred_accuracy")
        CLUENASPredictCellPths(model,rank_embds,context_embds,eval_ns301_cell_pths)
        ns301psp=CellPths2Psp(eval_ns301_cell_pths,"test_accuracy","pred_accuracy")
        ns101_psps.append(ns101psp)
        ns201_psps.append(ns201psp)
        ns301_psps.append(ns301psp)

    ns101_psp=np.mean(np.array(ns101_psps))
    ns201_psp=np.mean(np.array(ns201_psps))
    ns301_psp=np.mean(np.array(ns301_psps))

    print(ns101_psps)
    print(ns201_psps)
    print(ns301_psps)

    resulut_dict={}
    resulut_dict["budget"]=budget
    resulut_dict["training on"]=benchmarks
    resulut_dict["ns101_psp"]=ns101_psp
    resulut_dict["ns201_psp"]=ns201_psp
    resulut_dict["ns301_psp"]=ns301_psp

    save_path="exps_results/"+"".join(benchmarks)+"_"+str(budget)+".json"
    Dict2JSON(resulut_dict,save_path)
    return ns101_psp,ns201_psp,ns301_psp
    

# TrainEval(5,["nas101","nas201","nas301"])
# TrainEval(50,["nas101","nas201","nas301"])
TrainEval(100,["nas101","nas201","nas301"])

# TrainEval(5,["nas301"])
# TrainEval(50,["nas301"])


# TrainEval(100,["nas101"])
# TrainEval(100,["nas201"])
# TrainEval(100,["nas301"])
