import os
import numpy as np
import random
from json_io import Dict2JSON,JSON2Dict
from encoding_ops import EmbdsDict,EncodingOPs
from encoding_adj import EncodingADJ
EMBDSDICT=EmbdsDict()


def TransNas201ADJ(adj):
    adj_m=np.zeros([8,8])
    connections=[[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[3,6],[4,6],[5,6],[6,7]]
    for connection in connections:
        start_idx,end_idx=connection
        adj_m[end_idx][start_idx]=1
    return adj_m.tolist()

def TransNas301ADJ(adj):
    adj_m=np.zeros([17,17])
    connections=[[0,2],[0,4],[0,7],[0,11],
                 [1,3],[1,5],[1,8],[1,12],
                 [2,6],[2,9],[2,13],[3,6],[3,9],[3,13],
                 [4,10],[4,14],[5,10],[5,14],[6,10],[6,14],
                 [7,15],[8,15],[9,15],[10,15],
                 [11,16],[12,16],[13,16],[14,16],[15,16]]
    for connection in connections:
        start_idx,end_idx=connection
        adj_m[end_idx][start_idx]=1
    return adj_m.tolist()


def CellPth2Cell(cell_pth,preprcss=False):
    global EMBDSDICT
    cell=JSON2Dict(cell_pth)
    if(preprcss==True and cell["nas_type"]=="nas101"):
        cell["adj_matrix"]=cell["adj_matrix"]
        cell["operations"]=cell["operations"]
        cell["adj_matrix"]=EncodingADJ(cell["adj_matrix"])
        cell["operations"]=EncodingOPs(cell["operations"],EMBDSDICT)
    elif(preprcss==True and cell["nas_type"]=="nas201"):
        cell["adj_matrix"]=TransNas201ADJ(cell["adj_matrix"])
        cell["operations"]=cell["operations"]
        cell["adj_matrix"]=EncodingADJ(cell["adj_matrix"])
        cell["operations"]=EncodingOPs(cell["operations"],EMBDSDICT)
    elif(preprcss==True and cell["nas_type"]=="nas301"):
        cell["adj_matrix"]=TransNas301ADJ(cell["norm_adj_matrix"])
        cell["operations"]=cell["norm_operations"]
        cell["adj_matrix"]=EncodingADJ(cell["adj_matrix"])
        cell["operations"]=EncodingOPs(cell["operations"],EMBDSDICT)
    return cell

def CellPths2Cells(cell_pths,preprcss=False):
    return list(map(lambda x:CellPth2Cell(x,preprcss),cell_pths))

def CellPthInit(cell_pth):
    cell=CellPth2Cell(cell_pth)
    cell["pred_accuracy"]=-1
    cell["confidence"]=1
    Dict2JSON(cell,cell_pth)
    return cell_pth

def CellPthsInit(cell_pths):
    return list(map(lambda x:CellPthInit(x),cell_pths))

def CellPthPredicting(cell_pth,predictor):
    cell=CellPth2Cell(cell_pth,preprcss=True)
    preds=predictor.predict_on_batch((np.array([cell["adj_matrix"]]),np.array([cell["operations"]])))
    pred_acc=preds[0][0]
    cell=CellPth2Cell(cell_pth,preprcss=False)
    cell["pred_accuracy"]=float(pred_acc)
    Dict2JSON(cell,cell_pth)
    return

def CellPredicting(cell,predictor):
    preds=predictor.predict_on_batch((np.array([cell["adj_matrix"]]),np.array([cell["operations"]])))
    pred_acc=preds[0][0]
    cell["pred_accuracy"]=float(pred_acc)
    return cell

def CellPthsPredicting(cell_pths,predictor,cell_pth_type="nas201"):
    for cell_pth in cell_pths:
        CellPthPredicting(cell_pth,predictor,cell_pth_type)
    return

def CellsPredicting(cells,predictor):
    for cell in cells:
        CellPredicting(cell,predictor)
    return

def RankingCellPths(cell_pths,rank_key="pred_accuracy",reverse=True):
    cell_pths=cell_pths.copy()
    ranking_cell_pths=[]
    for cell_path in cell_pths:
        cell=CellPth2Cell(cell_path)
        metric=cell[rank_key]
        ranking_cell_pths.append([cell_path,metric])
    ranking_cell_pths=sorted(ranking_cell_pths,key=lambda x:x[1],reverse=reverse)
    ranking_cell_pths=list(map(lambda x:x[0],ranking_cell_pths))
    return ranking_cell_pths

def RankingCells(cells,rank_key="pred_accuracy",reverse=True):
    ranking_cells=[]
    for cell in cells:
        accuracy=cell[rank_key]
        ranking_cells.append([cell,accuracy])
    ranking_cells=sorted(ranking_cells,key=lambda x:x[1],reverse=reverse)
    ranking_cells=list(map(lambda x:x[0],ranking_cells))
    return ranking_cells

def SamplingCellPths(cells_dir,k=-1,shuffle=True):
    cell_pths=[]
    all_cells=os.listdir(cells_dir)
    if(shuffle==True):random.shuffle(all_cells)
    if(k==-1):k=len(all_cells)
    act_count=0
    for cell_name in all_cells:
        cell_path=cells_dir+"/"+cell_name
        if(os.path.isfile(cell_path)!=True):continue
        cell_pths.append(cell_path)
        act_count+=1
        if(act_count==k):break
    return cell_pths

def PartialSamplingCellPths(cells_dir,k=-1):
    if(k==-1):return SamplingCellPths(cells_dir,k=-1)
    cell_pths=SamplingCellPths(cells_dir,k=max(k,16000))
    cell_pths=RankingCellPths(cell_pths,"flops")
    partial_len=int(len(cell_pths)/k)
    chosen_pths=[]
    start_idx=0
    for i in range(k):
        end_idx=start_idx+partial_len
        if(i==k-1):end_idx=len(cell_pths)-1
        batch_cell_pths=cell_pths[start_idx:end_idx]
        chosen_pths.append(random.choice(batch_cell_pths))
        start_idx=end_idx
    return chosen_pths

def FilteringCellPths(cell_pths,key=None):
    if(key==None):return cell_pths
    _cell_pths=[]
    for cell_pth in cell_pths:
        cell=CellPth2Cell(cell_pth)
        if(key in cell):_cell_pths.append(cell_pth)
    return _cell_pths

def SamplingBatches(js_path,cells_dir,batch_size=150,batches_num=10,overwrite=False):
    if(overwrite==True or os.path.isfile(js_path)==False):
        btchs_dict={}
        btchs_list=[]
        for i in range(batches_num):
            btch_cell_pths=SamplingCellPths(cells_dir,k=batch_size)
            btchs_list.append(btch_cell_pths)
            btchs_dict[i]=btch_cell_pths
        Dict2JSON(btchs_dict,js_path)
    else:
        btchs_list=[]
        btchs_dict=JSON2Dict(js_path)
        for i in range(batches_num):
            btchs_list.append(btchs_dict[str(i)])
    return btchs_list

def MaxEsitmate(cell_pths,est_key="pred_accuracy"):
    ests=[]
    for cell_pth in cell_pths:
        cell=CellPth2Cell(cell_pth)
        ests.append(cell[est_key])
    return max(ests)

def MaxAccByEstimate(cell_pths,acc_key="test_accuracy_200",est_key="pred_accuracy"):
    cell_pths=RankingCellPths(cell_pths,est_key)
    cell=CellPth2Cell(cell_pths[0])
    return cell[acc_key]

def CostTime(cell_pths,time_key="train_time_200"):
    tot_time=0
    for cell_pth in cell_pths:
        cell=CellPth2Cell(cell_pth)
        tot_time+=cell[time_key]
    return tot_time

def ID2RankDict(cell_pths,est_key="pred_accuracy",reverse=False):
    id2rank={}
    cell_pths=RankingCellPths(cell_pths,est_key,reverse=reverse)
    for i,cell_pth in enumerate(cell_pths):
        cell=CellPth2Cell(cell_pth)
        id2rank[cell["id"]]=i
    return id2rank