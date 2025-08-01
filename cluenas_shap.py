import numpy as np
import math
from clue_nas import CreateCLUENASBodyforShap,CreateCLUENASHeadforShap
from json_io import JSON2Dict
from nas_prcss import SamplingBatches,CellPth2Cell
import shap

def AnalyzeShap(whts_path="ns101_100_cluenas.hdf5"):
    nas101_cell_dir="fake_data/nasbench101_cifar10"
    nas201_cell_dir="fake_data/nasbench201_cifar10"
    nas301_cell_dir="fake_data/nasbench301_cifar10"
    acc_anchors=[[1-math.pow(0.5,i),1.0] for i in range(5)]
    embds_js_path="embds_5.json"
    rank_embds=JSON2Dict(embds_js_path)["rank_embds"]
    context_embds=JSON2Dict(embds_js_path)["context_embds"]


    ns101_cell_pths_list=SamplingBatches("exps_setting/exp_setting_ns101_100.json",nas101_cell_dir,100,5,False)
    ns201_cell_pths_list=SamplingBatches("exps_setting/exp_setting_ns201_100.json",nas201_cell_dir,100,5,False)
    ns301_cell_pths_list=SamplingBatches("exps_setting/exp_setting_ns301_100.json",nas301_cell_dir,100,5,False)
    ns101_cell_pths=ns101_cell_pths_list[0]
    ns201_cell_pths=ns201_cell_pths_list[0]
    ns301_cell_pths=ns301_cell_pths_list[0]

    graph_shaps=[]
    context_shaps=[]
    for cell_pths in [ns101_cell_pths,ns201_cell_pths,ns301_cell_pths]:
        body_model=CreateCLUENASBodyforShap(acc_anchors,whts_path)
        head_model=CreateCLUENASHeadforShap(acc_anchors,whts_path)
        backgounds=[]
        for cell_pth in cell_pths:
            cell=CellPth2Cell(cell_pth,True)
            adj_mat=np.array(cell["adj_matrix"])
            ops_mat=np.array(cell["operations"])
            ctxt_tkn=np.array(cell["ctxt_embds"])
            backgound=body_model.predict_on_batch((np.array([adj_mat]),np.array([ops_mat]),np.array([ctxt_tkn]),np.array([rank_embds]),np.array([context_embds])))
            backgounds.append(backgound[0])
        backgounds=np.array(backgounds)
        explans=backgounds[64:96]
        backgounds=backgounds[:64]
        explainer=shap.DeepExplainer(head_model,backgounds)  # background
        shap_values=explainer.shap_values(explans)  # 要解釋的資料
        shap_values=np.abs(shap_values)
        shap_values=np.mean(shap_values,axis=0)
        shap_values=np.sum(shap_values,axis=-1)
        graph_shap=np.sum(shap_values[:128])
        context_shap=np.sum(shap_values[128:])
        graph_shaps.append(graph_shap)
        context_shaps.append(context_shap)
    return graph_shaps,context_shaps

graph_shaps,context_shaps=AnalyzeShap("fake_ns101_100_cluenas.hdf5")
print(graph_shaps,context_shaps)