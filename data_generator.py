import numpy as np
import random
from nas_prcss import CellPth2Cell
from json_io import JSON2Dict,Dict2JSON

class DataGenerator:
    def __init__(self,cell_pths=[],acc_anchors=[[0,1.0],[0.5,1.0],[0.75,1.0],[0.875,1.0],[0.925,1.0]],embds_js_path="embds.json"):
        self._cell_pths_dict={}
        self._cell_pths=self._Preprocess(cell_pths)
        self._acc_anchors=acc_anchors
        self._acc_anchors_len=len(acc_anchors)
        rank_embds=JSON2Dict(embds_js_path)["rank_embds"]
        rank_embds=rank_embds+[rank_embds[-1],rank_embds[-1]]
        context_embds=JSON2Dict(embds_js_path)["context_embds"]
        context_embds=context_embds+[context_embds[-1],context_embds[-1]]
        self._rank_embds=np.array(rank_embds)[:self._acc_anchors_len]
        self._context_embds=np.array(context_embds)[:self._acc_anchors_len]
    def _Preprocess(self,cell_pths):
        for cell_pth in cell_pths:
            cell_dict=CellPth2Cell(cell_pth)
            if(cell_dict["nas_type"]=="nas301"):
                acc=cell_dict["test_accuracy"]
            elif(cell_dict["nas_type"]=="nas201"):
                acc=cell_dict["test_accuracy_200"]
            elif(cell_dict["nas_type"]=="nas101"):
                acc=cell_dict["test_accuracy_108"]
            cell_dict["test_accuracy_buf"]=acc
            Dict2JSON(cell_dict,cell_pth)
            if(cell_dict["nas_type"] not in self._cell_pths_dict.keys()):
                self._cell_pths_dict[cell_dict["nas_type"]]=[]
            self._cell_pths_dict[cell_dict["nas_type"]].append(cell_pth)
        return cell_pths
    def _EncodingY(self,accuracy):
        gt_whts=[]
        gt_sims=[]
        gt_offsets=[]
        for i in range(self._acc_anchors_len):
            if(self._acc_anchors[i][0]<=accuracy and self._acc_anchors[i][1]>accuracy):
                acc_dist=(accuracy-self._acc_anchors[i][0])/(self._acc_anchors[i][1]-self._acc_anchors[i][0])
                gt_offsets.append(acc_dist)
                gt_whts.append(1-acc_dist)
                gt_sims.append(1)
            else:
                gt_offsets.append(0)
                gt_whts.append(0)
                gt_sims.append(0)
        gt_whts=np.array(gt_whts)
        gt_whts=(gt_whts-np.min(gt_whts))/(np.max(gt_whts)-np.min(gt_whts))
        return np.array([gt_sims,gt_offsets,gt_whts])
    def GetCellPths(self):
        return self._cell_pths
    def GetTypeCellPths(self,nas_type):
        return self._cell_pths_dict[nas_type]
    def AppendCellPths(self,cell_pths):
        self._Preprocess(cell_pths)
        self._cell_pths=self._cell_pths+cell_pths
        return self._cell_pths
    def Read(self,batch_size=16):
        act_cell_pths=random.choices(self._cell_pths,k=batch_size)
        adj_matrix_list=[]
        op_matrix_list=[]
        ctxt_embds_list=[]
        rank_embds_list=[]
        context_embds_list=[]
        y_list=[]
        for i,cell_path in enumerate(act_cell_pths):
            cell_dict=CellPth2Cell(cell_path,preprcss=True)
            adj_mat=np.array(cell_dict["adj_matrix"])
            ops_mat=np.array(cell_dict["operations"])
            ctxt_embds=np.array(cell_dict["ctxt_embds"])
            adj_matrix_list.append(adj_mat)
            op_matrix_list.append(ops_mat)
            ctxt_embds_list.append(ctxt_embds)
            rank_embds_list.append(self._rank_embds)
            context_embds_list.append(self._context_embds)
            acc=cell_dict["test_accuracy_buf"]
            y_list.append(self._EncodingY(acc))
        output_xy=(np.array(adj_matrix_list),np.array(op_matrix_list),np.array(ctxt_embds_list),np.array(rank_embds_list),np.array(context_embds_list)),np.array(y_list)
        return output_xy
    def Gen(self,batch_size=16):
        while(1):
            yield self.Read(batch_size)