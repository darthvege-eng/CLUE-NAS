from nas_prcss import SamplingCellPths,CellPthsPredicting,RankingCellPths
import random

class RndSampler:
    def __init__(self,cell_pths):
        self._cell_pths=cell_pths
    def __call__(self,sample_count):
        return self.Sampling(sample_count)
    def Sampling(self,sample_count):
        cell_pths=self._cell_pths.copy()
        random.shuffle(cell_pths)
        cell_pths=cell_pths[:sample_count]
        return cell_pths

class TopSampler:
    def __init__(self,cell_pths,act_cell_pths=[]):
        self._cell_pths=cell_pths
        self._act_cell_pths=act_cell_pths
    def __call__(self,pred_function,sample_count):
        return self.Sampling(pred_function,sample_count)
    def _RemoveRepeats(self,cell_pths):
        _cell_pths=[]
        for cell_pth in cell_pths:
            if(cell_pth not in self._act_cell_pths):
                _cell_pths.append(cell_pth)
        return _cell_pths
    def Sampling(self,pred_function,sample_count):
        smpld_cell_pths=[]
        while(sample_count>len(smpld_cell_pths)):
            part_cell_pths=self._cell_pths.copy()
            random.shuffle(part_cell_pths)
            part_cell_pths=part_cell_pths[:1000]
            pred_function(part_cell_pths)
            part_cell_pths=RankingCellPths(part_cell_pths)
            part_cell_pths=part_cell_pths[:sample_count*2]
            random.shuffle(part_cell_pths)
            top_cell_pths=part_cell_pths[:sample_count]
            top_cell_pths=self._RemoveRepeats(top_cell_pths)
            smpld_cell_pths+=top_cell_pths
        return smpld_cell_pths[:sample_count]

class DSSampler:
    def __init__(self,cells_dir,max_nodes,all_ops,read_type="gt",cell_pth_type="nas201"):
        self._data_dir=cells_dir
        self._max_nodes=max_nodes
        self._all_ops=all_ops
        self._read_type=read_type
        self._cell_pth_type=cell_pth_type
        self._all_cell_pths=RankingCellPths(SamplingCellPths(cells_dir,k=50000),"flops")
        self._orig_part_idxs=[0,len(self._all_cell_pths)-1]
        self._cur_part_idxs=[0,len(self._all_cell_pths)-1]
        self._recover_dict={}
        self._recover_dict[str(self._cur_part_idxs)]=self._cur_part_idxs
        self._recover_count=0
        self._init=True
        self._alpha=1.0
    def __call__(self,predictor,sample_count,update_best=True):
        return self.Sampling(predictor,sample_count,update_best)
    def _SplitFLOPsPart(self,part_idxs):
        begin_idx,end_idx=part_idxs
        begin_idx_1=begin_idx
        end_idx_1=begin_idx+int((end_idx-begin_idx)*0.75)
        begin_idx_2=begin_idx+int((end_idx-begin_idx)*0.25)
        end_idx_2=end_idx
        part_idxs_1=[begin_idx_1,end_idx_1]
        part_idxs_2=[begin_idx_2,end_idx_2]
        self._recover_dict[str(part_idxs_1)]=part_idxs
        self._recover_dict[str(part_idxs_2)]=part_idxs
        return part_idxs_1,part_idxs_2
    def _UpdatePartIdxs(self,all_act_pths,part_idx_1,part_idx_2):
        part_1_count=0
        part_2_count=0
        for cell_pth in all_act_pths:
            if (cell_pth in self._all_cell_pths[part_idx_1[0]:part_idx_1[1]]):
                part_1_count+=1
            if(cell_pth in self._all_cell_pths[part_idx_2[0]:part_idx_2[1]]):
                part_2_count+=1
        if(part_1_count-part_2_count>len(all_act_pths)*0.75):
            self._recover_count=0
            self._cur_part_idxs=part_idx_1
        elif(part_2_count-part_1_count>len(all_act_pths)*0.75):
            self._recover_count=0
            self._cur_part_idxs=part_idx_2
        else:
            self._recover_count+=1
            for i in range(self._recover_count):
                self._cur_part_idxs=self._recover_dict[str(self._cur_part_idxs)]
        
        if((self._cur_part_idxs[1]-self._cur_part_idxs[0])<1000):
            self._cur_part_idxs=self._recover_dict[str(self._cur_part_idxs)]
        return 
    def Sampling(self,predictor,sample_count,update_best=True):
        part_cell_pths=self._all_cell_pths[self._cur_part_idxs[0]:self._cur_part_idxs[1]]
        random.shuffle(part_cell_pths)
        part_cell_pths=part_cell_pths[:1000]

        # if(self._init==True):
        #     CellPthsPredicting(part_cell_pths,predictor,self._all_ops,self._max_nodes,self._cell_pth_type)
        #     self._init=False

        CellPthsPredicting(part_cell_pths,predictor,self._all_ops,self._max_nodes,self._cell_pth_type)

        ranked_cell_pths=RankingCellPths(part_cell_pths)
        part_idx_1,part_idx_2=self._SplitFLOPsPart(self._cur_part_idxs)
        self._UpdatePartIdxs(ranked_cell_pths[:int(len(part_cell_pths)*0.1)],part_idx_1,part_idx_2)
        batch_cell_pths=self._all_cell_pths[self._cur_part_idxs[0]:self._cur_part_idxs[1]]

        random.shuffle(batch_cell_pths)
        batch_cell_pths=batch_cell_pths[:1000]

        CellPthsPredicting(batch_cell_pths,predictor,self._all_ops,self._max_nodes,self._cell_pth_type)
        batch_cell_pths=RankingCellPths(batch_cell_pths)

        topk=sample_count*2
        top_cell_pths=[]
        while(topk>0):
            _top_cell_pths=self._cells_pool.CheckPths(batch_cell_pths[:topk])
            top_cell_pths+=_top_cell_pths
            del batch_cell_pths[:topk]
            topk=topk-len(_top_cell_pths)
        random.shuffle(top_cell_pths)
        top_cell_pths=top_cell_pths[:sample_count]

        ##############
        self._cells_pool.AppendPths(top_cell_pths,self._read_type)
        if(update_best==True):
            self._cells_pool.UpdateBestAcc(self._read_type)
        ##############
        return