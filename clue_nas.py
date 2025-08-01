import tensorflow as tf
import numpy as np
from modules import DenseBN,gelu
from nas_prcss import CellPth2Cell
from json_io import Dict2JSON,JSON2Dict

class CLIPFixedPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self,embds_js_path="clip_fixed_pstn_embd.json",name="clippstnembd"):
        super(CLIPFixedPositionalEmbedding,self).__init__(name=name)
        rank_embds=JSON2Dict(embds_js_path)["rank_embds"]
        context_embds=JSON2Dict(embds_js_path)["context_embds"]
        self._fix_pstn_embbd=np.array(rank_embds)+np.array(context_embds)
        self._name=name
    def build(self,in_shape):
        self._fix_pstn_embbd=tf.constant(self._fix_pstn_embbd,dtype=tf.float32)
        return
    def call(self,input_ts):
        return input_ts+self._fix_pstn_embbd
    
class FixedPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self,name="pstnembd"):
        super(FixedPositionalEmbedding,self).__init__(name=name)
        self._name=name
    def build(self,in_shape):
        _,seq_len,dims=in_shape
        positions=tf.range(seq_len,dtype=tf.float32)
        scale=1.0/seq_len
        positions=tf.expand_dims(positions,axis=-1)*scale
        fix_pstn_embbd=tf.tile(positions,[1,dims])
        self._fix_pstn_embbd=tf.constant(fix_pstn_embbd,dtype=tf.float32)
        return
    def call(self,input_ts):
        return input_ts+self._fix_pstn_embbd

class ArchsEncoder(tf.Module):
    def __init__(self,name="archsencoder"):
        super(ArchsEncoder,self).__init__(name=name)
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._flatten=tf.keras.layers.Flatten(name=self._name+"_flatten")
        self._res1=DenseBN(1024,activation=None,use_bn=False,name=self._name+"_res1")
        self._res2=DenseBN(1024,activation=None,use_bn=False,name=self._name+"_res2")
        self._fcx1=DenseBN(128,activation=gelu,name=self._name+"_fcx1")
        self._fcx2=DenseBN(128,activation=gelu,name=self._name+"_fcx2")
        self._fc_embd1=DenseBN(256,activation=gelu,name=self._name+"_fc_embd1")
        self._fc_embd2=DenseBN(1024,activation=None,use_bn=False,name=self._name+"_fc_embd2")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        _x1,_x2=input_ts
        _x1=self._flatten(_x1)
        _x2=self._flatten(_x2)
        res_x=self._res1(_x1)+self._res2(_x2)
        x1=self._fcx1(_x1)
        x2=self._fcx2(_x2)
        x=tf.concat([x1,x2],axis=-1)
        x=self._fc_embd1(x)
        embd_ts=self._fc_embd2(x)+res_x
        return embd_ts
    
class ContextAlignment(tf.Module):
    def __init__(self,name="contextalignment"):
        super(ContextAlignment,self).__init__(name=name)
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        # self._pstnembd=CLIPFixedPositionalEmbedding(name=self._name+"_pstnembd")
        self._pstnembd=FixedPositionalEmbedding(name=self._name+"_pstnembd")
        self._fc1=DenseBN(128,activation=None,name=self._name+"_fc1")
        self._fc2=DenseBN(128,activation=None,name=self._name+"_fc2")
    @tf.Module.with_name_scope
    def _Sim(self,cnfd_embds,archs_embd):
        archs_embd=tf.expand_dims(archs_embd,axis=-2)
        archs_embds=tf.tile(archs_embd,[1,tf.shape(cnfd_embds)[1],1])
        kl_dist=tf.keras.losses.kl_divergence(tf.nn.softmax(cnfd_embds/5),tf.nn.softmax(archs_embds/5))*25
        cnfd_embds=tf.nn.l2_normalize(cnfd_embds,axis=-1)
        archs_embds=tf.nn.l2_normalize(archs_embds,axis=-1)
        square_dist=(cnfd_embds-archs_embds)**2
        square_dist=tf.reduce_sum(square_dist,axis=-1)
        cos_sims=tf.reduce_sum(cnfd_embds*archs_embds,axis=-1)
        sims=tf.nn.sigmoid(cos_sims)
        embds_dist=kl_dist+square_dist
        return sims,embds_dist
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        archs_embd,context_embd=input_ts
        context_embd=tf.expand_dims(context_embd,axis=-2)
        context_embds=tf.tile(context_embd,[1,5,1]) #copy m=5
        context_embds=self._pstnembd(context_embds)
        embds_sims,embds_dist=self._Sim(context_embds,archs_embd)
        max_sim=tf.reduce_max(embds_sims,axis=-1,keepdims=True)
        chosen_mask=tf.cast(embds_sims>=max_sim,tf.float32)
        chosen_mask=tf.expand_dims(chosen_mask,axis=-1)
        chosen_context_embd=context_embds*chosen_mask
        chosen_context_embd=tf.reduce_sum(chosen_context_embd,axis=-2)
        archs_embd=self._fc1(archs_embd)
        chosen_context_embd=self._fc2(chosen_context_embd)
        fused_embd=tf.concat([archs_embd,chosen_context_embd],axis=-1)
        return fused_embd,embds_sims,embds_dist
    
#ContrastiveLearnableUnfyingEncoder
class CLUENAS(tf.Module):
    def __init__(self,acc_anchors,name="cluenas"):
        super(CLUENAS,self).__init__(name=name)
        self._anchors_len=len(acc_anchors)
        self._acc_anchors=np.array(acc_anchors)
        self._acc_bases=self._acc_anchors[...,0]
        self._scale_vals=self._acc_anchors[...,1]-self._acc_bases
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._archencoder=ArchsEncoder(name=self._name+"_archencoder")
        self._contextalignment=ContextAlignment(name=self._name+"_contextalignment")
        self._fc1=DenseBN(64,activation=gelu,name=self._name+"_fc1")
        self._fc2=DenseBN(32,activation=None,name=self._name+"_fc2")
        self._offset=DenseBN(self._anchors_len,activation=tf.nn.sigmoid,use_bn=False,name=self._name+"_offset")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        info_flow,metrics,context_embd=input_ts
        archs_embd=self._archencoder([info_flow,metrics])
        concat_embd,embds_sims,embds_dist=self._contextalignment([archs_embd,context_embd])
        fused_embd=self._fc1(concat_embd)
        fused_embd=self._fc2(fused_embd)
        pred_offset=self._offset(fused_embd)
        pred_acc=self._acc_bases+(pred_offset*self._scale_vals)
        pred_sims=tf.expand_dims(embds_sims,axis=-2)
        embds_dist=tf.expand_dims(embds_dist,axis=-2)
        pred_offset=tf.expand_dims(pred_offset,axis=-2)
        pred_acc=tf.expand_dims(pred_acc,axis=-2)
        out_ts=tf.concat([pred_sims,pred_offset,embds_dist,pred_acc],axis=-2)
        return fused_embd,out_ts
    
class CLUELoss(tf.Module):
    def __init__(self,name="clueloss"):
        super(CLUELoss,self).__init__(name)
        self._name=name
    def _SquareDist(self,square_dist,gt_whts):
        sqloss=square_dist*gt_whts
        sqloss=tf.reduce_sum(sqloss,axis=-1)
        return sqloss
    def _SimsLoss(self,gt_sims,pred_sims,gt_whts):
        sims_loss=tf.keras.backend.binary_crossentropy(gt_sims,pred_sims)
        sims_loss=tf.reduce_sum(sims_loss*gt_whts,axis=-1)
        return sims_loss
    def _OffsetLoss(self,gt_offsets,pred_offsets):
        offset_loss=tf.keras.backend.binary_crossentropy(gt_offsets,pred_offsets)
        offset_loss=tf.reduce_sum(offset_loss,axis=-1)
        return offset_loss
    def __call__(self):
        def _Loss(true_y,pred_y):
            gt_sims=true_y[...,0,:]
            gt_offsets=true_y[...,1,:]
            gt_whts=true_y[...,2,:]
            pred_sims=pred_y[...,0,:]
            pred_offsets=pred_y[...,1,:]
            kl_dist=pred_y[...,2,:]
            sqloss=self._SquareDist(kl_dist,gt_whts)
            sims_loss=self._SimsLoss(gt_sims,pred_sims,gt_whts)
            offset_loss=self._OffsetLoss(gt_offsets,pred_offsets)
            return sims_loss+sqloss+offset_loss
        return _Loss
    
def CreateCLUENAS(acc_anchors):
    info_flow=tf.keras.Input(shape=(320))
    metrics=tf.keras.Input(shape=(128))
    context_embd=tf.keras.Input(shape=(1024))
    _,preds=CLUENAS(acc_anchors)([info_flow,metrics,context_embd])
    model=tf.keras.Model(inputs=[info_flow,metrics,context_embd],outputs=preds)
    return model

def CompileCLUENAS(model,lr=0.01):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),loss=CLUELoss()())
    return model

def CLUENASPredictCellPth(model,cell_pth):
    cell=CellPth2Cell(cell_pth,True)
    adj_mat=np.array(cell["adj_matrix"])
    ops_mat=np.array(cell["operations"])
    ctxt_tkn=np.array(cell["ctxt_embds"])
    preds=model.predict_on_batch((np.array([adj_mat]),np.array([ops_mat]),np.array([ctxt_tkn])))
    pred_sims=preds[0][0]
    pred_accs=preds[0][-1]
    max_sim_idx=np.argmax(pred_sims)
    pred_acc=pred_accs[max_sim_idx]
    cell=CellPth2Cell(cell_pth)
    cell["pred_accuracy"]=float(pred_acc)
    Dict2JSON(cell,cell_pth)
    return pred_acc

def CLUENASPredictCellPths(model,cell_pths):
    for i,cell_pth in enumerate(cell_pths):
        CLUENASPredictCellPth(model,cell_pth)
    return 