
import numpy as np

def EncodingADJ(adj,embds_len=320):
    adj=np.array(adj)
    adj=(adj+np.transpose(adj))
    embds=np.reshape(adj,[-1])
    zero_pads=np.zeros([embds_len-len(embds)])
    embds=np.concatenate([embds,zero_pads],axis=0)
    return embds
