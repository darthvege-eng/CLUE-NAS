import numpy as np
import clip
from json_io import Dict2JSON
from nas_prcss import CellPth2Cell,SamplingCellPths
from text_encoding import LoadTextEncoder
from encoding_ops import UnifyOPs

def Cell2Context(cell):
    if(cell["nas_type"]=="nas301"):
        ops=cell["norm_operations"]
    else:
        ops=cell["operations"]
    u_ops=UnifyOPs(ops)
    context="The cell has "+str(len(ops))+"nodes: "+str(u_ops)
    ctxt_tkn=clip.tokenize(context,truncate=True,context_length=77)
    cell["context"]=context
    cell["ctxt_tkn"]=ctxt_tkn[0].numpy().tolist()
    return cell

def CellPth2Context(cell_pth):
    cell=CellPth2Cell(cell_pth)
    cell=Cell2Context(cell)
    Dict2JSON(cell,cell_pth)
    return

def CellPths2Context(cell_pths):
    for i,cell_pth in enumerate(cell_pths):
        print((i+1)/len(cell_pths))
        CellPth2Context(cell_pth)
    return

def CellPths2ContextEmbds(text_encoder,cell_pths,overwrite=False):
    for i,cell_pth in enumerate(cell_pths):
        print(i)
        cell=CellPth2Cell(cell_pth)
        if("ctxt_embds" in cell and overwrite==False):continue
        embds=text_encoder.predict_on_batch(np.array([cell["ctxt_tkn"]]))[0]
        cell["ctxt_embds"]=embds.tolist()
        Dict2JSON(cell,cell_pth)
    return


# text_encoder=LoadTextEncoder()

# CellPths2Context(SamplingCellPths("data/nasbench101_cifar10"))
# CellPths2Context(SamplingCellPths("data/nasbench201_cifar10"))
# CellPths2Context(SamplingCellPths("data/nasbench301_cifar10"))

# CellPths2ContextEmbds(text_encoder,SamplingCellPths("data/nasbench101_cifar10"))
# CellPths2ContextEmbds(text_encoder,SamplingCellPths("data/nasbench201_cifar10"))
# CellPths2ContextEmbds(text_encoder,SamplingCellPths("data/nasbench301_cifar10"))