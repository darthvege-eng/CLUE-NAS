import clip
import numpy as np
from nas_prcss import CellPth2Cell,SamplingCellPths
from json_io import Dict2JSON,JSON2Dict
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

# cell_pths=SamplingCellPths("fake_data/nasbench201_cifar10")
# CellPths2Context(cell_pths)
# cell_pths=SamplingCellPths("fake_data/nasbench101_cifar10")
# CellPths2Context(cell_pths)

