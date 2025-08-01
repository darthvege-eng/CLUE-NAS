import numpy as np
from json_io import Dict2JSON
from nas_prcss import CellPth2Cell,SamplingCellPths
from text_encoding import LoadTextEncoder

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
# CellPths2ContextEmbds(text_encoder,SamplingCellPths("fake_data/nasbench101_cifar10"))
# CellPths2ContextEmbds(text_encoder,SamplingCellPths("fake_data/nasbench201_cifar10"))
# CellPths2ContextEmbds(text_encoder,SamplingCellPths("fake_data/nasbench301_cifar10"))