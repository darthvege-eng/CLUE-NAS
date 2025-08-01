from json_io import Dict2JSON,JSON2Dict
from encoding_ops import UnifyOPs
import os

FakeOpsDict={}
FakeOpsDict["conv1x1"]="op1"
FakeOpsDict["conv3x3"]="op2"
FakeOpsDict["conv5x5"]="op3"
FakeOpsDict["dilconv3x3"]="op4"
FakeOpsDict["dilconv5x5"]="op5"
FakeOpsDict["identity"]="op6"
FakeOpsDict["none"]="op7"
FakeOpsDict["pooling"]="op8"
FakeOpsDict["sepconv3x3"]="op9"
FakeOpsDict["sepconv5x5"]="op10"


def CreateFakeOps():
    all_js=os.listdir("ops_embds")
    for js_name in all_js:
        op_name=js_name.split(".")[0]
        op_dict=JSON2Dict("ops_embds/"+js_name)
        op_dict["operation"]=FakeOpsDict[op_name]
        Dict2JSON(op_dict,"ops_embds/"+op_dict["operation"]+".json")
    return 

def CreateFakeNasBench(in_dir,out_dir):
    all_js=os.listdir(in_dir)
    for js_name in all_js:
        in_path=in_dir+"/"+js_name
        out_path=out_dir+"/"+js_name
        cell_dict=JSON2Dict(in_path)
        fake_ops=[]
        for op in UnifyOPs(cell_dict["norm_operations"]):
            fake_op=FakeOpsDict[op]
            fake_ops.append(fake_op)
        cell_dict["norm_operations"]=fake_ops
        fake_ops=[]
        for op in UnifyOPs(cell_dict["rdce_operations"]):
            fake_op=FakeOpsDict[op]
            fake_ops.append(fake_op)
        cell_dict["rdce_operations"]=fake_ops
        Dict2JSON(cell_dict,out_path)
    return

# CreateFakeNasBench("data/nasbench101_cifar10","fake_data/nasbench101_cifar10")
        


