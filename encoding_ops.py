import tensorflow as tf
import numpy as np
import time
import os
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from modules import ConvBN,DilConvBN,SepConvBN
from json_io import Dict2JSON,JSON2Dict


def AgentModel(op):
    in_ts=tf.keras.Input(shape=(32,32,32))
    out_ts=op(in_ts)
    model=tf.keras.Model(inputs=in_ts,outputs=out_ts)
    return model

def FLOPs(model,scale_factor=10**7):
    forward_pass=tf.function(model.call,input_signature=[tf.TensorSpec(shape=(1,)+model.input_shape[1:])])
    graph_info=profile(forward_pass.get_concrete_function().graph,options=ProfileOptionBuilder.float_operation())
    flops=graph_info.total_float_ops//2
    return flops/scale_factor

def Params(model,scale_factor=10**4):
    tot_whts_count=0
    for whts in model.weights:
        tot_whts_count+=tf.keras.backend.count_params(whts)
    return tot_whts_count/scale_factor

def Latency(model,repeat=10,scale_factor=10**(-1)):
    latencies=[]
    for i in range(repeat):
        indata=np.ones([128,32,32,32])
        bts=time.time()
        model(indata)
        model.predict(indata)
        ets=time.time()
        latency=ets-bts
        latencies.append(latency)
    latency=sum(latencies)/repeat
    return latency/scale_factor

def DataSensitive(model,repeat=10,scale_factor=10**(-1)):
    indata=np.ones([128,32,32,32])
    outdata=model.predict(indata)

    data_sensitives=[]
    for i in range(repeat):
        indata_rnd=indata+np.random.rand(128,32,32,32)
        outdata_rnd=model.predict(indata_rnd)
        data_sensitive=np.abs(outdata_rnd-outdata)
        data_sensitive=np.mean(data_sensitive,axis=(1,2,3))
        data_sensitive=np.mean(data_sensitive)
        data_sensitives.append(data_sensitive)
    data_sensitive=sum(data_sensitives)/repeat
    return data_sensitive/scale_factor

def ConvOP(kernel_size):
    model=AgentModel(ConvBN(32,(kernel_size,kernel_size)))
    flops=FLOPs(model)
    params=Params(model)
    latency=Latency(model)
    data_sensitive=DataSensitive(model)
    embd_dict={}
    embd_dict["operation"]="conv"+str(kernel_size)+"x"+str(kernel_size)
    embd_dict["flops"]=flops
    embd_dict["params"]=params
    embd_dict["latency"]=latency
    embd_dict["sensitive"]=data_sensitive
    embd_dict["passing"]=1.0 #passing rate
    Dict2JSON(embd_dict,"ops_embds/"+embd_dict["operation"]+".json")
    return

def DilConvOP(kernel_size):
    model=AgentModel(DilConvBN(32,(kernel_size,kernel_size)))
    flops=FLOPs(model)
    params=Params(model)
    latency=Latency(model)
    data_sensitive=DataSensitive(model)
    embd_dict={}
    embd_dict["operation"]="dilconv"+str(kernel_size)+"x"+str(kernel_size)
    embd_dict["flops"]=flops
    embd_dict["params"]=params
    embd_dict["latency"]=latency
    embd_dict["sensitive"]=data_sensitive
    embd_dict["passing"]=1.0 #passing rate
    Dict2JSON(embd_dict,"ops_embds/"+embd_dict["operation"]+".json")
    return

def SepConvOP(kernel_size):
    model=AgentModel(SepConvBN(32,(kernel_size,kernel_size)))
    flops=FLOPs(model)
    params=Params(model)
    latency=Latency(model)
    data_sensitive=DataSensitive(model)
    embd_dict={}
    embd_dict["operation"]="sepconv"+str(kernel_size)+"x"+str(kernel_size)
    embd_dict["flops"]=flops
    embd_dict["params"]=params
    embd_dict["latency"]=latency
    embd_dict["sensitive"]=data_sensitive
    embd_dict["passing"]=1.0 #passing rate
    Dict2JSON(embd_dict,"ops_embds/"+embd_dict["operation"]+".json")
    return

def OtherOP(op_name,passing_rate=0):
    embd_dict={}
    embd_dict["operation"]=op_name
    embd_dict["flops"]=0
    embd_dict["params"]=0
    embd_dict["latency"]=0
    embd_dict["sensitive"]=0
    embd_dict["passing"]=passing_rate
    Dict2JSON(embd_dict,"ops_embds/"+embd_dict["operation"]+".json")
    return

def EmbdsDict(js_dir="ops_embds"):
    embds_dict={}
    keys=["flops","params","latency","sensitive","passing"]
    all_js=os.listdir(js_dir)
    for js_name in all_js:
        js_path=js_dir+"/"+js_name
        js_dict=JSON2Dict(js_path)
        embd=[]
        for key in keys:
            embd.append(js_dict[key])
        embds_dict[js_dict["operation"]]=embd
    return embds_dict

def UnifyOPs(operations):
    trans_dict={}
    trans_dict["identity"]=["identity","input","output","skip_connect"]
    trans_dict["none"]=["none","void","zeros"]
    trans_dict["conv1x1"]=["conv1x1","nor_conv_1x1","conv1x1-bn-relu"]
    trans_dict["conv3x3"]=["conv3x3","nor_conv_3x3","conv3x3-bn-relu"]
    trans_dict["sepconv3x3"]=["sepconv3x3","sep_conv_3x3"]
    trans_dict["sepconv5x5"]=["sepconv5x5","sep_conv_5x5"]
    trans_dict["dilconv3x3"]=["dilconv3x3","dil_conv_3x3"]
    trans_dict["dilconv5x5"]=["dilconv5x5","dil_conv_5x5"]
    trans_dict["pooling"]=["pooling","avg_pool_3x3","max_pool_3x3","maxpool3x3"]
    u_operations=[]
    for operation in operations:
        check=False
        for key in list(trans_dict.keys()):
            if(operation in trans_dict[key]):
                u_operations.append(key)
                check=True
                break
        if(check==False):u_operations.append(operation)
    return u_operations

def EncodingOPs(operations,embds_dict,embds_len=128):
    operations=UnifyOPs(operations)
    out_embds=[]
    for operation in operations:
        out_embds+=embds_dict[operation]
    for i in range(embds_len-len(out_embds)):
        out_embds.append(0)
    return out_embds