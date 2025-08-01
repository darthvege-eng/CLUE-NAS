import tensorflow as tf
import numpy as np

class QuickGELU(tf.keras.layers.Layer):
    def __init__(self,name="QuickGELU"):
        super(QuickGELU,self).__init__(name=name)
    def call(self,x:tf.Tensor):
        return x*tf.sigmoid(1.702 * x)
    
class LayerNorm(tf.keras.layers.LayerNormalization):
    def __init__(self, name="LayerNorm"):
        super(LayerNorm,self).__init__(epsilon=1e-05, name=name)
    def call(self,in_ts):
        return super(LayerNorm,self).call(in_ts)

class ResidualAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, d_model: int, n_head: int, attn_mask: tf.Tensor = None, name="ResidualAttentionBlock", idx=0):
        super().__init__(name=name)
        self.idx = idx

        self.d_model = d_model
        self.n_head = n_head

        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=n_head, key_dim=d_model // n_head, name="attn")
        self.ln_1 = LayerNorm(name="ln_1")
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model * 4, name=name + "/mlp/c_fc"),
            QuickGELU(name=name + "/mlp/gelu"),
            tf.keras.layers.Dense(d_model, name=name + "/mlp/c_proj")
        ], name="mlp")
        self.ln_2 = LayerNorm(name="ln_2")
        self.attn_mask = attn_mask
    def attention(self, x):
        return self.attn(x, x, x, attention_mask=self.attn_mask)
    def get_config(self):
        return {
            "d_model": self.d_model,
            "n_head": self.n_head,
            "name": self.name
        }
    def from_config(cls, config):
        return cls(**config)
    def call(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(tf.Module):
    def __init__(self,width,layers,heads,attn_mask,name="transformer"):
        super(Transformer,self).__init__(name=name)
        self.width=width
        self.num_layers=layers
        self.heads=heads
        self.attn_mask=attn_mask
        self.resblocks=tf.keras.Sequential([
            ResidualAttentionBlock(width,heads,attn_mask,name=name + f".resblocks.{i}",idx=i)
            for i in range(layers)
        ], name=name + ".resblocks")
    def get_config(self):
        return {
            "width":self.width,
            "layers":self.num_layers,
            "heads":self.heads,
            "name":self.name
        }
    def from_config(cls, config):
        return cls(**config)
    def __call__(self, x: tf.Tensor):
        return self.resblocks(x)

class TokenEmbeding(tf.keras.layers.Layer):
    def __init__(self,name="tokenembd"):
        super(TokenEmbeding,self).__init__(name=name)
        self._vocab_size=49408
        self._transformer_width=512
        self._context_length=77
        self._name=name
    def build(self,input_shape):
        self._tokenembd=self.add_weight(name=self._name+"_tokenembd",
                                        shape=(self._vocab_size,self._transformer_width),
                                        initializer=tf.keras.initializers.get("glorot_uniform"),
                                        constraint=None,
                                        trainable=True)
        self._pstnembd=self.add_weight(name=self._name+"_positionembd",
                                       shape=(self._context_length,self._transformer_width),
                                       initializer=tf.keras.initializers.get("glorot_uniform"),
                                       constraint=None,
                                       trainable=True)
        super(TokenEmbeding,self).build(input_shape)
    def call(self,input_ts):
        x=tf.nn.embedding_lookup(self._tokenembd,input_ts)
        out_ts=x+self._pstnembd
        return out_ts
        
class PostProject(tf.keras.layers.Layer):
    def __init__(self,name="postproject"):
        super(PostProject,self).__init__(name=name)
        self._transformer_width=512
        self._embed_dim=1024
        self._name=name
    def build(self,input_shape):
        self._projtembd=self.add_weight(name=self._name+"_projtembd",
                                        shape=(self._transformer_width,self._embed_dim),
                                        initializer=tf.keras.initializers.get("glorot_uniform"),
                                        constraint=None,
                                        trainable=True)
        super(PostProject,self).build(input_shape)
    def call(self,input_ts):
        out_ts=tf.matmul(input_ts,self._projtembd)
        return out_ts
    
class TextEncoder(tf.Module):
    def __init__(self,name="textencoder"):
        super(TextEncoder,self).__init__(name=name)
        self._name=name
        self._Build()
    def _Build(self):
        self._transformer=Transformer(512,12,8,self._AttentionMask(),name=self._name+"_transformer")
        self._postproj=PostProject(name=self._name+"_postproj")
        self._norm=LayerNorm(name=self._name+"_norm")
        return
    def _AttentionMask(self):
        self._context_length=77
        n_dest=self._context_length
        n_src=self._context_length
        dtype=tf.bool
        batch_size=1
        i=tf.range(n_dest)[:,None]
        j=tf.range(n_src)
        m=i>=j-n_src+n_dest
        mask=tf.cast(m, dtype)
        mask=tf.reshape(mask, [1, n_dest, n_src])
        mult=tf.concat([tf.expand_dims(batch_size,-1),tf.constant([1,1],dtype=tf.int32)],0)
        return tf.tile(mask,mult)
    def __call__(self,in_ts_list):
        embd_ts,eot_tkn_idx=in_ts_list
        x=self._transformer(embd_ts)
        x=self._norm(x)
        x_shape=tf.shape(x)

        idx=tf.transpose(tf.stack((tf.range(0,x_shape[0],dtype=tf.int32),eot_tkn_idx),axis=0))
        x=tf.gather_nd(x,idx)
        out_ts=self._postproj(x)
        return out_ts
    
def GetTextEncoder(whts_path="text_encoder_whts.hdf5"):
    inputs=tf.keras.Input(shape=(77,),dtype=tf.int32)
    # eot_tkn_idx=tf.argmax(inputs,axis=-1)
    eot_tkn_idx=tf.keras.layers.Lambda(lambda x:tf.argmax(x,axis=-1))(inputs)
    eot_tkn_idx=tf.cast(eot_tkn_idx,tf.int32)
    x=TokenEmbeding()(inputs)
    out_ts=TextEncoder()([x,eot_tkn_idx])
    text_encoder=tf.keras.models.Model(inputs=inputs,outputs=out_ts)
    text_encoder.load_weights(whts_path)
    for layer in text_encoder.layers:
        layer.trainable=False
    return text_encoder

def GetTextEncoderNoEmbedLayer(whts_path="text_encoder_whts.hdf5"):
    _inputs=tf.keras.Input(shape=(77,),dtype=tf.int32)
    inputs_2=tf.keras.Input(shape=(77,512),dtype=tf.float32)
    x=TokenEmbeding()(_inputs)
    eot_tkn_idx=tf.keras.layers.Lambda(lambda x:tf.zeros_like(x[:,0],dtype=tf.int32)+76)(_inputs)
    text_encoder=TextEncoder()
    _out_ts=text_encoder([x,eot_tkn_idx])
    _text_encoder=tf.keras.Model(inputs=_inputs, outputs=_out_ts)
    _text_encoder.load_weights(whts_path)
    for layer in _text_encoder.layers:
        layer.trainable=False
    
    batch_size = tf.shape(inputs_2)[0]
    eot_tkn_idx = tf.fill([batch_size],76)
    out_ts=text_encoder([inputs_2,eot_tkn_idx])
    text_encoder=tf.keras.Model(inputs=inputs_2,outputs=out_ts)
    return text_encoder

def GetEmbeddingLayer(whts_path="txttkn_whts.hdf5"):
    in_ts=tf.keras.Input(shape=(77,),dtype=tf.int32)
    out_ts=TokenEmbeding()(in_ts)
    model=tf.keras.models.Model(inputs=in_ts,outputs=out_ts)
    model.load_weights(whts_path)
    for layer in model.layers:
            layer.trainable=False
    return model