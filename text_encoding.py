import tensorflow as tf
import clip
import numpy as np

def LoadTextEncoder(model_path="CLIP_text_RN50"):
    return tf.keras.models.load_model(model_path)

def Texts2Embds(model,texts):
    texts=clip.tokenize(texts,truncate=False)
    embds=model.predict_on_batch(np.array(texts))
    return embds

def Texts2Tokens(texts):
    tokens=clip.tokenize(texts,truncate=False)
    return np.array(tokens).tolist()

def Texts2TokenEmbeddings(model,texts):
    texts=clip.tokenize(texts,truncate=False)
    embds=model.predict_on_batch(np.array(texts))
    embds=np.array(embds).tolist()
    return embds