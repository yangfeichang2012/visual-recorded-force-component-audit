"""Frozen cached-input inference only. No fit, optimizer, search, image decode or network."""
import numpy as np
def descriptors(embedding,reference,local72,feature):
 # D76 descriptors intentionally preserve cached float32 arithmetic.
 x=embedding[:,-1];mean=embedding.mean(axis=1);diff=x-embedding[:,0]
 if feature=='relative_temporal1536':return np.concatenate([x-reference,mean-reference,diff],axis=1)
 if feature=='delta512':return x-reference
 if feature=='static512':return x
 if feature=='temporal1536':return np.concatenate([x,mean,diff],axis=1)
 if feature=='local72':return local72[:,-1]
 if feature=='local144':return np.concatenate([local72[:,-1],local72[:,-1]-local72[:,0]],axis=1)
 raise ValueError(feature)
def scale(x,mean,sd):return (np.asarray(x,dtype=np.float64)-mean)/sd/np.sqrt(len(mean))
def head(model,x,train=None):
 z=scale(x,model['mean'],model['scale'])
 if str(model['method']).endswith('_L1'):v=model['beta'][0]+z@model['beta'][1:]
 else:
  t=scale(train,model['mean'],model['scale']);dd=np.maximum((z*z).sum(1)[:,None]+(t*t).sum(1)[None,:]-2*z@t.T,0.)
  v=np.exp(-dd/(2*float(model['bw'])))@model['alpha']+float(model['b'])
 return v*float(model['ys'])+float(model['ym'])
def gru(embedding,reference,preprocessing,state):
 import torch
 from gru_head import build_head
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 model=build_head();model.load_state_dict(state);model.eval()
 # Unlike D76 descriptors the learned-head subtraction was FP64.
 x=embedding.astype(np.float64)-reference.astype(np.float64)
 x=(x-preprocessing['feature_mean'])/preprocessing['feature_scale']/np.sqrt(512)
 with torch.no_grad():p=model(torch.from_numpy(x)).numpy()
 return p*float(preprocessing['target_scale'])+float(preprocessing['target_mean'])
