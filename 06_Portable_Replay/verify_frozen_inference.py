"""Three small frozen cached-input examples; inference only, no optimizer or labels."""
from pathlib import Path
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
import json,csv,gzip,numpy as np
from inference import descriptors,head,gru
D=Path(__file__).resolve().parent;checks=0;maximum=0.;endpoint_links=0;identity_checks=0
def load(p):
 with np.load(p,allow_pickle=False) as z:return {a:z[a] for a in z.files}
for case in ['chicken','singlepsm','GRU']:
 p=D/'examples'/case;info=json.loads((p/'example.json').read_text(encoding='utf-8'));x=load(p/'cached-inputs.npz')
 if case=='GRU':
  import torch
  got={'GRU':gru(x['embedding'],x['reference'],load(p/'preprocessing.npz'),torch.load(p/'model-state.pt',map_location='cpu',weights_only=True))};expected={'GRU':info['expected']}
 else:
  got={}
  for a in ['A','B']:
   m=load(p/(a+'-model.npz'));feature=str(m['method']).rsplit('_',1)[0];got['branch_'+a]=head(m,descriptors(x['embedding'],x['reference'],x['local72'],feature),load(p/(a+'-training-features.npz'))['x'])
  got['fusion']=(got['branch_A']+got['branch_B'])/2;expected=info['expected']
 with gzip.open(D/'endpoints'/(info['record']+'.csv.gz'),'rt',encoding='utf-8',newline='') as f:rr=list(csv.DictReader(f))
 if info['offsets']!=[0,len(rr)//2,len(rr)-1]:raise AssertionError('Non-prespecified row selection')
 for i,offset in enumerate(info['offsets']):
  r=rr[offset]
  if int(x['row_index'][i])!=int(r['row_index']) or int(x['record_ns'][i])!=int(r['record_ns']):raise AssertionError('Example endpoint identity mismatch')
  identity_checks+=2
  for name,v in expected.items():
   col='GRU64_FP64_5frames_selectedL1' if case=='GRU' else name
   if float(v[i])!=float(r[col]):raise AssertionError(('Expected value not linked to frozen endpoint',case,name,offset))
   endpoint_links+=1
 for name,v in expected.items():
  delta=float(np.max(np.abs(got[name]-np.array(v))));maximum=max(maximum,delta);checks+=1
  if not delta<=1e-12:raise AssertionError((case,name,delta))
print(json.dumps({'status':'PASS_THREE_FROZEN_CACHED_INPUT_INFERENCE_EXAMPLES','cases':3,'rows_per_case':3,'vector_checks':checks,'scalar_expected_to_endpoint_links':endpoint_links,'row_time_identity_checks':identity_checks,'max_prediction_difference_N':maximum,'absolute_tolerance_N':1e-12,'optimizer_steps':0,'raw_image_encoder_replay':False,'GPU':False,'writes':0,'network':False}))
