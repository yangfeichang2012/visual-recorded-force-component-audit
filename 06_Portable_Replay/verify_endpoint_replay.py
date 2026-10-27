"""Portable stdlib-only replay: frozen per-row targets/predictions -> record metrics.
Read only. No model fitting, image decoding, network, private absolute paths or writes.
"""
from pathlib import Path
import csv,gzip,json,math,statistics,hashlib
D=Path(__file__).resolve().parent;B=D.parent;SD=B/'05_Review_Only/source_data';checks=0;maximum=0.
def rd(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):
 op=gzip.open if p.suffix=='.gz' else open
 with op(p,'rt',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def ck(v,msg):
 global checks
 checks+=1
 if not v:raise AssertionError(msg)
def eq(a,b,msg):
 global maximum
 if a is None or b in [None,'']:ck(a is None and b in [None,''],msg);return
 d=abs(float(a)-float(b));maximum=max(maximum,d);ck(math.isfinite(d) and d<=1e-10,(msg,a,b))
def quantile(v,q):
 s=sorted(v);h=(len(s)-1)*q;i=int(h);return s[i]+(s[min(i+1,len(s)-1)]-s[i])*(h-i)
def metrics(y,p,q):
 e=[a-b for a,b in zip(p,y)];a=[abs(x) for x in e];hi=[v for v,t in zip(a,y) if t>=q];peak=max(range(len(y)),key=y.__getitem__)
 return {'MAE_N':statistics.fmean(a),'RMSE_N':math.sqrt(statistics.fmean(x*x for x in e)),'P95_N':quantile(a,.95),'bias_N':statistics.fmean(e),'high_Q90_MAE_N':statistics.fmean(hi) if hi else None,'high_n':len(hi),'peak_underestimate_N':max(0,-e[peak]),'peak_signed_error_N':e[peak],'n':len(y),'training_Q90_N':q}
def checkrow(actual,expected,label):
 for f,v in actual.items():eq(v,expected[f],(label,f))
manifest=rd(D/'endpoint-manifest.json');data={};out={}
ref76={(r['record'],r['method']):r for r in rows(SD/'record-metrics.csv')}
ref204={(r['record'],r['method']):r for r in rows(SD/'D204/record-metrics.csv')}
for item in manifest['records']:
 p=D/item['file'];ck(hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],'endpoint SHA');rr=rows(p);ck(len(rr)==item['rows'],'rows');y=[float(r['target_N']) for r in rr];q=float(rr[0]['training_Q90_N']);grids=[int(r['row_index']) for r in rr];times=[int(r['record_ns']) for r in rr]
 ck(all(g2>g1 for g1,g2 in zip(grids,grids[1:])),'sorted unique grid');ck(all(t2-t1==200000000*(g2-g1) for t1,t2,g1,g2 in zip(times,times[1:],grids,grids[1:])),'5Hz stable clock');ck(all(float(r['training_Q90_N'])==q for r in rr),'constant training Q90');peak=max(range(len(y)),key=y.__getitem__);ck([i for i,r in enumerate(rr) if int(r['peak_row'])]==[peak],'first eligible argmax peak')
 k=item['record'];data[k]={int(r['row_index']):(float(r['target_N']),int(r['record_ns'])) for r in rr}
 for method in manifest['methods']:
  pp=[float(r[method]) for r in rr];mm=metrics(y,pp,q);checkrow(mm,ref76[k,method],(k,method));out[k,method]=mm
  if (k,method) in ref204:checkrow(mm,ref204[k,method],('D204 inherited',k,method))
 method='GRU64_FP64_5frames_selectedL1'
 if rr[0][method]:checkrow(metrics(y,[float(r[method]) for r in rr],q),ref204[k,method],('D204 GRU',k))
ck(sum(len(v) for v in data.values())==35303,'all targets');ck(len(out)==1494,'D76 18x83')
inner={}
for r in rows(D/'D205-inner-predictions.csv.gz'):inner.setdefault((r['role'],r['cap'],r['record']),[]).append(r)
nulls={};zero_pairs={}
for expected in rows(SD/'D205/record-metrics.csv'):
 key=(expected['role'],expected['cap'],expected['record']);rr=inner[key];ids=[int(r['row_index']) for r in rr];ck(ids==list(data[key[2]]),'D205 denominator identity');y=[data[key[2]][g][0] for g in ids];q=float(rr[0]['training_Q90_N']);ck(all(float(r['training_Q90_N'])==q for r in rr),'D205 Q90 constant');actual=metrics(y,[float(r['prediction_N']) for r in rr],q);checkrow(actual,expected,('D205',key))
 if actual['high_n']==0:nulls[key[1]]=nulls.get(key[1],0)+1;zero_pairs.setdefault(key[1],[]).append({'role':key[0],'record':key[2]})
ck(len(inner)==180,'D205 all role-record-cap cells');ck(nulls=={'269':3,'538':3,'1076':3},'explicit null support')
summ=rd(SD/'scientific-summary.json');paired={}
for label in ['chicken__V1','singlepsm__all']:
 groups=summ['results'][label]['group_MAE_N'];wins=sum(x['fusion']<x['branch_B'] for x in groups.values());paired[label]={'fusion_lower_MAE_than_B':wins,'groups':len(groups)}
ck(paired['chicken__V1']['fusion_lower_MAE_than_B']==4,'paired V1 4/5');ck(paired['singlepsm__all']['fusion_lower_MAE_than_B']==6,'paired PSM 6/12')
print(json.dumps({'status':'PASS_PER_ROW_ENDPOINT_TO_RECORD_METRIC_REPLAY','checks':checks,'max_difference_N':maximum,'D76_record_method_cells':1494,'D204_record_method_cells':380,'D205_role_record_cap_cells':180,'unique_target_rows':35303,'zero_high_support_per_cap':nulls,'zero_support_pairs':zero_pairs,'paired_fusion_vs_B':paired,'new_fits':0,'new_model_inference':0,'network':False,'writes':0,'next':'Run the two adjacent aggregation verifiers for macro metrics and unchanged gates.'}))
