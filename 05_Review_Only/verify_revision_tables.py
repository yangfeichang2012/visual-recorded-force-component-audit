"""Read-only stdlib check of added output tables and stopping gates; no model inference."""
from pathlib import Path
import json,csv,math,statistics,hashlib
B=Path(__file__).resolve().parents[1];D=B/'05_Review_Only/source_data';checks=0;maxdiff=0.
def rd(p):return json.loads(p.read_text(encoding='utf-8'))
def ck(v,name):
 global checks
 checks+=1
 if not v:raise AssertionError(name)
def eq(a,b,name):
 global maxdiff
 if a is None or b is None:ck(a is b,name);return
 diff=abs(float(a)-float(b));maxdiff=max(maxdiff,diff);ck(math.isfinite(diff) and diff<=1e-10,(name,a,b))
def mean(v):return statistics.fmean(v)
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return [{k:(None if v=='' else float(v)) if k not in ['dataset','view','group','record','method','role'] else v for k,v in r.items()} for r in csv.DictReader(f)]
a=rd(D/'D204/scientific-summary.json');rr=rows(D/'D204/record-metrics.csv');ck(len(rr)==380 and len({r['record'] for r in rr})==20,'D20419x20')
fields=['MAE_N','RMSE_N','P95_N','bias_N','high_Q90_MAE_N','peak_underestimate_N','peak_signed_error_N'];groups=sorted({r['group'] for r in rr});methods=sorted({r['method'] for r in rr});table={};group={}
for m in methods:
 bag=[r for r in rr if r['method']==m];table[m]={};ck(len(bag)==20 and sum(r['n'] for r in bag)==5490,'D204denominator')
 for f in fields:
  good=[r for r in bag if r[f] is not None];gs=sorted({r['group'] for r in good});v=mean(mean(r[f] for r in good if r['group']==g) for g in gs) if gs else None;table[m][f]=v;eq(v,a['metrics'][m][f],'D204macro')
 for g in groups:group.setdefault(g,{})[m]=mean(r['MAE_N'] for r in bag if r['group']==g);eq(group[g][m],a['group_MAE_N'][g][m],'D204group')
candidate='GRU64_FP64_5frames_selectedL1';refs=['full_median','fixed_temporal1536_L1'];gains={m:1-table[candidate]['MAE_N']/table[m]['MAE_N'] for m in refs+['global6_L1','branch_A','branch_B','fusion']};wins={m:sum(group[g][candidate]<group[g][m] for g in groups) for m in refs};tail={f:table[candidate][f]<=table['fixed_temporal1536_L1'][f]+1e-12 for f in ['P95_N','high_Q90_MAE_N','peak_underestimate_N']};gate=all(gains[m]>=.1 for m in refs) and gains['global6_L1']>=.05 and all(v/5>=.8 for v in wins.values()) and all(tail.values());ck(gate==a['scientific_gate']['passed']==False,'D204scientificFAIL');ck(wins==a['scientific_gate']['positive_groups'] and tail==a['scientific_gate']['tails_no_worse'],'D204protections')
for m,v in gains.items():eq(v,a['scientific_gate']['MAE_relative_gain'][m],'D204gain')
primary=rd(D/'scientific-summary.json')['results']['chicken__V1']['metrics']
for m in methods:
 if m==candidate:continue
 for f in fields:eq(table[m][f],primary[m][f],'originalD76notreplaced')
b=rd(D/'D205/scientific-summary.json');rr=rows(D/'D205/record-metrics.csv');ck(len(rr)==180,'D2053x15x4');t={};gm={};fs=['MAE_N','P95_N','high_Q90_MAE_N','peak_underestimate_N']
for cap in [269,538,1076]:
 bag=[r for r in rr if r['cap']==cap];ck(len(bag)==60 and len({r['role'] for r in bag})==15 and len({r['record'] for r in bag})==20,'D205role_denominator');gm[cap]={}
 for g in groups:
  local=[r for r in bag if r['group']==g];roles=sorted({r['role'] for r in local});ck(len(roles)==3 and len(local)==12,'D205repeatedroles')
  gm[cap][g]={}
  for f in fs:
   vals=[mean(r[f] for r in local if r['role']==role and r[f] is not None) for role in roles if any(r['role']==role and r[f] is not None for r in local)];v=mean(vals) if vals else None;gm[cap][g][f]=v;eq(v,b['validation_proxy_ID_metrics'][str(cap)][g][f],'D205group')
 t[cap]={f:mean(gm[cap][g][f] for g in groups if gm[cap][g][f] is not None) for f in fs}
 for f,v in t[cap].items():eq(v,b['metrics'][str(cap)][f],'D205macro')
gain={f:1-t[1076][f]/t[538][f] for f in fs};win=[g for g in groups if all(gm[1076][g][f]<gm[538][g][f]-1e-12 for f in ['MAE_N','P95_N'])];protection={f:t[1076][f]<=t[538][f]+1e-12 for f in ['high_Q90_MAE_N','peak_underestimate_N']};cont=all(gain[f]>=.05 for f in ['MAE_N','P95_N']) and len(win)>=4 and all(protection.values());ck(cont==b['continue_signal']==False,'D205STOP');ck(win==b['proxy_IDs_both_MAE_P95_improved']==['M4'] and protection==b['tail_protection'],'D205guards')
for f,v in gain.items():eq(v,b['primary_relative_gain_1076_vs538'][f],'D205gain')
print(json.dumps({'status':'PASS_READ_ONLY_REVISION_METRICS_AND_GATES','checks':checks,'max_difference_N':maxdiff,'D204scientific_gate':gate,'D205continue_signal':cont,'new_fits':0,'new_model_inference':0,'writes':0,'network':False,'scope':'D204/D205 numerical aggregation and unchanged stopping gates. Unpublished manuscript wording is not included or checked in this public distribution.'}))
