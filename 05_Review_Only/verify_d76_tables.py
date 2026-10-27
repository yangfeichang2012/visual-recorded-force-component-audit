"""Read-only standard-library aggregation verification; no fitting, writes or network."""
from pathlib import Path
import json,csv,math
B=Path(__file__).resolve().parents[1];D=B/'05_Review_Only/source_data';s=json.loads((D/'scientific-summary.json').read_text(encoding='utf-8'));rows=list(csv.DictReader((D/'record-metrics.csv').open(encoding='utf-8')));checks=0;maximum=0.
def check(value,name):
 global checks
 checks+=1
 if not value:raise AssertionError(name)
def equal(a,b,name):
 global maximum
 maximum=max(maximum,abs(a-b));check(abs(a-b)<=1e-10,name)
def mean(a):return sum(a)/len(a)
def aggregate(rr,col,conditionequal):
 valid=[r for r in rr if r[col]!='']
 if not valid:return None
 if not conditionequal:return mean([float(r[col]) for r in valid])
 gs=sorted({r['group'] for r in valid});return mean([mean([float(r[col]) for r in valid if r['group']==g]) for g in gs])
for key,result in s['results'].items():
 dataset,view=key.split('__');scope=[r for r in rows if r['dataset']==dataset and r['view']==view];cond=dataset=='chicken';group_table={}
 for method,mt in result['metrics'].items():
  rr=[r for r in scope if r['method']==method];check(len(rr)==mt['records'],'records');check(sum(int(r['n']) for r in rr)==mt['endpoints'],'endpoints');check(sum(int(r['high_n']) for r in rr)==mt['high_endpoints'],'highsupport')
  for col in ['MAE_N','RMSE_N','P95_N','bias_N','high_Q90_MAE_N','peak_underestimate_N','peak_signed_error_N']:
   value=aggregate(rr,col,cond)
   if value is None:check(mt[col] is None,'null')
   else:equal(value,mt[col],key+method+col)
  equal(aggregate(rr,'MAE_N',True),mt['condition_equal_MAE_N'],'condition equal')
 for g,mt in result['group_MAE_N'].items():
  group_table[g]={}
  for method,value in mt.items():
   calc=mean([float(r['MAE_N']) for r in scope if r['group']==g and r['method']==method]);equal(calc,value,'group');group_table[g][method]=calc
 table=result['metrics'];gain={m:1-table['fusion']['MAE_N']/table[m]['MAE_N'] for m in ['full_median','fixed_temporal1536_L1','global6_L1']};positive={m:sum(v['fusion']<v[m] for v in group_table.values()) for m in ['full_median','fixed_temporal1536_L1']};tails={col:table['fusion'][col]<=table['fixed_temporal1536_L1'][col]+1e-12 for col in ['P95_N','high_Q90_MAE_N','peak_underestimate_N']};gate=all(gain[m]>=.1 for m in ['full_median','fixed_temporal1536_L1']) and gain['global6_L1']>=.05 and all(n/len(group_table)>=.8 for n in positive.values()) and all(tails.values());stored=s['scientific_gates'][key]
 check(gate==stored['passed'],'gate');check(positive==stored['positive_groups'],'group directions')
 for m,value in gain.items():equal(value,stored['MAE_relative_gain'][m],'gain')
check(s['scientific_gates']['singlepsm__all']['passed'] is True,'SinglePSM positive retained');check(not s['scientific_gates']['chicken__V1']['passed'] and not s['scientific_gates']['chicken__V2']['passed'],'chicken failures retained');check(s['results']['singlepsm__all']['metrics']['branch_B']['MAE_N']<s['results']['singlepsm__all']['metrics']['fusion']['MAE_N'],'motion B superiority retained')
print(json.dumps({'status':'PASS_READ_ONLY_D76_AGGREGATION','checks':checks,'max_difference_N':maximum,'source_record_method_rows':len(rows),'new_fits':0,'new_inference':0,'writes':0,'network':False,'scope':'Output-table aggregation/gates only;not raw endpoint prediction replay.'}))
