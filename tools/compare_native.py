"""Compare fresh native results against the inspected Table I and Figure 1."""
import argparse
import csv
import json
import re
from decimal import Decimal
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def compare(run: Path):
    reference = list(csv.DictReader((ROOT / 'evidence/JKP_TableI_reference.csv').open()))
    comparisons = []
    for row in reference:
        sample = row['sample']
        estimates = pd.read_csv(run / 'numeric' / f'hyperparameters_{sample}.csv').set_index('estimate')
        for published_name, native_name in [('tau_c','tau_c'),('tau_w','tau_s'),('tau_s','tau_w')]:
            printed = row[published_name+'_pct']
            if printed:
                fresh = float(estimates.loc[native_name,'ml_est'])
                comparisons.append(dict(sample=sample, paper_parameter=published_name,
                    native_parameter=native_name, printed_pct=float(printed), fresh_pct=fresh,
                    difference_pp=fresh-float(printed), within_rounding=abs(fresh-float(printed))<=0.005+1e-12))
    table = pd.DataFrame(comparisons)
    table.to_csv(run/'numeric/TableI_comparison.csv',index=False)
    info = pd.read_csv(run/'numeric/factor_metadata.csv')
    rows = []
    for region in ['us','world']:
        factors = pd.read_csv(run/'numeric'/f'factors_{region}.csv')
        merged = factors.merge(info[['characteristic','significance']],on='characteristic',validate='one_to_one')
        assert len(merged)==len(factors) and merged.significance.notna().all()
        sig = merged[merged.significance==1]
        lower = sig.p025 if region == 'us' else sig.post_mean - 1.96*sig.post_sd
        rate = 100*(lower>0).mean()
        rows.append(dict(region=region,eligible=len(sig),positive=int((lower>0).sum()),
            fresh_pct=float(rate),paper_pct=82.4,within_rounding=bool(abs(rate-82.4)<=0.05)))
    pd.DataFrame(rows).to_csv(run/'numeric/headline_comparison.csv',index=False)
    result={'table_cells':len(table),'table_matched':int(table.within_rounding.sum()),'headline':rows,
        'scope':'Table I and originally significant US/world EB rates from computed model objects. Other numerical outputs require separate assessment.'}
    (run/'numeric/comparison_status.json').write_text(json.dumps(result,indent=2))
    return result

def compare_interrupted_console():
    reference=list(csv.DictReader((ROOT/'evidence/JKP_TableI_reference.csv').open()))
    lines=(ROOT/'logs/native_attempt_01.log').read_text(encoding='utf-8').splitlines()
    found=[]; idx=-1
    for line in lines:
        if line.startswith('[1] "Region:'): idx+=1
        match=re.match(r'^\d+ (tau_[csw])\s+\S+\s+(\S+)',line)
        if match and idx>=0:
            native, token=match.groups()
            printed=Decimal(token)
            ref=Decimal(reference[idx][{'tau_c':'tau_c_pct','tau_s':'tau_w_pct','tau_w':'tau_s_pct'}[native]])
            uncertainty=Decimal(1).scaleb(printed.as_tuple().exponent)/2
            distance=abs(printed-ref)
            status=('unambiguous_from_console' if distance+uncertainty<Decimal('.005')
                    else 'overlapping_rounding_bins' if distance-uncertainty<=Decimal('.005')
                    else 'inconsistent')
            found.append({'sample':reference[idx]['sample'],'native_label':native,'console':str(printed),
                'paper_pct':str(ref),'comparison':status})
    assert len(found)==23
    out=ROOT/'evidence/interrupted_console_comparison.csv'
    pd.DataFrame(found).to_csv(out,index=False)
    return pd.DataFrame(found).comparison.value_counts().to_dict()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path)
    args=parser.parse_args()
    print(json.dumps(compare(args.run) if args.run else compare_interrupted_console(),indent=2))
