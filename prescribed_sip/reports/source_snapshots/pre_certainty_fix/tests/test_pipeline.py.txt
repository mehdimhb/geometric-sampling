"""End-to-end artifacts must be inspectable, paired and reproducible."""
from pathlib import Path
import json
import pandas as pd
import yaml
from prescribed_sip.run_all import execute


def test_smoke_pipeline_complete_and_resumable(tmp_path):
    config={'master_seed':12,'output':str(tmp_path),'methods':['Package-GFS','GFS-Directed','CP2-enumerated','Sinha'],
            'budget':{'evaluations':40,'seconds':2,'batch':10},
            'scenarios':[{'experiment':'small','N':6,'n':2,'target_type':'srs','geometry':'1d','replications':2}]}
    execute(config)
    raw=pd.read_csv(tmp_path/'results/runs.csv')
    assert len(raw)==10 # 2 replications, 4 methods + exact LP/QP
    assert all(raw.geometry=='1d')
    assert {'final_RMSE2','seed','QP_optimal_D2','max_FIP_error'}.issubset(raw.columns)
    assert raw.loc[raw.status.isin(['completed','budget_exhausted','timeout']),'max_FIP_error'].max()<1e-10
    for i in range(1,10):
        assert (tmp_path/f'figures/figure_{i}.pdf').is_file()
        assert (tmp_path/f'figures/figure_{i}.png').is_file()
    for i in range(1,5):
        assert (tmp_path/f'tables/table_{i}.csv').is_file()
        assert (tmp_path/f'tables/table_{i}.tex').is_file()
    assert 'MAIN FINDINGS FOR THE PAPER' in (tmp_path/'reports/full_simulation_report.md').read_text()
    assert (tmp_path/'results/paired_comparisons.csv').is_file()
    execute(config)
    assert len(pd.read_csv(tmp_path/'results/runs.csv'))==10


def test_csv_preserves_large_seed_digits_with_missing_optional_seed(tmp_path):
    import csv
    from prescribed_sip.run_all import collect
    folder=tmp_path/'results/runs'; folder.mkdir(parents=True)
    (folder/'a.json').write_text(json.dumps({'row':{'seed':9007199254740993,'target_seed':9007199254740993},'checkpoints':[]}))
    (folder/'b.json').write_text(json.dumps({'row':{'seed':1},'checkpoints':[]}))
    collect(tmp_path)
    with (tmp_path/'results/runs.csv').open() as f: rows=list(csv.DictReader(f))
    assert rows[0]['target_seed']=='9007199254740993'
