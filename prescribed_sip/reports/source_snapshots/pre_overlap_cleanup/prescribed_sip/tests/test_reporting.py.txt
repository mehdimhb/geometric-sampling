import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from prescribed_sip.experiments.artifacts import plot_curves,tables


def test_curves_keep_variants_separate_and_select_evaluation_regime():
    rows=[]
    for variant,regime,ys in [('a','evaluations',[.3,.2]),('b','evaluations',[.1,.05]),('a','time',[.6,.5])]:
        for candidate,y in zip([10,20],ys):
            rows.append({'scenario_id':'s','replication':0,'candidate_moves':candidate,'runtime_seconds':candidate/10,
                         'method':'GFS-Directed','method_variant':variant,'regime':regime,
                         'target_RMSE2':y,'incumbent_target_RMSE2':y})
    fig,ax=plt.subplots();plot_curves(ax,pd.DataFrame(rows),'candidate_moves','s')
    assert len(ax.lines)==2
    assert all(max(line.get_ydata())<.4 for line in ax.lines)
    plt.close(fig)


def test_tables_retain_regime_density_structure_and_budgets(tmp_path):
    (tmp_path/'tables').mkdir()
    summary=pd.DataFrame([{'experiment':'sparse','N':250,'n':5,'geometry':'1d','target_type':'srs','FIP_type':'equal',
                          'method':'Package-GFS','method_variant':'default','runs':2,'target_density':.01,
                          'pair_structure':'nearby','regime':'time','evaluation_budget':100000,'time_budget_seconds':1,
                          'candidate_batch':50}])
    tables(tmp_path,summary)
    table=pd.read_csv(tmp_path/'tables/table_3.csv')
    assert {'target_density','pair_structure','regime','evaluation_budget','time_budget_seconds','candidate_batch'}.issubset(table.columns)
