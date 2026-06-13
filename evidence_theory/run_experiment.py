import pandas as pd
import os
import sys
import traceback
import click

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evidence_theory import core, experiment

@click.command()
@click.option('-m', default=4, help='The number of elements in X.')
@click.option('-n', help='A name for the experiment.')
def main(m, n):
    N = 1000 # number of samples
    # Definisci il percorso corretto per la cartella 'notebooks' dentro 'evidence-theory-master'
    notebooks_dir = os.path.join(os.path.dirname(__file__), 'notebooks')
    os.makedirs(notebooks_dir, exist_ok=True)
    
    f_name = os.path.join(notebooks_dir, f"results{m}-{n}.csv")
    
    entropy_measures = [
        core.hohle,
        core.smets,
        core.yager,
        core.nguyen,
        core.dubois_prade,
        core.lamata_moral,
        core.klir_and_ramer,
        core.klir_and_parviz,
        core.pal_et_al,
        core.deng,
        core.jirousek_and_shenoy,
        core.qin_et_al,
        core.yan_and_deng,
        core.li_et_al,
        core.harmanec_and_klir,
        core.li_and_pan,
        core.pan_and_deng,
        core.deng_and_wang,
        core.jousselme_et_al,
        core.yang_and_han,
        core.fractal_based_entropy,
        core.cui_et_al,
        core.george_and_pal,
        core.wang_and_song,
        core.zhou_et_al
    ]
    
    if os.path.isfile(f_name):
        r = pd.read_csv(f_name, index_col=0)
    else:
        try:
            r = experiment.experiment(N, m, entropy_measures)
            r.to_csv(f_name)
        except Exception as e:
            print("An error occurred:", e)
            print(traceback.format_exc())

if __name__ == '__main__':
    main()


