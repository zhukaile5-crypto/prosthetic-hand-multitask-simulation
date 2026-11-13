"""Presentation-only spacing, retaining frozen plot/data definitions unchanged."""
from pathlib import Path
import hashlib,json
import plot_results as frozen

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    source=ROOT/'output/confirmation_combined/analysis/summary.json'
    result=json.loads(source.read_text(encoding='utf-8'))
    destination=ROOT/'output/confirmation_combined/figures_verified';destination.mkdir(exist_ok=False)
    original=frozen._save
    def spaced(figure,out,name):
        if name.startswith('全景五轴'):
            for axis in figure.axes:
                label=axis.get_ylabel()
                if '平均法向握力' in label:axis.set_ylim(0,14.)
                elif '各自有效窗' in label and 'TV' in label:axis.set_ylim(0,2.8)
                elif '释放完成' in label:
                    axis.set_ylim(0,112)
                    for item in axis.texts:
                        if item.get_text()=='100.0':item.set_y(103.5)
        original(figure,out,name)
    frozen._save=spaced
    files=frozen.plot_results(result,destination)
    report=dict(scope='presentation only; same frozen data/cohorts/statistics; original plots preserved',
        changes=['equal TV/mean-force scales across parameter-domain overviews','headroom for counts and bar values','release labels separated from coverage markers'],
        summary_sha256=sha(source),frozen_plot_source_sha256=sha(ROOT/'plot_results.py'),
        wrapper_sha256=sha(Path(__file__)),files={p.name:sha(p) for p in destination.iterdir() if p.suffix in ('.png','.svg')})
    (destination/'figure_provenance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(figures=files,statistics_changed=False),ensure_ascii=False))
if __name__=='__main__':main()
