"""Generate LaTeX tables and PDF figures from the completed saved comparison.

Run with the project virtual environment. No models are trained or results
modified; all rounded table entries come from results/comparison artifacts.
"""

import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'results' / 'comparison'
MODELS = ['gain', 'gain_ratio', 'gini', 'logistic_regression', 'knn', 'gaussian_nb']
LABELS = dict(zip(MODELS, ['Information gain', 'Gain ratio', 'Gini impurity', 'Logistic regression', 'Five-nearest neighbors', 'Gaussian naive Bayes']))


def escape(value):
    """Escape text fields so dataset names and platform identifiers compile."""
    replacements = {'&': r'\&', '%': r'\%', '_': r'\_', '#': r'\#', '$': r'\$',
                    '{': r'\{', '}': r'\}', '\\': r'\textbackslash{}'}
    return ''.join(replacements.get(c, c) for c in str(value))


def write_table(name, spec, heading, rows):
    """Write a booktabs tabular fragment to be included by report.tex."""
    content = [r'\begin{tabular}{' + spec + '}', r'\toprule', heading + r' \\', r'\midrule']
    content += [' & '.join(row) + r' \\' for row in rows]
    content += [r'\bottomrule', r'\end{tabular}']
    (HERE / 'tables' / name).write_text('\n'.join(content) + '\n', encoding='utf-8')


def main():
    """Build report assets and verify complete six-model, ten-dataset coverage."""
    for name in ('tables', 'figures'):
        (HERE / name).mkdir(parents=True, exist_ok=True)
    runs = json.loads((RESULTS / 'results.json').read_text())
    validation = json.loads((RESULTS / 'validation.json').read_text())
    manifest = json.loads((RESULTS / 'manifest.json').read_text())
    if validation['status'] != 'passed' or manifest['status'] != 'complete' or len(runs) != 180:
        raise ValueError('Report requires the validated 180-run comparison')
    if {(r['benchmark'], r['model'], r['repetition']) for r in runs} != {
            (b, m, rep) for b in range(1, 11) for m in MODELS for rep in range(1, 4)}:
        raise ValueError('Unexpected run coverage')
    with (RESULTS / 'summary.csv').open(newline='') as stream:
        summary = {(int(r['benchmark']), r['model']): r for r in csv.DictReader(stream)}
    first = {(r['benchmark'], r['model']): r for r in runs if r['repetition'] == 1}
    # Ensure references to unchanged report configuration are accurate.
    for r in runs:
        if r['family'] == 'custom_tree' and r['config'] != dict(max_depth=12, min_samples_leaf=2, min_samples_split=2, min_impurity_decrease=0.0):
            raise ValueError('Tree settings changed; revise report prose')
    rows = []
    for b in range(1, 11):
        r = first[b, 'gain']
        rows.append([str(b), escape(r['dataset']), f"{r['train_rows']:,}", f"{r['test_rows']:,}",
                     str(r['features']), str(len(r['classes'])), r['split']])
    write_table('datasets.tex', '@{}lp{1.75in}rrrrl@{}', r'Benchmark & Dataset & Training & Testing & Features & Classes & Split', rows)
    for metric, scale, filename in [('accuracy', 100, 'accuracy.tex'), ('macro_f1', 1, 'f1.tex')]:
        rows = []
        for b in range(1, 11):
            values = [float(summary[b, m][metric + '_mean']) * scale for m in MODELS]
            best = max(values)
            cells = [f'{v:.2f}' if scale == 100 else f'{v:.4f}' for v in values]
            cells = [r'\textbf{' + cell + '}' if abs(v - best) < 1e-9 else cell for cell, v in zip(cells, values)]
            rows.append([str(b)] + cells)
        write_table(filename, '@{}lrrrrrr@{}', r'Benchmark & \shortstack{Information\\gain} & \shortstack{Gain\\ratio} & \shortstack{Gini\\impurity} & \shortstack{Logistic\\regression} & \shortstack{Five-nearest\\neighbors} & \shortstack{Gaussian\\naive Bayes}', rows)
    with (RESULTS / 'overall.csv').open(newline='') as stream:
        overall = list(csv.DictReader(stream))
    rows = [[LABELS[r['model']], f"{100 * float(r['mean_accuracy']):.2f}", f"{float(r['mean_macro_f1']):.4f}",
             f"{float(r['total_train_seconds']):.3f}", f"{float(r['total_predict_seconds']):.3f}"] for r in overall]
    write_table('overall.tex', '@{}lrrrr@{}', r'Model & Accuracy (\%) & \shortstack{Macro-averaged\\F1 score (0--1)} & \shortstack{Total training\\(seconds)} & \shortstack{Total prediction\\(seconds)}', rows)
    rows = []
    for b in range(1, 11):
        for model in MODELS[:3]:
            r = first[b, model]
            rows.append([str(b), LABELS[model], str(r['tree']['nodes']), str(r['tree']['depth']),
                         f"{100*r['training_evaluation']['accuracy']:.1f}", f"{100*r['evaluation']['accuracy']:.1f}"])
    write_table('trees.tex', '@{}llrrrr@{}', r'Benchmark & Split criterion & Nodes & Depth & \shortstack{Training\\accuracy (\%)} & \shortstack{Testing\\accuracy (\%)}', rows)
    # Longtable retains all 60 configurations, rather than only favorable runs.
    content = [r'\begin{longtable}{@{}l>{\raggedright\arraybackslash}p{1.05in}rrrr@{}}',
               r'\caption{Resource measurements for every configuration. Each entry is mean $\pm$ sample standard deviation across three repetitions. Memory is total sampled resident process memory, not model-only memory.}\label{tab:resources}\\',
               r'\toprule Benchmark & Model & \shortstack{Training\\(seconds)} & \shortstack{Prediction\\(seconds)} & \shortstack{Training peak\\(mebibytes)} & \shortstack{Prediction peak\\(mebibytes)} \\ \midrule',
               r'\endfirsthead', r'\toprule Benchmark & Model & \shortstack{Training\\(seconds)} & \shortstack{Prediction\\(seconds)} & \shortstack{Training peak\\(mebibytes)} & \shortstack{Prediction peak\\(mebibytes)} \\ \midrule',
               r'\endhead', r'\midrule \multicolumn{6}{r}{Continued on next page}\\ \endfoot',
               r'\bottomrule \endlastfoot']
    for b in range(1, 11):
        for model in MODELS:
            r = summary[b, model]
            cells = [str(b), LABELS[model]]
            for metric, digits in [('training_seconds', 4), ('prediction_seconds', 4),
                                   ('training_sampled_peak_rss_mib', 1), ('prediction_sampled_peak_rss_mib', 1)]:
                cells.append(f"${float(r[metric + '_mean']):.{digits}f} \\pm {float(r[metric + '_std']):.{digits}f}$")
            content.append(' & '.join(cells) + r' \\')
        content.append(r'\addlinespace')
    content.append(r'\end{longtable}')
    (HERE / 'tables' / 'resources.tex').write_text('\n'.join(content) + '\n')
    # Detailed class behavior on a strongly imbalanced binary task.
    rows = []
    for model in MODELS:
        r = first[2, model]
        tn, fp = r['evaluation']['confusion_matrix'][0]
        fn, tp = r['evaluation']['confusion_matrix'][1]
        rows.append([LABELS[model], str(tn), str(fp), str(fn), str(tp),
                     f"{r['evaluation']['recall_per_class'][1]:.4f}", f"{r['evaluation']['macro_f1']:.4f}"])
    write_table('cervical.tex', r'@{}>{\raggedright\arraybackslash}p{1.05in}rrrrrr@{}', r'Model & \shortstack{True\\negatives} & \shortstack{False\\positives} & \shortstack{False\\negatives} & \shortstack{True\\positives} & \shortstack{Positive\\recall (0--1)} & \shortstack{Macro-averaged\\F1 score (0--1)}', rows)
    environment = first[1, 'gain']['environment']
    environment_names = dict(os='Operating system', processor='Processor', logical_cpus='Logical processors', python='Python', numpy='NumPy', scipy='SciPy', sklearn='scikit-learn', psutil='Process measurement library')
    rows = [[environment_names[k], escape(environment[k])] for k in ['os', 'processor', 'logical_cpus', 'python', 'numpy', 'scipy', 'sklearn', 'psutil']]
    rows.insert(3, ['Physical memory', f"{environment['system_memory_gib']:.2f} gibibytes"])
    write_table('environment.tex', '@{}ll@{}', 'Item & Recorded value', rows)
    # Separate vector figures remain readable when printed in the report.
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
    names = [LABELS[m] for m in MODELS]
    for metric, title, filename in [('accuracy', 'Test accuracy', 'accuracy.pdf'), ('macro_f1', 'Test macro-averaged F1 score', 'f1.pdf')]:
        values = np.array([[float(summary[b, m][metric + '_mean']) for m in MODELS] for b in range(1, 11)])
        fig, ax = plt.subplots(figsize=(6.6, 4.5), layout='constrained')
        image = ax.imshow(values, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        ax.set_xticks(range(6), [name.replace(' ', '\n', 1) if name != 'Gaussian naive Bayes' else 'Gaussian\nnaive Bayes' for name in names])
        ax.set_yticks(range(10), [str(b) for b in range(1, 11)])
        for i in range(10):
            for j in range(6):
                ax.text(j, i, f'{values[i,j]:.3f}', ha='center', va='center', color='black' if values[i,j] > 0.6 else 'white', fontsize=8)
        ax.set_title(title + ' (0?1; higher is better)')
        ax.set_ylabel('Benchmark number')
        fig.colorbar(image, ax=ax, label='Score (0?1; higher is better)')
        fig.savefig(HERE / 'figures' / filename)
        plt.close(fig)
    for metrics, titles, filename, log in [
            (['training_seconds', 'prediction_seconds'], ['Training time (seconds)', 'Prediction time (seconds)'], 'times.pdf', True),
            (['training_sampled_peak_rss_mib', 'prediction_sampled_peak_rss_mib'], ['Training resident memory\n(mebibytes)', 'Prediction resident memory\n(mebibytes)'], 'memory.pdf', False)]:
        fig, axes = plt.subplots(2, 1, figsize=(6.6, 5.8), layout='constrained')
        for ax, metric, title in zip(axes, metrics, titles):
            for m in MODELS:
                means = [float(summary[b, m][metric + '_mean']) for b in range(1, 11)]
                ax.plot(range(1, 11), means, marker='o', markersize=3, linewidth=1, label=LABELS[m])
            ax.set_xticks(range(1, 11), [str(b) for b in range(1, 11)])
            ax.set_ylabel(title)
            ax.set_xlabel('Benchmark number')
            ax.grid(alpha=0.25)
            if log:
                ax.set_yscale('log')
        axes[0].legend(ncol=3, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 1.22))
        fig.savefig(HERE / 'figures' / filename)
        plt.close(fig)
    # Preserve compact source tables beside the PDF for numerical traceability.
    for filename in ['summary.csv', 'overall.csv', 'validation.json', 'manifest.json']:
        shutil.copy2(RESULTS / filename, HERE / filename)
    print('Generated seven result tables, environment table and four vector figures from 180 validated runs.')


if __name__ == '__main__':
    main()
