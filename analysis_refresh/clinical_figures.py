"""Clinical figures for the two retained cohorts; numeric inputs are revalidated by refresh_results.py."""
import numpy as np
from sklearn.metrics import roc_curve,precision_recall_curve
from types import SimpleNamespace
import clinical_statistics as s
OLD=['round1_c2','round1_c4','round1_c6']
NEW=['clinical_seed20261021','clinical_seed20261022','clinical_seed20261023']
c=SimpleNamespace(OLD=OLD,NEW=NEW,ORDER=OLD+NEW+['round1_clinical_ensemble','new_clinical_ensemble'])
TITLES={'besga_reused_test':'Reused Besga test: 11 patients (7 AD, 4 CN)','openneuro_reused_external':'Reused OpenNeuro: 15 patients (1 AD, 14 CN)'}
DISPLAY={n:n for n in c.ORDER}
COLORS=['#999999','#666666','#333333','#4c9bda','#36a26f','#a481c6','#c16421','#00758f']
NOTE='Patient-level research pilot; reused convenience cohorts. No clinical superiority or population risk-calibration claim.'
def cohort_note(cohort):
    return 'Only one AD patient: exact sensitivity intervals remain broad; conditional bootstrap does not establish population sensitivity precision.' if cohort == 'openneuro_reused_external' else NOTE

def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42, 'axes.spines.right': False, 'axes.spines.top': False})
    return plt

def save(fig, out, stem, synthetic=False):
    if synthetic:
        fig.text(0.5, 0.5, 'SYNTHETIC QA', ha='center', fontsize=36, alpha=0.15, rotation=20)
    for ext in ['png']:
        fig.savefig(out / f'{stem}.{ext}', dpi=220, facecolor='white')
    plotting().close(fig)

def cohort_figures(cohort, d, out, synthetic):
    plt = plotting()
    names = c.ORDER
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.6))
    for ax, metric, ci in zip(axes, ['accuracy', 'sensitivity_ad', 'specificity_cn'], ['accuracy_exact95', 'sensitivity_exact95', 'specificity_exact95']):
        for i, (name, color) in enumerate(zip(names, COLORS)):
            m = d['models'][name]['clean']['metrics']
            v = m[metric]
            lo, hi = m[ci]
            ax.errorbar(v * 100, i, xerr=np.array([[v - lo], [hi - v]]) * 100, fmt='o', color=color, capsize=3, markersize=6 if 'ensemble' in name else 4)
        ax.set_yticks(range(8), [DISPLAY[n] for n in names] if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set(xlim=(-2, 102), xlabel=metric.replace('_', ' ') + ' (%)')
        ax.grid(axis='x', alpha=0.2)
    fig.suptitle(TITLES[cohort] + ' | exact 95% binomial intervals', fontweight='bold')
    fig.text(0.02, 0.025, cohort_note(cohort) + '\nSeparate sensitivity/specificity denominators; seed and patient uncertainty are distinct.', fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 0.92))
    save(fig, out, cohort + '_exact_intervals', synthetic)
    fig, axes = plt.subplots(2, 4, figsize=(13, 8.3))
    for ax, name in zip(axes.flat, names):
        m = d['models'][name]['clean']['metrics']
        cm = np.array(m['confusion_cn_ad'])
        den = cm.sum(1, keepdims=True)
        fraction = cm / den
        ax.imshow(fraction, cmap='Blues', vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if fraction[i, j] > 0.55 else 'black', fontsize=15)
        ax.set_xticks([0, 1], ['CN', 'AD'])
        ax.set_yticks([0, 1], ['CN', 'AD'])
        ax.set(title=DISPLAY[name], xlabel='Predicted', ylabel='Provider diagnosis')
    fig.suptitle(TITLES[cohort] + ' | patient confusion counts', fontweight='bold')
    fig.text(0.02, 0.02, cohort_note(cohort), fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94), h_pad=3)
    save(fig, out, cohort + '_confusion', synthetic)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    y = d['y']
    for name, color in zip(names, COLORS):
        z = d['models'][name]['clean']
        prob = z['prob'][:, 1]
        m = z['metrics']
        xx, yy, _ = roc_curve(y, prob, drop_intermediate=False)
        axes[0].plot(xx, yy, color=color, lw=2.5 if 'ensemble' in name else 1, ls='-' if 'ensemble' in name else '--', label=DISPLAY[name])
        yy, xx, _ = precision_recall_curve(y, prob)
        axes[1].step(xx, yy, where='post', color=color, lw=2.5 if 'ensemble' in name else 1, ls='-' if 'ensemble' in name else '--')
    axes[0].plot([0, 1], [0, 1], ls=':', color='gray')
    axes[0].set(xlabel='False-positive rate', ylabel='AD sensitivity')
    axes[1].axhline(y.mean(), ls=':', color='gray')
    axes[1].set(xlabel='Recall', ylabel='AD precision')
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1.02))
    fig.suptitle(TITLES[cohort] + ' | ROC and precision–recall', fontweight='bold')
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, loc='lower center', bbox_to_anchor=(0.5, 0.035), fontsize=8)
    fig.text(0.02, 0.014, 'Small fixed case-control samples; precision reflects sampled prevalence. All scores and uncertainty appear in analysis.json.', fontsize=8)
    fig.tight_layout(rect=(0, 0.16, 1, 0.93))
    save(fig, out, cohort + '_roc_pr', synthetic)
    for kind in ['reliability', 'confidence']:
        fig, axes = plt.subplots(2, 4, figsize=(14, 7.8))
        for ax, name, color in zip(axes.flat, names, COLORS):
            z = d['models'][name]['clean']
            m = z['metrics']
            prob = z['prob'][:, 1]
            if kind == 'reliability':
                bins = [b for b in m['reliability_bins'] if b['n']]
                ax.plot([0, 1], [0, 1], ls=':', color='gray')
                ax.scatter([b['mean_p_ad'] for b in bins], [b['observed_ad_fraction'] for b in bins], s=[20 + 150 * b['n'] / len(y) for b in bins], color=color)
                ax.text(0.04, 0.06, f"Brier {m['binary_brier']:.3f}\nNLL {m['nll']:.3f}", transform=ax.transAxes, fontsize=8)
                ax.set(xlim=(0, 1), ylim=(0, 1.03), xlabel='Mean P(AD) in fixed bin', ylabel='Observed AD fraction')
            else:
                ax.hist([prob[y == 0], prob[y == 1]], bins=np.linspace(0, 1, 11), stacked=True, color=['#7299bb', '#c77959'], label=['Provider CN', 'Provider AD'])
                ax.axvline(0.5, ls=':', color='black')
                ax.set(xlim=(0, 1), xlabel='Patient AD score', ylabel='Patient count')
                ax.legend(fontsize=7)
            ax.set_title(DISPLAY[name], fontsize=10)
        caption = 'Five fixed probability bins are descriptive; sparse bins and case-control prevalence do not establish clinical-risk calibration.' if kind == 'reliability' else 'Ten fixed histogram bins show patient scores by provider diagnosis; case-control score distributions do not establish clinical-risk calibration.'
        fig.suptitle(TITLES[cohort] + ' | ' + kind, fontweight='bold')
        fig.text(0.02, 0.02, caption, fontsize=9)
        fig.tight_layout(rect=(0, 0.07, 1, 0.94))
        save(fig, out, cohort + '_' + kind, synthetic)

def stress_figures(cohort, d, out, synthetic):
    plt = plotting()
    families = [('Intensity', ['clean', 'intensity_0.9', 'intensity_1.1'], ['Clean', '0.9', '1.1']), ('Noise sigma', ['clean', 'gaussian_noise_0.01', 'gaussian_noise_0.02'], ['Clean', '0.01', '0.02']), ('Blur radius', ['clean', 'blur_0.5', 'blur_1.0'], ['Clean', '0.5', '1.0'])]
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 10.8))
    for j, (family, scenarios, ticks) in enumerate(families):
        for i, metric in enumerate(['balanced_accuracy', 'label_flip_fraction', 'mean_probability_total_variation']):
            ax = axes[i, j]
            for name, color in zip(c.ORDER, COLORS):
                ax.plot(range(3), [d['models'][name][scenario]['metrics'][metric] for scenario in scenarios], color=color, lw=2.4 if 'ensemble' in name else 1, ls='-' if 'ensemble' in name else '--', marker='o', ms=3, label=DISPLAY[name])
            ax.set_xticks(range(3), ticks)
            ax.set_xlabel(family)
            ax.set_ylabel(metric.replace('_', ' '))
            ax.grid(alpha=0.2)
    fig.suptitle(TITLES[cohort] + ' | all 7 fixed pixel-stress conditions', fontweight='bold')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), ncol=4, loc='lower center', bbox_to_anchor=(0.5, 0.04), fontsize=8)
    fig.text(0.025, 0.015, 'Same patient, perturbation and clean reference for every model. These synthetic 2D controls do not simulate MRI acquisition physics.', fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    save(fig, out, cohort + '_all_stress', synthetic)

def analyze(data, draws=10000):
    analyses = {}
    for cohort, d in data.items():
        models = {n: d['models'][n]['clean']['metrics'] for n in c.ORDER}
        older = d['models']['round1_clinical_ensemble']['clean']['prob']
        paired = {}
        for name in ['new_clinical_ensemble'] + c.NEW:
            paired[name + '_minus_round1_clinical_ensemble'] = s.paired_comparison(d['y'], d['models'][name]['clean']['prob'], older, 'clinical_fixed_v1/' + cohort + '/' + name, draws=draws)
        spread = {}
        for condition, names in [('new', c.NEW), ('old', list(c.OLD))]:
            spread[condition] = {metric: dict(values={n: models[n][metric] for n in names}, mean=float(np.mean([models[n][metric] for n in names])), sample_sd=float(np.std([models[n][metric] for n in names], ddof=1))) for metric in ['accuracy', 'balanced_accuracy', 'sensitivity_ad', 'specificity_cn', 'nll', 'binary_brier', 'roc_auc']}
        analyses[cohort] = dict(patient_ids=d['keys'], models=models, paired_comparisons=paired, optimization_seed_spread=spread)
    return analyses
