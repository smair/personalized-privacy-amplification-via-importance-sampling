import os.path
import numpy as np
import matplotlib.pylab as plt
import matplotlib.ticker as mtick

results_path = './results/'

lam2name = {0.5:'core',
            1.0:'unif',
            -1.0:'opt'}

# https://scottplot.net/cookbook/4.1/colors/#colorblind-friendly
color = {'unif':'#0072B2',
        'core':'#009E73',
        'opt':'#E69F00'}

hatch = {'unif':'*',
        'core':'/',
        'opt':'o'}

line_style = {'full':':',
            'unif':'-',
            'core':'--',
            'opt':'-.'}

dataset_size = {'ijcnn1':48740,
                'pose':34936,
                'miniboone':126812,
                'kdd-protein':142107,
                'rna':476350,
                'song':502461,
                'covertype':566486,
                'fma':103909,
                'sun-attribute':13981}

dataset_name_fix = {'ijcnn1':'Ijcnn1',
                    'pose':'Pose',
                    'miniboone':'MiniBooNE',
                    'kdd-protein':'KDD-Protein',
                    'rna':'RNA',
                    'song':'Song',
                    'covertype':'Covertype',
                    'fma':'FMA',
                    'sun-attribute':'SUN-Attribute'}


def get_full(dataset):
    k = 25
    eps = 3.0
    p = 97.5
    T = 10
    norm = 2

    print(f'setting: dataset={dataset} k={k} p={p} T={T} norm={norm} eps={eps}')

    filename = f'result_{dataset}_{k}_-1_1.0_{norm}_{eps}_{T}_{p}_1.npz'
    npz = np.load(os.path.join(results_path, filename))
    res_obj = npz['res_obj']
    res_time = npz['res_time']
    print(f'full                  obj={np.median(res_obj[:,-1]):.2f} time_KM={np.median(res_time[:,-1]):.2f}')
    time_full = np.median(res_time[:,-1])
    return time_full


#
# Figure 5 (right)
#

k = 25
m = 20000
eps = 3.0
p = 97.5
reps = 50
T = 10
norm = 2
L = [1.0, 0.5, -1.0]
lam = 1.0

dataset = 'covertype'

bar_full = get_full(dataset)

bar_kmeans = []
bar_weights = []
bar_sample = []
for lam in L:
    filename = f'result_{dataset}_{k}_{m}_{lam}_{norm}_{eps}_{T}_{p}_{reps}.npz'
    npz = np.load(os.path.join(results_path, filename))
    res_C = npz['res_C']
    res_obj = npz['res_obj']
    res_obj_sub = npz['res_obj_sub']
    res_time = npz['res_time']
    res_time_weights = npz['res_time_weights']
    res_time_sample = npz['res_time_sample']
    # take the new time for optimal
    if lam == -1.0:
        print('old', res_time_weights.flatten()[-1])
        filename = filename.replace('result', 'weights')
        npz = np.load(os.path.join(results_path, filename))
        res_time_weights = npz['res_time_weights']
        print('new', res_time_weights.flatten()[-1])
    bar_kmeans.append(np.median(res_time[:,-1]))
    bar_weights.append(res_time_weights.flatten()[-1])
    bar_sample.append(np.median(res_time_sample))
    print(f'm={m} sample={lam2name[lam]} obj={np.median(res_obj[:,-1]):.2f} time_KM={np.median(res_time[:,-1]):.2f} time_weights={res_time_weights.flatten()[-1]:.4f} time_sample={np.median(res_time_sample):.4f}')

bar_kmeans = np.array(bar_kmeans) / bar_full * 100.0
bar_weights = np.array(bar_weights) / bar_full * 100.0
bar_sample = np.array(bar_sample) / bar_full * 100.0


fig, ax = plt.subplots(figsize=(5.0, 4.0))
ax.grid(zorder=0)
ax.set_title(f'{dataset_name_fix[dataset]}   $m$={m}')
ax.set_ylabel('Total relative computation time [%]')
ax2 = ax.twinx()
ax2.set_ylabel('Relative subset size [%]')
ax.axhline(m/dataset_size[dataset]*100, zorder=0, ls=':', c='k')
bar1 = ax.bar([lam2name[l] for l in L], bar_kmeans, facecolor='#8FBC8F', hatch='', label='DP-Lloyd', zorder=3)
bar2 = ax.bar([lam2name[l] for l in L], bar_sample, bottom=bar_kmeans, hatch='', color='#966FD6', label='Subset sampling', zorder=3)
bar3 = ax.bar([lam2name[l] for l in L], bar_weights, bottom=bar_kmeans+bar_sample, hatch='', color='#779ECB', label='Weight computation', zorder=3)
ax2.plot([], [], 'k:', label=r'$m/n_{\operatorname{data}}$')
ax2.set_ylim(ax.get_ylim())
ax2.legend(loc=4)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1])
fig.savefig(f'/tmp/timing_{dataset}.pdf', dpi=300, transparent=True, bbox_inches='tight')
fig.savefig(f'/tmp/timing_{dataset}.png', dpi=300, transparent=False, bbox_inches='tight')


#
# Figure 5 (left)
#

fig, ax = plt.subplots(figsize=(5.0, 4.0))
ax.grid()
ax.set_xlim(2500, 52000)
ax.set_ylim(-0.5, 32)
ax.set_xlabel('Subsample size $m$')
ax.set_ylabel('Total relative computation time [%]')
ax2 = ax.twinx()
ax2.set_ylabel('Relative subset size [%]')
for dataset in ['covertype','ijcnn1','pose','kdd-protein','fma']:
    time_full = get_full(dataset)
    print(f'{dataset_name_fix[dataset]} & full & $n$ & - & - & {time_full:.2f} & {time_full:.2f} & 100 & 100 \\\\ ')
    for lam in L:
        plot_m = []
        plot_rel_time = []
        for m in [3000, 4000, 5000, 6000, 7000, 8000, 10000, 15000, 20000, 50000]:
            filename = f'result_{dataset}_{k}_{m}_{lam}_{norm}_{eps}_{T}_{p}_{reps}.npz'
            if os.path.exists(os.path.join(results_path, filename)):
                npz = np.load(os.path.join(results_path, filename))
                res_obj = npz['res_obj']
                res_time = npz['res_time']
                res_time_weights = npz['res_time_weights']
                res_time_sample = npz['res_time_sample']
                # take the new time for optimal
                if lam == -1.0:
                    # print('old', res_time_weights.flatten()[-1])
                    filename = filename.replace('result', 'weights')
                    npz = np.load(os.path.join(results_path, filename))
                    res_time_weights = npz['res_time_weights']
                    # print('new', res_time_weights.flatten()[-1])
                total_time = res_time_weights.flatten()[-1] + np.median(res_time_sample) + np.median(res_time[:,-1])
                # print(f'{dataset} lam={lam2name[lam]} m={m} n={dataset_size[dataset]} full={time_full:.4f} weights={res_time_weights.flatten()[-1]:.4f} sample={np.median(res_time_sample):.4f} DPKM={np.median(res_time[:,-1]):.4f} total={total_time:.4f}')
                print(f'{dataset_name_fix[dataset]} & {lam2name[lam]} & {m} & {res_time_weights.flatten()[-1]:.4f} & {np.median(res_time_sample):.4f} & {np.median(res_time[:,-1]):.2f} & {total_time:.2f} & {total_time / time_full * 100.0:.2f} & {m / dataset_size[dataset] * 100:.2f} \\\\ ')
                rel_total_time = total_time / time_full * 100.0
                plot_rel_time.append(rel_total_time)
                plot_m.append(m)
        # if lam == -1.0:
        #     print(f'{dataset} {plot_rel_time[-1]-plot_m[-1]/dataset_size[dataset]*100}')
        ax.plot(plot_m, plot_rel_time, ls=line_style[lam2name[lam]], color=color[lam2name[lam]]) #, label=lam2name[lam])
        ax.plot(plot_m, np.array(plot_m)/dataset_size[dataset]*100, 'k:')
for lam in L:
    ax.plot([], [], ls=line_style[lam2name[lam]], color=color[lam2name[lam]], label=lam2name[lam])
ax.legend(loc=2, ncol=3)
ax2.plot([], [], 'k:', label=r'$m/n_{\operatorname{data}}$')
ax2.set_ylim(ax.get_ylim())
ax2.legend(loc=1)
# ax.annotate('RNA', [46000, 12], [46000, 12])
ax.annotate('Ijcnn1', [14000, 26.5], [14000, 26.5])
ax.annotate('Pose', [3500, 23.5], [3500, 23.5])
ax.annotate('FMA', [13500, 15.5], [13500, 15.5])
ax.annotate('KDD-Protein', [20500, 15.5], [20500, 15.5])
# ax.annotate('Song', [45000, 5.75], [45000, 5.75])
# ax.annotate('Covertype', [42500, 3.75], [42500, 3.75])
# ax.annotate('MiniBooNE', [7500, 4], [7500, 4])
ax.annotate('Covertype', [42500, 10.5], [42500, 10.5])
# beginning (and end) of line has a marker, marker per data set
fig.savefig('/tmp/timing_full.pdf', dpi=300, transparent=True, bbox_inches='tight')
fig.savefig('/tmp/timing_full.png', dpi=300, transparent=False, bbox_inches='tight')



