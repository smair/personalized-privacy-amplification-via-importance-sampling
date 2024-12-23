import os.path
import numpy as np
import matplotlib.pylab as plt

import matplotlib.patches as patches

results_path = './results/'

dataset_name_fix = {'ijcnn1':'Ijcnn1',
                    'pose':'Pose',
                    'miniboone':'MiniBooNE',
                    'kdd-protein':'KDD-Protein',
                    'rna':'RNA',
                    'song':'Song',
                    'covertype':'Covertype',
                    'fma':'FMA'}

dataset_size = {'ijcnn1':48740,
                'pose':34936,
                'miniboone':126812,
                'kdd-protein':142107,
                'rna':476350,
                'song':502461,
                'covertype':566486,
                'fma':103909}

lam2name = {0.5:'core',
            1.0:'unif',
            -1.0:'opt'}

lam2name_nonpriv = {0.5:'core_nonpriv',
                    1.0:'unif_nonpriv'}
lam2name_nonpriv_legend = {0.5:'core $\epsilon=\infty$',
                           1.0:'unif $\epsilon=\infty$'}

# https://scottplot.net/cookbook/4.1/colors/#colorblind-friendly
color = {'unif':'#0072B2',
        'core':'#009E73',
        'opt':'#E69F00',
        'full':'black',
        'unif_nonpriv':'#56B4E9',
        'core_nonpriv':'black'}

hatch = {'unif':'*',
        'core':'/',
        'opt':'o',
        'full':'.',
        'unif_nonpriv':'*',
        'core_nonpriv':'/'}

line_style = {'unif':':',
            'core':'--',
            'opt':'-.',
            'full':'-',
            'unif_nonpriv':':',
            'core_nonpriv':'--'}

m_for_dataset = {'kdd-protein':[5000, 10000, 15000], # max m = 15532
                 'rna':[5000, 10000, 15000, 20000, 50000], # max m = 67033
                 'song':[5000, 10000, 15000, 20000, 50000], # max m = 60353
                 'covertype':[5000, 10000, 15000, 20000, 50000], # max m = 107935
                 'ijcnn1':[3000, 4000, 5000, 6000, 7000, 10000, 15000, 20000], # max m = 26972
                 'pose':[3000, 4000, 5000, 6000, 7000, 8000], # max m = 16740
                 'miniboone':[4000, 5000, 6000, 7000], # max m = 7662
                 'fma':[5000, 10000, 15000], # max m = 16740
                 'sun-attribute':[3000, 4000, 5000, 6000],} # max m = 6456


q_lower = 0.25
q_upper = 0.75

norm = 2
T = 10
p = 97.5
k = 25
reps = 50


###############################################################################


location = ['appendix','main']

m_for_dataset = {'kdd-protein':[5000, 10000, 15000], # max m = 15532
                 'rna':[5000, 20000, 50000], # max m = 67033
                 'song':[5000, 20000, 50000], # max m = 60353
                 'covertype':[5000, 20000, 75000], # max m = 107935
                 'ijcnn1':[5000, 10000, 15000], # max m = 26972
                 'pose':[3000, 5000, 8000], # max m = 16740
                 'miniboone':[4000, 5000, 7000], # max m = 7662
                 'fma':[5000, 10000, 15000], # max m = 16740
                 'sun-attribute':[3000, 5000, 6000],} # max m = 6456

plot_main = True
plot_main = False

if plot_main:
    datasets = ['kdd-protein', 'rna', 'song', 'covertype']
else:
    datasets = ['ijcnn1', 'pose', 'miniboone', 'fma'] #, 'sun-attribute']

fig, axv = plt.subplots(3, len(datasets), sharex='col', figsize=(10.0, 8.0))
for ax in axv.flatten():
    ax.grid()
    ax.set_xscale('log')
    ax.set_xticks([10**k for k in range(-1, 4)])
for ax in axv[:,0]:
    ax.set_ylabel('Obj. func. on all data')
for ax in axv[-1,:]:
    ax.set_xlabel('$\epsilon$')
for j in range(len(datasets)):
    dataset = datasets[j]
    # for i in range(len(M)):
    #     m = M[i]
    for i in range(len(m_for_dataset[dataset])):
        m = m_for_dataset[dataset][i]

        ax = axv[i,j]
        if i==0:
            ax.set_title(f'{dataset_name_fix[dataset]}\n$m$={m}')
        else:
            ax.set_title(f'$m$={m}')


        # w/o privacy
        for lam in [1.0, 0.5]:
            filename = f'result_{dataset}_{k}_{m}_{lam}_{norm}_kMeans_{T}_{p}_{reps}.npz'
            if os.path.exists(os.path.join(results_path, filename)):
                npz = np.load(os.path.join(results_path, filename))
                if 'res_obj' in npz:
                    res_obj = npz['res_obj'] / dataset_size[dataset]
                    print(f'dataset={dataset} m={m} lam={lam} kMeans found and complete')
                    plot_eps = [0.5, 1.0, 3.0, 10.0, 50.0, 100.0, 300.0, 1000.0]
                    lower = np.ones_like(plot_eps) * np.nanquantile(res_obj[:,-1], q=q_lower)
                    upper = np.ones_like(plot_eps) * np.nanquantile(res_obj[:,-1], q=q_upper)
                    median = np.ones_like(plot_eps) * np.nanmedian(res_obj[:,-1])
                    ax.fill_between(plot_eps, lower, upper, alpha=0.1, color=color[lam2name_nonpriv[lam]], hatch=hatch[lam2name_nonpriv[lam]])
                    ax.plot(plot_eps, median, ls=line_style[lam2name_nonpriv[lam]], color=color[lam2name_nonpriv[lam]])
                else:
                    print(f'm={m} lam={lam} kMeans run not finished or problematic: {filename}')
            else:
                print(f'm={m} lam={lam} kMeans run not found: {filename}')


        for lam in [1.0, 0.5, -1.0]:
            plot_eps = []
            plot_obj_median = []
            plot_obj_lower = []
            plot_obj_upper = []
            for eps in [0.5, 1.0, 3.0, 10.0, 50.0, 100.0, 300.0, 1000.0]:
                filename = f'result_{dataset}_{k}_{m}_{lam}_{norm}_{eps}_{T}_{p}_{reps}.npz'
                if os.path.exists(os.path.join(results_path, filename)):
                    npz = np.load(os.path.join(results_path, filename))
                    if 'res_obj' in npz:
                        res_obj = npz['res_obj'] / dataset_size[dataset]
                        # print(f'm={m} lam={lam} eps={eps} found and complete')
                        plot_eps.append(eps)
                        plot_obj_median.append(np.median(res_obj[:,-1]))
                        plot_obj_lower.append(np.quantile(res_obj[:,-1], q=q_lower))
                        plot_obj_upper.append(np.quantile(res_obj[:,-1], q=q_upper))
                    else:
                        print(f'm={m} lam={lam} eps={eps} run not finished or problematic: {filename}')
                else:
                    print(f'm={m} lam={lam} eps={eps} run not found: {filename}')
            ax.fill_between(plot_eps, plot_obj_lower, plot_obj_upper, alpha=0.1, color=color[lam2name[lam]], hatch=hatch[lam2name[lam]])
            ax.plot(plot_eps, plot_obj_median, ls=line_style[lam2name[lam]], color=color[lam2name[lam]])

i,j = 0,0
if plot_main:
    i,j = 1,1
handles = []
labels = []
for lam in [1.0, 0.5, -1.0]:
    fill = patches.Patch(color=color[lam2name[lam]], alpha=0.1, hatch=hatch[lam2name[lam]])
    line = axv[i,j].plot([], [], ls=line_style[lam2name[lam]], color=color[lam2name[lam]])
    handles.append((fill, line[0]))
    labels.append(lam2name[lam])
for lam in [1.0, 0.5]:
    fill = patches.Patch(color=color[lam2name_nonpriv[lam]], alpha=0.1, hatch=hatch[lam2name_nonpriv[lam]])
    line = axv[i,j].plot([], [], ls=line_style[lam2name_nonpriv[lam]], color=color[lam2name_nonpriv[lam]])
    handles.append((fill, line[0]))
    labels.append(lam2name_nonpriv_legend[lam])
legend = axv[i,j].legend(handles=handles, labels=labels, prop={'size':9},
                         labelspacing=1.2, handlelength=4)

for patch in legend.get_patches():
    patch.set_height(15)
    patch.set_y(-4)

fig.tight_layout()

fig.savefig(f'/tmp/results_fix-m_{location[plot_main]}.pdf', dpi=300, transparent=True, bbox_inches='tight')
fig.savefig(f'/tmp/results_fix-m_{location[plot_main]}.png', dpi=300, transparent=False, bbox_inches='tight')



###############################################################################

m_for_dataset = {'kdd-protein':[5000, 10000, 15000], # max m = 15532
                 'rna':[5000, 10000, 15000, 20000, 50000], # max m = 67033
                 'song':[5000, 10000, 15000, 20000, 50000], # max m = 60353
                 'covertype':[5000, 10000, 15000, 20000, 50000, 75000], # max m = 107935
                 'ijcnn1':[3000, 4000, 5000, 6000, 7000, 10000, 15000, 20000], # max m = 26972
                 'pose':[3000, 4000, 5000, 6000, 7000, 8000], # max m = 16740
                 'miniboone':[4000, 5000, 6000, 7000], # max m = 7662
                 'fma':[5000, 10000, 15000], # max m = 16740
                 'sun-attribute':[3000, 4000, 5000, 6000],} # max m = 6456

eps_values = [3.0, 100.0]

plot_main = True
plot_main = False

if plot_main:
    datasets = ['kdd-protein', 'rna', 'song', 'covertype']
else:
    datasets = ['ijcnn1', 'pose', 'miniboone', 'fma'] #, 'sun-attribute']

fig, axv = plt.subplots(2, len(datasets), sharex='col', figsize=(10.0, 8.0*2/3))
for ax in axv.flatten():
    ax.grid()
for ax in axv[:,0]:
    ax.set_ylabel('Obj. func. on all data')
for ax in axv[-1,:]:
    ax.set_xlabel('Subsample size $m$')
for j in range(len(datasets)):
    dataset = datasets[j]
    for i in range(len(eps_values)):
        eps = eps_values[i]

        ax = axv[i,j]
        if i==0:
            ax.set_title(f'{dataset_name_fix[dataset]}\n$\epsilon$={eps}')
        else:
            ax.set_title(f'$\epsilon$={eps}')

        if 50000 in m_for_dataset[dataset]:
            ax.set_xticks([5000, 25000, 50000])
        if 75000 in m_for_dataset[dataset]:
            ax.set_xticks([5000, 25000, 50000, 75000])

        # full w/ privacy
        filename = f'result_{dataset}_{k}_-1_1.0_{norm}_{eps}_{T}_{p}_{reps}.npz'
        if os.path.exists(os.path.join(results_path, filename)):
            npz = np.load(os.path.join(results_path, filename))
            if 'res_obj' in npz:
                res_obj = npz['res_obj'] / dataset_size[dataset]
                # print(f'dataset={dataset} m=full lam=1.0 eps={eps} found and complete')
                plot_m = [m_for_dataset[dataset][0], m_for_dataset[dataset][-1]]
                lower = np.ones_like(plot_m) * np.nanquantile(res_obj[:,-1], q=q_lower)
                upper = np.ones_like(plot_m) * np.nanquantile(res_obj[:,-1], q=q_upper)
                median = np.ones_like(plot_m) * np.nanmedian(res_obj[:,-1])
                ax.fill_between(plot_m, lower, upper, alpha=0.1, color=color['full'], hatch=hatch['full'])
                ax.plot(plot_m, median, ls=line_style['full'], color=color['full'])
            else:
                print(f'm={m} lam={lam} eps={eps} run not finished or problematic: {filename}')
        else:
            print(f'm={m} lam={lam} eps={eps} run not found: {filename}')


        for lam in [1.0, 0.5, -1.0]:
            plot_m = []
            plot_obj_median = []
            plot_obj_lower = []
            plot_obj_upper = []
            for m in m_for_dataset[dataset]:
                filename = f'result_{dataset}_{k}_{m}_{lam}_{norm}_{eps}_{T}_{p}_{reps}.npz'
                if os.path.exists(os.path.join(results_path, filename)):
                    npz = np.load(os.path.join(results_path, filename))
                    if 'res_obj' in npz:
                        res_obj = npz['res_obj'] / dataset_size[dataset]
                        # print(f'm={m} lam={lam} eps={eps} found and complete')
                        plot_m.append(m)
                        plot_obj_median.append(np.median(res_obj[:,-1]))
                        plot_obj_lower.append(np.quantile(res_obj[:,-1], q=q_lower))
                        plot_obj_upper.append(np.quantile(res_obj[:,-1], q=q_upper))
                    else:
                        print(f'm={m} lam={lam} eps={eps} run not finished or problematic: {filename}')
                else:
                    print(f'm={m} lam={lam} eps={eps} run not found: {filename}')
            ax.fill_between(plot_m, plot_obj_lower, plot_obj_upper, alpha=0.1, color=color[lam2name[lam]], hatch=hatch[lam2name[lam]])
            ax.plot(plot_m, plot_obj_median, ls=line_style[lam2name[lam]], color=color[lam2name[lam]])

i,j = 0,0
if plot_main:
    i,j = 1,1
handles = []
labels = []
for lam in [1.0, 0.5, -1.0]:
    fill = patches.Patch(color=color[lam2name[lam]], alpha=0.1, hatch=hatch[lam2name[lam]])
    line = axv[i,j].plot([], [], ls=line_style[lam2name[lam]], color=color[lam2name[lam]])
    handles.append((fill, line[0]))
    labels.append(lam2name[lam])
fill = patches.Patch(color=color['full'], alpha=0.1, hatch=hatch['full'])
line = axv[i,j].plot([], [], ls=line_style['full'], color=color['full'])
handles.append((fill, line[0]))
labels.append('full')
legend = axv[i,j].legend(handles=handles, labels=labels, prop={'size':9},
                         labelspacing=1.2, handlelength=4)

for patch in legend.get_patches():
    patch.set_height(15)
    patch.set_y(-4)

fig.tight_layout()

fig.savefig(f'/tmp/results_fix-eps_{location[plot_main]}.pdf', dpi=300, transparent=True, bbox_inches='tight')
fig.savefig(f'/tmp/results_fix-eps_{location[plot_main]}.png', dpi=300, transparent=False, bbox_inches='tight')


