import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import imageio 
import cartopy 
import cartopy.crs as ccrs
import numpy as np
from scipy.stats import binned_statistic, binned_statistic_dd
from matplotlib.animation import FuncAnimation, PillowWriter
import xarray as xr 

def to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def plot_psd(ax, x, y, minimum_freq_in_ccam = 12.5, lbl='CCAM', spt_nm='Spatial Domain'):
    if lbl == 'CCAM':
        ax.loglog(y[1:] / minimum_freq_in_ccam, np.mean(x, axis=0), lw=3, color='black', label=lbl)
    else:
        ax.loglog(y[1:] / minimum_freq_in_ccam, np.mean(x, axis=0), lw=3, label=lbl)
        
    frequency_ticks = [1/25, 1/50, 1/100, 1/500]
    [ax.axvline(frequency_ticks[i], color='k', ls='--', alpha=0.3) for i in range(len(frequency_ticks))]
    ax.legend(fontsize=12)
    ax.set_ylabel('PSD', weight='bold', fontsize=13)
    ax.set_xlabel('Frequency', weight='bold', fontsize=13)
    frequency_labels = [f'1/{int(1/tick)}' for tick in frequency_ticks]
    plt.xticks(frequency_ticks, frequency_labels)
    ax.set_title(f'Frequency Composition of {spt_nm}', fontsize=15, weight='bold')
    
    plt.savefig('metric_plots/freq.pdf')


def qq_plot(val_dict, loc_dict):
    def quantiles(arr, q_space=np.linspace(0.0, 0.99, 20)):
        return [np.quantile(arr, qs, method='linear').squeeze() for qs in q_space]
    
    def add_plot(axs, x, y, lbl='', color='grey'):
        if lbl == 'CCAM':
            axs.plot(x, y, color='grey', linestyle='--', linewidth=3, label=lbl)
        else:
            axs.plot(x, y, marker='o', linewidth=3, label=lbl, color=color)
            
    colors = ['#36213E', '#E0AFA0', '#557A95', '#D9BF77', '#A799B7', '#E0AFA0']

    for ln, mdl_dict in val_dict.items():
        y_true = val_dict[ln]['CCAM']
        y_true = y_true[y_true > 1]
        q_true = quantiles(y_true)
        ymax = np.max(q_true)
        ymin = np.min(q_true)
        
        fig, axs = plt.subplots()        
        # plot predictions 
        for (name, predictions), color in zip(mdl_dict.items(), colors):
            predictions = predictions[predictions > 1]
            q_pred = quantiles(predictions)
            qmax =  np.max(q_pred)
            qmin = np.min(q_pred)
            ymax = np.max([ymax, qmax])
            ymin = np.min([ymin, qmin])
            add_plot(axs, q_true, q_pred, lbl=name, color=color)
        
        # set title 
        xmax = np.max(q_true)
        xmin = np.min(q_true)
        llat, llon = loc_dict[ln]
        if llat is None or llon is None:
            title = f'Quantile Quantile Plot for {ln}'
        else:
            lat0, lat1, lon0, lon1 = llat[0], llat[-1], llon[0], llon[-1]
            title = f'Quantile Quantile Plot for {ln} ({lat0}-{lat1}, {lon0}-{lon1})'
        
        add_plot(axs, q_true, q_true, lbl='CCAM')
        axs.legend(fontsize=12)
        axs.set_xlim([xmin, xmax])
        axs.set_ylim([ymin, ymax])
        axs.tick_params(labelsize=12)
        axs.set_xlabel(r'Real', fontsize=13, weight='bold')
        axs.set_ylabel(r'Prediction', fontsize=13, weight='bold')
        axs.set_title(title, fontsize=15)

        plt.savefig(f'metric_plots/qq_{ln}.pdf')

def plot_hist(
                  val_dict,            # gives the prediction value 
                  color=None,
                  path='/home/bailiet/DDM/mlde/slurm/metric_plots/hist'
             ):
    
    def add_hist(z, ax, label, color=None, lw=1):
        hist, bins = np.histogram(z, bins=np.linspace(0.0, np.max(z), 25))
        ax.vlines(bins[0], 0, hist[0], label=label, color=color, linewidth=lw)
        ax.vlines(bins[-1], 0, hist[-1], color=color, linewidth=lw)

        for i, val in enumerate(hist):
            ax.hlines(val, bins[i], bins[i + 1], color=color, linewidth=lw)
            if (i < len(hist) - 1):
                ax.vlines(bins[i + 1], hist[i], hist[i + 1], color=color, linewidth=lw)
    
    colors = ['#36213E', '#557A95', '#D9BF77', '#A799B7', '#E0AFA0']
    for ln, mdl_dict in val_dict.items():
        fig, axs = plt.subplots() 
        for n, (lbl, arr) in enumerate(mdl_dict.items()):
            if lbl == 'CCAM':
                color = 'grey'
                lw = 2 
            else:
                color=colors[n%len(colors)]
                lw = 2
                
            add_hist(arr.ravel(), axs, lbl, color=color, lw=lw) 

        axs.legend(fontsize=13)
        axs.set_yscale('log')
        axs.set_xlabel("Precipitation (mm)", weight='bold', fontsize=13)
        axs.set_ylabel("Frequency", weight='bold', fontsize=13)
        axs.set_title(f"Precipitation Intensity over {ln}", fontsize=15, weight='bold')
        axs.tick_params(labelsize=12)
        plt.savefig(f'{path}_{ln}.pdf')

def to_numpy(x):
    return x.cpu().detach().numpy()

def sort_pairs(x, y):
    # sorts arrays x and y but reorders y based off x 
    sorted_pairs = sorted(zip(x, y))
    _, y_sorted = zip(*sorted_pairs)
    
    return y_sorted
    
def plot_stats(stat_dict=None):
    stat_names = ['MSE', 'PSNR']
    model_names = ['segfull_stat', 'fullstat', 'seg_stat']

    full_inten = np.load('full_inten.npy')
    seg_inten = np.load('seg_inten.npy')

    stat_dict = {sn: [sort_pairs(seg_inten, np.load(f'{mn}_{sn}.npy')) for mn in model_names] for sn in stat_names}
    seg_inten = np.sort(seg_inten)

    fig, axs = plt.subplots(2, 1)
    for n, (ax, sn) in enumerate(zip(axs.ravel(), stat_dict)):
        for mn, starr in zip(model_names, stat_dict[sn]):
            if 'seg' in mn:
                x = seg_inten

            ax.set_title(sn)
            ax.plot(x, starr, label=mn)
        ax.legend()

    plt.savefig('/home/bailiet/DDM/mlde/slurm/stats.pdf')

    fig, axs = plt.subplots(2, 1)
    axs[0].set_title('MSE')
    axs[0].plot(stat_dict['MSE'])
    axs[1].set_title('PSNR')
    axs[1].plot(stat_dict['PSNR'])
    plt.savefig('/home/bailiet/DDM/mlde/slurm/stat2s.pdf')
    

def plot_train_info(training_stats, flag=''):
    training_statistics, validation_statistics = training_stats
    
    # Iterate over training and validation statistics
    for (stat_name, train_stat), (_, val_stat) in zip(training_statistics.get_gen(), validation_statistics.get_gen()):
        
        if isinstance(train_stat, list):
            train_stat = torch.stack(train_stat)
            val_stat = torch.stack(val_stat)
        if isinstance(train_stat, torch.Tensor):
            train_stat = to_numpy(train_stat)
            val_stat = to_numpy(val_stat)

        
        # Create a new figure for plotting
        fig, ax = plt.subplots()
        
        # Plot the training and validation statistics
        ax.plot(train_stat, label=f'train_{stat_name}')
        ax.plot(val_stat, label=f'val_{stat_name}')
        
        # Add a legend to the plot
        ax.legend()
        
        # Save the figure to a PDF file
        plt.savefig(f'train_plots/{flag}_{stat_name}.pdf')
        
        # Close the plot to free up resources
        plt.close(fig)
        
def animation(flag='', nrepeat=1, ntimestep=1, frames_path=None, predictions=None):
    # if in .png style then use imageio to get GIF 
    if frame_paths is not None:
        for k in range(nrepeat):
            img_list = [imageio.imread(f"plots/{flag}_n{n}_k{k}.png") for n in range(ntimestep)]
            imageio.mimsave(f"gifs/{flag}_{k}.gif", img_list, duration=3, format='GIF')  # duration of each frame in seconds
    elif predictions is not None:
        pass 
        # plot predictions with matplotlib 
    else:
        raise ValueError('One of frames_path and predictions must not be None')
        
def single_plot(xlr, xhr, pred_xhr, coords, lat, lon, figax=None, cplot=None, cmap='Blues', save_name='', do_save=True):
    if figax is None:
        fig, axs = plt.subplots(1, 3, subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=171.77)))
    else:
        fig, axs = figax
            
    def mk_xr(x, coords, b, lat, lon):
        if b:
            x = to_numpy(x)
        if x.shape[0] == 2:
            x = x[0]
            
        if type(x) is xr.DataArray:
            x = x.rename({kw: v for kw, v in zip(x.coords, coords)})

        x = xr.DataArray(x, coords)
        if lat is not None and lon is not None:
            x = x.interp(lat=lat, lon=lon, method='nearest')
        
        return x 
    
    xlr = mk_xr(xlr.squeeze(), coords, True, lat, lon)
    xhr = mk_xr(xhr.squeeze(), coords, True, lat, lon)
    pred_xhr = mk_xr(pred_xhr.squeeze(), coords, False, lat, lon)
    
    plots = [] 
    for ax, x in zip(axs.ravel(), (xlr, xhr, pred_xhr)):
        plots.append(x.plot(ax=ax, cmap=cmap, add_colorbar=False, transform=ccrs.PlateCarree()))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.coastlines('10m')
    
    cbar_ax = fig.add_axes([0.92, 0.30, 0.02, 0.40])
    fig.colorbar(plots[-1] if cplot is None else cplot, cax=cbar_ax, cmap=cmap)
    
    axs[0].set_title('Low Res')
    axs[1].set_title('High Res')
    axs[2].set_title('Sample')
            
    if do_save:
        plt.savefig(f'/home/bailiet/DDM/mlde/slurm/{save_name}.pdf')
        plt.close()
        
    return plots  
        
def show_samples(config, eval_data_loader, samples, coords, lat, lon, nrepeat=1, nsample=1, flag=''):
    cmap = 'Blues'
    
    for n, (xlr, xhr), pred_xhr in zip(range(nsample), eval_data_loader, samples):
        xlr = xlr.to(config.device)
        xhr = xhr.to(config.device)
        name = f'{flag}_n{n}'
        single_plot(xlr, xhr, pred_xhr, coords, lat, lon, save_name=name)      
                    
def func_animation(eval_dl, predictions, coords, lat, lon, flag=''):
    fig, axs = plt.subplots(1, 3, subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=171.77)))
    eval_gen = ((xlr, xhr) for xlr, xhr in eval_dl)
    cplot = None 
    
    def update(data):
        _, pred_xhr, (xlr, xhr) = data
        plots = single_plot(xlr, xhr, pred_xhr, coords, lat, lon, figax=(fig,axs), cplot=None, cmap='Blues', do_save=False)
        return plots 

    ani = FuncAnimation(fig, update, frames=zip(range(100), predictions, eval_gen), blit=False, repeat=True)
    writer = PillowWriter(fps=1)
    ani.save(f'{flag}.gif', writer=writer)