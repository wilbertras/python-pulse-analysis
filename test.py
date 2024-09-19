import numpy as np
import matplotlib.pyplot as plt
import functions as f
from scipy.signal import fftconvolve
import matplotlib as mpl


try:
    plt.style.use('matplotlibrc')
except:
    print('mpl stylesheet not used')
    pass

def pulse_analysis(dir, kid, pread, file_type, chuncksize, nr_chuncks, pw, pw_offset, filter, lifetime, mph, mpp, iterate=True, exclude_dc=True, plot=False, tmax=5, coord='smith', response='phase', fit_tqp=None):
    pulse_files, info_files = f.get_files(dir, kid, pread, type=file_type)
    nr_files = len(pulse_files)
    f0, Q, Qc, Qi, S21_min, dt, T = f.get_info(info_files[0])
    sff = 1 / dt / 1e6

    pw = int(pw * sff)
    pw_offset = int(pw_offset * sff)
    sw = int(lifetime * sff)
    if not fit_tqp:
        fit_tqp = [2*pw_offset, 3*pw_offset]
    if nr_chuncks:
        nr_req_files = np.amin((chuncksize * nr_chuncks, nr_files))
    else:
        nr_req_files = nr_files

    window = f.get_window(filter, sw)
    analysed_files = 0
    while analysed_files < nr_req_files:
        if analysed_files + chuncksize > nr_files:
            chuncksize = nr_files - analysed_files
        amp, phase, _ = f.get_data(pulse_files[analysed_files:analysed_files+chuncksize])
        signal = f.coord_transformation(phase, amp, coord=coord, response=response)

        if analysed_files == 0:
            std = f.get_sigma(signal, window)
            ph = mph*std
            pp = mpp*std
            pulses = []
            single_idx = []
            too_high_idx = []

        locs, props = f.find_pks(signal, ph[0], pp, window)
        too_high_chunck = props['peak_heights'] >= ph[1]
        args = np.argwhere(~too_high_chunck).flatten()
        pulses_chunck, single_idx_chunck = f.get_single_pulses(signal, locs, pw, pw_offset, args)
        pulses.append(pulses_chunck)
        single_idx.append(single_idx_chunck)
        too_high_idx.append(too_high_chunck)
        if analysed_files==0:
            noises = f.get_single_noises(signal, locs, pw+pw_offset)
            _, noise_psd = f.get_avg_psd(noises, pw+pw_offset, sff, exclude_dc=False)
        analysed_files += chuncksize
        print('Analysed %d out of %d files' % (analysed_files, nr_req_files), end='\r')
        if iterate:
            fig, ax = plt.subplots()
            ax.plot(window)
            mean_pulse = np.mean(pulses_chunck, axis=0)
            ax.plot(mean_pulse/np.sum(mean_pulse))
            # window = f.opt_filter(mean_pulse, noise_psd, exclude_dc=False)
            window = mean_pulse[::-1]
            window /= np.sum(window)
            ax.plot(window)
            plt.show()
            analysed_files = 0
            iterate -= 1


    t_file = len(signal)*dt / chuncksize
    t_files = analysed_files*t_file
    single_idx = np.hstack(single_idx)
    pulses = np.vstack(pulses)
    too_high_idx = np.hstack(too_high_idx)
    
    nr_total = len(single_idx)
    nr_single = np.sum(single_idx)
    nr_too_high = np.sum(too_high_idx)
    nr_too_close = np.sum(~single_idx) - nr_too_high

    Nph_total = nr_total / t_files
    Nph_single = nr_single / t_files
    Nph_too_close = nr_too_close / t_files
    Nph_too_high = nr_too_high / t_files

    perc_single = nr_single/nr_total * 100
    perc_too_close = nr_too_close/nr_total * 100
    perc_too_high = nr_too_high/nr_total * 100
    print('Found %d single pulses (%d%% of all detected pulses)' % (nr_single, perc_single))
    freqs, pulse_psd = f.get_avg_psd(pulses, pw+pw_offset, sff, exclude_dc=False)
    pulse_template = np.mean(pulses, axis=0)
    H_opt, R_sn, _, _ = f.optimal_filter(pulses, pulse_template, sff*1e6, 1, noise_psd)
    binedges = np.histogram_bin_edges(H_opt, bins='auto')
    pulse_binsize = binedges[1] - binedges[0]
    H_0, _, _, _ = f.optimal_filter(noises, pulse_template, sff*1e6, 1, noise_psd)
    R_opt, pdf_y, pdf_x, _, _ = f.resolving_power(H_opt, pulse_binsize)

    tqp, _, popt = f.fit_decaytime(pulse_template, pw, fit_tqp)
    tqp /= sff
    tqp_x = np.linspace(fit_tqp[0]/sff, fit_tqp[1]/sff, int((fit_tqp[1] - fit_tqp[0])))
    fit_x = np.arange(len(tqp_x))
    tqp_y = f.exp_decay(fit_x, *popt)

    if plot:
        max = int(tmax/dt)+1
        plot_signal = signal[:max]
        t = np.linspace(0, tmax, len(plot_signal))
        if len(window):
            smoothed_plot_signal = fftconvolve(plot_signal, window, mode='valid')
            window_offset = int(np.argmax(window[::-1]))
            t_smooth = t[window_offset:-len(window)+window_offset+1]
        t_pulse = np.linspace(0, (pw+pw_offset)*dt, len(pulse_template)) * 1e6
        single_locs = locs[single_idx_chunck]
        too_close_locs = locs[(~single_idx_chunck) & (~too_high_chunck)]
        too_high_locs = locs[too_high_chunck]
        single_locs = single_locs[single_locs < max]
        too_close_locs = too_close_locs[too_close_locs < max]
        too_high_locs = too_high_locs[too_high_locs < max]
        title = 'KID%d, P%d' % (kid, pread)
        
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        b, o, y, p, g, lb, r = colors
        pulse_color = b
        noise_color = 'tab:gray'
        smooth_color = o
        single_color = 'tab:green'
        not_single_color = r
        mp_color = r

        fig, axes = plt.subplot_mosaic('bbbf;aecd', figsize=(12, 6), constrained_layout=True, num=str(dir))
        fig.suptitle(title)
        
        ax = axes['a']
        ax.plot(t_pulse, pulse_template, c=pulse_color, label='avg. pulse', zorder=2)
        ax.plot(t_pulse, np.mean(noises, axis=0), c=noise_color, label='avg. noise', zorder=1)
        ax.set_xlabel('t [$\mu$s]')
        ax.set_ylabel(str(response))
        ax.set_xlim([t_pulse[0], t_pulse[-1]])
        ax.legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left',
                ncols=2, mode="expand", borderaxespad=0., fontsize=9)

        ax = axes['f']
        ax.plot(window[::-1], c=pulse_color, label='avg. pulse', zorder=2)
        ax.set_xlabel('t [$\mu$s]')
        # ax.set_ylabel(str(response))
        # ax.set_ylim([1e-3, (np.amax(pulse_template)//.1+1)*.1])
        # ax.set_xlim([t_pulse[0], t_pulse[-1]])
        # ax.legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left',
        #         ncols=2, mode="expand", borderaxespad=0., fontsize=9)
        
        ax = axes['e']
        ax.semilogy(t_pulse, pulse_template, c=pulse_color, label='avg. pulse', zorder=2)
        ax.semilogy(tqp_x, tqp_y, c=smooth_color, ls='--', label='$\\tau_{qp}$=%d $\mu$s' % tqp, zorder=3)
        ax.set_xlabel('t [$\mu$s]')
        ax.set_ylabel(str(response))
        ax.set_ylim([1e-3, (np.amax(pulse_template)//.1+1)*.1])
        ax.set_xlim([t_pulse[0], t_pulse[-1]])
        ax.legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left',
                ncols=2, mode="expand", borderaxespad=0., fontsize=9)

        ax = axes['b']
        if len(window):
            ax.plot(t, plot_signal, lw=.1, c=pulse_color, zorder=0, alpha=.5, label='response')
            ax.plot(t_smooth, smoothed_plot_signal, lw=.5, zorder=1, c=smooth_color, label='smoothed response')
            ax.scatter(t[single_locs], smoothed_plot_signal[single_locs-window_offset], facecolor='None', edgecolor=single_color, marker='v', zorder=2, label='single pulses $N_{ph}$=%d cps' % Nph_single)
            ax.scatter(t[too_close_locs], smoothed_plot_signal[too_close_locs-window_offset], facecolor='None', edgecolor=not_single_color, marker='o', zorder=2, label='%d%% too close' % (perc_too_close))
            ax.scatter(t[too_high_locs], smoothed_plot_signal[too_high_locs-window_offset], facecolor='None', edgecolor=not_single_color, marker='v', zorder=2, label='%d%% too high' % (perc_too_high))
        else:
            ax.plot(t, plot_signal, lw=.5, c=pulse_color, zorder=0, alpha=1, label='response')
            ax.scatter(t[single_locs], plot_signal[single_locs], facecolor='None', edgecolor=single_color, marker='v', zorder=2, label='single pulses $N_{ph}$=%d cps' % Nph_single)
            ax.scatter(t[too_close_locs], plot_signal[too_close_locs], facecolor='None', edgecolor=not_single_color, marker='o', zorder=2, label='%d%% too close' % (perc_too_close))
            ax.scatter(t[too_high_locs], plot_signal[too_high_locs], facecolor='None', edgecolor=not_single_color, marker='v', zorder=2, label='%d%% too high' % (perc_too_high))
        if isinstance(mph, np.ndarray):
            ax.axhline(mph[0]*std, zorder=2, c=mp_color, lw=.5)
            ax.axhline(mph[1]*std, zorder=2, c=mp_color, lw=.5, label='mph=[%d, %d]$\\sigma$' % (mph[0], mph[1]))
        else:
            ax.axhline(mph*std, zorder=2, c=mp_color, lw=.5, label='mph=%d$\\sigma$' % (mph))
        if isinstance(mpp, np.ndarray):
            ax.axhline(mpp[0]*std, zorder=2, c=mp_color, lw=.5, ls='--')
            ax.axhline(mpp[1]*std, zorder=2, c=mp_color, lw=.5, ls='--', label='mpp=[%d, %d]$\\sigma$' % (mpp[0], mpp[1]))
        else:
            ax.axhline(mpp*std, zorder=2, c=mp_color, lw=.5, ls='--', label='mpp=%d$\\sigma$' % (mpp))
        ax.axhline(mpp*std, zorder=2, c=mp_color, ls='--', lw=.5)
        ax.set_xlabel('t [s]')
        ax.set_ylabel(str(response))
        ax.set_xlim([t[0], t[-1]])
        ax.legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left',
                ncols=6, mode="expand", borderaxespad=0., fontsize=9)

        ax = axes['c']
        ax.semilogx(freqs[exclude_dc:], 10*np.log10(pulse_psd[exclude_dc:]), c=pulse_color,label='avg. pulse', zorder=1)
        ax.semilogx(freqs[exclude_dc:], 10*np.log10(noise_psd[exclude_dc:]), c=noise_color, label='noise', zorder=0)
        ax.set_xlabel('f [Hz]')
        ax.set_ylabel('$S_{xx}$ [dBc/Hz]')
        ax.set_xlim([freqs[0], sff*1e6/2])
        ax.grid(which='major', lw=0.5)
        ax.grid(which='minor', lw=0.2)
        ax.xaxis.get_major_locator().set_params(numticks=99)
        ax.xaxis.get_minor_locator().set_params(numticks=99, subs=np.arange(2, 10, 2)*.1)
        ax.legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left',
                ncols=2, mode="expand", borderaxespad=0., fontsize=9)

        ax = axes['d']
        ax.hist(H_0, bins='auto', facecolor='None', edgecolor=noise_color, zorder=0, label='noise heights')
        ax.hist(H_opt, bins=binedges, facecolor='None', edgecolor=pulse_color, zorder=1, label='pulse heights')
        ax.plot(pdf_x, pdf_y, c=pulse_color, label='R=%.1f, $R_{sn}$=%.1f' % (R_opt, R_sn))
        ax.set_ylabel('counts')
        ax.set_xlabel(str(response))
        ax.legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left',
                ncols=1, mode="expand", borderaxespad=0., fontsize=9)
        plt.show()


if __name__ == "__main__":
    # dir = "./README Example Data/LT218Chip1_BF_20240116_MIR18_5/12KIDS_185um_BB200K/TD_Power"
    # kid = 5
    # pread = 113
    # file_type = 'vis'
    # chuncksize = 10
    # nr_chuncks = 1
    # pw = 1500
    # pw_offset = 100
    # filter = 'exp'
    # lifetime = 300
    # mph = np.array([5, 20])
    # mpp = 1
    # iterate = 1

    # dir = "./README Example Data/LT192Chip1_BF_20220235/1KIDs laser on 1545 50nW 46dB/TD_Power"
    # kid = 1
    # pread = 102
    # file_type = 'vis'
    # chuncksize = 10
    # nr_chuncks = None
    # pw = 200
    # pw_offset = 20
    # filter = 'None'
    # lifetime = 300
    # mph = np.array([5, 15])
    # mpp = 5
    # iterate = 0

    dir = "Z:/KIDonSun/experiments/Entropy ADR/LT361W1chip6_BF_20240917/10KIDs_402nm_1090nW_24dB/TD_Power"
    kid = 4
    pread = 106
    file_type = 'vis'
    chuncksize = 10
    nr_chuncks = 1
    pw = 200
    pw_offset = 20
    filter = 'None'
    lifetime = 300
    mph = np.array([10, 50])
    mpp = 5
    iterate = 0

    # dir = "Z:/KIDonSun/experiments/Entropy ADR/LT361W1chip6_BF_20240917/10KIDs_402nm_1090nW_24dB/TD_Power"
    # kid = 381
    # pread = 106
    # file_type = 'vis'
    # chuncksize = 10
    # nr_chuncks = 4
    # pw = 200
    # pw_offset = 20
    # filter = 'None'
    # lifetime = 300
    # mph = np.array([5, 50])
    # mpp = 5
    # iterate = 0

    # dir = "Z:/KIDonSun/experiments/Entropy ADR/LT361W1chip6_BF_20240917/10KIDs_986nm_100nW_33dB/TD_Power"
    # kid = 4
    # pread = 106
    # file_type = 'vis'
    # chuncksize = 10
    # nr_chuncks = 4
    # pw = 200
    # pw_offset = 20
    # filter = 'None'
    # lifetime = 300
    # mph = np.array([5, 50])
    # mpp = 5
    # iterate = 0

    # dir = "Z:/KIDonSun/experiments/Entropy ADR/LT361W1chip6_BF_20240917/10KIDs_986nm_100nW_33dB/TD_Power"
    # kid = 381
    # pread = 106
    # file_type = 'vis'
    # chuncksize = 10
    # nr_chuncks = 4
    # pw = 200
    # pw_offset = 20
    # filter = 'None'
    # lifetime = 300
    # mph = np.array([5, 50])
    # mpp = 5
    # iterate = 0

    # dir = "Z:/KIDonSun/experiments/Entropy ADR/LT343chip5_BF_20240530_mapping/6KIDs_2Pread_500nm_40W/TD_Power"
    # kid = 2
    # pread = 104
    # file_type = 'vis'
    # chuncksize = 10
    # nr_chuncks = 4
    # pw = 200
    # pw_offset = 20
    # filter = 'None'
    # lifetime = 300
    # mph = np.array([10, 50])
    # mpp = 5
    # iterate = 0

    # dir = "Z:/KIDonSun/experiments/Entropy ADR/LT342_Chip3_bTa_hybrid_SiN/5KIDs_673nm_160nW_34dB/TD_Power"
    # kid = 267
    # pread = 99
    # file_type = 'vis'
    # chuncksize = 10
    # nr_chuncks = 4
    # pw = 200
    # pw_offset = 20
    # filter = 'None'
    # lifetime = 300
    # mph = np.array([5, 20])
    # mpp = 5
    # iterate = 0

    # dir = "Z:/KIDonSun/experiments/Entropy ADR/LT342_chip3_QEmeasurement/500nm_13_05_2024/TD_Power"
    # kid = 9
    # pread = 102
    # file_type = 'vis'
    # chuncksize = 10
    # nr_chuncks = 4
    # pw = 200
    # pw_offset = 20
    # filter = 'None'
    # lifetime = 300
    # mph = np.array([7, 20])
    # mpp = 5
    # iterate = 0

    pulse_analysis(dir, kid, pread, file_type, chuncksize, nr_chuncks, pw, pw_offset, filter, lifetime, mph, mpp, iterate=iterate, plot=True, coord='circle')

