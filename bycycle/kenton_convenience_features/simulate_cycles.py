from neurodsp.sim import sim_bursty_oscillation, sim_powerlaw

FS = 500
def create_signals(self):
        n_seconds = 10
        # bursts and signals taken from tutorial pages.
        burst0 = sim_bursty_oscillation(n_seconds=n_seconds, fs=FS, freq=10, burst_def='durations', burst_params={
            'n_cycles_burst': 3, 'n_cycles_off': 3})
        burst1 = sig = sim_bursty_oscillation(n_seconds=n_seconds, fs=FS, freq=10, burst_params={
            'enter_burst': 0.2, 'leave_burst': 0.8})
        burst2 = sim_bursty_oscillation(n_seconds=n_seconds, fs=FS, freq=10, burst_def='durations',
                                        burst_params={'n_cycles_burst': 3, 'n_cycles_off': 5})
        burst3 = sim_bursty_oscillation(n_seconds=n_seconds, fs=FS, freq=10, burst_def='durations', burst_params={
            'n_cycles_burst': 8, 'n_cycles_off': 20})
        burst4 = sig = sim_bursty_oscillation(n_seconds=n_seconds, fs=FS, freq=10, burst_params={
            'enter_burst': 0.2, 'leave_burst': 0.8})
        burst5 = sim_bursty_oscillation(n_seconds=n_seconds, fs=FS, freq=10, burst_def='durations',
                                        burst_params={'n_cycles_burst': 3, 'n_cycles_off': 3})

        sig0 = sim_powerlaw(n_seconds=n_seconds, fs=FS, exponent=-2.0)

        sig1 = sim_powerlaw(n_seconds=n_seconds, fs=FS,
                            exponent=-1.5, f_range=(2, None))
        sig2 = sim_powerlaw(n_seconds=n_seconds, fs=FS, exponent=-0.5)

        sig3 = sim_powerlaw(n_seconds=n_seconds, fs=FS,
                            exponent=-3, f_range=(2, None))

        bursts = [burst1, burst2, burst3, burst4, burst5]
        # bursts = [burst0]
        sigs = [sig0, sig1, sig2, sig3]
        # sigs = [sig0]
        # ratios = [10, 1, 0.5, 0.1, 0.0]
        ratios = [10, 1, 0.5]
        # ratios=[20,10]
        combined_sigs = [None]*(len(bursts) * len(sigs)*len(ratios))

        # for y_vals in all_to_plot:
        #     plot_time_series(times=times, sigs=y_vals)
        # tested, looks good.

        for i in range(len(bursts)):
            for j in range(len(sigs)):
                for k in range(len(ratios)):
                    combined_sigs[len(sigs)*len(ratios)*i + len(ratios)
                                  * j+k] = bursts[i]+10*ratios[len(ratios)-k-1]*sigs[j]

        return combined_sigs