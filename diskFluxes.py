import sys
import time
from pathlib import Path
from multiprocessing import Pool
import functools
import argparse
import numpy as np
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['legend.labelspacing'] = 0.2
import matplotlib.pyplot as plt
import h5py as h5
import discopy as dp

def getPlanetPot(r, phi, z, planets):

    PhiG = np.zeros(r.shape)

    for ip in range(planets.shape[0]):
        ptype = int(planets[ip, 6])
        GM = planets[ip, 0]
        rp = planets[ip, 3]
        phip = planets[ip, 4]
        eps = planets[ip, 5]

        dx = r*np.cos(phi) - rp*np.cos(phip)
        dy = r*np.sin(phi) - rp*np.sin(phip)
        dz = z
        R = np.sqrt(dx*dx + dy*dy + dz*dz)

        if ptype == 0:
            PhiG += -GM/np.sqrt(R*R + eps*eps)
        elif ptype == 1:
            PhiG += -GM / (R - 2*GM)
        elif ptype == 2:
            PhiG += -GM * R
        elif ptype == 3:
            epsS = eps * 2.8
            u = R / epsS
            A = u < 0.5
            B = (u >= 0.5) & (u < 1.0)
            C = u >= 1.0
            PhiG[A] += GM * (16./3.*u**2 - 48./5.*u**4 + 32./5.*u**5
                             - 14./5.)[A] / eps
            PhiG[B] += GM * (1./(15.*u) + 32./3.*u**2 - 16.*u**3 + 48./5.*u**4
                             - 32./15.*u**5 - 3.2)[B] / eps
            PhiG[C] += -GM/R[C]
        else:
            print("Unknown Planet Type")

    return PhiG


def getPlanetGacc(r, phi, z, planets):

    gr = np.zeros(r.shape)
    gp = np.zeros(r.shape)
    gz = np.zeros(r.shape)

    for ip in range(planets.shape[0]):
        ptype = int(planets[ip, 6])
        GM = planets[ip, 0]
        rp = planets[ip, 3]
        phip = planets[ip, 4]
        eps = planets[ip, 5]

        cosphi = np.cos(phi)
        sinphi = np.sin(phi)

        dx = r*cosphi - rp*np.cos(phip)
        dy = r*sinphi - rp*np.sin(phip)
        dz = z
        Rc = np.sqrt(dx*dx + dy*dy)
        R = np.sqrt(dx*dx + dy*dy + dz*dz)

        if ptype == 0:
            g = -GM*R/np.power(R*R + eps*eps, 1.5)
        elif ptype == 1:
            g = -GM / (R - 2*GM)**2
        elif ptype == 2:
            g = -GM * np.ones(r.shape)
        elif ptype == 3:
            epsS = eps * 2.8
            u = R / epsS
            A = u < 0.5
            B = (u >= 0.5) & (u < 1.0)
            C = u >= 1.0
            g = np.empty(r.shape)

            g[A] = -GM * (32./3.*u - 192./5.*u**3 + 32.*u**4)[A] / eps**2
            g[B] = -GM * (-1./(15.*u**2) + 64./3.*u - 48.*u**2 + 192./5.*u**3
                             - 32./3.*u**4)[B] / eps**2
            g[C] = -GM/R[C]**2
        else:
            g = np.zeros(r.shape)

        gx = g * dx/R
        gy = g * dy/R
        gz += g * dz/R

        gr += gx*cosphi + gy*sinphi
        gp += Rc * (-gx*sinphi + gy*cosphi)

    return gr, gp, gz


def integrate_over_r(f, dat=None, rjph=None, axis=None):
  
    if rjph is None:
        rjph = dat[0]
    dr = rjph[1:] - rjph[:-1]
    I = np.cumsum(f * dr, axis=axis)

    return I


def plotCombo(ax, x, ypack, mask, kwargs, alpha):

    if len(ypack.shape) == 1:
        ax.plot(x[mask], ypack[mask], **kwargs)

    else:
        y = ypack[0]
        for yerr in ypack[1:]:
            ax.fill_between(x[mask], (y-yerr)[mask], (y+yerr)[mask],
                            alpha=alpha, lw=0, color=kwargs['color'])
        ax.plot(x[mask], y[mask], **kwargs)


def makeFramePlots(t, R, Sig, Pi, Vr, Om, Mdot, Jdot, FJ_adv, FJ_rey,
                   FJ_visc, TG, Qheat, nu, Rmin, Rmax, Mdot0, name,
                   plotDir, ext, alphaErr=0.0):

    toPlot = (R >= Rmin) & (R <= Rmax)

    Jdot0 = 0.0
    Jdot1 = 1.0 * Mdot0
    Jdot2 = -1.0 * Mdot0

    vr0 = -1.5 * nu / R
    Sig0 = Mdot0 / (-2*np.pi*R*vr0)
    Om0 = np.power(R, -1.5)

    f1 = 1.0 - Jdot1 / (Mdot0 * R*R*Om0)
    f2 = 1.0 - Jdot2 / (Mdot0 * R*R*Om0)
    Sig1 = Sig0 * f1
    Sig2 = Sig0 * f2
    vr1 = vr0 / f1
    vr2 = vr0 / f2

    Qheat0 = 9./4. * Sig0*nu * Om0**2
    Qheat1 = 9./4. * Sig1*nu * Om0**2
    Qheat2 = 9./4. * Sig2*nu * Om0**2

    FJadv0 = -Mdot0 * np.power(R, 0.5)
    FJvisc0 = 3*np.pi * Sig0*nu * np.power(R, 0.5)
    FJvisc1 = 3*np.pi * Sig1*nu * np.power(R, 0.5)
    FJvisc2 = 3*np.pi * Sig2*nu * np.power(R, 0.5)

    kwargsSim = {'ls': '-', 'lw': 1, 'color': 'tab:blue',
                 'label': r'\tt{Disco}'}
    kwargs0 = {'ls': '-.', 'lw': 2, 'color': 'grey',
               'label': (r'$\dot{M} = \dot{M}_0$,'
                       + r'  $\dot{J} = 0$')}
    kwargs1 = {'ls': '--', 'lw': 2, 'color': 'grey',
               'label': (r'$\dot{M} = \dot{M}_0$,'
                       + r'  $\dot{J} = +\dot{M}_0 \ell_{\mathrm{bin}}$')}
    kwargs2 = {'ls': ':', 'lw': 2, 'color': 'grey',
               'label': (r'$\dot{M} = \dot{M}_0$,'
                       + r'  $\dot{J} = -\dot{M}_0 \ell_{\mathrm{bin}}$')}

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    
    ax[0, 0].plot(R[toPlot], Sig0[toPlot], **kwargs0)
    ax[0, 0].plot(R[toPlot], Sig1[toPlot], **kwargs1)
    ax[0, 0].plot(R[toPlot], Sig2[toPlot], **kwargs2)
    plotCombo(ax[0, 0], R, Sig, toPlot, kwargsSim, alphaErr)
    
    plotCombo(ax[0, 1], R, Pi, toPlot, kwargsSim, alphaErr)
    
    ax[1, 0].plot(R[toPlot], -vr0[toPlot], **kwargs0)
    ax[1, 0].plot(R[toPlot], -vr1[toPlot], **kwargs1)
    ax[1, 0].plot(R[toPlot], -vr2[toPlot], **kwargs2)
    plotCombo(ax[1, 0], R, -Vr, toPlot, kwargsSim, alphaErr)
    
    ax[1, 1].plot(R[toPlot], Om0[toPlot], **kwargs0)
    plotCombo(ax[1, 1], R, Om, toPlot, kwargsSim, alphaErr)

    ax[0, 0].set(xlabel=r'$R$ (a)', ylabel=r'$\Sigma$')
    ax[0, 1].set(xlabel=r'$R$ (a)', ylabel=r'$\Pi$')
    ax[1, 0].set(xlabel=r'$R$ (a)', ylabel=r'$-v_r$',
                 xscale='log', yscale='log')
    ax[1, 1].set(xlabel=r'$R$ (a)', ylabel=r'$\Omega$')

    ax[0, 0].legend()

    figname = plotDir / "primAve_{0:s}.{1:s}".format(name, ext)
    print("Saving", figname)
    fig.savefig(figname)
    plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    
    ax[0, 0].axhline(Mdot0, **kwargs0)
    plotCombo(ax[0, 0], R, Mdot, toPlot, kwargsSim, alphaErr)
    
    ax[0, 1].axhline(Jdot0, **kwargs0)
    ax[0, 1].axhline(Jdot1, **kwargs1)
    ax[0, 1].axhline(Jdot2, **kwargs2)
    plotCombo(ax[0, 1], R, Jdot, toPlot, kwargsSim, alphaErr)
    
    ax[1, 0].plot(R[toPlot], Qheat0[toPlot], **kwargs0)
    ax[1, 0].plot(R[toPlot], Qheat1[toPlot], **kwargs1)
    ax[1, 0].plot(R[toPlot], Qheat2[toPlot], **kwargs2)
    plotCombo(ax[1, 0], R, Qheat, toPlot, kwargsSim, alphaErr)

    ax[0, 0].set(xlabel=r'$R$ (a)', ylabel=r'$\dot{M}$')
    ax[0, 1].set(xlabel=r'$R$ (a)', ylabel=r'$\dot{J}$')
    ax[1, 0].set(xlabel=r'$R$ (a)', ylabel=r'$Q_{\mathrm{visc}}$',
                 yscale='log')

    ax[1, 0].legend()

    figname = plotDir / "diskDiag_{0:s}.{1:s}".format(name, ext)
    print("Saving", figname)
    fig.savefig(figname)
    plt.close(fig)

    kwargsTot = {'ls': '-', 'lw': 2, 'color': 'k',
                 'label': r'$\dot{J}_{\mathrm{tot}}$'}
    kwargsAdv = {'ls': '-', 'lw': 1, 'color': 'C0',
                 'label': r'$\dot{J}_{\mathrm{adv}}$'}
    kwargsVisc = {'ls': '-', 'lw': 1, 'color': 'C1',
                 'label': r'$\dot{J}_{\mathrm{visc}}$'}
    kwargsGrav = {'ls': '-', 'lw': 1, 'color': 'C2',
                 'label': r'$\dot{J}_{\mathrm{grav}}$'}
    kwargsRey = {'ls': '-', 'lw': 1, 'color': 'C3',
                 'label': r'$\dot{J}_{\mathrm{Rey}}$'}

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax[0, 0].axhline(Jdot0, **kwargs0)
    ax[0, 0].axhline(Jdot1, **kwargs1)
    ax[0, 0].axhline(Jdot2, **kwargs2)
    plotCombo(ax[0, 0], R, -FJ_adv, toPlot, kwargsAdv, alphaErr)
    plotCombo(ax[0, 0], R, -FJ_visc, toPlot, kwargsVisc, alphaErr)
    plotCombo(ax[0, 0], R, TG, toPlot, kwargsGrav, alphaErr)
    plotCombo(ax[0, 0], R, Jdot, toPlot, kwargsTot, alphaErr)
    ax[0, 1].plot(R[toPlot], -FJadv0[toPlot], **kwargs0)
    plotCombo(ax[0, 1], R, -FJ_adv, toPlot, kwargsAdv, alphaErr)
    ax[1, 0].plot(R[toPlot], -FJvisc0[toPlot], **kwargs0)
    ax[1, 0].plot(R[toPlot], -FJvisc1[toPlot], **kwargs1)
    ax[1, 0].plot(R[toPlot], -FJvisc2[toPlot], **kwargs2)
    plotCombo(ax[1, 0], R, -FJ_visc, toPlot, kwargsVisc, alphaErr)
    plotCombo(ax[1, 1], R, -FJ_rey, toPlot, kwargsRey, alphaErr)
    plotCombo(ax[1, 2], R, TG, toPlot, kwargsGrav, alphaErr)

    ax[0, 0].set(xlabel=r'$R$ (a)', ylabel=r'$\dot{J}$')
    ax[0, 1].set(xlabel=r'$R$ (a)', ylabel=r'$\dot{J}_{\mathrm{adv}}$')
    ax[1, 0].set(xlabel=r'$R$ (a)', ylabel=r'$\dot{J}_{\mathrm{visc}}$')
    ax[1, 1].set(xlabel=r'$R$ (a)', ylabel=r'$\dot{J}_{\mathrm{Rey}}$')
    ax[1, 2].set(xlabel=r'$R$ (a)', ylabel=r'$\dot{J}_{\mathrm{grav}}$')

    ax[0, 0].legend()
    ax[1, 1].legend()

    figname = plotDir / "angMomFlux_{0:s}.{1:s}".format(name, ext)
    print("Saving", figname)
    fig.savefig(figname)
    plt.close(fig)


def analyzeSingle(filepack, archiveName="diskFluxArchive.h5", Rmin=0.0,
                  Rmax=np.inf, makeFrames=False):

    idx, filename = filepack

    print("Loading", filename)
    opts = dp.util.loadOpts(filename)
    pars = dp.util.loadPars(filename)
    plain_names, tex_names, num_c, num_n = dp.util.getVarNames(filename)
    t, r, phi, z, prim, dat = dp.util.loadCheckpoint(filename)
    
    name = (filename.stem).split("_")[-1]
    plotDir = Path("plots")
    if not plotDir.exists():
        plotDir.mkdir()

    nu = pars['Viscosity']
    planetDat = dat[4]

    rho = prim[:, 0]  # r * np.cos(phi)
    P = prim[:, 1]
    vr = prim[:, 2]
    vp = prim[:, 3]
    vz = prim[:, 4]
    q0 = prim[:, 5]

    print("Computing gradients...")
    dvdr, dvdp, _ = dp.geom.calculateGrad(r, phi, z, prim[:, 2:4],
                                            dat, opts, pars)
    print("    Done.")

    dr_vr = dvdr[:, 0]
    dp_vr = dvdp[:, 0]
    dr_vp = dvdr[:, 1]
    dp_vp = dvdp[:, 1]

    print("Computing gravity...")
    PhiG = getPlanetPot(r, phi, z, planetDat)
    gr, gp, gz = getPlanetGacc(r, phi, z, planetDat)
    print("    Done.")

    print("Computing fluxes...")
    R = np.unique(r)
    Sig = dp.geom.integrate2(rho, dat, opts, pars) / (2*np.pi)
    Sig2 = dp.geom.integrate2(rho**2, dat, opts, pars) / (2*np.pi)
    Pi = dp.geom.integrate2(P, dat, opts, pars) / (2*np.pi)
    Vr_ave = dp.geom.integrate2(vr, dat, opts, pars) / (2*np.pi)
    Om_ave = dp.geom.integrate2(vp, dat, opts, pars) / (2*np.pi)

    dMdr = 2*np.pi * R * Sig
    dSdr = dp.geom.integrate2(r * rho*vr, dat, opts, pars)
    dJdr = dp.geom.integrate2(r * rho*r*r*vp, dat, opts, pars)
    Vr = dSdr / dMdr
    l = dJdr / dMdr
    Om = l / (R*R)


    Mdot = -dSdr
    
    fJ_adv = r * rho*vr*r*r*vp

    # shear s_ab == s_a^b
    divV = dr_vr + dp_vp + vr / r
    s_rr = dr_vr - divV/3.0
    s_pp = dp_vp + vr/r - divV/3.0
    s_pr = 0.5 * (dp_vr + r*r*dr_vp)
    s_rp = s_pr / (r*r)
    s2 = s_rr*s_rr + 2 * s_rp*s_pr + s_pp*s_pp
    
    fJ_visc = r * -2 * rho*nu * s_pr

    visc_diss = 2*rho*nu * s2
    
    tG = rho*gp

    FJ_adv = dp.geom.integrate2(fJ_adv, dat, opts, pars)
    FJ_visc = dp.geom.integrate2(fJ_visc, dat, opts, pars)
    dTG_dr = dp.geom.integrate2(r * tG, dat, opts, pars)
    Qheat_Ave = dp.geom.integrate2(visc_diss, dat, opts, pars) / (2*np.pi)
    Qheat_Ave2 = dp.geom.integrate2(visc_diss**2, dat, opts, pars) / (2*np.pi)
    Qheat_asig = np.sqrt(Qheat_Ave2 - Qheat_Ave**2)

    TG = integrate_over_r(dTG_dr, dat)
    TG = integrate_over_r(dTG_dr, dat)

    FJ_acc = -Mdot*l
    FJ_rey = FJ_adv - FJ_acc

    Jdot = TG - FJ_visc - FJ_adv
    print("    Done.")

    print("Storing fluxes in", archiveName, "...")

    maxTries = 100
    tries = 0
    f = None

    while tries < maxTries:
        try:
            f = h5.File(archiveName, "r+")
            break
        except OSError:
            tries += 1
            # print("File busy, attempt", tries, "of", maxTries)
            time.sleep(0.05)
    if f is None:
        f = h5.File(archiveName, "r+")

    f['t'][idx] = t
    f['Sig'][idx, :] = Sig
    f['Pi'][idx, :] = Pi
    f['Vr'][idx, :] = Vr
    f['Om'][idx, :] = Om
    f['Mdot'][idx, :] = Mdot
    f['FJ_adv'][idx, :] = FJ_adv
    f['FJ_visc'][idx, :] = FJ_visc
    f['T_grav'][idx, :] = TG
    f['Qheat_visc'][idx, :] = Qheat_Ave
    f['planets'][idx, :, :] = planetDat
    f.close()

    print("    Done.")
    
    Mdot0 = 1.0  # 3*np.pi*nu

    if makeFrames:
        makeFramePlots(t, R, Sig, Pi, Vr, Om, Mdot, Jdot, FJ_adv, FJ_rey,
                       FJ_visc, TG, Qheat_Ave, nu, Rmin, Rmax, Mdot0, name,
                       plotDir, "png")

    
def getPackStats(f, axis=0):

    f_mean = f.mean(axis=axis)
    f2_mean = (f**2).mean(axis=axis)
    df2 = f2_mean - f_mean**2
    df2[df2<0.0] = 0.0
    f_sig = np.sqrt(df2)

    f_mean_err = f_sig / np.sqrt(f.shape[axis])

    return np.array((f_mean, f_sig, f_mean_err))


def analyzeSingleArchive(archiveName, Rmin=0.0, Rmax=np.inf,
                         makeFrames=False):

    with h5.File(archiveName, "r") as f:
        t = f['t'][...]
        R = f['R'][...]
        rjph = f['rjph'][...]
        Sig = f['Sig'][...]
        Pi = f['Pi'][...]
        Vr = f['Vr'][...]
        Om = f['Om'][...]
        Mdot = f['Mdot'][...]
        FJ_adv = f['FJ_adv'][...]
        FJ_visc = f['FJ_visc'][...]
        T_grav = f['T_grav'][...]
        Qheat_visc = f['Qheat_visc'][...]
        planets = f['planets'][...]
        nu = f['Pars/Viscosity'][()]

    nt = len(t)
    tP = t / (2*np.pi)

    Mdot0 = 1.0

    Jdot = T_grav - FJ_adv - FJ_visc
    l = R[None, :]**2 * Om
    FJ_acc = -Mdot*l
    FJ_rey = FJ_adv - FJ_acc
    
    plotDir = Path("plots")
    if not plotDir.exists():
        plotDir.mkdir()

    if makeFrames:
        for i in range(nt):
            name = "{0:04d}".format(i)
            makeFramePlots(t[i], R, Sig[i], Pi[i], Vr[i], Om[i], Mdot[i],
                           Jdot[i], FJ_adv[i], FJ_rey[i], FJ_visc[i],
                           T_grav[i], Qheat_visc[i],
                           nu, Rmin, Rmax, Mdot0, name, plotDir)

    SigPack = getPackStats(Sig)
    PiPack = getPackStats(Pi)
    VrPack = getPackStats(Vr)
    OmPack = getPackStats(Om)
    MdotPack = getPackStats(Mdot)
    JdotPack = getPackStats(Jdot)
    FJ_advPack = getPackStats(FJ_adv)
    FJ_reyPack = getPackStats(FJ_rey)
    FJ_viscPack = getPackStats(FJ_visc)
    T_gravPack = getPackStats(T_grav)
    Qheat_viscPack = getPackStats(Qheat_visc)

    makeFramePlots(t[-1]-t[0], R, SigPack, PiPack, VrPack, OmPack, MdotPack,
                   JdotPack, FJ_advPack, FJ_reyPack, FJ_viscPack, T_gravPack,
                   Qheat_viscPack, nu, Rmin, Rmax, Mdot0, "tAvg", plotDir,
                   "pdf", 0.4)

    dMdr = 2*np.pi * R[None, :] * Sig
    Mr = integrate_over_r(dMdr, rjph=rjph, axis=1)
    Mtot = Mr[:, -1]

    dJdr = 2*np.pi * R[None, :]**3 * Sig * Om
    Jr = integrate_over_r(dJdr, rjph=rjph, axis=1)
    Jtot = Jr[:, -1]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sortInds = np.argsort(tP)
    ax[0].plot(tP[sortInds], Mtot[sortInds])
    ax[1].plot(tP[sortInds], Jtot[sortInds])

    ax[0].set(xlabel=r'$t$ ($T_{\mathrm{bin}}^{-1}$)',
              ylabel=r'$M_{\mathrm{tot}}$')
    ax[1].set(xlabel=r'$t$ ($T_{\mathrm{bin}}^{-1}$)',
              ylabel=r'$J_{\mathrm{tot}}$')

    figname = plotDir / "cons_timeSeries.pdf"
    print("Saving", figname)
    fig.savefig(figname)
    plt.close(fig)


def analyzeCheckpoints(filenames, archiveName='diskFluxArchive.h5',
                       ncpu=None, Rmin=0.0, Rmax=np.inf, makeFrames=False):

    nt = len(filenames)
    pars = dp.util.loadPars(filenames[0])
    t, r, phi, z, prim, dat = dp.util.loadCheckpoint(filenames[0])
    rjph = dat[0]
    R = np.unique(r)

    planetDat = dat[4]

    nr = len(rjph)-1

    print("Initializing archive HDF5 file:", archiveName)

    with h5.File(archiveName, "w") as f:
        f.create_dataset("t", (nt, ), dtype=float)
        f.create_dataset("rjph", data=rjph)
        f.create_dataset("R", data=R)
        f.create_dataset("Sig", (nt, nr), dtype=float)
        f.create_dataset("Pi", (nt, nr), dtype=float)
        f.create_dataset("Vr", (nt, nr), dtype=float)
        f.create_dataset("Om", (nt, nr), dtype=float)
        f.create_dataset("Mdot", (nt, nr), dtype=float)
        f.create_dataset("FJ_adv", (nt, nr), dtype=float)
        f.create_dataset("FJ_visc", (nt, nr), dtype=float)
        f.create_dataset("T_grav", (nt, nr), dtype=float)
        f.create_dataset("Qheat_visc", (nt, nr), dtype=float)
        f.create_dataset("planets", (nt,)+planetDat.shape, dtype=float)

        f.create_group("Pars")
        for key in pars:
            f['Pars'].create_dataset(key, data=pars[key])

    if ncpu is None or ncpu <= 1:
        for i, fname in enumerate(filenames):
            analyzeSingle((i, fname), archiveName, Rmin, Rmax, makeFrames)
    else:
        map_fn = functools.partial(analyzeSingle, archiveName=archiveName,
                                   Rmin=Rmin, Rmax=Rmax, makeFrames=makeFrames)
        with Pool(ncpu) as p:
            p.map(map_fn, enumerate(filenames))

    return archiveName


def analyzeArchives(filenames, Rmin, Rmax, makeFrames):

    for filename in filenames:
        analyzeSingleArchive(filename, Rmin, Rmax, makeFrames)


def analyze(filenames, archiveName='diskFluxArchive.h5', ncpu=1,
            Rmin=0.0, Rmax=np.inf, makeFrames=False):

    checkpoints = False
    archives = False

    with h5.File(filenames[0], "r") as f:
        if 'GIT_VERSION' in f and 'Grid' in f and 'Data' in f:
            checkpoints = True
        elif 'Mdot' in f and 'Pars' in f and 'rjph' in f:
            archives = True

    if checkpoints:
        analyzeCheckpoints(filenames, archiveName, ncpu, Rmin, Rmax,
                           makeFrames)
        analyzeSingleArchive(archiveName, Rmin, Rmax, False)

    elif archives:
        analyzeArchives(filenames, Rmin, Rmax, makeFrames)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analyze accretion disks")
    parser.add_argument('files', nargs='+')
    parser.add_argument('--ncpu', nargs='?', default=1, type=int)
    parser.add_argument('--Rmin', nargs='?', default=0.0, type=np.float)
    parser.add_argument('--Rmax', nargs='?', default=np.inf, type=np.float)
    parser.add_argument('--makeFrames', action='store_true')
    parser.add_argument('--archive', nargs=1, default='diskFluxArchive.h5')

    args = parser.parse_args()

    filenames = [Path(x) for x in args.files]
    analyze(filenames, args.archive, args.ncpu, args.Rmin, args.Rmax,
            args.makeFrames)
