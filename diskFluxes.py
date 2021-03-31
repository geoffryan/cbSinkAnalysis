import sys
import time
import math
import warnings
from pathlib import Path
from multiprocessing import Pool
import functools
import argparse
import numpy as np
import scipy.signal
import scipy.optimize
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['legend.labelspacing'] = 0.2
import matplotlib.pyplot as plt
import h5py as h5
import discopy as dp

warnings.simplefilter('ignore', RuntimeWarning)

def f_bpl(x, al1, al2, x0, f0, s, fa):

    f1 = np.power(x/x0, -al1*s)
    f2 = np.power(x/x0, -al2*s)
    return f0 * np.power(f1 + f2, -1.0/s) + fa

def f_M20(x, l0, f0, rcav, z):

    return f0 * (1-l0*np.power(x, -0.5)) * np.exp(-np.power(rcav/x, z))

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


def makeFramePlots(t, R, Sig, Pi, Vr, Om, Mdot, Jdot, ell, FJ_adv, FJ_rey,
                   FJ_visc, TG, Qheat, nu, Mdot0, name, plotDir, args,
                   ext, alphaErr=0.0):

    toPlot = (R >= args.Rmin) & (R <= args.Rmax)

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
    
    ax[1, 0].axhline(Jdot0 / Mdot0, **kwargs0)
    ax[1, 0].axhline(Jdot1 / Mdot0, **kwargs1)
    ax[1, 0].axhline(Jdot2 / Mdot0, **kwargs2)
    plotCombo(ax[1, 0], R, ell, toPlot, kwargsSim, alphaErr)
    
    ax[1, 1].plot(R[toPlot], Qheat0[toPlot], **kwargs0)
    ax[1, 1].plot(R[toPlot], Qheat1[toPlot], **kwargs1)
    ax[1, 1].plot(R[toPlot], Qheat2[toPlot], **kwargs2)
    plotCombo(ax[1, 1], R, Qheat, toPlot, kwargsSim, alphaErr)

    ax[0, 0].set(xlabel=r'$R$ (a)', ylabel=r'$\dot{M}$',
                 ylim=(-4, 4))
    ax[0, 1].set(xlabel=r'$R$ (a)', ylabel=r'$\dot{J}$',
                 ylim=(-4, 4))
    ax[1, 0].set(xlabel=r'$R$ (a)', ylabel=r'$\ell = \dot{J} / \dot{M}$',
                 ylim=(-4, 4))
    ax[1, 1].set(xlabel=r'$R$ (a)', ylabel=r'$Q_{\mathrm{visc}}$',
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

    ax[0, 0].set(xlabel=r'$R$ (a)', ylabel=r'$\dot{J}$',
                 ylim=(-10, 10))
    ax[0, 1].set(xlabel=r'$R$ (a)', ylabel=r'$\dot{J}_{\mathrm{adv}}$',
                 ylim=(-10, 10))
    ax[1, 0].set(xlabel=r'$R$ (a)', ylabel=r'$\dot{J}_{\mathrm{visc}}$')
    ax[1, 1].set(xlabel=r'$R$ (a)', ylabel=r'$\dot{J}_{\mathrm{Rey}}$',
                 ylim=(-3, 3))
    ax[1, 2].set(xlabel=r'$R$ (a)', ylabel=r'$\dot{J}_{\mathrm{grav}}$',
                 ylim=(-1, 1))

    ax[0, 0].legend()
    ax[1, 1].legend()

    figname = plotDir / "angMomFlux_{0:s}.{1:s}".format(name, ext)
    print("Saving", figname)
    fig.savefig(figname)
    plt.close(fig)


def analyzeSingle(filepack, args):

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
    rjph = dat[0]

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

    ell = Jdot / Mdot
    print("    Done.")

    print("Calculating orbital elements")

    GM = 1.0
    cosp = np.cos(phi)
    sinp = np.sin(phi)
    x = r * cosp
    y = r * sinp
    vx = vr * cosp - r*vp * sinp
    vy = vr * sinp + r*vp * cosp
    v2 = vr*vr + r*r*vp*vp

    slr = (r*r*vp)**2 / GM  # semi-latus rectum
    e = 0.5*v2 - GM/r  # specific energy, this should be negative
    
    ecc = np.sqrt(1 + 2 * slr * e / GM)  # eccentricity squared
    eccx = (v2*x - r*vr*vx) / GM - cosp  # eccentricity vector-x
    eccy = (v2*y - r*vr*vy) / GM - sinp  # eccentricity vector-y
    phip = np.arctan2(eccy, eccx)  # argument of periapse

    a = slr / (-2 * slr * e / GM)  # semi-major axis

    idxa = np.searchsorted(rjph, a) - 1
    idxp = np.searchsorted(rjph, slr) - 1

    nr = len(R)-1
    gooda = (idxa >= 0) & (idxa < nr)
    goodp = (idxp >= 0) & (idxp < nr)
    idxa = idxa[gooda]
    idxp = idxp[goodp]

    dV = dp.geom.getDV(dat, opts, pars)

    Rc = 0.5*(rjph[:-1] + rjph[1:])
    dr = rjph[1:] - rjph[:-1]

    dMe_r = dp.geom.integrate2(rho * e, dat, opts, pars) * Rc * dr
    dMecc_r = dp.geom.integrate2(rho * ecc, dat, opts, pars) * Rc * dr
    dMeccx_r = dp.geom.integrate2(rho * eccx, dat, opts, pars) * Rc * dr
    dMeccy_r = dp.geom.integrate2(rho * eccy, dat, opts, pars) * Rc * dr
    dMslr_r = dp.geom.integrate2(rho * slr, dat, opts, pars) * Rc * dr
    dMa_r = dp.geom.integrate2(rho * a, dat, opts, pars) * Rc * dr
    dMphip_r = dp.geom.integrate2(rho * phip, dat, opts, pars) * Rc * dr

    dV_a = np.zeros(R.shape)
    dM_a = np.zeros(R.shape)

    np.add.at(dV_a, idxa, dV[gooda])
    np.add.at(dM_a, idxa, (rho * dV)[gooda])
    
    dMe_a = np.zeros(R.shape)
    np.add.at(dMe_a, idxa, (rho*e * dV)[gooda])
    dMecc_a = np.zeros(R.shape)
    np.add.at(dMecc_a, idxa, (rho*ecc * dV)[gooda])
    dMeccx_a = np.zeros(R.shape)
    np.add.at(dMeccx_a, idxa, (rho*eccx * dV)[gooda])
    dMeccy_a = np.zeros(R.shape)
    np.add.at(dMeccy_a, idxa, (rho*eccy * dV)[gooda])
    dMslr_a = np.zeros(R.shape)
    np.add.at(dMslr_a, idxa, (rho*slr * dV)[gooda])
    dMa_a = np.zeros(R.shape)
    np.add.at(dMa_a, idxa, (rho*a * dV)[gooda])
    dMphip_a = np.zeros(R.shape)
    np.add.at(dMphip_a, idxa, (rho*phip * dV)[gooda])

    dV_p = np.zeros(R.shape)
    dM_p = np.zeros(R.shape)

    np.add.at(dV_p, idxp, dV[goodp])
    np.add.at(dM_p, idxp, (rho * dV)[goodp])
    
    dMe_p = np.zeros(R.shape)
    np.add.at(dMe_p, idxp, (rho*e * dV)[goodp])
    dMecc_p = np.zeros(R.shape)
    np.add.at(dMecc_p, idxp, (rho*ecc * dV)[goodp])
    dMeccx_p = np.zeros(R.shape)
    np.add.at(dMeccx_p, idxp, (rho*eccx * dV)[goodp])
    dMeccy_p = np.zeros(R.shape)
    np.add.at(dMeccy_p, idxp, (rho*eccy * dV)[goodp])
    dMslr_p = np.zeros(R.shape)
    np.add.at(dMslr_p, idxp, (rho*slr * dV)[goodp])
    dMa_p = np.zeros(R.shape)
    np.add.at(dMa_p, idxp, (rho*a * dV)[goodp])
    dMphip_p = np.zeros(R.shape)
    np.add.at(dMphip_p, idxp, (rho*phip * dV)[goodp])

    print("    Done.")

    print("Storing fluxes in", args.archive, "...")

    maxTries = 100
    tries = 0
    f = None

    while tries < maxTries:
        try:
            f = h5.File(args.archive, "r+")
            break
        except OSError:
            tries += 1
            # print("File busy, attempt", tries, "of", maxTries)
            time.sleep(0.05)
    if f is None:
        f = h5.File(args.archive, "r+")

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

    f['orb/dM_e'][idx, :] = dMe_r
    f['orb/dM_ecc'][idx, :] = dMecc_r
    f['orb/dM_eccx'][idx, :] = dMeccx_r
    f['orb/dM_eccy'][idx, :] = dMeccy_r
    f['orb/dM_p'][idx, :] = dMslr_r
    f['orb/dM_a'][idx, :] = dMa_r
    f['orb/dM_phip'][idx, :] = dMphip_r

    f['orb_a/dV'][idx, :] = dV_a
    f['orb_a/dM'][idx, :] = dM_a
    f['orb_a/dM_e'][idx, :] = dMe_a
    f['orb_a/dM_ecc'][idx, :] = dMecc_a
    f['orb_a/dM_eccx'][idx, :] = dMeccx_a
    f['orb_a/dM_eccy'][idx, :] = dMeccy_a
    f['orb_a/dM_p'][idx, :] = dMslr_a
    f['orb_a/dM_a'][idx, :] = dMa_a
    f['orb_a/dM_phip'][idx, :] = dMphip_a

    f['orb_p/dV'][idx, :] = dV_p
    f['orb_p/dM'][idx, :] = dM_p
    f['orb_p/dM_e'][idx, :] = dMe_p
    f['orb_p/dM_ecc'][idx, :] = dMecc_p
    f['orb_p/dM_eccx'][idx, :] = dMeccx_p
    f['orb_p/dM_eccy'][idx, :] = dMeccy_p
    f['orb_p/dM_p'][idx, :] = dMslr_p
    f['orb_p/dM_a'][idx, :] = dMa_p
    f['orb_p/dM_phip'][idx, :] = dMphip_p
    f.close()

    print("    Done.")
    
    Mdot0 = 1.0  # 3*np.pi*nu

    if args.makeFrames:
        makeFramePlots(t, R, Sig, Pi, Vr, Om, Mdot, Jdot, ell, FJ_adv, FJ_rey,
                       FJ_visc, TG, Qheat_Ave, nu, Mdot0, name,
                       plotDir, args, "png")
        makeOrbFrames(t, R, rjph, Sig, dMe_r, dMecc_r, dMeccx_r, dMeccy_r,
                      dMslr_r, dMa_r, dMphip_r, dV_a, dM_a, dMe_a, dMecc_a,
                      dMeccx_a, dMeccy_a, dMslr_a, dMa_a, dMphip_a, dV_p, dM_p,
                      dMe_p, dMecc_p, dMeccx_p, dMeccy_p, dMslr_p, dMa_p,
                      dMphip_p, name, plotDir, args, "png",
                      r, phi, z, dV, rho, e, ecc, eccx, eccy, slr, a, phip,
                      dat, opts, pars)

def makeOrbFrames(t, R, rjph, Sig,
                  dMe_r, dMecc_r, dMeccx_r, dMeccy_r, dMp_r, dMa_r, dMphip_r,
                  dV_a, dM_a, dMe_a, dMecc_a, dMeccx_a, dMeccy_a, dMp_a, dMa_a,
                  dMphip_a, dV_p, dM_p, dMe_p, dMecc_p, dMeccx_p, dMeccy_p,
                  dMp_p, dMa_p, dMphip_p, name, plotDir, args, ext,
                  r=None, phi=None, z=None, dV=None, rho=None, e=None,
                  ecc=None, eccx=None, eccy=None, p=None, a=None, phip=None,
                  dat=None, opts=None, pars=None):
    
    dr = rjph[1:]-rjph[:-1]
    Rc = 0.5*(rjph[1:] + rjph[:-1])
    dV_r = 2*np.pi*Rc * dr
    dM_r = Sig * dV_r

    n2 = 2*(len(R) // 2)
    n4 = 4*(len(R) // 4)

    ecc_r = dMecc_r / dM_r
    phip_r = dMphip_r / dM_r
    eccx_r = dMeccx_r / dM_r
    eccy_r = dMeccy_r / dM_r

    Sig_a = dM_a / dV_a
    ecc_a = dMecc_a / dM_a
    phip_a = dMphip_a / dM_a
    eccx_a = dMeccx_a / dM_a
    eccy_a = dMeccy_a / dM_a
    
    Sig_p = dM_p / dV_p
    ecc_p = dMecc_p / dM_p
    phip_p = dMphip_p / dM_p
    eccx_p = dMeccx_p / dM_p
    eccy_p = dMeccy_p / dM_p

    Sig2_r = (dM_r[:n2:2] + dM_r[1:n2:2]) / (dV_r[:n2:2] + dV_r[1:n2:2])
    Sig4_r = (dM_r[:n4:4] + dM_r[1:n4:4] + dM_r[2:n4:4] + dM_r[3:n4:4]
              ) / (dV_r[:n4:4] + dV_r[1:n4:4] + dV_r[2:n4:4] + dV_r[3:n4:4])

    Sig2_a = (dM_a[:n2:2] + dM_a[1:n2:2]) / (dV_a[:n2:2] + dV_a[1:n2:2])
    Sig4_a = (dM_a[:n4:4] + dM_a[1:n4:4] + dM_a[2:n4:4] + dM_a[3:n4:4]
              ) / (dV_a[:n4:4] + dV_a[1:n4:4] + dV_a[2:n4:4] + dV_a[3:n4:4])

    Sig2_p = (dM_p[:n2:2] + dM_p[1:n2:2]) / (dV_p[:n2:2] + dV_p[1:n2:2])
    Sig4_p = (dM_p[:n4:4] + dM_p[1:n4:4] + dM_p[2:n4:4] + dM_p[3:n4:4]
              ) / (dV_p[:n4:4] + dV_p[1:n4:4] + dV_p[2:n4:4] + dV_p[3:n4:4])

    R2 = 0.5*(R[:n2:2] + R[1:n2:2])
    R4 = 0.25*(R[:n4:4] + R[1:n4:4] + R[2:n4:4] + R[3:n4:4])

    goodR = (R >= args.Rmin) & (R <= args.Rmax)
    goodR2 = (R2 >= args.Rmin) & (R2 <= args.Rmax)
    goodR4 = (R4 >= args.Rmin) & (R4 <= args.Rmax)

    fig, ax = plt.subplots(3, 5, figsize=(16, 10))
    ax[0, 0].plot(R[goodR], Sig[goodR])
    ax[0, 0].plot(R2[goodR2], Sig2_r[goodR2])
    ax[0, 0].plot(R4[goodR4], Sig4_r[goodR4])
    ax[1, 0].plot(R[goodR], Sig_a[goodR])
    ax[1, 0].plot(R2[goodR2], Sig2_a[goodR2])
    ax[1, 0].plot(R4[goodR4], Sig4_a[goodR4])
    ax[2, 0].plot(R[goodR], Sig_p[goodR])
    ax[2, 0].plot(R2[goodR2], Sig2_p[goodR2])
    ax[2, 0].plot(R4[goodR4], Sig4_p[goodR4])
    ax[0, 1].plot(R[goodR], ecc_r[goodR])
    ax[0, 1].plot(R[goodR], np.sqrt(eccx_r**2 + eccy_r**2)[goodR])
    ax[1, 1].plot(R[goodR], ecc_a[goodR])
    ax[1, 1].plot(R[goodR], np.sqrt(eccx_a**2 + eccy_a**2)[goodR])
    ax[2, 1].plot(R[goodR], ecc_p[goodR])
    ax[2, 1].plot(R[goodR], np.sqrt(eccx_p**2 + eccy_p**2)[goodR])
    ax[0, 2].plot(R[goodR], phip_r[goodR])
    ax[0, 2].plot(R[goodR], np.arctan2(eccy_r, eccx_r)[goodR])
    ax[1, 2].plot(R[goodR], phip_a[goodR])
    ax[1, 2].plot(R[goodR], np.arctan2(eccy_a, eccx_a)[goodR])
    ax[2, 2].plot(R[goodR], phip_p[goodR])
    ax[2, 2].plot(R[goodR], np.arctan2(eccy_p, eccx_p)[goodR])
    ax[0, 3].plot(R[goodR], (ecc_r * np.cos(phip_r))[goodR])
    ax[0, 3].plot(R[goodR], eccx_r[goodR])
    ax[1, 3].plot(R[goodR], (ecc_a * np.cos(phip_a))[goodR])
    ax[1, 3].plot(R[goodR], eccx_a[goodR])
    ax[2, 3].plot(R[goodR], (ecc_p * np.cos(phip_p))[goodR])
    ax[2, 3].plot(R[goodR], eccx_p[goodR])
    ax[0, 4].plot(R[goodR], (ecc_r * np.sin(phip_r))[goodR])
    ax[0, 4].plot(R[goodR], eccy_r[goodR])
    ax[1, 4].plot(R[goodR], (ecc_a * np.sin(phip_a))[goodR])
    ax[1, 4].plot(R[goodR], eccy_a[goodR])
    ax[2, 4].plot(R[goodR], (ecc_p * np.sin(phip_p))[goodR])
    ax[2, 4].plot(R[goodR], eccy_p[goodR])


    figname = plotDir / "sig_grid_{0:s}.{1:s}".format(name, ext)
    print("Saving", figname)
    fig.savefig(figname, dpi=200)
    plt.close(fig)
   
    if r is not None:

        mapRmax = 5

        rs = np.arange(2.0, 20.0, 0.2)
        phi_orb = np.linspace(0.0, 2*np.pi, 300)
        cosp = np.cos(phi_orb)
        sinp = np.sin(phi_orb)

        figR, axR = plt.subplots(1, 1, figsize=(8, 6))
        figA, axA = plt.subplots(1, 1, figsize=(8, 6))
        figP, axP = plt.subplots(1, 1, figsize=(8, 6))
        dp.plot.plotZSlice(figR, axR, dat[0], dat[3], r, rho, z, r"$\Sigma$",
                           pars, opts, log=True, rmax=mapRmax)
        dp.plot.plotZSlice(figA, axA, dat[0], dat[3], r, rho, z, r"$\Sigma$",
                           pars, opts, log=True, rmax=mapRmax)
        dp.plot.plotZSlice(figP, axP, dat[0], dat[3], r, rho, z, r"$\Sigma$",
                           pars, opts, log=True, rmax=mapRmax)

        for rr in rs:
            i = np.searchsorted(rjph, rr) - 1
            # print(R[i], dr[i], dM_r[i], dM_a[i], dM_p[i])
            p_orb = dMp_r[i] / dM_r[i]
            ecc_orb = dMecc_r[i] / dM_r[i]
            eccx_orb = dMeccx_r[i] / dM_r[i]
            eccy_orb = dMeccy_r[i] / dM_r[i]
            phip_orb = dMphip_r[i] / dM_r[i]
            r_orb1 = p_orb / (1 + ecc_orb*np.cos(phi_orb-phip_orb))

            ecc_orb2 = math.sqrt(eccx_orb**2 + eccy_orb**2)
            phip_orb2 = math.atan2(eccy_orb, eccx_orb)
            r_orb2 = p_orb / (1 + ecc_orb2*np.cos(phi_orb-phip_orb2))

            # axR.plot(r_orb1 * cosp, r_orb1 * sinp, lw=1, color='w', ls='-')
            axR.plot(r_orb2 * cosp, r_orb2 * sinp, lw=1, color='w', ls='--')
            
            p_orb = dMp_a[i] / dM_a[i]
            ecc_orb = dMecc_a[i] / dM_a[i]
            eccx_orb = dMeccx_a[i] / dM_a[i]
            eccy_orb = dMeccy_a[i] / dM_a[i]
            phip_orb = dMphip_a[i] / dM_a[i]
            r_orb1 = p_orb / (1 + ecc_orb*np.cos(phi_orb-phip_orb))

            ecc_orb2 = math.sqrt(eccx_orb**2 + eccy_orb**2)
            phip_orb2 = math.atan2(eccy_orb, eccx_orb)
            r_orb2 = p_orb / (1 + ecc_orb2*np.cos(phi_orb-phip_orb2))

            # axA.plot(r_orb1 * cosp, r_orb1 * sinp, lw=1, color='w', ls='-')
            axA.plot(r_orb2 * cosp, r_orb2 * sinp, lw=1, color='w', ls='--')
            
            p_orb = R[i]
            ecc_orb = dMecc_p[i] / dM_p[i]
            eccx_orb = dMeccx_p[i] / dM_p[i]
            eccy_orb = dMeccy_p[i] / dM_p[i]
            phip_orb = dMphip_p[i] / dM_p[i]
            r_orb1 = p_orb / (1 + ecc_orb*np.cos(phi_orb-phip_orb))

            ecc_orb2 = math.sqrt(eccx_orb**2 + eccy_orb**2)
            phip_orb2 = math.atan2(eccy_orb, eccx_orb)
            r_orb2 = p_orb / (1 + ecc_orb2*np.cos(phi_orb-phip_orb2))

            # axP.plot(r_orb1 * cosp, r_orb1 * sinp, lw=1, color='w', ls='-')
            axP.plot(r_orb2 * cosp, r_orb2 * sinp, lw=1, color='w', ls='--')

        figname = plotDir / "sigOrb_r_{0:s}.{1:s}".format(name, ext)
        print("Saving", figname)
        figR.savefig(figname, dpi=200)
        plt.close(figR)

        figname = plotDir / "sigOrb_a_{0:s}.{1:s}".format(name, ext)
        print("Saving", figname)
        figA.savefig(figname, dpi=200)
        plt.close(figA)

        figname = plotDir / "sigOrb_p_{0:s}.{1:s}".format(name, ext)
        print("Saving", figname)
        figP.savefig(figname, dpi=200)
        plt.close(figP)



        cm = plt.cm.inferno
        cm.set_under('k')
        cm.set_bad('k')

        eccb = np.linspace(0.0, 1.0, 101)
        dM_dedr, _, _ = np.histogram2d(r, ecc, bins=(rjph, eccb),
                                       weights=rho*dV, density=True)
        dM_deda, _, _ = np.histogram2d(a, ecc, bins=(rjph, eccb),
                                       weights=rho*dV,density=True)
        dM_dedp, _, _ = np.histogram2d(p, ecc, bins=(rjph, eccb),
                                       weights=rho*dV, density=True)

        i1 = np.searchsorted(rjph, args.Rmin)
        i2 = np.searchsorted(rjph, args.Rmax)
        rb = rjph[i1:i2]

        dM_dedr /= R[:, None]
        dM_deda /= R[:, None]
        dM_dedp /= R[:, None]

        dM_dedr = dM_dedr[i1:i2-1, :]
        dM_deda = dM_deda[i1:i2-1, :]
        dM_dedp = dM_dedp[i1:i2-1, :]

        fig, ax = plt.subplots(1, 1)
        C = ax.pcolormesh(rb, eccb, dM_dedr.T,
                          norm=mpl.colors.LogNorm(), cmap=cm)
        ax.set(xlabel=r'$r$', xlim=(1.0, rjph[-1]),
               ylabel=r'$e$')
        fig.colorbar(C, extend='min')

        figname = plotDir / "ecc_r_{0:s}.{1:s}".format(name, ext)
        print("Saving", figname)
        fig.savefig(figname, dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(1, 1)
        C = ax.pcolormesh(rb, eccb, dM_deda.T,
                          norm=mpl.colors.LogNorm(), cmap=cm)
        ax.set(xlabel=r'$a$', xlim=(1.0, rjph[-1]),
               ylabel=r'$e$')
        fig.colorbar(C, extend='min')

        figname = plotDir / "ecc_a_{0:s}.{1:s}".format(name, ext)
        print("Saving", figname)
        fig.savefig(figname, dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(1, 1)
        C = ax.pcolormesh(rb, eccb, dM_dedp.T,
                          norm=mpl.colors.LogNorm(), cmap=cm)
        ax.set(xlabel=r'$p$', xlim=(1.0, rjph[-1]),
               ylabel=r'$e$')
        fig.colorbar(C, extend='min')

        figname = plotDir / "ecc_p_{0:s}.{1:s}".format(name, ext)
        print("Saving", figname)
        fig.savefig(figname, dpi=300)
        plt.close(fig)

    
def getPackStats(f, axis=0):

    f_mean = f.mean(axis=axis)
    f2_mean = (f**2).mean(axis=axis)
    df2 = f2_mean - f_mean**2
    df2[df2<0.0] = 0.0
    f_sig = np.sqrt(df2)

    f_mean_err = f_sig / np.sqrt(f.shape[axis])

    return np.array((f_mean, f_sig, f_mean_err))


def analyzeSingleArchive(archiveName, args):

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

    T_grav = T_grav - T_grav[:, -1][:, None]

    Jdot = T_grav - FJ_adv - FJ_visc
    ell = Jdot / Mdot

    l = R[None, :]**2 * Om
    FJ_acc = -Mdot*l
    FJ_rey = FJ_adv - FJ_acc
    
    plotDir = Path("plots")
    if not plotDir.exists():
        plotDir.mkdir()

    if args.makeFrames:
        for i in range(nt):
            name = "{0:04d}".format(i)
            makeFramePlots(t[i], R, Sig[i], Pi[i], Vr[i], Om[i], Mdot[i],
                           Jdot[i], ell[i], FJ_adv[i], FJ_rey[i], FJ_visc[i],
                           T_grav[i], Qheat_visc[i],
                           nu, Mdot0, name, plotDir, args, 'png')

    mask = (tP >= args.Tmin) & (tP <= args.Tmax)

    SigPack = getPackStats(Sig[mask])
    PiPack = getPackStats(Pi[mask])
    VrPack = getPackStats(Vr[mask])
    OmPack = getPackStats(Om[mask])
    MdotPack = getPackStats(Mdot[mask])
    JdotPack = getPackStats(Jdot[mask])
    # ellPack = getPackStats(ell[mask])
    FJ_advPack = getPackStats(FJ_adv[mask])
    FJ_reyPack = getPackStats(FJ_rey[mask])
    FJ_viscPack = getPackStats(FJ_visc[mask])
    T_gravPack = getPackStats(T_grav[mask])
    Qheat_viscPack = getPackStats(Qheat_visc[mask])

    ell = JdotPack[0] / MdotPack[0]

    makeFramePlots(t[-1]-t[0], R, SigPack, PiPack, VrPack, OmPack, MdotPack,
                   JdotPack, ell, FJ_advPack, FJ_reyPack, FJ_viscPack,
                   T_gravPack, Qheat_viscPack,
                   nu, Mdot0, "tAvg", plotDir, args, "pdf", 0.4)

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

    ax[0].set(xlabel=r'$t$ ($T_{\mathrm{bin}}$)',
              ylabel=r'$M_{\mathrm{tot}}$')
    ax[1].set(xlabel=r'$t$ ($T_{\mathrm{bin}}$)',
              ylabel=r'$J_{\mathrm{tot}}$')

    figname = plotDir / "cons_timeSeries.pdf"
    print("Saving", figname)
    fig.savefig(figname)
    plt.close(fig)

    iR1 = np.searchsorted(R, 4.0)
    iR2 = np.searchsorted(R, 5.0)
    iR3 = np.searchsorted(R, 6.0)
    iR4 = np.searchsorted(R, 7.0)
    iR5 = np.searchsorted(R, 8.0)

    kwargs1 = {'ls': '-', 'lw': 1, 'color': 'C0',
               'label': r'$R = {0:.1f}$ a'.format(R[iR1])}
    kwargs2 = {'ls': '-', 'lw': 1, 'color': 'C1',
               'label': r'$R = {0:.1f}$ a'.format(R[iR2])}
    kwargs3 = {'ls': '-', 'lw': 1, 'color': 'C2',
               'label': r'$R = {0:.1f}$ a'.format(R[iR3])}
    kwargs4 = {'ls': '-', 'lw': 1, 'color': 'C3',
               'label': r'$R = {0:.1f}$ a'.format(R[iR4])}
    kwargs5 = {'ls': '-', 'lw': 1, 'color': 'C5',
               'label': r'$R = {0:.1f}$ a'.format(R[iR5])}

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    ax[0, 0].plot(tP[sortInds], Sig[sortInds, iR1], **kwargs1)
    ax[0, 0].plot(tP[sortInds], Sig[sortInds, iR2], **kwargs2)
    ax[0, 0].plot(tP[sortInds], Sig[sortInds, iR3], **kwargs3)
    ax[0, 0].plot(tP[sortInds], Sig[sortInds, iR4], **kwargs4)
    ax[0, 0].plot(tP[sortInds], Sig[sortInds, iR5], **kwargs5)
    ax[0, 1].plot(tP[sortInds], Mdot[sortInds, iR1], **kwargs1)
    ax[0, 1].plot(tP[sortInds], Mdot[sortInds, iR2], **kwargs2)
    ax[0, 1].plot(tP[sortInds], Mdot[sortInds, iR3], **kwargs3)
    ax[0, 1].plot(tP[sortInds], Mdot[sortInds, iR4], **kwargs4)
    ax[0, 1].plot(tP[sortInds], Mdot[sortInds, iR5], **kwargs5)
    ax[1, 0].plot(tP[sortInds], Jdot[sortInds, iR1], **kwargs1)
    ax[1, 0].plot(tP[sortInds], Jdot[sortInds, iR2], **kwargs2)
    ax[1, 0].plot(tP[sortInds], Jdot[sortInds, iR3], **kwargs3)
    ax[1, 0].plot(tP[sortInds], Jdot[sortInds, iR4], **kwargs4)
    ax[1, 0].plot(tP[sortInds], Jdot[sortInds, iR5], **kwargs5)
    ax[1, 1].plot(tP[sortInds], (Jdot/Mdot)[sortInds, iR1], **kwargs1)
    ax[1, 1].plot(tP[sortInds], (Jdot/Mdot)[sortInds, iR2], **kwargs2)
    ax[1, 1].plot(tP[sortInds], (Jdot/Mdot)[sortInds, iR3], **kwargs3)
    ax[1, 1].plot(tP[sortInds], (Jdot/Mdot)[sortInds, iR4], **kwargs4)
    ax[1, 1].plot(tP[sortInds], (Jdot/Mdot)[sortInds, iR5], **kwargs5)

    ax[0, 0].legend()
    ax[0, 0].set(xlabel=r'$t$ ($T_{\mathrm{bin}}$)',
                 ylabel=r'$\Sigma(R)$')
    ax[0, 1].set(xlabel=r'$t$ ($T_{\mathrm{bin}}$)',
                 ylabel=r'$\dot{M}(R)$')
    ax[1, 0].set(xlabel=r'$t$ ($T_{\mathrm{bin}}$)',
                 ylabel=r'$\dot{J}(R)$')
    ax[1, 1].set(xlabel=r'$t$ ($T_{\mathrm{bin}}$)',
                 ylabel=r'$\ell(R)$', ylim=(-2, 2))

    figname = plotDir / "flux_timeSeries.pdf"
    print("Saving", figname)
    fig.savefig(figname)
    plt.close(fig)

    dt = (t[-1] - t[0]) / (nt - 1)

    nw = max(1, int(2*np.pi*args.window / dt))

    Mdot_window = scipy.signal.convolve2d(Mdot, np.ones((nw, 1))/nw,
                                          mode='valid')
    Jdot_window = scipy.signal.convolve2d(Jdot, np.ones((nw, 1))/nw,
                                          mode='valid')
    ell_window = Jdot_window/Mdot_window

    di = max(1, int(2*np.pi*args.interval / dt))

    goodR = (R >= args.Rmin) & (R <= args.Rmax)

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    ax[0, 0].plot(R[goodR], Mdot_window[-1, goodR])
    ax[0, 0].plot(R[goodR], Mdot_window[-di-1, goodR])
    ax[0, 0].plot(R[goodR], Mdot_window[-2*di-1, goodR])
    ax[0, 0].plot(R[goodR], Mdot_window[-3*di-1, goodR])
    ax[0, 0].plot(R[goodR], Mdot_window[-4*di-1, goodR])
    
    ax[0, 1].plot(R[goodR], Jdot_window[-1, goodR])
    ax[0, 1].plot(R[goodR], Jdot_window[-di-1, goodR])
    ax[0, 1].plot(R[goodR], Jdot_window[-2*di-1, goodR])
    ax[0, 1].plot(R[goodR], Jdot_window[-3*di-1, goodR])
    ax[0, 1].plot(R[goodR], Jdot_window[-4*di-1, goodR])
    
    ax[1, 0].plot(R[goodR], ell_window[-1, goodR])
    ax[1, 0].plot(R[goodR], ell_window[-di-1, goodR])
    ax[1, 0].plot(R[goodR], ell_window[-2*di-1, goodR])
    ax[1, 0].plot(R[goodR], ell_window[-3*di-1, goodR])
    ax[1, 0].plot(R[goodR], ell_window[-4*di-1, goodR])
    
    ax[1, 0].set(ylim=(-2, 2))

    figname = plotDir / "Jdot_r_timeWindow.pdf"
    print("Saving", figname)
    fig.savefig(figname)
    plt.close(fig)

    iR1 = np.searchsorted(R, 4.0)
    iR2 = np.searchsorted(R, 5.0)
    iR3 = np.searchsorted(R, 6.0)
    iR4 = np.searchsorted(R, 7.0)
    iR5 = np.searchsorted(R, 8.0)

    ntw = Mdot_window.shape[0]

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    ax[0, 0].plot(tP[-ntw:], Mdot_window[:, iR1])
    ax[0, 0].plot(tP[-ntw:], Mdot_window[:, iR2])
    ax[0, 0].plot(tP[-ntw:], Mdot_window[:, iR3])
    ax[0, 0].plot(tP[-ntw:], Mdot_window[:, iR4])
    ax[0, 0].plot(tP[-ntw:], Mdot_window[:, iR5])

    ax[0, 1].plot(tP[-ntw:], Jdot_window[:, iR1])
    ax[0, 1].plot(tP[-ntw:], Jdot_window[:, iR2])
    ax[0, 1].plot(tP[-ntw:], Jdot_window[:, iR3])
    ax[0, 1].plot(tP[-ntw:], Jdot_window[:, iR4])
    ax[0, 1].plot(tP[-ntw:], Jdot_window[:, iR5])

    ax[1, 0].plot(tP[-ntw:], ell_window[:, iR1])
    ax[1, 0].plot(tP[-ntw:], ell_window[:, iR2])
    ax[1, 0].plot(tP[-ntw:], ell_window[:, iR3])
    ax[1, 0].plot(tP[-ntw:], ell_window[:, iR4])
    ax[1, 0].plot(tP[-ntw:], ell_window[:, iR5])

    ax[1, 0].set(ylim=(-2, 2))
    

    figname = plotDir / "Jdot_t_timeWindow.pdf"
    print("Saving", figname)
    fig.savefig(figname)
    plt.close(fig)


    

def analyzeSingleArchiveOrb(archiveName, args):

    with h5.File(archiveName, "r") as f:
        t = f['t'][...]
        R = f['R'][...]
        rjph = f['rjph'][...]
        Sig = f['Sig'][...]
        dMecc_r = f['orb/dM_ecc'][...]
        dMeccx_r = f['orb/dM_eccx'][...]
        dMeccy_r = f['orb/dM_eccy'][...]
        dMphip_r = f['orb/dM_phip'][...]
        dV_a = f['orb_a/dV'][...]
        dM_a = f['orb_a/dM'][...]
        dMecc_a = f['orb_a/dM_ecc'][...]
        dMeccx_a = f['orb_a/dM_eccx'][...]
        dMeccy_a = f['orb_a/dM_eccy'][...]
        dMphip_a = f['orb_a/dM_phip'][...]
        dV_p = f['orb_p/dV'][...]
        dM_p = f['orb_p/dM'][...]
        dMecc_p = f['orb_p/dM_ecc'][...]
        dMeccx_p = f['orb_p/dM_eccx'][...]
        dMeccy_p = f['orb_p/dM_eccy'][...]
        dMphip_p = f['orb_p/dM_phip'][...]
    
    plotDir = Path("plots")
    if not plotDir.exists():
        plotDir.mkdir()

    nt = len(t)
    tP = t / (2*np.pi)

    dr = rjph[1:]-rjph[:-1]
    Rc = 0.5*(rjph[1:] + rjph[:-1])
    dV_r = (2*np.pi*Rc * dr)
    dM_r = Sig * dV_r[None, :]

    n2 = 2*(len(R) // 2)
    n4 = 4*(len(R) // 4)

    goodT = (tP >= args.Tmin) & (tP <=args.Tmax)
    
    Sig_r_t = Sig
    ecc_r_t = (dMecc_r / dM_r)
    phip_r_t = (dMphip_r / dM_r)
    eccx_r_t = (dMeccx_r / dM_r)
    eccy_r_t = (dMeccy_r / dM_r)

    Sig_a_t = (dM_a / dV_a)
    ecc_a_t = (dMecc_a / dM_a)
    phip_a_t = (dMphip_a / dM_a)
    eccx_a_t = (dMeccx_a / dM_a)
    eccy_a_t = (dMeccy_a / dM_a)
    
    Sig_p_t = (dM_p / dV_p)
    ecc_p_t = (dMecc_p / dM_p)
    phip_p_t = (dMphip_p / dM_p)
    eccx_p_t = (dMeccx_p / dM_p)
    eccy_p_t = (dMeccy_p / dM_p)

    Sig_r = Sig_r_t[goodT, :].mean(axis=0)
    dSig_r = np.sqrt(((Sig_r_t[goodT, :] - Sig_r[None, :])**2).mean(axis=0))
    ecc_r = ecc_r_t[goodT, :].mean(axis=0)
    decc_r = np.sqrt(((ecc_r_t[goodT, :] - ecc_r[None, :])**2).mean(axis=0))

    Sig_a = Sig_a_t[goodT, :].mean(axis=0)
    dSig_a = np.sqrt(((Sig_a_t[goodT, :] - Sig_a[None, :])**2).mean(axis=0))
    ecc_a = ecc_a_t[goodT, :].mean(axis=0)
    decc_a = np.sqrt(((ecc_a_t[goodT, :] - ecc_a[None, :])**2).mean(axis=0))

    Sig_p = Sig_p_t[goodT, :].mean(axis=0)
    dSig_p = np.sqrt(((Sig_p_t[goodT, :] - Sig_p[None, :])**2).mean(axis=0))
    ecc_p = ecc_p_t[goodT, :].mean(axis=0)
    decc_p = np.sqrt(((ecc_p_t[goodT, :] - ecc_p[None, :])**2).mean(axis=0))

    ecc_xy_r_t = np.sqrt(eccx_r_t**2 + eccy_r_t**2)
    ecc_xy_a_t = np.sqrt(eccx_a_t**2 + eccy_a_t**2)
    ecc_xy_p_t = np.sqrt(eccx_p_t**2 + eccy_p_t**2)
    ecc_xy_r = ecc_xy_r_t[goodT, :].mean(axis=0)
    decc_xy_r = np.sqrt(((ecc_xy_r_t[goodT, :]
                          - ecc_xy_r[None, :])**2).mean(axis=0))
    ecc_xy_a = ecc_xy_a_t[goodT, :].mean(axis=0)
    decc_xy_a = np.sqrt(((ecc_xy_a_t[goodT, :]
                          - ecc_xy_a[None, :])**2).mean(axis=0))
    ecc_xy_p = ecc_xy_p_t[goodT, :].mean(axis=0)
    decc_xy_p = np.sqrt(((ecc_xy_p_t[goodT, :]
                          - ecc_xy_p[None, :])**2).mean(axis=0))

    


    goodR = (R >= args.Rmin) & (R <= args.Rmax)

    """
    Sig2_r = ((dM_r[:, :n2:2] + dM_r[:, 1:n2:2])
            / (dV_r[:n2:2] + dV_r[1:n2:2])[None, :]
              )[goodT, :].mean(axis=0)
    Sig4_r = ((dM_r[:, :n4:4] + dM_r[:, 1:n4:4] + dM_r[:, 2:n4:4]
               + dM_r[:, 3:n4:4])
              / (dV_r[:n4:4] + dV_r[1:n4:4] + dV_r[2:n4:4]
                 + dV_r[3:n4:4])[None, :]
              )[goodT, :].mean(axis=0)

    Sig2_a = ((dM_a[:, :n2:2] + dM_a[:, 1:n2:2])
              / (dV_a[:, :n2:2] + dV_a[:, 1:n2:2])
              )[goodT, :].mean(axis=0)
    Sig4_a = ((dM_a[:, :n4:4] + dM_a[:, 1:n4:4] + dM_a[:, 2:n4:4]
               + dM_a[:, 3:n4:4])
              / (dV_a[:, :n4:4] + dV_a[:, 1:n4:4] + dV_a[:, 2:n4:4]
                 + dV_a[:, 3:n4:4])
              )[goodT, :].mean(axis=0)

    Sig2_p = ((dM_p[:, :n2:2] + dM_p[:, 1:n2:2])
              / (dV_p[:, :n2:2] + dV_p[:, 1:n2:2])
              )[goodT, :].mean(axis=0)
    Sig4_p = ((dM_p[:, :n4:4] + dM_p[:, 1:n4:4] + dM_p[:, 2:n4:4]
               + dM_p[:, 3:n4:4])
               / (dV_p[:, :n4:4] + dV_p[:, 1:n4:4] + dV_p[:, 2:n4:4]
                  + dV_p[:, 3:n4:4])
              )[goodT, :].mean(axis=0)

    R2 = 0.5*(R[:n2:2] + R[1:n2:2])
    R4 = 0.25*(R[:n4:4] + R[1:n4:4] + R[2:n4:4] + R[3:n4:4])

    goodR2 = (R2 >= args.Rmin) & (R2 <= args.Rmax)
    goodR4 = (R4 >= args.Rmin) & (R4 <= args.Rmax)
    """


    fitR = (R >= 1.5) & (R <= 6)
    p0 = (10.0, 0.0, 3.0, 100.0, 2.0, 0.1)
    bnds = ((0.0, -5, 1.5, 1.0, 0.1, 0.01), (20.0, 5, 6.0, 200.0, 10.0, 10.0))
    popt_r, pcov_r = scipy.optimize.curve_fit(f_bpl, R[fitR], Sig_r[fitR], p0,
                                              dSig_r[fitR], absolute_sigma=True,
                                              bounds=bnds)
    popt_a, pcov_a = scipy.optimize.curve_fit(f_bpl, R[fitR], Sig_a[fitR], p0,
                                              dSig_a[fitR], absolute_sigma=True,
                                              bounds=bnds)
    popt_p, pcov_p = scipy.optimize.curve_fit(f_bpl, R[fitR], Sig_p[fitR], p0,
                                              dSig_p[fitR], absolute_sigma=True,
                                              bounds=bnds)

    print("Sig_r fit:", popt_r)
    print("Sig_a fit:", popt_a)
    print("Sig_p fit:", popt_p)
    
    SigFit_r = f_bpl(R, *popt_r)
    SigFit_a = f_bpl(R, *popt_a)
    SigFit_p = f_bpl(R, *popt_p)
    
    fitR = (R >= 1.0) & (R <= 10.0)
    p0 = (0.5, 100.0, 3.0, 1.0)
    bnds = ((-2.0, 1.0, 1.0, 0.1), (2.0, 200.0, 6.0, 20.0))
    popt_r, pcov_r = scipy.optimize.curve_fit(f_M20, R[fitR], Sig_r[fitR], p0,
                                              dSig_r[fitR], absolute_sigma=True,
                                              bounds=bnds)
    popt_a, pcov_a = scipy.optimize.curve_fit(f_M20, R[fitR], Sig_a[fitR], p0,
                                              dSig_a[fitR], absolute_sigma=True,
                                              bounds=bnds)
    popt_p, pcov_p = scipy.optimize.curve_fit(f_M20, R[fitR], Sig_p[fitR], p0,
                                              dSig_p[fitR], absolute_sigma=True,
                                              bounds=bnds)

    print("Sig_r fit2:", popt_r)
    print("Sig_a fit2:", popt_a)
    print("Sig_p fit2:", popt_p)
    
    SigFit2_r = f_M20(R, *popt_r)
    SigFit2_a = f_M20(R, *popt_a)
    SigFit2_p = f_M20(R, *popt_p)

    fig, ax = plt.subplots(3, 5, figsize=(16, 10))
    ax[0, 0].fill_between(R[goodR], (Sig_r-dSig_r)[goodR],
                          (Sig_r+dSig_r)[goodR], color='C0', alpha=0.1)
    ax[0, 0].plot(R[goodR], Sig_r[goodR])
    ax[0, 0].plot(R[goodR], SigFit_r[goodR], color='grey', ls='--', lw=1)
    ax[0, 0].plot(R[goodR], SigFit2_r[goodR], color='grey', ls=':', lw=1)
    ax[1, 0].fill_between(R[goodR], (Sig_a-dSig_a)[goodR],
                          (Sig_a+dSig_a)[goodR], color='C0', alpha=0.1)
    ax[1, 0].plot(R[goodR], Sig_a[goodR])
    ax[1, 0].plot(R[goodR], SigFit_a[goodR], color='grey', ls='--', lw=1)
    ax[1, 0].plot(R[goodR], SigFit2_a[goodR], color='grey', ls=':', lw=1)
    ax[2, 0].fill_between(R[goodR], (Sig_p-dSig_p)[goodR],
                          (Sig_p+dSig_p)[goodR], color='C0', alpha=0.1)
    ax[2, 0].plot(R[goodR], Sig_p[goodR])
    ax[2, 0].plot(R[goodR], SigFit_p[goodR], color='grey', ls='--', lw=1)
    ax[2, 0].plot(R[goodR], SigFit2_p[goodR], color='grey', ls=':', lw=1)
    
    ax[0, 1].fill_between(R[goodR], (ecc_r-decc_r)[goodR],
                          (ecc_r+decc_r)[goodR], color='C0', alpha=0.1, lw=0)
    ax[0, 1].plot(R[goodR], ecc_r[goodR], color='C0')
    ax[0, 1].fill_between(R[goodR], (ecc_xy_r-decc_xy_r)[goodR],
                          (ecc_xy_r+decc_xy_r)[goodR],
                          color='C1', alpha=0.1, lw=0)
    ax[0, 1].plot(R[goodR], ecc_xy_r[goodR], color='C1')
    ax[1, 1].fill_between(R[goodR], (ecc_a-decc_a)[goodR],
                          (ecc_a+decc_a)[goodR], color='C0', alpha=0.1, lw=0)
    ax[1, 1].plot(R[goodR], ecc_a[goodR])
    ax[1, 1].fill_between(R[goodR], (ecc_xy_a-decc_xy_a)[goodR],
                          (ecc_xy_a+decc_xy_a)[goodR],
                          color='C1', alpha=0.1, lw=0)
    ax[1, 1].plot(R[goodR], ecc_xy_a[goodR], color='C1')
    ax[2, 1].fill_between(R[goodR], (ecc_p-decc_p)[goodR],
                          (ecc_p+decc_p)[goodR], color='C0', alpha=0.1, lw=0)
    ax[2, 1].plot(R[goodR], ecc_p[goodR])
    ax[2, 1].fill_between(R[goodR], (ecc_xy_p-decc_xy_p)[goodR],
                          (ecc_xy_p+decc_xy_p)[goodR],
                          color='C1', alpha=0.1, lw=0)
    ax[2, 1].plot(R[goodR], ecc_xy_p[goodR], color='C1')

    sigScaleX = 'linear'
    sigScaleY = 'linear'

    ax[0, 0].set(xlabel=r'$r$', ylabel=r'$\Sigma$',
                 xscale=sigScaleX, yscale=sigScaleY)
    ax[1, 0].set(xlabel=r'$a$', ylabel=r'$\Sigma$',
                 xscale=sigScaleX, yscale=sigScaleY)
    ax[2, 0].set(xlabel=r'$p$', ylabel=r'$\Sigma$',
                 xscale=sigScaleX, yscale=sigScaleY)
    ax[0, 1].set(xlabel=r'$r$', ylabel=r'$e$')
    ax[1, 1].set(xlabel=r'$a$', ylabel=r'$e$')
    ax[2, 1].set(xlabel=r'$p$', ylabel=r'$e$')


    figname = plotDir / "sig_grid_{0:s}.{1:s}".format("ave", "pdf")
    print("Saving", figname)
    fig.savefig(figname, dpi=200)
    plt.close(fig)

    rcav_r = popt_r[2]
    acav_a = popt_a[2]
    pcav_p = popt_p[2]

    GM = 1.0

    ir2 = np.searchsorted(R, 2.0)
    ir3 = np.searchsorted(R, 3.0)
    ir4 = np.searchsorted(R, 4.0)
    ir5 = np.searchsorted(R, 5.0)

    cols = ['C{0:d}'.format(i) for i in range(10)]

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    for i, ir in enumerate([ir2, ir3, ir4, ir5]):
        ax[0, 0].plot(tP, ecc_r_t[:, ir], color=cols[i], ls='-')
        ax[0, 0].plot(tP, ecc_a_t[:, ir], color=cols[i], ls='--')
        ax[0, 0].plot(tP, ecc_p_t[:, ir], color=cols[i], ls=':')
        ax[0, 1].plot(tP, phip_r_t[:, ir], color=cols[i], ls='-')
        ax[0, 1].plot(tP, phip_a_t[:, ir], color=cols[i], ls='--')
        ax[0, 1].plot(tP, phip_p_t[:, ir], color=cols[i], ls=':')
        ax[1, 0].plot(tP, eccx_r_t[:, ir], color=cols[i], ls='-')
        ax[1, 0].plot(tP, eccx_a_t[:, ir], color=cols[i], ls='--')
        ax[1, 0].plot(tP, eccx_p_t[:, ir], color=cols[i], ls=':')
        ax[1, 1].plot(tP, eccy_r_t[:, ir], color=cols[i], ls='-')
        ax[1, 1].plot(tP, eccy_a_t[:, ir], color=cols[i], ls='--')
        ax[1, 1].plot(tP, eccy_p_t[:, ir], color=cols[i], ls=':')

    ax[0, 0].set(xlabel=r'Time (orbits)', ylabel=r'$e$')
    ax[0, 1].set(xlabel=r'Time (orbits)', ylabel=r'$\phi_p$')
    ax[1, 0].set(xlabel=r'Time (orbits)', ylabel=r'$e_x$')
    ax[1, 1].set(xlabel=r'Time (orbits)', ylabel=r'$e_y$')

    figname = plotDir / "cav_phase.pdf"
    print("Saving", figname)
    fig.savefig(figname)
    plt.close(fig)


def analyzeCheckpoints(filenames, args):

    nt = len(filenames)
    pars = dp.util.loadPars(filenames[0])
    t, r, phi, z, prim, dat = dp.util.loadCheckpoint(filenames[0])
    rjph = dat[0]
    R = np.unique(r)

    planetDat = dat[4]

    nr = len(rjph)-1

    print("Initializing archive HDF5 file:", args.archive)

    with h5.File(args.archive, "w") as f:
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

        f.create_group("orb")
        f.create_dataset("orb/dM_e", (nt, nr), dtype=float)
        f.create_dataset("orb/dM_ecc", (nt, nr), dtype=float)
        f.create_dataset("orb/dM_eccx", (nt, nr), dtype=float)
        f.create_dataset("orb/dM_eccy", (nt, nr), dtype=float)
        f.create_dataset("orb/dM_p", (nt, nr), dtype=float)
        f.create_dataset("orb/dM_a", (nt, nr), dtype=float)
        f.create_dataset("orb/dM_phip", (nt, nr), dtype=float)
        f.create_group("orb_a")
        f.create_dataset("orb_a/dV", (nt, nr), dtype=float)
        f.create_dataset("orb_a/dM", (nt, nr), dtype=float)
        f.create_dataset("orb_a/dM_e", (nt, nr), dtype=float)
        f.create_dataset("orb_a/dM_ecc", (nt, nr), dtype=float)
        f.create_dataset("orb_a/dM_eccx", (nt, nr), dtype=float)
        f.create_dataset("orb_a/dM_eccy", (nt, nr), dtype=float)
        f.create_dataset("orb_a/dM_p", (nt, nr), dtype=float)
        f.create_dataset("orb_a/dM_a", (nt, nr), dtype=float)
        f.create_dataset("orb_a/dM_phip", (nt, nr), dtype=float)
        f.create_group("orb_p")
        f.create_dataset("orb_p/dV", (nt, nr), dtype=float)
        f.create_dataset("orb_p/dM", (nt, nr), dtype=float)
        f.create_dataset("orb_p/dM_e", (nt, nr), dtype=float)
        f.create_dataset("orb_p/dM_ecc", (nt, nr), dtype=float)
        f.create_dataset("orb_p/dM_eccx", (nt, nr), dtype=float)
        f.create_dataset("orb_p/dM_eccy", (nt, nr), dtype=float)
        f.create_dataset("orb_p/dM_p", (nt, nr), dtype=float)
        f.create_dataset("orb_p/dM_a", (nt, nr), dtype=float)
        f.create_dataset("orb_p/dM_phip", (nt, nr), dtype=float)

        f.create_group("Pars")
        for key in pars:
            f['Pars'].create_dataset(key, data=pars[key])

    if args.ncpu is None or args.ncpu <= 1:
        for i, fname in enumerate(filenames):
            analyzeSingle((i, fname), args)
    else:
        map_fn = functools.partial(analyzeSingle, args=args)
        with Pool(args.ncpu) as p:
            p.map(map_fn, enumerate(filenames))

    return args.archive


def analyzeArchives(filenames, args):

    for filename in filenames:
        if not args.noDisk:
            analyzeSingleArchive(filename, args)
        if not args.noOrb:
            analyzeSingleArchiveOrb(filename, args)


def analyze(filenames, args):

    checkpoints = False
    archives = False

    with h5.File(filenames[0], "r") as f:
        if 'GIT_VERSION' in f and 'Grid' in f and 'Data' in f:
            checkpoints = True
        elif 'Mdot' in f and 'Pars' in f and 'rjph' in f:
            archives = True

    if checkpoints:
        analyzeCheckpoints(filenames, args)
        args.makeFrames = False
        analyzeSingleArchive(args.archive, args)

    elif archives:
        analyzeArchives(filenames, args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analyze accretion disks")
    parser.add_argument('files', nargs='+')
    parser.add_argument('--ncpu', nargs='?', default=1, type=int)
    parser.add_argument('--Rmin', nargs='?', default=0.0, type=np.float)
    parser.add_argument('--Rmax', nargs='?', default=np.inf, type=np.float)
    parser.add_argument('--Tmin', nargs='?', default=0, type=np.float)
    parser.add_argument('--Tmax', nargs='?', default=np.inf, type=np.float)
    parser.add_argument('--window', nargs='?', default=10, type=np.float)
    parser.add_argument('--interval', nargs='?', default=100, type=np.float)
    parser.add_argument('--makeFrames', action='store_true')
    parser.add_argument('--archive', nargs='?', default='diskFluxArchive.h5')
    parser.add_argument('--noDisk', action='store_true')
    parser.add_argument('--noOrb', action='store_true')

    args = parser.parse_args()

    filenames = [Path(x) for x in args.files]
    analyze(filenames, args) 
