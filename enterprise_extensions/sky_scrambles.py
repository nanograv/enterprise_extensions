# -*- coding: utf-8 -*-

import pickle
import sys
import time

import numpy as np
from enterprise.signals import utils


def compute_match(orf1, orf1_mag, orf2, orf2_mag):
    """Computes the match between two different ORFs."""

    match = np.abs(np.dot(orf1, orf2))/(orf1_mag*orf2_mag)

    return match


def make_true_orf(psrs):
    """Computes the ORF by looping over pulsar pairs"""

    npsr = len(psrs)

    orf = np.zeros(int(npsr*(npsr-1)/2))

    idx = 0
    for i in range(npsr):
        for j in range(i+1, npsr):

            orf[idx] = utils.hd_orf(psrs[i].pos, psrs[j].pos)

            idx += 1

    return orf


def compute_orf(ptheta, pphi):
    """
    Computes the ORF coefficient. Takes different input than utils.hd_orf().

    :param ptheta: Array of values of pulsar positions theta
    :param pphi: Array of values of pulsar positions phi

    :returns:
        orf: ORF for the given positions
        orf_mag: Magnitude of the ORF
    """
    npsr = len(ptheta)
    pos = [np.array([np.cos(phi)*np.sin(theta),
                     np.sin(phi)*np.sin(theta),
                     np.cos(theta)]) for phi, theta in zip(pphi, ptheta)]

    x = []
    for i in range(npsr):
        for j in range(i+1, npsr):
            x.append(np.dot(pos[i], pos[j]))
    x = np.array(x)

    orf = 1.5*(1./3. + (1.-x)/2.*(np.log((1.-x)/2.)-1./6.))

    return orf, np.sqrt(np.dot(orf, orf))


def get_scrambles(psrs, N=500, Nmax=10000, thresh=0.1,
                  filename='sky_scrambles.npz', resume=False):
    """
    Get sky scramble ORFs and matches.

    :param psrs: List of pulsar objects
    :param N: Number of desired sky scrambles
    :param Nmax: Maximum number of tries to get independent scrambles
    :param thresh: Threshold value for match statistic.
    :param filename: Name of the file where the sky scrambles should be saved.
                     Sky scrambles should be saved in *.npz file.
    :param resume: Whether to resume from an earlier run.
    """

    # compute the unscrambled ORF
    orf_true = make_true_orf(psrs)
    orf_true_mag = np.sqrt(np.dot(orf_true, orf_true))

    npsr = len(psrs)

    print('Generating {0} sky scrambles from {1} attempts with threshold {2}...'.format(N, Nmax, thresh))

    orf_mags = []

    if resume:
        print('Resuming from earlier run... loading sky scrambles from file {0}'.format(filename))
        npzfile = np.load(filename)
        matches, orfs = npzfile['matches'], npzfile['orfs']
        thetas, phis = npzfile['thetas'], npzfile['phis']
        print('{0} sky scrambles have already been generated.'.format(len(matches)))
        for o in orfs:
            orf_mags.append(np.sqrt(np.dot(o, o)))
    else:
        matches, orfs, thetas, phis = [], [], [], []

    ct = 0
    tstart = time.time()
    while ct <= Nmax and len(matches) <= N:
        ptheta = np.arccos(np.random.uniform(-1, 1, npsr))
        pphi = np.random.uniform(0, 2*np.pi, npsr)
        orf_s, orf_s_mag = compute_orf(ptheta, pphi)
        match = compute_match(orf_true, orf_true_mag, orf_s, orf_s_mag)
        if thresh == 1.0:
            if ct == 0:
                print('There is no threshold! Keep all the sky scrambles')
            if len(orfs) == 0:
                orfs.append(orf_s)
                matches.append(match)
                orfs = np.array(orfs)
                matches = np.array(matches)
                thetas = ptheta[np.newaxis, ...]
                phis = pphi[np.newaxis, ...]
                orf_mags.append(np.sqrt(np.dot(orf_s, orf_s)))
            else:
                matches = np.append(matches, match)
                orf_reshape = np.vstack(orf_s).T
                orfs = np.append(orfs, orf_reshape, axis=0)
                orf_mags.append(orf_s_mag)
                thetas = np.concatenate((thetas, [ptheta]))
                phis = np.concatenate((phis, [pphi]))
        elif thresh < 1.0 and match <= thresh:
            if len(orfs) == 0:
                orfs.append(orf_s)
                matches.append(match)
                orfs = np.array(orfs)
                matches = np.array(matches)
                thetas = ptheta[np.newaxis, ...]
                phis = pphi[np.newaxis, ...]
                orf_mags.append(np.sqrt(np.dot(orf_s, orf_s)))
            else:
                check = np.all(map(lambda x, y: compute_match(orf_s, orf_s_mag, x, y)<=thresh, orfs, orf_mags))
                if check:
                    matches = np.append(matches, match)
                    orf_reshape = np.vstack(orf_s).T
                    orfs = np.append(orfs, orf_reshape, axis=0)
                    orf_mags.append(orf_s_mag)
                    thetas = np.concatenate((thetas, [ptheta]))
                    phis = np.concatenate((phis, [pphi]))

        ct += 1
        if ct % 1000 == 0:
            sys.stdout.write('\r')
            sys.stdout.write('Finished %2.1f percent in %f min'
                             % (float(ct)/N*100, (time.time() - tstart)/60.))
            sys.stdout.flush()

    if len(matches) < N:
        print('\nGenerated {0} matches rather than the desired {1} matches'.format(len(matches), N))
    else:
        print('\nGenerated the desired {0} matches in {1} attempts'.format(len(matches), ct))
    print('Total runtime: {0:.1f} min'.format((time.time()-tstart)/60.))

    np.savez(filename, matches=matches, orfs=orfs, thetas=thetas, phis=phis)

    return (matches, orfs, thetas, phis)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--picklefile',
                        help='pickle file for the pulsars')
    parser.add_argument('--threshold', default=0.1,
                        help='threshold for sky scrambles (DEFAULT 0.1)')
    parser.add_argument('--nscrambles', default=1000,
                        help='number of sky scrambles to generate (DEFAULT 1000)')
    parser.add_argument('--nmax', default=1000,
                        help='maximum number of attempts (DEFAULT 1000)')
    parser.add_argument('--savefile', default='../data/scrambles_nano.npz',
                        help='outputfile name')
    parser.add_argument('--resume', action='store_true',
                        help='resume from existing savefile?')

    args = parser.parse_args()

    with open(args.picklefile, 'rb') as f:
        psrs = pickle.load(f)

    get_scrambles(psrs, N=int(args.nscrambles), Nmax=int(args.nmax), thresh=float(args.threshold),
                  filename=args.savefile, resume=args.resume)
