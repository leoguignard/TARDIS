#!python
# This file is subject to the terms and conditions defined in
# file 'LICENCE', which is part of this source code package.
# Author: Leo Guignard (guignardl...@AT@...janelia.hhmi.org)

import pandas as pd
import scipy as sp
from scipy import interpolate
from itertools import product
import re
import sys
import os
from TGMMlibraries import lineageTree
import sys
from time import time
import struct
from multiprocessing import Pool
from itertools import combinations
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import kdtree
from scipy.spatial import Delaunay
from scipy import spatial
kdtree.node = kdtree.KDTree.node
kdtree.leafnode = kdtree.KDTree.leafnode
kdtree.innernode = kdtree.KDTree.innernode
import itertools as it
from scipy import spatial
from scipy.optimize import linear_sum_assignment
from copy import deepcopy
from scipy import ndimage as nd
from copy import deepcopy
from Transformations.transformations import quaternion_from_matrix, quaternion_matrix, quaternion_slerp

def rigid_transform_3D(A, B):
    ''' Compute the 4x4 matrix reprenting the 3D rigid rotation
        minimizing the least squares between two paired sets of points *A* and *B*
        Args:
            A: Nx3 array, the first set of 3D points
            B: Nx3 array, the second set of 3D points
        Returns:
            out: 4x4 array, 3D rigid rotation
    '''
    assert len(A) == len(B)
    
    if not type(A) == np.matrix:
        A = np.matrix(A)

    if not type(B) == np.matrix:
        B = np.matrix(B)

    N = A.shape[0]; # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.transpose(AA) * BB

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T * U.T

    t = -R*centroid_A.T + centroid_B.T

    out = np.identity(4)
    out[:3, :3] = R
    out[:-1, 3:] = t
    return out


def write_header_am_2(f, nb_points, length):
    ''' Header for Amira .am files
    '''
    f.write('# AmiraMesh 3D ASCII 2.0\n')
    f.write('define VERTEX %d\n'%(nb_points*2))
    f.write('define EDGE %d\n'%nb_points)
    f.write('define POINT %d\n'%((length)*nb_points))
    f.write('Parameters {\n')
    f.write('\tContentType "HxSpatialGraph"\n')
    f.write('}\n')

    f.write('VERTEX { float[3] VertexCoordinates } @1\n')
    f.write('EDGE { int[2] EdgeConnectivity } @2\n')
    f.write('EDGE { int NumEdgePoints } @3\n')
    f.write('POINT { float[3] EdgePointCoordinates } @4\n')
    f.write('VERTEX { float Vcolor } @5\n')
    f.write('VERTEX { int Vbool } @6\n')
    f.write('EDGE { float Ecolor } @7\n')
    f.write('VERTEX { int Vbool2 } @8\n')

def write_to_am_2(path_format, LT_to_print, t_b = None, t_e = None, length = 5, manual_labels = None, 
                  default_label = 5, new_pos = None, to_take_time = None):
    ''' Writes a lineageTree into an Amira readable data (.am format).
        Args:
            path_format: string, path to the output. It should contain 1 %03d where the time step will be entered
            LT_to_print: lineageTree, lineageTree to write
            t_b: int, first time point to write (if None, min(LT.to_take_time) is taken)
            t_e: int, last time point to write (if None, max(LT.to_take_time) is taken)
                note: if there is no 'to_take_time' attribute, LT_to_print.time_nodes is considered instead
                    (historical)
            length: int, length of the track to print (how many time before).
            manual_labels: {id: label, }, dictionary that maps cell ids to 
            default_label: int, default value for the manual label
            new_pos: {id: [x, y, z]}, dictionary that maps a 3D position to a cell ID.
                if new_pos == None (default) then LT_to_print.pos is considered.
            to_take_time: {t, [int, ]}
    '''
    if to_take_time is None:
        to_take_time = LT_to_print.time_nodes
    if t_b is None:
        t_b = min(to_take_time.keys())
    if t_e is None:
        t_e = max(to_take_time.keys())
    if new_pos is None:
        new_pos = LT_to_print.pos

    if manual_labels is None:
        manual_labels = {}
    for t in range(t_b, t_e + 1):
        f = open(path_format%t, 'w')
        nb_points = len(to_take_time[t])
        write_header_am_2(f, nb_points, length)
        points_v = {}
        for C in to_take_time[t]:
            C_tmp = C
            positions = []
            for i in xrange(length):
                c_before = C_tmp
                C_tmp = LT_to_print.predecessor.get(C_tmp, [C_tmp])[0]
                while not C_tmp in new_pos and C_tmp in LT_to_print.predecessor:
                    C_tmp = LT_to_print.predecessor.get(C_tmp, [C_tmp])[0]
                if not C_tmp in new_pos:
                    C_tmp = c_before
                positions.append(np.array(new_pos[C_tmp]))
            points_v[C] = positions

        f.write('@1\n')
        for i, C in enumerate(to_take_time[t]):
            f.write('%f %f %f\n'%tuple(points_v[C][0]))
            f.write('%f %f %f\n'%tuple(points_v[C][-1]))

        f.write('@2\n')
        for i, C in enumerate(to_take_time[t]):
            f.write('%d %d\n'%(2*i, 2*i+1))

        f.write('@3\n')
        for i, C in enumerate(to_take_time[t]):
            f.write('%d\n'%(length))

        f.write('@4\n')
        tmp_velocity = {}
        for i, C in enumerate(to_take_time[t]):
            for p in points_v[C]:
                f.write('%f %f %f\n'%tuple(p))

        f.write('@5\n')
        for i, C in enumerate(to_take_time[t]):
            f.write('%f\n'%(manual_labels.get(C, default_label)))
            f.write('%f\n'%(0))

        f.write('@6\n')
        for i, C in enumerate(to_take_time[t]):
            f.write('%d\n'%(int(manual_labels.get(C, default_label) != default_label)))
            f.write('%d\n'%(0))
        
        f.write('@7\n')
        for i, C in enumerate(to_take_time[t]):
            f.write('%f\n'%(np.linalg.norm(points_v[C][0] - points_v[C][-1])))

        f.write('@8\n')
        for i, C in enumerate(to_take_time[t]):
            f.write('%d\n'%(1))
            f.write('%d\n'%(0))

        f.close()

def write_to_am_rem(path_format, LT_to_print, t_b = None, t_e = None, length = 5, manual_labels = None, 
                  default_label = 5, new_pos = None, to_take_time = None, to_remove = None, predecessor = None):
    ''' Writes a lineageTree into an Amira readable data (.am format).
        Args:
            path_format: string, path to the output. It should contain 1 %03d where the time step will be entered
            LT_to_print: lineageTree, lineageTree to write
            t_b: int, first time point to write (if None, min(LT.to_take_time) is taken)
            t_e: int, last time point to write (if None, max(LT.to_take_time) is taken)
                note: if there is no 'to_take_time' attribute, LT_to_print.time_nodes is considered instead
                    (historical)
            length: int, length of the track to print (how many time before).
            manual_labels: {id: label, }, dictionary that maps cell ids to 
            default_label: int, default value for the manual label
            new_pos: {id: [x, y, z]}, dictionary that maps a 3D position to a cell ID.
                if new_pos == None (default) then LT_to_print.pos is considered.
            to_take_time: {t, [int, ]}
            to_remove: [int, ], list of id not to take into account
            predecessor: {int: [int, ]}, dictionary that maps a cell id to the list of its
                predecesors
    '''
    if to_take_time is None:
        to_take_time = LT_to_print.time_nodes
    if t_b is None:
        t_b = min(to_take_time.keys())
    if t_e is None:
        t_e = max(to_take_time.keys())
    if new_pos is None:
        new_pos = LT_to_print.pos
    if to_remove is None:
        to_remove = set()
    if predecessor is None:
        predecessor = LT_to_print.predecessor

    if manual_labels is None:
        manual_labels = {}
    for t in range(t_b, t_e + 1):
        f = open(path_format%t, 'w')
        Points = [c for c in to_take_time[t] if not c in to_remove]
        nb_points = len(Points)
        write_header_am_2(f, nb_points, length)
        points_v = {}
        for C in Points:
            C_tmp = C
            positions = []
            for i in xrange(length):
                c_before = C_tmp
                C_tmp = predecessor.get(C_tmp, [C_tmp])[0]
                while not C_tmp in new_pos and C_tmp in predecessor:
                    C_tmp = predecessor.get(C_tmp, [C_tmp])[0]
                if not C_tmp in new_pos:
                    C_tmp = c_before
                positions.append(np.array(new_pos[C_tmp]))
            points_v[C] = positions

        f.write('@1\n')
        for i, C in enumerate(Points):
            f.write('%f %f %f\n'%tuple(points_v[C][0]))
            f.write('%f %f %f\n'%tuple(points_v[C][-1]))

        f.write('@2\n')
        for i, C in enumerate(Points):
            f.write('%d %d\n'%(2*i, 2*i+1))

        f.write('@3\n')
        for i, C in enumerate(Points):
            f.write('%d\n'%(length))

        f.write('@4\n')
        tmp_velocity = {}
        for i, C in enumerate(Points):
            for p in points_v[C]:
                f.write('%f %f %f\n'%tuple(p))

        f.write('@5\n')
        for i, C in enumerate(Points):
            f.write('%f\n'%(manual_labels.get(C, default_label)))
            f.write('%f\n'%(0))

        f.write('@6\n')
        for i, C in enumerate(Points):
            f.write('%d\n'%(int(manual_labels.get(C, default_label) != default_label)))
            f.write('%d\n'%(0))
        
        f.write('@7\n')
        for i, C in enumerate(Points):
            f.write('%f\n'%(np.linalg.norm(points_v[C][0] - points_v[C][1])))
            # if 15 < np.linalg.norm(points_v[C][0] - points_v[C][1]):
            #     print C

        f.write('@8\n')
        for i, C in enumerate(Points):
            f.write('%d\n'%(1))
            f.write('%d\n'%(0))

        f.close()

def interpolate_transformations(matrices):
    ''' Provided a mapping time -> 4D rigid transformation matrices,
        interpolates the missing rigid matrices
        Args:
            matrices: {int: 4x4 array, }, dictionary where the key is time
                and the value is the associated transformation for that time
        Returns:
            all_matrices: {int, 4x4 array, }, dictionary where the key is time
                and the value is the associated transformation for that time.
                The previously missing times from *matrices* being interpolated
    '''
    all_matrices = {}
    existing_times = np.array(sorted(matrices.keys()))
    times = np.arange(min(existing_times), max(existing_times)+1)
    for t in times:
        t_b, t_a = sorted(existing_times[np.argsort(np.abs(t - existing_times))][:2])
        if t != t_b and t != t_b:
            mat1 = matrices[t_b]
            mat2 = matrices[t_a]
            qa = quaternion_from_matrix(mat1)
            qb = quaternion_from_matrix(mat2)
            m = quaternion_matrix(quaternion_slerp(qa, qb, float(t - t_b)/(t_a - t_b)))
            m[0, -1] = np.interp(t, [t_b, t_a], [mat1[0, -1], mat2[0, -1]])
            m[1, -1] = np.interp(t, [t_b, t_a], [mat1[1, -1], mat2[1, -1]])
            m[2, -1] = np.interp(t, [t_b, t_a], [mat1[2, -1], mat2[2, -1]])
        else:
            m = matrices[t]
        all_matrices[t] = m
    return all_matrices

def applyTrsf_full(mats, p, t):
    ''' Provided a dictionary of matrices, a position and a time,
        applies the appropriate transformation to the position *p*
        Args:
            mats: {int: 4x4 array, }, dictionary where the key is time
                and the value is the associated transformation for that time
            p: [float, float, float], x, y, z coordinate in the Cartesian referential
            t: int, time to which p belongs to
        Returns:
            new_p: [float, float, float], the new position of the original point *p*
    '''
    if not t in mats:
        t = mats.keys()[np.argmin(np.abs(t - np.array(mats.keys())))]
    new_p = applyMatrixTrsf(p, mats[t])
    return new_p

def applyTrsf_time_laps(LT, timeTrsf, mats):
    ''' Apply the appropriate time and rigid Transformations
        to the points in a lineageTree provided the time and rigid transformations.
        If necessary, time points are added/removed.
        The nodes are stored in LT.new_time_nodes for there right times,
        The transformed positions are stored in LT.new_pos
        Args:
            LT: lineageTree
            timeTrsf: time transformation function
            mats: {int: 4x4 array, }, dictionary where the key is time
                and the value is the associated transformation for that time
    '''
    LT.new_time_nodes = {}
    LT.new_pos = {}
    t_total = []
    t_to_fill = []
    time_evol = []
    if not hasattr(LT, 'to_take_time'):
        LT.to_take_time = LT.time_nodes

    time_mapping = np.round(timeTrsf(sorted(LT.time_nodes.keys()))).astype(np.int)
    times_to_fuse = time_mapping[1:][(time_mapping[1:] == time_mapping[:-1])]
    fill = zip(time_mapping[:-1][1 < (time_mapping[1:] - time_mapping[:-1])],
               time_mapping[1:][1 < (time_mapping[1:] - time_mapping[:-1])])

    cells_removed = set()
    treated = set()
    for t in sorted(LT.time_nodes.keys()):
        if not t in treated:
            treated.add(t)
            new_t = int(np.round(timeTrsf(t)))
            LT.new_time_nodes[new_t] = []
            if not new_t in times_to_fuse:
                for c in LT.to_take_time[t]:
                    LT.new_pos[c] = np.array(applyTrsf_full(mats, LT.pos[c], t))
                    LT.new_time_nodes[new_t].append(c)
            else:
                ti = t
                t_to_aggregate = []
                while int(np.round(timeTrsf(ti))) == new_t:
                    t_to_aggregate += [ti]
                    treated.add(ti)
                    ti += 1
                for ci in LT.to_take_time[t_to_aggregate[0]]:
                    track = [ci]
                    while LT.successor.get(track[-1], []) != [] and LT.time[track[-1]] < t_to_aggregate[-1]:
                        track += LT.successor[track[-1]]
                    glob_pred = -1
                    glob_pred = -1
                    if (LT.time.get(LT.successor.get(track[-1], [-1])[0], -1) == t_to_aggregate[-1] + 1 and
                        LT.predecessor.get(track[0], [-1])[0] != -1):
                        all_pos = [applyTrsf_full(mats, LT.pos[cii], LT.time[cii]) for cii in track]
                        avg_p = np.mean(all_pos, axis = 0)
                        glob_pred = LT.predecessor[track[0]][0]
                        glob_succ = LT.successor[track[-1]][0]
                    for cii in track:
                        cells_removed.add(cii)
                        cii_succ = LT.successor.pop(cii, [])
                        for succ in cii_succ:
                            if cii in LT.predecessor.get(succ, []):
                                LT.predecessor[succ].remove(cii)
                            if LT.predecessor.get(succ, []) == []:
                                LT.predecessor.pop(succ)
                        cii_pred = LT.predecessor.pop(cii, [])
                        for pred in cii_pred:
                            if cii in LT.successor.get(pred, []):
                                LT.successor[pred].remove(cii)
                            if LT.successor.get(pred, []) == []:
                                LT.successor.pop(pred)
                    if glob_pred != -1 and glob_succ != -1:
                        C_id = LT.get_next_id()
                        LT.successor[pred] = [C_id]
                        LT.predecessor[C_id] = [glob_pred]
                        LT.successor[C_id] = [glob_succ]
                        LT.predecessor[succ] = [C_id]
                        LT.edges.append((C_id, glob_succ))
                        LT.edges.append((glob_pred, C_id))
                        LT.nodes.append(C_id)
                        LT.new_time_nodes[new_t] += [C_id]
                        LT.new_pos[C_id] = avg_p
    for tb, te in fill:
        for t_inter in range(tb + 1, te):
            LT.new_time_nodes[t_inter] = []
        for ci in LT.new_time_nodes[tb]:
            next_C = LT.successor.get(ci, [-1])[0]
            if next_C != -1:
                p_init = LT.new_pos[ci]
                p_final = LT.new_pos[LT.successor[ci][0]]
                vals = np.zeros((2, 4))
                vals[0, :-1] = p_init
                vals[0, -1] = tb
                vals[1, :-1] = p_final
                vals[1, -1] = te
                pos_interp = interp_3d_coord(vals)
                c_pred = ci
                for t_inter in range(tb + 1, te):
                    C_id = LT.get_next_id()
                    LT.nodes += [C_id]
                    LT.new_time_nodes[t_inter] += [C_id]
                    LT.successor[c_pred] = [C_id]
                    LT.predecessor[C_id] = [c_pred]
                    LT.edges.append((c_pred, C_id))
                    c_pred = C_id
                    LT.new_pos[C_id] = pos_interp(t_inter)
                LT.successor[c_pred] = [next_C]
                LT.predecessor[next_C] = [c_pred]
                LT.edges.append((c_pred, C_id))
                # LT.edges.remove((ci, next_C))
    LT.nodes = list(set(LT.nodes).difference(cells_removed))


def get_points_from_ray(points, A, B, r = 5):
    ''' Build the projections of points on a line AB that are at a distance
        maximum of *r*. It return three lists.
        Args:
            points: [[float, float, float], ], list of 3D positions
            A: [float, float, float], 3D position, first point of the line
            B: [float, float, float], 3D position, second point of the line
            r: float, distance maximum between the line and the points
        Returns:
            [[float, float, float], ], list for position of the selected points after projection
            [float, ], list of distances of the selected points to *A*
            [[float, float, float], ], list for position of the selected points before projection
    '''
    AB = B - A
    Ap = points - A
    n_Aproj = (np.dot(Ap, AB)/np.linalg.norm(AB)).reshape(len(Ap), 1)
    proj = (A + n_Aproj * AB/np.linalg.norm(AB))
    distances = np.linalg.norm(points - proj, axis = 1)
    distr_on_AB = n_Aproj.reshape(len(Ap))
    return (proj[(distances<r) & (distr_on_AB >= 0)],
            distr_on_AB[(distances<r) & (distr_on_AB >= 0)],
            points[(distances<r) & (distr_on_AB >= 0)])

N_func = lambda x, mu, sigma: np.exp(-(x - mu)**2/(2*sigma**2)) / (np.sqrt(2*np.pi*sigma**2))
def get_low_and_high(points, A, B, r = 150, percentile = 80):
    ''' Retrieves the high and low points along a ray casted from a point A to a point B
        Args:
            points: [[float, float, float], ], list of 3D positions
            A: [float, float, float], 3D position, first point of the line
            B: [float, float, float], 3D position, second point of the line
            r: float, distance maximum between the line and the points
            percentile: float, percentile of the distribution kept, has to be between 0 and 100.
    '''
    kept_points, distribution_on_AB, kept_full_points = get_points_from_ray(points, A, B, 150)
    m, s = np.mean(distribution_on_AB), np.std(distribution_on_AB)
    mask = ((m - 5 * s) < distribution_on_AB) & (distribution_on_AB < (m + 5 * s))
    distribution_on_AB = distribution_on_AB[mask]
    kept_points = kept_points[mask]
    if len(distribution_on_AB)>10:
        th_v_m = np.percentile(distribution_on_AB, percentile)
        th_v_p = np.percentile(distribution_on_AB, 100-percentile)

        low_p = kept_points[np.argmin(np.abs(distribution_on_AB - th_v_m))]
        high_p = kept_points[np.argmin(np.abs(distribution_on_AB - th_v_p))]
        return low_p, high_p, distribution_on_AB, kept_points, kept_full_points
    else:
        return None, None,None, None, None

percentile = 85
def build_points(t, A):
    ''' Builds the inner and outer shell of the two lineage trees LT_1 and LT_2
        for time point *t*t from rays casted from the center of mass *A*.
        The points are written in the files:
        pts01_high_DS_%03d.txt, pts01_low_DS_%03d.txt, pts02_high_DS_%03d.txt, pts02_low_DS_%03d.txt
        Args:
            t: int, time point to treat
            A: [float, float, float], 3D coordinates of the center of mass
    '''
    tic = time()
    A = np.array(A)
    points_2 = np.array([LT_2.new_pos[c] for c in LT_2.new_time_nodes[t]])
    points_1 = np.array([LT_1.pos[c] for c in LT_1.time_nodes[t]])
    low_points_2 = []
    high_points_2 = []
    low_points_1 = []
    high_points_1 = []
    tmp_pts = []
    tmp_dist = []
    nb_p_for_phi = lambda phi: np.round(np.interp(phi, [0, np.pi/2], [1, 50])).astype(np.int)
    for phi in np.linspace(0, np.pi/2, 50):
        for theta in np.linspace(0, np.pi/2, nb_p_for_phi(phi)):
            Z = 500*np.cos(theta)*np.sin(phi)
            Y = 500*np.sin(theta)*np.sin(phi)
            X = 500*np.cos(phi)
            sphere_points = [[ X,  Y,  Z], [ X,  Y, -Z], [ X, -Y,  Z], [ X, -Y, -Z]]
            for B in sphere_points:
                tmp_pts += [A - B]
                low_p_2, high_p_2, distribution_on_AB_2, kept_points_2, kept_full_points_2 = get_low_and_high(points_2, A, A - B, 150, percentile)
                low_p_1, high_p_1, distribution_on_AB_1, kept_points_1, kept_full_points_1 = get_low_and_high(points_1, A, A - B, 150, percentile)
                if (not low_p_1 is None) and (not low_p_2 is None):
                    tmp_dist += [[distribution_on_AB_2, kept_points_2, kept_full_points_2]]
                    low_points_2 += [low_p_2]
                    high_points_2 += [high_p_2]
                    low_points_1 += [low_p_1]
                    high_points_1 += [high_p_1]
                low_p_2, high_p_2, distribution_on_AB_2, kept_points_2, kept_full_points_2 = get_low_and_high(points_2, A, A + B, 150, percentile)
                low_p_1, high_p_1, distribution_on_AB_1, kept_points_1, kept_full_points_1 = get_low_and_high(points_1, A, A + B, 150, percentile)
                if (not low_p_1 is None) and (not low_p_2 is None):
                    low_points_2.append(low_p_2)
                    high_points_2.append(high_p_2)
                    low_points_1.append(low_p_1)
                    high_points_1.append(high_p_1)

    np.savetxt(match_points_folder + 'pts01_high_DS_%03d.txt'%t, np.array(high_points_1)/DS_value)
    np.savetxt(match_points_folder + 'pts01_low_DS_%03d.txt'%t, np.array(low_points_1)/DS_value)
    np.savetxt(match_points_folder + 'pts02_high_DS_%03d.txt'%t, np.array(high_points_2)/DS_value)
    np.savetxt(match_points_folder + 'pts02_low_DS_%03d.txt'%t, np.array(low_points_2)/DS_value)
    print 't%03d: %.2f'%(t, time() - tic)

def interp_3d_coord(vals, ext = 3):
    ''' Does a piecewise lineare interpolation between 3D points
        Args:
            vals: [[float, float, float, int], ],
                list of 3D values plus there associatedf time
    '''
    if len(vals) > 1:
        vals = vals[np.argsort(vals[:,-1])]
        X_som_front = sp.interpolate.InterpolatedUnivariateSpline(vals[:, -1], vals[:, 0], k=1, ext = ext)
        Y_som_front = sp.interpolate.InterpolatedUnivariateSpline(vals[:, -1], vals[:, 1], k=1, ext = ext)
        Z_som_front = sp.interpolate.InterpolatedUnivariateSpline(vals[:, -1], vals[:, 2], k=1, ext = ext)

        return lambda t: np.array([X_som_front(t), Y_som_front(t), Z_som_front(t)])
    elif ext == 3:
        return lambda t: deepcopy(vals[0][:-1])
    else:
        return lambda t: deepcopy(vals[0][:-1]) if t == vals[0][-1] else np.array([0., 0., 0.])

def get_somite_pos(coord):
    ''' Extract the average position between left and right somites
        Args:
            coord: {string: [float, float, float], }, dictionary that maps the name
                of a landmark to its position
    '''
    s_end_pairing = {}
    for name, p in coord.iteritems():
        if 'LS' in name:
            name_R = name.replace('L', 'R')
            if name_R in coord:
                s_end_pairing.setdefault(int(name[2]), []).append([p, coord[name_R]])
    s_end_pairing = {t:np.mean(v, axis = 1) for t, v in s_end_pairing.iteritems()}
    final_somites = {}
    for S_id in sorted(s_end_pairing):
        final_somites[S_id] = {}
        for S in s_end_pairing[S_id]:
            final_somites[S_id][np.round(S[-1])] = S[:-1]
    return final_somites


def get_somite_front(coord):
    ''' Extract the average position between left and right
        somites at the first time they appear
        Args:
            coord: {string: [float, float, float], }, dictionary that maps the name
                of a landmark to its position
    '''
    s1_paring = []
    for name, p in coord.iteritems():
        if 'LS1.' in name:
            name_R = name.replace('L', 'R')
            if name_R in coord:
                s1_paring.append((p, coord[name_R]))

    somite_front = np.mean(s1_paring, axis = 1)
    return interp_3d_coord(somite_front, ext = 1), {k[-1]: k[:-1] for k in somite_front}

def get_somite_pairing(coord):
    ''' Extract the average position between left and right somites
        Args:
            coord: {string: [float, float, float], }, dictionary that maps the name
                of a landmark to its position
    '''
    s_end_pairing = {}
    for name, p in coord.iteritems():
        if 'LS' in name:
            name_R = name.replace('L', 'R')
            if name_R in coord:
                s_end_pairing.setdefault(int(name[2]), []).append([p, coord[name_R]])
    s_end_pairing_mean = {t:np.mean(v, axis = 1) for t, v in s_end_pairing.iteritems()}
    return (np.array([ki[:-1] for k in s_end_pairing_mean.values() for ki in k]),
            sum(s_end_pairing.values(), []))

def get_somite_end(coord):
    ''' Extract the average position between left and right somites
        For the last time the somites have been marked
        Args:
            coord: {string: [float, float, float], }, dictionary that maps the name
                of a landmark to its position
    '''
    s_end_pairing = {}
    for name, p in coord.iteritems():
        if 'LS' in name:
            name_R = name.replace('L', 'R')
            if name_R in coord:
                s_end_pairing.setdefault(int(name[2]), []).append([p, coord[name_R]])
    s_end_pairing = {t:np.mean(v, axis = 1) for t, v in s_end_pairing.iteritems()}
    final = {}
    s_start = {np.int(np.round(np.min(v[:,-1]))):t for t, v in s_end_pairing.iteritems()}
    for s, vals in s_end_pairing.iteritems():
        final[s] = interp_3d_coord(vals)
    choosed_somite = {}
    times = sorted(s_start)
    end_time = times[-1]
    for start in times:
        for t in range(start, end_time + 1):
            choosed_somite.setdefault(t, []).append(s_start[start])

    return final, choosed_somite

def get_somite_interp(coord, somite_to_look):
    s_end_pairing = []
    for name, p in coord.iteritems():
        if 'LS' in name:
            name_R = name.replace('L', 'R')
            if name_R in coord:
                if somite_to_look[np.round((p[-1] + coord[name_R][-1])/2.)] == int(name[2]):
                    s_end_pairing += [np.mean([p, coord[name_R]], axis = 0)]

    return interp_3d_coord(np.array(s_end_pairing), ext = 1), {k[-1]: k[:-1] for k in s_end_pairing}

def get_common_somites_end(coord_1, coord_2_trsf):
    ''' Extract the average position between left and right somites
        Args:
            coord: {string: [float, float, float], }, dictionary that maps the name
                of a landmark to its position
    '''
    somite_end_1, somite_present_1 = get_somite_end(coord_1)
    somite_end_2, somite_present_2 = get_somite_end(coord_2_trsf)

    somite_to_look = {}
    for t in set(somite_present_1.keys() + somite_present_2.keys()):
        somite_to_look[t] = min(max(somite_present_1.get(t, [somite_to_look.get(t-1, 2)])),
                                max(somite_present_2.get(t, [somite_to_look.get(t-1, 2)])))
    
    end_s_interp_1 = get_somite_interp(coord_1, somite_to_look)
    end_s_interp_2 = get_somite_interp(coord_2_trsf, somite_to_look)

    return end_s_interp_1, end_s_interp_2


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    if (v1 == 0).all() or (v2 == 0).all():
        return 0
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def oriented_angle_between(v1, v2, n):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    if (v1 == 0).all() or (v2 == 0).all():
        return 0
    else:
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        A = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        if np.dot(n, np.cross(v1_u, v2_u)) > 0:
            return -A
        else:
            return A
        
def get_angles(params):
    ''' Calculates the angles between the equivalent somites
        between the two embryos
    '''
    t, origin = params
    try:
        start_time = time()
        b, n = symmetry_plane[t]
        angles = []
        to_consider = []
        proj_ori = projection_onto_P(origin, b, n)[0] - b
        vE_2 = projection_onto_P(E_2(t), b, n)[0] - b
        vB_2 = projection_onto_P(B_2(t), b, n)[0] - b
        for i, L_2 in enumerate(LM_2):
            vP_2 = projection_onto_P(L_2(t), b, n)[0] - b
            a = oriented_angle_between(proj_ori, vP_2, n)
            if a < 0:
                a = 2 * np.pi - np.abs(a)
            if (np.linalg.norm(vP_2 - vE_2)>1 and 
                (angles==[] or (angles!=[] and np.round(a, 3) != angles[-1]))):
                to_consider += [i]
            angles += [np.round(a, 3)]
        angles = np.array(angles)[to_consider]
        tmp_LM_2 = [np.array(LM_2)[to_consider][i] for i in np.argsort(angles)]
        tmp_LM_1 = [np.array(LM_1)[to_consider][i] for i in np.argsort(angles)]
        angles = np.sort(angles)
        theta_disc = []
        for L_1, L_2 in zip(tmp_LM_1, tmp_LM_2):
            vP_1 = projection_onto_P(L_1(t), b, n)[0] - b
            vP_2 = projection_onto_P(L_2(t), b, n)[0] - b
            theta_disc += [oriented_angle_between(vP_1, vP_2, n)]
        theta_disc = np.array(theta_disc)
        return t, angles, theta_disc, to_consider, proj_ori
    except Exception as e:
        print e, t
        return(e, t)

def apply_rot(params):
    ''' Applies the piecewise rotation to the floating embryo
    '''
    t, angles, theta_disc, proj_ori, to_consider = params
    try:
        new_pos = {}
        theta = sp.interpolate.InterpolatedUnivariateSpline(angles,
                                                            theta_disc, k=1, ext = 3)
        b, n = symmetry_plane[t]
        start_time = time()
        new_embryo = []
        previous_embryo = []
        new_embryo_angle = {}
        old_embryo_angle = {}
        for c in LT_for_multiprocess.new_time_nodes[t]:
            pos = np.array(LT_for_multiprocess.new_pos[c])
            previous_embryo += [pos]
            proj_pos = projection_onto_P(pos, b, n)[0] - b
            A_to_ORI = oriented_angle_between(proj_ori, proj_pos, n)
            if A_to_ORI < 0:
                A_to_ORI = 2 * np.pi - np.abs(A_to_ORI)
            theta_p = theta(A_to_ORI)
            T_v = rotation(pos - b, n, theta_p) + b
            new_pos[c] = T_v
            new_embryo += [T_v]
            new_embryo_angle[tuple(T_v)] = theta_p
            old_embryo_angle[tuple(pos)] = A_to_ORI

        tmp_LM_2 = [np.array(LM_2)[to_consider][i] for i in np.argsort(angles)]
        tmp_LM_1 = [np.array(LM_1)[to_consider][i] for i in np.argsort(angles)]
        LM_TRSF = []
        for L in tmp_LM_2:
            pos = L(t)
            proj_pos = projection_onto_P(pos, b, n)[0] - b
            A_to_ORI = oriented_angle_between(proj_ori, proj_pos, n)
            if A_to_ORI < 0:
                A_to_ORI = 2 * np.pi - np.abs(A_to_ORI)
            theta_p = theta(A_to_ORI)
            T_v = rotation(pos - b, n, theta_p) + b
            LM_TRSF += [T_v]
        print 't %03d done in %.3f seconds'%(t, time() - start_time)
        return new_pos
    except Exception as e:
        print e, t
        return(e, t)

def dict_cmp(d1, d2):
    ''' Function to order two dictionaries
    '''
    return int(np.round(min(d1[0]) - min(d2[0])))

def extrapolate_pos(all_dicts, decay_time = 10):
    ''' Inter/extrapolate in time the position of the landmarks
        Args:
            all_dicts: {string: [float, float, float], }, dictionary that maps the name
                of a landmark to its position
    '''
    for tmp in sorted(all_dicts, cmp = dict_cmp):
        D, F = tmp
        t_start = min(D)
        if min(LT_2.new_time_nodes) < t_start:
            closest = np.inf
            for di, fi in all_dicts:
                times = [t for t in di.keys() if t <= t_start - decay_time]
                if times != []:
                    closest_t = max(times)
                    if np.linalg.norm(di[closest_t] - D[t_start]) < closest:
                        closest_dict = di
                        closest_f = fi
            D[t_start - decay_time] = closest_f(t_start - decay_time)
            for t, p in closest_dict.iteritems():
                if t < t_start - decay_time:
                    D[t] = p
        tmp[-1] = interp_3d_coord(np.array([np.array(list(v)+[k]) for k, v in D.iteritems()]))

    final_dists = []
    for D, F in all_dicts:
        final_dists += [interp_3d_coord(np.array([np.array(list(v)+[k]) for k, v in D.iteritems()]))]
    return final_dists

def compute_plane(p1, p2, p3):
    ''' Compute the parameters a, b, c, d, of a plane equation:
        ax + by + cz + d = 0 from 3 non-aligned points
        Args:
            p1, p2, p3: [float, float, float], 3D coordinates
        Returns:
            a, b, c, d: float, parameters of the equation of the plan
    '''
    v1 = p1 - p2
    v2 = p1 - p3
    cross_prod = np.cross(v1, v2)
    a, b, c = cross_prod
    d = np.sum([a, b, c] * -p1)
    return np.array([a, b, c, d])

def get_barycenter(fname):
    ''' Reads and coes a linear piecewise interpolation/extrapolation barycenters
        Args:
            fname: string, name of the barycenter file (each line as 'x, y, z, t')
        Returns:
            barycenters_interp: {int:[float, float, float], }, dictionary mapping
                        a time point to the interpolated barycenter at that time
            barycenters: {int:[float, float, float], }, dictionary mapping
                        a time point to the barycenter for each time in fname
    '''
    f = open(fname)
    lines = f.readlines()
    f.close()
    barycenters = {}
    for l in lines[1:]:
        split_l = l.split(',')
        barycenters[int(split_l[-1])] = tuple(float(v) for v in split_l[:-1])
    times = sorted(barycenters)
    Xb, Yb, Zb = np.array([barycenters[t] for t in times]).T
    Xb_f = interpolate.InterpolatedUnivariateSpline(times, Xb, k=1)
    Yb_f = interpolate.InterpolatedUnivariateSpline(times, Yb, k=1)
    Zb_f = interpolate.InterpolatedUnivariateSpline(times, Zb, k=1)
    Ti = np.arange(11, 522)
    barycenters_interp = dict(zip(Ti, zip(Xb_f(Ti), Yb_f(Ti), Zb_f(Ti))))

    return barycenters_interp, barycenters

def build_gg(data):
    ''' Build a Gabriel graph from a list of 3D points.
        In the resulting Gabriel graph, the id of the nodes correspond
        to the position of points in *data*
        Args:
            data: Nx3 array of floats, array of 3D coordinates
        Returns:
            data: Nx3 array of floats, array of 3D coordinates (same as input)
            GG_out: {int: [int, ], }, dictionary that maps object id to the 
                list of the ids of its neighbors
            idx3d: kdtree structure
    '''
    D_graph = Delaunay(data, incremental = True)
    idx3d = kdtree.KDTree(data)
    delaunay_graph = {}
    for N in D_graph.simplices:
        for e1, e2 in combinations(np.sort(N), 2):
            delaunay_graph.setdefault(e1, set([])).add(e2)
            delaunay_graph.setdefault(e2, set([])).add(e1)

    tmp = time()
    GG_out = {}
    for e1, neighbs in delaunay_graph.iteritems():
        for ni in neighbs:
            if not any([np.linalg.norm((data[ni] + data[e1])/2 - data[i])<np.linalg.norm(data[ni] - data[e1])/2
                    for i in delaunay_graph[e1].intersection(delaunay_graph[ni])]):
                GG_out.setdefault(e1, set()).add(ni)
                GG_out.setdefault(ni, set()).add(e1)
    return data, GG_out, idx3d

def get_V_W(P, GG, data, points1, pos):
    ''' From a set of 3D points, computes the set of vector+weights associated to
        including the vectors of the neighbors in the Gabriel graph
        Args:
            P; Nx3 array of floats, list of 3D points
            GG: {int: [int, ], }, dictionary that maps object id to the 
                list of the ids of its neighbors
            data: Mx3 array of floats, array of 3D points
            points1: Kx3 array of floats, array of 3D points
            pos: [float, float, float], 3D coordinate
        Returns:
            vector: Lx3 array (as a list), list of 3D vectors
            weights: Lx1 array (as a list), list of weights
    '''
    N = set()
    for pi in P:
        N.add(pi)
        for k in GG.get(pi, []):
            N.add(k)
    N = list(N)
    if N != []:
        weights = 1./np.linalg.norm(data[N] - pos, axis = 1)
        vector = points1[N] - data[N]
    return list(vector), list(weights)

def apply_trsf(t, k = 2):
    ''' Computes and apply a non-linear transformation between two embryos
        provided that the embryo shells Have been already computed.
        Args:
            t: int, time
            k: int, number of neighbors to consider (default 2)
    '''
    tic = time()
    points2_NL_1 = np.loadtxt(match_points_folder + 'pts02_low_DS_%03d.txt'%(t)) * DS_value
    points2_NL_2 = np.loadtxt(match_points_folder + 'pts02_high_DS_%03d.txt'%(t)) * DS_value

    points1_NL_1 = np.loadtxt(match_points_folder + 'pts01_low_DS_%03d.txt'%(t)) * DS_value
    points1_NL_2 = np.loadtxt(match_points_folder + 'pts01_high_DS_%03d.txt'%(t)) * DS_value

    data_1, GG_1, idx3d_1 = build_gg(points2_NL_1)
    data_2, GG_2, idx3d_2 = build_gg(points2_NL_2)
    neighbs = {}
    count = 0
    cells = VF_2.new_time_nodes[t]
    closest_1 = idx3d_1.query([VF_2.new_pos[c] for c in cells], k)
    closest_2 = idx3d_2.query([VF_2.new_pos[c] for c in cells], k)
    final_pos = {}

    for i, (D, P) in enumerate(zip(*closest_1)):
        pos = VF_2.new_pos[cells[i]]
        V_1, W_1 = get_V_W(P, GG_1, data_1, points1_NL_1, pos)
        V_2, W_2 = get_V_W(closest_2[1][i], GG_2, data_2, points1_NL_2, pos)

        V = np.array(V_1 + V_2)
        W = W_1 + W_2

        vector = np.sum(V * zip(W, W, W), axis = 0) / np.sum(W)

        final_pos[cells[i]] = pos + vector

    print 't%03d: %.2f'%(t, time() - tic)
    return final_pos


def read_param_file():
    ''' Asks for, reads and formats the parameter file
    '''
    p_param = raw_input('Please enter the path to the parameter file:\n')
    p_param = p_param.replace('"', '')
    p_param = p_param.replace("'", '')
    p_param = p_param.replace(" ", '')
    if p_param[-4:] == '.csv':
        f_names = [p_param]
    else:
        f_names = [os.path.join(p_param, f) for f in os.listdir(p_param) if '.csv' in f and not '~' in f]
    for file_name in f_names:
        f = open(file_name)
        lines = f.readlines()
        f.close()
        param_dict = {}
        i = 0
        nb_lines = len(lines)
        while i < nb_lines:
            l = lines[i]
            split_line = l.split(',')
            param_name = split_line[0]
            if param_name in ['time_match_manual']:
                name = param_name
                out = []
                while (name == param_name or param_name == '') and  i < nb_lines:
                    out += [[int(split_line[1]), int(split_line[2])]]
                    i += 1
                    if i < nb_lines:
                        l = lines[i]
                        split_line = l.split(',')
                        param_name = split_line[0]
                param_dict[name] = np.array(out)
            else:
                param_dict[param_name] = split_line[1].strip()
                i += 1
            if param_name == 'time':
                param_dict[param_name] = int(split_line[1])
        path_1 = param_dict['path_ref']
        csv_file_name_1 = param_dict['annotation_ref']
        TARDIS_folder_1 = param_dict['TARDIS_folder_ref']
        manual_annotation_1 = param_dict['annotation_ref_2']
        path_barycenter = param_dict['path_barycenter']
        path_2 = param_dict['path_flo']
        csv_file_name_2 = param_dict['annotation_flo']
        manual_annotation_2 = param_dict['annotation_flo_2']
        TARDIS_folder_2 = param_dict['TARDIS_folder_flo']
        match_points_folder = param_dict['match_points_folder']
        time_match_manual = param_dict['time_match_manual']
    return (path_1, csv_file_name_1, TARDIS_folder_1,
            manual_annotation_1, path_barycenter, path_2,
            csv_file_name_2, manual_annotation_2,
            TARDIS_folder_2, match_points_folder, time_match_manual)

if __name__ == '__main__':
    (path_1, csv_file_name_1, TARDIS_folder_1,
            manual_annotation_1, path_barycenter, path_2,
            csv_file_name_2, manual_annotation_2,
            TARDIS_folder_2, match_points_folder, time_match_manual) = read_param_file()

    ### Preparation of the necessary folders
    if not os.path.exists(TARDIS_folder_1):
        os.makedirs(TARDIS_folder_1)
    if not os.path.exists(TARDIS_folder_2):
        os.makedirs(TARDIS_folder_2)
    if not os.path.exists(match_points_folder):
        os.makedirs(match_points_folder)
    if not os.path.exists(TARDIS_folder_2 + 'SVF_TARDIS_filtered/'):
        os.makedirs(TARDIS_folder_2 + 'SVF_TARDIS_filtered/')
    if not os.path.exists(TARDIS_folder_2 + 'SVF_TARDIS'):
        os.makedirs(TARDIS_folder_2 + 'SVF_TARDIS')


    ### Reading of the different lineage trees
    VF_1 = lineageTree(path_1 + 'SVF.bin')
    LT_1 = lineageTree(path_1 + 'TGMM_clean.bin')

    VF_2 = lineageTree(path_2 + 'SVF.bin')
    LT_2 = lineageTree(path_2 + 'TGMM_clean.bin')

    ### Reading the landmarks
    f = pd.read_csv(TARDIS_folder_1 + csv_file_name_1)
    coord_1 = f.dropna().as_matrix()[:,1:].astype(float)
    coord_1 = dict(zip(f.dropna().as_matrix()[:,0], coord_1))

    f = pd.read_csv(TARDIS_folder_2 + csv_file_name_2)
    coord_2 = f.dropna().as_matrix()[:,1:].astype(float)
    coord_2 = dict(zip(f.dropna().as_matrix()[:,0], coord_2))

    corres = {}
    import re
    for p_name, pos in coord_2.iteritems():
        if re.search('([RL]S[0-9].[0-9])|(^NS$)|(^INTCS$)|(^AIP$)', p_name) and p_name in coord_1:
            corres[p_name] = [coord_1[p_name], pos]

    p1, p2 = zip(*corres.values())
    times_2 = np.array(p2)[:, -1]
    times_1 = np.array(p1)[:, -1][np.argsort(times_2)]
    times_2 = np.sort(times_2)
    t_corres = {}
    for t2, t1 in zip(times_2, times_1):
        t_corres.setdefault(t2, []).append(t1)
    t_corres = [(k, np.mean(v)) for k, v in t_corres.iteritems()]
    times_2, times_1 = zip(*sorted(t_corres))

    X_1 = ([pos[0][0] for pos in corres.values()])
    X_2 = ([pos[1][0] for pos in corres.values()])

    Y_1 = ([pos[0][1] for pos in corres.values()])
    Y_2 = ([pos[1][1] for pos in corres.values()])

    Z_1 = ([pos[0][2] for pos in corres.values()])
    Z_2 = ([pos[1][2] for pos in corres.values()])

    ### Builds the first rigid transformation
    mat = rigid_transform_3D(np.array(p2)[:, :-1], np.array(p1)[:, :-1])
    timeTrsf = sp.interpolate.InterpolatedUnivariateSpline(times_2, times_1, k=1)#lambda x, t1, t2: x + np.mean(t1 - t2)
    applyMatrixTrsf = lambda x, mat: list(np.dot(mat, np.transpose([list(x) + [1]])).reshape(4)[:3])
    trsf4d = lambda x, mat, t1, t2: applyMatrixTrsf(list(x[:-1]) + [timeTrsf(x[-1], t1, t2)], mat)

    ### Builds the second rigid transformation
    somite_end_1, somite_present_1 = get_somite_end(coord_1)
    somite_end_2, somite_present_2 = get_somite_end(coord_2)

    somite_apparition_1 = {}
    for t, s in somite_present_1.iteritems():
        for si in s:
            somite_apparition_1[si] = min(somite_apparition_1.get(si, np.inf), t)

    somite_apparition_2 = {}
    for t, s in somite_present_2.iteritems():
        for si in s:
            somite_apparition_2[si] = min(somite_apparition_2.get(si, np.inf), t)

    time_match_somites = []
    for s in sorted(set(somite_apparition_1).intersection(set(somite_apparition_2))):
        time_match_somites += [[somite_apparition_1[s], somite_apparition_2[s]]]

    time_match = []
    for v1, v2 in time_match_manual:
        if v1 < time_match_somites[0][0] and v2 < time_match_somites[0][1]:
            time_match += [[v1, v2]]

    time_match = np.array(time_match + time_match_somites)

    timeTrsf = sp.interpolate.InterpolatedUnivariateSpline(time_match[:,1], time_match[:,0], k=1)
    manual_ann_1 = lineageTree(TARDIS_folder_1 + manual_annotation_1, MaMuT = True)
    manual_ann_2 = lineageTree(TARDIS_folder_2 + manual_annotation_2, MaMuT = True)

    times_in_1 = []
    names = []
    for t, v in manual_ann_1.time_nodes.iteritems():
        if 4 < np.sum([not re.search('^ID.', manual_ann_1.node_name[c]) for c in v]):
            times_in_1 += [t]
            names += [manual_ann_1.node_name[c]]
    times_in_1 = np.array(times_in_1)

    times_in_2 = []#np.array(manual_ann_2.time_nodes.keys())
    for t, v in manual_ann_2.time_nodes.iteritems():
        if 4 < np.sum([not re.search('^ID.', manual_ann_2.node_name[c]) for c in v]):
            times_in_2 += [t]
    times_in_2 = np.array(times_in_2)

    common_labels = {}
    tmp_dict_t1 = {}
    time_distances = []
    for t2 in times_in_2:
        common_labels[t2] = {}
        tmp_dict_t1[t2] = {}
        if (np.abs(times_in_1 - timeTrsf(t2))<3).any():
            possible_times = list(times_in_1[np.abs(times_in_1 - timeTrsf(t2))<3])
        else:
            possible_times = [times_in_1[np.argmin(np.abs(times_in_1 - timeTrsf(t2)))]]
        time_distances += [np.min(np.abs(times_in_1 - timeTrsf(t2)))]
        for t1 in possible_times:
            for n in manual_ann_1.time_nodes[t1]:
                name = manual_ann_1.node_name[n]
                if not re.search('^ID.', name):
                    if re.search('.[0-9]', name):
                        tmp_n = name[:re.search('.[0-9]', name).start()+1]
                    else:
                        tmp_n = name
                    if tmp_n[-1] in 'LR':
                        tmp_n = tmp_n[:-1]
                    tmp_dict_t1[t2].setdefault(tmp_n, []).append(n)

        t0 = []
        t0_full = []
        for n in manual_ann_2.time_nodes[t2]:
            name = manual_ann_2.node_name[n]
            if not re.search('^ID.', name):
                    if re.search('.[0-9]', name) :
                        tmp_n = name[:re.search('.[0-9]', name).start()+1]
                    else:
                        tmp_n = name
                    if tmp_n[-1] in 'LR':
                        tmp_n = tmp_n[:-1]
                    if tmp_n in tmp_dict_t1[t2]:
                        common_labels[t2].setdefault(tmp_n, []).append(n)

    final_mapping = {t: {g_n: [tmp_dict_t1[t][g_n], nodes] for g_n, nodes in mapping_t.iteritems()} for t, mapping_t in common_labels.iteritems()}

    mat_per_t = {}
    for t, annotations in final_mapping.iteritems():
        if annotations != {}:
            points1 = []
            points2 = []
            for annotation_name, (cells1, cells2) in annotations.iteritems():
                a_ = np.array([manual_ann_1.pos[c] for c in cells1])
                b_ = np.array([applyMatrixTrsf(manual_ann_2.pos[c], mat) for c in cells2])
                b_post_trsf = np.array([manual_ann_2.pos[c] for c in cells2])
                cost_matrix = spatial.distance_matrix(a_, b_)
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                points1 += list(a_[row_ind])
                points2 += list(b_post_trsf[col_ind])
            mat_per_t[t] = rigid_transform_3D(points2[:-1], points1[:-1])

    mats = interpolate_transformations(mat_per_t)

    ### Apply the rigid transformations
    applyTrsf_time_laps(VF_2, timeTrsf, mats)
    applyTrsf_time_laps(LT_2, timeTrsf, mats)

    VF_2_rigid = VF_2.new_pos

    ### Computes the piecewise rotations
    ###### Preparation of the landmarks
    pos_all_1 = np.array(LT_1.pos.values())
    max_x = np.max(pos_all_1, axis = 0)[0]
    new_time_nodes_BU = deepcopy(VF_2.new_time_nodes)
    to_take_time_new = {}
    for t, cells in VF_2.new_time_nodes.iteritems():
        to_take_time_new[t] = []
        for c in cells:
            if VF_2.new_pos[c][0] < max_x:
                to_take_time_new[t] += [c]
    VF_2.new_time_nodes = to_take_time_new

    DS_value = 10.
    VF_dimension = np.max(LT_2.new_pos.values() + LT_1.pos.values(), axis = 0)/DS_value + 50
    VF_dimension = np.round(VF_dimension)

    coord_2_trsf = {}
    for k, v in coord_2.iteritems():
        t = v[-1]
        p = v[:-1]
        v_trsf = list(applyTrsf_full(mats, v[:-1], v[-1])) + [int(np.round(timeTrsf(t)))]
        coord_2_trsf[k] = v_trsf

    barycenters, b_dict = get_barycenter(path_barycenter)

    line, somites_pairing = get_somite_pairing(coord_1)

    avg = np.mean(line, axis = 0)
    uu, dd, vv = np.linalg.svd(line - avg)
    fitted_line = avg + vv[0] * (line[:, 0].reshape(-1, 1) - avg[0])

    projection_onto_P = lambda p, origin, n: ((p - np.dot(n, p - origin)/(np.linalg.norm(n))
                                              * (n/np.linalg.norm(n))), np.dot(n, p - origin)/(np.linalg.norm(n)))

    rotation = lambda v, r, theta: (1 - np.cos(theta)) * (np.dot(v, r)) * r + np.cos(theta) * v + np.sin(theta) * (np.cross(r, v))


    somite_front_1, pos_SF_1 = get_somite_front(coord_1)
    somite_front_2, pos_SF_2 = get_somite_front(coord_2_trsf)

    (somite_end_1, pos_SE_1), (somite_end_2, pos_SE_2) = get_common_somites_end(coord_1, coord_2_trsf)



    pos_AMN_1 = {v[-1]:v[:-1] for k, v in coord_1.iteritems() if 'AMN' in k}
    pos_AMN_2 = {int(np.round(timeTrsf(v[-1]))):np.array(applyTrsf_full(mats, v[:-1], v[-1])) for k, v in coord_2.iteritems() if 'AMN' in k}

    pos_HFAP_1 = {v[-1]:v[:-1] for k, v in coord_1.iteritems() if 'HFAP' in k}
    pos_HFAP_2 = {int(np.round(timeTrsf(v[-1]))):np.array(applyTrsf_full(mats, v[:-1], v[-1])) for k, v in coord_2.iteritems() if 'HFAP' in k}

    pos_HFPP_1 = {v[-1]:v[:-1] for k, v in coord_1.iteritems() if 'HFPP' in k}
    pos_HFPP_2 = {int(np.round(timeTrsf(v[-1]))):np.array(applyTrsf_full(mats, v[:-1], v[-1])) for k, v in coord_2.iteritems() if 'HFPP' in k}

    AMN_1 = interp_3d_coord(np.array([np.array(list(v)+[k]) for k, v in pos_AMN_1.iteritems()]), ext = 3)
    AMN_2 = interp_3d_coord(np.array([np.array(list(v)+[k]) for k, v in pos_AMN_2.iteritems()]), ext = 3)
    HFPP_1 = interp_3d_coord(np.array([np.array(list(v)+[k]) for k, v in pos_HFPP_1.iteritems()]), ext = 3)
    HFPP_2 = interp_3d_coord(np.array([np.array(list(v)+[k]) for k, v in pos_HFPP_2.iteritems()]), ext = 3)

    pos_S_1 = get_somite_pos(coord_1)
    pos_S_2 = get_somite_pos(coord_2_trsf)

    common_S = set(pos_S_1.keys()).intersection(pos_S_2.keys())

    S_1 = []
    for S_name in common_S:
        Si = interp_3d_coord(np.array([np.array(list(v)+[k]) for k, v in pos_S_1[S_name].iteritems()]), ext = 3)
        S_1 += [[pos_S_1[S_name], Si]]

    S_2 = []
    for S_name in common_S:
        Si = interp_3d_coord(np.array([np.array(list(v)+[k]) for k, v in pos_S_2[S_name].iteritems()]), ext = 3)
        S_2 += [[pos_S_2[S_name], Si]]

    TP_starting_LM = min([min(S[0]) for S in S_1 + S_2])

    static_barycenter_val = barycenters[TP_starting_LM]#np.mean(np.array(barycenters.values()), axis = 0)
    static_barycenter = {k:v if TP_starting_LM <= k else static_barycenter_val for k, v in barycenters.iteritems()}
    b_dict = {k:v if TP_starting_LM <= k else static_barycenter_val for k,v in b_dict.iteritems()}

    pos_B_1 = pos_B_2 = pos_E_1 = pos_E_2 = b_dict
    B_1 = B_2 = interp_3d_coord(np.array([np.array(list(v)+[k]) for k, v in pos_B_1.iteritems()]), ext = 1)
    E_1 = E_2 = interp_3d_coord(np.array([np.array(list(v)+[k]) for k, v in pos_E_1.iteritems()]), ext = 1)

    all_dicts_1 = [[pos_E_1, E_1], [pos_HFPP_1, HFPP_1], [pos_AMN_1, AMN_1]] + S_1
    all_dicts_1 = all_dicts_1[::-1]
    all_dicts_2 = [[pos_E_2, E_2], [pos_HFPP_2, HFPP_2], [pos_AMN_2, AMN_2]] + S_2
    all_dicts_2 = all_dicts_2[::-1]

    for (d1, f1), (d2, f2) in zip(all_dicts_1, all_dicts_2):
        t_min = max(min(d1), min(d2))
        t_max = min(max(d1), max(d2))
        for t in d2.keys():
            if not (t_min <= t <= t_max):
                d2.pop(t)
        for t in d1.keys():
            if not (t_min <= t <= t_max):
                d1.pop(t)
        d1[t_min] = f1(t_min)
        d1[t_max] = f1(t_max)
        d2[t_min] = f2(t_min)
        d2[t_max] = f2(t_max)


    symmetry_plane = {}
    symmetry_plane_eq = {}
    for t, b in static_barycenter.iteritems():
        plane_params = compute_plane(np.array(b), fitted_line[0], fitted_line[-1])
        symmetry_plane[t] = b, plane_params[:-1]/np.linalg.norm(plane_params[:-1])
        symmetry_plane_eq[t] = plane_params

    LM_1 = extrapolate_pos(all_dicts_1, decay_time = 1)
    LM_2 = extrapolate_pos(all_dicts_2, decay_time = 1)

    ###
    ###### Computes the angles between the the landmarks
    LM_1 = LM_1[:-1]
    LM_2 = LM_2[:-1]
    LM_all = zip(LM_1, LM_2)
    origin = np.array([ 4000, 3000, 2500])
    out = []
    for t in LT_2.new_time_nodes.iterkeys():
        out += [get_angles([t, origin])]

    ###
    ###### Interpolates between the landmarks
    angles = np.zeros((len(LT_2.new_time_nodes.keys()), len(LM_1)))
    theta_disc = np.zeros((len(LT_2.new_time_nodes.keys()), len(LM_1)))
    proj_ori = {}
    first_time = min(LT_2.new_time_nodes)
    last_time = max(LT_2.new_time_nodes)
    for t, a, d, lm, proj in out:
        proj_ori[t] = proj
        for ai, di, pi in zip(a, d, lm):
            angles[t - first_time][pi] = ai
            theta_disc[t - first_time][pi] = di


    lms = [k[3] for k in out]
    max_size = angles.shape[0]
    delay = 40
    order_start = []
    order_end = []
    for i in range(len(LM_1)):
        s_time = np.min(np.argwhere(theta_disc[:,i] != 0))
        e_time = np.max(np.argwhere(theta_disc[:,i] != 0))
        order_start += [(s_time, i)]
        order_end += [(e_time, i)]

    order_start = sorted(order_start, cmp = lambda x1, x2: x1[0]-x2[0])
    order_end = sorted(order_end, cmp = lambda x1, x2: x2[0]-x1[0])

    for s_time, i in order_start:
        add_time = []
        delay = s_time 
        if 0 < s_time:
            A, T = angles[s_time - delay], theta_disc[s_time - delay]
            non_zeros = np.argwhere(angles[s_time - delay]!=0).flatten()
            if len(non_zeros) == 1:
                T = [T[non_zeros][0], ]*2
                A = [A[non_zeros][0], A[non_zeros][0] + 1e-3]
            elif len(lm) == 0:
                T = [0, 0]
                A = [0, 1e-3]
            else:
                T = T[non_zeros]
                A = A[non_zeros]
            if len(A) == 0:
                init_val = 0
            elif len(A) == 1:
                init_val = T[0]
            else:
                init_val = sp.interpolate.InterpolatedUnivariateSpline(A, T, k = 1, ext = 3)(angles[s_time, i])
            v_s = theta_disc[s_time, i]
            iterp = sp.interpolate.InterpolatedUnivariateSpline(np.array([max(0, s_time - delay), s_time]),
                                                                np.array([init_val, v_s]), k = 1, ext = 1)
            for t in range(max(0, s_time - delay), s_time):
                theta_disc[t, i] = iterp(t)
                angles[t, i] = angles[s_time, i]
                lms[t] += [i]
                add_time += [t]

    for e_time, i in order_end:
        delay = last_time - e_time
        if e_time < last_time - first_time:
            A, T = angles[e_time + delay], theta_disc[e_time + delay]
            non_zeros = np.argwhere(angles[e_time + delay]!=0).flatten()
            if len(non_zeros) == 1:
                T = [T[non_zeros][0], ]*2 #theta_disc[t][lm][0]]
                A = [A[non_zeros][0], A[non_zeros][0] + 1e-3]
            elif len(lm) == 0:
                T = [0, 0]
                A = [0, 1e-3]
            else:
                T = T[non_zeros]
                A = A[non_zeros]
            if len(A) == 0:
                init_val = 0
            elif len(A) == 1:
                init_val = T[0]
            else:
                init_val = sp.interpolate.InterpolatedUnivariateSpline(A, T, k = 1, ext = 3)(angles[e_time, i])

            v_s = theta_disc[e_time, i]
            iterp = sp.interpolate.InterpolatedUnivariateSpline(np.array([e_time, min(max_size - 1, e_time + delay)]),
                                                                np.array([v_s, init_val]), k = 1, ext = 1)
            for t in range(e_time, min(max_size - 1, e_time + delay)):
                theta_disc[t, i] = iterp(t)
                angles[t, i] = angles[e_time, i] 
                lms[t] += [i]
                add_time += [t]


    for l in out:
        l[3].sort()

    theta_disc = dict(zip(LT_2.new_time_nodes.keys(), theta_disc))
    angles = dict(zip(LT_2.new_time_nodes.keys(), angles))

    t_for_1 = []
    for t, a, d, lm, proj in out:
        if len(lm) == 1:
            theta_disc[t] = [theta_disc[t][lm][0], ]*2 #theta_disc[t][lm][0]]
            angles[t] = [angles[t][lm], angles[t][lm] + 1e-3]
            t_for_1 += [t]
        elif len(lm) == 0:
            theta_disc[t] = [0, 0]
            angles[t] = [0, 1e-3]
        else:
            theta_disc[t] = theta_disc[t][lm]
            angles[t] = angles[t][lm]

    ### Apply the piecewise rotation
    mapping = [(t, angles[t], theta_disc[t], proj_ori[t], out[t - first_time][3]) for t in VF_2.new_time_nodes]

    LT_for_multiprocess = LT_2
    pool = Pool()
    try:
        out1 = pool.map(apply_rot, mapping)
    except Exception as e:
        print e
        pool.terminate()
        pool.close()
    pool.terminate()
    pool.close()

    LT_2.new_pos = {}
    for d in out1:
        LT_2.new_pos.update(d)

    LT_for_multiprocess = VF_2
    pool = Pool()
    try:
        out1 = pool.map(apply_rot, mapping)
    except Exception as e:
        print e
        pool.terminate()
        pool.close()
    pool.terminate()
    pool.close()


    VF_2.new_pos = {}
    for d in out1:
        VF_2.new_pos.update(d)

    pos_all_1 = np.array(LT_1.pos.values())
    max_x = np.max(pos_all_1, axis = 0)[0]
    new_time_nodes_BU = deepcopy(VF_2.new_time_nodes)
    to_take_time_new = {}
    for t, cells in VF_2.new_time_nodes.iteritems():
        to_take_time_new[t] = []
        for c in cells:
            if VF_2.new_pos[c][0] < max_x:
                to_take_time_new[t] += [c]

    VF_2.new_time_nodes = to_take_time_new

    ### Computes the non-linear registration
    x, y, z = 0, 2, 1
    B_all = []
    R = 500
    nb_std = 1
    percentile = 80
    from itertools import product
    from time import time
    VF_2.final_pos = {}
    mapping = [(t, static_barycenter[t]) for t in sorted(LT_2.new_time_nodes.keys())]


    from itertools import product
    from time import time
    VF_2.final_pos = {}
    LT_2.new_time_nodes.keys()
    mapping = sorted(LT_2.new_time_nodes.keys())

    pos_all_1 = np.array(LT_1.pos.values())
    max_x = np.max(pos_all_1, axis = 0)[0]
    final_time_nodes = {}
    for t, N in VF_2.new_time_nodes.iteritems():
        final_time_nodes[t] = []
        for c in N:
            if VF_2.new_pos[c][0] < max_x - 100:
                final_time_nodes[t] += [c]

    VF_2.new_time_nodes = final_time_nodes

    ### Apply the non-linear registration
    out = []
    tic = time()
    pool = Pool()
    out = pool.map(apply_trsf, mapping)
    pool.terminate()
    pool.close()

    VF_2.final_pos = {}
    for f_pos in out:
        VF_2.final_pos.update(f_pos)
    print 'Overall time:', time()-tic

    # 3D time smoothing of the tracks.
    done = set()
    corresponding_track = {}
    smoothed_pos_2 = {}
    num_track = 0
    t_l = []
    for C in VF_2.final_pos:
        if not C in done:
            track = [C]
            tmp_track = [C]
            while tmp_track[-1] in VF_2.successor:
                if VF_2.successor[tmp_track[-1]][0] in VF_2.final_pos:
                    track.append(VF_2.successor[tmp_track[-1]][0])
                tmp_track.append(VF_2.successor[tmp_track[-1]][0])
            while tmp_track[0] in VF_2.predecessor:
                if VF_2.predecessor[tmp_track[0]][0] in VF_2.final_pos:
                    track.insert(0, VF_2.predecessor[tmp_track[0]][0])
                tmp_track.insert(0, VF_2.predecessor[tmp_track[0]][0])
            pos_track = np.array([VF_2.final_pos[Ci] for Ci in track if Ci in VF_2.final_pos])
            X = sp.ndimage.filters.gaussian_filter1d(pos_track[:, 0], sigma = 5)
            Y = sp.ndimage.filters.gaussian_filter1d(pos_track[:, 1], sigma = 5)
            Z = sp.ndimage.filters.gaussian_filter1d(pos_track[:, 2], sigma = 5)
            track_smoothed = np.zeros_like(pos_track)
            track_smoothed[:, 0] = X
            track_smoothed[:, 1] = Y
            track_smoothed[:, 2] = Z
            smoothed_pos_2.update(zip(track, list(track_smoothed)))
            done.update(set(track))

    if os.path.exists(path_2 + 'Database.csv'):
        f = open(path_2 + 'Database.csv')
        lines = f.readlines()
        f.close()

        tissue_label = {}
        max_label = -np.inf
        for l in lines[1:]:
            split_l = l.split(',')
            c_id = int(split_l[0])
            label = int(split_l[9])
            tissue_label[c_id] = label
            max_label = max(max_label, label)

        cells = smoothed_pos_2
        starting_cells = [c for c in smoothed_pos_2 if not VF_2.predecessor.get(c, [-1])[0] in smoothed_pos_2]
        th = 15
        to_remove = set()
        new_pred = deepcopy(VF_2.predecessor)
        max_label = max(tissue_label.values())
        for c in starting_cells:
            track = [c]
            pos_track = [smoothed_pos_2[c]] # 
            wrong = []
            i = 0
            while VF_2.successor.get(track[-1], [-1])[0] != -1:
                track += VF_2.successor[track[-1]]
                pos_track += [smoothed_pos_2.get(track[-1], None)]
                i+=1    
                if smoothed_pos_2.get(track[-1], None) is None:
                    wrong += [i]
            label = max([tissue_label.get(ci, -np.inf) for ci in track])
            for ci in track:
                if label < 0:
                    label = -1
                tissue_label[ci] = label
            if wrong != []:
                for i in wrong:
                    if i < len(track) - 1:
                        new_pred.pop(track[i + 1], None)
                tracks = []
                pos_tracks = []
                for i, p in enumerate(wrong):
                    if i == 0:
                        tracks += [track[:p]]
                        pos_tracks += [pos_track[:p]]
                    elif i == len(wrong) - 1:
                        tracks += [track[p+1:]]
                        tracks += [track[wrong[i-1]+1:p]]
                        pos_tracks += [pos_track[p+1:]]
                        pos_tracks += [pos_track[wrong[i-1]+1:p]]
                    else:
                        tracks += [track[wrong[i-1]+1:p]]
                        pos_tracks += [pos_track[wrong[i-1]+1:p]]
            else:
                tracks = [track]
                pos_tracks = [pos_track]
            tracks = [tr for tr in tracks if tr != []]
            pos_tracks = [tr for tr in pos_tracks if tr != []]
            for track, pos_track in zip(tracks, pos_tracks):
                track = np.array(track)
                pos_track = np.array(pos_track)
                if 1 < len(track):
                    speed = np.linalg.norm(pos_track[1:] - pos_track[:-1], axis = 1)
                    if (th < speed).any():
                        borders = np.ones(len(speed) + 6).astype(np.bool)
                        borders[3:-3] = nd.binary_dilation(th < speed, iterations=3)
                        th_scal = nd.binary_erosion(borders, iterations=3)[3:-3]
                        to_remove.update(track[1:][th_scal])
                        for ci in track[1:][th_scal]:
                            new_pred.pop(ci, None)
    else:
        print "No database for labeling"
        tissue_label = {}

    ### Writing of the output
    VF_for_speed = lineageTree(path_2 + 'SVF.bin')

    f = open(TARDIS_folder_2 + 'TARDIS_coord.csv', 'w')
    f.write('cell_id, m_id,x r,y r,z r,x r+,y r+,z r+,x NL,y NL,z NL,label,t,dx,dy,dz,speed\n')
    for t, cells in VF_2.new_time_nodes.iteritems(): 
        for c_id in cells:
            m_id = VF_2.predecessor.get(c_id, [-1])[0]
            if c_id in VF_for_speed.predecessor:
                speed = np.linalg.norm(VF_for_speed.pos[c_id] - VF_for_speed.pos[VF_for_speed.predecessor[c_id][0]])
                d_M_p = tuple(VF_for_speed.pos[c_id] - VF_for_speed.pos[VF_for_speed.predecessor[c_id][0]])
            else:
                speed = -1
                d_M_p = (-1, -1, -1)
            f.write('%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d,%d,%.5f,%.5f,%.5f,%.5f\n'%((c_id,m_id) +
                                        tuple(VF_2_rigid.get(c_id, [-1, -1, -1])) +
                                        tuple(VF_2.new_pos.get(c_id, [-1, -1, -1])) +
                                        tuple(smoothed_pos_2.get(c_id, [-1, -1, -1])) +
                                        (tissue_label.get(c_id, -1), t) + d_M_p + (speed,)))
    f.close()


    tissue_label = {k:l if 0<=l else max_label+1 for k, l in tissue_label.iteritems()}
    write_to_am_2(TARDIS_folder_2 + 'SVF_TARDIS/seg_t%04d.am', VF_2, t_b = None, t_e = None,
                  manual_labels = tissue_label, default_label = max_label + 1, 
                  length = 7, new_pos = smoothed_pos_2, to_take_time = VF_2.new_time_nodes)


    write_to_am_rem(TARDIS_folder_2 + 'SVF_TARDIS_filtered/seg_t%04d.am', VF_2, t_b = None, t_e = None,
                  manual_labels = tissue_label, default_label = max_label + 1, 
                  length = 7, new_pos = smoothed_pos_2, to_take_time = VF_2.new_time_nodes,
                  to_remove = to_remove, predecessor = new_pred)
