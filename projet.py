# -*- coding: utf-8 -*-
'''
author: 
Damian BIMBENET 
Bailin CAI
Mohamed SOUANE 

date: 2021-03-25
'''

from mpi4py import MPI

comm = MPI.COMM_WORLD.Dup()
rank = comm.rank
nbp = comm.size

import mesh
import visu_split_mesh as VSM
import splitter
import numpy as np
import visu_solution as VS
import fem
import fem_laplacian as laplacian
from math import cos,sin,pi,sqrt
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from conjugate_gradient import *

nb_domains = nbp
msh,cl = mesh.read("CarrePetit.msh")
vert2elts = msh.comp_vertices_to_elements()
# vert2elts <-> vert2elt

splitted_elements = splitter.split_element_mesh(nb_domains, msh.elt2verts, msh.vertices)[rank]
# splitted_elements <-> element_loc2glob

local_nodes = set()
for e in splitted_elements:
    local_nodes.add(msh.elt2verts[e][0])
    local_nodes.add(msh.elt2verts[e][1])
    local_nodes.add(msh.elt2verts[e][2])

print(f"noeuds locaux : {local_nodes}")

loc2glob_aux = np.array(list(local_nodes), dtype = msh.elt2verts.dtype)

interface = [None]*nbp

for p in range(nbp):
    nodes = comm.bcast(local_nodes, root=p)
    if rank != p:
        interface[p] = np.array(list(local_nodes.intersection(nodes)), dtype = msh.elt2verts.dtype)

interface_aux_array = np.array([], dtype = np.uint8)
for element in interface:
    if element is not None:
        #interface_aux_array = element
        #interface_aux_array = np.concatenate((interface_aux_array, element), dtype = np.uint8)
        interface_aux_array = np.concatenate((interface_aux_array, element))
        #break

interface_aux = list(interface_aux_array)

loc2glob = np.array([], dtype = np.uint8)
for value in loc2glob_aux:
    if value not in interface_aux:
        loc2glob = np.concatenate((loc2glob, np.array([value], dtype = np.uint8)))
for value in loc2glob_aux:
    if value in interface_aux:
        loc2glob = np.concatenate((loc2glob, np.array([value], dtype = np.uint8)))

glob2local = -1*np.ones(msh.vertices.shape[0], dtype = msh.elt2verts.dtype)

for index, n in enumerate(loc2glob):
    glob2local[n] = index

# Premier indice correspondant aux éléments dans l'interface.
indice_interface = loc2glob.shape[0] - 1 - len(interface_aux)

elt2verts = msh.elt2verts
nbElts  = elt2verts.shape[0]

glob2local_elements = -1*np.ones(nbElts)

for index, n in enumerate(splitted_elements):
    glob2local_elements[n] = index

#interface = [None]*nbp

#for p in range(nbp):
#    nodes = globCom.bcast(local_nodes, root=p)
#    if rank != p:
#        interface[p] = np.array(list(local_nodes.intersection(nodes)), dtype = msh.elt2verts.dtype)

#print(f"type d'interface : {type(interface)}")

print(f"loc2glob : {loc2glob}")
print(f"interface : {interface}")

#print(f"{interface[1]}")
#print(f"{type(interface[1])}")

#print(f"{list(interface[1])}")
#print(f"{type(list(interface[1]))}")

local_coords = msh.vertices[loc2glob]

# local_elt2verts <-> locElmt2vert

local_elt2verts = np.empty((splitted_elements.shape[0],3),dtype = msh.elt2verts.dtype)

for index,e in enumerate(splitted_elements):
    local_elt2verts[index,:] = glob2local[msh.elt2verts[e]]

sol = np.zeros(local_coords.shape[0], dtype = np.double)

VS.view(local_coords, local_elt2verts, sol)

def g(x,y) :
    return cos(2*pi*x)+sin(2*pi*y)

#m,cl = mesh.read("CarreMedium.msh")
coords    = msh.vertices
elt2verts = msh.elt2verts
# elt2verts <->globElt2vert
nbVerts = coords.shape[0]
nbElts  = elt2verts.shape[0]
# nbElts <-> nb_globElem
nbVertsloc = local_coords.shape[0]
nbEltsloc = local_elt2verts.shape[0]
print('nbVerts : {}'.format(nbVerts))
print('nbElts  : {}'.format(nbElts))
print('nbVertsloc : {}'.format(nbVertsloc))
print('nbEltsloc : {}'.format(nbEltsloc))

# local_coords <-> locVertices

msh_loc = mesh.Mesh(local_coords,local_elt2verts)
begVert2Elts, vert2elts = msh_loc.comp_vertices_to_elements()

#locMesh = mesh.Mesh(locVertices, locElmt2vert)
#begVert2Elts, vert2elts = locMesh.comp_vertices_to_elements()

#begVert2Elts, vert2elts = msh.comp_vertices_to_elements()

begRows, indCols = fem.compute_skeleton_sparse_matrix(local_elt2verts, (begVert2Elts, vert2elts) )
#begRows, indCols = fem.compute_skeleton_sparse_matrix(elt2verts, (begVert2Elts, vert2elts) )
nz = begRows[-1]
print("Number of non zero in sparse matrix : {}".format(nz))

spCoefs = np.zeros( (nz,), np.double)
for iElt in range(nbEltsloc):
#for iElt in range(nbElts):
    iVertices = local_elt2verts[iElt,:]
    #iVertices = elt2verts[iElt,:]
    crd1 = local_coords[iVertices[0],:]
    crd2 = local_coords[iVertices[1],:]
    crd3 = local_coords[iVertices[2],:]
    #crd1 = coords[iVertices[0],:]
    #crd2 = coords[iVertices[1],:]
    #crd3 = coords[iVertices[2],:]
    matElem = laplacian.compute_elementary_matrix(crd1, crd2, crd3)
    #print(matElem)
    fem.add_elementary_matrix_to_csr_matrix((begRows,indCols,spCoefs), (iVertices, iVertices, matElem))

def prodMatVect(x):
    dim_row = begRows.shape[0]
    y = np.zeros(dim_row-1)
    for indice_i in range(len(begRows)-1):
        for indice_j in range(begRows[indice_i], begRows[indice_i+1]):
            y[indice_i] = spCoefs[indice_j]*x[indCols[indice_j]]
    auxiliary = y[indice_interface - 1:]
    comm.Allreduce(MPI.IN_PLACE, auxiliary, op = MPI.SUM)
    y[indice_interface - 1:] = auxiliary
    return y

def prodMatVect_loc(x):
    dim_row = begRows.shape[0]
    y = np.zeros(dim_row-1)
    for indice_i in range(len(begRows)-1):
        for indice_j in range(begRows[indice_i], begRows[indice_i+1]):
            y[indice_i] = spCoefs[indice_j]*x[indCols[indice_j]]
    #auxiliary = y[indice_interface - 1:]
    #comm.Allreduce(MPI.IN_PLACE, auxiliary, op = MPI.SUM)
    #y[indice_interface - 1:] = auxiliary
    return y

def prodScal(x, y):
    if x.shape != y.shape:
        raise ValueError("x et y doivent avoir la même dimension.")
    scalar_product_loc = 0.
    for count,value in enumerate(x):
        buff = (1/nbp)*value*y[count] if count >= indice_interface else value*y[count]
        scalar_product_loc += buff
    return scalar_product_loc

# Assemblage second membre :
f = np.zeros(nbVertsloc, np.double)
#f = np.zeros(nbVerts, np.double)
for iVert in range(nbVertsloc):
#for iVert in range(nbVerts):
    if ( cl[iVert] > 0 ) :
        f[iVert] += g(local_coords[iVert,0],local_coords[iVert,1])
        #f[iVert] += g(coords[iVert,0],coords[iVert,1])
b = np.zeros(nbVertsloc, np.double)
#b = np.zeros(nbVerts, np.double)
for i in range(nbVertsloc):
#for i in range(nbVerts) :
    for ptR in range(begRows[i],begRows[i+1]):
        b[i] -= spCoefs[ptR]*f[indCols[ptR]]        
# Il faut maintenant tenir compte des conditions limites :
for iVert in range(nbVertsloc):
#for iVert in range(nbVerts):
    if cl[iVert] > 0: # C'est une condition limite !
        # Suppression de la ligne avec 1 sur la diagonale :
        for i in range(begRows[iVert],begRows[iVert+1]):
            if indCols[i] != iVert :
                spCoefs[i] = 0.
            else :
                spCoefs[i] = 1.
        # Suppression des coefficients se trouvant sur la colonne iVert :
        for iRow in range(nbVertsloc):
        #for iRow in range(nbVerts):
            if iRow != iVert :
                for ptCol in range(begRows[iRow],begRows[iRow+1]):
                    if indCols[ptCol] == iVert :
                        spCoefs[ptCol] = 0.
                       
        b[iVert] = f[iVert]
# On definit ensuite la matrice :
spMatrix = sparse.csc_matrix((spCoefs, indCols, begRows), shape=(nbVertsloc,nbVertsloc))
#spMatrix = sparse.csc_matrix((spCoefs, indCols, begRows),
#                             shape=(nbVerts,nbVerts))
print("Matrice creuse : {}".format(spMatrix))

# Visualisation second membre :
VS.view(local_coords, local_elt2verts, b, title = "Second membre")
#VS.view(local_coords, elt2verts, b, title = "Second membre")
#VS.view( coords, elt2verts, b, title = "second membre" )

# Résolution du problème séquentiel avec gradient conjugue :
sol, info = solver_gc(spMatrix, b, None, tol=1.E-14, M=None, verbose=True)
print(f"||r_final|| = {info[0]}, nombre itérations : {info[1]}")
print(sol)
d = spMatrix.dot(sol) - b
print("||A.x-b||/||b|| = {}".format(sqrt(d.dot(d)/b.dot(b))))
# Visualisation de la solution :
VS.view(local_coords, local_elt2verts, sol, title = "Solution")
#VS.view(local_coords, elt2verts, sol, title = "Solution")
#VS.view( coords, elt2verts, sol, title = "Solution" )


'''
####################################Version incorrecte############################################################
# Résolution du problème séquentiel avec gradient conjugue version parallèle:
def solver_gc_parallel( A, b, x0=None, M=None, tol=1.E-7, niterMax=100,
                        prodMatVect = None, prodPrecvect = None, prodScal=None,
                        verbose=False):
    """Résolution du système linéaire Ax = b avec le gradient conjugue.
    Paramètres :
    - A : matrice creuse (sparse matrix)
    - b : second membre (numpy array)
    - x0 : solution initiale (numpy array)
    - tol : critère d'arrêt
    - M : matrice de préconditionnement (sparse matrix)
    - niterMax : nombre d'itérations maximum
    - prodMatVect : produit matrice-vecteur version parallèle sinon np.dot
    - prodPrecvect : produit matrice de préconditionnement-vecteur version parallèle sinon np.dot
    - prodScal : produit scalaire version parallèle sinon np.dot
    - verbose : affichage des informations
    Résultat :
    - x : solution du système linéaire
    - info : liste contenant les informations sur la convergence
    """
    # Initialisation :
    n = b.shape[0]

    # produit matrice-vecteur
    prodA = prodMatVect 
    if prodMatVect is None:
        prodA = A.dot

    # produit matrice de préconditionnement-vecteur
    prodM = prodPrecvect
    if prodM is None:
        if M is not None:
            prodM = M.dot
    # produit scalaire
    dotS = prodScal 
    if prodScal is None:
        dotS = np.dot

    # le vecteur direction de descente
    if x0 is not None:
        r = b - prodA(x0)
    else:
        r = b

    # si avec le preconditionnement
    if prodM:
        p = prodM(r)
    else:
        p = r
    z = p

    if x0:
        x = x0.copy()
    else:
        x = np.zeros(b.shape, np.double)

    # diviser par le nombre de processus
    n_loc = n // nbp

    b_loc = b[rank*n_loc:(rank+1)*n_loc]
    x_loc = x[rank*n_loc:(rank+1)*n_loc]
    print("x_loc = ", x_loc)
    print("b_loc = ", b_loc)
    print("A = ", A)
    #r_loc = b_loc - prodA(x)
    r_loc = r[rank*n_loc:(rank+1)*n_loc]
    #r = np.zeros(n, np.double)
    #comm.Allreduce(r_loc, r, op=MPI.SUM)

    if M is not None:
        z_loc = prodM(r_loc)
        z = np.zeros(n, np.double)
        comm.Allreduce(z_loc, z, op=MPI.SUM)
    else:
        z = r.copy()

    p_loc = z[rank * n_loc:(rank + 1) * n_loc].copy()

    r_dot_z_old = dotS(r, z)
    err0 = r_dot_z_old

    if verbose:
        print("Norme L2 initiale de l'erreur = {}".format(sqrt(err0)))
    err = err0
    nit = 0

    # main du gradient conjugue :
    while (err > tol**2 * err0) and (nit < niterMax):
        Ap_k = prodA(p)
        Ap = np.zeros(n, np.double)
        print("Ap_k = ", Ap_k)
        print("Ap = ", Ap)
        comm.Allgather(Ap_k, Ap)

        alpha = r_dot_z_old / dotS(p, Ap)

        x_loc += alpha * p_loc
        r_loc -= alpha * Ap_k

        comm.Allgather(r_loc, r)

        if M is not None:
            z_loc = prodM(r_loc)
            z = np.zeros(n, np.double)
            comm.Allgather(z_loc, z)
        else:
            z = r.copy()

        r_dot_z_new = dotS(r, z)

        if verbose and rank == 0:
            print(f"Iteration {nit+1} : ||r|| = {sqrt(r_dot_z_new)}")
        
        err = r_dot_z_new
        nit += 1

        beta = r_dot_z_new / r_dot_z_old

        p_loc = z[rank * n_loc:(rank + 1) * n_loc] + beta * p_loc
        r_dot_z_old = r_dot_z_new

    x = np.zeros(n, np.double)
    comm.Allgather(x_loc, x)
    return x, (sqrt(err/err0), nit)
'''
