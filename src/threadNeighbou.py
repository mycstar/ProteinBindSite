from Bio.PDB import *
import urllib.request
import numpy as np
import pandas as pd
from math import sqrt
import time
import os
import math
import heapq
from datetime import datetime
from multiprocessing import Pool
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GroupKFold

dir_path = os.getcwd()

peptidasesList = pd.read_csv("./MCSA_EC3.4_peptidases.csv")
peptidasesList = peptidasesList[peptidasesList.iloc[:, 4] == "residue"]

peptidasesList = peptidasesList.reset_index(drop=True)
print(len(peptidasesList))

bindingSiteDic = {}
for i in range(len(peptidasesList)):
    # print(bindingSiteDic)
    if peptidasesList.loc[i, "PDB"] not in bindingSiteDic:
        bindingSiteDic[peptidasesList.loc[i, "PDB"]] = {
            peptidasesList.loc[i, "chain/kegg compound"]: [peptidasesList.loc[i, "resid/chebi id"]]}
    elif peptidasesList.loc[i, "chain/kegg compound"] not in bindingSiteDic[peptidasesList.loc[i, "PDB"]]:
        bindingSiteDic[peptidasesList.loc[i, "PDB"]] = {
            peptidasesList.loc[i, "chain/kegg compound"]: [peptidasesList.loc[i, "resid/chebi id"]]}
    else:
        bindingSiteDic[peptidasesList.loc[i, "PDB"]][peptidasesList.loc[i, "chain/kegg compound"]].append(
            peptidasesList.loc[i, "resid/chebi id"])
for protein in bindingSiteDic:
    for chain in bindingSiteDic[protein]:
        bindingSiteDic[protein][chain] = [int(x) for x in list(set(bindingSiteDic[protein][chain]))]

uniqueList = peptidasesList[["PDB", "chain/kegg compound"]].drop_duplicates()

uniqueList.reset_index(drop=True).iloc[20:, ]

backbone = ["N", "CA", "C", "O"]
aminoAcidCodes = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLY", "GLU", "HIS", "ILE", "LEU", "LYS",
                  "MET", "PHE", "PRO", "PYL", "SER", "SEC", "THR", "TRP", "TYR", "TRP", "VAL"]

neighhor_df = pd.DataFrame(columns=["proteinid", "chain", "aaid", "neighborid"])
n_bigger = 10
target_list = []
start_time = datetime.now()


def threadCalc(fold, protein_list):
    for eachRow in range(0, len(protein_list)):
        pdbID = protein_list[eachRow][0]
        chainOrder = protein_list[eachRow][1]
        PDB = PDBList()
        PDB.retrieve_pdb_file(pdb_code=pdbID, pdir="./pdb", file_format="pdb")

        protein_start_time = datetime.now()
        p = PDBParser()
        structure = p.get_structure("X", "./pdb/pdb" + pdbID + ".ent")
        oneChain = pd.DataFrame(columns=["Seq", "Residue", "Center", "Direction"])

        if structure.header["resolution"] <= 3.0:
            if chainOrder in [x.id for x in list(structure[0].get_chains())]:
                chain = chainOrder
                for residue in structure[0][chainOrder]:
                    if residue.get_resname() in aminoAcidCodes:
                        if len(list(residue.get_atoms())) > 3:
                            if residue.get_resname() != "GLY":
                                point = vectors.Vector([0, 0, 0])
                                for atom in residue:
                                    if (atom.get_name() not in backbone):
                                        point = point + atom.get_vector()
                                center = point.__div__(len(residue) - 4)
                                cToRGroup = residue["CA"].get_vector() - center
                                oneChain.loc[len(oneChain)] = [residue.get_id()[1], residue.get_resname(), center,
                                                               cToRGroup]
                            else:
                                center = residue["CA"].get_vector()
                                cToRGroup = center - (residue["C"].get_vector() + residue["N"].get_vector() + residue[
                                    "O"].get_vector()).__div__(3)
                                oneChain.loc[len(oneChain)] = [residue.get_id()[1], residue.get_resname(), center,
                                                               cToRGroup]

                columns = np.array(list(oneChain.iloc[:, 0]))
                row_index = oneChain.iloc[:, 0]

                distanceMatrix = pd.DataFrame(columns=list(oneChain.iloc[:, 0]), index=list(oneChain.iloc[:, 0]))
                print(time.time())
                numResidue = len(oneChain)
                for row in range(0, numResidue):
                    if row % 50 == 0:
                        print(str(row) + "th row")
                    for column in range(0, numResidue):
                        coordinatesSubstraction = list(oneChain.loc[row, "Center"] - oneChain.loc[column, "Center"])
                        distanceMatrix.iloc[row, column] = sqrt(
                            sum(list(map(lambda x: x * x, coordinatesSubstraction))))
                        # distanceMatrix.iloc[row, column] = sqrt(sum(list(map(lambda x: x * x, coordinatesSubstraction))))

                for row in range(0, numResidue):
                    row_list = list(distanceMatrix.iloc[row, :])
                    result = list(map(row_list.index, heapq.nsmallest(n_bigger, row_list)))
                    target_col = columns[result]
                    target_list.append(target_col)
                    neighhor_df.loc[len(neighhor_df)] = [pdbID, chain, row_index[row], str(target_col)]

        protein_end_time = datetime.now()
        print(pdbID, " Duration: {}".format(protein_end_time - protein_start_time))
    neighhor_df.to_csv("./mid_result/" + str(fold) + ".csv")
    print('  sub process %s finished  ' % fold)


def chunks(l, m):
    """Yield successive n-sized chunks from l."""
    n = int(math.ceil(len(l) / float(m)))
    for i in range(0, len(l), n):
        yield l[i:i + n]


thread_size = 50
protein_list = np.array(uniqueList).tolist()
#kfold = StratifiedKFold(n_splits=thread_size, shuffle=False)
group_list = list(chunks(protein_list, thread_size))

print('Parent process %s.' % os.getpid())
#p = Pool()

for fold, fold_list in enumerate(group_list):
    print(fold)
    threadCalc(fold, fold_list)
    #p.apply_async(threadCalc, args=(fold, fold_list))

print('Waiting for all subprocesses done...')
#p.close()
#p.join()
print('All subprocesses done.')

end_time = datetime.now()
print("The total Duration: {}".format(end_time - start_time))
print(time.time())
