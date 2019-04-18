from Bio.PDB import *
import urllib.request
import numpy as np
import pandas as pd
from math import sqrt
import time
import os
import heapq
from datetime import datetime

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
n_bigger = 5
target_list = []
start_time = datetime.now()

for eachRow in range(0, len(uniqueList)):
    pdbID = uniqueList.iloc[eachRow, 0]
    chainOrder = uniqueList.iloc[eachRow, 1]
    PDB = PDBList()
    PDB.retrieve_pdb_file(pdb_code=pdbID, pdir="../pdb", file_format="pdb")
    p = PDBParser()
    structure = p.get_structure("X", "../pdb/pdb" + pdbID + ".ent")
    oneChain = pd.DataFrame(columns=["Seq", "Residue", "Center", "Direction"])

    protein_start_time = datetime.now()

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
                    distanceMatrix.iloc[row, column] = sqrt(sum(list(map(lambda x: x * x, coordinatesSubstraction))))
                    # distanceMatrix.iloc[row, column] = sqrt(sum(list(map(lambda x: x * x, coordinatesSubstraction))))

            row_list = list(distanceMatrix.iloc[row, :])
            result = list(map(row_list.index, heapq.nsmallest(n_bigger, row_list)))
            target_col = columns[result]
            target_list.append(target_col)
            neighhor_df.loc[len(neighhor_df)] = [pdbID, chain, row_index[row], str(target_col)]

    protein_end_time = datetime.now()
    print(pdbID, " Duration: {}".format(protein_end_time - protein_start_time))

end_time = datetime.now()
print("The total Duration: {}".format(end_time - start_time))
print(time.time())

pdbID = uniqueList.iloc[35, 0]
chainOrder = uniqueList.iloc[35, 1]
PDB = PDBList()
for pdbid in uniqueList.iloc[:, 0]:
    exist = os.path.isfile('../pdb/pdb' + pdbID + '.ent')
    if not exist:
        PDB.retrieve_pdb_file(pdb_code=pdbid, pdir="../pdb", file_format="pdb")

p = PDBParser()
structure = p.get_structure("X", "../pdb/pdb" + pdbID + ".ent")

oneChain = pd.DataFrame(columns=["Seq", "Residue", "Center", "Direction", "pdbid", "chain"])
if structure.header["resolution"] <= 3.0:
    if chainOrder in [x.id for x in list(structure[0].get_chains())]:  # Chain information not in pdb file
        for residue in structure[0][chainOrder]:
            if residue.get_resname() in aminoAcidCodes:  # Only treat common amino acid
                if len(list(residue.get_atoms())) > 3:
                    if residue.get_resname() != "GLY":  # Glysine as a special case
                        point = vectors.Vector([0, 0, 0])
                        for atom in residue:
                            if (atom.get_name() not in backbone):
                                point = point + atom.get_vector()
                        center = point.__div__(len(residue) - 4)
                        cToRGroup = residue["CA"].get_vector() - center
                        oneChain.loc[len(oneChain)] = [residue.get_id()[1], residue.get_resname(), center, cToRGroup,
                                                       pdbID, chainOrder]
                    else:
                        center = residue["CA"].get_vector()
                        cToRGroup = center - (residue["C"].get_vector() + residue["N"].get_vector() + residue[
                            "O"].get_vector()).__div__(3)
                        oneChain.loc[len(oneChain)] = [residue.get_id()[1], residue.get_resname(), center, cToRGroup,
                                                       pdbID, chainOrder]

distanceMatrix = pd.DataFrame(columns=list(oneChain.iloc[:, 0]), index=list(oneChain.iloc[:, 0]))
print(len(oneChain))

print(time.time())
numResidue = len(oneChain)
columns = np.array(list(oneChain.iloc[:, 0]))
n_bigger = 3
target_list = []
for row in range(0, numResidue):
    if row % 50 == 0:
        print(str(row) + "th row")
    for column in range(0, numResidue):
        coordinatesSubstraction = list(oneChain.loc[row, "Center"] - oneChain.loc[column, "Center"])
        distanceMatrix.iloc[row, column] = sqrt(sum(list(map(lambda x: x * x, coordinatesSubstraction))))
    row_list = list(distanceMatrix.iloc[row, :])
    result = list(map(row_list.index, heapq.nlargest(n_bigger, row_list)))
    target_col = columns[result]
    target_list.append(target_col)

print(time.time())

sortedDistance = distanceMatrix.apply(lambda x: np.sort(x), axis=1)

sortedD = np.array(sortedDistance.tolist())
# get 10 biggest value
sortedD[:, len(oneChain) - 10:]

# get the index 10 biggest value
distanceMatrix.apply(lambda x: np.argsort(x), axis=1).iloc[:, len(oneChain) - 10:]

for eachRow in range(0, len(uniqueList)):
    pdbID = uniqueList.iloc[eachRow, 0]
    chainOrder = uniqueList.iloc[eachRow, 1]
    PDB = PDBList()
    PDB.retrieve_pdb_file(pdb_code=pdbID, pdir="../pdb", file_format="pdb")
    p = PDBParser()
    structure = p.get_structure("X", "../pdb/pdb" + pdbID + ".ent")
    oneChain = pd.DataFrame(columns=["Seq", "Residue", "Center", "Direction"])
    if structure.header["resolution"] <= 3.0:
        if chainOrder in [x.id for x in list(structure[0].get_chains())]:
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
            distanceMatrix = pd.DataFrame(columns=list(oneChain.iloc[:, 0]), index=list(oneChain.iloc[:, 0]))
            print(time.time())
            numResidue = len(oneChain)
            for row in range(0, numResidue):
                if row % 50 == 0:
                    print(str(row) + "th row")
                for column in range(0, numResidue):
                    coordinatesSubstraction = list(oneChain.loc[row, "Center"] - oneChain.loc[column, "Center"])
                    distanceMatrix.iloc[row, column] = sqrt(sum(list(map(lambda x: x * x, coordinatesSubstraction))))
        print(time.time())
