import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

dict_of_pattern = {
    "Suzuki": ['c!@c', 'c!@[CX3]=[CX3]', '[CX3]=[CX3]!@[CX3]=[CX3]'],
    "Buchwald_Hartwig": ['c!@N', 'c!@n'],
    "Alkylation_ONS": ['C!@N', 'C!@n', 'C!@O', 'C!@S'],
    "Michael": ['[O-][N+](=O)CC!@O', '[O-][N+](=O)CC!@N', '[O-][N+](=O)CC!@S',
                    'N#C!@CC!@N', 'N#C!@CC!@S', 'N#C!@CC!@O',
                    'CC(=O)CC!@N', 'CC(=O)CC!@S', 'CC(=O)CC!@O',
                    'N!@CCC=O', 'S!@CCC=O', 'O!@CCC=O',
                    'N!@CCC(O)=O', 'S!@CCC(O)=O', 'O!@CCC(O)=O',
                    'N!@CCC(N)=O', 'S!@CCC(N)=O', 'O!@CCC(N)=O'],
    "Wittig": ['C!@=C'],
    "Grignard": ['CC(C)[OH]', 'CC(C)(C)[OH]', 'CC(c)[OH]', 'CC(C)(c)[OH]'],
    "Acylation_ON": ['CC(=O)OC', 'CC(=O)NC', 'O=C(C)Oc', 'O=C(C)Nc', 'O=C(c)Nc', 'O=C(c)Oc'],
    "Fridel": ['CC(=O)c', 'cC(=O)c', ]

}

def max_percent_atom(smiles, reactants_list):
    mol = Chem.MolFromSmiles(smiles)
    count_atom_mol = 0
    list_count = []
    for reactant in reactants_list:
        react = Chem.MolFromSmiles(reactant)
        count = 0
        for atom in react.GetAtoms():
            if atom.GetAtomMapNum():
                count += 1
        list_count.append(count)
    for _ in mol.GetAtoms():
        count_atom_mol += 1

    return max(list_count)/count_atom_mol


def max_count_fused_cycle(smile_string):
    mol = Chem.MolFromSmiles(smile_string)
    rings_list = [set(atom) for atom in mol.GetRingInfo().AtomRings()
                      if all([mol.GetAtomWithIdx(index).GetAtomMapNum() for index in atom
                              if mol.GetAtomWithIdx(index).GetAtomMapNum() != 0])]
    list_fused_rings = []
    while rings_list:
        if rings_list:
            temp = rings_list.pop()
            count = 1
            for i in range(len(rings_list)):
                for ring in rings_list:
                    if temp.intersection(ring):
                        temp = temp | ring
                        count += 1
                        rings_list.remove(ring)
            list_fused_rings.append(count)
        else:
            break
    if list_fused_rings:
        return max(list_fused_rings)
    else:
        return 0

def count_cycle(smile_string, count):
    if smile_string.find(f']{count+1}[') != -1:
        count += 1
        return count_cycle(smile_string, count)
    else:
        return count

def count_ring_heteroatom(smile_string):
    mol = Chem.MolFromSmiles(smile_string)
    rings_list_set = [set(atom) for atom in mol.GetRingInfo().AtomRings()
                      if all([mol.GetAtomWithIdx(index).GetAtomMapNum() for index in atom
                              if mol.GetAtomWithIdx(index).GetAtomMapNum() != 0])]
    set_1 = set()
    for j in rings_list_set:
        set_1 |= j
    list_ring_atom = list(set_1)
    count = 0
    for index in list_ring_atom:
        atom_num= mol.GetAtomWithIdx(index).GetAtomicNum()
        if atom_num != 6:
            count += 1
    return count


def pattern_search(smile_string, pattern_list):

    res = any([all([smile_string.GetAtomWithIdx(index).GetAtomMapNum() for index in
                    tuple if smile_string.GetAtomWithIdx(index).GetAtomMapNum() != 0])
               for tuple in smile_string.GetSubstructMatches(Chem.MolFromSmarts(pattern))]
              for pattern in pattern_list)
    return res

def count_substituens(smile_string):
    mol = Chem.MolFromSmiles(smile_string)
    rings_list_set = [set(atom) for atom in mol.GetRingInfo().AtomRings()
                      if all([mol.GetAtomWithIdx(index).GetAtomMapNum() for index in atom
                              if mol.GetAtomWithIdx(index).GetAtomMapNum() != 0])]
    set_1 = set()
    for j in rings_list_set:
        set_1 |= j
    list_ring_atom = list(set_1)
    set_substituent = set()
    for index in list_ring_atom:
        atom = mol.GetAtomWithIdx(index)

        for bond in atom.GetNeighbors()[-1].GetBonds():
            if not bond.IsInRing():
                set_substituent.add(mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomMapNum())
    return len(set_substituent)



data = pd.read_excel('Conversion_Dataset_full.xlsx')


data['reactants_list'] = [i.split(', ') for i in data['reactants']]
data["max_percent_atom"] = [max_percent_atom(smiles, reactants_list) for smiles, reactants_list in zip(data["smiles"], data["reactants_list"])]
data['Suzuki'] = [1 if any([pattern_search(Chem.MolFromSmiles(j), dict_of_pattern["Suzuki"]) for j in i]) else 0 for i in data['reactants_list']]
data['Buchwald_Hartwig'] = [1 if any([pattern_search(Chem.MolFromSmiles(j), dict_of_pattern["Buchwald_Hartwig"]) for j in i]) else 0 for i in data['reactants_list']]
data['max_count_cycles'] = [max([count_cycle(j, 0) for j in i]) for i in data['reactants_list']]
data['count_multycycles'] = [len([count_cycle(j, 0) for j in i if count_cycle(j, 0) >= 2]) for i in data['reactants_list']]
data['max_count_fused_cycles'] = [max([max_count_fused_cycle(j) for j in i]) for i in data['reactants_list']]
data['Michael'] = [1 if any([pattern_search(Chem.MolFromSmiles(j), dict_of_pattern["Michael"]) for j in i]) else 0 for i in data['reactants_list']]
data['Wittig'] = [1 if any([pattern_search(Chem.MolFromSmiles(j), dict_of_pattern["Wittig"]) for j in i]) else 0 for i in data['reactants_list']]
data['Gringard'] = [1 if any([pattern_search(Chem.MolFromSmiles(j), dict_of_pattern["Grignard"]) for j in i]) else 0 for i in data['reactants_list']]
data['Acylation'] = [1 if any([pattern_search(Chem.MolFromSmiles(j), dict_of_pattern["Acylation_ON"]) for j in i]) else 0 for i in data['reactants_list']]
data['Fridel'] = [1 if any([pattern_search(Chem.MolFromSmiles(j), dict_of_pattern["Fridel"]) for j in i]) else 0 for i in data['reactants_list']]
data['count_substituens'] = [max([count_substituens(j) for j in i]) for i in data['reactants_list']]
data['count_ring_heteroatom'] = [max([count_ring_heteroatom(j) for j in i]) for i in data['reactants_list']]
data['count_presence'] = [ sum([a, b, c, d, e, f, g]) for a, b, c, d, e, f, g in
    zip(data['Fridel'], data['Acylation'], data['Gringard'],data['Wittig'],
        data['Michael'], data['Buchwald_Hartwig'], data['Suzuki'])]



features = data.drop(['reactants_list', 'reactants', 'smiles', 'name', 'Fridel', 'Acylation', 'Gringard','Wittig',
        'Michael', 'Buchwald_Hartwig', 'Suzuki'], axis=1)

X = features.drop(['conversion'], axis=1)

features['conversion'] = features['conversion'].apply(lambda x: 0 if x == 0 or x == 1 else 1)
y = features['conversion']
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y,
                                                    test_size=0.2, random_state=42)


rf = RandomForestClassifier(criterion='entropy', random_state=0)
parameters = {'n_estimators': range(1, 5), 'max_depth': range(1, 13, 2),
              'min_samples_leaf': range(1, 7), 'min_samples_split': range(2, 10, 9)}
grid_search_cv_clf = GridSearchCV(rf, param_grid=parameters, cv=6, n_jobs=-1)
grid_search_cv_clf.fit(X_train, y_train)
best_decision = grid_search_cv_clf.best_estimator_

response = best_decision.predict(X_val)
print(response)
print(roc_auc_score(y_val, response))

importance = best_decision.feature_importances_
name_feature = list(X.columns)
print(pd.DataFrame({'name_feature':name_feature, 'importance':importance}))
