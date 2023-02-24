import string
import matplotlib.pyplot as plt
import numpy as np
import tadasets
import gudhi as gd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from classical_betti_calc import boundary, homology, betti

characters = string.digits + string.ascii_letters

all_accs = [[], []]
all_noises = np.linspace(0, 5, 30)
for noise in all_noises:
    torus = [
        tadasets.torus(n=len(characters), c=5, a=1, ambient=10, noise=noise)
        for _ in range(50)
    ]
    swiss_roll = [
        tadasets.swiss_roll(n=len(characters), r=5, ambient=10, noise=noise)
        for _ in range(50)
    ]

    all_shapes = []
    all_shapes.extend(torus)
    all_shapes.extend(swiss_roll)

    labels = np.zeros(100)
    labels[50:] = 1

    # tadasets.plot3d(torus[0])
    # plt.show()

    # tadasets.plot3d(swiss_roll[0])
    # plt.show()

    skeletons_2d = [gd.RipsComplex(points=x, max_edge_length=2) for x in all_shapes]
    data_2d_simplex_tree = [
        skeleton.create_simplex_tree(max_dimension=5) for skeleton in skeletons_2d
    ]
    num_ver_simp = [
        [simplex_tree.num_vertices(), simplex_tree.num_simplices()]
        for simplex_tree in data_2d_simplex_tree
    ]
    rips_lists = [
        list(simplex_tree.get_filtration()) for simplex_tree in data_2d_simplex_tree
    ]

    scs = []
    for rips_list in rips_lists:
        sc = []
        for simplex in rips_list:
            temp = ""
            for vertex in simplex[0]:
                temp = temp + characters[vertex]
            sc.append(temp)
        scs.append(sc)

    new_features = []
    skipped = []
    for i, sc in enumerate(scs):
        try:
            bnd, simplicies = boundary(sc)
            H = homology(bnd)
            b = betti(H)
            new_features.append([b[0], b[1]])
        except:
            skipped.append(i)

    new_features = np.array(new_features)
    labels = np.delete(labels, skipped)

    X_train, X_test, y_train, y_test = train_test_split(
        new_features, labels, test_size=0.5, stratify=labels, random_state=0
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_predict_test = model.predict(X_test)
    y_predict_train = model.predict(X_train)

    test_acc = accuracy_score(y_test, y_predict_test)
    train_acc = accuracy_score(y_train, y_predict_train)

    all_accs[0].append(train_acc)
    all_accs[1].append(test_acc)

plt.plot(all_noises, all_accs[0], "*-")
plt.plot(all_noises, all_accs[1], "o-")
plt.legend(["tra", "tes"])
plt.show()
