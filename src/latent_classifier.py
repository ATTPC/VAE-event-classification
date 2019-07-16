import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import  cross_val_score, train_test_split

def longform_latent(latent,):
    longform_samples = np.zeros((
                        latent.shape[1], 
                        latent.shape[0]*latent.shape[2]
                        ))

    latent_dim = latent.shape[2]

    for i, evts in enumerate(latent):
        longform_samples[:, i*latent_dim:(i+1)*latent_dim] = evts

    return longform_samples


def fit_logreg(X, y,):

    model = LogisticRegression(
            penalty="l2",
            solver="newton-cg",
            multi_class="multinomial",
            class_weight="balanced",
            max_iter=10000,
        )

    model.fit(X, y)
    #train_score = f1_score(y, model.predict(X), average=None)
    #test_score = f1_score(ytest, model.predict(Xtest), average=None)
    return model 

def test_model(X, y, model, sess):
    latent_test = model.run_large(
                                sess,
                                model.z_seq,
                                X,
                                )
    np.save("../drawing/latent/test_latent.npy", latent_test)

    latent_test = np.array(latent_test)
    latent_test = longform_latent(latent_test)

    lr_train, lr_test, lry_train, lry_test = train_test_split(latent_test, y)
    try:
        lr_model = fit_logreg(lr_train, lry_train)
        pred_train = lr_model.predict(lr_train)
        pred_test = lr_model.predict(lr_test)

        train_f1 = f1_score(lry_train, pred_train, average=None)
        test_f1 = f1_score(lry_test, pred_test, average=None)

        train_cm = confusion_matrix(lry_train, pred_train)
        test_cm = confusion_matrix(lry_test, pred_test)

        train_recall = train_cm.diagonal()/train_cm.sum(axis=1)
        test_recall = test_cm.diagonal()/test_cm.sum(axis=1)
        train_precision = train_cm.diagonal()/train_cm.sum(axis=0)
        test_precision = test_cm.diagonal()/test_cm.sum(axis=0)

        train_score = [train_f1, train_recall, train_precision]
        test_score = [test_f1, test_recall, test_precision]

    except ValueError:
        train_score = np.zeros((3, np.unique(np.argmax(y, axis=1))))
        test_score = np.zeros((3, np.unique(np.argmax(y, axis=1))))
    return train_score, test_score
