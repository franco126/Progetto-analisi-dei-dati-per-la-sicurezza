import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from sklearn.feature_selection import mutual_info_classif

"""     COSTANTI    """
PATH_TRAINING_SET = "C:/Users/cloud/Desktop/ProgettoAnalisiDati/Train_OneClsNumeric.csv"
PATH_TESTING_SET = "C:/Users/cloud/Desktop/ProgettoAnalisiDati/Test_OneClsNumeric.csv"
BOXPLOT_PATH = "BoxPlots/"
SCATTERPLOT_PATH = "ScatterPlots/"
STATISTICS_FILE_NAME = 'FeatureStatistics.csv'

LABEL_NAME = 'classification'

NUMBER_OF_FEATURES = 10

NUMBER_OF_FOLDS = 10
RANDOM_STATE_VALUE = 35

#Il numero minimo di campioni richiesti per dividere un nodo interno
MIN_NUMBER_OF_SAMPLES_TO_SPLIT = 0.05

RANGE_ESTIMATORS = [10, 20, 30]
RANGE_MAX_FEATURES = ["sqrt", "log2"]
RANGE_MAX_SAMPLES = [0.5, 0.6, 0.7, 0.8, 0.9]


def load_data_from_csv(path):
    """
        Carica i dati da un file CSV in base al path indicato.
        :param path: percorso del CSV da caricare
        :return: dataset dei dati (oggetto pandas dataframe)
    """
    data = pd.read_csv(path, low_memory=False)
    return data


def preElaboration(data, list_of_features):
    """
        Fornisce le statistiche degli attributi del dataset
        I valori min,max, avg ecc sono salvati su un file csv
        :param data: dataset
        :param list_of_features: elenco degli attributi
    """
    values = []
    for feature in list_of_features:
        values.append(data[feature].describe())
    statistics = pd.DataFrame(values)
    statistics.to_csv(STATISTICS_FILE_NAME, sep=";")


def preElaborationBox(data, independent_list):
    """
        Crea un box plot per ciascun attributo elencato nella lista raggrupati per classe
        Il box plot è salvato come immagine
        :param data: dataset
        :param independent_list: elenco degli attributi
    """
    path_file = BOXPLOT_PATH

    if not os.path.exists(path_file):
        os.makedirs(path_file)

    for feature in independent_list:
        data.boxplot(column=[feature], by=LABEL_NAME)  # Boxplot della colonna confrontato con la classe di appartenenza
        plt.savefig(path_file + feature + '.png')
        plt.close()


def preElaborationScatter(data, independent_list):
    """
        Crea uno scatter plot per ciascun attributo elencato nella lista rispetto la classe
        Lo scatter plot è salvato come immagine
        :param data: dataset
        :param independent_list: elenco degli attributi
    """
    path_file = SCATTERPLOT_PATH

    if not os.path.exists(path_file):
        os.makedirs(path_file)

    for feature in independent_list:
        data.plot.scatter(x=feature, y=LABEL_NAME, title=feature)  # Scatterplot della singola colonna
        plt.savefig(path_file + feature + '.png')
        plt.close()


def stratified_cross_validation(x_data, y_data):
    """
        Esegue la stratified cross validation in base al numero fold specificato
        Lo shuffle è abilitato. Gli esempi sono quindi mescolati casualmente
        Per controllare il generatore di numeri casuali si specifica un valore per il parametro random_state.
        :param x_data: dataset sui cui effettuare il partizionamento
        :param y_data: colonna della classe
        :return: 4 liste di dataframe pandas, di cui 2 di train e 2 di test
    """
    list_x_train = []
    list_x_test = []
    list_y_train = []
    list_y_test = []

    cross_validation = StratifiedKFold(n_splits=NUMBER_OF_FOLDS, shuffle=True, random_state=RANDOM_STATE_VALUE)
    # genera gli indici per dividere il dataset in training set e validation set
    split_indices = cross_validation.split(x_data, y_data)

    for train_index, test_index in split_indices:
        x_train, x_test = x_data.iloc[train_index], x_data.iloc[test_index]
        y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]

        list_x_train.append(x_train)
        list_x_test.append(x_test)
        list_y_train.append(y_train)
        list_y_test.append(y_test)

    return list_x_train, list_x_test, list_y_train, list_y_test


def learn_randomForest(x_data, y_data, number_of_estimators, max_number_features, max_number_samples):
    """
        APPRENDE UNA RANDOM FOREST SU UN PARTICOLARE TRANING SET DI INPUT
        :param x_data: dataset su cui apprendere la random forest
        :param y_data: colonna della classe
        :param number_of_estimators: numero di alberi da apprendere
        :param  max_number_features: numero di attributi da considerare quando si cerca lo split migliore
                (radice quadrata o log in base del 2 del numero totale di attributi)
        :param  max_number_samples: con bootstrap = true, indica il numero di esempi da estrarre dal dataset
                per costruire ogni bag
        :return: random forest addestrata sul dataset in input secondo i parametri specificati
        bootstrap indica che il sampling sugli esempi per costruire un bag avviene con rimessa
    """
    rf_classifier = RandomForestClassifier(n_estimators=number_of_estimators, min_samples_split=MIN_NUMBER_OF_SAMPLES_TO_SPLIT,
                                           max_features=max_number_features, bootstrap=True, max_samples=max_number_samples)
    rf_classifier.fit(x_data, y_data)
    return rf_classifier


def evaluate_classifier(x_data, y_true, clf):
    """
    Effettua il calcolo delle metriche per la valutazione del classificatore in input
    :param x_data: dataset di input
    :param y_true: colonna della classe
    :param clf: classificatore appreso
    :return: dizionario contenente tutte le metriche di valutazione delle predizioni effettuate dal classificatore
    """
    metrics = {}
    y_predicted = clf.predict(x_data)

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import balanced_accuracy_score

    overall_accuracy = accuracy_score(y_true, y_predicted)
    balance_accuracy = balanced_accuracy_score(y_true, y_predicted)
    precision = precision_score(y_true, y_predicted)
    recall = recall_score(y_true, y_predicted)
    f_score = f1_score(y_true, y_predicted)

    metrics["OVERALL_ACCURACY"] = overall_accuracy
    metrics["BALANCE_ACCURACY"] = balance_accuracy
    metrics["PRECISION"] = precision
    metrics["RECALL"] = recall
    metrics["F_SCORE"] = f_score

    return metrics


def evaluate_randomforest_CV(list_x_train, list_x_test, list_y_train, list_y_test,
                             number_of_estimators, max_number_features,max_number_samples ):
    """
    Calcola le metriche di valutazione del classificatore per ogni fold e ne restituisce la media, al fine di stabilire
    la migliore configurazione da utilizzare
    :param list_x_train: porzione di dataset ottenuta tramite cross validation per l'addestramento del classificatore
    :param list_x_test: porzione di dataset ottenuta tramite cross validation per la valutazione del classificatore
    :param list_y_train: porzione di dataset, contenente solo la classe, ottenuta tramite corss validation per l'addestramento del classificatore
    :param list_y_test: porzione di dataset, contenente solo la classe, ottenuta tramite cross validation per la valutazione del classificatore
    :param number_of_estimators:
    :param max_number_features:
    :param max_number_samples:
    :return: dizionario contenente la valutazione media sui fold
    """
    avgTest = {}
    avgTest["OVERALL_ACCURACY"] = 0.0
    avgTest["BALANCE_ACCURACY"] = 0.0
    avgTest["PRECISION"] = 0.0
    avgTest["RECALL"] = 0.0
    avgTest["F_SCORE"] = 0.0

    for i in range(NUMBER_OF_FOLDS):

        x_training_data = list_x_train[i]
        y_training_data = list_y_train[i]
        rf_classifier = learn_randomForest(x_training_data,y_training_data,
                                           number_of_estimators,max_number_features,max_number_samples)

        x_testing_data = list_x_test[i]
        y_testing_data = list_y_test[i]
        metrics = evaluate_classifier(x_testing_data,y_testing_data,rf_classifier)

        for key in avgTest.keys():
            avgTest[key]=avgTest[key]+metrics[key]

    for key in avgTest.keys():
        avgTest[key]=avgTest[key]/NUMBER_OF_FOLDS

    return avgTest


def best_randomforest_configuration(x_data, y_data):
    """
    Calcola i migliori parametri per l'addestramento della random forest
    :param x_data: dataset costituito dalle variabili indipendenti
    :param y_data: colonna classe del dataset
    :return: dizionario contenente i migliori parametri per l'addestramento
    """
    max_fscore = 0.0
    best_configuration = {}
    best_configuration["N_ESTIMATORS"] = 0
    best_configuration["MAX_FEATURES"] = ""
    best_configuration["MAX_SAMPLES"] = 0

    list_x_train, list_x_test, list_y_train, list_y_test = stratified_cross_validation(x_data, y_data)

    for i in range(len(RANGE_ESTIMATORS)):
        for j in range (len(RANGE_MAX_SAMPLES)):
            for k in range (len(RANGE_MAX_FEATURES)):
                avgTest = evaluate_randomforest_CV(list_x_train, list_x_test, list_y_train, list_y_test,
                                                   RANGE_ESTIMATORS[i], RANGE_MAX_FEATURES[k], RANGE_MAX_SAMPLES[j])
                if ( avgTest["F_SCORE"] > max_fscore ):
                    max_fscore = avgTest["F_SCORE"]
                    best_configuration["N_ESTIMATORS"] = RANGE_ESTIMATORS[i]
                    best_configuration["MAX_FEATURES"] = RANGE_MAX_FEATURES[k]
                    best_configuration["MAX_SAMPLES"] = RANGE_MAX_SAMPLES[j]

    return best_configuration


def pca(x_data):
    """
    Addestra la PCA sul dataset in input (senza colonna della classe)
    :param x_data: dataset in input
    :return: PCA addestrata sul dataset e lista dei nomi delle colonne del nuovo dataset costituito dalle componenti principali
    """
    pca_analysis = PCA(n_components=NUMBER_OF_FEATURES)
    columns_list = ["PCA" + str(index) for index in range(1, NUMBER_OF_FEATURES + 1)]
    return pca_analysis.fit(x_data), columns_list


def applyPCA(pca_analysis, x_data, pca_columns_list):
    """
    Applica la PCA addestrata sul dataset in input
    :param pca_analysis: pca addestrata
    :param x_data: dataset
    :param pca_columns_list: lista dei nomi delle colonne del nuovo dataset costituto dalle componenti principali
    :return: un nuovo dataset formato dalle sole componenti principali
    """
    data = pca_analysis.transform(x_data)
    pca_dataset = pd.DataFrame(data, columns=pca_columns_list)
    return pca_dataset

def applyPCA_on_CV(x_train, x_test):
    """
    Addestra e applica la PCA sui vari fold della cross validation
    :param x_train: partizione di dataset con le sole variabili indipendenti, ottenuta tramite cross valdiation, da utilizzare per l'addestramento
    :param x_test: partizione di dataset con le sole variabili indipendenti, ottenuta tramite cross valdiation, da utilizzare per la validazione
    :return: lista contenenti le PCA su ogni partizione
    """
    x_train_pca = []
    x_test_pca = []
    for i in range(0, len(x_train)):
        pca_value, pca_columns = pca(x_train[i])
        pca_train_dataset = applyPCA(pca_value, x_train[i], pca_columns)
        pca_test_dataset = applyPCA(pca_value, x_test[i], pca_columns)
        x_train_pca.append(pca_train_dataset)
        x_test_pca.append(pca_test_dataset)

    return x_train_pca, x_test_pca


def best_randomforest_configuration_for_PCA(x_data, y_data):
    """
    Calcola la migliore configurazione per la random forrest da addestrare sulla PCA
    :param x_data: dataset di input, con le sole variabili indipendenti
    :param y_data: colonna classe del dataset
    :return: dizionario contenente i parametri per la migliore configurazione
    """
    max_fscore = 0.0
    best_configuration = {}
    best_configuration["N_ESTIMATORS"] = 0
    best_configuration["MAX_FEATURES"] = ""
    best_configuration["MAX_SAMPLES"] = 0

    list_x_train, list_x_test, list_y_train, list_y_test = stratified_cross_validation(x_data, y_data)
    list_x_train, list_x_test = applyPCA_on_CV(list_x_train, list_x_test)

    for i in range(len(RANGE_ESTIMATORS)):
        for j in range(len(RANGE_MAX_SAMPLES)):
            for k in range(len(RANGE_MAX_FEATURES)):
                avgTest = evaluate_randomforest_CV(list_x_train, list_x_test, list_y_train, list_y_test,
                                                   RANGE_ESTIMATORS[i], RANGE_MAX_FEATURES[k], RANGE_MAX_SAMPLES[j])
                if (avgTest["F_SCORE"] > max_fscore):
                    max_fscore = avgTest["F_SCORE"]
                    best_configuration["N_ESTIMATORS"] = RANGE_ESTIMATORS[i]
                    best_configuration["MAX_FEATURES"] = RANGE_MAX_FEATURES[k]
                    best_configuration["MAX_SAMPLES"] = RANGE_MAX_SAMPLES[j]

    return best_configuration


def evaluate_knn_CV(list_x_train, list_x_test, list_y_train, list_y_test, number_of_neighbours):
    """
    Calcola le metriche di valutazione per il knn su ogni fold e restituisce la media delle valutazioni
    :param list_x_train: partizione di dataset con le sole variabili indipendenti, ottenuta tramite cross valdiation, da utilizzare per l'addestramento
    :param list_x_test: partizione di dataset con le sole variabili indipendenti, ottenuta tramite cross valdiation, da utilizzare per la valutazione
    :param list_y_train: partizione di dataset con la sola classe, ottenuta tramite cross valdiation, da utilizzare per l'addestramento
    :param list_y_test:  partizione di dataset con la sola classe, ottenuta tramite cross valdiation, da utilizzare per la valutazione
    :param number_of_neighbours: numero di vicini, parametro per il classificatore knn
    :return: media delle valutazioni
    """
    avgTest = {}
    avgTest["OVERALL_ACCURACY"] = 0.0
    avgTest["BALANCE_ACCURACY"] = 0.0
    avgTest["PRECISION"] = 0.0
    avgTest["RECALL"] = 0.0
    avgTest["F_SCORE"] = 0.0

    for i in range(NUMBER_OF_FOLDS):

        x_training_data = list_x_train[i]
        y_training_data = list_y_train[i]
        knn_classifier = learn_knn(x_training_data, y_training_data, number_of_neighbours)

        x_testing_data = list_x_test[i]
        y_testing_data = list_y_test[i]
        metrics = evaluate_classifier(x_testing_data, y_testing_data, knn_classifier)

        for key in avgTest.keys():
            avgTest[key] = avgTest[key] + metrics[key]

    for key in avgTest.keys():
        avgTest[key] = avgTest[key] / NUMBER_OF_FOLDS

    return avgTest


def best_knn_configuration(x_data, y_data):
    """
    Calcola il miglior numero di vicini da utilizzare per il classificatore KNN
    :param x_data: dataset con le sole variabili indipendenti
    :param y_data: colonna classe del dataset
    :return: miglior numero di vicini
    """
    max_fscore = 0.0
    best_number_of_neighbours = 0

    list_x_train, list_x_test, list_y_train, list_y_test = stratified_cross_validation(x_data, y_data)

    for i in range(1,101):
        avgTest = evaluate_knn_CV(list_x_train, list_x_test, list_y_train, list_y_test, i)
        #print("Numero vicini= ",i,"\nValutazione\n",avgTest,"\n")
        if (avgTest["F_SCORE"] > max_fscore):
            max_fscore = avgTest["F_SCORE"]
            best_number_of_neighbours = i

    return best_number_of_neighbours


def learn_knn(x_data, y_data, number_of_neighbours):
    """
    Addestra il classificatore k nearest neighbors sul dataset in input
    :param x_data: dataset in input formato dalle sole variabili indipendenti
    :param y_data: colonna classe del dataset
    :param number_of_neighbours: numero di vicini, parametro per il classifcatore
    :return: classificatore knn addestrato sul dataset
    """
    knn = KNeighborsClassifier(n_neighbors=number_of_neighbours, metric="euclidean")
    knn.fit(x_data, y_data)

    return knn


def mutual_info_rank(data, features_list, label):
    """
    Calcola la mutualInfoClassif sulla lista delle variabili indipendenti
    :param data: dataset su cui effettuare il calcolo
    :param features_list: lista nomi variabili indipendenti (attributi)
    :param label: nome della classe
    :return: lista di tuple (coppie nome/valore) ordinate in modo decrescente per mutual info
    """
    res = dict(zip(features_list, mutual_info_classif(data[features_list], data[label], discrete_features=False)))
    sorted_x = sorted(res.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_x


def top_attribute_selection ( sorted_features_list, number_of_top_features ):
    """
    metodo utilizzato per estrarre il nome dei primi n attributi dal dataset ordinato in base alla mutual info
    :param data:
    :param sorted_features_list:
    :param number_of_top_features:
    :return:
    """
    dictionary = dict(sorted_features_list)
    topAttributeList = []
    for attribute in dictionary.keys():
        topAttributeList.append(attribute)
        if (len(topAttributeList) == number_of_top_features):
            break
    # all'elenco dei migliori n attributi aggiungo il nome dell'attributo classe
    #topAttributeList.append(data.columns.values[data.shape[1] - 1])
    return topAttributeList


def feature_selection (data, topAttributeList):
    """
    crea un nuovo data frame selezionando le top (10) feature indipendenti ottenute dalla mutual info
    Seleziona un determinato numero di feature dopo il calcolo del rispettivo Info Gain
    Si presuppone che la lista di feature sia ordinata in modo decrescente
    :param data:  dataset su cui effettuare la feature selection
    :param topAttributeList: elenco dei primi n attributi orindati in modo decrescente per mutual info
    :return: nuovo dataset
    """
    df = data.loc[:,topAttributeList] #primo parametro: righe secondo parametro: lista di coloonne
    df[LABEL_NAME] = data[LABEL_NAME]
    return df

if __name__ == '__main__':

    training_set = load_data_from_csv(PATH_TRAINING_SET)
    """
        PRE ELABORATION
    """
    features_list = training_set.columns[0:training_set.shape[1]-1].values
    #print("\nLista degli attributi\n",features_list,"\n")

    preElaboration(training_set, features_list)
    preElaborationBox(training_set, features_list)
    preElaborationScatter(training_set, features_list)

    sorted_features_list = mutual_info_rank( training_set, features_list, LABEL_NAME )
    print("ATTRIBUTI ORDINATI PER IMPORTANZA\n", sorted_features_list,"\n")

    number_of_top_features = 10
    top_attribute_list = top_attribute_selection(sorted_features_list, number_of_top_features)
    print("I migliori ", number_of_top_features," attributi sono\n", top_attribute_list,"\n")

    dataset_of_top_features = feature_selection( training_set, top_attribute_list)
    print("Feature selection sul Training Set\n", dataset_of_top_features,"\n")

    """ Addestramento della random forest sull'intero dataset
        Scelta della configurazione migliore tramite stratified cross validation    """

    number_of_features = training_set.shape[1]
    independent_features = training_set.iloc[:, 0:number_of_features - 1]
    dependent_features = training_set.iloc[:, number_of_features - 1]

    """ INDIVIDUAZIONE DEI MIGLIORI PARAMETRI PER LA RANDOM FOREST  """
    randomforest_configuration = best_randomforest_configuration(independent_features, dependent_features)
    print("Migliore configurazione Random Forest su intero Training Set\n",randomforest_configuration)

    """ APPRENDO LA RANDOM FOREST CON LA MIGLIORE CONFIGURAZIONE SUL TRAINING SET   """
    number_of_estimators = randomforest_configuration["N_ESTIMATORS"]
    max_number_features = randomforest_configuration["MAX_FEATURES"]
    max_number_samples = randomforest_configuration["MAX_SAMPLES"]

    #Apprendo random forest con i migliori parametri su training test
    randomforest = learn_randomForest(independent_features,dependent_features,number_of_estimators, max_number_features, max_number_samples)


    """ Addestramento della random forest sulle TOP 10 Features """
    pca_analysis, pca_columns_list = pca(independent_features)
    pca_dataset = applyPCA(pca_analysis, independent_features, pca_columns_list)
    pca_dataset[LABEL_NAME] = training_set[LABEL_NAME]
    #print("\nPCA DATASET\n", pca_dataset,"\n")

    pca_independent_features = pca_dataset.iloc[:, 0:pca_dataset.shape[1]-1]
    pca_dependent_features = pca_dataset.iloc[:, pca_dataset.shape[1]-1]

    pca_randomforest_configuration = best_randomforest_configuration_for_PCA(independent_features,dependent_features)
    print("\nMigliore configurazione Random Forest costruita dalle 10 top componenti principali\n", pca_randomforest_configuration)

    pca_number_of_estimators = pca_randomforest_configuration["N_ESTIMATORS"]
    pca_max_number_features = pca_randomforest_configuration["MAX_FEATURES"]
    pca_max_number_samples = pca_randomforest_configuration["MAX_SAMPLES"]

    # Apprendo random forest con i migliori parametri sulle 10 top componenti principali
    pca_randomforest = learn_randomForest( pca_independent_features, pca_dependent_features, pca_number_of_estimators, pca_max_number_features, pca_max_number_samples)


    """ STACKER """
    """ Creo dataset formato dalle classi predette dalla random forest appresa su intero training set e random forest 
    appresa sulle top 10 features """
    stacker_dataset = pd.DataFrame(columns=["P1", "P2"])
    stacker_dataset["P1"] = pd.Series( randomforest.predict( independent_features ) )
    stacker_dataset["P2"] = pd.Series ( pca_randomforest.predict( pca_independent_features ) )
    stacker_dataset[LABEL_NAME] = training_set[LABEL_NAME]
    #print("\nDataset costituito dalle classi predette\n", stacker_dataset,"\n")

    stacker_independent_features = stacker_dataset.iloc[:, 0: stacker_dataset.shape[1]-1 ]
    stacker_dependent_features = stacker_dataset.iloc[:, stacker_dataset.shape[1]-1 ]

    best_neighbours = best_knn_configuration(stacker_independent_features, stacker_dependent_features)
    print("\nMiglior numero di vicini per l'algoritmo KNN: ", best_neighbours)

    #Apprendo knn con il miglior numero di vicini
    knn_classifier = learn_knn(stacker_independent_features, stacker_dependent_features, best_neighbours)


    """ VALUTAZIONE DEI PATTERN APPRESI """
    testing_set = load_data_from_csv(PATH_TESTING_SET)
    number_of_features = testing_set.shape[1]
    independent_features = testing_set.iloc[:, 0:number_of_features-1]
    dependent_features = testing_set.iloc[:, number_of_features-1]

    """ VALUTAZIONE RANDOM FOREST ADDESTRATA SULL'INTERO TRAINING SET """
    metrics = evaluate_classifier(independent_features,dependent_features,randomforest)
    print("\nVALUTAZIONE DELLA RANDOM FOREST APPRESA SULL'INTERO TRAINING SET\n",metrics)


    """ VALUTAZIONE RANDOM FOREST ADDESTRATA SULLE (10) TOP COMPONENTI PRINCIPALI SU TESTING SET """
    #-- PCA calcolata per il Testing Set
    #pca_testing_set, pca_testing_set_columns = pca ( independent_features )
    #pca_testing_dataset = applyPCA(pca_testing_set, independent_features, pca_testing_set_columns)
    #--
    # PCA calcolata sul Training Set applicata sul Testing Set
    pca_testing_dataset = applyPCA(pca_analysis, independent_features, pca_columns_list)
    pca_testing_dataset[LABEL_NAME] = testing_set[LABEL_NAME]
    #print("\nPCA TESTING SET\n", pca_testing_dataset,"\n")
    """ Valutazione """
    pca_independent_features = pca_testing_dataset.iloc[:, 0:pca_testing_dataset.shape[1] - 1]
    pca_dependent_features = pca_testing_dataset.iloc[:, pca_testing_dataset.shape[1] - 1]
    pca_metrics = evaluate_classifier( pca_independent_features, pca_dependent_features, pca_randomforest )
    print("\nVALUTAZIONE DELLA RANDOM FOREST APPRESA SULLE TOP 10 FEATURE SUL TRAINING SET\n", pca_metrics)


    """ VALUTAZIONE DEL PATTERN STACKER APPRESO SU TESTING SET"""
    class_predicted_one = randomforest.predict(independent_features)
    class_predicted_two = pca_randomforest.predict (pca_independent_features)

    testing_stacker_dataset = pd.DataFrame(columns=["P1", "P2"])
    testing_stacker_dataset["P1"] = pd.Series(class_predicted_one)
    testing_stacker_dataset["P2"] = pd.Series(class_predicted_two)
    testing_stacker_dataset[LABEL_NAME] = testing_set[LABEL_NAME]
    #print("\nSTACKER DATASET COSTRUITO CON LE PREDIZIONI SUL TESTING SET\n",testing_stacker_dataset,"\n")

    stacker_independent_features = testing_stacker_dataset.iloc[:, 0: testing_stacker_dataset.shape[1]-1 ]
    stacker_dependent_features = testing_stacker_dataset.iloc[:, testing_stacker_dataset.shape[1]-1 ]
    stacker_metrics = evaluate_classifier(stacker_independent_features, stacker_dependent_features, knn_classifier )
    print("\nVALUTAZIONE STACKER TRAMITE KNN SUL TESTING SET\n", stacker_metrics)

    quit()
