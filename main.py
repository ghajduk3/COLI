import logging

logger = logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.DEBUG)
from utils import pipeline,classifier_manage

from classifiers import bert

if __name__ == '__main__':

    ds = pipeline.load_labeled_datasets(dataset_number=(1,1))
    x, y = pipeline.run_dataset_preparation(ds)

    model = bert.setup_classifier(
        model_name = "classifiers/bert/CroSloEngual",
        num_labels = 2
    )

    dataset = bert.setup_data(
        model_name = "classifiers/bert/CroSloEngual",
        x = x,
        y = y,
        do_lower_case = False,
        max_length = 180
    )

    model, stats = bert.train_classifier(
        model = model,
        dataset = dataset,
        validation_ratio = 0.9,
        batch_size = 32,
        freeze_embeddings_layer = True,
        freeze_encoder_layers = 8,
        epochs = 1
    )

    predictions, true_labels = bert.test_classifier(
        model = model,
        dataset = dataset,
        batch_size = 32
    )

    """
    ds = pipeline.load_labeled_datasets(dataset_number=(1,1))
    x,y = pipeline.run_dataset_preparation(ds)
    model,vectorizer,x_test,y_true = pipeline.train_and_split('LOGISTIC REGRESSION','tfidf',x,y)
    # classifier_manage.save_classifier(model,'SVM',vectorizer)
    y_test = pipeline.transform_and_predict(model,vectorizer,x_test)

    # f1,acc = pipeline.evaluate_cross_validation('SVM','tfidf',x,y)
    # print(f1,acc)
    report = pipeline.evaluate(y_true,y_test,target_names = ['no-hate','hate'])
    print(report)
    # pipeline.explore()
    """
    

