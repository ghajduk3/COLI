import logging

logger = logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.DEBUG)
from utils import pipeline
from classifiers import bert

if __name__ == '__main__':
    # pipeline.prepare_labeled_datasets()
    # ds = pipeline.load_labeled_datasets(dataset_number=(1,4))
    pipeline.combine_multiclass_datasets()
    # pipeline.combine_binary_datasets()
    ds = pipeline.load_multiclass_datasets()
    x, y = pipeline.run_dataset_preparation(ds)
    # new_df = pd.DataFrame(x).join(y)
    # utils.utilities.write_to_file_pd(new_df,os.path.join(rootpath.detect(), 'data', 'temp', 'eng', 'binary', 'data.csv'))



    model = bert.setup_classifier(
        model_name = "classifiers/bert/CroSloEngual",
        num_labels = 2
    )

    # model.load_state_dict(bert.load_model("models/m1.pt"))

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

    bert.save_model("models/m1.pt", model)

    """
    ds = pipeline.load_labeled_datasets(dataset_number=(6,6), type="multiclass")
    x,y = pipeline.run_dataset_preparation(ds)
    # model,vectorizer,x_test,y_true = pipeline.train_and_split('SVM','tfidf',x,y)
    # # print(y_true.value_counts())
    # # classifier_manage.save_classifier(model,'SVM',vectorizer)
    # y_test = pipeline.transform_and_predict(model,vectorizer,x_test)
    f1,acc = pipeline.evaluate_cross_validation('SVM','tfidf',x,y)
    print(f1,acc)
    # report = pipeline.evaluate(y_true,y_test,target_names = ['0','1' , '2','3', '4','5'])
    # print(report)
    # pipeline.explore()
    """





