import logging
import sys

import argparse
logger = logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.DEBUG)
from utils import pipeline, utilities, validation
from classifiers import bert




def process_args():
    parser = argparse.ArgumentParser(description="Cross-lingual offensive language classifier project by Simon Dimc and Gojko Hajduković.")
    parser.add_argument('-pd', '--prepareData', required=True, default='false', type=str, help="Prepare data. Can be done just once at the beginning.")
    parser.add_argument('-t', '--type', required=True, default='bi', type=str, help="Specifies type of classification task to be performed. It is either binary task - bi or multiclass task - multi, the default one is binary - bi. ")
    parser.add_argument('-m', '--model', required=True, default='LR', type=str, help="Specifies the model that is used for classification. Available models are : BERT, LR, SVM, XGBOOST. LR, SVM and XGBOOST perform training and evaluation in-place for the english data defined. BERT model uses pre-trained model for classification trained on english data and performs evaluation on English and Slovene data.")
    return parser.parse_args()


def validate_args():
    args = process_args()
    prepareData = args.prepareData
    type = args.type
    model = args.model
    if not validation.validate_type_argument(type):
        raise SystemExit("Type argument is invalid! Valid types are bi or multi.")
    if not validation.validate_model_argument(model):
        raise SystemExit("Model argument is invalid! Valid models are BERT, LR, SVM, XGBOOST.")
    if not validation.validate_prepareData_argument(prepareData):
        raise SystemExit("PrepareData argument is invalid! Valid models are true, false.")
    return prepareData, type, model


def main():
    prepareData, type, model= validate_args()

    if prepareData == 'true':
        pipeline.prepare_labeled_datasets("eng")
        pipeline.prepare_labeled_datasets("slo")

        pipeline.create_final_datasets("eng")
        pipeline.create_final_datasets("slo")

    if type == 'bi':
        ds_eng = pipeline.load_binary_datasets("eng")
        ds_slov = pipeline.load_binary_datasets("slo")
    else:
        ds_eng = pipeline.load_multiclass_datasets("eng")
        ds_slov = pipeline.load_multiclass_datasets("slo")

    if model == 'BERT':
        x_eng,y_eng =  pipeline.run_dataset_preparation(ds_eng, "eng", remove_stopwords=False, do_lemmatization=False)
        x_slov, y_slov = pipeline.run_dataset_preparation(ds_slov, "slo", remove_stopwords=False, do_lemmatization=False)

        pipeline.run_bert_experiment(x_eng, y_eng, x_slov, y_slov, type)
    else:
        x_eng,y_eng =  pipeline.run_dataset_preparation(ds_eng, "eng", remove_stopwords=True, do_lemmatization=True)
        x_slov, y_slov = pipeline.run_dataset_preparation(ds_slov, "slo", remove_stopwords=True, do_lemmatization=True)

        pipeline.evaluate_cross_validation(model, 'tfidf', x_eng, y_eng)


if __name__ == '__main__':
    main()
    """
    pipeline.prepare_labeled_datasets("eng")
    pipeline.prepare_labeled_datasets("slo")

    pipeline.create_final_datasets("eng")
    pipeline.create_final_datasets("slo")

    ds = pipeline.load_binary_datasets("slo")
    x, y = pipeline.run_dataset_preparation(ds, "slo", remove_stopwords=True, do_lemmatization=True)

    model = bert.setup_classifier(
        model_name = "classifiers/bert/CroSloEngual",
        num_labels = 2
    )

    model.load_state_dict(bert.load_model("models/bert/binary.pt"))

    dataset = bert.setup_data(
        model_name = "classifiers/bert/CroSloEngual",
        x = x,
        y = y,
        do_lower_case = False,
        max_length = 180
    )

   
    # model, stats = bert.train_classifier(
    #     model = model,
    #     dataset = dataset,
    #     validation_ratio = 0.9,
    #     batch_size = 32,
    #     freeze_embeddings_layer = True,
    #     freeze_encoder_layers = 8,
    #     epochs = 1
    # )
    

    predictions, true_labels = bert.test_classifier(
        model = model,
        dataset = dataset,
        batch_size = 32
    )

    utilities.print_performance_metrics(predictions, true_labels)

    bert.save_model("models/m1.pt", model)
    
    """

    """
    ds = pipeline.load_labeled_datasets(dataset_number=(1,4), type="binary")
    x,y = pipeline.run_dataset_preparation(ds)
    model,vectorizer,topic_model_dict,x_test,y_true = pipeline.train_and_split('LR','tfidf',x,y)
    # print(*topic_model_dict)
    # # print(y_true.value_counts())
    # # classifier_manage.save_classifier(model,'SVM',vectorizer)
    # y_test = pipeline.transform_and_predict(model,vectorizer,topic_model_dict,x_test)
    f1,acc = pipeline.evaluate_cross_validation('LR','tfidf',x,y)
    print(f1,acc)
    # report = pipeline.evaluate(y_true,y_test,target_names = ['0','1'])
    # print(report)
    # pipeline.explore()
    """






