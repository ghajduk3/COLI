import logging

logger = logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.DEBUG)
from utils import pipeline,classifier_manage

if __name__ == '__main__':
    ds = pipeline.load_labeled_datasets(dataset_number=(1,1))
    x,y = pipeline.run_dataset_preparation(ds)
    # model,vectorizer,x_test,y_true = pipeline.train_and_split('XGB','tfidf',x,y)
    # classifier_manage.save_classifier(model,'SVM',vectorizer)
    # y_test = pipeline.transform_and_predict(model,vectorizer,x_test)

    f1,acc = pipeline.evaluate_cross_validation('LOGISTIC REGRESSION','tfidf',x,y)
    print(f1,acc)
    # report = pipeline.evaluate(y_true,y_test,target_names = ['no-hate','hate'])
    # print(report)
    # pipeline.explore()

