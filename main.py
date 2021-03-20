import logging

logger = logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.DEBUG)
from utils import pipeline,classifier_manage

# def transform_text(text,indexes):
#     indexes = eval(indexes) if isinstance(indexes,str) else []
#     comment_by_idx = []
#     for index,comment in enumerate(re.split('\d.\s+',text)[1:]):
#         if indexes and index+1 in indexes:
#             comment_by_idx.append([comment,1])
#         else:
#             comment_by_idx.append([comment,0])
#     return comment_by_idx

if __name__ == '__main__':
    ds = pipeline.load_labeled_datasets(dataset_number=(2,2))
    x,y = pipeline.run_dataset_preparation(ds)
    model,vectorizer,x_test,y_true = pipeline.train_and_split('LOGISTIC REGRESSION','tfidf',x,y)
    classifier_manage.save_classifier(model,'SVM',vectorizer)
    y_test = pipeline.transform_and_predict(model,vectorizer,x_test)

    # f1,acc = pipeline.evaluate_cross_validation('LOGISTIC REGRESSION','tfidf',x,y)
    # print(f1,acc)
    report = pipeline.evaluate(y_true,y_test,target_names = ['no-hate','hate'])
    print(report)
    # pipeline.explore()

