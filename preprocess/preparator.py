from utils import utilities
import rootpath,os,logging
logger = logging.getLogger(__name__)
class Preparator(object):

    @staticmethod
    def prepare_dataset_1(input_file_path:str):
        """
        Reads the data from dataset_1. Converts it into format Text,Label(1-Hate,0-Non hate) and writes the contents to file.
        """
        labels,data = utilities.read_file_contents(input_file_path, ':')
        labels = [labels[1],labels[0]]
        data = [[row[1],row[0]] for row in data]
        data.insert(0,labels)
        output_path = os.path.join(rootpath.detect(), 'structured_data', 'dataset_1', 'data.csv')
        utilities.write_to_file(data, output_path)
        logger.info("Dataset_1 has been prepared")

    @staticmethod
    def prepare_dataset_2(input_file_path:str):
        """
        Reads the data from dataset_2. Converts it into format Text,Label(1-Hate,0-Non hate) and writes the contents to file.
        """
        data = utilities.read_file_contents_pd(input_file_path)
        data['text'] = data.apply(lambda row: utilities.transform_text(row.text, row.hate_speech_idx), axis=1)
        comments = [item for sublist in data['text'].tolist() for item in sublist]
        comments.insert(0, ['Text', 'Label'])
        output_path = os.path.join(rootpath.detect(), 'structured_data', 'dataset_2', 'data.csv')
        utilities.write_to_file(comments, output_path)
        logger.info("Dataset_2 has been prepared")

    @staticmethod
    def prepare_dataset_3(input_file_path:str):
        """
        Reads the data from dataset_3. Converts it into format Text,Label(1-Hate,0-Non hate) and writes the contents to file.
        """
        data = utilities.read_file_contents_pd(input_file_path)
        data['text'] = data.apply(lambda row: utilities.transform_dataset_2_3(row.text, row.hate_speech_idx), axis=1)
        comments = [item for sublist in data['text'].tolist() for item in sublist]
        comments.insert(0, ['Text', 'Label'])
        output_path = os.path.join(rootpath.detect(), 'structured_data', 'dataset_3', 'data.csv')
        utilities.write_to_file(comments, output_path)
        logger.info("Dataset_3 has been prepared")

    @staticmethod
    def prepare_dataset_4(input_file_path:str):
        """
        Reads the data from dataset_4. Converts it into format Text,Label(1-Hate,0-Non hate) and writes the contents to file.
        """
        data = utilities.read_file_contents_pd(input_file_path)
        data = data[data['label'] !=1]
        data['label'].where(~(data.label == 0),other=1, inplace=True)
        data['label'].where(~(data.label == 2), other=0, inplace=True)
        data['comment'] = data.apply(lambda row: [row.tweet,row.label],axis=1)
        output_path = os.path.join(rootpath.detect(), 'structured_data', 'dataset_4', 'data.csv')
        utilities.write_to_file(data['comment'].tolist(), output_path)
        logger.info("Dataset_4 has been prepared")

    @staticmethod
    def prepare_dataset_5(input_file_path:str):
        """
        Reads the data from dataset_3. Converts it into format Text,Label(1-Hate,0-Non hate) and writes the contents to file.
        """

        labels, data = utilities.read_file_contents(os.path.join(rootpath.detect(), 'source_data', 'dataset_5', 'annotations_metadata.csv'), ',')
        comments = []
        for row in data:
            file, user, forum, cont, label = row
            text = utilities.read_file_contents_txt(os.path.join(rootpath.detect(), 'source_data', 'dataset_5', 'all_files', file + '.txt'))
            label = 1 if label =='hate' else 0
            comments.append([text,label])
        comments.insert(0,['Text','Label'])
        output_path = os.path.join(rootpath.detect(), 'structured_data', 'dataset_5', 'data.csv')
        utilities.write_to_file(comments, output_path)
        logger.info("Dataset_5 has been prepared")



    @staticmethod
    def prepare_all(source_data_path:str):
        """
        Preprocesses and prepares unstructured source datasets into structured datasets. Each processed dataset has two columns [Text,Label].
        Label represents binary class [1-Hate speech, 0-Non-hate speech].
        """
        for index,directory in enumerate(reversed(os.listdir(source_data_path))):
            input_file_path = os.path.join(source_data_path,'dataset_1','data.csv')
            # eval('Preparator.prepare_dataset_' + str(index+1) + '(input_file_path)')
            eval('Preparator.prepare_dataset_' + '1' + '(input_file_path)')




