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
        data = [[' '.join(row[1].split(' ')[1:]),row[0]] for row in data]
        data.insert(0,labels)
        data = utilities.strip_and_replace_new_lines(data)
        output_path = os.path.join(rootpath.detect(), 'structured_data', 'dataset_1', 'data.csv')
        utilities.write_to_file(data, output_path)
        logger.info("Dataset_1 has been prepared")

    @staticmethod
    def prepare_dataset_2(input_file_path:str):
        """
        Reads the data from dataset_2. Converts it into format Text,Label(1-Hate,0-Non hate) and writes the contents to file.
        """
        data = utilities.read_file_contents_pd(input_file_path)
        data['text'] = data.apply(lambda row: utilities.transform_dataset_2_3(row.text, row.hate_speech_idx), axis=1)
        comments = [item for sublist in data['text'].tolist() for item in sublist]
        comments.insert(0, ['Text', 'Label'])
        comments = utilities.strip_and_replace_new_lines(comments)
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
        comments = utilities.strip_and_replace_new_lines(comments)
        output_path = os.path.join(rootpath.detect(), 'structured_data', 'dataset_3', 'data.csv')
        utilities.write_to_file(comments, output_path)
        logger.info("Dataset_3 has been prepared")

    @staticmethod
    def prepare_dataset_4(input_file_path:str):
        """
        Reads the data from dataset_4. Converts it into format Text,Label(1-Hate,0-Non hate) and writes the contents to file.
        """

        labels, data = utilities.read_file_contents(os.path.join(rootpath.detect(), 'source_data', 'eng', 'dataset_4', 'annotations_metadata.csv'), ',')
        comments = []
        for row in data:
            file, user, forum, cont, label = row
            text = utilities.read_file_contents_txt(os.path.join(rootpath.detect(), 'source_data', 'eng', 'dataset_4', 'all_files', file + '.txt'))
            label = 1 if label =='hate' else 0
            comments.append([text,label])
        comments.insert(0,['Text','Label'])
        comments = utilities.strip_and_replace_new_lines(comments)
        output_path = os.path.join(rootpath.detect(), 'structured_data', 'dataset_4', 'data.csv')
        utilities.write_to_file(comments, output_path)
        logger.info("Dataset_4 has been prepared")

    @staticmethod
    def prepare_dataset_5(input_file_path:str):
        """
        Reads the data from dataset_5. Converts it into format Text,Label(0-Archaic,1-Class,2-Disability,3-Ethnicity,4-Gender,5-Nationality,6-Religion,7-Sexual Orientation) and writes the contents to file.
        """

        class_text_number_map = {
            'archaic': 0,
            'class': 1,
            'disability': 2,
            'ethn': 3,
            'gender': 4,
            'nation': 5,
            'rel': 6,
            'sexorient': 7
        }

        directory = os.path.join(rootpath.detect(), 'source_data', 'eng', 'dataset_5', 'downloaded_tweets_dataset')

        comments = []
        for filename in os.listdir(directory):
            labels, data = utilities.read_file_contents(os.path.join(directory, filename), ',')
            class_text = filename.split("_")[0]
            comments_file = [[row[1], class_text_number_map[class_text]] for row in data]
            comments.extend(comments_file)
        
        comments.insert(0,['Text','Label'])
        comments = utilities.strip_and_replace_new_lines(comments)
        output_path = os.path.join(rootpath.detect(), 'structured_data', 'dataset_5', 'data.csv')
        utilities.write_to_file(comments, output_path)
        logger.info('Dataset_5 has been prepared')
    
    @staticmethod
    def prepare_dataset_8(input_file_path:str):
        """
        Reads the data from dataset_8. Converts it into format Text,Label(0-Appearance,1-Intelligence,2-Political,3-Racial,4-Sextual) and writes the contents to file.
        """

        class_text_number_map = {
            'Appearance Data': 0,
            'Intelligence Data': 1,
            'Political Data': 2,
            'Racial Data': 3,
            'Sextual Data': 4
        }

        directory = os.path.join(rootpath.detect(), 'source_data', 'eng', 'dataset_8', 'tweets_dataset')

        comments = []
        for filename in os.listdir(directory):
            labels, data = utilities.read_file_contents(os.path.join(directory, filename), ',', 'unicode_escape')
            class_text = filename.split(".")[0]
            comments_file = [[row[0], class_text_number_map[class_text]] for row in data if row[1].lower() == "yes"]
            comments.extend(comments_file)
        
        comments.insert(0,['Text','Label'])
        comments = utilities.strip_and_replace_new_lines(comments)
        output_path = os.path.join(rootpath.detect(), 'structured_data', 'dataset_8', 'data.csv')
        utilities.write_to_file(comments, output_path)
        logger.info('Dataset_8 has been prepared')

    @staticmethod
    def prepare_all(source_data_path:str):
        """
        Preprocesses and prepares unstructured source datasets into structured datasets. Each processed dataset has two columns [Text,Label].
        """
        for index,directory in enumerate(reversed(os.listdir(source_data_path))):
            dataset_name = 'dataset_' + str(index + 1)
            if index + 1 in [1, 2, 3]:
                input_file_path = os.path.join(source_data_path,dataset_name,'data.csv')
                if os.path.exists(input_file_path):
                    eval('Preparator.prepare_dataset_' + str(index+1) + '(input_file_path)')
            else:
                eval('Preparator.prepare_dataset_' + str(index+1) + '("")')




