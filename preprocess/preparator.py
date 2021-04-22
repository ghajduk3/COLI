from utils import utilities
import rootpath,os,logging
logger = logging.getLogger(__name__)
class Preparator(object):

    @staticmethod
    def prepare_dataset_1_eng(input_file_path:str, lang:str, type:str):
        """
        Reads the data from dataset_1. Converts it into format Text,Label(1-Hate,0-Non hate) and writes the contents to file.
        """
        labels,data = utilities.read_file_contents(input_file_path, ':')
        labels = [labels[1],labels[0]]
        data = [[' '.join(row[1].split(' ')[1:]),row[0]] for row in data]
        data.insert(0,labels)
        data = utilities.strip_and_replace_new_lines(data)
        output_path = os.path.join(rootpath.detect(), 'data','structured_data',lang, type, 'dataset_1', 'data.csv')
        utilities.write_to_file(data, output_path)
        logger.info("Eng Dataset_1 has been prepared")

    @staticmethod
    def prepare_dataset_2_eng(input_file_path:str, lang:str, type:str):
        """
        Reads the data from dataset_2. Converts it into format Text,Label(1-Hate,0-Non hate) and writes the contents to file.
        """
        data = utilities.read_file_contents_pd(input_file_path)
        data['text'] = data.apply(lambda row: utilities.transform_dataset_2_3(row.text, row.hate_speech_idx), axis=1)
        comments = [item for sublist in data['text'].tolist() for item in sublist]
        comments.insert(0, ['Text', 'Label'])
        comments = utilities.strip_and_replace_new_lines(comments)
        output_path = os.path.join(rootpath.detect(), 'data' , 'structured_data', lang, type, 'dataset_2', 'data.csv')
        utilities.write_to_file(comments, output_path)
        logger.info("Eng Dataset_2 has been prepared")

    @staticmethod
    def prepare_dataset_3_eng(input_file_path:str, lang:str, type:str):
        """
        Reads the data from dataset_3. Converts it into format Text,Label(1-Hate,0-Non hate) and writes the contents to file.
        """
        data = utilities.read_file_contents_pd(input_file_path)
        data['text'] = data.apply(lambda row: utilities.transform_dataset_2_3(row.text, row.hate_speech_idx), axis=1)
        comments = [item for sublist in data['text'].tolist() for item in sublist]
        comments.insert(0, ['Text', 'Label'])
        comments = utilities.strip_and_replace_new_lines(comments)
        output_path = os.path.join(rootpath.detect(), 'data' , 'structured_data', lang, type, 'dataset_3', 'data.csv')
        utilities.write_to_file(comments, output_path)
        logger.info("Eng Dataset_3 has been prepared")

    @staticmethod
    def prepare_dataset_4_eng(input_file_path:str, lang:str, type:str):
        """
        Reads the data from dataset_4. Converts it into format Text,Label(1-Hate,0-Non hate) and writes the contents to file.
        """

        labels, data = utilities.read_file_contents(os.path.join(rootpath.detect(), 'data' , 'source_data', lang, type,'dataset_4', 'annotations_metadata.csv'), ',')
        comments = []
        for row in data:
            file, user, forum, cont, label = row
            text = utilities.read_file_contents_txt(os.path.join(rootpath.detect(), 'data' ,'source_data', lang, type,'dataset_4', 'all_files', file + '.txt'))
            label = 1 if label =='hate' else 0
            comments.append([text,label])
        comments.insert(0,['Text','Label'])
        comments = utilities.strip_and_replace_new_lines(comments)
        output_path = os.path.join(rootpath.detect(),'data', 'structured_data', lang, type, 'dataset_4', 'data.csv')
        utilities.write_to_file(comments, output_path)
        logger.info("Eng Dataset_4 has been prepared")

    @staticmethod
    def prepare_dataset_5_eng(input_file_path:str, lang:str, type:str):
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

        directory = os.path.join(rootpath.detect(),'data','source_data', lang, type, 'dataset_5', 'downloaded_tweets_dataset')

        comments = []
        for filename in os.listdir(directory):
            labels, data = utilities.read_file_contents(os.path.join(directory, filename), ',')
            class_text = filename.split("_")[0]
            comments_file = [[row[1], class_text_number_map[class_text]] for row in data]
            comments.extend(comments_file)
        
        comments.insert(0,['Text','Label'])
        comments = utilities.strip_and_replace_new_lines(comments)
        output_path = os.path.join(rootpath.detect(), 'data', 'structured_data', lang ,type, 'dataset_5', 'data.csv')
        utilities.write_to_file(comments, output_path)
        logger.info('Eng Dataset_5 has been prepared')
    
    @staticmethod
    def prepare_dataset_6_eng(input_file_path:str, lang:str, type:str):
        """
        Reads the data from dataset_6. Converts it into format Text,Label(0-Appearance,1-Intelligence,2-Political,3-Racial,4-Sextual) and writes the contents to file.
        """

        class_text_number_map = {
            'Appearance Data': 0,
            'Intelligence Data': 1,
            'Political Data': 2,
            'Racial Data': 3,
            'Sextual Data': 4
        }

        directory = os.path.join(rootpath.detect(), 'data' ,'source_data', lang, type, 'dataset_6', 'tweets_dataset')

        comments = []
        for filename in os.listdir(directory):
            labels, data = utilities.read_file_contents(os.path.join(directory, filename), ',', 'unicode_escape')
            class_text = filename.split(".")[0]
            comments_file = [[row[0], class_text_number_map[class_text]] for row in data if row[1].lower() == "yes"]
            comments.extend(comments_file)
        
        comments.insert(0,['Text','Label'])
        comments = utilities.strip_and_replace_new_lines(comments)
        output_path = os.path.join(rootpath.detect(),'data', 'structured_data',lang, type, 'dataset_6', 'data.csv')
        utilities.write_to_file(comments, output_path)
        logger.info('Eng Dataset_6 has been prepared')
    
    @staticmethod
    def prepare_dataset_1_slo(input_file_path:str, lang:str, type:str):

        data = utilities.read_file_contents_pd(input_file_path)

        comments = []
        comments.extend(data.values)

        comments.insert(0, ['Text', 'Label'])
        comments = utilities.strip_and_replace_new_lines(comments)
        output_path = os.path.join(rootpath.detect(),'data', 'structured_data',lang, type, 'dataset_1', 'data.csv')
        utilities.write_to_file(comments, output_path)
        logger.info('Slo Dataset_1 has been prepared')

    @staticmethod
    def prepare_dataset_2_slo(input_file_path:str, lang:str, type:str):

        data = utilities.read_file_contents_pd(input_file_path)

        non_hate = data.loc[data['Class'].isin([0, 1])]
        del non_hate['Type']
        non_hate.columns = ['Text', 'Label']
        non_hate['Label'] = 0

        hate = data.loc[data['Class'].isin([3])]
        del hate['Class']
        hate.columns = ['Text', 'Label']
        hate['Label'] = hate['Label'].fillna(12)

        non_hate['Label'] = non_hate['Label'].astype(int)
        hate['Label'] = hate['Label'].astype(int)

        comments = []
        comments.extend(non_hate.values)
        comments.extend(hate.values)

        comments.insert(0, ['Text', 'Label'])
        comments = utilities.strip_and_replace_new_lines(comments)
        output_path = os.path.join(rootpath.detect(),'data', 'structured_data',lang, type, 'dataset_2', 'data.csv')
        utilities.write_to_file(comments, output_path)
        logger.info('Slo Dataset_2 has been prepared')


    @staticmethod
    def prepare_all(source_data_path_base:str,lang:str):
        """
        Preprocesses and prepares unstructured source datasets into structured datasets. Each processed dataset has two columns [Text,Label].
        """
        data_types = ['binary', 'multiclass']
        for type in data_types:
            source_data_path = os.path.join(source_data_path_base,type)
            if os.path.exists(source_data_path):
                for directory in reversed(os.listdir(source_data_path)):
                    dataset_number = int(directory.split('_')[1])
                    dataset_name = 'dataset_' + str(dataset_number)

                    input_file_path = os.path.join(source_data_path,dataset_name,'data.csv')
                    if os.path.exists(input_file_path):
                        eval('Preparator.prepare_dataset_' + str(dataset_number) + '_' + lang + '(input_file_path,lang,type)')
                    else:
                        eval('Preparator.prepare_dataset_' + str(dataset_number) + '_' + lang + '("",lang,type)')




