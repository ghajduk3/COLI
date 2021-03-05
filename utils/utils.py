import os, csv
import logging

logger = logging.getLogger(__name__)


def read_file_contents(input_file_path: str, delimiter=",") -> tuple:
    """

    :param file_path:
    :param delimiter:
    :return: labels,data
    """
    try:
        with open(input_file_path) as csv_file:
            reader = csv.reader(csv_file, delimiter)
            labels = next(reader)
            data = [line for line in reader]
            logger.info("Data has been read succesfully")
            return labels, data
    except IOError:
        logger.exception("I/O error")
    except FileNotFoundError:
        logger.exception("FileNotFound error")
    except Exception:
        logger.exception("Unexpected error")

def write_to_file(content,output_file_path:str):
    """

    :param content:
    :param output_file_path:
    :return:
    """
    try:
        with open(output_file_path,'w+') as output_file:
            writer = csv.writer(output_file)
            writer.writerows(content)
            logger.info("Data has been succesfully written to file")
    except IOError:
        logger.exception("I/O error")
    except FileNotFoundError:
        logger.exception("FileNotFound error")
    except Exception:
        logger.exception("Unexpected error")
