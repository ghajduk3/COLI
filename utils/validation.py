

# def validate_input_arguments(type:str, model:str,)
def validate_type_argument(type:str) -> bool:
    return True if type == 'bi' or type == 'multi' else False

def validate_model_argument(model:str) -> bool:
    models = ('SVM', 'LR', 'XGBOOST', 'BERT')
    return True if model in models else False

