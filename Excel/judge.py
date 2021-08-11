import pandas as pd


def F(exception=Exception):
    def decorator(f):
        def wrapper(*args):
            try:
                return f(*args)
            except exception:
                return None

        return wrapper

    return decorator


sint = F(ValueError)(pd.to_numeric)
sbool = F(ValueError)(bool)
sdate = F(ValueError)(pd.to_datetime)
sstr = F(ValueError)(str)

def parse_string_type(str):
    if sint(str):
        return sint(str).astype('float64')
    if sdate(str):
        return sdate(str)
    if str == "True" or str == "False":
        return str == "True"
    return str

print(bool("False"))