
import pandas as pd

styles = pd.read_csv('static/myntradataset/new_styles.csv', on_bad_lines='skip')

def get_random_products(gender='All', article_type='All', num_products=20):
    if gender == 'All' and article_type == 'All':
        random_products = styles.sample(n=num_products, random_state=42)
    elif gender == 'All':
        random_products = styles[styles['articleType'] == article_type].sample(n=num_products, random_state=42)
    elif article_type == 'All':
        random_products = styles[styles['gender'] == gender].sample(n=num_products, random_state=42)
    else:
        random_products = styles[(styles['gender'] == gender) & (styles['articleType'] == article_type)].sample(n=num_products, random_state=42)

    return random_products


def get_product_details_by_id(product_id):
    # Assuming 'styles' is your DataFrame containing product information
    product_details = styles[styles['id'] == product_id].iloc[0].to_dict()
    return product_details