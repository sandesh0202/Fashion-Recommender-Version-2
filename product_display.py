
import pandas as pd

styles = pd.read_csv('static/myntradataset/new_styles.csv', on_bad_lines='skip')

def get_random_products(gender='All', article_type='All', num_products=20):
    filtered_products = styles

    if gender != 'All':
        filtered_products = filtered_products[filtered_products['gender'] == gender]

    if article_type != 'All':
        filtered_products = filtered_products[filtered_products['articleType'] == article_type]

    if len(filtered_products) < num_products:
        random_products = filtered_products.sample(n=len(filtered_products), random_state=42)
    else:
        random_products = filtered_products.sample(n=num_products, random_state=42)

    return random_products



def get_product_details_by_id(product_id):
    # Assuming 'styles' is your DataFrame containing product information
    product_details = styles[styles['id'] == product_id].iloc[0].to_dict()
    return product_details