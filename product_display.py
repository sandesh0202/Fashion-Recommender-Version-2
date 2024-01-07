
import pandas as pd

styles = pd.read_csv('static/myntradataset/new_styles.csv', on_bad_lines='skip')

def get_random_products(num_products=20):
    # Select random products
    random_products = styles.sample(n=num_products, random_state=42)

    return random_products


def get_product_details_by_id(product_id):
    # Assuming 'styles' is your DataFrame containing product information
    product_details = styles[styles['id'] == product_id].iloc[0].to_dict()
    return product_details