import pandas as pd
import os
import random
from faker import Faker
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, CopulaGANSynthesizer
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from urllib.parse import quote
import numpy as np
import re

base_url = settings.BASE_URL
faker = Faker()

MODELS = {
    "copulagan": CopulaGANSynthesizer,
    "gaussian": GaussianCopulaSynthesizer,
    "ctgan": CTGANSynthesizer,
}

# ----------------------
# SMART FAKER-BASED FIELD FILLING
# ----------------------
def smart_fill(value, col_name, gender=None):
    col = col_name.lower()
    needs_replacement = (
        pd.isnull(value)
        or value is None
        or (isinstance(value, str) and value.strip().lower() in ["", "-", "?", "nan", "none", "null"])
        or (isinstance(value, str) and re.match(r"sdv-pii", value.lower()))
    )

    if needs_replacement:
        if 'email' in col:
            return faker.email()
        elif 'first_name' in col:
            return faker.first_name_male() if gender == 'male' else faker.first_name_female() if gender == 'female' else faker.first_name()
        elif 'last_name' in col:
            return faker.last_name()
        elif 'name' in col:
            return faker.name_male() if gender == 'male' else faker.name_female() if gender == 'female' else faker.name()
        elif 'address' in col:
            return faker.address()
        elif 'city' in col:
            return faker.city()
        elif 'state' in col:
            return faker.state()
        elif 'country' in col:
            return faker.country()
        elif 'postcode' in col or 'zip' in col:
            return faker.postcode()
        elif 'currency' in col or 'amount' in col or 'price' in col or 'salary' in col:
            return round(random.uniform(30000, 70000), 2)
        elif 'phone' in col or 'mobile' in col:
            return faker.phone_number()
        elif 'date' in col:
            return faker.date()
        elif 'time' in col:
            return faker.time()
        elif 'dob' in col or 'birth' in col:
            return faker.date_of_birth().strftime('%Y-%m-%d')
        elif 'gender' in col:
            return random.choice(["Male", "Female"])
        elif 'company' in col:
            return faker.company()
        elif 'job' in col or 'position' in col or 'title' in col:
            return faker.job()
        elif 'url' in col or 'website' in col:
            return faker.url()
        elif 'domain' in col:
            return faker.domain_name()
        elif 'ip' in col:
            return faker.ipv4()
        elif 'credit_card' in col or 'cc' in col:
            return faker.credit_card_number(card_type=None)
        elif 'bool' in col or 'flag' in col or col.startswith('is_') or col.startswith('has_'):
            return random.choice([True, False])
        elif 'id' == col or 'code' in col or 'number' in col:
            # If ID/code, generate numeric or alphanumeric based on original type
            if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
                return random.randint(1, 10000)
            else:
                return faker.bothify(text='??##??##')
        elif 'text' in col or 'description' in col or 'comment' in col or 'note' in col:
            return faker.sentence(nb_words=6)
        else:
            return faker.word()
    return value

# ----------------------
# MAIN FUNCTION
# ----------------------
def handle_upload_and_generate(uploaded_file, num_rows, model_type, output_file_type):
    file_path = default_storage.save(uploaded_file.name, ContentFile(uploaded_file.read()))
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    file_abs_path = default_storage.path(file_path)

    if ext == ".csv":
        df = pd.read_csv(file_abs_path)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_abs_path)
    elif ext == ".tsv":
        df = pd.read_csv(file_abs_path, sep="\t")
    else:
        raise ValueError("Unsupported file format")
    try:
        if df.empty:
            # Just use smart_fill to generate synthetic data from column names
           #if df.empty or df.columns.size == 0:
            columns = df.columns.tolist()  # e.g., ['id', 'name']
            # Generate all rows synthetically
            synthetic_rows = []
            for i in range(num_rows):
                gender = random.choice(["Male", "Female"])
                row = {}
                for col in columns:
                    row[col] = smart_fill(None, col, gender)
                synthetic_rows.append(row)
            combined_df = pd.DataFrame(synthetic_rows)
            print(combined_df.shape)
        else:
            original_row_count = len(df)
            # Clean missing values like '-', '', 'nan', etc.
            df.replace(["-", " ","?", "", "nan", "NaN", "None", "null"], np.nan, inplace=True)
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(df)
            model_class = MODELS.get(model_type.lower(), CopulaGANSynthesizer)
            model = model_class(metadata)
            model.fit(df.dropna(how="all"))
            df_filled = df.copy()
            for i, row in df.iterrows():
                gender = row.get("gender", None)
                for col in df.columns:
                    val = row[col]
                    df_filled.at[i, col] = smart_fill(val, col, gender)

            if num_rows > original_row_count:
                extra = num_rows - original_row_count
                new_rows = model.sample(extra)

                # Fix 'id' if exists
                if 'id' in df.columns:
                    max_id = pd.to_numeric(df['id'], errors='coerce').max()
                    new_rows['id'] = range(int(max_id) + 1, int(max_id) + 1 + len(new_rows))

                # Replace synthetic pii values and NaNs
                for i, row in new_rows.iterrows():
                    gender = row.get("gender", None)
                    for col in new_rows.columns:
                        new_rows.at[i, col] = smart_fill(row[col], col, gender)

                combined_df = pd.concat([df_filled, new_rows], ignore_index=True)
            else:
                combined_df = df_filled.head(num_rows)

        # Final cleanup and reindex
        if 'id' in combined_df.columns:
            # First, just ignore the current 'id' values and assign new unique ones
            combined_df['id'] = range(1, len(combined_df) + 1)
            # Optionally drop duplicates across the entire row, not just 'id'
            combined_df = combined_df.drop_duplicates()
            combined_df.reset_index(drop=True, inplace=True)

        ext_map = {"csv": ".csv", "tsv": ".tsv", "xlsx": ".xlsx", "xls": ".xls"}
        output_file_type = output_file_type.lower()
        if output_file_type not in ext_map:
            raise ValueError("Unsupported output file type")

        output_ext = ext_map[output_file_type]
        filename, _ = os.path.splitext(uploaded_file.name)
        output_file = f"{filename}_cleaned_synthetic{output_ext}"

        output_dir = os.path.join(settings.MEDIA_ROOT, "synthetic")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)

        if output_file_type in ["csv", "tsv"]:
            sep = "\t" if output_file_type == "tsv" else ","
            combined_df.to_csv(output_path, index=False, sep=sep)
        else:
            engine = "openpyxl" if output_file_type == "xlsx" else "xlwt"
            combined_df.to_excel(output_path, index=False, engine=engine)
        return f"{base_url}/media/synthetic/{quote(output_file)}"

    except Exception as e:
        print(e)
        return f'Error: {e}'
