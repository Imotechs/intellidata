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

base_url = settings.BASE_URL

MODELS = {
    "copulagan": CopulaGANSynthesizer,
    "gaussian": GaussianCopulaSynthesizer,
    "ctgan": CTGANSynthesizer
}

faker = Faker()

def generate_name(gender=None):
    if gender:
        if gender.lower() == 'male':
            return faker.name_male()
        elif gender.lower() == 'female':
            return faker.name_female()
    return faker.name()

def generate_first_name(gender=None):
    if gender:
        if gender.lower() == 'male':
            return faker.first_name_male()
        elif gender.lower() == 'female':
            return faker.first_name_female()
    return faker.first_name()

def generate_last_name():
    return faker.last_name()

def handle_upload_and_generate(uploaded_file, num_rows, model_type, output_file_type):
    # Save uploaded file
    file_path = default_storage.save(uploaded_file.name, ContentFile(uploaded_file.read()))
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    file_abs_path = default_storage.path(file_path)

    # Load DataFrame
    if ext == ".csv":
        df = pd.read_csv(file_abs_path)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_abs_path)
    elif ext == ".tsv":
        df = pd.read_csv(file_abs_path, sep="\t")
    else:
        raise ValueError("Unsupported file format. Please upload CSV, Excel (.xls/.xlsx), or TSV.")

    original_row_count = len(df)

    # Metadata detection
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    # Train model
    model_class = MODELS.get(model_type.lower(), CopulaGANSynthesizer)
    model = model_class(metadata)
    model.fit(df.dropna())

    # Fill missing values
    df_filled = df.copy()
    if df.isnull().values.any():
        filler_df = model.sample(len(df) * 2)
        filler_index = 0
        for i in range(len(df)):
            for col in df.columns:
                if pd.isnull(df_filled.at[i, col]):
                    while pd.isnull(filler_df.at[filler_index, col]):
                        filler_index += 1
                    df_filled.at[i, col] = filler_df.at[filler_index, col]
                    filler_index += 1

    # Add synthetic rows if needed
    if num_rows > original_row_count:
        extra_rows = num_rows - original_row_count
        new_rows = model.sample(extra_rows)

        # Fix ID if needed
        if 'id' in df.columns:
            max_id = int(df['id'].max())
            new_rows['id'] = list(range(max_id + 1, max_id + 1 + len(new_rows)))
            new_rows['id'] = new_rows['id'].astype(int)

        # Fix name column
        if 'name' in df.columns:
            real_names = df['name'].dropna().unique().tolist()
            def replace_name(row):
                if isinstance(row['name'], str) and ('sdv-pii' in row['name'] or not row['name'].strip()):
                    return random.choice(real_names) if real_names else generate_name(row.get('gender', None))
                return row['name']
            new_rows['name'] = new_rows.apply(replace_name, axis=1)

        # Fix first_name and last_name
        if 'first_name' in df.columns:
            new_rows['first_name'] = new_rows.apply(
                lambda row: generate_first_name(row.get('gender')) if not isinstance(row['first_name'], str) or 'sdv-pii' in row['first_name'] else row['first_name'],
                axis=1
            )

        if 'last_name' in df.columns:
            new_rows['last_name'] = new_rows.apply(
                lambda row: generate_last_name() if not isinstance(row['last_name'], str) or 'sdv-pii' in row['last_name'] else row['last_name'],
                axis=1
            )

        combined_df = pd.concat([df_filled, new_rows], ignore_index=True)
    else:
        combined_df = df_filled.head(num_rows)

    if 'id' in combined_df.columns:
        combined_df['id'] = combined_df['id'].astype(int)

    # File type logic
    ext_map = {
        "csv": ".csv",
        "tsv": ".tsv",
        "xlsx": ".xlsx",
        "xls": ".xls",
    }
    output_file_type = output_file_type.lower()
    if output_file_type not in ext_map:
        raise ValueError(f"Unsupported output file type: {output_file_type}")

    output_ext = ext_map[output_file_type]
    filename, _ = os.path.splitext(uploaded_file.name)
    output_file = f"{filename}_cleaned_synthetic{output_ext}"

    output_dir = os.path.join(settings.MEDIA_ROOT, "synthetic")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    # Save the file
    if output_file_type in ["csv", "tsv"]:
        sep = "\t" if output_file_type == "tsv" else ","
        combined_df.to_csv(output_path, index=False, sep=sep)
    elif output_file_type in ["xls", "xlsx"]:
        engine = "openpyxl" if output_file_type == "xlsx" else "xlwt"
        combined_df.to_excel(output_path, index=False, engine=engine)

    return f"{base_url}/media/synthetic/{quote(output_file)}"
