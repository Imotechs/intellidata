import streamlit as st
import pandas as pd
import requests
import io


def run():
    # ========================
    # File Upload
    # ========================
    st.subheader("📤 Upload Your Dataset")
    uploaded_file = st.file_uploader(
        "Choose your file (CSV, Excel, or TSV)",
        type=["csv", "xlsx", "xls", "tsv"],
        help="This should be a clean dataset with headers.",
        key='test_uploader'
    )

    # ========================
    # Process File
    # ========================
    if uploaded_file:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        ext = None
        try:
            if file_ext == "csv":
                df = pd.read_csv(uploaded_file)
                ext = 'csv'
            elif file_ext in ["xls", "xlsx"]:
                df = pd.read_excel(uploaded_file)
                ext = 'xlsx'
            elif file_ext == "tsv":
                df = pd.read_csv(uploaded_file, sep="\t")
                ext = 'tsv'
            else:
                st.error("Unsupported file type.")
                st.stop()
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

        # ========================
        # Column Management
        # ========================
        st.subheader("🧩 Modify Dataset Columns")

        # Drop columns
        columns_to_keep = st.multiselect(
            "🧹 Select columns to keep:",
            options=list(df.columns),
            default=list(df.columns),
            help="Uncheck any column you want to exclude from the synthetic dataset."
        )
        df = df[columns_to_keep]

        # Add new custom columns
        st.markdown("### ➕ Add Custom Columns (Optional)")

        num_new_fields = st.number_input(
            "How many new columns do you want to add?",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )

        new_fields = {}
        if num_new_fields > 0:
            st.info("👇 Define each new column below:")
            for i in range(num_new_fields):
                col_name = st.text_input(f"Column {i+1} name:", key=f"col_name_{i}")
                col_type = st.selectbox(
                    f"Data type for '{col_name}'", ["text", "number", "date"], key=f"col_type_{i}"
                )
                if col_name:
                    new_fields[col_name] = col_type

        for col_name in new_fields:
            df[col_name] = None

        st.success(f"✅ Final structure: {len(df.columns)} columns total.")
        st.dataframe(df.head(), use_container_width=True)
        st.divider()

        # ========================
        # Controls for Generation
        # ========================
        st.subheader("⚙️ Data Generation Settings")
        original_row_count = df.shape[0]

        num_rows = st.slider(
            f"📊 Your original data has {original_row_count} rows. You can add extra synthetic rows (useful for testing):",
            min_value=original_row_count,
            max_value=original_row_count * 5,
            value=original_row_count,
            step=1
        )


        extensions = ["csv", "xlsx", "tsv", "xls"] 

        output_file_type = st.selectbox(
            "Output format:",
            options= [ext] + [e for e in extensions if e != ext],
            index=0
        )

        # ========================
        # Generate Button
        # ========================
        if st.button("🚀 Generate "):
            with st.spinner("🔄 Generating synthetic data..."):
                # Convert modified DataFrame to file-like object
                output = io.BytesIO()
                if output_file_type == "csv":
                    df.to_csv(output, index=False)
                    mime_type = "text/csv"
                elif output_file_type in ["xlsx", "xls"]:
                    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                        df.to_excel(writer, index=False)
                    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                elif output_file_type == "tsv":
                    df.to_csv(output, sep="\t", index=False)
                    mime_type = "text/tab-separated-values"
                else:
                    st.error("Unsupported output format.")
                    st.stop()

                output.seek(0)
                files = {"file": ("modified_data." + output_file_type, output, mime_type)}
                data = {
                    "num_rows": num_rows,
                    "output_file_type": output_file_type
                }

                try:
                    response = requests.post("http://localhost:8000/api/generate/smart/", files=files, data=data)
                    if response.status_code == 200:
                        file_url = response.json().get("file")
                        st.success("✅ Synthetic data generated successfully!")
                        st.markdown(f"[📥 Click to Download {output_file_type.upper()}]({file_url})", unsafe_allow_html=True)
                    else:
                        st.error("❌ Failed to generate data. Please check server logs.")
                except Exception as e:
                    st.error(f"❌ Server error: {e}")

    # else:
    #     st.info("📎 Please upload a file to begin. OR generate new test dataset from scratch Below")
    # else:
    #     st.info("📎 Please upload a file to begin. OR generate new test dataset from scratch below")

    else:
        st.info("📎 Please upload a file to begin. OR generate new test dataset from scratch below")

        st.subheader("✨ Build Test Dataset From Scratch")

        if "custom_columns" not in st.session_state:
            st.session_state.custom_columns = []

        # Add column inputs
        with st.form("add_column_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            new_col_name = col1.text_input("Column name")
            new_col_type = col2.selectbox("Data type", ["text", "number", "date"])
            submitted = st.form_submit_button("➕ Add Column")

            if submitted:
                if not new_col_name:
                    st.warning("Please enter a column name.")
                elif new_col_name in [col["name"] for col in st.session_state.custom_columns]:
                    st.warning("This column name already exists.")
                else:
                    st.session_state.custom_columns.append({
                        "name": new_col_name,
                        "type": new_col_type
                    })

        # Live column preview
        if st.session_state.custom_columns:
            st.markdown("#### 🧩 Live Column Preview")

            # Build and show empty preview DataFrame with dtypes
            preview_dict = {
                col["name"]: [f"({col['type']})"] for col in st.session_state.custom_columns
            }
            preview_df = pd.DataFrame(preview_dict)
            st.dataframe(preview_df, use_container_width=True)

            # Remove column(s) if needed
            with st.expander("🗑️ Remove a column"):
                col_to_remove = st.selectbox(
                    "Select column to remove:",
                    [col["name"] for col in st.session_state.custom_columns]
                )
                if st.button("❌ Remove"):
                    st.session_state.custom_columns = [
                        col for col in st.session_state.custom_columns if col["name"] != col_to_remove
                    ]
                    st.rerun()

            # Set number of rows
            num_rows = st.slider("📏 Number of rows to generate", min_value=10, max_value=1000, value=50, step=10)
            output_format = st.selectbox("File format", ["csv", "xlsx", "tsv"])
         
            # Generate synthetic data
            if st.button("🚀 Generate Data"):
        
                with st.spinner("🔄 Generating data ..."):
                    if not st.session_state.custom_columns:
                        st.error("Please add at least one column.")
                    else:
                        try:
                            # Step 1: Create empty CSV file in memory
                            headers = [col["name"] for col in st.session_state.custom_columns]
                            df = pd.DataFrame(columns=headers)
                            csv_buffer = io.StringIO()
                            df.to_csv(csv_buffer, index=False)
                            csv_buffer.seek(0)

                            # Step 2: Prepare request
                            files = {
                                "file": ("schema.csv", csv_buffer.getvalue(), "text/csv")
                            }
                            data = {
                                "num_rows": str(num_rows),
                                "output_file_type": output_format
                            }

                            # Step 3: Send request to backend
                            response = requests.post("http://localhost:8000/api/generate/smart/", files=files, data=data)

                            # Step 4: Handle response
                            if response.status_code == 200:
                                file_url = response.json().get("file")
                                st.success("✅ Synthetic data generated successfully!")
                                st.markdown(f"[📥 Click to Download {output_format.upper()}]({file_url})", unsafe_allow_html=True)
                            else:
                                st.error("❌ Failed to generate data. Please check server logs.")

                        except Exception as e:
                            st.error(f"❌ Server error: {e}")


