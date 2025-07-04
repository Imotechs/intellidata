import streamlit as st
import pandas as pd
import requests
import io

base_url ="http://backend:8000"
def run():
    # ========================
    # File Upload
    # ========================
    st.subheader("📤 Upload Your Dataset")

    real_uploaded_file = st.file_uploader(
        "Choose your file (CSV, Excel, or TSV)",
        type=["csv", "xlsx", "xls", "tsv"],
        help="This should be a clean dataset with headers.",
        key='real_uploader'
    )

    # ========================
    # Process File
    # ========================
    if real_uploaded_file:
        file_ext = real_uploaded_file.name.split('.')[-1].lower()
        ext = None
        try:
            if file_ext == "csv":
                df = pd.read_csv(real_uploaded_file)
                ext = 'csv'
            elif file_ext in ["xls", "xlsx"]:
                df = pd.read_excel(real_uploaded_file)
                ext = 'xlsx'
            elif file_ext == "tsv":
                df = pd.read_csv(real_uploaded_file, sep="\t")
                ext = 'tsv'
            else:
                st.error("Unsupported file type.")
                st.stop()
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

        # ========================
        # Show File Preview
        # ========================
        st.success(f"✅ File `{real_uploaded_file.name}` loaded successfully!")
        st.write("📊 **Preview of uploaded data:**")
        st.dataframe(df.head(), use_container_width=True)
        st.write(f"📈 Rows: {df.shape[0]}, 📊 Columns: {df.shape[1]}")
        #st.divider()
        # ========================
        # Controls for Generation
        # ========================
        st.subheader("Data Output format")
        original_row_count = df.shape[0]

        # num_rows = st.slider(
        #     f"📊 Your original data has {original_row_count} rows. You can add extra synthetic rows (useful for testing):",
        #     min_value=original_row_count,
        #     max_value=original_row_count * 5,
        #     value=original_row_count,
        #     step=1
        # )
        extensions = ["csv", "xlsx", "tsv", "xls"] 

        output_file_type = st.selectbox(
            "Output format:",
            options= [ext] + [e for e in extensions if e != ext],
            index=0
        )

        # ========================
        # Generate Button
        # ========================
        if st.button("⚙️ Generate "):
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
                    st.error("❌ Unsupported output format.")
                    st.stop()

                output.seek(0)
                files = {"file": ("modified_data." + output_file_type, output, mime_type)}
                data = {
                    "num_rows": original_row_count,
                    "output_file_type": output_file_type
                }

                try:
                    response = requests.post(f"{base_url}/api/generate/smart/", files=files, data=data)
                    if response.status_code == 200:
                        file_url = response.json().get("file")
                        st.success("✅ Synthetic data generated successfully!")
                        st.markdown(f"[📥 Click to Download {output_file_type.upper()}]({file_url})", unsafe_allow_html=True)
                    else:
                        st.error("❌ Failed to generate data. Please check server logs.")
                except Exception as e:
                    st.error(f"❌ Server error: {e}")

    else:
        st.info("📎 Please upload a file to begin.")


