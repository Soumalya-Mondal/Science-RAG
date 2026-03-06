if __name__ == '__main__':
    # importing python modules:S1
    try:
        from pathlib import Path
        import urllib.request
        import os
        # import json
        # from typing import List

        # redirect model caches
        parent_folder_path = Path.cwd()
        models_folder_path = parent_folder_path / "models"
        os.environ["UNSTRUCTURED_HOME"] = str(models_folder_path)
        os.environ["HF_HOME"] = str(models_folder_path / "hf_cache")

        from unstructured.partition.pdf import partition_pdf
        # from unstructured.chunking.title import chunk_by_title
        # from langchain_core.documents import Document
        # from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        # from langchain_chroma import Chroma
        # from langchain_core.messages import HumanMessage
    except Exception as error:
        print(f'ERROR - [Main:S1] - {error}')

    # define folder path:S2
    try:
        parent_folder_path = Path.cwd()
        input_folder_path = parent_folder_path / 'input'
        pdf_document_path = input_folder_path / 'Sample-DOC.pdf'
    except Exception as error:
        print(f'ERROR - [Main:S2] - {error}')

    # partition pdf document:S3
    try:
        pdf_doc_elements = partition_pdf(
            filename = str(pdf_document_path),  # Path to your PDF file
            strategy = "hi_res", # Use the most accurate (but slower) processing method of extraction
            infer_table_structure = True, # Keep tables as structured HTML, not jumbled text
            extract_image_block_types = ["Image"], # Grab images found in the PDF
            extract_image_block_to_payload = True # Store images as base64 data you can actually use
        )
        print(pdf_doc_elements)
    except Exception as error:
        print(f'ERROR - [Main:S3] - {error}')