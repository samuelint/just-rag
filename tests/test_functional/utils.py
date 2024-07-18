import os
import shutil

assets_directory_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets")
)

test_chromadb_path = "./tests_chroma_db"
test_record_manager_db_path = "tests_record_manager_cache.sql"
test_record_manager_db_url = f"sqlite:///{test_record_manager_db_path}"


def delete_test_records():
    if os.path.exists(test_chromadb_path):
        shutil.rmtree(test_chromadb_path)

    if os.path.exists(test_record_manager_db_path):
        os.remove(test_record_manager_db_path)
