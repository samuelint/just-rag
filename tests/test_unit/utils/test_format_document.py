from langchain_core.documents import Document
from just_rag.utils.format_document import format_documents_with_sources


class TestFormatDocumentWithSources:

    def test_each_source_have_id(self):
        documents = [
            Document(
                page_content="6 (1) Tout citoyen canadien a le droit de demeurer au Canada, d’y entrer ou d’en sortir.",
            ),
            Document(
                page_content="Note marginale :Droits et libertés au Canada",
            ),
        ]

        result = format_documents_with_sources(documents)

        assert "Source ID: 0" in result
        assert "Source ID: 1" in result

    def test_each_source_have_page_content(self):
        documents = [
            Document(
                page_content="6 (1) Tout citoyen canadien a le droit de demeurer au Canada, d’y entrer ou d’en sortir.",
            ),
            Document(
                page_content="Note marginale :Droits et libertés au Canada",
            ),
        ]

        result = format_documents_with_sources(documents)

        assert (
            "Content: 6 (1) Tout citoyen canadien a le droit de demeurer au Canada, d’y entrer ou d’en sortir."
            in result
        )
        assert "Content: Note marginale :Droits et libertés au Canada" in result

    def test_metadata_containing_last_modified_is_included(self):
        documents = [
            Document(
                page_content="6 (1) Tout citoyen canadien a le droit de demeurer au Canada, d’y entrer ou d’en sortir.",
                metadata={
                    "last_modified": "2022-01-01",
                },
            ),
            Document(
                page_content="Note marginale :Droits et libertés au Canada",
                metadata={
                    "last_modified": "2022-01-02",
                },
            ),
        ]

        result = format_documents_with_sources(documents, meta_keys=["last_modified"])

        assert (
            "Content: 6 (1) Tout citoyen canadien a le droit de demeurer au Canada, d’y entrer ou d’en sortir."
            in result
        )
        assert "last_modified: 2022-01-01" in result
        assert "last_modified: 2022-01-02" in result

    def test_metadata_containing_filename_is_included(self):
        documents = [
            Document(
                page_content="6 (1) Tout citoyen canadien a le droit de demeurer au Canada, d’y entrer ou d’en sortir.",
                metadata={
                    "filename": "Charte canadienne des droits et facultés.html",
                },
            ),
            Document(
                page_content="Note marginale :Droits et libertés au Canada",
                metadata={
                    "filename": "Charte canadienne des droits et facultés.html2",
                },
            ),
        ]

        result = format_documents_with_sources(documents, meta_keys=["filename"])

        assert "\nfilename: Charte canadienne des droits et facultés.html" in result
        assert "\nfilename: Charte canadienne des droits et facultés.html2" in result

    def test_metadata_not_in_keys_is_not_formatted(self):
        documents = [
            Document(
                page_content="6 (1) Tout citoyen canadien a le droit de demeurer au Canada, d’y entrer ou d’en sortir.",
                metadata={
                    "filename": "Charte canadienne des droits et facultés.html",
                },
            ),
            Document(
                page_content="Note marginale :Droits et libertés au Canada",
                metadata={
                    "filename": "Charte canadienne des droits et facultés.html",
                },
            ),
        ]

        result = format_documents_with_sources(documents)

        assert "\nfilename: Charte canadienne des droits et facultés.html" not in result
