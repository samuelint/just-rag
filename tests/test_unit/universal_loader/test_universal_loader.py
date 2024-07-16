import pytest
from decoy import Decoy, matchers
from langchain_core.documents import Document
from just_rag.universal_loader.loader import FileOrUrlLoader
from just_rag.universal_loader import UniversalDocumentLoader


@pytest.fixture
def loader_1(decoy: Decoy):
    mocked = decoy.mock(cls=FileOrUrlLoader)
    decoy.when(mocked.load(path_or_url=matchers.Anything())).then_return([])

    return mocked


@pytest.fixture
def loader_2(decoy: Decoy):
    mocked = decoy.mock(cls=FileOrUrlLoader)
    decoy.when(mocked.load(path_or_url=matchers.Anything())).then_return([])

    return mocked


@pytest.fixture
def document_1(decoy: Decoy):
    return decoy.mock(cls=Document)


@pytest.fixture
def document_2(decoy: Decoy):
    return decoy.mock(cls=Document)


class TestLoad:

    def test_files_are_loaded_in_loaders_order(
        self,
        decoy: Decoy,
        loader_1: FileOrUrlLoader,
        loader_2: FileOrUrlLoader,
        document_1: Document,
        document_2: Document,
    ):
        decoy.when(loader_1.can_load(path_or_url="a")).then_return(True)
        decoy.when(loader_1.can_load(path_or_url="b")).then_return(False)
        decoy.when(loader_1.load(path_or_url="a")).then_return([document_1])
        decoy.when(loader_2.can_load(path_or_url="a")).then_return(False)
        decoy.when(loader_2.can_load(path_or_url="b")).then_return(True)
        decoy.when(loader_2.load(path_or_url="b")).then_return([document_2])

        instance = UniversalDocumentLoader(
            paths=["a", "b"], loaders=[loader_1, loader_2]
        )

        result = instance.load()

        assert result == [document_1, document_2]

    def test_a_file_is_not_loaded_twice(
        self,
        decoy: Decoy,
        loader_1: FileOrUrlLoader,
        loader_2: FileOrUrlLoader,
    ):
        decoy.when(loader_1.can_load(path_or_url="a")).then_return(True)
        decoy.when(loader_2.can_load(path_or_url="a")).then_return(True)
        instance = UniversalDocumentLoader(paths=["a"], loaders=[loader_1, loader_2])

        instance.load()

        decoy.verify(
            loader_2.load(),
            times=0,
        )
