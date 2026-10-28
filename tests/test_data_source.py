from src.dcsa.data_source import PROJECT_ROOT, input_paths, verify_inputs


def test_copied_clean_inputs_are_local_to_dcsa():
    paths = input_paths()
    assert all(path.is_file() for path in paths.values())
    assert all(PROJECT_ROOT in path.parents for path in paths.values())


def test_input_provenance_has_sha256_and_nonzero_size():
    metadata = verify_inputs()
    assert all(len(item["sha256"]) == 64 for item in metadata.values())
    assert all(item["size_bytes"] > 0 for item in metadata.values())
