def test_project_is_independent_from_reference_project():
    from src.dcsa.data_source import PROJECT_ROOT, input_paths

    assert PROJECT_ROOT.name == "crct"
    for path in input_paths().values():
        assert PROJECT_ROOT in path.parents
