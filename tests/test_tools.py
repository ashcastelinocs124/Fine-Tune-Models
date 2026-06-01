from macro_ds import tools


def test_web_search_uses_formatting_and_handles_error(mocker):
    mocker.patch.object(
        tools, "_sonar_raw", return_value=[{"title": "A", "url": "http://u", "snippet": "s"}]
    )
    out = tools.web_search("fed path")
    assert "http://u" in out

    mocker.patch.object(tools, "_sonar_raw", side_effect=RuntimeError("429"))
    err = tools.web_search("fed path")
    assert err.startswith("ERROR:")


def test_fetch_url_extracts_and_handles_error(mocker):
    mocker.patch.object(tools, "_http_get", return_value="<html><body><p>Yields rose.</p></body></html>")
    mocker.patch("macro_ds.tools.trafilatura.extract", return_value="Yields rose.")
    out = tools.fetch_url("http://u")
    assert "Yields rose." in out
    assert "PAGE http://u" in out

    mocker.patch.object(tools, "_http_get", side_effect=RuntimeError("timeout"))
    assert tools.fetch_url("http://u").startswith("ERROR:")


def test_fred_series_formats_and_handles_error(mocker):
    mocker.patch.object(
        tools,
        "_fred_raw",
        return_value=[{"date": "2026-05-01", "value": "4.5"}, {"date": "2026-05-02", "value": "4.6"}],
    )
    out = tools.fred_series("DGS10")
    assert "DGS10" in out and "4.6" in out

    mocker.patch.object(tools, "_fred_raw", side_effect=RuntimeError("bad series"))
    assert tools.fred_series("NOPE").startswith("ERROR:")


def test_tools_registry_shape():
    assert set(tools.TOOLS) == {"web_search", "fetch_url", "fred_series", "fred_search"}
    for name, (fn, schema) in tools.TOOLS.items():
        assert callable(fn)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == name
        assert "parameters" in schema["function"]


def test_openai_tool_schemas_returns_list():
    schemas = tools.openai_tool_schemas()
    assert isinstance(schemas, list) and len(schemas) == 4
