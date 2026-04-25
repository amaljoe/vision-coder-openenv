import pytest


def pytest_addoption(parser):
    parser.addoption("--force-render", action="store_true", default=False,
                     help="Re-render all test case PNGs even if they already exist")


@pytest.fixture(scope="session")
def force_render(request):
    return request.config.getoption("--force-render")
