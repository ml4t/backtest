import vectorbtpro as vbt

from tests.utils import *

# ############# Global ############# #


def teardown_module():
    vbt.settings.reset()


# ############# settings ############# #


def is_lambda(v):
    test_lambda = lambda: 0
    return isinstance(v, type(test_lambda)) and v.__name__ == test_lambda.__name__


def dicts_equal(dct1, dct2):
    for k, v in dct1.items():
        if isinstance(v, dict):
            dicts_equal(v, dct2[k])
        elif is_lambda(v) and is_lambda(dct2[k]) or isinstance(v, float) and np.isnan(v) and np.isnan(dct2[k]):
            pass
        else:
            assert v is dct2[k] or v == dct2[k]


class TestSettings:
    def test_save_and_load(self, tmp_path):
        vbt.settings.set_theme("seaborn")
        vbt.settings.set_option("pickle_reset_dct", True)
        vbt.settings.save(tmp_path / "settings")
        new_settings = vbt.settings.load(tmp_path / "settings")
        dicts_equal(vbt.settings, new_settings)
        dicts_equal(vbt.settings.__dict__, new_settings.__dict__)
        vbt.settings.save(tmp_path / "settings", file_format="ini")
        new_settings = vbt.settings.load(tmp_path / "settings", file_format="ini")
        dicts_equal(vbt.settings, new_settings)
        dicts_equal(vbt.settings.__dict__, new_settings.__dict__)
