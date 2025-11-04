from functools import wraps

import pytest
import vectorbtpro as vbt
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils import jitting

# ############# Global ############# #


class MyJitter1(jitting.Jitter):
    def decorate(self, py_func, tags=None):
        @wraps(py_func)
        def wrapper(*args, **kwargs):
            return py_func(*args, **kwargs)

        wrapper.config = self.config
        wrapper.py_func = py_func
        wrapper.suffix = "my1"
        return wrapper


class MyJitter2(MyJitter1):
    def decorate(self, py_func, tags=None):
        wrapper = MyJitter1.decorate(self, py_func, tags=tags)
        wrapper.suffix = "my2"
        return wrapper


def setup_module():
    vbt.settings.jitting.jitters["my1"] = dict(cls=MyJitter1)
    vbt.settings.jitting.jitters["my2"] = dict(cls=MyJitter2)


def teardown_module():
    vbt.settings.reset()


@pytest.fixture(scope="module")
def jitted_f_my1():
    @vbt.register_jitted(task_id_or_func="f", test="test_my1", tags={"tag_my1"})
    def f_my1():
        pass

    yield f_my1


@pytest.fixture(scope="module")
def jitted_f_my2():
    @vbt.register_jitted(task_id_or_func="f", test="test_my2", tags={"tag_my2"})
    def f_my2():
        pass

    yield f_my2


# ############# jit_registry ############# #


class TestJITRegistry:
    def test_register_jitted(self, jitted_f_my1, jitted_f_my2):
        my1_jitable_setup = jit_reg.jitable_setups["f"]["my1"]
        assert my1_jitable_setup.task_id == "f"
        assert my1_jitable_setup.jitter_id == "my1"
        assert my1_jitable_setup.tags == {"tag_my1"}
        assert my1_jitable_setup.jitter_kwargs == dict(test="test_my1")
        assert my1_jitable_setup.py_func is jitted_f_my1.py_func

        my2_jitable_setup = jit_reg.jitable_setups["f"]["my2"]
        assert my2_jitable_setup.task_id == "f"
        assert my2_jitable_setup.jitter_id == "my2"
        assert my2_jitable_setup.tags == {"tag_my2"}
        assert my2_jitable_setup.jitter_kwargs == dict(test="test_my2")
        assert my2_jitable_setup.py_func is jitted_f_my2.py_func

        assert len(jit_reg.jitted_setups[hash(my1_jitable_setup)]) == 1
        my1_jitted_setup = list(jit_reg.jitted_setups[hash(my1_jitable_setup)].values())[0]
        assert my1_jitted_setup.jitter == MyJitter1(test="test_my1")
        assert my1_jitted_setup.jitted_func is jitted_f_my1

        assert len(jit_reg.jitted_setups[hash(my2_jitable_setup)]) == 1
        my2_jitted_setup = list(jit_reg.jitted_setups[hash(my2_jitable_setup)].values())[0]
        assert my2_jitted_setup.jitter == MyJitter2(test="test_my2")
        assert my2_jitted_setup.jitted_func is jitted_f_my2

        def f3_my1():
            pass

        vbt.settings.jitting["jitters"]["my1"]["tasks"] = {"f2": {"replace_py_func": f3_my1}}

        @vbt.register_jitted(task_id_or_func="f2")
        def f2_my1():
            pass

        assert f2_my1.py_func is f3_my1

        vbt.settings.jitting["jitters"]["my1"]["tasks"].clear()

    def test_match_jitable_setups(self):
        assert jit_reg.match_jitable_setups('task_id == "f"') == {
            jit_reg.jitable_setups["f"]["my1"],
            jit_reg.jitable_setups["f"]["my2"],
        }

    def test_match_jitted_setups(self, jitted_f_my2):
        my1_jitable_setup = jit_reg.jitable_setups["f"]["my1"]
        assert jit_reg.match_jitted_setups(my1_jitable_setup) == {
            list(jit_reg.jitted_setups[hash(my1_jitable_setup)].values())[0]
        }

        my2_jitable_setup = jit_reg.jitable_setups["f"]["my2"]
        assert jit_reg.match_jitted_setups(
            my2_jitable_setup,
            "jitted_func == f_my2",
            context=dict(f_my2=jitted_f_my2),
        ) == {list(jit_reg.jitted_setups[hash(my2_jitable_setup)].values())[0]}
        assert (
            jit_reg.match_jitted_setups(
                my2_jitable_setup, "jitted_func != f_my2", context=dict(f_my2=jitted_f_my2)
            )
            == set()
        )

    def test_resolve(self, jitted_f_my1, jitted_f_my2):
        vbt.settings.jitting["disable_resolution"] = True
        assert jit_reg.resolve(task_id_or_func="f") == "f"
        vbt.settings.jitting["disable_resolution"] = False

        assert jit_reg.resolve(task_id_or_func="f", jitter="my1") is jitted_f_my1
        assert jit_reg.resolve(task_id_or_func="f", jitter="my2") is jitted_f_my2

        @vbt.register_jitted
        def f2_my1():
            pass

        def f2_my2():
            pass

        assert jit_reg.resolve(task_id_or_func=f2_my1) is f2_my1
        assert jit_reg.resolve(task_id_or_func="tests.test_jit_registry.f2_my1") is f2_my1
        assert jit_reg.resolve(task_id_or_func=f2_my1.py_func) is f2_my1

        with pytest.raises(Exception):
            jit_reg.resolve(task_id_or_func=f2_my2)
        assert jit_reg.resolve(task_id_or_func=f2_my2, return_missing_task=True) is f2_my2

        with pytest.raises(Exception):
            jit_reg.resolve(task_id_or_func="f")
        assert (
            jit_reg.resolve(
                task_id_or_func="f", jitter=vbt.RepEval("list(task_setups.keys())[0]")
            ).py_func
            is list(jit_reg.jitable_setups["f"].values())[0].py_func
        )

        @vbt.register_jitted(jitter="my1")
        def f3():
            pass

        assert jit_reg.resolve(task_id_or_func=f3) is f3

        def f4():
            pass

        with pytest.raises(Exception):
            jit_reg.resolve(task_id_or_func=f4)
        with pytest.raises(Exception):
            jit_reg.resolve(task_id_or_func=f4, allow_new=True)
        assert jit_reg.resolve(task_id_or_func=f4, jitter="my1", allow_new=True).py_func is f4
        assert jit_reg.resolve(task_id_or_func=f4, jitter="my1", allow_new=True).suffix == "my1"
        assert jit_reg.resolve(task_id_or_func=f4, jitter="my2", allow_new=True).suffix == "my2"
        with pytest.raises(Exception):
            jit_reg.resolve(task_id_or_func="f4", jitter="my2", allow_new=True)

        vbt.settings.jitting["disable"] = True
        assert jit_reg.resolve(task_id_or_func="f", jitter="my1") is jitted_f_my1.py_func
        assert jit_reg.resolve(task_id_or_func="f", jitter="my2") is jitted_f_my2.py_func
        vbt.settings.jitting["disable"] = False

        assert (
            jit_reg.resolve(task_id_or_func="f", jitter="my1", disable=True) is jitted_f_my1.py_func
        )
        assert (
            jit_reg.resolve(task_id_or_func=jitted_f_my1, disable=True, allow_new=True)
            is jitted_f_my1.py_func
        )
        assert (
            jit_reg.resolve(
                task_id_or_func="f", jitter="my1", disable=vbt.RepEval("jitter_id == 'my1'")
            )
            is jitted_f_my1.py_func
        )
        assert (
            jit_reg.resolve(
                task_id_or_func="f", jitter="my2", disable=vbt.RepEval("jitter_id == 'my1'")
            ).py_func
            is jitted_f_my2.py_func
        )

        vbt.settings.jitting["jitters"]["my1"]["resolve_kwargs"] = dict(
            hello="world", hello2="world2"
        )
        vbt.settings.jitting["jitters"]["my1"]["tasks"] = {
            "tests.test_jit_registry.f4": {"resolve_kwargs": dict(hello2="world3", hello3="world3")}
        }

        res_func = jit_reg.resolve(
            task_id_or_func=f4,
            jitter=MyJitter1,
            allow_new=True,
            my_jitter_id=vbt.Rep("jitter_id"),
        )
        assert res_func.config["hello"] == "world"
        assert res_func.config["hello2"] == "world3"
        assert res_func.config["hello2"] == "world3"
        assert res_func.config["my_jitter_id"] == "my1"

        vbt.settings.jitting["jitters"]["my1"]["resolve_kwargs"].clear()
        vbt.settings.jitting["jitters"]["my1"]["tasks"].clear()

        @vbt.register_jitted(task_id_or_func="f5", jitter="my1")
        def f5():
            pass

        jitable_setup = jit_reg.jitable_setups["f5"]["my1"]
        assert len(jit_reg.jitted_setups[hash(jitable_setup)]) == 1
        jit_reg.resolve(task_id_or_func="f5")
        assert len(jit_reg.jitted_setups[hash(jitable_setup)]) == 1
        jit_reg.resolve(task_id_or_func="f5", new_kwarg1="new_kwarg1", new_kwarg2="new_kwarg2")
        assert len(jit_reg.jitted_setups[hash(jitable_setup)]) == 2
        jit_reg.resolve(task_id_or_func="f5", new_kwarg2="new_kwarg2", new_kwarg1="new_kwarg1")
        assert len(jit_reg.jitted_setups[hash(jitable_setup)]) == 2

        def f6():
            pass

        jit_reg.resolve(
            task_id_or_func=f6,
            jitter=MyJitter1,
            allow_new=True,
            register_new=True,
            tags={"my_tag"},
            new_kwarg="new_kwarg",
        )
        jitable_setup = jit_reg.jitable_setups["tests.test_jit_registry.f6"]["my1"]
        assert jitable_setup.task_id == "tests.test_jit_registry.f6"
        assert jitable_setup.jitter_id == "my1"
        assert jitable_setup.tags == {"my_tag"}
        assert jitable_setup.jitter_kwargs == dict(new_kwarg="new_kwarg")
        assert jitable_setup.py_func is f6

        assert len(jit_reg.jitted_setups[hash(jitable_setup)]) == 1
        jitted_setup = list(jit_reg.jitted_setups[hash(jitable_setup)].values())[0]
        assert jitted_setup.jitter == MyJitter1(new_kwarg="new_kwarg")
        assert jitted_setup.jitted_func.suffix == "my1"
