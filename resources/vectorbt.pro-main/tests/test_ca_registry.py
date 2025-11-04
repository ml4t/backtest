import gc
import weakref

import pytest
import vectorbtpro as vbt
from vectorbtpro.registries.ca_registry import CAQuery, CARunResult, CARunSetup, ca_reg
from vectorbtpro.utils.caching import Cacheable

from tests.utils import *

# ############# Global ############# #


def setup_module():
    vbt.settings.pbar["disable"] = True
    vbt.settings.caching["register_lazily"] = False


def teardown_module():
    vbt.settings.reset()


# ############# ca_registry ############# #


class TestCacheableRegistry:
    def test_ca_query(self):
        class A(Cacheable):
            @vbt.cacheable_property(x=10, y=10)
            def f(self):
                return None

            @vbt.cacheable_property(x=10)
            def f2(self):
                return None

        class B(A):
            @vbt.cacheable_property(x=20, y=10)
            def f2(self):
                return None

            @vbt.cacheable_method(x=20)
            def f3(self):
                return None

        @vbt.cacheable(my_option1=True)
        def f4():
            return None

        a = A()
        b = B()

        def match_query(query):
            matched = set()
            if query.matches_setup(A.f.get_ca_setup()):
                matched.add("A.f")
            if query.matches_setup(A.f.get_ca_setup(a)):
                matched.add("a.f")
            if query.matches_setup(A.f2.get_ca_setup()):
                matched.add("A.f2")
            if query.matches_setup(A.f2.get_ca_setup(a)):
                matched.add("a.f2")
            if query.matches_setup(B.f.get_ca_setup()):
                matched.add("B.f")
            if query.matches_setup(B.f.get_ca_setup(b)):
                matched.add("b.f")
            if query.matches_setup(B.f2.get_ca_setup()):
                matched.add("B.f2")
            if query.matches_setup(B.f2.get_ca_setup(b)):
                matched.add("b.f2")
            if query.matches_setup(B.f3.get_ca_setup()):
                matched.add("B.f3")
            if query.matches_setup(B.f3.get_ca_setup(b)):
                matched.add("b.f3")
            if query.matches_setup(f4.get_ca_setup()):
                matched.add("f4")
            if query.matches_setup(A.get_ca_setup()):
                matched.add("A")
            if query.matches_setup(B.get_ca_setup()):
                matched.add("B")
            if query.matches_setup(a.get_ca_setup()):
                matched.add("a")
            if query.matches_setup(b.get_ca_setup()):
                matched.add("b")
            return matched

        assert match_query(CAQuery(cacheable=A.f)) == {"A.f", "B.f", "a.f", "b.f"}
        assert match_query(CAQuery(cacheable=A.f2)) == {"A.f2", "a.f2"}
        assert match_query(CAQuery(cacheable=B.f2)) == {"B.f2", "b.f2"}
        assert match_query(CAQuery(cacheable=B.f3)) == {"B.f3", "b.f3"}
        assert match_query(CAQuery(cacheable=f4)) == {"f4"}
        assert match_query(CAQuery(cacheable=A.f.func)) == {"A.f", "B.f", "a.f", "b.f"}
        assert match_query(CAQuery(cacheable=A.f2.func)) == {"A.f2", "a.f2"}
        assert match_query(CAQuery(cacheable=B.f2.func)) == {"B.f2", "b.f2"}
        assert match_query(CAQuery(cacheable=B.f3.func)) == {"B.f3", "b.f3"}
        assert match_query(CAQuery(cacheable=f4.func)) == {"f4"}
        assert match_query(CAQuery(cacheable="f")) == {"A.f", "B.f", "a.f", "b.f"}
        assert match_query(CAQuery(cacheable="f2")) == {"A.f2", "B.f2", "a.f2", "b.f2"}
        assert match_query(CAQuery(cacheable="f3")) == {"B.f3", "b.f3"}
        assert match_query(CAQuery(cacheable="f4")) == {"f4"}
        assert match_query(CAQuery(cacheable=vbt.Regex("(f2|f3)"))) == {
            "A.f2",
            "B.f2",
            "B.f3",
            "a.f2",
            "b.f2",
            "b.f3",
        }
        assert match_query(CAQuery(instance=a)) == {"a", "a.f", "a.f2"}
        assert match_query(CAQuery(instance=b)) == {"b", "b.f", "b.f2", "b.f3"}
        assert match_query(CAQuery(cls=A)) == {"A", "a", "a.f", "a.f2"}
        assert match_query(CAQuery(cls=B)) == {"B", "b", "b.f", "b.f2", "b.f3"}
        assert match_query(CAQuery(cls="A")) == {"A", "a", "a.f", "a.f2"}
        assert match_query(CAQuery(cls="B")) == {"B", "b", "b.f", "b.f2", "b.f3"}
        assert match_query(CAQuery(cls=("A", "B"))) == {
            "A",
            "B",
            "a",
            "a.f",
            "a.f2",
            "b",
            "b.f",
            "b.f2",
            "b.f3",
        }
        assert match_query(CAQuery(cls=vbt.Regex("(A|B)"))) == {
            "A",
            "B",
            "a",
            "a.f",
            "a.f2",
            "b",
            "b.f",
            "b.f2",
            "b.f3",
        }
        assert match_query(CAQuery(base_cls=A)) == {
            "A",
            "B",
            "a",
            "a.f",
            "a.f2",
            "b",
            "b.f",
            "b.f2",
            "b.f3",
        }
        assert match_query(CAQuery(base_cls=B)) == {"B", "b", "b.f", "b.f2", "b.f3"}
        assert match_query(CAQuery(base_cls="A")) == {
            "A",
            "B",
            "a",
            "a.f",
            "a.f2",
            "b",
            "b.f",
            "b.f2",
            "b.f3",
        }
        assert match_query(CAQuery(base_cls="B")) == {"B", "b", "b.f", "b.f2", "b.f3"}
        assert match_query(CAQuery(base_cls=("A", "B"))) == {
            "A",
            "B",
            "a",
            "a.f",
            "a.f2",
            "b",
            "b.f",
            "b.f2",
            "b.f3",
        }
        assert match_query(CAQuery(base_cls=vbt.Regex("(A|B)"))) == {
            "A",
            "B",
            "a",
            "a.f",
            "a.f2",
            "b",
            "b.f",
            "b.f2",
            "b.f3",
        }
        assert match_query(CAQuery(options=dict(x=10))) == {
            "A.f",
            "A.f2",
            "B.f",
            "a.f",
            "a.f2",
            "b.f",
        }
        assert match_query(CAQuery(options=dict(y=10))) == {
            "A.f",
            "B.f",
            "B.f2",
            "a.f",
            "b.f",
            "b.f2",
        }
        assert match_query(CAQuery(options=dict(x=20, y=10))) == {"B.f2", "b.f2"}

        assert CAQuery(
            options=dict(
                cacheable=A.f, instance=a, cls=A, base_cls="A", options=dict(my_option1=True)
            ),
        ) == CAQuery(
            options=dict(
                cacheable=A.f, instance=a, cls=A, base_cls="A", options=dict(my_option1=True)
            )
        )

        assert CAQuery.parse(None) == CAQuery()
        assert CAQuery.parse(A.get_ca_setup()) == CAQuery(base_cls=A)
        assert CAQuery.parse(a.get_ca_setup()) == CAQuery(instance=a)
        assert CAQuery.parse(A.f.get_ca_setup()) == CAQuery(cacheable=A.f)
        assert CAQuery.parse(A.f.get_ca_setup(a)) == CAQuery(cacheable=A.f, instance=a)
        assert CAQuery.parse(A.f) == CAQuery(cacheable=A.f)
        assert CAQuery.parse(B.f) == CAQuery(cacheable=B.f)
        assert CAQuery.parse(B.f.func) == CAQuery(cacheable=B.f.func)
        assert CAQuery.parse("f") == CAQuery(cacheable="f")
        assert CAQuery.parse("A.f") == CAQuery(cacheable="f", base_cls="A")
        assert CAQuery.parse("A.f", use_base_cls=False) == CAQuery(cacheable="f", cls="A")
        assert CAQuery.parse("A") == CAQuery(base_cls="A")
        assert CAQuery.parse("A", use_base_cls=False) == CAQuery(cls="A")
        assert CAQuery.parse(vbt.Regex("A")) == CAQuery(base_cls=vbt.Regex("A"))
        assert CAQuery.parse(vbt.Regex("A"), use_base_cls=False) == CAQuery(cls=vbt.Regex("A"))
        assert CAQuery.parse(A) == CAQuery(base_cls=A)
        assert CAQuery.parse(A, use_base_cls=False) == CAQuery(cls=A)
        assert CAQuery.parse((A, B)) == CAQuery(base_cls=(A, B))
        assert CAQuery.parse((A, B), use_base_cls=False) == CAQuery(cls=(A, B))
        assert CAQuery.parse(dict(my_option1=True)) == CAQuery(options=dict(my_option1=True))
        assert CAQuery.parse(a) == CAQuery(instance=a)

    def test_ca_run_setup(self):
        @vbt.cacheable(max_size=2)
        def f(a, b, c=3):
            return a + b + c

        setup = f.get_ca_setup()

        assert setup.readable_name == "f()"
        assert setup.readable_str == f"f():{setup.position_among_similar}"
        assert setup.short_str == "<func tests.test_ca_registry.f>"

        with pytest.raises(Exception):
            CARunSetup(f)
        assert setup.unbound_setup is None
        assert setup.instance_setup is None
        assert setup is CARunSetup.get(f)

        assert setup.run_func(10, 20, c=30) == 60
        assert setup.run_func_and_cache(10, 20, c=30) == 60
        assert setup.misses == 1
        assert setup.hits == 0
        assert setup.first_run_time == list(setup.cache.values())[0].run_time
        assert setup.last_run_time == list(setup.cache.values())[0].run_time
        assert setup.first_hit_time is None
        assert setup.last_hit_time is None
        assert len(setup.cache) == 1
        assert (
            setup.cache[CARunResult.get_hash(hash((("a", 10), ("b", 20), ("c", 30))))].result == 60
        )

        assert setup.run_func_and_cache(10, 20, c=30) == 60
        assert setup.misses == 1
        assert setup.hits == 1
        assert setup.first_run_time == list(setup.cache.values())[0].run_time
        assert setup.last_run_time == list(setup.cache.values())[0].run_time
        assert setup.first_hit_time == list(setup.cache.values())[0].first_hit_time
        assert setup.last_hit_time == list(setup.cache.values())[0].last_hit_time
        assert len(setup.cache) == 1
        assert (
            setup.cache[CARunResult.get_hash(hash((("a", 10), ("b", 20), ("c", 30))))].result == 60
        )

        assert setup.run_func_and_cache(10, 20, c=30) == 60
        assert setup.misses == 1
        assert setup.hits == 2
        assert setup.first_run_time == list(setup.cache.values())[0].run_time
        assert setup.last_run_time == list(setup.cache.values())[0].run_time
        assert setup.first_hit_time == list(setup.cache.values())[0].first_hit_time
        assert setup.last_hit_time == list(setup.cache.values())[0].last_hit_time
        assert len(setup.cache) == 1
        assert (
            setup.cache[CARunResult.get_hash(hash((("a", 10), ("b", 20), ("c", 30))))].result == 60
        )

        assert setup.run_func_and_cache(10, 20, c=40) == 70
        assert setup.misses == 2
        assert setup.hits == 2
        assert setup.first_run_time == list(setup.cache.values())[0].run_time
        assert setup.last_run_time == list(setup.cache.values())[1].run_time
        assert setup.first_hit_time == list(setup.cache.values())[0].first_hit_time
        assert setup.last_hit_time == list(setup.cache.values())[0].last_hit_time
        assert len(setup.cache) == 2
        assert (
            setup.cache[CARunResult.get_hash(hash((("a", 10), ("b", 20), ("c", 40))))].result == 70
        )

        assert setup.run_func_and_cache(10, 20, c=50) == 80
        assert setup.misses == 2
        assert setup.hits == 0
        assert setup.first_run_time == list(setup.cache.values())[0].run_time
        assert setup.last_run_time == list(setup.cache.values())[1].run_time
        assert setup.first_hit_time == list(setup.cache.values())[0].first_hit_time
        assert setup.last_hit_time == list(setup.cache.values())[0].last_hit_time
        assert len(setup.cache) == 2
        assert (
            setup.cache[CARunResult.get_hash(hash((("a", 10), ("b", 20), ("c", 40))))].result == 70
        )
        assert (
            setup.cache[CARunResult.get_hash(hash((("a", 10), ("b", 20), ("c", 50))))].result == 80
        )

        setup.clear_cache()
        assert setup.misses == 0
        assert setup.hits == 0
        assert setup.first_run_time is None
        assert setup.last_run_time is None
        assert setup.first_hit_time is None
        assert setup.last_hit_time is None
        assert len(setup.cache) == 0

        assert setup.run(10, 20, c=30) == 60
        assert len(setup.cache) == 0

        setup.enable_caching()
        assert setup.run(10, 20, c=30) == 60
        assert len(setup.cache) == 1

        setup.disable_caching(clear_cache=False)
        assert setup.run(10, 20, c=30) == 60
        assert len(setup.cache) == 1

        setup.disable_caching(clear_cache=True)
        assert len(setup.cache) == 0

        setup.enable_caching()
        vbt.settings["caching"]["disable"] = True
        assert setup.run(10, 20, c=30) == 60
        assert len(setup.cache) == 0

        setup.enable_whitelist()
        assert setup.run(10, 20, c=30) == 60
        assert len(setup.cache) == 1

        vbt.settings["caching"]["disable_whitelist"] = True
        assert setup.run(10, 20, c=30) == 60
        assert len(setup.cache) == 1

        setup.clear_cache()
        vbt.settings["caching"]["disable"] = False
        vbt.settings["caching"]["disable_whitelist"] = False

        np.testing.assert_array_equal(
            setup.run(10, 20, c=np.array([1, 2, 3])), np.array([31, 32, 33])
        )
        assert len(setup.cache) == 0

        @vbt.cached(ignore_args=["c"])
        def f(a, b, c=3):
            return a + b + c

        setup = f.get_ca_setup()

        assert setup.readable_name == "f()"
        assert setup.readable_str == f"f():{setup.position_among_similar}"
        assert setup.short_str == "<func tests.test_ca_registry.f>"

        np.testing.assert_array_equal(
            setup.run(10, 20, c=np.array([1, 2, 3])), np.array([31, 32, 33])
        )
        assert len(setup.cache) == 1

        class A(Cacheable):
            @vbt.cacheable_property
            def f(self):
                return 10

        with pytest.raises(Exception):
            CARunSetup.get(A.f)

        a = A()

        setup = A.f.get_ca_setup(a)

        assert setup.readable_name == "a.f"
        assert setup.readable_str == f"a:{setup.instance_setup.position_among_similar}.f"
        assert setup.short_str == "<instance property tests.test_ca_registry.A.f>"

        unbound_setup = A.f.get_ca_setup()
        instance_setup = a.get_ca_setup()
        assert setup.unbound_setup is unbound_setup
        assert setup.instance_setup is instance_setup
        assert setup.run() == 10
        assert len(setup.cache) == 0

        setup.enable_caching()
        assert setup.run() == 10
        assert len(setup.cache) == 1
        assert setup.run() == 10
        assert len(setup.cache) == 1

        class B(Cacheable):
            @vbt.cacheable_method
            def f(self, a, b, c=30):
                return a + b + c

        b = B()

        setup = B.f.get_ca_setup(b)

        assert setup.readable_name == "b.f()"
        assert setup.readable_str == f"b:{setup.instance_setup.position_among_similar}.f()"
        assert setup.short_str == "<instance method tests.test_ca_registry.B.f>"

        unbound_setup = B.f.get_ca_setup()
        instance_setup = b.get_ca_setup()
        assert setup.unbound_setup is unbound_setup
        assert setup.instance_setup is instance_setup
        assert setup.run(10, 20, c=30) == 60
        assert len(setup.cache) == 0

        setup.enable_caching()
        assert setup.run(10, 20, c=30) == 60
        assert len(setup.cache) == 1
        assert setup.run(10, 20, c=30) == 60
        assert len(setup.cache) == 1

        class C(Cacheable):
            @vbt.cacheable_property
            def f(self):
                return 10

        class D(C):
            @vbt.cacheable_property
            def f2(self):
                return 10

        class_setup1 = C.get_ca_setup()
        class_setup2 = D.get_ca_setup()
        unbound_setup1 = C.f.get_ca_setup()
        unbound_setup2 = D.f2.get_ca_setup()

        d = D()
        assert not D.f.get_ca_setup(d).whitelist
        assert not D.f.get_ca_setup(d).use_cache
        assert not D.f2.get_ca_setup(d).whitelist
        assert not D.f2.get_ca_setup(d).use_cache

        class_setup2.enable_caching()
        d = D()
        assert not D.f.get_ca_setup(d).whitelist
        assert D.f.get_ca_setup(d).use_cache
        assert not D.f2.get_ca_setup(d).whitelist
        assert D.f2.get_ca_setup(d).use_cache

        class_setup1.disable_caching()
        class_setup1.enable_whitelist()
        d = D()
        assert D.f.get_ca_setup(d).whitelist
        assert not D.f.get_ca_setup(d).use_cache
        assert D.f2.get_ca_setup(d).whitelist
        assert not D.f2.get_ca_setup(d).use_cache

        unbound_setup1.enable_caching()
        d = D()
        assert D.f.get_ca_setup(d).whitelist
        assert D.f.get_ca_setup(d).use_cache
        assert D.f2.get_ca_setup(d).whitelist
        assert not D.f2.get_ca_setup(d).use_cache

        class_setup1.disable_caching()
        d = D()
        assert D.f.get_ca_setup(d).whitelist
        assert not D.f.get_ca_setup(d).use_cache
        assert D.f2.get_ca_setup(d).whitelist
        assert not D.f2.get_ca_setup(d).use_cache

        unbound_setup2.enable_caching()
        d = D()
        assert D.f.get_ca_setup(d).whitelist
        assert not D.f.get_ca_setup(d).use_cache
        assert D.f2.get_ca_setup(d).whitelist
        assert D.f2.get_ca_setup(d).use_cache

    def test_ca_unbound_setup(self):
        class A(Cacheable):
            @vbt.cached_method(whitelist=True)
            def f(self, a, b, c=30):
                return a + b + c

        class B(A):
            @vbt.cacheable_method(whitelist=False)
            def f2(self, a, b, c=30):
                return a + b + c

        a = A()
        b = B()

        assert A.f.get_ca_setup() is B.f.get_ca_setup()
        assert A.f.get_ca_setup() is not B.f2.get_ca_setup()
        assert A.f.get_ca_setup(a) is not B.f.get_ca_setup(b)
        assert A.f.get_ca_setup(a) is not B.f2.get_ca_setup(b)
        assert A.f.get_ca_setup(a).unbound_setup is B.f.get_ca_setup(b).unbound_setup
        assert A.f.get_ca_setup(a).unbound_setup is not B.f2.get_ca_setup(b).unbound_setup

        assert A.f.get_ca_setup().run_setups == {A.f.get_ca_setup(a), B.f.get_ca_setup(b)}
        assert B.f.get_ca_setup().run_setups == {A.f.get_ca_setup(a), B.f.get_ca_setup(b)}
        assert B.f2.get_ca_setup().run_setups == {B.f2.get_ca_setup(b)}

        unbound_setup1 = A.f.get_ca_setup()

        assert unbound_setup1.readable_name == "f()"
        assert unbound_setup1.readable_str == f"f():{unbound_setup1.position_among_similar}"
        assert unbound_setup1.short_str == "<unbound method tests.test_ca_registry.f>"

        unbound_setup2 = B.f2.get_ca_setup()

        assert unbound_setup2.readable_name == "f2()"
        assert unbound_setup2.readable_str == f"f2():{unbound_setup2.position_among_similar}"
        assert unbound_setup2.short_str == "<unbound method tests.test_ca_registry.f2>"

        run_setup1 = A.f.get_ca_setup(a)
        run_setup2 = B.f.get_ca_setup(b)
        run_setup3 = B.f2.get_ca_setup(b)

        assert unbound_setup1.use_cache
        assert unbound_setup1.whitelist
        assert not unbound_setup2.use_cache
        assert not unbound_setup2.whitelist
        assert run_setup1.use_cache
        assert run_setup1.whitelist
        assert run_setup2.use_cache
        assert run_setup2.whitelist
        assert not run_setup3.use_cache
        assert not run_setup3.whitelist

        unbound_setup1.disable_whitelist()
        unbound_setup2.disable_whitelist()
        assert not unbound_setup1.whitelist
        assert not unbound_setup2.whitelist
        assert not run_setup1.whitelist
        assert not run_setup2.whitelist
        assert not run_setup3.whitelist

        unbound_setup1.enable_whitelist()
        unbound_setup2.enable_whitelist()
        assert unbound_setup1.whitelist
        assert unbound_setup2.whitelist
        assert run_setup1.whitelist
        assert run_setup2.whitelist
        assert run_setup3.whitelist

        unbound_setup1.disable_caching()
        unbound_setup2.disable_caching()
        assert not unbound_setup1.use_cache
        assert not unbound_setup2.use_cache
        assert not run_setup1.use_cache
        assert not run_setup2.use_cache
        assert not run_setup3.use_cache

        unbound_setup1.enable_caching()
        unbound_setup2.enable_caching()
        assert unbound_setup1.use_cache
        assert unbound_setup2.use_cache
        assert run_setup1.use_cache
        assert run_setup2.use_cache
        assert run_setup3.use_cache

        assert run_setup1.run(10, 20, c=30) == 60
        assert len(run_setup1.cache) == 1
        assert run_setup2.run(10, 20, c=30) == 60
        assert len(run_setup2.cache) == 1
        assert run_setup3.run(10, 20, c=30) == 60
        assert len(run_setup3.cache) == 1
        unbound_setup1.clear_cache()
        unbound_setup2.clear_cache()
        assert len(run_setup1.cache) == 0
        assert len(run_setup2.cache) == 0
        assert len(run_setup3.cache) == 0

        b2 = B()
        run_setup4 = B.f.get_ca_setup(b2)
        run_setup5 = B.f2.get_ca_setup(b2)
        assert run_setup4.use_cache
        assert run_setup4.whitelist
        assert run_setup5.use_cache
        assert run_setup5.whitelist

    def test_ca_instance_setup(self):
        class A(Cacheable):
            @vbt.cached_method(whitelist=True)
            def f(self, a, b, c=30):
                return a + b + c

        class B(A):
            @vbt.cacheable_method(whitelist=False)
            def f2(self, a, b, c=30):
                return a + b + c

        a = A()
        b = B()

        assert a.get_ca_setup() is not b.get_ca_setup()

        assert a.get_ca_setup().class_setup is A.get_ca_setup()
        assert b.get_ca_setup().class_setup is B.get_ca_setup()
        assert a.get_ca_setup().unbound_setups == {A.f.get_ca_setup()}
        assert b.get_ca_setup().unbound_setups == {B.f.get_ca_setup(), B.f2.get_ca_setup()}
        assert a.get_ca_setup().run_setups == {A.f.get_ca_setup(a)}
        assert b.get_ca_setup().run_setups == {B.f.get_ca_setup(b), B.f2.get_ca_setup(b)}

        instance_setup1 = a.get_ca_setup()

        assert instance_setup1.readable_name == "a"
        assert instance_setup1.readable_str == f"a:{instance_setup1.position_among_similar}"
        assert instance_setup1.short_str == "<instance of tests.test_ca_registry.A>"

        instance_setup2 = b.get_ca_setup()

        assert instance_setup2.readable_name == "b"
        assert instance_setup2.readable_str == f"b:{instance_setup2.position_among_similar}"
        assert instance_setup2.short_str == "<instance of tests.test_ca_registry.B>"

        run_setup1 = A.f.get_ca_setup(a)
        run_setup2 = B.f.get_ca_setup(b)
        run_setup3 = B.f2.get_ca_setup(b)

        assert instance_setup1.use_cache is None
        assert instance_setup1.whitelist is None
        assert instance_setup2.use_cache is None
        assert instance_setup2.whitelist is None
        assert run_setup1.use_cache
        assert run_setup1.whitelist
        assert run_setup2.use_cache
        assert run_setup2.whitelist
        assert not run_setup3.use_cache
        assert not run_setup3.whitelist

        instance_setup1.disable_whitelist()
        instance_setup2.disable_whitelist()
        assert not instance_setup1.whitelist
        assert not instance_setup2.whitelist
        assert not run_setup1.whitelist
        assert not run_setup2.whitelist
        assert not run_setup3.whitelist

        instance_setup1.enable_whitelist()
        instance_setup2.enable_whitelist()
        assert instance_setup1.whitelist
        assert instance_setup2.whitelist
        assert run_setup1.whitelist
        assert run_setup2.whitelist
        assert run_setup3.whitelist

        instance_setup1.disable_caching()
        instance_setup2.disable_caching()
        assert not instance_setup1.use_cache
        assert not instance_setup2.use_cache
        assert not run_setup1.use_cache
        assert not run_setup2.use_cache
        assert not run_setup3.use_cache

        instance_setup1.enable_caching()
        instance_setup2.enable_caching()
        assert instance_setup1.use_cache
        assert instance_setup2.use_cache
        assert run_setup1.use_cache
        assert run_setup2.use_cache
        assert run_setup3.use_cache

        assert run_setup1.run(10, 20, c=30) == 60
        assert len(run_setup1.cache) == 1
        assert run_setup2.run(10, 20, c=30) == 60
        assert len(run_setup2.cache) == 1
        assert run_setup3.run(10, 20, c=30) == 60
        assert len(run_setup3.cache) == 1
        instance_setup1.clear_cache()
        instance_setup2.clear_cache()
        assert len(run_setup1.cache) == 0
        assert len(run_setup2.cache) == 0
        assert len(run_setup3.cache) == 0

        B.get_ca_setup().disable_caching()
        B.get_ca_setup().disable_whitelist()
        b2 = B()
        instance_setup3 = b2.get_ca_setup()
        run_setup4 = B.f.get_ca_setup(b2)
        run_setup5 = B.f2.get_ca_setup(b2)
        assert not instance_setup3.use_cache
        assert not instance_setup3.whitelist
        assert not run_setup4.use_cache
        assert not run_setup4.whitelist
        assert not run_setup5.use_cache
        assert not run_setup5.whitelist

    def test_ca_class_setup(self):
        class A(Cacheable):
            @vbt.cacheable_method
            def f(self, a, b, c=30):
                return a + b + c

        class B(A):
            @vbt.cacheable_method
            def f2(self, a, b, c=30):
                return a + b + c

        class C(Cacheable):
            @vbt.cacheable_method
            def f3(self, a, b, c=30):
                return a + b + c

        a = A()
        b = B()
        c = C()

        assert A.get_ca_setup() is not B.get_ca_setup()
        assert A.get_ca_setup() is not C.get_ca_setup()

        class_setup1 = A.get_ca_setup()

        assert class_setup1.readable_name == "A"
        assert class_setup1.readable_str == f"A:{class_setup1.position_among_similar}"
        assert class_setup1.short_str == "<class tests.test_ca_registry.A>"

        class_setup2 = B.get_ca_setup()

        assert class_setup2.readable_name == "B"
        assert class_setup2.readable_str == f"B:{class_setup2.position_among_similar}"
        assert class_setup2.short_str == "<class tests.test_ca_registry.B>"

        class_setup3 = C.get_ca_setup()

        assert class_setup3.readable_name == "C"
        assert class_setup3.readable_str == f"C:{class_setup3.position_among_similar}"
        assert class_setup3.short_str == "<class tests.test_ca_registry.C>"

        assert class_setup1.superclass_setups == []
        assert class_setup2.superclass_setups == [class_setup1]
        assert class_setup3.superclass_setups == []
        assert class_setup1.subclass_setups == [class_setup2]
        assert class_setup2.subclass_setups == []
        assert class_setup3.subclass_setups == []
        assert class_setup1.unbound_setups == {A.f.get_ca_setup()}
        assert class_setup2.unbound_setups == {A.f.get_ca_setup(), B.f2.get_ca_setup()}
        assert class_setup3.unbound_setups == {C.f3.get_ca_setup()}
        assert class_setup1.instance_setups == {a.get_ca_setup()}
        assert class_setup2.instance_setups == {b.get_ca_setup()}
        assert class_setup3.instance_setups == {c.get_ca_setup()}
        assert class_setup1.child_setups == {a.get_ca_setup(), B.get_ca_setup()}
        assert class_setup2.child_setups == {b.get_ca_setup()}
        assert class_setup3.child_setups == {c.get_ca_setup()}

        class_setup1 = A.get_ca_setup()
        class_setup2 = B.get_ca_setup()
        class_setup3 = C.get_ca_setup()
        instance_setup1 = a.get_ca_setup()
        instance_setup2 = b.get_ca_setup()
        instance_setup3 = c.get_ca_setup()

        assert class_setup1.use_cache is None
        assert class_setup1.whitelist is None
        assert class_setup2.use_cache is None
        assert class_setup2.whitelist is None
        assert class_setup3.use_cache is None
        assert class_setup3.whitelist is None
        assert instance_setup1.use_cache is None
        assert instance_setup1.whitelist is None
        assert instance_setup2.use_cache is None
        assert instance_setup2.whitelist is None
        assert instance_setup3.use_cache is None
        assert instance_setup3.whitelist is None

        class_setup1.enable_whitelist()
        assert class_setup1.whitelist
        assert class_setup2.whitelist
        assert class_setup3.whitelist is None
        assert instance_setup1.whitelist
        assert instance_setup2.whitelist
        assert instance_setup3.whitelist is None

        class_setup1.enable_caching()
        assert class_setup1.use_cache
        assert class_setup2.use_cache
        assert class_setup3.use_cache is None
        assert instance_setup1.use_cache
        assert instance_setup2.use_cache
        assert instance_setup3.use_cache is None

        class_setup2.disable_whitelist()
        assert class_setup1.whitelist
        assert not class_setup2.whitelist
        assert class_setup3.whitelist is None
        assert instance_setup1.whitelist
        assert not instance_setup2.whitelist
        assert instance_setup3.whitelist is None

        class_setup2.disable_caching()
        assert class_setup1.use_cache
        assert not class_setup2.use_cache
        assert class_setup3.use_cache is None
        assert instance_setup1.use_cache
        assert not instance_setup2.use_cache
        assert instance_setup3.use_cache is None

        class D(A):
            @vbt.cacheable_method
            def f4(self, a, b, c=30):
                return a + b + c

        d = D()
        class_setup4 = D.get_ca_setup()
        instance_setup4 = d.get_ca_setup()

        assert class_setup4.use_cache
        assert class_setup4.whitelist
        assert instance_setup4.use_cache
        assert instance_setup4.whitelist

        class E(B):
            @vbt.cacheable_method
            def f5(self, a, b, c=30):
                return a + b + c

        e = E()
        class_setup5 = E.get_ca_setup()
        instance_setup5 = e.get_ca_setup()

        assert not class_setup5.use_cache
        assert not class_setup5.whitelist
        assert not instance_setup5.use_cache
        assert not instance_setup5.whitelist

    def test_match_setups(self):
        class A(Cacheable):
            @vbt.cacheable_property
            def f_test(self):
                return 10

        class B(A):
            @vbt.cacheable_method
            def f2_test(self, a, b, c=30):
                return a + b + c

        @vbt.cacheable
        def f3_test(a, b, c=30):
            return a + b + c

        a = A()
        b = B()

        b.get_ca_setup().enable_caching()

        queries = [
            A.get_ca_setup().query,
            B.get_ca_setup().query,
            A.f_test.get_ca_setup().query,
            B.f2_test.get_ca_setup().query,
            f3_test.get_ca_setup().query,
        ]
        assert ca_reg.match_setups(queries, kind=None) == {
            A.get_ca_setup(),
            B.get_ca_setup(),
            a.get_ca_setup(),
            b.get_ca_setup(),
            A.f_test.get_ca_setup(),
            B.f2_test.get_ca_setup(),
            A.f_test.get_ca_setup(a),
            B.f_test.get_ca_setup(b),
            B.f2_test.get_ca_setup(b),
            f3_test.get_ca_setup(),
        }
        assert ca_reg.match_setups(
            queries, kind=None, filter_func=lambda setup: setup.caching_enabled
        ) == {
            b.get_ca_setup(),
            B.f_test.get_ca_setup(b),
            B.f2_test.get_ca_setup(b),
        }
        assert ca_reg.match_setups(queries, kind="runnable") == {
            A.f_test.get_ca_setup(a),
            B.f_test.get_ca_setup(b),
            B.f2_test.get_ca_setup(b),
            f3_test.get_ca_setup(),
        }
        assert ca_reg.match_setups(queries, kind="unbound") == {
            A.f_test.get_ca_setup(),
            B.f2_test.get_ca_setup(),
        }
        assert ca_reg.match_setups(queries, kind="instance") == {a.get_ca_setup(), b.get_ca_setup()}
        assert ca_reg.match_setups(queries, kind="class") == {A.get_ca_setup(), B.get_ca_setup()}
        assert ca_reg.match_setups(queries, kind=("class", "instance")) == {
            A.get_ca_setup(),
            B.get_ca_setup(),
            a.get_ca_setup(),
            b.get_ca_setup(),
        }
        assert ca_reg.match_setups(
            queries, kind=("class", "instance"), exclude=b.get_ca_setup()
        ) == {
            A.get_ca_setup(),
            B.get_ca_setup(),
            a.get_ca_setup(),
        }
        assert ca_reg.match_setups(queries, collapse=True, kind=None) == {
            A.get_ca_setup(),
            A.f_test.get_ca_setup(),
            B.f2_test.get_ca_setup(),
            f3_test.get_ca_setup(),
        }
        assert ca_reg.match_setups(queries, collapse=True, kind="instance") == {
            a.get_ca_setup(),
            b.get_ca_setup(),
        }
        assert ca_reg.match_setups(queries, collapse=True, kind=("instance", "runnable")) == {
            a.get_ca_setup(),
            b.get_ca_setup(),
            f3_test.get_ca_setup(),
        }
        assert ca_reg.match_setups(
            queries, collapse=True, kind=("instance", "runnable"), exclude=a.get_ca_setup()
        ) == {
            b.get_ca_setup(),
            f3_test.get_ca_setup(),
        }
        assert ca_reg.match_setups(
            queries,
            collapse=True,
            kind=("instance", "runnable"),
            exclude=a.get_ca_setup(),
            exclude_children=False,
        ) == {b.get_ca_setup(), A.f_test.get_ca_setup(a), f3_test.get_ca_setup()}

    def test_ca_query_delegator(self):
        class A(Cacheable):
            @vbt.cacheable_property
            def f_test(self):
                return 10

        class B(A):
            @vbt.cacheable_method
            def f2_test(self, a, b, c=30):
                return a + b + c

        @vbt.cacheable
        def f3_test(a, b, c=30):
            return a + b + c

        a = A()
        b = B()

        class_setup1 = A.get_ca_setup()
        class_setup2 = B.get_ca_setup()

        assert class_setup1.use_cache is None
        assert class_setup1.whitelist is None
        assert class_setup2.use_cache is None
        assert class_setup2.whitelist is None

        queries = [
            A.get_ca_setup().query,
            B.get_ca_setup().query,
            A.f_test.get_ca_setup().query,
            B.f2_test.get_ca_setup().query,
            f3_test.get_ca_setup().query,
        ]
        query_delegator = vbt.CAQueryDelegator(queries, kind="class")
        assert query_delegator.child_setups == {class_setup1}

        query_delegator.enable_caching()
        query_delegator.disable_whitelist()
        assert class_setup1.use_cache
        assert not class_setup1.whitelist
        assert class_setup2.use_cache
        assert not class_setup2.whitelist

        assert query_delegator.hits == 0
        assert query_delegator.misses == 0
        assert query_delegator.first_run_time is None
        assert query_delegator.last_run_time is None
        assert query_delegator.first_hit_time is None
        assert query_delegator.last_hit_time is None

        assert a.f_test == 10
        assert query_delegator.hits == 0
        assert query_delegator.misses == 1
        assert query_delegator.first_run_time == A.f_test.get_ca_setup(a).first_run_time
        assert query_delegator.last_run_time == A.f_test.get_ca_setup(a).last_run_time
        assert query_delegator.first_hit_time is None
        assert query_delegator.last_hit_time is None

        assert a.f_test == 10
        assert query_delegator.hits == 1
        assert query_delegator.misses == 1
        assert query_delegator.first_run_time == A.f_test.get_ca_setup(a).first_run_time
        assert query_delegator.last_run_time == A.f_test.get_ca_setup(a).last_run_time
        assert query_delegator.first_hit_time == A.f_test.get_ca_setup(a).first_hit_time
        assert query_delegator.last_hit_time == A.f_test.get_ca_setup(a).last_hit_time

        assert b.f2_test(10, 20, c=30) == 60
        assert query_delegator.hits == 1
        assert query_delegator.misses == 2
        assert query_delegator.first_run_time == A.f_test.get_ca_setup(a).first_run_time
        assert query_delegator.last_run_time == B.f2_test.get_ca_setup(b).last_run_time
        assert query_delegator.first_hit_time == A.f_test.get_ca_setup(a).first_hit_time
        assert query_delegator.last_hit_time == A.f_test.get_ca_setup(a).last_hit_time

        assert b.f2_test(10, 20, c=30) == 60
        assert query_delegator.hits == 2
        assert query_delegator.misses == 2
        assert query_delegator.first_run_time == A.f_test.get_ca_setup(a).first_run_time
        assert query_delegator.last_run_time == B.f2_test.get_ca_setup(b).last_run_time
        assert query_delegator.first_hit_time == A.f_test.get_ca_setup(a).first_hit_time
        assert query_delegator.last_hit_time == B.f2_test.get_ca_setup(b).last_hit_time

        assert f3_test(10, 20, c=30) == 60
        assert query_delegator.hits == 2
        assert query_delegator.misses == 2
        assert query_delegator.first_run_time == A.f_test.get_ca_setup(a).first_run_time
        assert query_delegator.last_run_time == B.f2_test.get_ca_setup(b).last_run_time
        assert query_delegator.first_hit_time == A.f_test.get_ca_setup(a).first_hit_time
        assert query_delegator.last_hit_time == B.f2_test.get_ca_setup(b).last_hit_time

        setup_hierarchy = query_delegator.get_setup_hierarchy()
        assert len(setup_hierarchy) == 1
        assert len(setup_hierarchy[0]["children"]) == 2

        assert query_delegator.metrics == {
            "hits": query_delegator.hits,
            "misses": query_delegator.misses,
            "total_size": query_delegator.total_size,
            "total_elapsed": A.f_test.get_ca_setup(a).total_elapsed
            + B.f2_test.get_ca_setup(b).total_elapsed,
            "total_saved": A.f_test.get_ca_setup(a).total_saved
            + B.f2_test.get_ca_setup(b).total_saved,
            "first_run_time": query_delegator.first_run_time,
            "last_run_time": query_delegator.last_run_time,
            "first_hit_time": query_delegator.first_hit_time,
            "last_hit_time": query_delegator.last_hit_time,
        }

        columns = [
            "hash",
            "use_cache",
            "whitelist",
            "caching_enabled",
            "hits",
            "misses",
            "total_size",
            "total_elapsed",
            "total_saved",
            "first_run_time",
            "last_run_time",
            "first_hit_time",
            "last_hit_time",
            "creation_time",
            "last_update_time",
        ]
        stats = query_delegator.get_stats(readable=False, short_str=False)
        assert_index_equal(stats.columns, pd.Index(columns))
        np.testing.assert_array_equal(stats.index.values, np.array([str(class_setup1)]))
        np.testing.assert_array_equal(stats["hash"].values, np.array([hash(class_setup1)]))
        np.testing.assert_array_equal(stats["use_cache"].values, np.array([class_setup1.use_cache]))
        np.testing.assert_array_equal(stats["whitelist"].values, np.array([class_setup1.whitelist]))
        np.testing.assert_array_equal(
            stats["caching_enabled"].values,
            np.array([class_setup1.caching_enabled]),
        )
        np.testing.assert_array_equal(stats["hits"].values, np.array([class_setup1.hits]))
        np.testing.assert_array_equal(stats["misses"].values, np.array([class_setup1.misses]))
        np.testing.assert_array_equal(
            stats["total_size"].values, np.array([class_setup1.total_size])
        )
        np.testing.assert_array_equal(
            stats["total_elapsed"].values,
            pd.to_timedelta(np.array([class_setup1.total_elapsed])),
        )
        np.testing.assert_array_equal(
            stats["total_saved"].values,
            pd.to_timedelta(np.array([class_setup1.total_saved])),
        )
        np.testing.assert_array_equal(
            stats["first_run_time"].values,
            pd.to_datetime([class_setup1.first_run_time]).values,
        )
        np.testing.assert_array_equal(
            stats["last_run_time"].values,
            pd.to_datetime([class_setup1.last_run_time]).values,
        )
        np.testing.assert_array_equal(
            stats["first_hit_time"].values,
            pd.to_datetime([class_setup1.first_hit_time]).values,
        )
        np.testing.assert_array_equal(
            stats["last_hit_time"].values,
            pd.to_datetime([class_setup1.last_hit_time]).values,
        )
        np.testing.assert_array_equal(
            stats["creation_time"].values,
            pd.to_datetime([class_setup1.creation_time]).values,
        )
        np.testing.assert_array_equal(
            stats["last_update_time"].values,
            pd.to_datetime([class_setup1.last_update_time]).values,
        )

        stats = query_delegator.get_stats()
        assert_index_equal(stats.columns, pd.Index(columns))
        stats = query_delegator.get_stats(include=columns)
        assert_index_equal(stats.columns, pd.Index(columns))
        stats = query_delegator.get_stats(include=columns[0])
        assert_index_equal(stats.columns, pd.Index([columns[0]]))
        stats = query_delegator.get_stats(include=[columns[0]])
        assert_index_equal(stats.columns, pd.Index([columns[0]]))
        stats = query_delegator.get_stats(exclude=columns)
        assert stats is None
        stats = query_delegator.get_stats(exclude=columns[0])
        assert_index_equal(stats.columns, pd.Index(columns[1:]))
        stats = query_delegator.get_stats(exclude=[columns[0]])
        assert_index_equal(stats.columns, pd.Index(columns[1:]))

    def test_disable_machinery(self):
        vbt.settings["caching"]["disable_machinery"] = True

        class A(Cacheable):
            @vbt.cacheable_property
            def f_test(self):
                return 10

        class B(A):
            @vbt.cacheable_method
            def f2_test(self, a, b, c=30):
                return a + b + c

        @vbt.cacheable
        def f3_test(a, b, c=30):
            return a + b + c

        a = A()
        b = B()

        assert a.f_test == 10
        assert b.f_test == 10
        assert b.f2_test(10, 20, c=30) == 60
        assert f3_test(10, 20, c=30) == 60

        assert A.get_ca_setup() is None
        assert a.get_ca_setup() is None
        assert B.get_ca_setup() is None
        assert b.get_ca_setup() is None
        assert A.f_test.get_ca_setup() is None
        assert A.f_test.get_ca_setup(a) is None
        assert B.f_test.get_ca_setup() is None
        assert B.f_test.get_ca_setup(b) is None
        assert B.f2_test.get_ca_setup() is None
        assert B.f2_test.get_ca_setup(b) is None
        assert f3_test.get_ca_setup() is None

        vbt.settings["caching"]["disable_machinery"] = False

    def test_gc(self):
        class A(Cacheable):
            @vbt.cacheable_property
            def f(self):
                return 10

        a = A()

        assert ca_reg.match_setups(CAQuery(cls=A), kind=None) == {
            A.get_ca_setup(),
            a.get_ca_setup(),
            A.f.get_ca_setup(a),
        }
        a_ref = weakref.ref(a)
        del a
        gc.collect()
        assert a_ref() is None
        assert ca_reg.match_setups(CAQuery(cls=A), kind=None) == {A.get_ca_setup()}

        class B(Cacheable):
            @vbt.cacheable_method
            def f(self):
                return 10

        b = B()

        assert ca_reg.match_setups(CAQuery(cls=B), kind=None) == {
            B.get_ca_setup(),
            B.f.get_ca_setup(b),
            b.get_ca_setup(),
        }
        b_ref = weakref.ref(b)
        del b
        gc.collect()
        assert b_ref() is None
        assert ca_reg.match_setups(CAQuery(cls=B), kind=None) == {B.get_ca_setup()}
