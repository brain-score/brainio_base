import pytest
from xarray import DataArray

from brainio_base.assemblies import DataAssembly, get_levels


def test_get_levels():
    assy = DataAssembly(
        data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]],
        coords={
            'up': ("a", ['alpha', 'alpha', 'beta', 'beta', 'beta', 'beta']),
            'down': ("a", [1, 1, 1, 1, 2, 2]),
            'sideways': ('b', ['x', 'y', 'z'])
        },
        dims=['a', 'b']
    )
    assert get_levels(assy) == ["up", "down"]

def test_wrap_dataarray():
    assy = DataAssembly(
        data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]],
        coords={
            'up': ("a", ['alpha', 'alpha', 'beta', 'beta', 'beta', 'beta']),
            'down': ("a", [1, 1, 1, 1, 2, 2]),
            'sideways': ('b', ['x', 'y', 'z'])
        },
        dims=['a', 'b']
    )
    assert assy.coords.variables["a"].level_names == ["up", "down"]
    da = DataArray(assy)
    assert da.coords.variables["a"].level_names == ["up", "down"]


def test_reset_index():
    assy = DataAssembly(
        data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]],
        coords={
            'up': ("a", ['alpha', 'alpha', 'beta', 'beta', 'beta', 'beta']),
            'down': ("a", [1, 1, 1, 1, 2, 2]),
            'sideways': ('b', ['x', 'y', 'z'])
        },
        dims=['a', 'b']
    )
    da = DataArray(assy)
    da = da.reset_index(["up", "down"])
    assert get_levels(da) == []


class TestIndex:
    def test_single_element(self):
        d = DataAssembly([0], coords={'coordA': ('dim', [0]), 'coordB': ('dim', [1])}, dims=['dim'])
        d.sel(coordA=0)
        d.sel(coordB=1)

    def test_multi_elements(self):
        d = DataAssembly([0, 1, 2, 3, 4],
                         coords={'coordA': ('dim', [0, 1, 2, 3, 4]),
                                 'coordB': ('dim', [1, 2, 3, 4, 5])},
                         dims=['dim'])
        d.sel(coordA=0)
        d.sel(coordA=4)
        d.sel(coordB=1)
        d.sel(coordB=5)

    def test_incorrect_coord(self):
        d = DataAssembly([0], coords={'coordA': ('dim', [0]), 'coordB': ('dim', [1])}, dims=['dim'])
        with pytest.raises(KeyError):
            d.sel(coordA=1)
        with pytest.raises(KeyError):
            d.sel(coordB=0)


class TestMultiGroupby:
    def test_single_dimension(self):
        d = DataAssembly([[1, 2, 3], [4, 5, 6]], coords={'a': ['a', 'b'], 'b': ['x', 'y', 'z']}, dims=['a', 'b'])
        g = d.multi_groupby(['a']).mean(...)
        assert g.equals(DataAssembly([2, 5], coords={'a': ['a', 'b']}, dims=['a']))

    def test_single_dimension_int(self):
        d = DataAssembly([[1, 2, 3], [4, 5, 6]], coords={'a': [1, 2], 'b': [3, 4, 5]}, dims=['a', 'b'])
        g = d.multi_groupby(['a']).mean(...)
        assert g.equals(DataAssembly([2., 5.], coords={'a': [1, 2]}, dims=['a']))

    def test_single_coord(self):
        d = DataAssembly(
            data=[
                [ 0,  1,  2,  3,  4,  5,  6],
                [ 7,  8,  9, 10, 11, 12, 13],
                [14, 15, 16, 17, 18, 19, 20]
            ],
            coords={
                "greek": ("a", ['alpha', 'beta', 'gamma']),
                "colors": ("a", ['red', 'green', 'blue']),
                "compass": ("b", ['north', 'south', 'east', 'west', 'northeast', 'southeast', 'southwest']),
                "integer": ("b", [0, 1, 2, 3, 4, 5, 6]),
            },
            dims=("a", "b")
        )
        g = d.multi_groupby(['greek']).mean(...)
        c = DataAssembly(
            data=[3, 10, 17],
            coords={'greek': ('greek', ['alpha', 'beta', 'gamma'])},
            dims=['greek']
        )
        assert g.equals(c)

    def test_single_dim_multi_coord(self):
        d = DataAssembly([1, 2, 3, 4, 5, 6],
                         coords={'a': ('multi_dim', ['a', 'a', 'a', 'a', 'a', 'a']),
                                 'b': ('multi_dim', ['a', 'a', 'a', 'b', 'b', 'b']),
                                 'c': ('multi_dim', ['a', 'b', 'c', 'd', 'e', 'f'])},
                         dims=['multi_dim'])
        g = d.multi_groupby(['a', 'b']).mean()
        assert g.equals(DataAssembly([2, 5],
                                     coords={'a': ('multi_dim', ['a', 'a']), 'b': ('multi_dim', ['a', 'b'])},
                                     dims=['multi_dim']))

    def test_int_multi_coord(self):
        d = DataAssembly([1, 2, 3, 4, 5, 6],
                         coords={'a': ('multi_dim', [1, 1, 1, 1, 1, 1]),
                                 'b': ('multi_dim', ['a', 'a', 'a', 'b', 'b', 'b']),
                                 'c': ('multi_dim', ['a', 'b', 'c', 'd', 'e', 'f'])},
                         dims=['multi_dim'])
        g = d.multi_groupby(['a', 'b']).mean()
        assert g.equals(DataAssembly([2., 5.],
                                     coords={'a': ('multi_dim', [1, 1]), 'b': ('multi_dim', ['a', 'b'])},
                                     dims=['multi_dim']))

    def test_two_coord(self):
        assy = DataAssembly([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]],
                         coords={'up': ("a", ['alpha', 'alpha', 'beta', 'beta', 'beta', 'beta']),
                                 'down': ("a", [1, 1, 1, 1, 2, 2]),
                                 'sideways': ('b', ['x', 'y', 'z'])},
                         dims=['a', 'b'])
        assy_grouped = assy.multi_groupby(['up', 'down']).mean(dim="a")
        assy_2 = DataAssembly([[2.5, 3.5, 4.5], [8.5, 9.5, 10.5], [14.5, 15.5, 16.5]],
                                     coords={'up': ("a", ['alpha', 'beta', 'beta']),
                                             'down': ("a", [1, 1, 2]),
                                             'sideways': ('b', ['x', 'y', 'z'])},
                                     dims=['a', 'b'])
        assert assy_grouped.equals(assy_2)


class TestMultiDimApply:
    def test_unique_values(self):
        d = DataAssembly([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                         coords={'a': ['a', 'b', 'c', 'd'],
                                 'b': ['x', 'y', 'z']},
                         dims=['a', 'b'])
        g = d.multi_dim_apply(['a', 'b'], lambda x, **_: x)
        assert g.equals(d)

    def test_unique_values_swappeddims(self):
        d = DataAssembly([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                         coords={'a': ['a', 'b', 'c', 'd'],
                                 'b': ['x', 'y', 'z']},
                         dims=['a', 'b'])
        g = d.multi_dim_apply(['b', 'a'], lambda x, **_: x)
        assert g.equals(d)

    def test_nonindex_coord(self):
        d = DataAssembly([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                         coords={'a': ['a', 'b', 'c', 'd'],
                                 'b': ['x', 'y', 'z'],
                                 # additional coordinate that has no index values.
                                 # This could e.g. be the result of `.sel(c='remnant')`
                                 'c': 'remnant'},
                         dims=['a', 'b'])
        g = d.multi_dim_apply(['a', 'b'], lambda x, **_: x)
        assert g.equals(d)  # also tests that `c` persists

    def test_subtract_mean(self):
        d = DataAssembly([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                         coords={'a': ['a', 'b', 'c', 'd'],
                                 'aa': ('a', ['a', 'a', 'b', 'b']),
                                 'b': ['x', 'y', 'z']},
                         dims=['a', 'b'])
        g = d.multi_dim_apply(['aa', 'b'], lambda x, **_: x - x.mean())
        assert g.equals(DataAssembly([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5], [-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]],
                                     coords={'a': ['a', 'b', 'c', 'd'],
                                             'aa': ('a', ['a', 'a', 'b', 'b']),
                                             'b': ['x', 'y', 'z']},
                                     dims=['a', 'b']))

    def test_multi_level(self):
        d = DataAssembly([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]],
                         coords={'up': ("a", ['alpha', 'alpha', 'beta', 'beta', 'beta', 'beta']),
                                 'down': ("a", [1, 1, 1, 1, 2, 2]),
                                 'sideways': ('b', ['x', 'y', 'z'])},
                         dims=['a', 'b'])
        g = d.multi_dim_apply(['a', 'b'], lambda x, **_: x)
        assert g.equals(d)

