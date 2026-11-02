from boardwatch.prepare import _assign_grouped_splits, _group_similar_hashes


def test_perceptual_groups_use_transitive_connected_components():
    hashes = {
        "a.jpg": 0b0000,
        "b.jpg": 0b0001,
        "c.jpg": 0b0011,
        "d.jpg": 0b11110000,
        "e.jpg": 0b11110001,
    }

    group_by_name, groups = _group_similar_hashes(hashes, max_distance=1)

    assert group_by_name["a.jpg"] == group_by_name["b.jpg"] == group_by_name["c.jpg"]
    assert group_by_name["d.jpg"] == group_by_name["e.jpg"]
    assert group_by_name["a.jpg"] != group_by_name["d.jpg"]
    assert sorted(map(len, groups.values())) == [2, 3]


def test_grouped_split_holds_out_entire_smallest_multi_image_group_deterministically():
    hashes = {
        "train-a.jpg": 0,
        "train-b.jpg": 0,
        "train-c.jpg": 0,
        "test-a.jpg": (1 << 10) - 1,
        "test-b.jpg": (1 << 10) - 1,
        "single.jpg": (1 << 20) - 1,
    }

    first = _assign_grouped_splits(hashes, seed=2026)
    second = _assign_grouped_splits(hashes, seed=2026)
    assignments, group_by_name, groups, test_group = first

    assert first == second
    assert assignments["test-a.jpg"] == assignments["test-b.jpg"] == "test"
    assert groups[test_group] == ["test-a.jpg", "test-b.jpg"]
    assert group_by_name["test-a.jpg"] == group_by_name["test-b.jpg"]
    assert all(assignments[name] != "test" for name in ("train-a.jpg", "train-b.jpg", "train-c.jpg"))
    assert set(assignments.values()) == {"train", "validation", "test"}
