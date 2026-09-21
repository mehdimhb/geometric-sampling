from prescribed_sip.experiments.profile import representation_benchmark


def test_profile_representations_encode_identical_sample_differences():
    rows=representation_benchmark(N=25,n=5,repeats=10,seed=1)
    assert {r['representation'] for r in rows}=={'tuple','integer_mask','frozenset'}
    assert len({r['difference_checksum'] for r in rows})==1
    assert all(r['elapsed_seconds']>=0 and r['memory_bytes']>0 for r in rows)
